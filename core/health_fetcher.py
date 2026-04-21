"""
data/fetchers/health_fetcher.py
--------------------------------
Fetches health threat data from the ECDC Communicable Disease Threats Report
(CDTR), a weekly bulletin published every Thursday by the European Centre for
Disease Prevention and Control.

Source:
    https://www.ecdc.europa.eu/en/publications-and-data/monitoring/weekly-threats-reports

Strategy:
    1. Fetch the ECDC CDTR listing page to find the latest week's report URL.
    2. Fetch that report page to extract the PDF link and active threat list
       from the description text (no PDF parsing needed for the summary).
    3. Score the threats using a keyword-weighted approach.

Health score (0-50):
    - Routine seasonal activity (ILI, RSV, seasonal flu) → low baseline
    - Known EU threats at monitoring level                → moderate
    - Novel pathogen / pandemic-potential event           → high
    - WHO PHEIC declared                                  → maximum

The score is capped at 50 — higher than geopolitical (30) because a pandemic
directly affects supply chains, medication availability, and the 72h kit items
the user needs. Kept below weather (100) because weather is the primary daily
driver for most users.

Design note — Option C architecture:
    Health, geopolitical, and weather are THREE INDEPENDENT signals.
    They are NEVER summed into a combined score.
    The UI displays them as three separate gauges.
    The alert prioritizer reads all three and surfaces the most actionable
    items from each dimension independently.
"""

import re
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CDTR_LISTING_URL = (
    "https://www.ecdc.europa.eu/en/publications-and-data/monitoring/weekly-threats-reports"
)

MAX_HEALTH_SCORE = 50

# Threat keywords → score contribution
# Higher scores for novel/pandemic-potential threats
_THREAT_WEIGHTS = {
    # Pandemic-potential — maximum concern
    "pandemic":              50,
    "pheic":                 50,
    "novel":                 35,
    "unknown pathogen":      40,
    "new variant":           25,

    # High-concern known threats — only if actively causing EU cases
    "marburg":               30,
    "ebola":                 30,
    "mpox":                  15,
    "avian influenza":       12,   # ↓ routine monitoring, not outbreak
    "h5n1":                  15,
    "h5":                    10,
    "mers":                  12,
    "sars":                  25,
    "sars-cov-2":             8,
    "covid":                  8,

    # Moderate threats
    "measles":               10,
    "cholera":                8,
    "dengue":                 6,   # ↓ tropical, rarely EU-acquired
    "chikungunya":            5,
    "west nile":              6,
    "meningococcal":         10,
    "diphtheria":             8,
    "polio":                 12,
    "tuberculosis":           5,

    # Routine — very low contribution
    "influenza":              3,
    "ili":                    2,
    "respiratory":            2,
    "rsv":                    2,
}

# Phrases that indicate the threat is actively impacting the EU/EEA
# (vs. being monitored globally but not present in Europe)
_EU_IMPACT_PHRASES = [
    "eu/eea",
    "european",
    "reported in eu",
    "reported in europe",
    "cases in eu",
    "member state",
    "outbreak in",
]

# Phrases that indicate a threat is distant / low EU risk
_DISTANT_PHRASES = [
    "no cases in eu",
    "risk to eu is very low",
    "risk to human health in the eu/eea is currently considered very low",
    "no evidence of community transmission",
    "no additional measures",
    "not yet detected in eu",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThreatSignal:
    name:           str
    score:          int       # contribution to health_score
    eu_impact:      bool      # True if actively affecting EU/EEA
    source_text:    str       # excerpt from CDTR that triggered this signal


@dataclass
class HealthSnapshot:
    week_label:       str     # e.g. "Week 14, 2026"
    period:           str     # e.g. "28 March – 3 April 2026"
    threats:          List[ThreatSignal]
    health_score:     int     # 0-50
    level:            str     # ROUTINE / MEDIUM / HIGH / CRITICAL
    top_threats:      List[str]   # human-readable names of active threats
    cdtr_url:         str         # URL of the CDTR page used
    fetched_at:       str


# ---------------------------------------------------------------------------
# Score → level
# ---------------------------------------------------------------------------

def health_score_to_level(score: int) -> str:
    """
    Convert a numeric health score to a level label.

    Thresholds are calibrated so that:
    - Typical week (ILI + 1-2 monitored threats) → ROUTINE
    - Active EU outbreak (e.g. measles + mpox)   → MEDIUM
    - Novel pathogen or pandemic signal           → HIGH / CRITICAL
    """
    if score >= 40:
        return "CRITICAL"
    if score >= 25:
        return "HIGH"
    if score >= 10:
        return "MEDIUM"
    return "ROUTINE"


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------

def _extract_threats_from_text(text: str) -> List[ThreatSignal]:
    """
    Scan CDTR summary text for known threat keywords and score them.

    Args:
        text: Plain text extracted from the CDTR page description.

    Returns:
        List of ThreatSignal objects, deduplicated by name.
    """
    text_lower = text.lower()
    signals: dict = {}  # name → ThreatSignal (deduped)

    for keyword, base_score in _THREAT_WEIGHTS.items():
        if keyword not in text_lower:
            continue

        # Find a short excerpt around the keyword
        idx = text_lower.find(keyword)
        start = max(0, idx - 80)
        end   = min(len(text), idx + 120)
        excerpt = text[start:end].strip()

        # Check if any distant phrase neutralises this threat
        is_distant = any(phrase in text_lower for phrase in _DISTANT_PHRASES
                         if phrase in excerpt.lower())

        score = base_score if not is_distant else max(1, base_score // 4)

        # Check EU impact
        eu_impact = any(phrase in excerpt.lower() for phrase in _EU_IMPACT_PHRASES)

        # EU-present threats get a 20% boost
        if eu_impact and not is_distant:
            score = min(int(score * 1.2), MAX_HEALTH_SCORE)

        if keyword not in signals or signals[keyword].score < score:
            signals[keyword] = ThreatSignal(
                name=keyword.title(),
                score=score,
                eu_impact=eu_impact,
                source_text=excerpt,
            )

    return list(signals.values())


def _compute_health_score(threats: List[ThreatSignal]) -> int:
    """
    Aggregate threat signals into a single 0-50 score.

    Uses diminishing returns: each additional threat adds less than the last.
    This prevents 10 low-level threats from outscoring one pandemic signal.
    """
    if not threats:
        return 0

    sorted_scores = sorted([t.score for t in threats], reverse=True)

    total = 0
    for i, score in enumerate(sorted_scores):
        # Diminishing weight: 100%, 60%, 40%, 25%, 15%...
        weight = 1.0 / (1 + i * 0.8)
        total += score * weight

    return min(int(total), MAX_HEALTH_SCORE)


# ---------------------------------------------------------------------------
# ECDC CDTR fetcher
# ---------------------------------------------------------------------------

def fetch_latest_cdtr_summary(timeout: int = 15) -> dict:
    headers = {"User-Agent": "HAVEN/1.0 (health risk monitor; non-commercial)"}

    resp = requests.get(CDTR_LISTING_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    link_pattern = re.compile(
        r'href="(/en/publications-data/communicable-disease-threats-report-[^"]+)"'
    )
    matches = link_pattern.findall(html)
    if not matches:
        raise ValueError("No CDTR links found on the ECDC listing page.")

    latest_path = matches[0]
    latest_url  = f"https://www.ecdc.europa.eu{latest_path}"

    week_match = re.search(r'week-(\d+)', latest_path)
    year_match = re.search(r'(\d{4})', latest_path)
    week_label = f"Week {week_match.group(1)}, {year_match.group(1)}" \
        if week_match and year_match else "Unknown week"

    page_resp = requests.get(latest_url, headers=headers, timeout=timeout)
    page_resp.raise_for_status()
    page_html = page_resp.text

    # Extract meta description
    desc_pattern = re.compile(
        r'<meta name="description" content="([^"]+)"', re.IGNORECASE
    )
    desc_match = desc_pattern.search(page_html)
    full_meta = desc_match.group(1) if desc_match else ""

    # Keep ONLY the first sentence — everything before the sidebar bleeds in
    # The real content always ends with the threat list sentence
    # Split on the page title which always follows the description
    clean = full_meta.split("Communicable disease threats report,")[0].strip()
    if not clean:
        clean = full_meta[:400]

    period_match = re.search(r'(\d+\s*[–-]\s*\d+\s+\w+\s+\d{4}|\d+\s+\w+\s*[–-]\s*\d+\s+\w+\s+\d{4})', clean)
    period = period_match.group(1) if period_match else week_label

    return {
        "week_label":  week_label,
        "period":      period,
        "description": clean,     # ← only the actual threat sentence
        "url":         latest_url,
    }


def get_health_snapshot(timeout: int = 15) -> HealthSnapshot:
    """
    Full pipeline: fetch CDTR → extract threats → score → snapshot.

    No authentication required — ECDC data is fully public.

    Returns:
        HealthSnapshot ready to store in the DB and display in the UI.
    """
    cdtr = fetch_latest_cdtr_summary(timeout=timeout)
    threats = _extract_threats_from_text(cdtr["description"])
    score   = _compute_health_score(threats)
    level   = health_score_to_level(score)

    # Top threats: EU-impacting first, then by score
    top = sorted(threats, key=lambda t: (-int(t.eu_impact), -t.score))
    top_names = [t.name for t in top[:5]]

    return HealthSnapshot(
        week_label=   cdtr["week_label"],
        period=       cdtr["period"],
        threats=      threats,
        health_score= score,
        level=        level,
        top_threats=  top_names,
        cdtr_url=     cdtr["url"],
        fetched_at=   datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Simulate snapshot (for testing without network)
# ---------------------------------------------------------------------------

def simulate_health_snapshot(
    scenario: str = "routine",
) -> HealthSnapshot:
    """
    Generate a simulated HealthSnapshot for testing and demos.

    Args:
        scenario: One of 'routine', 'medium', 'pandemic'

    Returns:
        HealthSnapshot with realistic simulated data.
    """
    scenarios = {
        "routine": {
            "threats": [
                ThreatSignal("Influenza", 5, False, "Seasonal flu at baseline levels"),
                ThreatSignal("Sars-Cov-2", 15, True, "Low SARS-CoV-2 activity EU/EEA"),
                ThreatSignal("Rsv", 3, False, "RSV at low levels"),
            ],
            "week_label": "Week 15, 2026",
            "period":     "4–10 April 2026",
        },
        "medium": {
            "threats": [
                ThreatSignal("Measles", 12, True, "Measles outbreak reported in EU/EEA"),
                ThreatSignal("Mpox", 20, True, "Mpox cases reported in EU member states"),
                ThreatSignal("Avian Influenza", 25, False, "H5N1 monitoring ongoing globally"),
                ThreatSignal("Influenza", 5, False, "Seasonal activity"),
            ],
            "week_label": "Week 15, 2026",
            "period":     "4–10 April 2026",
        },
        "pandemic": {
            "threats": [
                ThreatSignal("Novel", 40, True, "Novel pathogen detected in EU/EEA member states"),
                ThreatSignal("Pandemic", 50, True, "WHO PHEIC declared — pandemic underway"),
                ThreatSignal("Sars-Cov-2", 30, True, "New SARS-CoV-2 variant driving surge"),
            ],
            "week_label": "Week 15, 2026",
            "period":     "4–10 April 2026",
        },
    }

    s = scenarios.get(scenario, scenarios["routine"])
    threats = s["threats"]
    score   = _compute_health_score(threats)

    return HealthSnapshot(
        week_label=   s["week_label"],
        period=       s["period"],
        threats=      threats,
        health_score= score,
        level=        health_score_to_level(score),
        top_threats=  [t.name for t in threats],
        cdtr_url=     "https://www.ecdc.europa.eu (simulated)",
        fetched_at=   datetime.now(timezone.utc).isoformat(),
    )
