"""
core/regional_risk_fetcher.py
------------------------------
Computes a Regional Risk Score (0-30) from two free, official sources:

    1. GDACS (Global Disaster Alert and Coordination System)
       UN + European Commission joint system.
       Free REST API, no authentication.
       Covers: earthquakes, floods, volcanoes, cyclones, wildfires, droughts.
       Score contribution: 0-15

    2. ReliefWeb API (UN OCHA)
       Free REST API, no authentication.
       Covers: conflict, humanitarian crises, displacement reports.
       Score contribution: 0-15

Why these sources instead of ACLED:
    ACLED API requires a paid license for non-institutional accounts.
    GDACS and ReliefWeb are fully free, programmatic, and from official
    UN/EC bodies — making them more appropriate for a citizen-facing
    emergency preparedness application.

Design notes (Option C):
    This score is NEVER added to weather or health scores.
    It is displayed as an independent gauge in the UI (0-30 scale).
    It feeds into the alert prioritizer as a separate signal.
"""

import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GDACS RSS feed — free, no auth, covers last 24h by default
GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"

# ReliefWeb API v2 — free, no auth, 1000 calls/day limit
RELIEFWEB_URL = "https://api.reliefweb.int/v2/reports"
RELIEFWEB_APP = "ildebrando-prepsense-itdf0"   # required appname parameter

MAX_REGIONAL_SCORE = 30

# Countries to monitor alongside Spain for regional spillover
SPAIN_REGION = ["Spain", "France", "Portugal", "Morocco", "Algeria"]

# GDACS alert level → score
_GDACS_ALERT_SCORES = {
    "Red":    15,
    "Orange":  8,
    "Green":   2,
}

# ReliefWeb theme keywords relevant to citizen preparedness
_CRISIS_THEMES = [
    "Conflict and Violence",
    "Refugees and Internally Displaced Persons",
    "Humanitarian Financing",
    "Food and Nutrition",
    "Shelter and Non-Food Items",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisasterEvent:
    title:       str
    event_type:  str    # EQ, FL, TC, VO, WF, DR
    alert_level: str    # Green / Orange / Red
    country:     str
    score:       int    # 0-15
    url:         str    # GDACS event URL


@dataclass
class CrisisReport:
    title:   str
    country: str
    date:    str
    theme:   str
    url:     str
    score:   int        # 0-15


@dataclass
class RegionalSnapshot:
    country:          str
    region_countries: List[str]
    disaster_events:  List[DisasterEvent]
    crisis_reports:   List[CrisisReport]
    disaster_score:   int   # 0-15 from GDACS
    crisis_score:     int   # 0-15 from ReliefWeb
    regional_score:   int   # 0-30 combined, capped
    level:            str   # MINIMAL / LOW / MEDIUM / HIGH
    fetched_at:       str


# ---------------------------------------------------------------------------
# Level mapping
# ---------------------------------------------------------------------------

def regional_score_to_level(score: int) -> str:
    if score >= 22: return "HIGH"
    if score >= 12: return "MEDIUM"
    if score >= 4:  return "LOW"
    return "MINIMAL"


# ---------------------------------------------------------------------------
# GDACS fetcher
# ---------------------------------------------------------------------------

def fetch_gdacs_events(
    region_countries: List[str],
    timeout: int = 15,
) -> List[DisasterEvent]:
    """
    Fetch recent disaster events from GDACS RSS feed.
    Filters to events affecting the specified countries.

    Returns:
        List of DisasterEvent objects, sorted by score descending.
    """
    resp = requests.get(GDACS_RSS_URL, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    channel = root.find("channel")
    if channel is None:
        return []

    # GDACS namespaces
    gdacs_ns = "https://www.gdacs.org"

    events = []
    countries_lower = [c.lower() for c in region_countries]

    for item in channel.findall("item"):
        title       = item.findtext("title", "")
        link        = item.findtext("link", "")
        country_el  = item.find(f"{{{gdacs_ns}}}country")
        alert_el    = item.find(f"{{{gdacs_ns}}}alertlevel")
        type_el     = item.find(f"{{{gdacs_ns}}}eventtype")

        country     = country_el.text.strip()  if country_el  is not None else ""
        alert_level = alert_el.text.strip()    if alert_el    is not None else "Green"
        event_type  = type_el.text.strip()     if type_el     is not None else ""

        # Filter: only events in our region
        if not any(c in country.lower() for c in countries_lower):
            continue

        score = _GDACS_ALERT_SCORES.get(alert_level, 2)
        events.append(DisasterEvent(
            title=       title,
            event_type=  event_type,
            alert_level= alert_level,
            country=     country,
            score=       score,
            url=         link,
        ))

    events.sort(key=lambda e: -e.score)
    return events


def compute_disaster_score(events: List[DisasterEvent]) -> int:
    """
    Aggregate GDACS events into a 0-15 score.
    Uses diminishing returns to prevent many small events from
    outscoring one major event.
    """
    if not events:
        return 0

    scores = sorted([e.score for e in events], reverse=True)
    total = 0.0
    for i, s in enumerate(scores):
        weight = 1.0 / (1 + i * 0.7)
        total += s * weight

    return min(int(total), 15)


# ---------------------------------------------------------------------------
# ReliefWeb fetcher
# ---------------------------------------------------------------------------

_THEME_SCORES = {
    "Conflict and Violence":                         8,
    "Refugees and Internally Displaced Persons":     6,
    "Disaster Management":                           8,
    "Food and Nutrition":                            4,
    "Agriculture":                                   2,
    "Protection and Human Rights":                   4,
    "Humanitarian Financing":                        2,
}

_SKIP_TITLE_KEYWORDS = ["location map", "carte", "bulletin", "country brief"]


def fetch_reliefweb_reports(
    region_countries: List[str],
    lookback_days: int = 30,
    limit: int = 10,
    timeout: int = 15,
) -> List[CrisisReport]:
    since = (date.today() - timedelta(days=lookback_days)).isoformat()

    payload = {
        "limit": limit,
        "sort": ["date.created:desc"],
        "filter": {
            "operator": "AND",
            "conditions": [
                {
                    "field":    "primary_country",
                    "value":    region_countries,
                    "operator": "OR",
                },
                {
                    "field": "date.created",
                    "value": {"from": f"{since}T00:00:00+00:00"},
                },
            ],
        },
        "fields": {
            "include": [
                "title",
                "primary_country.name",
                "date.created",
                "theme.name",
            ],
        },
    }

    resp = requests.post(
        RELIEFWEB_URL,
        params={"appname": RELIEFWEB_APP},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    reports = []
    for item in data.get("data", []):
        f         = item.get("fields", {})
        title     = f.get("title", "")
        created   = f.get("date", {}).get("created", "")[:10] if isinstance(f.get("date"), dict) else ""
        themes    = [t.get("name", "") for t in f.get("theme", [])] if f.get("theme") else []
        theme_str = themes[0] if themes else ""
        country   = f.get("primary_country", {}).get("name", "") if isinstance(f.get("primary_country"), dict) else ""

        if not theme_str:
            continue
        if any(kw in title.lower() for kw in _SKIP_TITLE_KEYWORDS):
            continue

        score = _THEME_SCORES.get(theme_str, 3)
        reports.append(CrisisReport(
            title=   title,
            country= country,
            date=    created,
            theme=   theme_str,
            url=     "",
            score=   score,
        ))

    # Deduplicate: same country + theme + date = same report in different languages
    seen = set()
    deduped = []
    for r in reports:
        key = (r.country, r.theme, r.date)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    reports = deduped

    reports.sort(key=lambda r: -r.score)
    return reports

def compute_crisis_score(reports: List[CrisisReport]) -> int:
    """
    Aggregate ReliefWeb reports into a 0-15 score.
    Diminishing returns so many low-score reports don't inflate the signal.
    """
    if not reports:
        return 0

    scores = sorted([r.score for r in reports], reverse=True)
    total  = 0.0
    for i, s in enumerate(scores):
        weight = 1.0 / (1 + i * 0.8)
        total += s * weight

    return min(int(total), 15)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_regional_snapshot(
    country: str = "Spain",
    region_countries: Optional[List[str]] = None,
    lookback_days: int = 30,
    timeout: int = 15,
) -> RegionalSnapshot:
    """
    Full pipeline: fetch GDACS + ReliefWeb → score → snapshot.
    No authentication required.

    Args:
        country:          Primary country name.
        region_countries: Countries to monitor (defaults to SPAIN_REGION).
        lookback_days:    How far back to look for ReliefWeb reports.
        timeout:          HTTP request timeout in seconds.

    Returns:
        RegionalSnapshot ready to store in the DB and display in the UI.
    """
    if region_countries is None:
        region_countries = SPAIN_REGION

    # Fetch both sources — graceful degradation if one fails
    disaster_events: List[DisasterEvent] = []
    crisis_reports:  List[CrisisReport]  = []

    try:
        disaster_events = fetch_gdacs_events(region_countries, timeout=timeout)
    except Exception as e:
        print(f"  [GDACS] fetch failed: {e}")

    try:
        crisis_reports = fetch_reliefweb_reports(
            region_countries, lookback_days=lookback_days, timeout=timeout
        )
    except Exception as e:
        print(f"  [ReliefWeb] fetch failed: {e}")

    disaster_score = compute_disaster_score(disaster_events)
    crisis_score   = compute_crisis_score(crisis_reports)
    regional_score = min(disaster_score + crisis_score, MAX_REGIONAL_SCORE)

    return RegionalSnapshot(
        country=          country,
        region_countries= region_countries,
        disaster_events=  disaster_events,
        crisis_reports=   crisis_reports,
        disaster_score=   disaster_score,
        crisis_score=     crisis_score,
        regional_score=   regional_score,
        level=            regional_score_to_level(regional_score),
        fetched_at=       datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Simulate snapshot (for testing without network)
# ---------------------------------------------------------------------------

def simulate_regional_snapshot(scenario: str = "calm") -> RegionalSnapshot:
    """
    Generate a simulated RegionalSnapshot for testing and demos.

    Args:
        scenario: 'calm' | 'medium' | 'crisis'
    """
    scenarios = {
        "calm": {
            "disasters": [],
            "reports":   [],
        },
        "medium": {
            "disasters": [
                DisasterEvent("Flood in Portugal", "FL", "Orange", "Portugal", 8, ""),
                DisasterEvent("Wildfire in Spain",  "WF", "Green",  "Spain",   2, ""),
            ],
            "reports": [
                CrisisReport("Humanitarian update Morocco", "Morocco", "2026-04-01", "Food and Nutrition", "", 4),
            ],
        },
        "crisis": {
            "disasters": [
                DisasterEvent("Major earthquake Spain", "EQ", "Red",    "Spain",   15, ""),
                DisasterEvent("Severe flood France",    "FL", "Orange", "France",   8, ""),
            ],
            "reports": [
                CrisisReport("Conflict displacement report", "Algeria", "2026-04-01", "Conflict and Violence", "", 8),
                CrisisReport("Refugee crisis update",        "Morocco", "2026-04-01", "Refugees and Internally Displaced Persons", "", 8),
            ],
        },
    }

    s = scenarios.get(scenario, scenarios["calm"])
    disasters = s["disasters"]
    reports   = s["reports"]
    d_score   = compute_disaster_score(disasters)
    c_score   = compute_crisis_score(reports)
    total     = min(d_score + c_score, MAX_REGIONAL_SCORE)

    return RegionalSnapshot(
        country=          "Spain",
        region_countries= SPAIN_REGION,
        disaster_events=  disasters,
        crisis_reports=   reports,
        disaster_score=   d_score,
        crisis_score=     c_score,
        regional_score=   total,
        level=            regional_score_to_level(total),
        fetched_at=       datetime.now(timezone.utc).isoformat(),
    )
