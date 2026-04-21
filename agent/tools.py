"""
agent/tools.py
---------------
The four HAVEN agent tools.

Each tool is a pure function that takes structured inputs and returns
a structured result. The agent orchestrates them — tools themselves
have no LLM dependency.

Tools:
    get_risk_score       → current HavenSignals snapshot
    get_kit_gaps         → InventoryReport from current kit
    retrieve_guidelines  → top-k RAG chunks for a query
    run_scenario         → survival estimate for a given event + duration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Tool result data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskSummary:
    """Formatted snapshot of all three risk signals."""
    weather_score:  int
    weather_level:  str
    weather_detail: str          # alert description if any

    geo_score:      int
    geo_level:      str
    geo_trend:      str

    health_score:   int
    health_level:   str
    top_threats:    List[str]

    overall_concern: str         # LOW / MODERATE / HIGH / CRITICAL
    narrative:       str         # one-sentence human summary

    def to_prompt_str(self) -> str:
        lines = [
            "CURRENT RISK SIGNALS:",
            f"  Weather  : {self.weather_score}/100 — {self.weather_level}"
            + (f" ({self.weather_detail})" if self.weather_detail else ""),
            f"  Regional : {self.geo_score}/30  — {self.geo_level} (trend: {self.geo_trend})",
            f"  Health   : {self.health_score}/50  — {self.health_level}"
            + (f" | Active: {', '.join(self.top_threats[:2])}" if self.top_threats else ""),
            f"  Overall  : {self.overall_concern}",
        ]
        return "\n".join(lines)


@dataclass
class GapSummary:
    """Formatted inventory gap summary for prompt injection."""
    total_gaps:    int
    critical_gaps: int
    gap_score:     int
    gaps:          List[Dict]    # name, current, recommended, unit, priority, gap_pct
    has_gaps:      bool

    def to_prompt_str(self) -> str:
        if not self.has_gaps:
            return "KIT STATUS: Complete — no gaps detected."
        lines = [f"KIT GAPS ({self.total_gaps} items, score {self.gap_score}/100):"]
        for g in self.gaps:
            lines.append(
                f"  [{g['priority']:<6}] {g['name']:<25} "
                f"{g['current']:.1f}/{g['recommended']:.1f} {g['unit']} "
                f"({g['gap_pct']:.0f}% missing)"
            )
        return "\n".join(lines)


@dataclass
class ScenarioResult:
    """Survival estimate for a given emergency scenario."""
    event_type:     str          # power_outage / flood / earthquake / heat_wave / general
    duration_hours: int
    people:         int

    # Per-resource survival estimates
    water_hours:    float        # how many hours water supply lasts
    food_hours:     float
    light_hours:    float        # flashlight / candles
    comms_ok:       bool         # has radio or charged power bank
    meds_ok:        bool         # has medication supply

    # Critical gaps — items that run out before duration ends
    critical_gaps:  List[str]
    survival_pct:   int          # rough % of duration covered (0-100)

    narrative:      str          # human-readable summary
    recommendations: List[str]   # specific actions to take

    def to_prompt_str(self) -> str:
        lines = [
            f"SCENARIO ANALYSIS: {self.event_type.replace('_', ' ').title()} "
            f"({self.duration_hours}h, {self.people} person(s))",
            f"  Water supply    : lasts {self.water_hours:.0f}h "
            f"({'OK' if self.water_hours >= self.duration_hours else 'INSUFFICIENT'})",
            f"  Food supply     : lasts {self.food_hours:.0f}h "
            f"({'OK' if self.food_hours >= self.duration_hours else 'INSUFFICIENT'})",
            f"  Lighting        : {'available' if self.light_hours > 0 else 'NONE'}",
            f"  Communications  : {'OK (radio/power bank)' if self.comms_ok else 'NONE — cannot receive alerts'}",
            f"  Medication      : {'stocked' if self.meds_ok else 'MISSING'}",
            f"  Coverage        : {self.survival_pct}% of {self.duration_hours}h scenario",
        ]
        if self.critical_gaps:
            lines.append(f"  Critical gaps   : {', '.join(self.critical_gaps)}")
        lines.append(f"  Summary: {self.narrative}")
        return "\n".join(lines)


@dataclass
class GuidelineResult:
    """Retrieved chunks from the RAG knowledge base."""
    query:    str
    chunks:   List[Any]          # RetrievedChunk objects
    context:  str                # formatted for prompt injection

    def to_prompt_str(self) -> str:
        return f"GUIDELINE SOURCES:\n{self.context}"


# ---------------------------------------------------------------------------
# Tool: get_risk_score
# ---------------------------------------------------------------------------

def get_risk_score(signals) -> RiskSummary:
    """
    Extract and format the current HavenSignals into a RiskSummary.

    Args:
        signals: HavenSignals dataclass from the risk engine.

    Returns:
        RiskSummary with all three dimensions and an overall concern level.
    """
    # Determine overall concern from the highest individual signal
    weather_norm = signals.weather.risk_score / 100
    geo_norm     = signals.geo_score / 30
    health_norm  = signals.health_score / 50
    max_norm     = max(weather_norm, geo_norm, health_norm)

    if max_norm >= 0.7:
        overall = "CRITICAL"
    elif max_norm >= 0.45:
        overall = "HIGH"
    elif max_norm >= 0.2:
        overall = "MODERATE"
    else:
        overall = "LOW"

    # Build narrative
    concerns = []
    if signals.weather.risk_score >= 45:
        concerns.append(f"weather at {signals.weather.risk_level}")
    if signals.geo_score >= 12:
        concerns.append(f"regional risk {signals.geo_level}")
    if signals.health_score >= 10:
        concerns.append(f"health {signals.health_level}")

    if concerns:
        narrative = f"Current concern: {', '.join(concerns)}."
    else:
        narrative = "All signals at routine levels — no immediate concern."

    # Get weather alert detail from risk_result if available
    weather_detail = ""
    if hasattr(signals.weather, "alert_severity") and signals.weather.alert_severity > 0:
        weather_detail = f"alert active, severity score {signals.weather.alert_severity}"

    return RiskSummary(
        weather_score=  signals.weather.risk_score,
        weather_level=  signals.weather.risk_level,
        weather_detail= weather_detail,
        geo_score=      signals.geo_score,
        geo_level=      _geo_level(signals.geo_score),
        geo_trend=      signals.geo_trend,
        health_score=   signals.health_score,
        health_level=   signals.health_level,
        top_threats=    signals.top_health_threats,
        overall_concern= overall,
        narrative=      narrative,
    )


def _geo_level(score: int) -> str:
    if score >= 22: return "HIGH"
    if score >= 12: return "MEDIUM"
    if score >= 4:  return "LOW"
    return "MINIMAL"


# ---------------------------------------------------------------------------
# Tool: get_kit_gaps
# ---------------------------------------------------------------------------

def get_kit_gaps(inv_report) -> GapSummary:
    """
    Format an InventoryReport into a GapSummary for prompt injection.

    Args:
        inv_report: InventoryReport from analyze_inventory().

    Returns:
        GapSummary with structured gap data.
    """
    gaps_data = []
    for g in inv_report.gaps:
        gaps_data.append({
            "name":        g.name,
            "current":     g.current,
            "recommended": g.recommended,
            "unit":        g.unit,
            "priority":    g.priority,
            "gap_pct":     g.gap_pct,
            "category":    g.category,
        })

    return GapSummary(
        total_gaps=    len(inv_report.gaps),
        critical_gaps= sum(1 for g in inv_report.gaps if g.priority == "HIGH"),
        gap_score=     inv_report.total_gap_score,
        gaps=          gaps_data,
        has_gaps=      len(inv_report.gaps) > 0,
    )


# ---------------------------------------------------------------------------
# Tool: retrieve_guidelines
# ---------------------------------------------------------------------------

def retrieve_guidelines(retriever, query: str, k: int = 4) -> GuidelineResult:
    """
    Retrieve relevant chunks from the RAG knowledge base.

    Args:
        retriever: HavenRetriever instance.
        query:     Natural language query to search for.
        k:         Number of chunks to retrieve.

    Returns:
        GuidelineResult with chunks and formatted context string.
    """
    chunks = retriever.query(query, k=k, min_score=0.15)

    context = "\n\n".join(
        f"[{i+1}] Source: {c.source}, Page {c.page} (score: {c.score:.2f})\n{c.text}"
        for i, c in enumerate(chunks)
    )

    return GuidelineResult(
        query=   query,
        chunks=  chunks,
        context= context if context else "No relevant guidelines found for this query.",
    )


# ---------------------------------------------------------------------------
# Tool: run_scenario
# ---------------------------------------------------------------------------

# Water consumption rates by scenario (litres per person per day)
_WATER_RATES = {
    "power_outage": 3.0,   # no pumping, drinking + basic hygiene
    "flood":        4.0,   # contamination risk, need more clean water
    "earthquake":   3.5,   # dust, injury — slightly higher
    "heat_wave":    5.0,   # hydration critical in heat
    "general":      3.0,   # EU standard baseline
}

# Food consumption (days per person)
_FOOD_RATE_DAYS = 1.0      # 1 unit of food = 1 person-day

# Event-specific critical items beyond water/food
_EVENT_CRITICAL_ITEMS = {
    "power_outage": ["flashlight", "battery-powered radio", "power bank", "candles"],
    "flood":        ["waterproof bag", "spare keys", "documents"],
    "earthquake":   ["first aid kit", "whistle", "dust mask"],
    "heat_wave":    ["water", "fan", "medication"],
    "general":      ["flashlight", "radio", "first aid kit"],
}


def run_scenario(
    inv_report,
    event_type:     str = "general",
    duration_hours: int = 72,
    people:         int = 1,
) -> ScenarioResult:
    """
    Estimate how long the current kit covers a given emergency scenario.

    Args:
        inv_report:     InventoryReport from analyze_inventory().
        event_type:     One of: power_outage, flood, earthquake, heat_wave, general.
        duration_hours: Duration of the emergency in hours.
        people:         Number of people in the household.

    Returns:
        ScenarioResult with per-resource survival estimates and recommendations.
    """
    event_type = event_type.lower().replace(" ", "_")
    if event_type not in _WATER_RATES:
        event_type = "general"

    water_rate_per_day = _WATER_RATES[event_type]
    water_rate_per_hour = water_rate_per_day / 24

    # Build a lookup of current kit quantities by category and name.
    # Use all_items (full inventory) so fully-stocked items are included.
    # Fall back to gaps list for InventoryReport objects built without all_items.
    kit_lookup: Dict[str, float] = {}
    source_items = inv_report.all_items if inv_report.all_items else inv_report.gaps
    for item in source_items:
        current = getattr(item, "quantity", None) or getattr(item, "current", 0)
        kit_lookup[item.name.lower()] = current
        kit_lookup[item.category.lower()] = kit_lookup.get(item.category.lower(), 0) + current

    # Water survival
    water_available = kit_lookup.get("drinking water", kit_lookup.get("water", 0))
    water_needed_total = water_rate_per_hour * duration_hours * people
    if water_available > 0:
        water_hours = (water_available / (water_rate_per_day * people)) * 24
    else:
        water_hours = 0.0

    # Food survival (assume 1 unit = 1 day for 1 person)
    food_available = kit_lookup.get("non-perishable food", kit_lookup.get("food", 0))
    if food_available > 0:
        food_hours = (food_available / people) * 24
    else:
        food_hours = 0.0

    # Lighting
    has_flashlight = kit_lookup.get("flashlight", kit_lookup.get("light", 0)) > 0
    light_hours = duration_hours if has_flashlight else 0.0

    # Communications
    has_radio     = kit_lookup.get("battery-powered radio", kit_lookup.get("comms", 0)) > 0
    has_powerbank = kit_lookup.get("power bank", 0) > 0
    comms_ok      = has_radio or has_powerbank

    # Medication
    meds_available = kit_lookup.get("regular medication", kit_lookup.get("meds", 0))
    meds_ok        = meds_available > 0

    # Identify critical gaps (run out before scenario ends)
    critical_gaps = []
    if water_hours < duration_hours:
        critical_gaps.append(
            f"Water runs out at hour {water_hours:.0f} "
            f"(need {water_needed_total:.1f}L, have {water_available:.1f}L)"
        )
    if food_hours < duration_hours:
        critical_gaps.append(
            f"Food runs out at hour {food_hours:.0f}"
        )
    if not comms_ok:
        critical_gaps.append("No communications device — cannot receive emergency alerts")
    if not meds_ok:
        critical_gaps.append("No medication stock")

    # Overall survival coverage score
    coverages = [
        min(water_hours / duration_hours, 1.0),
        min(food_hours  / duration_hours, 1.0),
        1.0 if comms_ok else 0.0,
        1.0 if meds_ok  else 0.5,   # meds weighted less — not everyone needs daily meds
    ]
    survival_pct = int(sum(coverages) / len(coverages) * 100)

    # Build narrative
    event_label = event_type.replace("_", " ")
    if not critical_gaps:
        narrative = (
            f"Your kit fully covers a {duration_hours}h {event_label} "
            f"for {people} person(s). Well prepared."
        )
    else:
        first_failure = min(
            [water_hours if water_hours < duration_hours else duration_hours,
             food_hours  if food_hours  < duration_hours else duration_hours],
        )
        narrative = (
            f"Your kit covers approximately {survival_pct}% of a "
            f"{duration_hours}h {event_label} for {people} person(s). "
            f"First critical shortage at hour {first_failure:.0f}."
        )

    # Recommendations
    recommendations = []
    if water_hours < duration_hours:
        needed_extra = round(water_needed_total - water_available, 1)
        recommendations.append(
            f"Store {needed_extra}L more water "
            f"({water_rate_per_day}L/person/day × {people} people × "
            f"{duration_hours/24:.0f} days)"
        )
    if food_hours < duration_hours:
        days_short = round((duration_hours - food_hours) / 24, 1)
        recommendations.append(f"Add {days_short * people:.1f} more days of food per person")
    if not comms_ok:
        recommendations.append("Get a battery-powered radio or charged power bank")
    if not meds_ok:
        recommendations.append("Keep a 7-day supply of regular medication")
    if event_type == "power_outage" and not has_flashlight:
        recommendations.append("Add flashlight with spare batteries or candles")

    return ScenarioResult(
        event_type=      event_type,
        duration_hours=  duration_hours,
        people=          people,
        water_hours=     water_hours,
        food_hours=      food_hours,
        light_hours=     light_hours,
        comms_ok=        comms_ok,
        meds_ok=         meds_ok,
        critical_gaps=   critical_gaps,
        survival_pct=    survival_pct,
        narrative=       narrative,
        recommendations= recommendations,
    )
