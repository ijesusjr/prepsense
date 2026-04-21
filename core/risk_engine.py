"""
core/risk_engine.py
-------------------
Computes a 0-100 risk score from OWM weather data and One Call 3.0 alerts.

Score composition:
    weather_severity (0-40)  — derived from OWM weather_id
    alert_severity   (0-60)  — derived from One Call alert tags
    wind_bonus       (0-15)  — based on wind speed
    rain_bonus       (0-10)  — based on rain intensity
    Total capped at 100.

Risk levels:
    LOW       0-19
    MEDIUM  20-44
    HIGH      45-69
    CRITICAL  70-100
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WeatherSnapshot:
    weather_id:    int
    wind_speed_ms: float = 0.0
    rain_1h_mm:    float = 0.0
    wind_gust_ms:  float = 0.0


@dataclass
class Alert:
    event:    str
    severity: str          # Minor / Moderate / Severe / Extreme / Unknown
    tags:     List[str] = field(default_factory=list)
    onset:    str = ""
    expires:  str = ""


@dataclass
class RiskResult:
    risk_score:       int
    risk_level:       str
    weather_severity: int
    alert_severity:   int
    wind_bonus:       int
    rain_bonus:       int

    def breakdown(self) -> dict:
        return {
            "weather_severity": self.weather_severity,
            "alert_severity":   self.alert_severity,
            "wind_bonus":       self.wind_bonus,
            "rain_bonus":       self.rain_bonus,
        }


# ---------------------------------------------------------------------------
# Severity maps
# ---------------------------------------------------------------------------

# OWM weather condition code groups → severity score (0-40)
# Reference: https://openweathermap.org/weather-conditions
_WEATHER_ID_SCORES = {
    range(200, 300): 35,   # Thunderstorm
    range(300, 400): 10,   # Drizzle
    range(600, 700): 25,   # Snow
    range(700, 800): 15,   # Atmosphere (fog, smoke, tornado group)
    range(800, 900): 0,    # Clear / clouds
}

# Specific overrides within rain group (500-599)
_RAIN_ID_SCORES = {
    500: 15,   # Light rain
    501: 20,   # Moderate rain
    502: 30,   # Heavy intensity rain
    503: 35,   # Very heavy rain
    504: 40,   # Extreme rain
    511: 25,   # Freezing rain
}

# Tornado gets its own maximum score
_TORNADO_ID = 781

# Alert severity label → score (0-60)
_ALERT_SEVERITY_SCORES = {
    "Minor":    10,
    "Moderate": 25,
    "Severe":   45,
    "Extreme":  60,
    "Unknown":   0,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def weather_id_to_severity(weather_id: int) -> int:
    """
    Convert an OWM weather condition ID to a base severity score (0-40).

    Args:
        weather_id: OWM weather condition code.

    Returns:
        Integer score between 0 and 40.
    """
    if weather_id == _TORNADO_ID:
        return 40

    if 500 <= weather_id <= 599:
        return _RAIN_ID_SCORES.get(weather_id, 20)

    for id_range, score in _WEATHER_ID_SCORES.items():
        if weather_id in id_range:
            return score

    return 0


def alert_severity_to_score(severity: str) -> int:
    """
    Convert a severity label to a numeric score (0-60).

    Args:
        severity: One of Minor / Moderate / Severe / Extreme / Unknown.

    Returns:
        Integer score between 0 and 60.
    """
    return _ALERT_SEVERITY_SCORES.get(severity, 0)


def _wind_bonus(wind_speed_ms: float) -> int:
    if wind_speed_ms >= 20:
        return 15
    if wind_speed_ms >= 14:
        return 8
    if wind_speed_ms >= 8:
        return 3
    return 0


def _rain_bonus(rain_1h_mm: float) -> int:
    if rain_1h_mm >= 20:
        return 10
    if rain_1h_mm >= 10:
        return 5
    return 0


def score_to_level(score: int) -> str:
    """
    Convert a numeric risk score to a level label.

    Args:
        score: Integer between 0 and 100.

    Returns:
        One of: LOW / MEDIUM / HIGH / CRITICAL
    """
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH"
    if score >= 20:
        return "MEDIUM"
    return "LOW"


def compute_risk_score(weather: WeatherSnapshot, alerts: List[Alert]) -> RiskResult:
    """
    Combine weather conditions and active alerts into a 0-100 risk score.

    Args:
        weather: Current weather snapshot.
        alerts:  List of active Alert objects.

    Returns:
        RiskResult with score, level, and component breakdown.
    """
    weather_severity = weather_id_to_severity(weather.weather_id)

    alert_severity = max(
        (alert_severity_to_score(a.severity) for a in alerts),
        default=0,
    )

    wb = _wind_bonus(weather.wind_speed_ms)
    rb = _rain_bonus(weather.rain_1h_mm)

    total = min(weather_severity + alert_severity + wb + rb, 100)

    return RiskResult(
        risk_score=total,
        risk_level=score_to_level(total),
        weather_severity=weather_severity,
        alert_severity=alert_severity,
        wind_bonus=wb,
        rain_bonus=rb,
    )


# ---------------------------------------------------------------------------
# Option C — Three independent signals (never summed)
# ---------------------------------------------------------------------------

@dataclass
class HavenSignals:
    """
    Container for the three independent risk signals in HAVEN.

    Option C design: signals are kept separate, displayed as three
    independent gauges in the UI. The alert prioritizer reads all three
    and surfaces actionable items from each dimension independently.

    Never add these scores together. Each has its own scale:
        weather_result.risk_score  → 0-100
        geo_score                  → 0-30
        health_score               → 0-50
    """
    weather:      RiskResult
    geo_score:    int           # 0-30  from GeopoliticalSnapshot.geo_score
    geo_trend:    str           # STABLE / INCREASING / DECREASING
    geo_country:  str
    health_score: int           # 0-50  from HealthSnapshot.health_score
    health_level: str           # ROUTINE / MEDIUM / HIGH / CRITICAL
    top_health_threats: List[str]

    def summary(self) -> dict:
        """Human-readable summary for the Streamlit dashboard."""
        return {
            "weather": {
                "score": self.weather.risk_score,
                "level": self.weather.risk_level,
                "scale": "0-100",
            },
            "geopolitical": {
                "score": self.geo_score,
                "level": _geo_level(self.geo_score),
                "trend": self.geo_trend,
                "scale": "0-30",
            },
            "health": {
                "score": self.health_score,
                "level": self.health_level,
                "top_threats": self.top_health_threats,
                "scale": "0-50",
            },
        }


def _geo_level(geo_score: int) -> str:
    """Convert geo_score (0-30) to a level label."""
    if geo_score >= 22:
        return "HIGH"
    if geo_score >= 12:
        return "MEDIUM"
    if geo_score >= 4:
        return "LOW"
    return "MINIMAL"
