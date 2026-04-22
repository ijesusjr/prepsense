"""
app_cloud.py
-------------
HAVEN — Streamlit Cloud deployment (Option A: direct imports, no FastAPI).

Differences from app.py (FastAPI version):
    - No HTTP calls — all modules imported directly
    - st.cache_data(ttl=...) replaces APScheduler for signal refresh
    - st.session_state + SQLite replace AppState singleton for kit persistence
    - Secrets read from st.secrets (Streamlit Cloud) with os.getenv fallback (local)

Architecture note:
    The full FastAPI stack (api/main.py) runs locally with `uvicorn api.main:app`.
    This file is the Streamlit Cloud deployment version — same logic, simpler runtime.
"""

import os
import copy
import time
import sqlite3
import requests
from datetime import date, datetime, timezone

import streamlit as st
import folium
from streamlit_folium import st_folium

from core.regions import is_supported, get_region
from core.inventory_analyzer import KitItem, analyze_inventory
from core.alert_prioritizer import prioritize
from agent.tools import run_scenario

# ---------------------------------------------------------------------------
# Secret / env helper — works both locally (.env) and on Streamlit Cloud
# ---------------------------------------------------------------------------

def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to os.getenv."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEVEL_COLORS = {
    "LOW":      "#2ecc71",
    "MINIMAL":  "#2ecc71",
    "ROUTINE":  "#2ecc71",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e74c3c",
    "CRITICAL": "#8e44ad",
}

URGENCY_COLORS = {
    "IMMEDIATE": "#e74c3c",
    "SOON":      "#f39c12",
    "ROUTINE":   "#95a5a6",
}

CATEGORY_ICONS = {
    "WEATHER":    "🌤",
    "COMBINED":   "⚡",
    "HEALTH_KIT": "🦠💊",
    "EXPIRY":     "⏰",
    "KIT_GAP":    "📦",
    "HEALTH":     "🦠",
    "GEO":        "⚔",
}

EVENT_TYPES = {
    "General emergency": "general",
    "Power outage":      "power_outage",
    "Flood":             "flood",
    "Earthquake":        "earthquake",
    "Heat wave":         "heat_wave",
}

PER_PERSON_ITEMS = {
    "Drinking water":      9.0,
    "Non-perishable food": 3.0,
    "Regular medication":  7.0,
    "Hand sanitizer":      1.0,
    "Cash":                70.0,
}

DEFAULT_KIT = [
    KitItem(name="Drinking water",        category="water",   quantity=2.0,  eu_recommended=9.0,  unit="liters", expiry_date=None),
    KitItem(name="Non-perishable food",   category="food",    quantity=1.0,  eu_recommended=3.0,  unit="days",   expiry_date=None),
    KitItem(name="First aid kit",         category="meds",    quantity=1.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
    KitItem(name="Regular medication",    category="meds",    quantity=0.0,  eu_recommended=7.0,  unit="days",   expiry_date=None),
    KitItem(name="Battery-powered radio", category="comms",   quantity=0.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
    KitItem(name="Flashlight",            category="light",   quantity=1.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
    KitItem(name="Cash",                  category="cash",    quantity=20.0, eu_recommended=70.0, unit="EUR",    expiry_date=None),
    KitItem(name="Hand sanitizer",        category="hygiene", quantity=0.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
    KitItem(name="Copies of documents",   category="docs",    quantity=0.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
    KitItem(name="Spare keys",            category="tools",   quantity=0.0,  eu_recommended=1.0,  unit="units",  expiry_date=None),
]

DB_PATH = _secret("DB_PATH", "haven.db")


# ---------------------------------------------------------------------------
# SQLite persistence (same as api/state.py)
# ---------------------------------------------------------------------------

def _db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kit_items (
            name           TEXT PRIMARY KEY,
            category       TEXT NOT NULL,
            quantity       REAL NOT NULL,
            eu_recommended REAL NOT NULL,
            unit           TEXT NOT NULL,
            expiry_date    TEXT,
            updated_at     TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _db_save(kit_items: list, household_size: int):
    conn = sqlite3.connect(DB_PATH)
    for item in kit_items:
        conn.execute("""
            INSERT INTO kit_items (name, category, quantity, eu_recommended, unit, expiry_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                quantity    = excluded.quantity,
                expiry_date = excluded.expiry_date,
                updated_at  = datetime('now')
        """, (
            item.name, item.category, item.quantity,
            item.eu_recommended, item.unit,
            item.expiry_date.isoformat() if item.expiry_date else None,
        ))
    conn.execute("""
        INSERT INTO settings (key, value) VALUES ('household_size', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (str(household_size),))
    conn.commit()
    conn.close()


def _db_load() -> tuple:
    """Returns (kit_items, household_size). Falls back to defaults if DB empty."""
    try:
        _db_init()
        conn  = sqlite3.connect(DB_PATH)
        rows  = conn.execute(
            "SELECT name, category, quantity, eu_recommended, unit, expiry_date "
            "FROM kit_items"
        ).fetchall()
        setting = conn.execute(
            "SELECT value FROM settings WHERE key = 'household_size'"
        ).fetchone()
        conn.close()

        if not rows:
            return list(DEFAULT_KIT), 1

        items = [
            KitItem(
                name=name, category=cat, quantity=qty,
                eu_recommended=rec, unit=unit,
                expiry_date=date.fromisoformat(exp) if exp else None,
            )
            for name, cat, qty, rec, unit, exp in rows
        ]
        return items, int(setting[0]) if setting else 1
    except Exception:
        return list(DEFAULT_KIT), 1


# ---------------------------------------------------------------------------
# Kit state — initialise once per session from DB
# ---------------------------------------------------------------------------

def _init_kit():
    if "kit_items" not in st.session_state:
        items, size = _db_load()
        st.session_state.kit_items      = items
        st.session_state.household_size = size

    n = st.session_state.household_size
    scaled = []
    for item in st.session_state.kit_items:
        s = copy.copy(item)
        s.eu_recommended = (
            item.eu_recommended * n
            if item.name in PER_PERSON_ITEMS else
            item.eu_recommended
        )
        scaled.append(s)
    return analyze_inventory(scaled)


# ---------------------------------------------------------------------------
# Cached signal fetchers — TTL replaces APScheduler
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_weather(lat: float, lon: float) -> dict:
    """Fetch weather risk — cached 60 minutes."""
    api_key = _secret("OWM_API_KEY")
    if not api_key:
        return _stub_weather()
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/3.0/onecall",
            params={"lat": lat, "lon": lon, "appid": api_key,
                    "exclude": "minutely,hourly,daily", "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data.get("current", {})
        alerts  = data.get("alerts", [])

        from core.risk_engine import WeatherSnapshot, Alert, compute_risk_score
        snap = WeatherSnapshot(
            weather_id=    current.get("weather", [{}])[0].get("id", 800),
            wind_speed_ms= current.get("wind_speed", 0.0),
            rain_1h_mm=    current.get("rain", {}).get("1h", 0.0),
            wind_gust_ms=  current.get("wind_gust", 0.0),
        )
        alert_objs = [
            Alert(event=a.get("event", ""), severity=_infer_severity(a),
                  tags=a.get("tags", []))
            for a in alerts
        ]
        risk = compute_risk_score(snap, alert_objs)
        weather_info = current.get("weather", [{}])[0]
        return {
            "score": risk.risk_score,
            "level": risk.risk_level,
            "breakdown": {
                "weather_severity": risk.weather_severity,
                "alert_severity":   risk.alert_severity,
                "wind_bonus":       risk.wind_bonus,
                "rain_bonus":       risk.rain_bonus,
            },
            "condition": {
                "main":        weather_info.get("main", ""),
                "description": weather_info.get("description", ""),
            },
            "alerts": [
                {
                    "event":       _clean_event(a.get("event", "")),
                    "severity":    _infer_severity(a),
                    "description": a.get("description", ""),
                    "sender":      a.get("sender_name", ""),
                }
                for a in alerts
            ],
        }
    except Exception:
        return _stub_weather()


def _stub_weather() -> dict:
    return {
        "score": 0, "level": "LOW",
        "breakdown": {"weather_severity": 0, "alert_severity": 0,
                      "wind_bonus": 0, "rain_bonus": 0},
        "condition": {"main": "", "description": ""},
        "alerts": [],
    }


def _infer_severity(alert: dict) -> str:
    tags = [t.lower() for t in alert.get("tags", [])]
    if "extreme"  in tags: return "Extreme"
    if "severe"   in tags: return "Severe"
    if "moderate" in tags: return "Moderate"
    return "Minor"


_AWARENESS_TYPES = {
    "1": "Wind", "2": "Snow/Ice", "3": "Thunderstorm", "4": "Fog",
    "5": "High Temperature", "6": "Low Temperature", "7": "Rain",
    "8": "Coastal Event", "9": "Forest Fire", "10": "Avalanche", "11": "Flood",
}

def _clean_event(event: str) -> str:
    """Replace OWM awareness_type/level key=value strings with readable labels."""
    import re
    at = re.search(r"awareness_type=(\d+)", event)
    if at:
        return _AWARENESS_TYPES.get(at.group(1), "Weather Alert")
    return event.strip() or "Weather Alert"


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_regional(country: str) -> dict:
    """Fetch regional risk — cached 7 days."""
    try:
        from core.regional_risk_fetcher import get_regional_snapshot, _THEME_SCORES
        region = get_region(country)
        geo    = get_regional_snapshot(country=country, region_countries=region)
        gdacs_alerts = [
            {"title": e.title, "alert_level": e.alert_level, "event_type": e.event_type, "country": e.country}
            for e in geo.disaster_events
        ]
        crisis_themes = list(dict.fromkeys(
            r.theme for r in geo.crisis_reports if r.theme in _THEME_SCORES
        ))
        return {
            "score":         geo.regional_score,
            "level":         _geo_level(geo.regional_score),
            "trend":         "STABLE",
            "country":       geo.country,
            "gdacs_alerts":  gdacs_alerts,
            "crisis_themes": crisis_themes,
        }
    except Exception:
        return {"score": 0, "level": "MINIMAL", "trend": "STABLE", "country": country,
                "gdacs_alerts": [], "crisis_themes": []}


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_health() -> dict:
    """Fetch health risk — cached 7 days."""
    try:
        from core.health_fetcher import get_health_snapshot
        hs = get_health_snapshot()
        return {
            "score":       hs.health_score,
            "level":       hs.level,
            "top_threats": hs.top_threats,
        }
    except Exception:
        return {"score": 0, "level": "ROUTINE", "top_threats": []}


def _geo_level(score: int) -> str:
    if score >= 22: return "HIGH"
    if score >= 12: return "MEDIUM"
    if score >= 4:  return "LOW"
    return "MINIMAL"


# ---------------------------------------------------------------------------
# Agent / RAG — initialise once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_agent():
    """Load retriever + LLM once per deployment instance."""
    from pathlib import Path
    try:
        from rag.retriever import HavenRetriever
        from rag.llm import HavenLLM
        from agent.agent import HavenAgent

        faiss_dir = Path("data/faiss")
        retriever = HavenRetriever.from_disk(
            index_path=str(faiss_dir / "index.bin"),
            meta_path= str(faiss_dir / "chunks.json"),
        )
        backend = _secret("LLM_BACKEND", "groq")
        llm     = HavenLLM(backend=backend)
        return retriever, llm
    except Exception as e:
        return None, None


# ---------------------------------------------------------------------------
# UI helpers (identical to app.py)
# ---------------------------------------------------------------------------

def level_badge(level: str) -> str:
    color = LEVEL_COLORS.get(level, "#95a5a6")
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.8em;font-weight:bold">{level}</span>')


def score_gauge(label: str, score: int, max_score: int, level: str):
    color = LEVEL_COLORS.get(level, "#95a5a6")
    pct   = int(score / max_score * 100)
    badge = level_badge(level)
    st.markdown(f"""
    <div style="background:#1e1e2e;border-radius:8px;padding:14px 18px;margin:4px 0">
      <div style="font-size:1.2em;font-weight:bold;color:#aaa;margin-bottom:4px">{label}</div>
      <div style="font-size:1.6em;font-weight:bold;color:{color}">{score}
        <span style="font-size:0.55em;color:#888">/ {max_score}</span>
      </div>
      <div style="background:#333;border-radius:4px;height:6px;margin:8px 0">
        <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
      </div>
      {badge}
    </div>
    """, unsafe_allow_html=True)


def alert_card_html(alert: dict) -> str:
    color = URGENCY_COLORS.get(alert["urgency"], "#95a5a6")
    icon  = CATEGORY_ICONS.get(alert["category"], "ℹ")
    return (
        f'<div style="border-left:3px solid {color};padding:8px 12px;margin:6px 0;'
        f'background:#1a1a2e;border-radius:0 6px 6px 0">'
        f'<div style="font-size:0.75em;color:{color};font-weight:bold;margin-bottom:2px">'
        f'{icon} {alert["category"]} · {alert["urgency"]} · score {alert["priority_score"]}'
        f'</div>'
        f'<div style="font-size:0.9em">{alert["message"]}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HAVEN",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; }
  .alert-scroll { max-height: 420px; overflow-y: auto; padding-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise kit state
# ---------------------------------------------------------------------------

inv_report = _init_kit()

# ---------------------------------------------------------------------------
# Load signals (cached)
# ---------------------------------------------------------------------------

if "user_lat" not in st.session_state:
    st.session_state.user_lat     = float(_secret("LAT", "41.3851"))
    st.session_state.user_lon     = float(_secret("LON", "2.1734"))
    st.session_state.user_city    = _secret("CITY", "Barcelona")
    st.session_state.user_country = _secret("COUNTRY", "Spain")

lat     = st.session_state.user_lat
lon     = st.session_state.user_lon
country = st.session_state.user_country

weather_data  = _fetch_weather(lat, lon)
regional_data = _fetch_regional(country)
health_data   = _fetch_health()

# Build HavenSignals for the alert prioritizer
from core.risk_engine import RiskResult, HavenSignals
_risk_result = RiskResult(
    risk_score=       weather_data["score"],
    risk_level=       weather_data["level"],
    weather_severity= weather_data["breakdown"]["weather_severity"],
    alert_severity=   weather_data["breakdown"]["alert_severity"],
    wind_bonus=       weather_data["breakdown"]["wind_bonus"],
    rain_bonus=       weather_data["breakdown"]["rain_bonus"],
)
signals = HavenSignals(
    weather=            _risk_result,
    geo_score=          regional_data["score"],
    geo_trend=          regional_data["trend"],
    geo_country=        regional_data["country"],
    health_score=       health_data["score"],
    health_level=       health_data["level"],
    top_health_threats= health_data["top_threats"],
)

# ---------------------------------------------------------------------------
# ── HEADER ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.image("haven_logo_horizontal.png")
with col_refresh:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.toast("Cache cleared — signals refreshing...", icon="🔄")
        time.sleep(1)
        st.rerun()
    st.markdown('<p style="font-size:0.65rem;color:#888;margin-top:4px;text-align:center;">🕐 Weather: 60 min cache · Regional/Health: 7 day cache</p>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# ── ROW 1: MAP | RISK INDEXES ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

col_map, col_risk = st.columns([0.55, 0.45])

# ── Map ──────────────────────────────────────────────────────────────────────
with col_map:
    st.markdown("### 📍 Location")

    m = folium.Map(location=[54, 15], zoom_start=4,
                   tiles="CartoDB positron", min_zoom=3, max_zoom=12)

    folium.Marker(
        location=[lat, lon],
        popup=st.session_state.user_city,
        tooltip="Your location",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    map_data = st_folium(m, width="100%", height=600,
                         returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        city = f"{clicked_lat:.2f}, {clicked_lon:.2f}"
        clicked_country = ""
        geocode_error   = ""
        try:
            geo      = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"lat": clicked_lat, "lon": clicked_lon,
                        "format": "json", "zoom": 10},
                headers={"User-Agent": "HAVEN/1.0", "Accept-Language": "en"},
                timeout=5,
            ).json()
            geo_addr        = geo.get("address", {})
            city            = (geo_addr.get("city") or geo_addr.get("town")
                               or geo_addr.get("village")
                               or geo_addr.get("municipality")
                               or geo.get("display_name", "").split(",")[0])
            clicked_country = geo_addr.get("country", "")
        except Exception as e:
            geocode_error = str(e)

        if geocode_error:
            st.caption(f"⚠ Geocoding unavailable ({geocode_error}) — you can still set the location by coordinates.")

        if clicked_country and not is_supported(clicked_country):
            st.warning(
                f"⚠ **{clicked_country}** not in supported region list — "
                f"regional risk uses default EU neighbours."
            )
        elif clicked_country:
            st.success(f"✓ **{clicked_country}** fully supported.")

        loc_col, btn_col = st.columns([3, 1])
        with loc_col:
            st.info(f"📍 **{city}**, {clicked_country} "
                    f"({clicked_lat:.3f}, {clicked_lon:.3f})")
        with btn_col:
            if st.button("✓ Set", type="primary", use_container_width=True):
                st.session_state.user_lat     = clicked_lat
                st.session_state.user_lon     = clicked_lon
                st.session_state.user_city    = city
                st.session_state.user_country = clicked_country
                # Clear weather + regional cache so they re-fetch for new location
                st.cache_data.clear()
                st.toast(f"📍 {city} — signals refreshing...", icon="📍")
                time.sleep(1)
                st.rerun()

    country_str = st.session_state.user_country
    city_str    = st.session_state.user_city
    st.markdown(
        f'<div style="font-size:0.82em;color:#aaa;margin-top:4px">'
        f'📍 <b style="color:#ddd">{city_str}'
        + (f', {country_str}' if country_str else '')
        + f'</b> · {lat:.4f}, {lon:.4f}'
        f'<br><span style="font-size:0.9em">Weather and regional signals refresh '
        f'automatically after location change. '
        f'Health signal is EU-wide and does not change with location.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Risk indexes ─────────────────────────────────────────────────────────────
with col_risk:
    st.markdown("### Risk Signals")

    w = weather_data
    r = regional_data
    h = health_data

    score_gauge("🌤 Weather Risk", w["score"], 100, w["level"])
    if w["breakdown"]["alert_severity"] > 0:
        st.caption(f"Active alert · severity {w['breakdown']['alert_severity']}")

    with st.expander("📊 Weather score breakdown", expanded=False):
        bd            = w["breakdown"]
        cond          = w.get("condition", {})
        active_alerts = w.get("alerts", [])
        severity_colors = {"Extreme": "#8e44ad", "Severe": "#e74c3c", "Moderate": "#f39c12", "Minor": "#f1c40f"}
        rows = [
            ("⚡ Active alert",  bd["alert_severity"],   60, "National alert tags (Minor/Moderate/Severe/Extreme)"),
            ("🌡 Condition",     bd["weather_severity"], 40, "OWM weather ID — thunderstorm, snow, fog, rain…"),
            ("💨 Wind bonus",    bd["wind_bonus"],       15, "Wind speed > 10 m/s"),
            ("🌧 Rain bonus",    bd["rain_bonus"],       10, "Rainfall > 2 mm/h"),
        ]
        for label, score, max_s, note in rows:
            color = "#e74c3c" if score >= max_s * 0.6 else "#f39c12" if score > 0 else "#555"
            pct   = int(score / max_s * 100) if max_s > 0 else 0
            st.markdown(
                f'<div style="margin:6px 0">'
                f'<div style="font-size:0.8em;color:#aaa">{label} '
                f'<span style="color:{color};font-weight:bold">+{score}</span>'
                f'<span style="color:#666"> / {max_s}</span></div>'
                f'<div style="background:#333;border-radius:3px;height:4px;margin:3px 0">'
                f'<div style="background:{color};width:{pct}%;height:4px;border-radius:3px"></div>'
                f'</div><div style="font-size:0.72em;color:#666">{note}</div></div>',
                unsafe_allow_html=True,
            )
            if label.startswith("⚡") and active_alerts:
                alerts_html = ""
                for al in active_alerts:
                    col  = severity_colors.get(al["severity"], "#f39c12")
                    desc = al["description"][:180] + "…" if len(al.get("description", "")) > 180 else al.get("description", "")
                    sender = f'<span style="color:#666"> · {al["sender"]}</span>' if al.get("sender") else ""
                    alerts_html += (
                        f'<div style="margin-bottom:6px;padding:6px 8px;background:#1a1a1a;border-left:3px solid {col};border-radius:3px">'
                        f'<div style="font-size:0.82em;font-weight:bold;color:white">{al["event"]} '
                        f'<span style="font-weight:normal;color:{col}">({al["severity"]})</span>{sender}</div>'
                        + (f'<div style="font-size:0.75em;color:#888;margin-top:3px">{desc}</div>' if desc else '')
                        + '</div>'
                    )
                st.markdown(f'<div style="margin:2px 0 8px 0">{alerts_html}</div>', unsafe_allow_html=True)
            if label.startswith("🌡") and cond.get("description"):
                st.markdown(
                    f'<div style="margin:2px 0 8px 0;padding:4px 8px;background:#1a1a1a;border-radius:3px;">'
                    f'<span style="font-size:0.78em;color:#aaa">'
                    f'{cond["main"]} — {cond["description"].capitalize()}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #333;'
            f'font-size:0.85em">Total: <b style="color:white">{w["score"]}</b>/100 → '
            f'<b>{w["level"]}</b></div>',
            unsafe_allow_html=True,
        )

    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""<div style="font-size:0.8em">

**🌤 Weather Risk** `0–100`
Measures active national weather alerts and current conditions
(wind speed, rainfall, weather type) at your location.
- **Source:** OpenWeatherMap One Call 3.0
- **Updated:** Every 60 minutes
- **Location-specific:** Yes — changes when you move the pin</div>
""", unsafe_allow_html=True)

    score_gauge("⚔ Regional Risk", r["score"], 30, r["level"])
    st.caption(f"Trend: {r['trend']} · {r['country']}")

    _GDACS_COLORS = {"Red": "🔴", "Orange": "🟠", "Green": "🟢"}
    gdacs_alerts  = r.get("gdacs_alerts", [])
    crisis_themes = r.get("crisis_themes", [])

    if gdacs_alerts:
        for ev in gdacs_alerts[:3]:
            dot = _GDACS_COLORS.get(ev["alert_level"], "⚪")
            st.caption(f"{dot} GDACS {ev['alert_level']} · {ev['event_type']} · {ev['country']}")
    else:
        st.caption("GDACS: no active alerts in the region")

    if crisis_themes:
        st.caption("ReliefWeb themes: " + " · ".join(crisis_themes[:4]))

    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""<div style="font-size:0.8em">

**⚔ Regional Risk** `0–30`
Combines natural disaster alerts and humanitarian crisis reports
in your country and neighbouring countries.
- **Sources:** GDACS (UN/EC — earthquakes, floods, wildfires) +
  ReliefWeb (UN OCHA — conflict, displacement, humanitarian crises)
- **Updated:** Every 7 days
- **Location-specific:** Yes — neighbours change with your country</div>
""", unsafe_allow_html=True)

    score_gauge("🦠 Health Risk", h["score"], 50, h["level"])
    if h["top_threats"]:
        st.caption(f"Active: {', '.join(h['top_threats'][:2])}")
    else:
        st.caption("EU-wide signal · not location-specific")
    with st.expander("ℹ About this risk", expanded=False):
        st.markdown("""<div style="font-size:0.8em">

**🦠 Health Risk** `0–50`
Monitors active communicable disease threats across the EU/EEA,
based on the weekly ECDC bulletin (CDTR).
- **Source:** ECDC Communicable Disease Threats Report
- **Updated:** Every 7 days (published Thursdays)
- **Location-specific:** No — EU-wide signal, same for all EU countries</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# ── ROW 2: KIT STATUS + KIT EDITOR | ACTION LIST ────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### Emergency Kit Management")

# Compute alerts from current signals + kit
alerts_list = prioritize(
    risk=               _risk_result,
    inventory_report=   inv_report,
    geo_score=          regional_data["score"],
    geo_trend=          regional_data["trend"],
    geo_country=        regional_data["country"],
    health_score=       health_data["score"],
    health_level=       health_data["level"],
    top_health_threats= health_data["top_threats"],
)

col_kit, col_alerts = st.columns([0.5, 0.5])

# ── Kit status ───────────────────────────────────────────────────────────────
with col_kit:
    n         = st.session_state.household_size
    gap_score = inv_report.total_gap_score
    gaps      = inv_report.gaps
    expiring  = inv_report.expiring

    st.markdown(
        f"**📦 Kit Status** — "
        f"`{len(gaps)} gaps · score {gap_score}/100 · {n} person(s)`"
    )

    if not gaps and not expiring:
        st.success("✓ Kit complete — all items at recommended levels.")
    else:
        if gaps:
            for g in gaps:
                to_buy = g.recommended - g.current
                color  = "#e74c3c" if g.priority == "HIGH" else "#f39c12"
                pct    = g.gap_pct
                bar    = int(pct / 5)
                st.markdown(
                    f'<div style="margin:5px 0;font-size:0.85em">'
                    f'<div style="margin-bottom:2px">'
                    f'<b>{g.name}</b> '
                    f'<span style="color:{color}">need {to_buy:.1f} {g.unit} more</span>'
                    f'<span style="color:#666;font-size:0.85em"> '
                    f'(have {g.current:.1f} / need {g.recommended:.1f})</span>'
                    f'</div>'
                    f'<div style="background:#333;border-radius:3px;height:4px">'
                    f'<div style="background:{color};width:{pct}%;height:4px;border-radius:3px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        if expiring:
            st.markdown("**⏰ Expiring soon:**")
            for e in expiring:
                color = "#e74c3c" if e.urgency == "CRITICAL" else "#f39c12"
                st.markdown(
                    f'<span style="color:{color}">⏰ {e.name}: '
                    f'{e.days_remaining}d ({e.expiry_date})</span>',
                    unsafe_allow_html=True,
                )

    # ── Kit editor ────────────────────────────────────────────────────────────
    st.markdown("")
    with st.expander("✏ Edit Kit", expanded=False):
        new_size = st.number_input(
            "👥 People in household",
            min_value=1, max_value=10,
            value=st.session_state.household_size,
            key="household_size_input",
            help="Scales water, food, medication, hand sanitizer, cash.",
        )

        st.divider()

        # Initialise edits from current kit_items
        if "kit_edits" not in st.session_state:
            st.session_state.kit_edits = {
                item.name: {
                    "quantity":    item.quantity,
                    "expiry_date": item.expiry_date.isoformat() if item.expiry_date else "",
                }
                for item in st.session_state.kit_items
            }

        categories: dict = {}
        for item in st.session_state.kit_items:
            categories.setdefault(item.category, []).append(item)

        for cat, cat_items in sorted(categories.items()):
            st.markdown(f"**{cat.title()}**")
            for item in cat_items:
                name      = item.name
                scales    = name in PER_PERSON_ITEMS
                base_rec  = PER_PERSON_ITEMS.get(name, item.eu_recommended)
                scaled_rec = base_rec * new_size if scales else item.eu_recommended
                rec_label  = (
                    f"{scaled_rec:.0f} {item.unit} ({base_rec:.0f} × {new_size} people)"
                    if scales and new_size > 1
                    else f"{scaled_rec:.0f} {item.unit}"
                )

                c1, c2, c3 = st.columns([2.5, 1.2, 1.5])
                with c1:
                    st.markdown(
                        f'<div style="padding-top:8px;font-size:0.9em">'
                        f'{"👥 " if scales else ""}{name}'
                        f'<span style="color:#888;font-size:0.8em"> (rec: {rec_label})</span></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.session_state.kit_edits[name]["quantity"] = st.number_input(
                        "Qty", value=float(item.quantity),
                        min_value=0.0, step=0.5,
                        key=f"qty_{name}", label_visibility="collapsed",
                    )
                with c3:
                    st.session_state.kit_edits[name]["expiry_date"] = st.text_input(
                        "Expiry",
                        value=item.expiry_date.isoformat() if item.expiry_date else "",
                        key=f"exp_{name}", label_visibility="collapsed",
                        placeholder="YYYY-MM-DD",
                    )

        st.markdown("")
        if st.button("💾 Save All", type="primary", use_container_width=True):
            # Apply edits to kit_items in session state
            for item in st.session_state.kit_items:
                edits = st.session_state.kit_edits.get(item.name, {})
                item.quantity = edits.get("quantity", item.quantity)
                exp_str = edits.get("expiry_date", "")
                item.expiry_date = date.fromisoformat(exp_str) if exp_str else None

            st.session_state.household_size = new_size
            _db_save(st.session_state.kit_items, new_size)

            # Clear kit_edits so they re-init from updated items on next render
            del st.session_state["kit_edits"]
            st.toast(f"✓ Kit saved for {new_size} person(s)", icon="✅")
            time.sleep(0.5)
            st.rerun()

# ── Action list ──────────────────────────────────────────────────────────────
with col_alerts:
    st.markdown(f"**🚨 Action List** — `{len(alerts_list)} items`")
    if not alerts_list:
        st.success("✓ No active alerts — kit looks good for current risk levels.")
    else:
        alerts_html = "".join(
            alert_card_html({
                "urgency":        a.urgency,
                "category":       a.category,
                "priority_score": a.priority_score,
                "message":        a.message,
            })
            for a in alerts_list
        )
        st.markdown(f'<div class="alert-scroll">{alerts_html}</div>',
                    unsafe_allow_html=True)
        if len(alerts_list) > 5:
            st.caption(f"↕ Scroll to see all {len(alerts_list)} alerts")

st.divider()

# ---------------------------------------------------------------------------
# ── ROW 3: CHAT | SCENARIO SIMULATOR ────────────────────────────────────────
# ---------------------------------------------------------------------------

col_chat, col_scenario = st.columns([0.5, 0.5])

# ── Chat ─────────────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Chat")
    st.markdown("<small>Ask about your kit, current risks, or preparedness advice.</small>",
                unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")

    if question := st.chat_input("Ask a question...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever, llm = _load_agent()
                if retriever is None:
                    answer  = ("⚠ Knowledge base not available — "
                               "FAISS index not found. Please check deployment.")
                    sources = []
                    st.warning(answer)
                else:
                    from agent.agent import HavenAgent
                    agent = HavenAgent(
                        retriever=  retriever,
                        inv_report= inv_report,
                        signals=    signals,
                        llm=        llm,
                        people=     st.session_state.household_size,
                    )
                    response = agent.ask(question)
                    answer   = response.answer
                    sources  = response.sources
                    intent   = response.intent
                    fallback = response.fallback

                    st.markdown(answer)
                    badge_color = "#555" if fallback else "#2980b9"
                    st.markdown(
                        f'<span style="font-size:0.75em;color:#888">Intent: '
                        f'<span style="background:{badge_color};color:white;'
                        f'padding:1px 6px;border-radius:3px">{intent}</span></span>',
                        unsafe_allow_html=True,
                    )
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for s in sources:
                                st.caption(f"• {s}")

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })

    if st.session_state.messages:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

# ── Scenario simulator ────────────────────────────────────────────────────────
with col_scenario:
    st.markdown("### ⚙ Scenario Simulator")
    st.markdown("<small>Estimate how long your kit lasts in an emergency.</small>",
                unsafe_allow_html=True)

    event_label    = st.selectbox("Event type", list(EVENT_TYPES.keys()), key="sc_event")
    event_type     = EVENT_TYPES[event_label]
    duration_hours = st.slider("Duration (hours)", 12, 168, 72, step=12, key="sc_duration")
    people         = st.number_input("People in household", 1, 10,
                                     value=st.session_state.household_size,
                                     key="sc_people")

    if st.button("▶ Run Scenario", use_container_width=True, type="primary", key="sc_run"):
        result = run_scenario(
            inv_report=     inv_report,
            event_type=     event_type,
            duration_hours= duration_hours,
            people=         people,
        )
        pct   = result.survival_pct
        color = (LEVEL_COLORS["HIGH"] if pct < 40 else
                 LEVEL_COLORS["MEDIUM"] if pct < 75 else
                 LEVEL_COLORS["LOW"])

        st.markdown(f"""
        <div style="background:#1e1e2e;border-radius:8px;padding:14px 18px;margin:8px 0">
          <div style="font-size:0.9em;color:#aaa">
            {event_label} · {duration_hours}h · {people} person(s)
          </div>
          <div style="font-size:2.8em;font-weight:bold;color:{color}">{pct}%</div>
          <div style="color:#aaa;font-size:0.85em">scenario coverage</div>
          <div style="background:#333;border-radius:4px;height:7px;margin:8px 0">
            <div style="background:{color};width:{pct}%;height:7px;border-radius:4px"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💧 Water", f"{result.water_hours:.0f}h")
        m2.metric("🍞 Food",  f"{result.food_hours:.0f}h")
        m3.metric("📻 Comms", "✓" if result.comms_ok else "✗")
        m4.metric("💊 Meds",  "✓" if result.meds_ok  else "✗")

        if result.critical_gaps:
            st.markdown("**⚠ Critical shortfalls:**")
            for g in result.critical_gaps:
                st.error(g)
        if result.recommendations:
            st.markdown("**Recommendations:**")
            for rec in result.recommendations:
                st.info(f"• {rec}")
        st.caption(result.narrative)
