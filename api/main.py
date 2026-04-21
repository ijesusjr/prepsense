"""
api/main.py
------------
HAVEN FastAPI backend.

Endpoints:
    GET  /                  health check
    GET  /risk              current three-signal risk snapshot
    GET  /kit               current kit inventory + gaps + expiry
    PUT  /kit/{item_name}   update a kit item quantity / expiry
    GET  /alerts            prioritised alert list (all signals × kit)
    POST /chat              agent query → cited answer
    POST /scenario          scenario simulation → survival estimate
    GET  /refresh           manually trigger signal refresh

Scheduler (APScheduler):
    Every 60 min  → fetch weather (OWM One Call)
    Every 7 days  → fetch regional risk (GDACS + ReliefWeb)
    Every 7 days  → fetch health risk (ECDC CDTR)
    On startup    → run all three immediately
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("haven")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: wire agent, run initial signal fetch, start scheduler."""
    log.info("HAVEN API starting up...")
    _wire_agent()
    _run_all_fetchers()
    _start_scheduler()
    log.info("HAVEN API ready.")
    yield
    log.info("HAVEN API shutting down.")
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


app = FastAPI(
    title="HAVEN API",
    description="AI-powered emergency preparedness copilot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# App state (imported singleton)
# ---------------------------------------------------------------------------

from api.state import app_state
from core.regions import get_region, is_supported


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

def _wire_agent():
    """Load RAG retriever, LLM, and HavenAgent on startup."""
    from pathlib import Path
    faiss_dir = Path("data/faiss")

    try:
        from rag.retriever import HavenRetriever
        app_state.retriever = HavenRetriever.from_disk(
            index_path=str(faiss_dir / "index.bin"),
            meta_path= str(faiss_dir / "chunks.json"),
        )
        log.info(f"Retriever loaded: {app_state.retriever.index.ntotal} chunks")
    except Exception as e:
        log.warning(f"Retriever not available: {e}")
        app_state.retriever = None

    try:
        from rag.llm import HavenLLM
        backend = os.getenv("LLM_BACKEND", "groq")
        app_state.llm = HavenLLM(backend=backend)
        log.info(f"LLM backend: {backend} (available: {app_state.llm.is_available()})")
    except Exception as e:
        log.warning(f"LLM not available: {e}")
        app_state.llm = None

    if app_state.retriever is not None:
        from agent.agent import HavenAgent
        app_state.agent = HavenAgent(
            retriever=  app_state.retriever,
            inv_report= app_state.inv_report,
            signals=    app_state.signals,
            llm=        app_state.llm,
            people=     int(os.getenv("HOUSEHOLD_SIZE", "1")),
        )
        log.info("HavenAgent wired.")


# ---------------------------------------------------------------------------
# Signal fetchers
# ---------------------------------------------------------------------------

def _fetch_weather():
    """Fetch live weather + alerts from OWM One Call 3.0."""
    try:
        import requests as req
        api_key = os.getenv("OWM_API_KEY", "")
        lat     = float(os.getenv("LAT", "41.3851"))
        lon     = float(os.getenv("LON", "2.1734"))
        if not api_key:
            log.warning("OWM_API_KEY not set — skipping weather fetch")
            return

        resp = req.get(
            "https://api.openweathermap.org/data/3.0/onecall",
            params={"lat": lat, "lon": lon, "appid": api_key,
                    "exclude": "minutely,hourly,daily", "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data.get("current", {})
        alerts  = data.get("alerts", [])

        from core.risk_engine import (
            WeatherSnapshot, Alert, compute_risk_score,
            HavenSignals, weather_id_to_severity,
        )

        weather_id = current.get("weather", [{}])[0].get("id", 800)
        snap = WeatherSnapshot(
            weather_id=    weather_id,
            wind_speed_ms= current.get("wind_speed", 0.0),
            rain_1h_mm=    current.get("rain", {}).get("1h", 0.0),
            wind_gust_ms=  current.get("wind_gust", 0.0),
        )
        alert_objs = [
            Alert(event=a.get("event",""), severity=_infer_severity(a),
                  tags=a.get("tags",[]))
            for a in alerts
        ]
        risk = compute_risk_score(snap, alert_objs)

        # Keep geo + health from existing signals
        existing = app_state.signals
        new_signals = HavenSignals(
            weather=            risk,
            geo_score=          existing.geo_score   if existing else 0,
            geo_trend=          existing.geo_trend   if existing else "STABLE",
            geo_country=        existing.geo_country if existing else "Spain",
            health_score=       existing.health_score   if existing else 0,
            health_level=       existing.health_level   if existing else "ROUTINE",
            top_health_threats= existing.top_health_threats if existing else [],
        )
        app_state.update_signals(new_signals)
        _rewire_agent()
        log.info(f"Weather updated: score={risk.risk_score} level={risk.risk_level}")
    except Exception as e:
        log.error(f"Weather fetch failed: {e}")


def _infer_severity(alert: dict) -> str:
    tags = [t.lower() for t in alert.get("tags", [])]
    if "extreme" in tags:  return "Extreme"
    if "severe"  in tags:  return "Severe"
    if "moderate" in tags: return "Moderate"
    return "Minor"


def _fetch_regional():
    """Fetch GDACS + ReliefWeb regional risk using current country."""
    try:
        from core.regional_risk_fetcher import get_regional_snapshot
        country = os.getenv("CITY_COUNTRY", "Spain")
        region  = get_region(country)
        geo = get_regional_snapshot(country=country, region_countries=region)
        existing = app_state.signals
        from core.risk_engine import HavenSignals
        new_signals = HavenSignals(
            weather=            existing.weather        if existing else _stub_risk(),
            geo_score=          geo.regional_score,
            geo_trend=          "STABLE",
            geo_country=        geo.country,
            health_score=       existing.health_score       if existing else 0,
            health_level=       existing.health_level       if existing else "ROUTINE",
            top_health_threats= existing.top_health_threats if existing else [],
        )
        app_state.update_signals(new_signals)
        _rewire_agent()
        log.info(f"Regional updated: score={geo.regional_score} level={geo.level} country={country}")
    except Exception as e:
        log.error(f"Regional fetch failed: {e}")


def _fetch_health():
    """Fetch ECDC CDTR health risk."""
    try:
        from core.health_fetcher import get_health_snapshot
        hs = get_health_snapshot()
        existing = app_state.signals
        from core.risk_engine import HavenSignals
        new_signals = HavenSignals(
            weather=            existing.weather    if existing else _stub_risk(),
            geo_score=          existing.geo_score  if existing else 0,
            geo_trend=          existing.geo_trend  if existing else "STABLE",
            geo_country=        existing.geo_country if existing else "Spain",
            health_score=       hs.health_score,
            health_level=       hs.level,
            top_health_threats= hs.top_threats,
        )
        app_state.update_signals(new_signals)
        _rewire_agent()
        log.info(f"Health updated: score={hs.health_score} level={hs.level}")
    except Exception as e:
        log.error(f"Health fetch failed: {e}")


def _stub_risk():
    from core.risk_engine import RiskResult
    return RiskResult(risk_score=0, risk_level="LOW",
                      weather_severity=0, alert_severity=0,
                      wind_bonus=0, rain_bonus=0)


def _run_all_fetchers():
    _fetch_weather()
    _fetch_regional()
    _fetch_health()


def _rewire_agent():
    """Update agent's signals and inv_report references after a refresh."""
    if app_state.agent is not None:
        app_state.agent.signals    = app_state.signals
        app_state.agent.inv_report = app_state.inv_report


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler = BackgroundScheduler()


def _start_scheduler():
    _scheduler.add_job(_fetch_weather,  "interval", minutes=60,  id="weather")
    _scheduler.add_job(_fetch_regional, "interval", hours=168,   id="regional")  # 7 days
    _scheduler.add_job(_fetch_health,   "interval", hours=168,   id="health")
    _scheduler.start()
    log.info("Scheduler started: weather=60min, regional=7d, health=7d")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KitUpdateRequest(BaseModel):
    quantity:    float
    expiry_date: Optional[str] = None   # ISO format YYYY-MM-DD


class ChatRequest(BaseModel):
    question:   str
    event_type: Optional[str] = "general"


class ScenarioRequest(BaseModel):
    event_type:     str = "power_outage"
    duration_hours: int = 72
    people:         int = 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status":      "ok",
        "service":     "HAVEN API",
        "version":     "1.0.0",
        "signals_age": f"{app_state.signals_age_minutes:.1f} min",
        "agent_ready": app_state.agent is not None,
        "llm_backend": os.getenv("LLM_BACKEND", "not set"),
    }


@app.get("/risk")
def get_risk():
    """Current three-signal risk snapshot."""
    if app_state.signals is None:
        raise HTTPException(status_code=503, detail="Signals not yet loaded")

    s = app_state.signals
    return {
        "weather": {
            "score": s.weather.risk_score,
            "level": s.weather.risk_level,
            "breakdown": {
                "weather_severity": s.weather.weather_severity,
                "alert_severity":   s.weather.alert_severity,
                "wind_bonus":       s.weather.wind_bonus,
                "rain_bonus":       s.weather.rain_bonus,
            },
        },
        "regional": {
            "score":   s.geo_score,
            "level":   _geo_level(s.geo_score),
            "trend":   s.geo_trend,
            "country": s.geo_country,
        },
        "health": {
            "score":       s.health_score,
            "level":       s.health_level,
            "top_threats": s.top_health_threats,
        },
        "meta": {
            "fetched_at":    app_state.signals_ts.isoformat() if app_state.signals_ts else None,
            "age_minutes":   round(app_state.signals_age_minutes, 1),
            "stale":         app_state.signals_stale,
        },
    }


@app.get("/kit")
def get_kit():
    """Current kit inventory — items, gaps, expiry, and household size."""
    r = app_state.inv_report
    return {
        "household_size": app_state.household_size,
        "items": [
            {
                "name":           item.name,
                "category":       item.category,
                "quantity":       item.quantity,
                "eu_recommended": item.eu_recommended,
                "unit":           item.unit,
                "expiry_date":    item.expiry_date.isoformat() if item.expiry_date else None,
            }
            for item in app_state.kit_items
        ],
        "gaps": [
            {
                "name":        g.name,
                "category":    g.category,
                "current":     g.current,
                "recommended": g.recommended,
                "unit":        g.unit,
                "gap_pct":     round(g.gap_pct, 1),
                "priority":    g.priority,
            }
            for g in r.gaps
        ],
        "expiring": [
            {
                "name":           e.name,
                "expiry_date":    e.expiry_date.isoformat(),
                "days_remaining": e.days_remaining,
                "urgency":        e.urgency,
            }
            for e in r.expiring
        ],
        "summary": {
            "total_items":   len(app_state.kit_items),
            "total_gaps":    len(r.gaps),
            "gap_score":     r.total_gap_score,
            "critical_gaps": sum(1 for g in r.gaps if g.priority == "HIGH"),
        },
    }


@app.put("/kit/{item_name}")
def update_kit_item(item_name: str, body: KitUpdateRequest):
    """Update a kit item quantity and optional expiry date — persisted to DB."""
    names = [i.name for i in app_state.kit_items]
    if item_name not in names:
        raise HTTPException(status_code=404,
                            detail=f"Item '{item_name}' not found. Available: {names}")
    app_state.update_kit_item(item_name, body.quantity, body.expiry_date)
    _rewire_agent()
    return {"status": "updated", "item": item_name,
            "quantity": body.quantity, "expiry_date": body.expiry_date}


class HouseholdRequest(BaseModel):
    size: int


@app.put("/household")
def update_household(body: HouseholdRequest):
    """Update household size — rescales per-person recommendations and persists to DB."""
    if not 1 <= body.size <= 10:
        raise HTTPException(status_code=422,
                            detail="Household size must be between 1 and 10")
    app_state.set_household_size(body.size)
    _rewire_agent()
    return {
        "status":         "updated",
        "household_size": app_state.household_size,
        "gaps":           len(app_state.inv_report.gaps),
        "gap_score":      app_state.inv_report.total_gap_score,
    }


@app.get("/alerts")
def get_alerts():
    """Prioritised alert list — all signals × kit gaps."""
    from core.alert_prioritizer import prioritize

    s = app_state.signals
    r = app_state.inv_report

    alerts = prioritize(
        risk=               s.weather,
        inventory_report=   r,
        geo_score=          s.geo_score,
        geo_trend=          s.geo_trend,
        geo_country=        s.geo_country,
        health_score=       s.health_score,
        health_level=       s.health_level,
        top_health_threats= s.top_health_threats,
    )

    return {
        "count":  len(alerts),
        "alerts": [
            {
                "priority_score": a.priority_score,
                "urgency":        a.urgency,
                "category":       a.category,
                "message":        a.message,
                "detail":         a.detail,
            }
            for a in alerts
        ],
    }


@app.post("/chat")
def chat(body: ChatRequest):
    """Agent query — returns cited answer from guidelines + risk + kit context."""
    if app_state.agent is None:
        raise HTTPException(status_code=503,
                            detail="Agent not ready — FAISS index may not be built yet")

    # Keep agent state in sync
    app_state.agent.inv_report = app_state.inv_report
    app_state.agent.signals    = app_state.signals

    try:
        response = app_state.agent.ask(
            query=      body.question,
            event_type= body.event_type or "general",
        )
        return {
            "question": response.query,
            "answer":   response.answer,
            "intent":   response.intent,
            "sources":  response.sources,
            "fallback": response.fallback,
            "routing":  response.routing,
        }
    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario")
def run_scenario_endpoint(body: ScenarioRequest):
    """Scenario simulation — survival estimate for given event × duration × people."""
    from agent.tools import run_scenario

    result = run_scenario(
        inv_report=     app_state.inv_report,
        event_type=     body.event_type,
        duration_hours= body.duration_hours,
        people=         body.people,
    )

    return {
        "event_type":      result.event_type,
        "duration_hours":  result.duration_hours,
        "people":          result.people,
        "survival_pct":    result.survival_pct,
        "water_hours":     result.water_hours,
        "food_hours":      result.food_hours,
        "comms_ok":        result.comms_ok,
        "meds_ok":         result.meds_ok,
        "critical_gaps":   result.critical_gaps,
        "narrative":       result.narrative,
        "recommendations": result.recommendations,
    }


@app.get("/refresh")
def manual_refresh(background_tasks: BackgroundTasks):
    """Manually trigger a full signal refresh (runs in background)."""
    background_tasks.add_task(_run_all_fetchers)
    return {"status": "refresh triggered", "note": "signals will update in ~10s"}


class LocationRequest(BaseModel):
    lat:     float
    lon:     float
    city:    Optional[str] = None
    country: Optional[str] = None


@app.put("/location")
def update_location(body: LocationRequest, background_tasks: BackgroundTasks):
    """Update user location and trigger weather + regional refresh."""
    os.environ["LAT"] = str(body.lat)
    os.environ["LON"] = str(body.lon)
    if body.city:
        os.environ["CITY"] = body.city
    if body.country:
        os.environ["CITY_COUNTRY"] = body.country

    # Weather always refreshes on location change
    # Regional refreshes too — neighbours differ by country
    # Health does NOT — it's EU-wide, not location-specific
    background_tasks.add_task(_fetch_weather)
    background_tasks.add_task(_fetch_regional)

    supported = is_supported(body.country or "")
    return {
        "status":    "updated",
        "lat":       body.lat,
        "lon":       body.lon,
        "city":      body.city,
        "country":   body.country,
        "supported": supported,
        "triggers":  ["weather", "regional"],
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geo_level(score: int) -> str:
    if score >= 22: return "HIGH"
    if score >= 12: return "MEDIUM"
    if score >= 4:  return "LOW"
    return "MINIMAL"
