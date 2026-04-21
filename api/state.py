"""
api/state.py
-------------
Singleton application state shared across FastAPI endpoints and the scheduler.

Kit persistence: SQLite (haven.db).
    - On startup: load kit + household_size from DB, fall back to defaults
    - On every kit/household update: write to DB immediately
    - On Render free tier: DB survives restarts but NOT redeploys (ephemeral FS)
    - Upgrade path: swap DB_PATH for DATABASE_URL + psycopg2 (Week 6 Postgres)
"""

import os
import copy
import sqlite3
import logging
from datetime import datetime, timezone, date
from typing import Optional

from core.risk_engine import RiskResult, HavenSignals
from core.inventory_analyzer import KitItem, analyze_inventory, InventoryReport

log = logging.getLogger("haven.state")

DB_PATH = os.getenv("DB_PATH", "haven.db")


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

def _db_init():
    """Create tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kit_items (
            name            TEXT PRIMARY KEY,
            category        TEXT NOT NULL,
            quantity        REAL NOT NULL,
            eu_recommended  REAL NOT NULL,
            unit            TEXT NOT NULL,
            expiry_date     TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
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


def _db_save_kit(kit_items: list, household_size: int):
    """Persist the full kit and household size to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    for item in kit_items:
        conn.execute("""
            INSERT INTO kit_items (name, category, quantity, eu_recommended, unit, expiry_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                quantity      = excluded.quantity,
                expiry_date   = excluded.expiry_date,
                updated_at    = datetime('now')
        """, (
            item.name,
            item.category,
            item.quantity,
            item.eu_recommended,
            item.unit,
            item.expiry_date.isoformat() if item.expiry_date else None,
        ))
    conn.execute("""
        INSERT INTO settings (key, value) VALUES ('household_size', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (str(household_size),))
    conn.commit()
    conn.close()


def _db_load_kit() -> tuple:
    """
    Load kit items and household size from SQLite.

    Returns:
        (list[KitItem], household_size: int)
        Returns ([], 1) if DB is empty or does not exist.
    """
    try:
        _db_init()
        conn = sqlite3.connect(DB_PATH)

        rows = conn.execute(
            "SELECT name, category, quantity, eu_recommended, unit, expiry_date "
            "FROM kit_items"
        ).fetchall()

        setting = conn.execute(
            "SELECT value FROM settings WHERE key = 'household_size'"
        ).fetchone()
        conn.close()

        if not rows:
            return [], 1

        items = []
        for name, cat, qty, rec, unit, exp in rows:
            items.append(KitItem(
                name=           name,
                category=       cat,
                quantity=       qty,
                eu_recommended= rec,
                unit=           unit,
                expiry_date=    date.fromisoformat(exp) if exp else None,
            ))

        household_size = int(setting[0]) if setting else 1
        log.info(f"DB loaded: {len(items)} kit items, household={household_size}")
        return items, household_size

    except Exception as e:
        log.warning(f"DB load failed ({e}) — using defaults")
        return [], 1


# ---------------------------------------------------------------------------
# Per-person scaling
# ---------------------------------------------------------------------------

PER_PERSON_ITEMS = {
    "Drinking water":      9.0,   # 3L/person/day × 3 days
    "Non-perishable food": 3.0,   # 3 days/person
    "Regular medication":  7.0,   # 7 days/person
    "Hand sanitizer":      1.0,   # 1 unit/person
    "Cash":                70.0,  # €70/adult
}


def scale_recommendation(name: str, base: float, people: int) -> float:
    """Return the scaled EU recommendation for a given household size."""
    if name in PER_PERSON_ITEMS:
        return base * people
    return base


# ---------------------------------------------------------------------------
# Default kit (base values — 1 person)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AppState singleton
# ---------------------------------------------------------------------------

class AppState:
    """Singleton holding all shared HAVEN runtime state."""

    def __init__(self):
        self.signals:        Optional[HavenSignals] = None
        self.signals_ts:     Optional[datetime]         = None
        self.agent          = None
        self.retriever      = None
        self.llm            = None

        # Initialise DB schema
        _db_init()

        # Load kit from DB — fall back to DEFAULT_KIT if empty
        saved_items, saved_size = _db_load_kit()
        if saved_items:
            self.kit_items      = saved_items
            self.household_size = saved_size
        else:
            self.kit_items      = list(DEFAULT_KIT)
            self.household_size = int(os.getenv("HOUSEHOLD_SIZE", "1"))
            # Seed the DB with defaults on first run
            _db_save_kit(self.kit_items, self.household_size)

        self.inv_report: Optional[InventoryReport] = None
        self._init_signals()
        self._refresh_inventory()

    def _init_signals(self):
        """Bootstrap with simulated signals — overwritten by scheduler on first run."""
        risk = RiskResult(
            risk_score=25, risk_level="MEDIUM",
            weather_severity=0, alert_severity=25,
            wind_bonus=0, rain_bonus=0,
        )
        self.signals = HavenSignals(
            weather=            risk,
            geo_score=          4,
            geo_trend=          "STABLE",
            geo_country=        "Spain",
            health_score=       19,
            health_level=       "MEDIUM",
            top_health_threats= ["Avian Influenza", "Dengue"],
        )
        self.signals_ts = datetime.now(timezone.utc)

    def _refresh_inventory(self):
        """Rebuild InventoryReport applying per-person scaling."""
        scaled_items = []
        for item in self.kit_items:
            scaled = copy.copy(item)
            scaled.eu_recommended = scale_recommendation(
                item.name, item.eu_recommended, self.household_size
            )
            scaled_items.append(scaled)
        self.inv_report = analyze_inventory(scaled_items)

    def set_household_size(self, n: int):
        self.household_size = max(1, min(n, 10))
        os.environ["HOUSEHOLD_SIZE"] = str(self.household_size)
        _db_save_kit(self.kit_items, self.household_size)
        self._refresh_inventory()
        log.info(f"Household size set to {self.household_size}")

    def update_kit_item(self, name: str, quantity: float, expiry_date=None):
        """Update a single kit item by name and persist to DB."""
        for item in self.kit_items:
            if item.name == name:
                item.quantity = quantity
                if expiry_date is not None:
                    item.expiry_date = (
                        date.fromisoformat(expiry_date)
                        if isinstance(expiry_date, str) else expiry_date
                    )
                break
        _db_save_kit(self.kit_items, self.household_size)
        self._refresh_inventory()
        log.info(f"Kit updated: {name} = {quantity}")

    def update_signals(self, signals: HavenSignals):
        self.signals    = signals
        self.signals_ts = datetime.now(timezone.utc)

    @property
    def signals_age_minutes(self) -> float:
        if self.signals_ts is None:
            return 9999
        delta = datetime.now(timezone.utc) - self.signals_ts
        return delta.total_seconds() / 60

    @property
    def signals_stale(self) -> bool:
        return self.signals_age_minutes > 60


# Module-level singleton
app_state = AppState()
