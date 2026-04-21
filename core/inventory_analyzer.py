"""
core/inventory_analyzer.py
--------------------------
Analyses the emergency kit against EU / Dutch government recommendations.

Two main outputs:
    1. Gap report  — items below recommended quantity
    2. Expiry report — items expiring within configurable thresholds

Reference: Dutch government emergency kit guidelines (denkvooruit.nl, Nov 2025)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, TYPE_CHECKING


# ---------------------------------------------------------------------------
# Expiry thresholds (days)
# ---------------------------------------------------------------------------

EXPIRY_CRITICAL_DAYS = 7
EXPIRY_WARNING_DAYS  = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KitItem:
    name:           str
    category:       str       # water / food / meds / tools / documents / comms / hygiene
    quantity:       float
    unit:           str       # liters / days / units / kg
    eu_recommended: float     # recommended quantity per person for 72h
    expiry_date:    Optional[date] = None
    notes:          str = ""


@dataclass
class GapItem:
    name:           str
    category:       str
    current:        float
    recommended:    float
    unit:           str
    gap:            float
    gap_pct:        float     # percentage missing vs recommended
    priority:       str       # HIGH / MEDIUM / LOW — set by category criticality


@dataclass
class ExpiryItem:
    name:           str
    category:       str
    expiry_date:    date
    days_remaining: int
    urgency:        str       # CRITICAL (<=7d) / WARNING (<=30d)


@dataclass
class InventoryReport:
    gaps:      List[GapItem]
    expiring:  List[ExpiryItem]
    all_items: List["KitItem"] = field(default_factory=list)

    @property
    def has_critical_gaps(self) -> bool:
        return any(g.priority == "HIGH" for g in self.gaps)

    @property
    def has_critical_expiry(self) -> bool:
        return any(e.urgency == "CRITICAL" for e in self.expiring)

    @property
    def total_gap_score(self) -> int:
        """
        Numeric gap severity (0-100) for use in alert prioritizer.
        Weighted by category criticality.
        """
        weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        raw = sum(weights.get(g.priority, 1) * g.gap_pct for g in self.gaps)
        # Normalise: assume worst case is 10 HIGH-priority items at 100% gap
        return min(int(raw / 30), 100)


# ---------------------------------------------------------------------------
# Category criticality
# ---------------------------------------------------------------------------

# Maps category → priority level for gap items
_CATEGORY_PRIORITY = {
    "water":     "HIGH",
    "meds":      "HIGH",
    "food":      "HIGH",
    "comms":     "MEDIUM",
    "tools":     "MEDIUM",
    "documents": "MEDIUM",
    "hygiene":   "LOW",
}


def _category_priority(category: str) -> str:
    return _CATEGORY_PRIORITY.get(category.lower(), "LOW")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def analyze_gaps(items: List[KitItem]) -> List[GapItem]:
    """
    Compare current kit quantities against EU recommendations.

    Args:
        items: List of KitItem objects representing the current kit.

    Returns:
        Sorted list of GapItem — items below recommendation, worst first.
        Items at or above recommendation are excluded.
    """
    gaps = []
    for item in items:
        if item.eu_recommended <= 0:
            continue
        gap = item.eu_recommended - item.quantity
        if gap <= 0:
            continue
        gap_pct = (gap / item.eu_recommended) * 100
        gaps.append(GapItem(
            name=item.name,
            category=item.category,
            current=item.quantity,
            recommended=item.eu_recommended,
            unit=item.unit,
            gap=round(gap, 2),
            gap_pct=round(gap_pct, 1),
            priority=_category_priority(item.category),
        ))

    # Sort: HIGH priority first, then by gap_pct descending
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    gaps.sort(key=lambda g: (priority_order.get(g.priority, 3), -g.gap_pct))
    return gaps


def analyze_expiry(
    items: List[KitItem],
    reference_date: Optional[date] = None,
) -> List[ExpiryItem]:
    """
    Identify items expiring within the warning thresholds.

    Args:
        items:          List of KitItem objects.
        reference_date: Date to compare against (defaults to today).

    Returns:
        Sorted list of ExpiryItem — most urgent first.
        Items without expiry date or expiring after warning threshold excluded.
    """
    ref = reference_date or date.today()
    expiring = []

    for item in items:
        if item.expiry_date is None:
            continue
        days = (item.expiry_date - ref).days
        if days > EXPIRY_WARNING_DAYS:
            continue
        urgency = "CRITICAL" if days <= EXPIRY_CRITICAL_DAYS else "WARNING"
        expiring.append(ExpiryItem(
            name=item.name,
            category=item.category,
            expiry_date=item.expiry_date,
            days_remaining=days,
            urgency=urgency,
        ))

    expiring.sort(key=lambda e: e.days_remaining)
    return expiring


def analyze_inventory(
    items: List[KitItem],
    reference_date: Optional[date] = None,
) -> InventoryReport:
    """
    Run full inventory analysis — gaps and expiry combined.

    Args:
        items:          List of KitItem objects.
        reference_date: Date for expiry comparison (defaults to today).

    Returns:
        InventoryReport with gaps and expiring lists.
    """
    return InventoryReport(
        gaps=      analyze_gaps(items),
        expiring=  analyze_expiry(items, reference_date),
        all_items= list(items),
    )
