"""
agent/router.py
----------------
Intent classifier for HAVEN agent queries.

Design: structured routing over ReAct loop.

Instead of letting the LLM freely decide which tools to call (ReAct pattern),
we classify the user query into one of 5 intents and deterministically dispatch
to the appropriate tools. The LLM is used for:
    1. Intent classification (one structured JSON call)
    2. Final answer composition (one generation call)

This is more reliable than ReAct for a small, well-defined tool set:
    - No risk of the agent looping or calling wrong tools
    - Easier to debug — routing decisions are explicit and logged
    - Still demonstrates the agent pattern clearly in the portfolio

Intents:
    KIT_QUESTION   → "why do I need X?" / "what should I have?"
    RISK_QUESTION  → "how dangerous is my situation?"
    SCENARIO       → "how long will my kit last if...?" / "what if...?"
    GENERAL_PREP   → "how do I prepare for...?"
    UNKNOWN        → fallback
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------

INTENTS = {
    "KIT_QUESTION": {
        "description": "Questions about specific kit items — why they are needed, "
                       "what to buy, how much to store.",
        "keywords":    ["why do i need", "what should i have", "how much",
                        "what is in", "explain", "recommend", "suggest", "item",
                        "supply", "stock", "buy", "get", "need", "medication",
                        "water", "food", "radio", "flashlight", "first aid",
                        "kit", "emergency kit", "what do i need", "store"],
        "tools":       ["retrieve_guidelines", "get_kit_gaps"],
    },
    "RISK_QUESTION": {
        "description": "Questions about the current risk level, active threats, "
                       "or whether the user should be concerned.",
        "keywords":    ["risk", "danger", "threat", "alert", "warning",
                        "safe", "situation", "current", "now", "today",
                        "weather", "health", "conflict", "concern", "level"],
        "tools":       ["get_risk_score"],
    },
    "SCENARIO": {
        "description": "Questions about hypothetical emergency scenarios — "
                       "survival estimates, duration, what happens if...",
        "keywords":    ["how long", "last", "scenario", "power outage",
                        "flood", "earthquake", "72 hours", "survive",
                        "what if", "if power", "if water", "estimate",
                        "days", "hours", "outage", "blackout", "emergency",
                        "heat wave", "blackout", "hurricane", "crisis", "kit"],
        "tools":       ["get_kit_gaps", "run_scenario", "retrieve_guidelines"],
    },
    "GENERAL_PREP": {
        "description": "General emergency preparedness questions — how to prepare, "
                       "best practices, evacuation, communication plans.",
        "keywords":    ["prepare", "preparation", "plan", "evacuate",
                        "communication", "family", "neighbours", "practice",
                        "ready", "guide", "tips", "advice", "how to",
                        "should i", "best way", "evacuation", "emergency plan"],
        "tools":       ["retrieve_guidelines"],
    },
    "UNKNOWN": {
        "description": "Query does not match any known intent.",
        "keywords":    [],
        "tools":       [],
    },
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    intent:     str
    confidence: str          # HIGH / MEDIUM / LOW
    tools:      list
    reasoning:  str


# ---------------------------------------------------------------------------
# Keyword-based fallback router (no LLM needed)
# ---------------------------------------------------------------------------

def _keyword_route(query: str) -> RoutingDecision:
    """
    Simple keyword-based intent classification.
    Used as fallback when LLM routing is unavailable.
    Fast, deterministic, zero cost.
    """
    query_lower = query.lower()
    scores = {}

    for intent, config in INTENTS.items():
        if intent == "UNKNOWN":
            continue
        score = sum(1 for kw in config["keywords"] if kw in query_lower)
        scores[intent] = score

    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]

    if best_score < 2:
        return RoutingDecision(
            intent=     "UNKNOWN",
            confidence= "LOW",
            tools=      [],
            reasoning=  f"Insufficient keyword matches (score={best_score}, need ≥2).",
        )

    confidence = "HIGH" if best_score >= 3 else "MEDIUM"

    return RoutingDecision(
        intent=     best_intent,
        confidence= confidence,
        tools=      INTENTS[best_intent]["tools"],
        reasoning=  f"Keyword match: {best_score} hits for {best_intent}.",
    )


# ---------------------------------------------------------------------------
# LLM-based router (more accurate, used when backend available)
# ---------------------------------------------------------------------------

ROUTING_PROMPT = """You are a query router for HAVEN, an emergency preparedness assistant.

Classify the following user query into exactly one intent:

KIT_QUESTION   — about specific kit items (why needed, how much, what to buy)
RISK_QUESTION  — about current risk level, active threats, or safety
SCENARIO       — hypothetical scenarios ("how long will X last", "what if power is out")
GENERAL_PREP   — general preparedness advice, plans, evacuation

Respond with ONLY a JSON object, no other text:
{{
  "intent": "KIT_QUESTION",
  "confidence": "HIGH",
  "reasoning": "one sentence"
}}

User query: {query}"""


def _llm_route(query: str, llm_func) -> Optional[RoutingDecision]:
    """
    Use the LLM to classify intent. Returns None if parsing fails.

    Args:
        query:    User query string.
        llm_func: Callable that takes a prompt string and returns a string response.
    """
    prompt = ROUTING_PROMPT.format(query=query)
    try:
        raw = llm_func(prompt)
        # Extract JSON from response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        intent = data.get("intent", "UNKNOWN").upper()
        if intent not in INTENTS:
            intent = "UNKNOWN"

        return RoutingDecision(
            intent=     intent,
            confidence= data.get("confidence", "MEDIUM"),
            tools=      INTENTS[intent]["tools"],
            reasoning=  data.get("reasoning", "LLM classification."),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

def route(query: str, llm_func=None) -> RoutingDecision:
    """
    Classify a user query into an intent and return the tools to call.

    Tries LLM routing first (if available), falls back to keyword matching.

    Args:
        query:    User query string.
        llm_func: Optional callable(prompt) → str for LLM-based routing.

    Returns:
        RoutingDecision with intent, confidence, and list of tool names.
    """
    if llm_func is not None:
        decision = _llm_route(query, llm_func)
        if decision is not None:
            decision.reasoning = f"[LLM] {decision.reasoning}"
            return decision

    # Fallback to keyword routing
    decision = _keyword_route(query)
    decision.reasoning = f"[keyword] {decision.reasoning}"
    return decision
