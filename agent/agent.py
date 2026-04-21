"""
agent/agent.py
---------------
HAVEN agent — orchestrates tools, routing, and LLM answer composition.

Architecture: structured router with closure-based node injection.

Dependencies (retriever, inv_report, signals, llm) are stored on the
HavenAgent instance and injected into LangGraph nodes via closures.
This avoids the TypedDict key-dropping issue that occurs when passing
objects through LangGraph state.

Graph structure:
    route → risk → gaps → scenario → guidelines → compose → END
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agent.tools import (
    get_risk_score, get_kit_gaps, retrieve_guidelines, run_scenario,
)
from agent.router import route


# ---------------------------------------------------------------------------
# Agent state — only serialisable values, no object references
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:          str
    people:         int
    event_type:     str
    duration_hours: int
    routing:        Optional[Dict]
    risk:           Optional[Dict]
    gaps:           Optional[Dict]
    scenario:       Optional[Dict]
    guidelines:     Optional[Dict]
    answer:         str
    sources:        List[str]
    fallback:       bool
    intent:         str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COMPOSE_SYSTEM = """You are HAVEN, an AI emergency preparedness advisor.
Answer based ONLY on the provided tool results and guidelines.
Always cite sources as [Source: document name, Page X].
Be practical and specific. If addressing kit gaps, mention the user's actual numbers.
If uncertain or information is missing, say so and recommend checking official guidelines."""

COMPOSE_TEMPLATE = """{risk_section}

{gaps_section}

{scenario_section}

{guidelines_section}

USER QUESTION: {question}

Provide a practical, cited answer. If the user has critical kit gaps relevant to
this question, address them specifically with numbers from their inventory."""

FALLBACK_RESPONSE = """I wasn't able to find specific guidance for this question in the
official EU emergency preparedness documents.

For authoritative information, please consult:
• Netherlands: english.denkvooruit.nl
• Belgium: centredecrise.be
• Sweden: krisinformation.se
• Czech Republic: 72hodin.cz
• EU Civil Protection: ec.europa.eu/echo
{partial_info}"""


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    query:        str
    intent:       str
    answer:       str
    sources:      List[str]
    fallback:     bool
    routing:      Dict
    tool_results: Dict


# ---------------------------------------------------------------------------
# HAVEN Agent
# ---------------------------------------------------------------------------

class HavenAgent:
    """
    HAVEN conversational agent.

    Uses closure-based node injection so object references (retriever,
    inv_report, signals, llm) never pass through LangGraph state —
    they live on self and are captured by each node method.

    Usage:
        agent = HavenAgent(retriever, inv_report, signals, llm)
        response = agent.ask("How long will my kit last in a power outage?")
        agent.print_response(response)
    """

    def __init__(self, retriever, inv_report, signals=None, llm=None, people: int = 1):
        self.retriever  = retriever
        self.inv_report = inv_report
        self.signals    = signals
        self.llm        = llm
        self.people     = people
        self._graph     = self._build_graph()

    # ── Nodes (methods → closures over self) ─────────────────────────────────

    def _node_route(self, state: AgentState) -> AgentState:
        decision = route(state["query"])
        state["routing"] = {
            "intent":     decision.intent,
            "confidence": decision.confidence,
            "tools":      decision.tools,
            "reasoning":  decision.reasoning,
        }
        state["intent"] = decision.intent
        return state

    def _node_risk(self, state: AgentState) -> AgentState:
        if "get_risk_score" not in state["routing"]["tools"]:
            return state
        if self.signals is None:
            return state
        rs = get_risk_score(self.signals)
        state["risk"] = {
            "weather_score": rs.weather_score, "weather_level": rs.weather_level,
            "geo_score":     rs.geo_score,     "geo_level":     rs.geo_level,
            "geo_trend":     rs.geo_trend,     "health_score":  rs.health_score,
            "health_level":  rs.health_level,  "top_threats":   rs.top_threats,
            "overall":       rs.overall_concern, "narrative":   rs.narrative,
            "prompt_str":    rs.to_prompt_str(),
        }
        return state

    def _node_gaps(self, state: AgentState) -> AgentState:
        if "get_kit_gaps" not in state["routing"]["tools"]:
            return state
        if self.inv_report is None:
            return state
        gs = get_kit_gaps(self.inv_report)
        state["gaps"] = {
            "total_gaps":    gs.total_gaps,   "critical_gaps": gs.critical_gaps,
            "gap_score":     gs.gap_score,    "gaps":          gs.gaps,
            "has_gaps":      gs.has_gaps,     "prompt_str":    gs.to_prompt_str(),
        }
        return state

    def _node_scenario(self, state: AgentState) -> AgentState:
        if "run_scenario" not in state["routing"]["tools"]:
            return state
        if self.inv_report is None:
            return state
        r = run_scenario(
            inv_report=     self.inv_report,
            event_type=     state.get("event_type", "general"),
            duration_hours= state.get("duration_hours", 72),
            people=         state.get("people", 1),
        )
        state["scenario"] = {
            "event_type":      r.event_type,      "duration_hours": r.duration_hours,
            "people":          r.people,           "water_hours":    r.water_hours,
            "food_hours":      r.food_hours,       "comms_ok":       r.comms_ok,
            "meds_ok":         r.meds_ok,          "critical_gaps":  r.critical_gaps,
            "survival_pct":    r.survival_pct,     "narrative":      r.narrative,
            "recommendations": r.recommendations,  "prompt_str":     r.to_prompt_str(),
        }
        return state

    def _node_guidelines(self, state: AgentState) -> AgentState:
        if "retrieve_guidelines" not in state["routing"]["tools"]:
            return state
        if self.retriever is None:
            return state
        gl = retrieve_guidelines(self.retriever, state["query"], k=4)
        sources = list(dict.fromkeys(f"{c.source}, p.{c.page}" for c in gl.chunks))
        state["guidelines"] = {
            "query":     gl.query,     "n_chunks": len(gl.chunks),
            "context":   gl.context,   "sources":  sources,
            "prompt_str": gl.to_prompt_str(),
        }
        return state

    def _node_compose(self, state: AgentState) -> AgentState:
        intent     = state.get("intent", "UNKNOWN")
        has_gl     = bool(state.get("guidelines") and state["guidelines"]["n_chunks"] > 0)
        has_risk   = bool(state.get("risk"))
        has_sc     = bool(state.get("scenario"))
        has_gaps   = bool(state.get("gaps"))

        sources = list(state["guidelines"]["sources"]) if has_gl else []

        # Fallback if nothing useful retrieved
        if intent == "UNKNOWN" or (not has_gl and not has_risk and not has_sc):
            partial = ""
            if has_gaps and state["gaps"]["has_gaps"]:
                partial = (f"\n\nYour kit analysis shows {state['gaps']['total_gaps']} "
                           f"gap(s), score {state['gaps']['gap_score']}/100.")
            state["answer"]   = FALLBACK_RESPONSE.format(partial_info=partial)
            state["sources"]  = []
            state["fallback"] = True
            return state

        # Build prompt sections
        risk_s = state["risk"]["prompt_str"]      if has_risk  else ""
        gaps_s = state["gaps"]["prompt_str"]      if has_gaps  else ""
        sc_s   = state["scenario"]["prompt_str"]  if has_sc    else ""
        gl_s   = state["guidelines"]["prompt_str"] if has_gl   else ""

        if self.llm is not None:
            prompt = (COMPOSE_SYSTEM + "\n\n" + COMPOSE_TEMPLATE.format(
                risk_section=risk_s, gaps_section=gaps_s,
                scenario_section=sc_s, guidelines_section=gl_s,
                question=state["query"],
            ))
            try:
                answer = self._call_llm_raw(prompt)
            except Exception as e:
                answer = self._rule_based_answer(state)
                answer += f"\n\n[LLM error: {e}]"
        else:
            answer = self._rule_based_answer(state)

        state["answer"]   = answer
        state["sources"]  = sources
        state["fallback"] = False
        return state

    # ── Build graph ───────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("route",      self._node_route)
        g.add_node("risk",       self._node_risk)
        g.add_node("gaps",       self._node_gaps)
        g.add_node("scenario",   self._node_scenario)
        g.add_node("guidelines", self._node_guidelines)
        g.add_node("compose",    self._node_compose)

        g.set_entry_point("route")
        g.add_edge("route",      "risk")
        g.add_edge("risk",       "gaps")
        g.add_edge("gaps",       "scenario")
        g.add_edge("scenario",   "guidelines")
        g.add_edge("guidelines", "compose")
        g.add_edge("compose",    END)
        return g.compile()

    # ── LLM helpers ──────────────────────────────────────────────────────────

    def _call_llm_raw(self, prompt: str) -> str:
        """Call the LLM backend directly with a raw prompt string."""
        from rag.llm import _call_groq, _call_ollama, _call_anthropic
        if self.llm.backend == "groq":
            return _call_groq(question=prompt, context="", kit_gaps="",
                              model=self.llm.groq_model, temperature=0.2)
        elif self.llm.backend == "anthropic":
            return _call_anthropic(question=prompt, context="", kit_gaps="", temperature=0.2)
        else:
            return _call_ollama(question=prompt, context="", kit_gaps="",
                                model=self.llm.ollama_model,
                                base_url=self.llm.ollama_url, temperature=0.2)

    def _rule_based_answer(self, state: AgentState) -> str:
        """Structured answer without LLM — used as fallback."""
        lines = [f"Answer for: {state['query']}\n"]
        if state.get("risk"):
            lines.append(state["risk"]["prompt_str"])
        if state.get("scenario"):
            lines.append(state["scenario"]["prompt_str"])
            if state["scenario"]["recommendations"]:
                lines.append("\nRecommendations:")
                for r in state["scenario"]["recommendations"]:
                    lines.append(f"  • {r}")
        if state.get("gaps") and state["gaps"]["has_gaps"]:
            lines.append(state["gaps"]["prompt_str"])
        if state.get("guidelines") and state["guidelines"]["n_chunks"] > 0:
            lines.append("\nFrom official EU guidelines:")
            lines.append(state["guidelines"]["context"][:600] + "...")
        lines.append("\n[Rule-based answer — LLM not available]")
        return "\n\n".join(lines)

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, query: str, event_type: str = "general",
            duration_hours: int = 72,
            people: Optional[int] = None) -> AgentResponse:
        """Ask the HAVEN agent a question.

        Args:
            query:          Natural language question from the user.
            event_type:     Scenario type for run_scenario (power_outage, flood, etc.).
            duration_hours: Scenario duration for run_scenario.
            people:         Household size override. Defaults to the value set at
                            construction time (self.people).
        """
        initial: AgentState = {
            "query":          query,
            "people":         people if people is not None else self.people,
            "event_type":     event_type,
            "duration_hours": duration_hours,
            "routing":        None,
            "risk":           None,
            "gaps":           None,
            "scenario":       None,
            "guidelines":     None,
            "answer":         "",
            "sources":        [],
            "fallback":       False,
            "intent":         "",
        }
        final = self._graph.invoke(initial)
        return AgentResponse(
            query=    query,
            intent=   final["intent"],
            answer=   final["answer"],
            sources=  final["sources"],
            fallback= final["fallback"],
            routing=  final["routing"],
            tool_results={
                "risk":     final.get("risk"),
                "gaps":     final.get("gaps"),
                "scenario": final.get("scenario"),
                "guidelines": {
                    "n_chunks": final["guidelines"]["n_chunks"] if final.get("guidelines") else 0,
                    "sources":  final["guidelines"]["sources"]  if final.get("guidelines") else [],
                },
            },
        )

    def print_response(self, r: AgentResponse) -> None:
        """Pretty-print an AgentResponse."""
        print(f"\n{'='*65}")
        print(f"Q: {r.query}")
        print(f"{'='*65}")
        print(f"Intent  : {r.intent}  |  Routing: {r.routing['reasoning']}")
        print(f"Fallback: {r.fallback}")
        print()
        print(r.answer)
        if r.sources:
            print(f"\n--- Sources ---")
            for s in r.sources:
                print(f"  • {s}")
        tr = r.tool_results
        print(f"\n--- Tools used ---")
        print(f"  risk      : {'✓' if tr['risk']     else '—'}")
        print(f"  gaps      : {'✓' if tr['gaps']     else '—'}")
        print(f"  scenario  : {'✓' if tr['scenario'] else '—'}")
        print(f"  guidelines: {tr['guidelines']['n_chunks']} chunks")
