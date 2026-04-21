"""
rag/llm.py
-----------
LLM wrapper for HAVEN RAG.

Three backends, one interface:
    LOCAL (development):  Ollama running Mistral 7B locally
    DEPLOYED (free):      Groq API — Llama 3.1 8B (14,400 req/day free tier)
    FALLBACK (paid):      Anthropic API — Claude Haiku

Backend selection via LLM_BACKEND environment variable:
    LLM_BACKEND=ollama      → local Mistral via Ollama (default)
    LLM_BACKEND=groq        → Groq API, free tier (deployment)
    LLM_BACKEND=anthropic   → Claude Haiku (paid fallback)

Deployment rationale:
    Groq provides a genuinely free tier (14,400 req/day, no credit card)
    running Llama 3.1 8B on custom inference hardware — fast and free.
    Ollama/Mistral is used locally since Render/Railway free tier has no GPU.
    The prompt template is identical across all backends — only the HTTP
    call changes, making local↔cloud switching transparent to the pipeline.
"""

import os
import requests
from typing import List, Optional

from rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are HAVEN, an AI emergency preparedness advisor.
Your role is to help users understand their emergency kit needs based on
official EU government guidance.

Rules:
- Answer ONLY using the provided source documents and kit gap information.
- Always cite your sources using [Source: document name, Page X] format.
- If the answer is not in the sources, say so clearly — do not guess.
- Be concise and practical. Users want actionable advice.
- When kit gaps are provided, always connect your answer to the user's specific situation."""

USER_PROMPT_TEMPLATE = """KNOWLEDGE BASE (Official EU emergency preparedness guidance):
{context}

USER'S CURRENT KIT GAPS:
{kit_gaps}

QUESTION: {question}

Please answer the question with specific citations and practical advice
tailored to the user's current kit gaps."""


# ---------------------------------------------------------------------------
# Kit gap formatter
# ---------------------------------------------------------------------------

def format_kit_gaps(gaps: list) -> str:
    """
    Format inventory gaps for injection into the LLM prompt.
    Accepts GapItem dataclass objects or plain dicts.
    """
    if not gaps:
        return "No kit gaps detected — kit appears complete."

    lines = []
    for g in gaps:
        if hasattr(g, "name"):
            name, current, recommend = g.name, g.current, g.recommended
            unit, priority, pct      = g.unit, g.priority, g.gap_pct
        else:
            name      = g.get("name", "Unknown")
            current   = g.get("current", 0)
            recommend = g.get("recommended", 0)
            unit      = g.get("unit", "")
            priority  = g.get("priority", "MEDIUM")
            pct       = g.get("gap_pct", 0)

        lines.append(
            f"- {name}: have {current:.1f} {unit}, need {recommend:.1f} {unit} "
            f"({pct:.0f}% missing) [{priority} priority]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama backend (local development)
# ---------------------------------------------------------------------------

def _call_ollama(
    question:    str,
    context:     str,
    kit_gaps:    str,
    model:       str = "mistral",
    base_url:    str = "http://localhost:11434",
    temperature: float = 0.2,
) -> str:
    """Call local Ollama instance. Model: mistral or llama3.1."""
    prompt = f"{SYSTEM_PROMPT}\n\n" + USER_PROMPT_TEMPLATE.format(
        context=  context,
        kit_gaps= kit_gaps,
        question= question,
    )

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 1024},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Then pull a model: ollama pull mistral"
        )


# ---------------------------------------------------------------------------
# Groq backend (free deployment — Llama 3.1 8B)
# ---------------------------------------------------------------------------

def _call_groq(
    question:    str,
    context:     str,
    kit_gaps:    str,
    model:       str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    max_tokens:  int = 1024,
) -> str:
    """
    Call Groq API — free tier, OpenAI-compatible.
    Free quota: 14,400 requests/day, 6,000 tokens/minute.
    Get your free API key at: console.groq.com

    Available models (all free):
        llama-3.1-8b-instant   — fast, good quality (recommended)
        llama-3.3-70b-versatile — better quality, slower
        mixtral-8x7b-32768     — long context option
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Get a free key at console.groq.com and add it to your .env file."
        )

    user_content = USER_PROMPT_TEMPLATE.format(
        context=  context,
        kit_gaps= kit_gaps,
        question= question,
    )

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":       model,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Anthropic backend (paid fallback — Claude Haiku)
# ---------------------------------------------------------------------------

def _call_anthropic(
    question:    str,
    context:     str,
    kit_gaps:    str,
    model:       str = "claude-haiku-4-5-20251001",
    temperature: float = 0.2,
    max_tokens:  int = 1024,
) -> str:
    """
    Call Anthropic API. Requires ANTHROPIC_API_KEY.
    Cost: ~$0.25/1M input tokens (cheapest capable Anthropic model).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      model,
            "max_tokens": max_tokens,
            "system":     SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                context=  context,
                kit_gaps= kit_gaps,
                question= question,
            )}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Unified LLM interface
# ---------------------------------------------------------------------------

class HavenLLM:
    """
    Unified LLM interface for HAVEN RAG.

    Backend selected via LLM_BACKEND env var:
        ollama     → local Mistral 7B (default, development)
        groq       → Llama 3.1 8B via Groq API (free deployment)
        anthropic  → Claude Haiku (paid fallback)

    Usage:
        llm = HavenLLM()                    # reads LLM_BACKEND from env
        llm = HavenLLM(backend="groq")      # explicit override
        answer = llm.answer(
            question="Why do I need a battery-powered radio?",
            retrieved_chunks=chunks,
            gaps=inv_report.gaps,
        )
    """

    # Groq model options for easy switching
    GROQ_MODELS = {
        "fast":    "llama-3.1-8b-instant",      # recommended: fast + free
        "quality": "llama-3.3-70b-versatile",   # better answers, same free tier
        "long":    "mixtral-8x7b-32768",         # long context (32k tokens)
    }

    def __init__(
        self,
        backend:      Optional[str] = None,
        ollama_model: str = "mistral",
        ollama_url:   str = "http://localhost:11434",
        groq_model:   str = "llama-3.1-8b-instant",
    ):
        self.backend      = backend or os.getenv("LLM_BACKEND", "ollama")
        self.ollama_model = ollama_model
        self.ollama_url   = ollama_url
        self.groq_model   = groq_model
        print(f"HavenLLM ready — backend: {self.backend}")
        if self.backend == "groq":
            print(f"  Groq model: {self.groq_model}")
            print(f"  Free quota: 14,400 req/day | Sign up: console.groq.com")

    def answer(
        self,
        question:         str,
        retrieved_chunks: List[RetrievedChunk],
        gaps:             list = None,
        temperature:      float = 0.2,
    ) -> str:
        """
        Generate a grounded, cited answer to a preparedness question.

        Args:
            question:         Natural language question from the user.
            retrieved_chunks: Top-k chunks from HavenRetriever.
            gaps:             GapItem list from analyze_inventory().
            temperature:      LLM temperature (0.2 = focused, 0.7 = creative).

        Returns:
            Cited answer string ready to display in the UI.
        """
        context = "\n\n".join(
            f"[{i+1}] Source: {c.source}, Page {c.page}\n{c.text}"
            for i, c in enumerate(retrieved_chunks)
        )
        kit_gaps = format_kit_gaps(gaps or [])

        if self.backend == "groq":
            return _call_groq(
                question=    question,
                context=     context,
                kit_gaps=    kit_gaps,
                model=       self.groq_model,
                temperature= temperature,
            )
        elif self.backend == "anthropic":
            return _call_anthropic(
                question=    question,
                context=     context,
                kit_gaps=    kit_gaps,
                temperature= temperature,
            )
        else:
            return _call_ollama(
                question=    question,
                context=     context,
                kit_gaps=    kit_gaps,
                model=       self.ollama_model,
                base_url=    self.ollama_url,
                temperature= temperature,
            )

    def is_available(self) -> bool:
        """Check whether the configured backend is reachable."""
        if self.backend == "groq":
            return bool(os.getenv("GROQ_API_KEY"))
        elif self.backend == "anthropic":
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        else:
            try:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                return resp.status_code == 200
            except Exception:
                return False

    def list_ollama_models(self) -> list:
        """List models available in local Ollama instance."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
