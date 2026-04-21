"""
rag/pipeline.py
----------------
End-to-end HAVEN RAG pipeline.

Wires retriever → LLM → cited answer.
This is the single entry point for the Streamlit UI and FastAPI backend.

Usage:
    pipeline = HavenPipeline.from_disk(
        index_path="data/faiss/index.bin",
        meta_path="data/faiss/chunks.json",
    )

    answer = pipeline.ask(
        question="Why do I need bottled water in my emergency kit?",
        gaps=inv_report.gaps,
        k=4,
    )
    print(answer.response)
    print(answer.sources)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from rag.retriever import HavenRetriever, RetrievedChunk
from rag.llm import HavenLLM


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    question:   str
    response:   str
    chunks:     List[RetrievedChunk]
    sources:    List[str]        # deduplicated source labels
    backend:    str              # "ollama" or "anthropic"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HavenPipeline:
    """
    End-to-end HAVEN RAG pipeline.

    Retrieves relevant chunks from the knowledge base,
    injects them + kit gaps into the LLM prompt,
    and returns a cited answer.
    """

    def __init__(
        self,
        retriever:    HavenRetriever,
        llm:          HavenLLM,
    ):
        self.retriever = retriever
        self.llm       = llm

    @classmethod
    def from_disk(
        cls,
        index_path:   str = "data/faiss/index.bin",
        meta_path:    str = "data/faiss/chunks.json",
        backend:      Optional[str] = None,
        ollama_model: str = "mistral",
    ) -> "HavenPipeline":
        """
        Load pipeline from persisted FAISS index.

        Args:
            index_path:   Path to FAISS index binary.
            meta_path:    Path to chunk metadata JSON.
            backend:      "ollama" or "anthropic" (defaults to LLM_BACKEND env var).
            ollama_model: Ollama model name (e.g. "mistral", "llama3.1").
        """
        retriever = HavenRetriever.from_disk(index_path, meta_path)
        llm       = HavenLLM(backend=backend, ollama_model=ollama_model)
        return cls(retriever=retriever, llm=llm)

    def ask(
        self,
        question:    str,
        gaps:        list = None,
        k:           int = 4,
        min_score:   float = 0.15,
        temperature: float = 0.2,
    ) -> RAGResponse:
        """
        Ask a preparedness question and get a cited, gap-aware answer.

        Args:
            question:    Natural language question.
            gaps:        List of GapItem objects from inventory analysis.
            k:           Number of chunks to retrieve.
            min_score:   Minimum retrieval similarity threshold.
            temperature: LLM temperature.

        Returns:
            RAGResponse with answer, chunks, and source list.
        """
        # Step 1: retrieve relevant chunks
        chunks = self.retriever.query(question, k=k, min_score=min_score)

        if not chunks:
            return RAGResponse(
                question= question,
                response= "I could not find relevant information in the knowledge base for this question.",
                chunks=   [],
                sources=  [],
                backend=  self.llm.backend,
            )

        # Step 2: generate answer
        response = self.llm.answer(
            question=         question,
            retrieved_chunks= chunks,
            gaps=             gaps or [],
            temperature=      temperature,
        )

        # Step 3: collect deduplicated sources
        sources = list(dict.fromkeys(
            f"{c.source}, p.{c.page}" for c in chunks
        ))

        return RAGResponse(
            question= question,
            response= response,
            chunks=   chunks,
            sources=  sources,
            backend=  self.llm.backend,
        )

    def print_answer(self, rag_response: RAGResponse) -> None:
        """Pretty-print a RAGResponse to stdout."""
        print(f"\n{'='*60}")
        print(f"Q: {rag_response.question}")
        print(f"{'='*60}")
        print(f"\n{rag_response.response}")
        print(f"\n--- Sources ---")
        for s in rag_response.sources:
            print(f"  • {s}")
        print(f"\n--- Retrieved chunks (top {len(rag_response.chunks)}) ---")
        for c in rag_response.chunks:
            print(f"  [{c.score:.3f}] {c.source}, p{c.page}: {c.text[:80]}...")
        print(f"\nBackend: {rag_response.backend}")
