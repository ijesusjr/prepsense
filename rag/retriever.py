"""
rag/retriever.py
-----------------
Given a natural language query, returns the top-k most relevant chunks
from the FAISS index with source citations.

The retriever is the bridge between the user's question and the LLM.
It ensures the LLM answers from actual document content, not hallucination.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from rag.chunker import Chunk
from rag.embedder import EMBEDDING_MODEL, load_index


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id:   int
    text:       str
    source:     str
    page:       int
    score:      float    # cosine similarity (0-1, higher = more relevant)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HavenRetriever:
    """
    Semantic retriever for HAVEN RAG pipeline.

    Usage:
        retriever = HavenRetriever.from_disk(
            index_path="data/faiss/index.bin",
            meta_path="data/faiss/chunks.json",
        )
        results = retriever.query("why do I need water in my emergency kit?", k=3)
        for r in results:
            print(f"[{r.source}, p{r.page}] score={r.score:.3f}")
            print(r.text)
    """

    def __init__(
        self,
        index:      faiss.Index,
        chunks:     List[Chunk],
        model_name: str = EMBEDDING_MODEL,
    ):
        self.index  = index
        self.chunks = chunks
        self._model = None
        self._model_name = model_name

    @classmethod
    def from_disk(
        cls,
        index_path: str,
        meta_path:  str,
        model_name: str = EMBEDDING_MODEL,
    ) -> "HavenRetriever":
        """Load retriever from persisted FAISS index and chunk metadata."""
        index, chunks = load_index(index_path, meta_path)
        return cls(index=index, chunks=chunks, model_name=model_name)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def query(
        self,
        question: str,
        k:        int = 4,
        min_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Args:
            question:  Natural language query.
            k:         Number of chunks to return.
            min_score: Minimum cosine similarity threshold (0-1).

        Returns:
            List of RetrievedChunk sorted by relevance descending.
        """
        # Embed the query with the same model and normalisation
        query_vec = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search FAISS
        k_actual  = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k_actual)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                  # FAISS returns -1 for empty slots
                continue
            if float(score) < min_score:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                chunk_id= chunk.chunk_id,
                text=     chunk.text,
                source=   chunk.source,
                page=     chunk.page,
                score=    float(score),
            ))

        return results

    def format_context(self, results: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as a numbered context block for LLM injection.

        Returns:
            String ready to insert into the LLM prompt.
        """
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] Source: {r.source}, Page {r.page} "
                f"(relevance: {r.score:.2f})\n{r.text}"
            )
        return "\n\n".join(lines)
