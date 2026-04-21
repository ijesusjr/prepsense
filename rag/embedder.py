"""
rag/embedder.py
----------------
Generates sentence embeddings for chunks and builds a FAISS index.

Model: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Fast inference, small footprint (~80MB)
    - Strong performance on semantic similarity tasks
    - Fully local, no API calls

FAISS index: IndexFlatL2
    - Exact nearest-neighbour search (no approximation)
    - Appropriate for our small corpus (~20 chunks)
    - Persisted to disk alongside chunk metadata
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
import faiss

from rag.chunker import Chunk, load_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[Chunk],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of chunks.

    Args:
        chunks:        List of Chunk objects.
        model_name:    SentenceTransformer model to use.
        batch_size:    Encoding batch size.
        show_progress: Show tqdm progress bar.

    Returns:
        numpy array of shape (n_chunks, embedding_dim), float32.
    """
    print(f"Loading embedding model: {model_name}")
    model  = SentenceTransformer(model_name)
    texts  = [c.text for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=    batch_size,
        show_progress_bar= show_progress,
        convert_to_numpy=  True,
        normalize_embeddings= True,   # L2 normalise → cosine similarity via dot product
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat index from embeddings.
    Uses IndexFlatIP (inner product) since embeddings are L2-normalised —
    inner product equals cosine similarity.

    Args:
        embeddings: float32 array of shape (n, dim).

    Returns:
        Populated FAISS index.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine similarity (post-normalisation)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(
    index:      faiss.Index,
    chunks:     List[Chunk],
    index_path: str,
    meta_path:  str,
) -> None:
    """
    Persist FAISS index and chunk metadata to disk.

    Args:
        index:      Populated FAISS index.
        chunks:     Corresponding list of Chunk objects.
        index_path: Output path for the FAISS binary (e.g. data/faiss/index.bin).
        meta_path:  Output path for chunk metadata JSON.
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved → {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"chunk_id": c.chunk_id, "text": c.text,
              "source": c.source, "page": c.page, "tokens": c.tokens}
             for c in chunks],
            f, indent=2, ensure_ascii=False,
        )
    print(f"Chunk metadata saved → {meta_path}")


def load_index(
    index_path: str,
    meta_path:  str,
) -> Tuple[faiss.Index, List[Chunk]]:
    """
    Load a persisted FAISS index and its chunk metadata.

    Returns:
        (faiss_index, list_of_chunks)
    """
    index = faiss.read_index(index_path)
    chunks = load_chunks(meta_path)
    print(f"Loaded FAISS index: {index.ntotal} vectors")
    print(f"Loaded {len(chunks)} chunks from metadata")
    return index, chunks


def build_and_save(
    pdf_dir:    str,
    faiss_dir:  str,
    model_name: str = EMBEDDING_MODEL,
) -> Tuple[faiss.Index, List[Chunk]]:
    """
    Full pipeline: extract chunks → embed → build index → persist.
    Call this once to build the knowledge base.

    Args:
        pdf_dir:   Directory containing source PDFs.
        faiss_dir: Directory to save index.bin and chunks.json.
        model_name: Embedding model name.

    Returns:
        (faiss_index, chunks)
    """
    from rag.chunker import extract_chunks

    print("=" * 50)
    print("HAVEN RAG — Building Knowledge Base")
    print("=" * 50)

    # Step 1: extract and chunk
    print("\n[1/3] Extracting and chunking PDFs...")
    chunks = extract_chunks(pdf_dir)

    # Step 2: embed
    print("\n[2/3] Generating embeddings...")
    embeddings = embed_chunks(chunks, model_name=model_name)

    # Step 3: index and persist
    print("\n[3/3] Building and saving FAISS index...")
    index      = build_faiss_index(embeddings)
    index_path = str(Path(faiss_dir) / "index.bin")
    meta_path  = str(Path(faiss_dir) / "chunks.json")
    save_index(index, chunks, index_path, meta_path)

    print(f"\nKnowledge base ready: {len(chunks)} chunks indexed")
    return index, chunks
