"""
Embedding generation and vector store management.
Wraps SentenceTransformers, Gemini API, and ChromaDB.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, CHROMA_DB_DIR, TOP_K_RESULTS, EMBEDDING_BACKEND


import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, CHROMA_DB_DIR, TOP_K_RESULTS, EMBEDDING_BACKEND, VECTOR_STORE_BACKEND


# Module-level singletons (loaded lazily)
_embedding_model = None
_chroma_client = None
_memory_collections = {}


def get_embedding_model():
    """Get or create the sentence transformer model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client():
    """Get or create the ChromaDB client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _chroma_client = chromadb.EphemeralClient()
        except Exception:
            _chroma_client = None
    return _chroma_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    """
    if EMBEDDING_BACKEND in ("gemini", "vertex_ai"):
        from core.llm_client import embed_texts_gemini_rest
        return embed_texts_gemini_rest(texts, model=EMBEDDING_MODEL)
    else:
        model = get_embedding_model()
        embeddings = model.encode(texts)
        return [e.tolist() for e in embeddings]


def store_chunks(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: list[dict] = None,
    id_prefix: str = "",
) -> None:
    """
    Store text chunks and their embeddings in ChromaDB or in-memory vector store.
    """
    if not chunks:
        return

    # In-memory store or fallback if VECTOR_STORE_BACKEND=="memory"
    if VECTOR_STORE_BACKEND == "memory" or get_chroma_client() is None:
        if collection_name not in _memory_collections:
            _memory_collections[collection_name] = []
        
        for i, chunk in enumerate(chunks):
            emb = embeddings[i] if i < len(embeddings) else [0.0] * 768
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            cid = f"{id_prefix}_{i}" if id_prefix else str(i)
            _memory_collections[collection_name].append({
                "id": cid,
                "document": chunk,
                "embedding": np.array(emb, dtype=np.float32),
                "metadata": meta
            })
        return

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    ids = [f"{id_prefix}_{i}" if id_prefix else str(i) for i in range(len(chunks))]

    metadatas = None
    if metadata_list:
        cleaned_metadatas = [m for m in metadata_list if isinstance(m, dict) and len(m) > 0]
        if len(cleaned_metadatas) == len(chunks):
            metadatas = cleaned_metadatas

    add_kwargs = {
        "documents": chunks,
        "embeddings": embeddings,
        "ids": ids,
    }
    if metadatas:
        add_kwargs["metadatas"] = metadatas

    collection.add(**add_kwargs)


def query_similar(
    collection_name: str,
    query_text: str,
    n_results: int = None,
) -> dict:
    """
    Query vector store for chunks most similar to query_text using cosine similarity.
    """
    if n_results is None:
        n_results = TOP_K_RESULTS

    query_embedding = np.array(embed_texts([query_text])[0], dtype=np.float32)

    # In-memory query logic
    if VECTOR_STORE_BACKEND == "memory" or get_chroma_client() is None:
        items = _memory_collections.get(collection_name, [])
        if not items:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        scores = []
        q_norm = np.linalg.norm(query_embedding) + 1e-9
        for item in items:
            emb = item["embedding"]
            e_norm = np.linalg.norm(emb) + 1e-9
            similarity = float(np.dot(query_embedding, emb) / (q_norm * e_norm))
            # Convert cosine similarity to distance
            distance = 1.0 - similarity
            scores.append((distance, item["document"], item["metadata"]))

        scores.sort(key=lambda x: x[0])
        top_k = scores[:n_results]

        return {
            "documents": [[x[1] for x in top_k]],
            "metadatas": [[x[2] for x in top_k]],
            "distances": [[x[0] for x in top_k]]
        }

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )
    return results


def clear_collection(collection_name: str) -> None:
    """Delete all documents from vector store collection."""
    if collection_name in _memory_collections:
        _memory_collections[collection_name] = []
    
    if get_chroma_client():
        try:
            get_chroma_client().delete_collection(name=collection_name)
        except Exception:
            pass  # Collection doesn't exist yet
