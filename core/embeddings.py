"""
Embedding generation and vector store management.
Wraps SentenceTransformers, Gemini API, and ChromaDB.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, CHROMA_DB_DIR, TOP_K_RESULTS, EMBEDDING_BACKEND


# Module-level singletons (loaded lazily)
_embedding_model = None
_chroma_client = None


def get_embedding_model():
    """Get or create the sentence transformer model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client():
    """Get or create the ChromaDB client (singleton).

    Uses the modern chromadb.Client() API (ephemeral, in-memory).
    The deprecated chroma_db_impl / persist_directory Settings
    kwargs were removed in ChromaDB >= 0.4.
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        # Modern ChromaDB: use EphemeralClient (in-memory, no deprecated Settings)
        _chroma_client = chromadb.EphemeralClient()
    return _chroma_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (as lists of floats).
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
    Store text chunks and their embeddings in ChromaDB.

    Args:
        collection_name: Name of the ChromaDB collection.
        chunks: Text chunks to store.
        embeddings: Corresponding embedding vectors.
        metadata_list: Optional list of metadata dicts per chunk.
        id_prefix: Prefix for chunk IDs to avoid collisions across reports.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        doc_id = f"{id_prefix}_{i}" if id_prefix else str(i)
        meta = metadata_list[i] if metadata_list else {}
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[doc_id],
            metadatas=[meta],
        )


def query_similar(
    collection_name: str,
    query_text: str,
    n_results: int = None,
) -> dict:
    """
    Query ChromaDB for chunks most similar to the query text.

    Args:
        collection_name: Name of the ChromaDB collection.
        query_text: User's question or search query.
        n_results: Number of results to return (default from config).

    Returns:
        ChromaDB query results dict with 'documents', 'metadatas', 'distances'.
    """
    if n_results is None:
        n_results = TOP_K_RESULTS

    query_embedding = embed_texts([query_text])[0]

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results


def clear_collection(collection_name: str) -> None:
    """Delete all documents from a ChromaDB collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection doesn't exist yet
