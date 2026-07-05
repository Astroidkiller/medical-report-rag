"""
Vector store wrapper.
Thin wrapper around core.embeddings for use by agents.
"""

from core.embeddings import (
    embed_texts,
    store_chunks,
    query_similar,
    clear_collection,
    get_embedding_model,
)

# Re-export for convenience
__all__ = [
    "embed_texts",
    "store_chunks",
    "query_similar",
    "clear_collection",
    "get_embedding_model",
]
