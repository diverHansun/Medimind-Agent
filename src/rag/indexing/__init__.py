"""Indexing module for building and managing FAISS vector indexes.

This module provides utilities for building, loading, and managing
FAISS vector indexes using LlamaIndex.
"""

from src.rag.indexing.builder import (
    build_index,
    load_index,
    index_exists,
    build_index_if_not_exists,
    add_documents_to_index,
    add_nodes_to_index,
)

__all__ = [
    "build_index",
    "load_index",
    "index_exists",
    "build_index_if_not_exists",
    "add_documents_to_index",
    "add_nodes_to_index",
]



