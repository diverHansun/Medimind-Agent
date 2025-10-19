"""RAG (Retrieval-Augmented Generation) module for MediMind Agent.

This module provides a complete RAG pipeline using LlamaIndex and FAISS,
including embeddings, document loading, text splitting, indexing, retrieval, and query engine building.

Submodules:
    - embeddings: Embedding models (ZhipuAI embedding-3)
    - document_loader: Load documents from various file formats
    - text_splitter: Split documents into chunks for indexing
    - indexing: Build and manage FAISS vector indexes
    - retrieval: Custom retrieval strategies (placeholder)
    - generation: Build query engines for RAG
    - postprocessors: Filter and enhance retrieved nodes
"""

# Import key functions for convenience
from src.rag.embeddings import build_embedding
from src.rag.document_loader import load_documents
from src.rag.text_splitter import split_documents
from src.rag.indexing import (
    build_index,
    load_index,
    index_exists,
    build_index_if_not_exists,
)
from src.rag.generation import build_query_engine, build_simple_query_engine
from src.rag.postprocessors import (
    EvidenceConsistencyChecker,
    SourceFilterPostprocessor,
    DeduplicationPostprocessor,
)

__all__ = [
    # Embeddings
    "build_embedding",
    # Document loading
    "load_documents",
    # Text splitting
    "split_documents",
    # Indexing
    "build_index",
    "load_index",
    "index_exists",
    "build_index_if_not_exists",
    # Query engine
    "build_query_engine",
    "build_simple_query_engine",
    # Postprocessors
    "EvidenceConsistencyChecker",
    "SourceFilterPostprocessor",
    "DeduplicationPostprocessor",
]
