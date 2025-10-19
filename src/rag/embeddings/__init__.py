"""Embedding models for medical document vectorization.

This module provides embedding model utilities using ZhipuAI embedding-3.
"""

from src.rag.embeddings.zhipu import build_embedding

__all__ = ["build_embedding"]
