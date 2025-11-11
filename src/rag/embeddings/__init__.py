"""Embedding models for medical document vectorization.

This module provides flexible embedding model utilities supporting:
- ZhipuAI embedding-3 (cloud API)
- BioBERT v1.1 (local model)

Factory pattern implementation following Open/Closed Principle (OCP).
"""

from src.common.config import get_embedding_provider, get_biobert_config
from src.rag.embeddings.base import BaseEmbeddingModel


def build_embedding() -> BaseEmbeddingModel:
    """Build embedding model based on configuration.

    Factory function that creates the appropriate embedding model instance
    based on the selected provider in configs/embeddings.yaml or environment variable.

    Follows:
    - Open/Closed Principle: Adding new providers requires no modification here
    - Dependency Inversion: Returns base type, not concrete implementation
    - Single Responsibility: Only responsible for model instantiation

    Returns:
        BaseEmbeddingModel: Configured embedding model instance

    Raises:
        ValueError: If provider is not supported
        RuntimeError: If model initialization fails

    Examples:
        # Using Zhipu (default)
        >>> embed_model = build_embedding()
        >>> embed_model.model_name
        'zhipu-embedding-3'

        # Using BioBERT (set provider: biobert in embeddings.yaml)
        >>> embed_model = build_embedding()
        >>> embed_model.model_name
        'biobert-768d'
    """
    provider = get_embedding_provider()

    if provider == "zhipu":
        # Import here to avoid circular dependency
        from src.rag.embeddings.zhipu import build_embedding as build_zhipu
        print(f"[Embedding] Using provider: {provider}")
        return build_zhipu()

    elif provider == "biobert":
        # Import here to avoid circular dependency
        from src.rag.embeddings.biobert import build_biobert_embedding

        # Get BioBERT configuration
        biobert_config = get_biobert_config()
        model_path = biobert_config.get('model_path', 'models/biobert-v1.1')
        normalize = biobert_config.get('normalize', True)
        device = biobert_config.get('device', 'cpu')

        print(f"[Embedding] Using provider: {provider}")
        print(f"[Embedding] Model path: {model_path}")
        print(f"[Embedding] Device: {device}")

        return build_biobert_embedding(
            model_path=model_path,
            normalize=normalize,
            device=device
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported providers: 'zhipu', 'biobert'. "
            f"Please check configs/embeddings.yaml or EMBEDDING_PROVIDER environment variable."
        )


__all__ = ["build_embedding", "BaseEmbeddingModel"]
