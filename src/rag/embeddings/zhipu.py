"""ZhipuAI Embedding builder.

Provides a factory function for LlamaIndex ZhipuAI embeddings (embedding-3).
Configuration priority: YAML config > environment variable > defaults.
"""

from typing import List
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from src.common.config import (
    get_zhipu_api_key,
    get_embedding_dimension,
    get_embedding_model,
    get_embeddings_config
)
from src.rag.embeddings.base import BaseEmbeddingModel


class ZhipuEmbeddingWrapper(BaseEmbeddingModel):
    """Wrapper for ZhipuAI Embedding to match BaseEmbeddingModel interface.

    Provides consistent interface across all embedding providers.
    Delegates actual embedding computation to LlamaIndex's ZhipuAIEmbedding.
    """

    def __init__(self, zhipu_embedding: ZhipuAIEmbedding, model_name: str):
        """Initialize wrapper.

        Args:
            zhipu_embedding: ZhipuAIEmbedding instance
            model_name: Model name for logging
        """
        super().__init__()
        self._zhipu_embedding = zhipu_embedding
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        """Return embedding dimension."""
        return self._zhipu_embedding.dimensions

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._zhipu_embedding._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._zhipu_embedding._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch text embeddings."""
        return self._zhipu_embedding._get_text_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return await self._zhipu_embedding._aget_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return await self._zhipu_embedding._aget_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch text embeddings asynchronously."""
        return await self._zhipu_embedding._aget_text_embeddings(texts)


def build_embedding() -> BaseEmbeddingModel:
    """Build and return a ZhipuAI Embedding instance.

    Configuration priority:
    1. configs/embeddings.yaml (zhipu section)
    2. Environment variables (EMBEDDING_MODEL, EMBEDDING_DIMENSION)
    3. Defaults (embedding-3, 1024 dimensions)

    Returns:
        BaseEmbeddingModel: Configured embedding instance

    Raises:
        ValueError: If ZHIPU_API_KEY is not set

    Example:
        >>> embed_model = build_embedding()
        >>> dim = embed_model.dimensions  # Returns 1024
        >>> embeddings = embed_model.get_text_embeddings(["sample text"])
    """
    # Get API key (required)
    api_key = get_zhipu_api_key()
    if not api_key:
        raise ValueError(
            "ZHIPU_API_KEY not set. Please set it in environment variables or .env file"
        )

    # Get model name and dimension with priority handling
    model = get_embedding_model()
    dimensions = get_embedding_dimension()

    # Create ZhipuAI embedding instance
    zhipu_embedding = ZhipuAIEmbedding(
        model=model,
        api_key=api_key,
        dimensions=dimensions
    )

    # Wrap it to match BaseEmbeddingModel interface
    return ZhipuEmbeddingWrapper(
        zhipu_embedding=zhipu_embedding,
        model_name=f"zhipu-{model}"
    )