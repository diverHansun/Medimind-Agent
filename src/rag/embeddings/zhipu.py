"""ZhipuAI Embedding builder.

Provides a factory function for LlamaIndex ZhipuAI embeddings (embedding-3).
Configuration priority: YAML config > environment variable > defaults.
"""

from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from src.common.config import (
    get_zhipu_api_key, 
    get_embedding_dimension, 
    get_embedding_model,
    get_embeddings_config
)


def build_embedding():
    """Build and return a ZhipuAI Embedding instance.
    
    Configuration priority:
    1. configs/embeddings.yaml (zhipu section)
    2. Environment variables (EMBEDDING_MODEL, EMBEDDING_DIMENSION)
    3. Defaults (embedding-3, 1024 dimensions)
    
    Returns:
        ZhipuAIEmbedding: Configured embedding instance with dimensions attribute
        
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
    
    # Create and return embedding instance
    return ZhipuAIEmbedding(
        model=model,
        api_key=api_key,
        dimensions=dimensions
    )