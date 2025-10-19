"""Configuration management for MediMind Agent.

Handles environment variables, YAML configuration files, and runtime configuration.
"""

import os
from dotenv import load_dotenv
import yaml


# Load environment variables from .env file
load_dotenv()


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}


def get_zhipu_api_key():
    """Get ZhipuAI API key from environment variables.
    
    Returns:
        str: API key for ZhipuAI services
    """
    return os.getenv("ZHIPU_API_KEY")


def get_embedding_dimension():
    """Get embedding dimension with priority: YAML config > environment variable > default.
    
    Priority order:
    1. configs/embeddings.yaml (zhipu.dim)
    2. EMBEDDING_DIMENSION environment variable
    3. Default: 1024 (matches ZhipuAI embedding-3)
    
    Returns:
        int: Embedding dimension
    """
    # Try to get from YAML config first
    embeddings_config = get_embeddings_config()
    if embeddings_config and 'embeddings' in embeddings_config:
        zhipu_config = embeddings_config['embeddings'].get('zhipu', {})
        if 'dim' in zhipu_config:
            return zhipu_config['dim']
    
    # Fall back to environment variable
    dim = os.getenv("EMBEDDING_DIMENSION")
    if dim and dim.isdigit():
        return int(dim)
    
    # Default to 1024 for embedding-3
    return 1024


def get_embedding_model():
    """Get embedding model name from environment variables or default to embedding-3.
    
    Returns:
        str: Embedding model name
    """
    return os.getenv("EMBEDDING_MODEL", "embedding-3")


def get_biobert_model_path():
    """Get BioBERT model path from environment variables or default to None.
    
    Returns:
        str: Path to BioBERT model files, or None to use Hugging Face model
    """
    return os.getenv("BIOBERT_MODEL_PATH")


def get_embeddings_config():
    """Get embeddings configuration from YAML file.
    
    Returns:
        dict: Embeddings configuration
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "embeddings.yaml")
    return load_yaml_config(config_path)


def get_rag_config():
    """Get RAG configuration from YAML file.
    
    Returns:
        dict: RAG configuration
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "rag.yaml")
    return load_yaml_config(config_path)


def get_llm_config():
    """Get LLM configuration from YAML file.
    
    Returns:
        dict: LLM configuration
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "llm.yaml")
    return load_yaml_config(config_path)


def get_llm_model():
    """Get LLM model name with priority: YAML config > environment variable > default.
    
    Priority order:
    1. configs/llm.yaml (llm.zhipu.model)
    2. LLM_MODEL environment variable
    3. Default: glm-4
    
    Returns:
        str: LLM model name
    """
    # Try YAML config first
    llm_config = get_llm_config()
    if llm_config and 'llm' in llm_config:
        zhipu_config = llm_config['llm'].get('zhipu', {})
        if 'model' in zhipu_config:
            return zhipu_config['model']
    
    # Fall back to environment variable
    model = os.getenv("LLM_MODEL")
    if model:
        return model
    
    # Default
    return "glm-4"


def get_llm_temperature():
    """Get LLM temperature with priority: YAML config > environment variable > default.
    
    Priority order:
    1. configs/llm.yaml (llm.zhipu.temperature)
    2. LLM_TEMPERATURE environment variable
    3. Default: 0.2
    
    Returns:
        float: LLM temperature
    """
    # Try YAML config first
    llm_config = get_llm_config()
    if llm_config and 'llm' in llm_config:
        zhipu_config = llm_config['llm'].get('zhipu', {})
        if 'temperature' in zhipu_config:
            return float(zhipu_config['temperature'])
    
    # Fall back to environment variable
    temp = os.getenv("LLM_TEMPERATURE")
    if temp:
        try:
            return float(temp)
        except ValueError:
            pass
    
    # Default
    return 0.2


def get_llm_max_tokens():
    """Get LLM max tokens with priority: YAML config > environment variable > default.
    
    Priority order:
    1. configs/llm.yaml (llm.zhipu.max_tokens)
    2. LLM_MAX_TOKENS environment variable
    3. Default: 2048
    
    Returns:
        int: Maximum tokens to generate
    """
    # Try YAML config first
    llm_config = get_llm_config()
    if llm_config and 'llm' in llm_config:
        zhipu_config = llm_config['llm'].get('zhipu', {})
        if 'max_tokens' in zhipu_config:
            return int(zhipu_config['max_tokens'])
    
    # Fall back to environment variable
    max_tokens = os.getenv("LLM_MAX_TOKENS")
    if max_tokens and max_tokens.isdigit():
        return int(max_tokens)
    
    # Default
    return 2048


def get_llm_top_p():
    """Get LLM top_p with priority: YAML config > environment variable > default.
    
    Priority order:
    1. configs/llm.yaml (llm.zhipu.top_p)
    2. LLM_TOP_P environment variable
    3. Default: 0.7
    
    Returns:
        float: Nucleus sampling parameter
    """
    # Try YAML config first
    llm_config = get_llm_config()
    if llm_config and 'llm' in llm_config:
        zhipu_config = llm_config['llm'].get('zhipu', {})
        if 'top_p' in zhipu_config:
            return float(zhipu_config['top_p'])
    
    # Fall back to environment variable
    top_p = os.getenv("LLM_TOP_P")
    if top_p:
        try:
            return float(top_p)
        except ValueError:
            pass
    
    # Default
    return 0.7


def get_llm_system_prompt():
    """Get LLM system prompt with priority: YAML config > environment variable > None.
    
    Priority order:
    1. configs/llm.yaml (llm.zhipu.system_prompt)
    2. LLM_SYSTEM_PROMPT environment variable
    3. None (no system prompt)
    
    Returns:
        str or None: System prompt for the LLM
    """
    # Try YAML config first
    llm_config = get_llm_config()
    if llm_config and 'llm' in llm_config:
        zhipu_config = llm_config['llm'].get('zhipu', {})
        if 'system_prompt' in zhipu_config:
            prompt = zhipu_config['system_prompt']
            if prompt and prompt.strip():
                return prompt.strip()
    
    # Fall back to environment variable
    prompt = os.getenv("LLM_SYSTEM_PROMPT")
    if prompt and prompt.strip():
        return prompt.strip()
    
    # No system prompt
    return None


def get_index_directory():
    """Get index directory from config or environment variable.
    
    Priority order:
    1. configs/embeddings.yaml (zhipu.index_dir)
    2. INDEX_DIR environment variable
    3. Default: data/indexes/zhipu
    
    Returns:
        str: Index directory path
    """
    # Try to get from YAML config first
    embeddings_config = get_embeddings_config()
    if embeddings_config and 'embeddings' in embeddings_config:
        zhipu_config = embeddings_config['embeddings'].get('zhipu', {})
        if 'index_dir' in zhipu_config:
            return zhipu_config['index_dir']
    
    # Fall back to environment variable
    index_dir = os.getenv("INDEX_DIR")
    if index_dir:
        return index_dir
    
    # Default directory
    return "data/indexes/zhipu"


def get_documents_directory():
    """Get documents directory from environment variable or default.
    
    Returns:
        str: Documents directory path
    """
    return os.getenv("DOCUMENTS_DIR", "data/documents")


def get_config():
    """Get all configuration as a dictionary.
    
    Returns:
        dict: Configuration values
    """
    return {
        "zhipu_api_key": get_zhipu_api_key(),
        "embedding_dimension": get_embedding_dimension(),
        "embedding_model": get_embedding_model(),
        "index_directory": get_index_directory(),
        "documents_directory": get_documents_directory(),
        "llm_model": get_llm_model(),
        "llm_temperature": get_llm_temperature(),
        "llm_max_tokens": get_llm_max_tokens(),
        "llm_top_p": get_llm_top_p(),
        "embeddings": get_embeddings_config(),
        "rag": get_rag_config(),
        "llm": get_llm_config(),
    }