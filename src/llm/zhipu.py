"""ZhipuAI LLM builder.

Provides a factory for LlamaIndex ZhipuAI LLM (e.g., 'glm-4').
Configuration priority: YAML config > environment variable > function parameters > defaults.
"""

from typing import Optional
from llama_index.llms.zhipuai import ZhipuAI
from src.common.config import (
    get_zhipu_api_key,
    get_llm_model,
    get_llm_temperature,
    get_llm_max_tokens,
    get_llm_top_p,
    get_llm_system_prompt,
)


def build_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    system_prompt: Optional[str] = None,
):
    """Build and return a ZhipuAI LLM instance.
    
    Configuration priority:
    1. Function parameters (if provided)
    2. configs/llm.yaml (llm.zhipu section)
    3. Environment variables (LLM_MODEL, LLM_TEMPERATURE, etc.)
    4. Defaults (glm-4, 0.2, 2048, 0.7)
    
    Args:
        model: Model name (e.g., glm-4, glm-4-plus, glm-4-turbo)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0-1.0)
        system_prompt: System prompt to guide the model's behavior
    
    Returns:
        ZhipuAI: Configured LLM instance
        
    Raises:
        ValueError: If ZHIPU_API_KEY is not set
        
    Example:
        >>> # Use default configuration from configs/llm.yaml
        >>> llm = build_llm()
        
        >>> # Override specific parameters
        >>> llm = build_llm(temperature=0.5, max_tokens=4096)
        
        >>> # Use custom system prompt
        >>> llm = build_llm(system_prompt="You are a helpful assistant.")
    """
    # Get API key (required)
    api_key = get_zhipu_api_key()
    if not api_key:
        raise ValueError(
            "ZHIPU_API_KEY not set. Please set it in environment variables or .env file"
        )
    
    # Use provided parameters or fall back to config
    final_model = model if model is not None else get_llm_model()
    final_temperature = temperature if temperature is not None else get_llm_temperature()
    final_max_tokens = max_tokens if max_tokens is not None else get_llm_max_tokens()
    final_top_p = top_p if top_p is not None else get_llm_top_p()
    final_system_prompt = system_prompt if system_prompt is not None else get_llm_system_prompt()
    
    # Build kwargs for ZhipuAI
    llm_kwargs = {
        "model": final_model,
        "api_key": api_key,
        "temperature": final_temperature,
        "max_tokens": final_max_tokens,
        "top_p": final_top_p,
    }
    
    # Add system_prompt if provided
    if final_system_prompt:
        llm_kwargs["system_prompt"] = final_system_prompt
    
    return ZhipuAI(**llm_kwargs)
