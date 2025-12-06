"""Chat memory management using LlamaIndex ChatSummaryMemoryBuffer.

This module provides functions to build and manage conversation memory
with automatic summarization when token limit is exceeded.
"""

import os
from typing import Optional
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.llms import LLM
from src.common.config import load_yaml_config


# Default summarization prompt
DEFAULT_SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, "
    "preserving key information and important context."
)


def load_memory_config(config_path: str = "configs/memory.yaml") -> dict:
    """Load memory configuration from YAML file.

    Args:
        config_path: Path to memory configuration file

    Returns:
        dict: Memory configuration dictionary

    Example:
        >>> config = load_memory_config("configs/memory.yaml")
        >>> print(config["memory"]["token_limit"])
        64000
    """
    config = load_yaml_config(config_path)
    return config.get("memory", {})


def build_chat_memory(
    llm: LLM,
    token_limit: int = 64000,
    summarize_prompt: Optional[str] = None,
) -> ChatSummaryMemoryBuffer:
    """Build a chat memory buffer for multi-turn conversations.

    Uses ChatSummaryMemoryBuffer to maintain conversation history with
    automatic summarization when token limit is exceeded.

    Args:
        llm: LLM instance for generating summaries
        token_limit: Maximum token count for memory buffer (default: 64000)
        summarize_prompt: Optional custom summarization prompt.
                         If None, uses default prompt.

    Returns:
        ChatSummaryMemoryBuffer: Configured memory buffer instance

    Raises:
        ValueError: If token_limit is less than 1000

    Example:
        >>> from src.llm.zhipu import build_llm
        >>>
        >>> llm = build_llm()
        >>> memory = build_chat_memory(llm=llm, token_limit=64000)
        >>>
        >>> # With custom prompt
        >>> custom_prompt = "Summarize the medical consultation..."
        >>> memory = build_chat_memory(llm=llm, summarize_prompt=custom_prompt)
    """
    if token_limit < 1000:
        raise ValueError(
            f"token_limit must be at least 1000, got {token_limit}"
        )

    # Use default prompt if not provided
    final_prompt = summarize_prompt or DEFAULT_SUMMARIZE_PROMPT

    # Create memory buffer
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=llm,
        token_limit=token_limit,
        summarize_prompt=final_prompt,
        count_initial_tokens=False,
    )

    return memory
