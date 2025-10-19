"""Postprocessor module for filtering and processing retrieved nodes.

This module provides custom postprocessors for filtering, deduplicating,
and enhancing retrieved nodes before response generation.
"""

from src.rag.postprocessors.processors import (
    EvidenceConsistencyChecker,
    SourceFilterPostprocessor,
    DeduplicationPostprocessor,
)

__all__ = [
    "EvidenceConsistencyChecker",
    "SourceFilterPostprocessor",
    "DeduplicationPostprocessor",
]



