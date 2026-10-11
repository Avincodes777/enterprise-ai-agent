"""RAG (Retrieval-Augmented Generation) package for Enterprise AI Agent."""

from .answer_generator import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    RAGAnswerGenerationError,
    RAGAnswerGenerator,
    RAGResponse,
    get_rag_answer_generator,
)
from .context_builder import (
    RAGContext,
    RAGContextBuilder,
    RAGContextError,
    get_rag_context_builder,
)

__all__ = [
    # Context Builder
    "RAGContext",
    "RAGContextBuilder",
    "RAGContextError",
    "get_rag_context_builder",
    # Answer Generator
    "RAGAnswerGenerator",
    "RAGResponse",
    "RAGAnswerGenerationError",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "get_rag_answer_generator",
]

