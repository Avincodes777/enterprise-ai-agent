"""FastAPI dependency providers for Enterprise AI Agent."""

from functools import lru_cache
import logging

from app.config import get_settings
from app.rag.answer_generator import RAGAnswerGenerator, get_rag_answer_generator

logger = logging.getLogger(__name__)


@lru_cache()
def get_shared_rag_answer_generator() -> RAGAnswerGenerator:
    """
    Return a cached singleton instance of RAGAnswerGenerator.

    Safely reuses the embedding service, vector store client, and LLM service
    across HTTP requests without re-instantiating them each time.
    """
    settings = get_settings()
    return get_rag_answer_generator(settings=settings)


def reset_rag_answer_generator_cache() -> None:
    """Clear the cached shared RAGAnswerGenerator instance."""
    get_shared_rag_answer_generator.cache_clear()


def get_rag_answer_generator_dependency() -> RAGAnswerGenerator:
    """
    FastAPI dependency that returns a RAGAnswerGenerator.

    Designed for injection in API routes and override in tests via
    app.dependency_overrides[get_rag_answer_generator_dependency].
    """
    return get_shared_rag_answer_generator()


@lru_cache()
def get_shared_ingestion_service() -> "IngestionService":
    """
    Return a cached singleton instance of IngestionService.

    Safely reuses the embedding service and vector store client
    across HTTP requests without re-instantiating them each time.
    """
    from app.services.ingestion import get_ingestion_service

    settings = get_settings()
    return get_ingestion_service(settings=settings)


def reset_ingestion_service_cache() -> None:
    """Clear the cached shared IngestionService instance."""
    get_shared_ingestion_service.cache_clear()


def get_ingestion_service_dependency():
    """
    FastAPI dependency that returns an IngestionService.

    Designed for injection in API routes and override in tests via
    app.dependency_overrides[get_ingestion_service_dependency].
    """
    return get_shared_ingestion_service()

