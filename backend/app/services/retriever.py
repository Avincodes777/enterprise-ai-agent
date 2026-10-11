"""Retriever service layer for orchestrating query embedding and vector search."""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional

from app.config import Settings, get_settings
from app.services.embedding import BaseEmbeddingService, EmbeddingError, get_embedding_service
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorSearchResult,
    VectorStoreError,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class RetrieverError(Exception):
    """Base exception for all retrieval operations."""
    pass


class RetrieverValidationError(RetrieverError):
    """Raised when query text, top_k, or score_threshold fail validation."""
    pass


# -----------------------------------------------------------------------------
# Abstract Base Interface
# -----------------------------------------------------------------------------
class BaseRetriever(ABC):
    """Abstract interface defining the document retrieval contract."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Retrieve relevant context chunks for a given query text.

        Args:
            query: User search query string.
            top_k: Optional maximum number of results (defaults to configured default_top_k).
            score_threshold: Optional similarity score threshold.
            collection_name: Optional vector collection name.

        Returns:
            List of VectorSearchResult items ordered by similarity.
        """
        pass


# -----------------------------------------------------------------------------
# Dense Vector Retriever Implementation
# -----------------------------------------------------------------------------
class VectorRetriever(BaseRetriever):
    """Provider-independent dense vector retriever."""

    MAX_QUERY_LENGTH: int = 4000

    def __init__(
        self,
        embedding_service: Optional[BaseEmbeddingService] = None,
        vector_store_service: Optional[BaseVectorStoreService] = None,
        default_top_k: int = 5,
        default_score_threshold: Optional[float] = None,
        default_collection_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()

        if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k <= 0:
            raise RetrieverValidationError(
                f"default_top_k must be a positive integer, got {default_top_k}"
            )

        if default_score_threshold is not None:
            self._validate_score_threshold(default_score_threshold)

        self._embedding_service = embedding_service or get_embedding_service(settings=current_settings)
        self._vector_store_service = vector_store_service or get_vector_store_service(settings=current_settings)
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold
        self._default_collection_name = default_collection_name or current_settings.qdrant_collection_name

    @property
    def embedding_service(self) -> BaseEmbeddingService:
        return self._embedding_service

    @property
    def vector_store_service(self) -> BaseVectorStoreService:
        return self._vector_store_service

    @property
    def default_top_k(self) -> int:
        return self._default_top_k

    @property
    def default_score_threshold(self) -> Optional[float]:
        return self._default_score_threshold

    @property
    def default_collection_name(self) -> Optional[str]:
        return self._default_collection_name

    def _validate_query(self, query: str) -> str:
        """Validate query type, presence, whitespace, and maximum length."""
        if query is None or not isinstance(query, str):
            raise RetrieverValidationError(
                f"Query must be a non-empty string, got {type(query).__name__ if query is not None else 'None'}"
            )
        cleaned = query.strip()
        if not cleaned:
            raise RetrieverValidationError("Query cannot be empty or whitespace-only")
        if len(cleaned) > self.MAX_QUERY_LENGTH:
            raise RetrieverValidationError(
                f"Query length ({len(cleaned)}) exceeds maximum allowed limit of {self.MAX_QUERY_LENGTH} characters"
            )
        return cleaned

    def _validate_top_k(self, top_k: Optional[int]) -> int:
        """Validate and resolve top_k parameter."""
        if top_k is None:
            return self._default_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrieverValidationError(
                f"top_k must be an integer, got {type(top_k).__name__}"
            )
        if top_k <= 0:
            raise RetrieverValidationError(
                f"top_k must be greater than 0, got {top_k}"
            )
        return top_k

    def _validate_score_threshold(self, threshold: Optional[float]) -> Optional[float]:
        """Validate score_threshold bounds for cosine similarity ([0.0, 1.0])."""
        if threshold is None:
            return self._default_score_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise RetrieverValidationError(
                f"score_threshold must be a float or integer, got {type(threshold).__name__}"
            )
        val = float(threshold)
        # Note: The [0.0, 1.0] range reflects the cosine similarity retrieval contract.
        if val < 0.0 or val > 1.0:
            raise RetrieverValidationError(
                f"score_threshold must be between 0.0 and 1.0 for cosine similarity, got {threshold}"
            )
        return val

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Execute dense vector retrieval for the input query string.

        Args:
            query: Raw user query text.
            top_k: Number of search results to retrieve (defaults to default_top_k).
            score_threshold: Minimum similarity threshold (defaults to default_score_threshold).
            collection_name: Target vector collection (defaults to default_collection_name).

        Returns:
            List of VectorSearchResult items ordered by similarity.
        """
        cleaned_query = self._validate_query(query)
        target_top_k = self._validate_top_k(top_k)
        target_score_threshold = self._validate_score_threshold(score_threshold)
        target_collection = collection_name or self._default_collection_name

        try:
            query_vector = self._embedding_service.embed_query(cleaned_query)
        except EmbeddingError as exc:
            logger.error(f"Failed to generate query embedding: {exc}")
            raise RetrieverError(f"Retrieval failed during query embedding: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error generating query embedding: {exc}")
            raise RetrieverError(f"Retrieval failed during query embedding: {exc}") from exc

        try:
            results = self._vector_store_service.search(
                query_vector=query_vector,
                top_k=target_top_k,
                score_threshold=target_score_threshold,
                collection_name=target_collection,
            )
        except VectorStoreError as exc:
            logger.error(f"Failed to perform vector store search: {exc}")
            raise RetrieverError(f"Retrieval failed during vector store search: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error performing vector store search: {exc}")
            raise RetrieverError(f"Retrieval failed during vector store search: {exc}") from exc

        # Defensive top_k safeguard without altering scoring or ranking
        return results[:target_top_k] if results else []


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def get_retriever(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
    default_top_k: int = 5,
    default_score_threshold: Optional[float] = None,
    default_collection_name: Optional[str] = None,
) -> BaseRetriever:
    """
    Factory function to retrieve a configured VectorRetriever.

    Args:
        settings: Optional Settings instance (defaults to get_settings()).
        use_mock: If True, injects MockEmbeddingService and MockVectorStoreService.
        default_top_k: Default top_k results.
        default_score_threshold: Optional default score threshold.
        default_collection_name: Optional default collection name.

    Returns:
        An instance conforming to BaseRetriever.
    """
    current_settings = settings or get_settings()
    embedding_service = get_embedding_service(settings=current_settings, use_mock=use_mock)
    vector_store_service = get_vector_store_service(settings=current_settings, use_mock=use_mock)

    return VectorRetriever(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        default_top_k=default_top_k,
        default_score_threshold=default_score_threshold,
        default_collection_name=default_collection_name,
        settings=current_settings,
    )
