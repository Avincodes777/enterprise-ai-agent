"""Embedding service layer for generating text, document, and query vectors."""

from abc import ABC, abstractmethod
import hashlib
import logging
import math
import threading
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""
    pass


class EmbeddingInputError(EmbeddingError):
    """Raised when input text or query is invalid (empty, None, or whitespace-only)."""
    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when the generated vector dimension does not match the configured dimension."""
    pass


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class EmbeddingResult(BaseModel):
    """Data model representing embedding outputs and provider metadata."""

    embeddings: List[List[float]] = Field(..., description="List of float vectors")
    model: str = Field(..., description="Model identifier used to produce vectors")
    dimension: int = Field(..., description="Dimensionality of the output vectors")
    total_tokens: Optional[int] = Field(
        default=None, description="Total tokens consumed if supported by the provider"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional provider-specific metadata"
    )


# -----------------------------------------------------------------------------
# Abstract Base Interface
# -----------------------------------------------------------------------------
class BaseEmbeddingService(ABC):
    """Abstract interface defining the embedding service contract."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by the embedding model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a dense vector.

        Args:
            text: Text content to embed.

        Returns:
            A list of floats representing the vector embedding.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a collection of document texts in batches.

        Args:
            texts: Sequence of document strings.

        Returns:
            A list of float vectors, one per input string.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user search query into a dense vector.

        Args:
            query: Query text to embed.

        Returns:
            A list of floats representing the query vector embedding.
        """
        pass


# -----------------------------------------------------------------------------
# FastEmbed Local Provider
# -----------------------------------------------------------------------------
class FastEmbedEmbeddingService(BaseEmbeddingService):
    """Local, provider-independent embedding service powered by FastEmbed ONNX runtime."""

    _models_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        expected_dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()
        self._model_name = model_name or current_settings.embedding_model
        self._dimension = expected_dimension or current_settings.embedding_dimension
        self._batch_size = batch_size or current_settings.embedding_batch_size
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Optional[Any] = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _get_or_load_model(self) -> Any:
        """Lazily load and cache the FastEmbed TextEmbedding model instance."""
        if self._model is not None:
            return self._model

        cache_key = f"{self._model_name}_{self._cache_dir}_{self._threads}"
        with self._cache_lock:
            if cache_key in self._models_cache:
                self._model = self._models_cache[cache_key]
                return self._model

            try:
                from fastembed import TextEmbedding  # type: ignore

                logger.info(f"Loading FastEmbed model '{self._model_name}'...")
                model_kwargs: Dict[str, Any] = {"model_name": self._model_name}
                if self._cache_dir:
                    model_kwargs["cache_dir"] = self._cache_dir
                if self._threads:
                    model_kwargs["threads"] = self._threads

                model = TextEmbedding(**model_kwargs)
                self._models_cache[cache_key] = model
                self._model = model
                return self._model
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load FastEmbed model '{self._model_name}': {exc}"
                ) from exc

    def _validate_single_text(self, text: Any, param_name: str = "text") -> str:
        if text is None or not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError(
                f"{param_name} cannot be None, empty, or whitespace-only"
            )
        return text.strip()

    def _validate_vector_dimension(self, vector: List[float], context: str = "output") -> List[float]:
        actual_dim = len(vector)
        if actual_dim != self._dimension:
            raise EmbeddingDimensionMismatchError(
                f"Generated vector dimension ({actual_dim}) does not match "
                f"expected dimension ({self._dimension}) for {context}"
            )
        return vector

    def embed_text(self, text: str) -> List[float]:
        clean_text = self._validate_single_text(text, param_name="text")
        model = self._get_or_load_model()
        try:
            embeddings_gen = model.embed([clean_text], batch_size=1)
            raw_vector = next(iter(embeddings_gen))
            vector = [float(x) for x in raw_vector]
            return self._validate_vector_dimension(vector, context="embed_text")
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate embedding for text using FastEmbed: {exc}"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts: List[str] = []
        for idx, t in enumerate(texts):
            clean_t = self._validate_single_text(t, param_name=f"texts[{idx}]")
            cleaned_texts.append(clean_t)

        model = self._get_or_load_model()
        try:
            # Use passage_embed for document chunks
            embeddings_gen = model.passage_embed(
                cleaned_texts, batch_size=self._batch_size
            )
            results: List[List[float]] = []
            for raw_vector in embeddings_gen:
                vector = [float(x) for x in raw_vector]
                self._validate_vector_dimension(vector, context="embed_documents")
                results.append(vector)
            return results
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate document embeddings using FastEmbed: {exc}"
            ) from exc

    def embed_query(self, query: str) -> List[float]:
        clean_query = self._validate_single_text(query, param_name="query")
        model = self._get_or_load_model()
        try:
            # Use query_embed for search queries
            embeddings_gen = model.query_embed([clean_query], batch_size=1)
            raw_vector = next(iter(embeddings_gen))
            vector = [float(x) for x in raw_vector]
            return self._validate_vector_dimension(vector, context="embed_query")
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate query embedding using FastEmbed: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Mock Provider (for deterministic, zero-network unit testing)
# -----------------------------------------------------------------------------
class MockEmbeddingService(BaseEmbeddingService):
    """Deterministic, zero-network mock embedding service for unit tests."""

    def __init__(
        self,
        model_name: str = "mock-embedding-model",
        dimension: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()
        self._model_name = model_name
        self._dimension = dimension or current_settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _validate_text(self, text: Any, param_name: str = "text") -> str:
        if text is None or not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError(
                f"{param_name} cannot be None, empty, or whitespace-only"
            )
        return text.strip()

    def _compute_deterministic_vector(self, text: str) -> List[float]:
        """Compute a deterministic, L2-normalized float vector of exact dimension from text."""
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vector: List[float] = []
        counter = 0

        while len(vector) < self._dimension:
            block = hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(block), 4):
                if len(vector) >= self._dimension:
                    break
                val = int.from_bytes(block[i : i + 4], "big", signed=True)
                vector.append(val / 2147483648.0)
            counter += 1

        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [float(x / norm) for x in vector]

    def embed_text(self, text: str) -> List[float]:
        clean_text = self._validate_text(text, param_name="text")
        return self._compute_deterministic_vector(clean_text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for idx, t in enumerate(texts):
            clean_t = self._validate_text(t, param_name=f"texts[{idx}]")
            results.append(self._compute_deterministic_vector(clean_t))
        return results

    def embed_query(self, query: str) -> List[float]:
        clean_query = self._validate_text(query, param_name="query")
        return self._compute_deterministic_vector(clean_query)


# -----------------------------------------------------------------------------
# Factory / Helper Function
# -----------------------------------------------------------------------------
def get_embedding_service(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
) -> BaseEmbeddingService:
    """
    Factory function to retrieve the configured embedding service.

    Args:
        settings: Optional Settings instance (uses get_settings() by default).
        use_mock: If True, returns a MockEmbeddingService regardless of settings.

    Returns:
        An instance of BaseEmbeddingService.
    """
    current_settings = settings or get_settings()

    if use_mock:
        return MockEmbeddingService(
            model_name="mock-embedding-model",
            dimension=current_settings.embedding_dimension,
            settings=current_settings,
        )

    provider = current_settings.embedding_provider.lower()
    if provider == "fastembed":
        return FastEmbedEmbeddingService(
            model_name=current_settings.embedding_model,
            expected_dimension=current_settings.embedding_dimension,
            batch_size=current_settings.embedding_batch_size,
            settings=current_settings,
        )
    elif provider == "openai":
        raise NotImplementedError(
            "OpenAI embedding provider will be implemented as an optional provider in a future milestone."
        )
    else:
        raise EmbeddingError(f"Unsupported embedding provider: '{provider}'")
