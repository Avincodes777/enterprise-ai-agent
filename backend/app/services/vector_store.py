"""Vector store service layer for dense vector indexing, storage, and retrieval."""

from abc import ABC, abstractmethod
import logging
import math
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store service is unreachable or connection fails."""
    pass


class VectorStoreValidationError(VectorStoreError):
    """Raised when input parameters, vectors, or schemas fail validation."""
    pass


class VectorStoreCollectionNotFoundError(VectorStoreError):
    """Raised when a requested vector store collection does not exist."""
    pass


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Generate a deterministic UUIDv5 string from a chunk identifier.

    Args:
        chunk_id: Unique string identifier for the chunk.

    Returns:
        String representation of the deterministic UUID.
    """
    if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
        raise VectorStoreValidationError("chunk_id must be a non-empty string")
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id.strip()))


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class VectorPoint(BaseModel):
    """Represents a vector embedding with associated document provenance and metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    vector: List[float] = Field(..., description="Dense float vector embedding")
    document_name: str = Field(default="", description="Source document file name")
    page_number: int = Field(default=1, description="1-indexed source page number")
    chunk_index: int = Field(default=0, description="0-indexed chunk index within page/doc")
    text: str = Field(default="", description="Cleaned chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata dictionary"
    )


class VectorSearchResult(BaseModel):
    """Represents a scored vector search result with source provenance."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_name: str = Field(default="", description="Source document file name")
    page_number: int = Field(default=1, description="1-indexed source page number")
    chunk_index: int = Field(default=0, description="0-indexed chunk index within page/doc")
    text: str = Field(default="", description="Chunk text content")
    score: float = Field(..., description="Similarity score (cosine)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Associated provenance and chunk metadata"
    )


# -----------------------------------------------------------------------------
# Abstract Base Interface
# -----------------------------------------------------------------------------
class BaseVectorStoreService(ABC):
    """Abstract interface defining provider-independent vector store capabilities."""

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store service is reachable and responsive.

        Returns:
            True if healthy, False otherwise.
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Verify whether the specified collection exists.

        Args:
            collection_name: Optional collection name (defaults to configured collection).

        Returns:
            True if collection exists, False otherwise.
        """
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        recreate_if_exists: bool = False,
    ) -> bool:
        """
        Create a vector collection configured for cosine similarity.

        Args:
            collection_name: Optional collection name (defaults to configured collection).
            dimension: Vector dimensionality (defaults to configured embedding dimension).
            recreate_if_exists: If True, existing collection will be deleted and recreated.

        Returns:
            True if collection was created or already exists.
        """
        pass

    @abstractmethod
    def upsert_vectors(
        self,
        points: List[VectorPoint],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Upsert a batch of vector points into the collection.

        Args:
            points: List of VectorPoint instances to upsert.
            collection_name: Optional collection name.

        Returns:
            Number of points successfully upserted.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Perform a cosine similarity search against the vector store.

        Args:
            query_vector: Query float vector embedding.
            top_k: Maximum number of search results to return.
            score_threshold: Optional minimum cosine similarity score threshold.
            collection_name: Optional collection name.

        Returns:
            List of VectorSearchResult items ordered by similarity score descending.
        """
        pass

    @abstractmethod
    def count(self, collection_name: Optional[str] = None) -> int:
        """
        Return the total number of points stored in the collection.

        Args:
            collection_name: Optional collection name.

        Returns:
            Total point count.
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection and its associated vectors.

        Args:
            collection_name: Optional collection name.

        Returns:
            True if collection was deleted, False if it did not exist.
        """
        pass


# -----------------------------------------------------------------------------
# Qdrant Concrete Service
# -----------------------------------------------------------------------------
class QdrantVectorStoreService(BaseVectorStoreService):
    """Concrete vector store service backed by Qdrant."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        dimension: Optional[int] = None,
        client: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()
        self._url = url or current_settings.qdrant_url or "http://localhost:6333"
        self._collection_name = (
            collection_name
            or current_settings.qdrant_collection_name
            or "enterprise_knowledge"
        )
        self._api_key = api_key if api_key is not None else current_settings.qdrant_api_key
        self._timeout = timeout or current_settings.qdrant_timeout_seconds
        self._dimension = dimension or current_settings.embedding_dimension

        if client is not None:
            self._client = client
        else:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=self._timeout,
            )

    @property
    def url(self) -> str:
        return self._url

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _resolve_collection_name(self, name: Optional[str]) -> str:
        target = name or self._collection_name
        if not target or not isinstance(target, str) or not target.strip():
            raise VectorStoreValidationError("Collection name cannot be empty")
        return target.strip()

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.debug(f"Qdrant health check failed: {exc}")
            return False

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        col = self._resolve_collection_name(collection_name)
        try:
            return bool(self._client.collection_exists(collection_name=col))
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to check collection existence for '{col}': {exc}"
            ) from exc

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        recreate_if_exists: bool = False,
    ) -> bool:
        col = self._resolve_collection_name(collection_name)
        dim = self._dimension if dimension is None else dimension
        if dim <= 0:
            raise VectorStoreValidationError(
                f"Collection vector dimension must be greater than 0, got {dim}"
            )

        from qdrant_client import models

        try:
            exists = self.collection_exists(col)
            if exists:
                if recreate_if_exists:
                    self._client.delete_collection(collection_name=col)
                else:
                    return True

            self._client.create_collection(
                collection_name=col,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            return True
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create collection '{col}': {exc}"
            ) from exc

    def upsert_vectors(
        self,
        points: List[VectorPoint],
        collection_name: Optional[str] = None,
    ) -> int:
        if not points:
            raise VectorStoreValidationError("Cannot upsert an empty list of vector points")

        col = self._resolve_collection_name(collection_name)

        from qdrant_client import models

        qdrant_points: List[models.PointStruct] = []
        for idx, pt in enumerate(points):
            if not isinstance(pt, VectorPoint):
                raise VectorStoreValidationError(
                    f"Expected VectorPoint instance at index {idx}, got {type(pt).__name__}"
                )
            if not pt.chunk_id or not pt.chunk_id.strip():
                raise VectorStoreValidationError(
                    f"VectorPoint at index {idx} has missing or empty chunk_id"
                )
            if len(pt.vector) != self._dimension:
                raise VectorStoreValidationError(
                    f"VectorPoint at index {idx} dimension ({len(pt.vector)}) "
                    f"does not match configured dimension ({self._dimension})"
                )

            point_id = chunk_id_to_point_id(pt.chunk_id)
            payload = {
                "chunk_id": pt.chunk_id,
                "document_name": pt.document_name,
                "page_number": pt.page_number,
                "chunk_index": pt.chunk_index,
                "text": pt.text,
                "metadata": pt.metadata,
            }

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector=pt.vector,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=col,
                points=qdrant_points,
            )
            return len(qdrant_points)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert {len(points)} points into collection '{col}': {exc}"
            ) from exc

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not query_vector:
            raise VectorStoreValidationError("Query vector cannot be empty")
        if len(query_vector) != self._dimension:
            raise VectorStoreValidationError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"configured dimension ({self._dimension})"
            )
        if top_k <= 0:
            raise VectorStoreValidationError(f"top_k must be greater than 0, got {top_k}")

        col = self._resolve_collection_name(collection_name)

        try:
            query_response = self._client.query_points(
                collection_name=col,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            scored_points = query_response.points if hasattr(query_response, "points") else query_response

            results: List[VectorSearchResult] = []
            for pt in scored_points:
                raw_payload = getattr(pt, "payload", None) or {}
                chunk_id = str(raw_payload.get("chunk_id") or pt.id)
                document_name = str(raw_payload.get("document_name") or "")
                page_number = int(raw_payload.get("page_number", 1))
                chunk_index = int(raw_payload.get("chunk_index", 0))
                text = str(raw_payload.get("text") or "")
                metadata = raw_payload.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}

                score_val = float(getattr(pt, "score", 0.0))

                results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=text,
                        score=score_val,
                        metadata=metadata,
                    )
                )
            return results
        except VectorStoreValidationError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to search collection '{col}': {exc}"
            ) from exc

    def count(self, collection_name: Optional[str] = None) -> int:
        col = self._resolve_collection_name(collection_name)
        try:
            if not self.collection_exists(col):
                raise VectorStoreCollectionNotFoundError(
                    f"Collection '{col}' does not exist"
                )
            res = self._client.count(collection_name=col)
            return int(res.count)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve point count for collection '{col}': {exc}"
            ) from exc

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        col = self._resolve_collection_name(collection_name)
        try:
            if not self.collection_exists(col):
                return False
            self._client.delete_collection(collection_name=col)
            return True
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete collection '{col}': {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# In-Memory Mock Vector Store (Zero-Network Testing)
# -----------------------------------------------------------------------------
class MockVectorStoreService(BaseVectorStoreService):
    """Deterministic, zero-network in-memory mock vector store service."""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        dimension: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()
        self._default_collection_name = collection_name or current_settings.qdrant_collection_name
        self._dimension = dimension or current_settings.embedding_dimension
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._is_healthy = True

    def set_healthy(self, healthy: bool) -> None:
        """Utility for simulating network / service outages in unit tests."""
        self._is_healthy = healthy

    def health_check(self) -> bool:
        return self._is_healthy

    def _resolve_collection_name(self, name: Optional[str]) -> str:
        target = name or self._default_collection_name
        if not target or not isinstance(target, str) or not target.strip():
            raise VectorStoreValidationError("Collection name cannot be empty")
        return target.strip()

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        col = self._resolve_collection_name(collection_name)
        return col in self._collections

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        recreate_if_exists: bool = False,
    ) -> bool:
        col = self._resolve_collection_name(collection_name)
        dim = self._dimension if dimension is None else dimension
        if dim <= 0:
            raise VectorStoreValidationError(
                f"Collection vector dimension must be greater than 0, got {dim}"
            )

        if col in self._collections:
            if recreate_if_exists:
                self._collections[col] = {"dimension": dim, "points": {}}
                return True
            return True

        self._collections[col] = {"dimension": dim, "points": {}}
        return True

    def upsert_vectors(
        self,
        points: List[VectorPoint],
        collection_name: Optional[str] = None,
    ) -> int:
        if not points:
            raise VectorStoreValidationError("Cannot upsert an empty list of vector points")

        col = self._resolve_collection_name(collection_name)
        if col not in self._collections:
            self.create_collection(col)

        expected_dim = self._collections[col]["dimension"]

        for idx, pt in enumerate(points):
            if not isinstance(pt, VectorPoint):
                raise VectorStoreValidationError(
                    f"Expected VectorPoint instance at index {idx}, got {type(pt).__name__}"
                )
            if not pt.chunk_id or not pt.chunk_id.strip():
                raise VectorStoreValidationError(
                    f"VectorPoint at index {idx} has missing or empty chunk_id"
                )
            if len(pt.vector) != expected_dim:
                raise VectorStoreValidationError(
                    f"VectorPoint at index {idx} dimension ({len(pt.vector)}) "
                    f"does not match collection dimension ({expected_dim})"
                )

            point_id = chunk_id_to_point_id(pt.chunk_id)
            self._collections[col]["points"][point_id] = {
                "chunk_id": pt.chunk_id,
                "vector": list(pt.vector),
                "document_name": pt.document_name,
                "page_number": pt.page_number,
                "chunk_index": pt.chunk_index,
                "text": pt.text,
                "metadata": dict(pt.metadata or {}),
            }

        return len(points)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not query_vector:
            raise VectorStoreValidationError("Query vector cannot be empty")
        if top_k <= 0:
            raise VectorStoreValidationError(f"top_k must be greater than 0, got {top_k}")

        col = self._resolve_collection_name(collection_name)
        if col not in self._collections:
            raise VectorStoreCollectionNotFoundError(
                f"Collection '{col}' does not exist"
            )

        expected_dim = self._collections[col]["dimension"]
        if len(query_vector) != expected_dim:
            raise VectorStoreValidationError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"collection dimension ({expected_dim})"
            )

        scored_candidates: List[VectorSearchResult] = []
        for _, pt_data in self._collections[col]["points"].items():
            score = self._cosine_similarity(query_vector, pt_data["vector"])
            if score_threshold is not None and score < score_threshold:
                continue

            scored_candidates.append(
                VectorSearchResult(
                    chunk_id=pt_data["chunk_id"],
                    document_name=pt_data["document_name"],
                    page_number=pt_data["page_number"],
                    chunk_index=pt_data["chunk_index"],
                    text=pt_data["text"],
                    score=score,
                    metadata=dict(pt_data["metadata"]),
                )
            )

        scored_candidates.sort(key=lambda r: r.score, reverse=True)
        return scored_candidates[:top_k]

    def count(self, collection_name: Optional[str] = None) -> int:
        col = self._resolve_collection_name(collection_name)
        if col not in self._collections:
            raise VectorStoreCollectionNotFoundError(
                f"Collection '{col}' does not exist"
            )
        return len(self._collections[col]["points"])

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        col = self._resolve_collection_name(collection_name)
        if col in self._collections:
            del self._collections[col]
            return True
        return False


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def get_vector_store_service(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
) -> BaseVectorStoreService:
    """
    Factory function to retrieve the configured Vector Store service.

    Args:
        settings: Optional Settings instance (uses get_settings() by default).
        use_mock: If True, returns an in-memory MockVectorStoreService.

    Returns:
        An instance conforming to BaseVectorStoreService.
    """
    current_settings = settings or get_settings()

    if use_mock:
        return MockVectorStoreService(
            collection_name=current_settings.qdrant_collection_name,
            dimension=current_settings.embedding_dimension,
            settings=current_settings,
        )

    return QdrantVectorStoreService(
        url=current_settings.qdrant_url,
        collection_name=current_settings.qdrant_collection_name,
        api_key=current_settings.qdrant_api_key,
        timeout=current_settings.qdrant_timeout_seconds,
        dimension=current_settings.embedding_dimension,
        settings=current_settings,
    )
