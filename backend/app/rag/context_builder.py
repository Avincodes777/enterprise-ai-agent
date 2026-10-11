"""RAG Context Builder service for assembling deterministic structured prompt context from retrieved knowledge."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import Settings
from app.services.retriever import BaseRetriever, get_retriever
from app.services.vector_store import VectorSearchResult

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class RAGContextError(Exception):
    """Base exception for errors encountered during RAG context assembly."""
    pass


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class RAGContext(BaseModel):
    """Structured container holding retrieval results and formatted prompt context."""

    query: str = Field(..., description="Original user search query")
    results: List[VectorSearchResult] = Field(
        default_factory=list, description="Ordered list of retrieved VectorSearchResult chunks"
    )
    context_text: str = Field(
        default="", description="Deterministic formatted plain text representation of retrieved sources"
    )

    @property
    def has_results(self) -> bool:
        """Derived property indicating whether retrieval returned any results."""
        return len(self.results) > 0

    @property
    def total_results(self) -> int:
        """Total number of retrieved chunks."""
        return len(self.results)


# -----------------------------------------------------------------------------
# Context Builder Implementation
# -----------------------------------------------------------------------------
class RAGContextBuilder:
    """Provider-independent RAG context builder formatting retrieved knowledge into structured context."""

    def __init__(
        self,
        retriever: BaseRetriever,
        default_top_k: int = 5,
        default_score_threshold: Optional[float] = None,
        default_collection_name: Optional[str] = None,
    ):
        if not isinstance(retriever, BaseRetriever):
            raise TypeError(f"retriever must be an instance of BaseRetriever, got {type(retriever).__name__}")

        if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k <= 0:
            raise ValueError(f"default_top_k must be a positive integer, got {default_top_k}")

        if default_score_threshold is not None:
            if isinstance(default_score_threshold, bool) or not isinstance(default_score_threshold, (int, float)) or not (0.0 <= float(default_score_threshold) <= 1.0):
                raise ValueError(
                    f"default_score_threshold must be a float between 0.0 and 1.0, got {default_score_threshold}"
                )

        self._retriever = retriever
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold
        self._default_collection_name = default_collection_name

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def default_top_k(self) -> int:
        return self._default_top_k

    @property
    def default_score_threshold(self) -> Optional[float]:
        return self._default_score_threshold

    @property
    def default_collection_name(self) -> Optional[str]:
        return self._default_collection_name

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata dictionary into deterministic key-sorted string lines."""
        if not metadata or not isinstance(metadata, dict):
            return ""

        lines: List[str] = []
        for key in sorted(metadata.keys()):
            val = metadata[key]
            lines.append(f"{key}={val}")
        return "\n".join(lines)

    def _format_context_text(self, results: List[VectorSearchResult]) -> str:
        """Format list of VectorSearchResult objects into deterministic plain text."""
        if not results:
            return ""

        blocks: List[str] = []
        for idx, item in enumerate(results, start=1):
            block_lines = [f"[SOURCE {idx}]"]
            if item.document_name:
                block_lines.append(f"Document: {item.document_name}")
            block_lines.append(f"Page: {item.page_number}")
            block_lines.append(f"Chunk: {item.chunk_index}")
            block_lines.append(f"Score: {item.score:.3f}" if isinstance(item.score, float) else f"Score: {item.score}")

            # Filter out standard positional fields already in header
            filtered_meta = {
                k: v
                for k, v in item.metadata.items()
                if k not in {"document_name", "page_number", "chunk_index"}
            } if item.metadata else {}

            meta_str = self._format_metadata(filtered_meta)
            if meta_str:
                block_lines.append(f"Metadata:\n{meta_str}")

            block_lines.append("Content:")
            block_lines.append(item.text)

            blocks.append("\n".join(block_lines))

        return "\n\n".join(blocks)

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> RAGContext:
        """
        Retrieve relevant knowledge and assemble structured RAGContext.

        Args:
            query: User search query string.
            top_k: Optional maximum number of results (defaults to builder default).
            score_threshold: Optional minimum similarity threshold (defaults to builder default).
            collection_name: Target vector collection (defaults to builder default).

        Returns:
            RAGContext instance with query, retrieved results, and formatted context text.
        """
        target_top_k = top_k if top_k is not None else self._default_top_k
        target_threshold = score_threshold if score_threshold is not None else self._default_score_threshold
        target_collection = collection_name if collection_name is not None else self._default_collection_name

        try:
            results = self._retriever.retrieve(
                query=query,
                top_k=target_top_k,
                score_threshold=target_threshold,
                collection_name=target_collection,
            )
        except Exception as exc:
            logger.error(f"Failed to retrieve documents for RAG context: {exc}")
            raise RAGContextError(f"Failed to build RAG context: {exc}") from exc

        formatted_text = self._format_context_text(results)

        return RAGContext(
            query=query,
            results=results,
            context_text=formatted_text,
        )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def get_rag_context_builder(
    retriever: Optional[BaseRetriever] = None,
    settings: Optional[Settings] = None,
    use_mock: bool = False,
    default_top_k: int = 5,
    default_score_threshold: Optional[float] = None,
    default_collection_name: Optional[str] = None,
) -> RAGContextBuilder:
    """
    Factory function to retrieve a configured RAGContextBuilder instance.

    Args:
        retriever: Optional BaseRetriever instance.
        settings: Optional Settings instance.
        use_mock: If True and retriever not supplied, instantiates a mock-backed retriever.
        default_top_k: Default top_k results.
        default_score_threshold: Optional default similarity threshold.
        default_collection_name: Optional default collection name.

    Returns:
        Configured RAGContextBuilder.
    """
    active_retriever = retriever or get_retriever(
        settings=settings,
        use_mock=use_mock,
        default_top_k=default_top_k,
        default_score_threshold=default_score_threshold,
        default_collection_name=default_collection_name,
    )
    return RAGContextBuilder(
        retriever=active_retriever,
        default_top_k=default_top_k,
        default_score_threshold=default_score_threshold,
        default_collection_name=default_collection_name,
    )
