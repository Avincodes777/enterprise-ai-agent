"""Document ingestion orchestration service for processing, embedding, and vector storage."""

import logging
from typing import Callable, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services.chunker import DocumentChunk, TextChunker
from app.services.embedding import BaseEmbeddingService, get_embedding_service
from app.services.pdf_loader import PDFDocument, PDFLoader, PDFProcessingError
from app.services.text_cleaner import TextCleaner
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorPoint,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class IngestionError(Exception):
    """Base exception raised for document ingestion failures."""
    pass


class IngestionValidationError(IngestionError):
    """Raised when input parameters fail validation."""
    pass


class IngestionEmptyDocumentError(IngestionError):
    """Raised when a document contains no extractable text or produces no chunks."""
    pass


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class IngestionResult(BaseModel):
    """Structured container detailing the outcome of document ingestion."""

    document_name: str = Field(..., description="Name or identifier of the ingested document")
    pages_processed: int = Field(..., ge=0, description="Number of pages parsed from the document")
    chunks_created: int = Field(..., ge=0, description="Total number of text chunks created")
    chunks_stored: int = Field(
        ..., ge=0, description="Total number of vector points upserted into the vector store"
    )
    collection_name: str = Field(
        ..., description="Target vector collection where chunks were stored"
    )
    chunk_ids: List[str] = Field(
        default_factory=list, description="Ordered list of deterministic chunk identifiers"
    )


# -----------------------------------------------------------------------------
# Ingestion Service
# -----------------------------------------------------------------------------
class IngestionService:
    """Orchestrates end-to-end PDF ingestion into chunked, embedded vector storage."""

    def __init__(
        self,
        pdf_loader: Optional[PDFLoader] = None,
        text_cleaner: Optional[TextCleaner] = None,
        chunker: Optional[TextChunker] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        vector_store_service: Optional[BaseVectorStoreService] = None,
        default_collection_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        current_settings = settings or get_settings()

        self._pdf_loader = pdf_loader or PDFLoader()
        self._text_cleaner = text_cleaner or TextCleaner()
        self._chunker = chunker or TextChunker()
        self._embedding_service = embedding_service or get_embedding_service(
            settings=current_settings
        )
        self._vector_store_service = vector_store_service or get_vector_store_service(
            settings=current_settings
        )
        self._default_collection_name = (
            default_collection_name
            or current_settings.qdrant_collection_name
            or "enterprise_knowledge"
        )
        self._settings = current_settings

    @property
    def pdf_loader(self) -> PDFLoader:
        return self._pdf_loader

    @property
    def text_cleaner(self) -> TextCleaner:
        return self._text_cleaner

    @property
    def chunker(self) -> TextChunker:
        return self._chunker

    @property
    def embedding_service(self) -> BaseEmbeddingService:
        return self._embedding_service

    @property
    def vector_store_service(self) -> BaseVectorStoreService:
        return self._vector_store_service

    @property
    def default_collection_name(self) -> str:
        return self._default_collection_name

    def ingest_pdf(
        self,
        file_bytes: bytes,
        document_name: str,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
        heartbeat_callback: Optional[Callable[[], None]] = None,
    ) -> IngestionResult:
        """
        Ingest a PDF from in-memory bytes, clean, chunk, embed, and upsert to vector store.

        Args:
            file_bytes: Raw bytes of the PDF file.
            document_name: Human-readable name or identifier for the document.
            collection_name: Optional target vector collection name override.
            document_id: Optional persistent unique document identifier. If omitted,
                         a unique fallback identity is generated to prevent point collisions.
            heartbeat_callback: Optional callback invoked before long-running stages to renew
                                document activity timestamp and verify ongoing processing state.

        Returns:
            IngestionResult summarizing the ingestion execution.

        Raises:
            IngestionValidationError: If input bytes, document name, document ID, or callback are invalid.
            IngestionEmptyDocumentError: If the document contains no extractable text.
            IngestionError: If PDF loading, embedding, or vector storage fails.
        """
        # Step 1: Input Validation
        if file_bytes is None or not isinstance(file_bytes, (bytes, bytearray)):
            raise IngestionValidationError("file_bytes must be a non-empty bytes object")
        if len(file_bytes) == 0:
            raise IngestionValidationError("file_bytes cannot be empty")

        if (
            document_name is None
            or not isinstance(document_name, str)
            or not document_name.strip()
        ):
            raise IngestionValidationError("document_name cannot be empty or whitespace-only")
        clean_doc_name = document_name.strip()

        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise IngestionValidationError("document_id cannot be empty or whitespace-only")
            clean_doc_id = document_id.strip()
        else:
            clean_doc_id = f"doc_{uuid.uuid4().hex[:16]}"

        if collection_name is not None:
            if not isinstance(collection_name, str) or not collection_name.strip():
                raise IngestionValidationError("collection_name cannot be empty or whitespace-only")
            target_collection = collection_name.strip()
        else:
            target_collection = self._default_collection_name

        if heartbeat_callback is not None and not callable(heartbeat_callback):
            raise IngestionValidationError("heartbeat_callback must be callable if provided")

        # Step 2: Load PDF from bytes
        try:
            raw_doc: PDFDocument = self._pdf_loader.load_from_bytes(
                file_bytes, filename=clean_doc_name
            )
        except PDFProcessingError as exc:
            logger.error("PDF loader failed for document '%s': %s", clean_doc_name, exc)
            raise IngestionError(f"Failed to load PDF document '{clean_doc_name}': {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error loading PDF '%s': %s", clean_doc_name, exc)
            raise IngestionError(f"Failed to load PDF document '{clean_doc_name}': {exc}") from exc

        if not raw_doc.pages or raw_doc.total_pages == 0:
            raise IngestionEmptyDocumentError(f"PDF document '{clean_doc_name}' contains no pages")

        # Step 3: Clean extracted text
        try:
            cleaned_doc: PDFDocument = self._text_cleaner.clean_document(raw_doc)
        except Exception as exc:
            logger.error("Text cleaning failed for document '%s': %s", clean_doc_name, exc)
            raise IngestionError(f"Text cleaning failed: {exc}") from exc

        if not cleaned_doc.full_text.strip():
            raise IngestionEmptyDocumentError(
                f"PDF document '{clean_doc_name}' contains no extractable text"
            )

        # Step 4: Chunk document
        try:
            chunks: List[DocumentChunk] = self._chunker.chunk_document(cleaned_doc)
        except Exception as exc:
            logger.error("Chunking failed for document '%s': %s", clean_doc_name, exc)
            raise IngestionError(f"Chunking failed: {exc}") from exc

        if not chunks:
            raise IngestionEmptyDocumentError(
                f"No chunks could be extracted from PDF document '{clean_doc_name}'"
            )

        # Checkpoint 1: Immediately before document embedding begins
        if heartbeat_callback is not None:
            heartbeat_callback()

        # Step 5: Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self._embedding_service.embed_documents(chunk_texts)
        except Exception as exc:
            logger.error(
                "Embedding generation failed for document '%s': %s", clean_doc_name, exc
            )
            raise IngestionError(f"Embedding generation failed: {exc}") from exc

        # Step 6: Verify embedding count matches chunk count
        if len(embeddings) != len(chunks):
            raise IngestionError(
                f"Embedding count mismatch for '{clean_doc_name}': "
                f"expected {len(chunks)} embeddings, got {len(embeddings)}"
            )

        # Step 7: Construct VectorPoint objects with document identity provenance
        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            scoped_chunk_id = f"{clean_doc_id}_{chunk.chunk_id}"
            chunk_metadata = dict(chunk.metadata)
            chunk_metadata["document_id"] = clean_doc_id
            points.append(
                VectorPoint(
                    chunk_id=scoped_chunk_id,
                    document_id=clean_doc_id,
                    vector=vec,
                    document_name=chunk.metadata.get("document_name", clean_doc_name),
                    page_number=chunk.page_number,
                    chunk_index=chunk.metadata.get("chunk_index", idx),
                    text=chunk.text,
                    metadata=chunk_metadata,
                )
            )

        # Checkpoint 2: Immediately after embedding completes and before vector-store collection setup or upsert proceeds
        if heartbeat_callback is not None:
            heartbeat_callback()

        # Step 8: Verify / create target collection if missing
        try:
            if not self._vector_store_service.collection_exists(target_collection):
                dim = getattr(self._embedding_service, "dimension", None) or self._settings.embedding_dimension
                self._vector_store_service.create_collection(
                    collection_name=target_collection,
                    dimension=dim,
                )
        except Exception as exc:
            logger.error(
                "Failed to verify or create collection '%s': %s", target_collection, exc
            )
            raise IngestionError(f"Vector store collection setup failed: {exc}") from exc

        # Step 9: Upsert vector points
        try:
            stored_count = self._vector_store_service.upsert_vectors(
                points=points,
                collection_name=target_collection,
            )
        except Exception as exc:
            logger.error(
                "Failed to upsert vector points for document '%s': %s", clean_doc_name, exc
            )
            raise IngestionError(f"Vector store upsert failed: {exc}") from exc

        # Step 10: Return structured IngestionResult
        return IngestionResult(
            document_name=clean_doc_name,
            pages_processed=raw_doc.total_pages,
            chunks_created=len(chunks),
            chunks_stored=stored_count,
            collection_name=target_collection,
            chunk_ids=[p.chunk_id for p in points],
        )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def get_ingestion_service(
    settings: Optional[Settings] = None,
    pdf_loader: Optional[PDFLoader] = None,
    text_cleaner: Optional[TextCleaner] = None,
    chunker: Optional[TextChunker] = None,
    embedding_service: Optional[BaseEmbeddingService] = None,
    vector_store_service: Optional[BaseVectorStoreService] = None,
    use_mock: bool = False,
) -> IngestionService:
    """
    Factory function to retrieve a configured IngestionService instance.

    Args:
        settings: Optional Settings instance.
        pdf_loader: Optional custom PDFLoader.
        text_cleaner: Optional custom TextCleaner.
        chunker: Optional custom TextChunker.
        embedding_service: Optional custom BaseEmbeddingService.
        vector_store_service: Optional custom BaseVectorStoreService.
        use_mock: If True, uses mock embedding and vector store services.

    Returns:
        Configured IngestionService.
    """
    current_settings = settings or get_settings()
    active_embedding = embedding_service or get_embedding_service(
        settings=current_settings, use_mock=use_mock
    )
    active_vector_store = vector_store_service or get_vector_store_service(
        settings=current_settings, use_mock=use_mock
    )
    return IngestionService(
        pdf_loader=pdf_loader,
        text_cleaner=text_cleaner,
        chunker=chunker,
        embedding_service=active_embedding,
        vector_store_service=active_vector_store,
        default_collection_name=current_settings.qdrant_collection_name,
        settings=current_settings,
    )
