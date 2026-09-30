"""Services package for the Enterprise AI Agent."""

from .chunker import (
    DocumentChunk,
    TextChunker,
    chunk_document,
    chunk_page,
    chunk_text,
)
from .embedding import (
    BaseEmbeddingService,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingResult,
    FastEmbedEmbeddingService,
    MockEmbeddingService,
    get_embedding_service,
)
from .pdf_loader import (
    PDFDocument,
    PDFEncryptedError,
    PDFLoader,
    PDFPage,
    PDFProcessingError,
    extract_pdf_text,
    load_pdf,
    load_pdf_from_bytes,
)
from .text_cleaner import (
    TextCleaner,
    clean_document,
    clean_page,
    clean_text,
)
from .vector_store import (
    BaseVectorStoreService,
    MockVectorStoreService,
    QdrantVectorStoreService,
    VectorPoint,
    VectorSearchResult,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreValidationError,
    chunk_id_to_point_id,
    get_vector_store_service,
)

__all__ = [
    # PDF Loader
    "PDFLoader",
    "PDFPage",
    "PDFDocument",
    "PDFEncryptedError",
    "PDFProcessingError",
    "load_pdf",
    "load_pdf_from_bytes",
    "extract_pdf_text",
    # Text Cleaner
    "TextCleaner",
    "clean_text",
    "clean_page",
    "clean_document",
    # Chunker
    "DocumentChunk",
    "TextChunker",
    "chunk_text",
    "chunk_page",
    "chunk_document",
    # Embedding
    "BaseEmbeddingService",
    "FastEmbedEmbeddingService",
    "MockEmbeddingService",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingDimensionMismatchError",
    "get_embedding_service",
    # Vector Store
    "BaseVectorStoreService",
    "QdrantVectorStoreService",
    "MockVectorStoreService",
    "VectorPoint",
    "VectorSearchResult",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreValidationError",
    "VectorStoreCollectionNotFoundError",
    "chunk_id_to_point_id",
    "get_vector_store_service",
]

