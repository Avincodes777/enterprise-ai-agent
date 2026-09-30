import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .pdf_loader import PDFDocument, PDFPage


class DocumentChunk(BaseModel):
    """Represents a discrete text chunk with source provenance and metadata."""

    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    page_number: int = Field(..., description="1-indexed source page number")
    char_count: int = Field(..., description="Length of the chunk text in characters")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Associated document and provenance metadata"
    )


class TextChunker:
    """Service for deterministic page-aware document chunking."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def _format_doc_prefix(self, doc_id: Optional[str]) -> str:
        """Format document identifier into a clean prefix for deterministic IDs."""
        if not doc_id:
            return "doc"
        filename = Path(doc_id).name
        # Remove extension and normalize special characters
        name_without_ext = Path(filename).stem
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name_without_ext).strip("_").lower()
        return clean or "doc"

    def chunk_text(
        self,
        text: Optional[str],
        page_number: int = 1,
        doc_id: Optional[str] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split a string into overlapping character chunks with deterministic IDs.

        Args:
            text: Text content to chunk.
            page_number: 1-indexed page number.
            doc_id: Document name or identifier.
            base_metadata: Base metadata dictionary to attach to chunks.

        Returns:
            List of DocumentChunk instances.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        text_len = len(text)
        step = self.chunk_size - self.overlap
        doc_prefix = self._format_doc_prefix(doc_id)

        chunks: List[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk_id = f"{doc_prefix}_p{page_number}_c{chunk_idx}"
                meta = dict(base_metadata or {})
                meta.update(
                    {
                        "chunk_index": chunk_idx,
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                if doc_id:
                    meta["document_name"] = Path(doc_id).name

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_content,
                        page_number=page_number,
                        char_count=len(chunk_content),
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break

            start += step

        return chunks

    def chunk_page(
        self,
        page: PDFPage,
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a single PDFPage while preserving its page number and metadata.

        Args:
            page: PDFPage instance to chunk.
            doc_id: Optional document identifier.

        Returns:
            List of DocumentChunk instances.
        """
        target_doc = doc_id or page.metadata.get("source") or "doc"
        return self.chunk_text(
            text=page.content,
            page_number=page.page_number,
            doc_id=target_doc,
            base_metadata=page.metadata,
        )

    def chunk_document(
        self,
        doc: PDFDocument,
    ) -> List[DocumentChunk]:
        """
        Chunk an entire PDFDocument page by page, preserving page traceability.

        Args:
            doc: PDFDocument instance.

        Returns:
            List of DocumentChunk instances for all pages.
        """
        doc_id = doc.filename or "doc"
        all_chunks: List[DocumentChunk] = []

        for page in doc.pages:
            page_meta = dict(doc.doc_metadata)
            page_meta.update(page.metadata)
            page_chunks = self.chunk_text(
                text=page.content,
                page_number=page.page_number,
                doc_id=doc_id,
                base_metadata=page_meta,
            )
            all_chunks.extend(page_chunks)

        return all_chunks


def chunk_text(
    text: Optional[str],
    chunk_size: int = 1000,
    overlap: int = 200,
    page_number: int = 1,
    doc_id: Optional[str] = None,
) -> List[DocumentChunk]:
    """Convenience helper to chunk raw text."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(
        text, page_number=page_number, doc_id=doc_id
    )


def chunk_page(
    page: PDFPage,
    chunk_size: int = 1000,
    overlap: int = 200,
    doc_id: Optional[str] = None,
) -> List[DocumentChunk]:
    """Convenience helper to chunk a single PDFPage."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_page(
        page, doc_id=doc_id
    )


def chunk_document(
    doc: PDFDocument,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """Convenience helper to chunk a PDFDocument."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_document(doc)
