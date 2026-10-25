"""Pydantic schemas for document ingestion and registry endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.ingestion import IngestionResult


class IngestionResponse(IngestionResult):
    """API response model for document ingestion, extending IngestionResult with registry document_id."""

    document_id: str = Field(
        ..., description="Unique persistent identifier for the document in the registry"
    )


class DocumentRecordResponse(BaseModel):
    """API response model for a single registered document record."""

    document_id: str = Field(..., description="Unique persistent identifier for the document")
    document_name: str = Field(..., description="Original filename of the uploaded document")
    status: str = Field(..., description="Ingestion lifecycle status: processing, succeeded, or failed")
    collection_name: str = Field(..., description="Target vector store collection name")
    created_at: str = Field(..., description="UTC ISO-8601 timestamp of registration")
    updated_at: str = Field(..., description="UTC ISO-8601 timestamp of last update")
    pages_processed: Optional[int] = Field(
        default=None, ge=0, description="Number of pages extracted from the PDF"
    )
    chunks_created: Optional[int] = Field(
        default=None, ge=0, description="Total text chunks generated"
    )
    chunks_stored: Optional[int] = Field(
        default=None, ge=0, description="Total vector points upserted to vector store"
    )
    error_code: Optional[str] = Field(
        default=None, description="Safe classified error code on ingestion failure"
    )
    error_message: Optional[str] = Field(
        default=None, description="Sanitized, non-sensitive summary of ingestion failure"
    )
    provenance_eligible: bool = Field(
        default=False,
        description="Indicates whether vector points carry verified document provenance",
    )


class DocumentDeleteResponse(BaseModel):
    """API response model for document deletion endpoint."""

    document_id: str = Field(..., description="Unique persistent identifier of the deleted document")
    deleted: bool = Field(default=True, description="Whether the document was successfully deleted")
    vectors_deleted: int = Field(
        ...,
        ge=0,
        description="Number of vector points deleted from the vector store (non-atomic under concurrent writes)",
    )
    message: str = Field(
        default="Document and associated vectors successfully deleted.",
        description="Human-readable outcome summary",
    )


class DocumentListResponse(BaseModel):
    """API response container for paginated document registry records."""

    documents: List[DocumentRecordResponse] = Field(
        default_factory=list, description="List of registered documents ordered newest first"
    )
    total: int = Field(..., ge=0, description="Total count of registered documents")
    limit: int = Field(..., ge=1, description="Pagination limit applied")
    offset: int = Field(..., ge=0, description="Pagination offset applied")


__all__ = [
    "IngestionResult",
    "IngestionResponse",
    "DocumentRecordResponse",
    "DocumentDeleteResponse",
    "DocumentListResponse",
]

