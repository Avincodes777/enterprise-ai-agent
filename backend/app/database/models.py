"""Data models and enums for persistent document registry."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle status of a document within the ingestion registry."""

    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DELETING = "deleting"
    DELETION_FAILED = "deletion_failed"
    RECONCILING = "reconciling"
    DELETED = "deleted"


class DocumentRecord(BaseModel):
    """Persistent record representing a registered document and its ingestion state."""

    document_id: str = Field(..., description="Unique persistent identifier for the document")
    document_name: str = Field(..., description="Original filename of the uploaded document")
    status: DocumentStatus = Field(
        default=DocumentStatus.PROCESSING,
        description="Ingestion status: processing, succeeded, failed, deleting, deletion_failed, reconciling, or deleted",
    )
    collection_name: str = Field(..., description="Target vector store collection name")
    created_at: str = Field(..., description="UTC ISO-8601 timestamp of initial registration")
    updated_at: str = Field(..., description="UTC ISO-8601 timestamp of latest status update")
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
        description="Whether points in vector store carry verified document_id provenance",
    )
    claim_token: Optional[str] = Field(
        default=None,
        description="Unique generation token identifying active claim ownership",
    )
    previous_status: Optional[str] = Field(
        default=None,
        description="Pre-claim status preserved across reconciliation attempts for safe recovery",
    )

