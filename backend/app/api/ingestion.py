"""API router for document ingestion endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_ingestion_service_dependency
from app.services.ingestion import (
    IngestionEmptyDocumentError,
    IngestionError,
    IngestionResult,
    IngestionService,
    IngestionValidationError,
)
from app.services.pdf_loader import PDFEncryptedError, PDFProcessingError
from app.services.vector_store import VectorStoreConnectionError, VectorStoreError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF document into knowledge base",
    description="Upload a PDF document, extract and clean text, chunk, embed, and store in the vector store.",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    collection_name: Optional[str] = Form(
        default=None, description="Optional target vector collection name override"
    ),
    ingestion_service: IngestionService = Depends(get_ingestion_service_dependency),
) -> IngestionResult:
    """
    Ingest an uploaded PDF document into the configured vector store.

    Args:
        file: Uploaded file object.
        collection_name: Optional target collection name.
        ingestion_service: Injected IngestionService dependency.

    Returns:
        Structured IngestionResult with processing and storage details.
    """
    try:
        # Validate filename presence
        if not file.filename or not file.filename.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing or invalid filename.",
            )

        cleaned_filename = file.filename.strip()

        # Validate file extension
        if not cleaned_filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported file type. Only PDF documents are accepted.",
            )

        # Validate content-type if provided by client
        if file.content_type:
            allowed_content_types = {
                "application/pdf",
                "application/x-pdf",
                "application/octet-stream",
                "binary/octet-stream",
            }
            if file.content_type.lower() not in allowed_content_types:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Unsupported content type. Expected a PDF document.",
                )

        # Read file bytes
        try:
            file_bytes = await file.read()
        except Exception as exc:
            logger.error("Failed to read uploaded file '%s': %s", cleaned_filename, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to read uploaded file content.",
            ) from exc

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded file is empty.",
            )

        target_collection = (
            collection_name.strip()
            if collection_name is not None and collection_name.strip()
            else None
        )

        try:
            result = ingestion_service.ingest_pdf(
                file_bytes=file_bytes,
                document_name=cleaned_filename,
                collection_name=target_collection,
            )
            return result
        except IngestionValidationError as exc:
            logger.warning("Ingestion validation error for '%s': %s", cleaned_filename, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Document validation failed: {exc}",
            ) from exc
        except IngestionEmptyDocumentError as exc:
            logger.warning("Empty document error for '%s': %s", cleaned_filename, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF document contains no extractable text.",
            ) from exc
        except IngestionError as exc:
            cause = getattr(exc, "__cause__", None)
            exc_str = str(exc).lower()

            # Check if root cause is PDF corruption or encryption
            if (
                isinstance(cause, (PDFProcessingError, PDFEncryptedError))
                or "pdf" in exc_str
                or "corrupt" in exc_str
                or "parse" in exc_str
            ):
                logger.warning("Malformed PDF uploaded '%s': %s", cleaned_filename, exc)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to parse PDF document. The file may be corrupt or invalid.",
                ) from exc

            # Check if root cause is vector storage / infrastructure failure
            if (
                isinstance(cause, (VectorStoreError, VectorStoreConnectionError))
                or "vector store" in exc_str
                or "qdrant" in exc_str
                or "connection refused" in exc_str
                or "database timeout" in exc_str
            ):
                logger.error(
                    "Vector storage failure during ingestion of '%s': %s",
                    cleaned_filename,
                    exc,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector storage service is currently unavailable.",
                ) from exc

            # General ingestion processing error
            logger.error(
                "Ingestion processing failure for '%s': %s", cleaned_filename, exc, exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while processing the document.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error ingesting '%s': %s", cleaned_filename, exc, exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during document ingestion.",
            ) from exc
    finally:
        await file.close()
