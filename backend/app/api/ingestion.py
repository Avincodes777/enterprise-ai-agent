"""API router for document ingestion endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_document_registry_dependency,
    get_ingestion_service_dependency,
    get_settings_dependency,
    get_vector_store_dependency,
)
from app.config import Settings
from app.database.registry import (
    DocumentNotFoundError,
    DocumentRegistryRepository,
    DocumentStateConflictError,
)
from app.schemas.ingestion import IngestionResponse
from app.services.ingestion import (
    IngestionEmptyDocumentError,
    IngestionError,
    IngestionService,
    IngestionValidationError,
)
from app.services.pdf_loader import PDFEncryptedError, PDFProcessingError
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


def _safe_record_failure(
    registry_repo: DocumentRegistryRepository,
    document_id: Optional[str],
    error_code: str,
    error_message: str,
    vector_store: Optional[BaseVectorStoreService] = None,
    collection_name: Optional[str] = None,
) -> None:
    """Safely record ingestion failure in the registry without throwing unhandled exceptions.

    If recording the failure encounters a DocumentStateConflictError (meaning the document
    already transitioned out of processing), and cleanup parameters are provided, perform
    compensating vector deletion using the exact registered document ID and target collection.
    """
    if not document_id:
        return
    try:
        registry_repo.update_status_failed(
            document_id=document_id,
            error_code=error_code,
            error_message=error_message,
        )
    except DocumentStateConflictError as exc:
        logger.warning(
            "Skipping failure status recording for document '%s': document has already left processing state: %s",
            document_id,
            exc,
        )
        if vector_store is not None and collection_name:
            try:
                deleted_count = vector_store.delete_by_document_id(
                    document_id=document_id,
                    collection_name=collection_name,
                )
                logger.info(
                    "Compensating cleanup successfully deleted %d vectors for document '%s' in collection '%s' after state conflict.",
                    deleted_count,
                    document_id,
                    collection_name,
                )
            except VectorStoreCollectionNotFoundError as cnf_exc:
                logger.warning(
                    "Collection '%s' does not exist during compensating cleanup for document '%s': zero vectors remain: %s",
                    collection_name,
                    document_id,
                    cnf_exc,
                )
            except Exception as cleanup_exc:
                logger.error(
                    "Compensating vector cleanup failed for document '%s' in collection '%s' after state conflict: %s",
                    document_id,
                    collection_name,
                    cleanup_exc,
                    exc_info=True,
                )
    except Exception as exc:
        logger.error(
            "Failed to record failure status in registry for document '%s': %s",
            document_id,
            exc,
            exc_info=True,
        )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF document into knowledge base",
    description="Upload a PDF document, extract and clean text, chunk, embed, store in the vector store, and record in registry.",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    collection_name: Optional[str] = Form(
        default=None, description="Optional target vector collection name override"
    ),
    ingestion_service: IngestionService = Depends(get_ingestion_service_dependency),
    settings: Settings = Depends(get_settings_dependency),
    registry_repo: DocumentRegistryRepository = Depends(get_document_registry_dependency),
    vector_store: BaseVectorStoreService = Depends(get_vector_store_dependency),
) -> IngestionResponse:
    """
    Ingest an uploaded PDF document into the configured vector store and track in registry.

    Args:
        file: Uploaded file object.
        collection_name: Optional target collection name.
        ingestion_service: Injected IngestionService dependency.
        settings: Injected application Settings dependency.
        registry_repo: Injected DocumentRegistryRepository dependency.

    Returns:
        Structured IngestionResponse with document_id and processing details.
    """
    registered_document_id: Optional[str] = None
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

        # Read file bytes in bounded chunks to prevent unbounded memory allocation and oversized uploads
        try:
            chunks = []
            total_bytes = 0
            chunk_size = min(64 * 1024, max(1, settings.max_upload_size_bytes))

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size_bytes:
                    logger.warning(
                        "Upload rejected for '%s': file size (%d bytes) exceeds configured limit (%d bytes)",
                        cleaned_filename,
                        total_bytes,
                        settings.max_upload_size_bytes,
                    )
                    raise HTTPException(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        detail=f"Uploaded file exceeds the maximum allowed size of {settings.max_upload_size_bytes} bytes.",
                    )
                chunks.append(chunk)

            file_bytes = b"".join(chunks)
        except HTTPException:
            raise
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
        effective_collection = (
            target_collection
            or getattr(ingestion_service, "default_collection_name", None)
            or "enterprise_knowledge"
        )
        active_vector_store = (
            getattr(ingestion_service, "vector_store_service", None)
            or vector_store
        )

        # Create document registry record with 'processing' status before ingestion begins
        try:
            record = registry_repo.create_record(
                document_name=cleaned_filename,
                collection_name=effective_collection,
                provenance_eligible=True,
            )
            registered_document_id = record.document_id

        except Exception as exc:
            logger.error(
                "Failed to register document '%s' in registry: %s",
                cleaned_filename,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while registering the document.",
            ) from exc

        try:
            def heartbeat() -> None:
                registry_repo.heartbeat_ingestion(registered_document_id)

            result = ingestion_service.ingest_pdf(
                file_bytes=file_bytes,
                document_name=cleaned_filename,
                collection_name=target_collection,
                document_id=registered_document_id,
                heartbeat_callback=heartbeat,
            )

            # Update registry record to succeeded
            try:
                registry_repo.update_status_succeeded(
                    document_id=registered_document_id,
                    pages_processed=result.pages_processed,
                    chunks_created=result.chunks_created,
                    chunks_stored=result.chunks_stored,
                    collection_name=result.collection_name,
                )
            except DocumentStateConflictError as conflict_exc:
                logger.error(
                    "Ingestion finalization rejected for '%s' due to state conflict: %s. "
                    "Initiating compensating vector cleanup without overwriting registry state.",
                    registered_document_id,
                    conflict_exc,
                    exc_info=True,
                )
                cleanup_succeeded = False
                target_col = result.collection_name
                try:
                    active_vector_store = (
                        getattr(ingestion_service, "vector_store_service", None)
                        or vector_store
                    )
                    if active_vector_store is not None:
                        deleted_count = active_vector_store.delete_by_document_id(
                            document_id=registered_document_id,
                            collection_name=target_col,
                        )
                        cleanup_succeeded = True
                        logger.info(
                            "Compensating cleanup successfully deleted %d vectors for late document '%s' in collection '%s'.",
                            deleted_count,
                            registered_document_id,
                            target_col,
                        )
                    else:
                        logger.warning(
                            "Compensating cleanup skipped for late document '%s': no vector store service available.",
                            registered_document_id,
                        )
                except VectorStoreCollectionNotFoundError as cnf_exc:
                    cleanup_succeeded = True
                    logger.warning(
                        "Collection '%s' does not exist during compensating cleanup for late document '%s': zero vectors remain: %s",
                        target_col,
                        registered_document_id,
                        cnf_exc,
                    )
                except Exception as cleanup_exc:
                    logger.error(
                        "Compensating vector cleanup failed for late document '%s' in collection '%s': %s",
                        registered_document_id,
                        target_col,
                        cleanup_exc,
                        exc_info=True,
                    )

                # Preserve the newer registry state without overwriting it
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Document ingestion conflict: document status was modified by another operation.",
                ) from conflict_exc
            except Exception as finalization_exc:
                logger.error(
                    "Failed to record ingestion success in registry for '%s': %s. "
                    "Initiating compensating vector cleanup.",
                    registered_document_id,
                    finalization_exc,
                    exc_info=True,
                )

                # Step 21F: Compensating Cleanup
                # Qdrant upsert succeeded, but registry success transition failed.
                # Attempt to delete points from the exact collection using the exact registered document_id.
                cleanup_succeeded = False
                target_col = result.collection_name
                try:
                    active_vector_store = (
                        getattr(ingestion_service, "vector_store_service", None)
                        or vector_store
                    )
                    if active_vector_store is not None:
                        deleted_count = active_vector_store.delete_by_document_id(
                            document_id=registered_document_id,
                            collection_name=target_col,
                        )
                        cleanup_succeeded = True
                        logger.info(
                            "Compensating cleanup successfully deleted %d vectors for document '%s' in collection '%s'.",
                            deleted_count,
                            registered_document_id,
                            target_col,
                        )
                    else:
                        logger.warning(
                            "Compensating cleanup skipped for document '%s': no vector store service available.",
                            registered_document_id,
                        )
                except VectorStoreCollectionNotFoundError as cnf_exc:
                    cleanup_succeeded = True
                    logger.warning(
                        "Collection '%s' does not exist during compensating cleanup for '%s': zero vectors remain: %s",
                        target_col,
                        registered_document_id,
                        cnf_exc,
                    )
                except Exception as cleanup_exc:
                    logger.error(
                        "Compensating vector cleanup failed for document '%s' in collection '%s': %s",
                        registered_document_id,
                        target_col,
                        cleanup_exc,
                        exc_info=True,
                    )

                # Attempt to mark the registry record as failed
                try:
                    error_code = "FINALIZATION_FAILED" if cleanup_succeeded else "COMPENSATION_FAILED"
                    error_message = (
                        "Document ingestion finalization failed; vectors were safely purged."
                        if cleanup_succeeded
                        else "Document ingestion finalization failed; compensating vector cleanup could not be completed."
                    )
                    registry_repo.update_status_failed(
                        document_id=registered_document_id,
                        error_code=error_code,
                        error_message=error_message,
                    )
                except Exception as status_exc:
                    logger.error(
                        "Failed to update registry status to failed for document '%s' after finalization error: %s",
                        registered_document_id,
                        status_exc,
                        exc_info=True,
                    )

                # Preserve the original finalization failure as primary error and return sanitized detail
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to persist ingestion success status in registry.",
                ) from finalization_exc

            return IngestionResponse(
                document_id=registered_document_id,
                document_name=result.document_name,
                pages_processed=result.pages_processed,
                chunks_created=result.chunks_created,
                chunks_stored=result.chunks_stored,
                collection_name=result.collection_name,
                chunk_ids=result.chunk_ids,
            )
        except (DocumentStateConflictError, DocumentNotFoundError) as conflict_exc:
            logger.warning(
                "Ingestion aborted for '%s' due to heartbeat state conflict: %s. "
                "Preserving newer registry state.",
                registered_document_id,
                conflict_exc,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document ingestion conflict: document status was modified by another operation.",
            ) from conflict_exc
        except IngestionValidationError as exc:
            logger.warning("Ingestion validation error for '%s': %s", cleaned_filename, exc)
            _safe_record_failure(
                registry_repo,
                registered_document_id,
                error_code="VALIDATION_ERROR",
                error_message="Document validation failed during ingestion.",
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Document validation failed: {exc}",
            ) from exc
        except IngestionEmptyDocumentError as exc:
            logger.warning("Empty document error for '%s': %s", cleaned_filename, exc)
            _safe_record_failure(
                registry_repo,
                registered_document_id,
                error_code="EMPTY_DOCUMENT",
                error_message="PDF document contains no extractable text.",
            )
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
                _safe_record_failure(
                    registry_repo,
                    registered_document_id,
                    error_code="MALFORMED_PDF",
                    error_message="Failed to parse PDF document. The file may be corrupt or encrypted.",
                )
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
                _safe_record_failure(
                    registry_repo,
                    registered_document_id,
                    error_code="VECTOR_STORE_UNAVAILABLE",
                    error_message="Vector storage service is currently unavailable.",
                    vector_store=active_vector_store,
                    collection_name=effective_collection,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector storage service is currently unavailable.",
                ) from exc

            # General ingestion processing error
            logger.error(
                "Ingestion processing failure for '%s': %s", cleaned_filename, exc, exc_info=True
            )
            _safe_record_failure(
                registry_repo,
                registered_document_id,
                error_code="INGESTION_ERROR",
                error_message="An error occurred while processing the document.",
                vector_store=active_vector_store,
                collection_name=effective_collection,
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
            _safe_record_failure(
                registry_repo,
                registered_document_id,
                error_code="UNEXPECTED_ERROR",
                error_message="An unexpected error occurred during document ingestion.",
                vector_store=active_vector_store,
                collection_name=effective_collection,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during document ingestion.",
            ) from exc
    finally:
        await file.close()

