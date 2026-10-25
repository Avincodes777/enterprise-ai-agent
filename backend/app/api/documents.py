"""API router for document registry exploration endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (
    get_document_registry_dependency,
    get_vector_store_dependency,
)
from app.database import (
    DocumentNotFoundError,
    DocumentRegistryRepository,
    DocumentStateConflictError,
    DocumentStatus,
)
from app.schemas.ingestion import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentRecordResponse,
)
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List registered documents",
    description=(
        "Retrieve a paginated list of registered enterprise knowledge documents and their ingestion "
        "lifecycle statuses ordered by creation time (newest first). "
        "Note: This endpoint currently operates without authentication or authorization controls. "
        "It is provided for the current development setup until multi-user access controls "
        "and organizational tenant boundaries are implemented in subsequent milestones."
    ),
)
def list_documents(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of document records to return (1 to 100)",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of document records to skip for pagination",
    ),
    registry_repo: DocumentRegistryRepository = Depends(get_document_registry_dependency),
) -> DocumentListResponse:
    """
    List registered documents with pagination and newest-first ordering.

    Args:
        limit: Maximum number of records to return (1-100).
        offset: Offset integer for pagination (>= 0).
        registry_repo: Injected DocumentRegistryRepository dependency.

    Returns:
        Structured DocumentListResponse containing document items and pagination totals.
    """
    try:
        total = registry_repo.count_records()
        records = registry_repo.list_records(limit=limit, offset=offset)

        documents = [
            DocumentRecordResponse(
                document_id=rec.document_id,
                document_name=rec.document_name,
                status=rec.status.value,
                collection_name=rec.collection_name,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
                pages_processed=rec.pages_processed,
                chunks_created=rec.chunks_created,
                chunks_stored=rec.chunks_stored,
                error_code=rec.error_code,
                error_message=rec.error_message,
                provenance_eligible=rec.provenance_eligible,
            )
            for rec in records
        ]

        return DocumentListResponse(
            documents=documents,
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("Failed to list registered documents: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving registered documents.",
        ) from exc


@router.get(
    "/documents/{document_id}",
    response_model=DocumentRecordResponse,
    status_code=status.HTTP_200_OK,
    summary="Get registered document by ID",
    description=(
        "Retrieve the registry record and ingestion status of a specific document by its unique identifier. "
        "Note: This endpoint currently operates without authentication or authorization controls. "
        "It is provided for the current development setup until multi-user access controls "
        "and organizational tenant boundaries are implemented in subsequent milestones."
    ),
)
def get_document_by_id(
    document_id: str,
    registry_repo: DocumentRegistryRepository = Depends(get_document_registry_dependency),
) -> DocumentRecordResponse:
    """
    Retrieve a single registered document by unique identifier.

    Args:
        document_id: Unique persistent identifier of the document.
        registry_repo: Injected DocumentRegistryRepository dependency.

    Returns:
        Structured DocumentRecordResponse with document ingestion metadata.

    Raises:
        HTTPException: 404 if the document is not found, or 500 on unexpected errors.
    """
    try:
        record = registry_repo.get_record(document_id=document_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found.",
            )

        return DocumentRecordResponse(
            document_id=record.document_id,
            document_name=record.document_name,
            status=record.status.value,
            collection_name=record.collection_name,
            created_at=record.created_at,
            updated_at=record.updated_at,
            pages_processed=record.pages_processed,
            chunks_created=record.chunks_created,
            chunks_stored=record.chunks_stored,
            error_code=record.error_code,
            error_message=record.error_message,
            provenance_eligible=record.provenance_eligible,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve document '%s': %s",
            document_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the document.",
        ) from exc


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete registered document and associated vectors",
    description=(
        "Safely delete a document's vector points from the vector store and transition its record "
        "to a durable tombstone ('deleted' status) in the persistent registry. Only documents with "
        "verified provenance eligibility can be deleted. Documents in active processing status, "
        "already-deleted documents, or legacy unverified records are protected from deletion."
    ),
)
def delete_document(
    document_id: str,
    registry_repo: DocumentRegistryRepository = Depends(get_document_registry_dependency),
    vector_store: BaseVectorStoreService = Depends(get_vector_store_dependency),
) -> DocumentDeleteResponse:
    """
    Delete an eligible document's vector points and record a durable tombstone.

    Coordinates exact-match vector deletion with atomic registry transition:
    1. Validates document ID presence.
    2. Atomically claims the document into 'deleting' state with an attempt token.
    3. Deletes vector points via exact document_id filter.
    4. Upon vector deletion success, conditionally finalizes the record to 'deleted'.
    5. On vector store failure, records a retryable failure state without marking 'deleted'.

    Args:
        document_id: Unique persistent identifier of the document.
        registry_repo: Injected DocumentRegistryRepository dependency.
        vector_store: Injected BaseVectorStoreService dependency.

    Returns:
        DocumentDeleteResponse detailing document_id and vector deletion count.

    Raises:
        HTTPException: 404 if missing, 409 if conflict/ineligible, 503 if vector store unavailable, 500 on unexpected errors.
    """
    if not document_id or not document_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing or invalid document ID.",
        )

    clean_id = document_id.strip()

    # Step 1: Atomic lifecycle transition to 'deleting'
    try:
        claimed_record = registry_repo.claim_for_deletion(clean_id)
    except Exception as exc:
        logger.error("Failed to claim document '%s' for deletion: %s", clean_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while initiating document deletion.",
        ) from exc

    if claimed_record is None:
        # Check current document state to return accurate, sanitized diagnostic status
        existing_record = registry_repo.get_record(clean_id)
        if existing_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{clean_id}' not found.",
            )

        if not existing_record.provenance_eligible:
            logger.warning(
                "Deletion rejected for document '%s': document has legacy unverified provenance",
                clean_id,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document has legacy unverified provenance and cannot be safely deleted.",
            )

        if existing_record.status == DocumentStatus.DELETED:
            logger.warning("Deletion rejected for document '%s': document has already been deleted", clean_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document has already been deleted.",
            )

        if existing_record.status == DocumentStatus.PROCESSING:
            logger.warning("Deletion rejected for document '%s': document is actively processing", clean_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is currently being processed and cannot be deleted.",
            )

        if existing_record.status == DocumentStatus.DELETING:
            logger.warning("Deletion rejected for document '%s': deletion is already in progress", clean_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document deletion is already in progress.",
            )

        if existing_record.status == DocumentStatus.RECONCILING:
            logger.warning("Deletion rejected for document '%s': document is undergoing reconciliation", clean_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document reconciliation is currently in progress.",
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document in status '{existing_record.status.value}' cannot be deleted.",
        )

    # Step 2: Delete vector points using exact document_id filter
    target_collection = claimed_record.collection_name
    vectors_deleted = 0
    try:
        vectors_deleted = vector_store.delete_by_document_id(
            document_id=clean_id,
            collection_name=target_collection,
        )
    except VectorStoreCollectionNotFoundError as exc:
        logger.warning(
            "Collection '%s' does not exist during deletion of '%s': %s",
            target_collection,
            clean_id,
            exc,
        )
        vectors_deleted = 0
    except (VectorStoreConnectionError, VectorStoreError) as exc:
        logger.error(
            "Vector store deletion failed for document '%s': %s",
            clean_id,
            exc,
            exc_info=True,
        )
        try:
            registry_repo.update_status_deletion_failed(
                document_id=clean_id,
                error_code="VECTOR_STORE_UNAVAILABLE",
                error_message="Vector storage service is currently unavailable.",
                claim_token=claimed_record.claim_token,
            )
        except Exception as repo_exc:
            logger.error("Failed to record deletion failure for '%s': %s", clean_id, repo_exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage service is currently unavailable. Deletion has been aborted and can be retried.",
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error deleting vector points for '%s': %s",
            clean_id,
            exc,
            exc_info=True,
        )
        try:
            registry_repo.update_status_deletion_failed(
                document_id=clean_id,
                error_code="VECTOR_DELETION_FAILED",
                error_message="An unexpected error occurred while deleting vector points.",
                claim_token=claimed_record.claim_token,
            )
        except Exception as repo_exc:
            logger.error("Failed to record deletion failure for '%s': %s", clean_id, repo_exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting document vectors. Deletion can be retried.",
        ) from exc

    # Step 3: Transition SQLite registry record to durable tombstone ('deleted')
    try:
        registry_repo.finalize_deletion(
            document_id=clean_id,
            claim_token=claimed_record.claim_token or "",
        )
    except DocumentStateConflictError as exc:
        logger.error(
            "Failed to finalize deletion for '%s': claim expired or record state conflict: %s",
            clean_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document deletion claim expired or was modified by another operation.",
        ) from exc
    except DocumentNotFoundError as exc:
        logger.error("Document '%s' not found during deletion finalization: %s", clean_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{clean_id}' not found.",
        ) from exc
    except Exception as exc:
        logger.error("Failed to finalize deletion for '%s': %s", clean_id, exc, exc_info=True)
        try:
            registry_repo.update_status_deletion_failed(
                document_id=clean_id,
                error_code="REGISTRY_DELETION_FAILED",
                error_message="Failed to finalize document removal in registry.",
                claim_token=claimed_record.claim_token,
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while finalizing document removal in registry.",
        ) from exc

    return DocumentDeleteResponse(
        document_id=clean_id,
        deleted=True,
        vectors_deleted=vectors_deleted,
        message=f"Document '{clean_id}' and {vectors_deleted} vector points successfully deleted.",
    )


