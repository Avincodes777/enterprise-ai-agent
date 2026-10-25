"""Reconciliation service for safe cleanup of orphaned vectors from failed ingestions."""

from enum import Enum
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.database import (
    DEFAULT_RECONCILIATION_LEASE_SECONDS,
    DEFAULT_RECONCILIATION_LIMIT,
    DEFAULT_RECONCILIATION_MIN_AGE_SECONDS,
    DocumentNotFoundError,
    DocumentRecord,
    DocumentRegistryError,
    DocumentRegistryRepository,
    DocumentStateConflictError,
    DocumentStatus,
)
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)


class ReconciliationOutcomeStatus(str, Enum):
    """Structured status classifications for reconciliation outcomes."""

    DRY_RUN_CANDIDATE = "dry_run_candidate"
    SKIPPED_CLAIM_UNAVAILABLE = "skipped_claim_unavailable"
    SKIPPED_INELIGIBLE = "skipped_ineligible"
    DELETION_SUCCEEDED = "deletion_succeeded"
    DELETION_FAILED = "deletion_failed"
    CLAIM_OWNERSHIP_LOST = "claim_ownership_lost"
    FINALIZATION_FAILED = "finalization_failed"
    DRY_RUN_COUNT_FAILED = "dry_run_count_failed"


class DocumentReconciliationOutcome(BaseModel):
    """Structured per-document reconciliation outcome."""

    document_id: str
    status: ReconciliationOutcomeStatus
    current_status: Optional[str] = None
    previous_status: Optional[str] = None
    provenance_eligible: bool = False
    vector_count: Optional[int] = None
    vectors_deleted: Optional[int] = None
    collection_name: Optional[str] = None
    claim_token: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ReconciliationResult(BaseModel):
    """Aggregate result from a reconciliation sweep or target execution."""

    dry_run: bool
    total_candidates_inspected: int
    processed_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    outcomes: List[DocumentReconciliationOutcome] = Field(default_factory=list)


class ReconciliationService:
    """
    Internal service executing idempotent reconciliation of orphaned vectors
    for eligible failed-ingestion records.

    Mandatory Safety Constraints:
    1. Processes only records whose original status is 'failed' and provenance_eligible is True.
    2. Does not automatically process 'deletion_failed' records (tombstones remain unresolved).
    3. Never processes legacy unverified records with provenance_eligible = False.
    4. Deletes points only through exact document_id filters; never drops or scans entire collections.
    5. Never deletes a registry row; preserves failed-ingestion records to catch late writes.
    """

    def __init__(
        self,
        registry_repo: DocumentRegistryRepository,
        vector_store: BaseVectorStoreService,
        settings: Optional[Settings] = None,
        lease_seconds: float = DEFAULT_RECONCILIATION_LEASE_SECONDS,
    ):
        self.registry_repo = registry_repo
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.lease_seconds = lease_seconds
        self.default_collection = self.settings.qdrant_collection_name

    def reconcile_failed_ingestion(
        self,
        dry_run: bool = True,
        min_age_seconds: int = DEFAULT_RECONCILIATION_MIN_AGE_SECONDS,
        limit: int = DEFAULT_RECONCILIATION_LIMIT,
        target_document_id: Optional[str] = None,
    ) -> ReconciliationResult:
        """
        Discover and reconcile orphaned vectors for eligible failed-ingestion documents.

        Args:
            dry_run: If True, inspects and counts candidates without claiming or deleting.
            min_age_seconds: Grace period duration; candidates updated more recently are excluded.
            limit: Maximum candidates to process in this invocation.
            target_document_id: Optional specific document ID to target instead of discovery.

        Returns:
            ReconciliationResult with detailed per-document outcomes.
        """
        candidates: List[DocumentRecord] = []
        if target_document_id and target_document_id.strip():
            clean_target = target_document_id.strip()
            rec = self.registry_repo.get_record(clean_target)
            if rec:
                candidates = [rec]
        else:
            candidates = self.registry_repo.list_reconciliation_candidates(
                min_age_seconds=min_age_seconds,
                limit=limit,
            )

        outcomes: List[DocumentReconciliationOutcome] = []
        processed_count = 0
        success_count = 0
        failure_count = 0
        skipped_count = 0

        # --- Dry-Run Mode ---
        if dry_run:
            for candidate in candidates:
                # Filter: Process only records whose eligible original status is failed and provenance_eligible is true
                if not candidate.provenance_eligible or (
                    candidate.status != DocumentStatus.FAILED
                    and candidate.previous_status != DocumentStatus.FAILED.value
                ):
                    outcomes.append(
                        DocumentReconciliationOutcome(
                            document_id=candidate.document_id,
                            status=ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE,
                            current_status=candidate.status.value,
                            previous_status=candidate.previous_status,
                            provenance_eligible=candidate.provenance_eligible,
                            collection_name=candidate.collection_name,
                            error_code="INELIGIBLE_CANDIDATE",
                            error_message="Only provenance-eligible failed ingestion records are eligible for reconciliation in this step.",
                        )
                    )
                    skipped_count += 1
                    continue

                target_collection = candidate.collection_name or self.default_collection
                try:
                    count = self.vector_store.count_by_document_id(
                        document_id=candidate.document_id,
                        collection_name=target_collection,
                    )
                    outcomes.append(
                        DocumentReconciliationOutcome(
                            document_id=candidate.document_id,
                            status=ReconciliationOutcomeStatus.DRY_RUN_CANDIDATE,
                            current_status=candidate.status.value,
                            previous_status=candidate.previous_status,
                            provenance_eligible=candidate.provenance_eligible,
                            vector_count=count,
                            collection_name=target_collection,
                        )
                    )
                    processed_count += 1
                except Exception as count_exc:
                    logger.warning(
                        "Dry-run vector count failed for document '%s' in collection '%s': %s",
                        candidate.document_id,
                        target_collection,
                        count_exc,
                    )
                    outcomes.append(
                        DocumentReconciliationOutcome(
                            document_id=candidate.document_id,
                            status=ReconciliationOutcomeStatus.DRY_RUN_COUNT_FAILED,
                            current_status=candidate.status.value,
                            previous_status=candidate.previous_status,
                            provenance_eligible=candidate.provenance_eligible,
                            vector_count=None,  # Never represent unknown count as zero
                            collection_name=target_collection,
                            error_code="VECTOR_COUNT_FAILED",
                            error_message=f"Failed to count vectors in vector store: {count_exc}",
                        )
                    )
                    failure_count += 1
                    processed_count += 1

            return ReconciliationResult(
                dry_run=True,
                total_candidates_inspected=len(candidates),
                processed_count=processed_count,
                success_count=success_count,
                failure_count=failure_count,
                skipped_count=skipped_count,
                outcomes=outcomes,
            )

        # --- Execute Mode ---
        for candidate in candidates:
            # 1. Pre-claim eligibility check
            if not candidate.provenance_eligible or (
                candidate.status != DocumentStatus.FAILED
                and candidate.previous_status != DocumentStatus.FAILED.value
            ):
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=candidate.document_id,
                        status=ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE,
                        current_status=candidate.status.value,
                        previous_status=candidate.previous_status,
                        provenance_eligible=candidate.provenance_eligible,
                        collection_name=candidate.collection_name,
                        error_code="INELIGIBLE_CANDIDATE",
                        error_message="Only provenance-eligible failed ingestion records are eligible for reconciliation in this step.",
                    )
                )
                skipped_count += 1
                continue

            # 2. Atomic claim
            try:
                claimed = self.registry_repo.claim_for_reconciliation(
                    document_id=candidate.document_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as claim_exc:
                logger.error("Error attempting claim on document '%s': %s", candidate.document_id, claim_exc)
                claimed = None

            if claimed is None:
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=candidate.document_id,
                        status=ReconciliationOutcomeStatus.SKIPPED_CLAIM_UNAVAILABLE,
                        current_status=candidate.status.value,
                        previous_status=candidate.previous_status,
                        provenance_eligible=candidate.provenance_eligible,
                        collection_name=candidate.collection_name,
                        error_code="CLAIM_UNAVAILABLE",
                        error_message="Unable to acquire atomic reconciliation claim (already claimed, concurrent lease, or modified).",
                    )
                )
                skipped_count += 1
                continue

            # 3. Post-claim verification: ensure claimed record is failed ingestion and provenance eligible
            if not claimed.provenance_eligible or claimed.previous_status != DocumentStatus.FAILED.value:
                try:
                    if claimed.claim_token:
                        self.registry_repo.release_reconciliation_claim(
                            document_id=claimed.document_id,
                            claim_token=claimed.claim_token,
                            error_code="INELIGIBLE_RECONCILIATION",
                            error_message="Record ineligible for failed ingestion cleanup.",
                        )
                except Exception as rel_exc:
                    logger.error("Failed to release ineligible claim on '%s': %s", claimed.document_id, rel_exc)

                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        collection_name=claimed.collection_name,
                        claim_token=claimed.claim_token,
                        error_code="INELIGIBLE_POST_CLAIM",
                        error_message="Record ineligible for failed ingestion cleanup after claim acquisition.",
                    )
                )
                skipped_count += 1
                continue

            # 4. Pre-side-effect ownership check: verify active unexpired ownership token
            if not claimed.claim_token:
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.CLAIM_OWNERSHIP_LOST,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        collection_name=claimed.collection_name,
                        error_code="MISSING_CLAIM_TOKEN",
                        error_message="Claim was acquired without an ownership token.",
                    )
                )
                failure_count += 1
                processed_count += 1
                continue

            try:
                is_owned = self.registry_repo.verify_reconciliation_claim(
                    document_id=claimed.document_id,
                    claim_token=claimed.claim_token,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as ver_exc:
                logger.error("Error verifying reconciliation claim on '%s': %s", claimed.document_id, ver_exc)
                is_owned = False

            if not is_owned:
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.CLAIM_OWNERSHIP_LOST,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        collection_name=claimed.collection_name,
                        claim_token=claimed.claim_token,
                        error_code="CLAIM_OWNERSHIP_LOST",
                        error_message="Reconciliation claim is no longer owned or lease expired before vector deletion.",
                    )
                )
                failure_count += 1
                processed_count += 1
                continue

            # 5. Exact document vector deletion
            target_collection = claimed.collection_name or self.default_collection
            vectors_deleted = 0
            deletion_failed = False
            deletion_error = None

            try:
                vectors_deleted = self.vector_store.delete_by_document_id(
                    document_id=claimed.document_id,
                    collection_name=target_collection,
                )
            except (VectorStoreCollectionNotFoundError, VectorStoreConnectionError, VectorStoreError) as vs_exc:
                deletion_failed = True
                deletion_error = f"Vector store error during reconciliation: {vs_exc}"
                logger.warning("Vector deletion failed for '%s': %s", claimed.document_id, vs_exc)
            except Exception as exc:
                deletion_failed = True
                deletion_error = f"Unexpected error during vector deletion: {exc}"
                logger.error("Unexpected vector deletion error for '%s': %s", claimed.document_id, exc, exc_info=True)

            if deletion_failed:
                # Attempt safe release of claim
                try:
                    self.registry_repo.release_reconciliation_claim(
                        document_id=claimed.document_id,
                        claim_token=claimed.claim_token,
                        error_code="RECONCILIATION_VECTOR_DELETION_FAILED",
                        error_message=deletion_error or "Vector deletion failed during reconciliation.",
                    )
                except Exception as rel_exc:
                    logger.error("Failed to release claim on '%s' after deletion failure: %s", claimed.document_id, rel_exc)

                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.DELETION_FAILED,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        collection_name=target_collection,
                        claim_token=claimed.claim_token,
                        error_code="VECTOR_DELETION_FAILED",
                        error_message=deletion_error,
                    )
                )
                failure_count += 1
                processed_count += 1
                continue

            # 6. Finalize reconciliation: verify exact token, restore failed, set chunks_stored=0, retain row
            try:
                finalized = self.registry_repo.finalize_reconciliation(
                    document_id=claimed.document_id,
                    claim_token=claimed.claim_token,
                    chunks_stored=0,
                )
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.DELETION_SUCCEEDED,
                        current_status=finalized.status.value,
                        previous_status=finalized.previous_status,
                        provenance_eligible=finalized.provenance_eligible,
                        vectors_deleted=vectors_deleted,
                        collection_name=target_collection,
                    )
                )
                success_count += 1
                processed_count += 1
            except DocumentStateConflictError as conflict_exc:
                logger.error(
                    "Finalization state conflict for document '%s' (claim token was likely superseded): %s",
                    claimed.document_id,
                    conflict_exc,
                )
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.FINALIZATION_FAILED,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        vectors_deleted=vectors_deleted,
                        collection_name=target_collection,
                        claim_token=claimed.claim_token,
                        error_code="FINALIZATION_CONFLICT",
                        error_message=f"State conflict finalizing reconciliation: {conflict_exc}",
                    )
                )
                failure_count += 1
                processed_count += 1
            except Exception as fin_exc:
                logger.error("Unexpected error finalizing reconciliation for '%s': %s", claimed.document_id, fin_exc, exc_info=True)
                outcomes.append(
                    DocumentReconciliationOutcome(
                        document_id=claimed.document_id,
                        status=ReconciliationOutcomeStatus.FINALIZATION_FAILED,
                        current_status=claimed.status.value,
                        previous_status=claimed.previous_status,
                        provenance_eligible=claimed.provenance_eligible,
                        vectors_deleted=vectors_deleted,
                        collection_name=target_collection,
                        claim_token=claimed.claim_token,
                        error_code="FINALIZATION_ERROR",
                        error_message=f"Unexpected error finalizing reconciliation: {fin_exc}",
                    )
                )
                failure_count += 1
                processed_count += 1

        return ReconciliationResult(
            dry_run=False,
            total_candidates_inspected=len(candidates),
            processed_count=processed_count,
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=skipped_count,
            outcomes=outcomes,
        )


def get_reconciliation_service(
    registry_repo: Optional[DocumentRegistryRepository] = None,
    vector_store: Optional[BaseVectorStoreService] = None,
    settings: Optional[Settings] = None,
) -> ReconciliationService:
    """Factory creating ReconciliationService with dependencies."""
    from app.api.dependencies import (
        get_shared_document_registry,
        get_shared_vector_store,
    )

    repo = registry_repo or get_shared_document_registry()
    vs = vector_store or get_shared_vector_store()
    cfg = settings or get_settings()
    return ReconciliationService(registry_repo=repo, vector_store=vs, settings=cfg)
