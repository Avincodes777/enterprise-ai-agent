"""Unit tests for the ReconciliationService failed-ingestion executor."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings
from app.database import (
    DEFAULT_RECONCILIATION_LEASE_SECONDS,
    DocumentNotFoundError,
    DocumentRecord,
    DocumentRegistryError,
    DocumentRegistryRepository,
    DocumentStateConflictError,
    DocumentStatus,
)
from app.services.reconciliation import (
    DocumentReconciliationOutcome,
    ReconciliationOutcomeStatus,
    ReconciliationResult,
    ReconciliationService,
    get_reconciliation_service,
)
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)


class TestReconciliationService(unittest.TestCase):
    """Test suite for ReconciliationService failed-ingestion reconciliation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_reconciliation.db")
        self.repo = DocumentRegistryRepository(db_path=self.db_path)
        self.mock_vector_store = MagicMock(spec=BaseVectorStoreService)
        self.settings = Settings(qdrant_collection_name="test_enterprise_collection")
        self.service = ReconciliationService(
            registry_repo=self.repo,
            vector_store=self.mock_vector_store,
            settings=self.settings,
            lease_seconds=60.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Dry-run performs no claim, mutation, or deletion
    # -------------------------------------------------------------------------
    def test_dry_run_zero_mutation(self):
        """Verify dry-run mode discovers candidates, counts vectors, but performs no claims or deletes."""
        rec = self.repo.create_record("fail1.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "INGEST_ERR", "Failed parse")

        # Age record beyond discovery grace period
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.count_by_document_id.return_value = 15

        res = self.service.reconcile_failed_ingestion(dry_run=True, min_age_seconds=60)

        self.assertTrue(res.dry_run)
        self.assertEqual(res.total_candidates_inspected, 1)
        self.assertEqual(res.processed_count, 1)
        self.assertEqual(len(res.outcomes), 1)

        outcome = res.outcomes[0]
        self.assertEqual(outcome.document_id, rec.document_id)
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DRY_RUN_CANDIDATE)
        self.assertEqual(outcome.vector_count, 15)

        # Zero deletions attempted
        self.mock_vector_store.delete_by_document_id.assert_not_called()
        self.mock_vector_store.count_by_document_id.assert_called_once_with(
            document_id=rec.document_id,
            collection_name="test_enterprise_collection",
        )

        # Registry row remains completely untouched (status is still 'failed', not 'reconciling')
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertIsNone(persisted.claim_token)
        self.assertEqual(persisted.updated_at, stale_time)

    # -------------------------------------------------------------------------
    # 2. Dry-run count failure is distinct from zero count
    # -------------------------------------------------------------------------
    def test_dry_run_count_failure_distinct_from_zero(self):
        """Verify that when vector store count fails in dry run, outcome is DRY_RUN_COUNT_FAILED with vector_count=None."""
        rec = self.repo.create_record("fail2.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Fail")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.count_by_document_id.side_effect = VectorStoreConnectionError("Qdrant offline")

        res = self.service.reconcile_failed_ingestion(dry_run=True, min_age_seconds=60)

        self.assertTrue(res.dry_run)
        self.assertEqual(res.failure_count, 1)
        self.assertEqual(len(res.outcomes), 1)

        outcome = res.outcomes[0]
        self.assertEqual(outcome.document_id, rec.document_id)
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DRY_RUN_COUNT_FAILED)
        self.assertIsNone(outcome.vector_count)  # MUST NOT be 0
        self.assertEqual(outcome.error_code, "VECTOR_COUNT_FAILED")
        self.assertIn("Qdrant offline", outcome.error_message)

    # -------------------------------------------------------------------------
    # 3. Successful execute mode cleans vectors and finalizes
    # -------------------------------------------------------------------------
    def test_execute_mode_successful_cleanup_and_finalization(self):
        """Verify successful reconciliation claims record, deletes vectors by ID, and finalizes with chunks_stored=0."""
        rec = self.repo.create_record("fail3.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "INGEST_ERR", "OOM error")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.delete_by_document_id.return_value = 8

        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertFalse(res.dry_run)
        self.assertEqual(res.success_count, 1)
        self.assertEqual(res.failure_count, 0)
        self.assertEqual(res.skipped_count, 0)

        outcome = res.outcomes[0]
        self.assertEqual(outcome.document_id, rec.document_id)
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DELETION_SUCCEEDED)
        self.assertEqual(outcome.vectors_deleted, 8)

        # Vector deletion called with exact document ID and collection
        self.mock_vector_store.delete_by_document_id.assert_called_once_with(
            document_id=rec.document_id,
            collection_name="test_enterprise_collection",
        )

        # Registry row is retained, status restored to 'failed', chunks_stored=0, claim_token=None
        persisted = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertEqual(persisted.chunks_stored, 0)
        self.assertIsNone(persisted.claim_token)
        # Original error info preserved for auditability
        self.assertEqual(persisted.error_code, "INGEST_ERR")
        self.assertEqual(persisted.error_message, "OOM error")

    # -------------------------------------------------------------------------
    # 4. Non-failed records (deletion_failed) are skipped in this step
    # -------------------------------------------------------------------------
    def test_deletion_failed_records_are_skipped(self):
        """Verify that deletion_failed records are skipped and never claimed or modified by failed-ingestion reconciliation."""
        rec = self.repo.create_record("del_failed.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec.document_id)
        self.repo.update_status_deletion_failed(rec.document_id, "DEL_ERR", "Timeout")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.total_candidates_inspected, 1)
        self.assertEqual(res.skipped_count, 1)
        self.assertEqual(res.success_count, 0)

        outcome = res.outcomes[0]
        self.assertEqual(outcome.document_id, rec.document_id)
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE)

        # Qdrant delete was NEVER called
        self.mock_vector_store.delete_by_document_id.assert_not_called()

        # Database record remains in deletion_failed
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.DELETION_FAILED)
        self.assertIsNone(persisted.claim_token)

    # -------------------------------------------------------------------------
    # 5. Legacy unverified provenance records are strictly excluded
    # -------------------------------------------------------------------------
    def test_legacy_unverified_provenance_records_excluded(self):
        """Verify records with provenance_eligible=False are excluded from candidate discovery and never claimed."""
        rec = self.repo.create_record("legacy.pdf", "test_enterprise_collection", provenance_eligible=False)
        self.repo.update_status_failed(rec.document_id, "LEGACY_ERR", "Old error")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        # Discovery automatically excludes provenance_eligible = 0
        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)
        self.assertEqual(res.total_candidates_inspected, 0)

        # Even if targeted directly by ID, execution skips as ineligible
        res_targeted = self.service.reconcile_failed_ingestion(
            dry_run=False,
            target_document_id=rec.document_id,
        )
        self.assertEqual(res_targeted.skipped_count, 1)
        self.assertEqual(res_targeted.outcomes[0].status, ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE)
        self.mock_vector_store.delete_by_document_id.assert_not_called()

    # -------------------------------------------------------------------------
    # 6. Skip when claim is unavailable
    # -------------------------------------------------------------------------
    def test_skip_when_claim_unavailable(self):
        """Verify that if a record is claimed by another worker concurrently, reconciliation skips without Qdrant call."""
        rec = self.repo.create_record("concurrent.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Fail")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        # Simulate another worker claiming the record right before our service claims it
        other_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(other_claim)

        res = self.service.reconcile_failed_ingestion(
            dry_run=False,
            target_document_id=rec.document_id,
        )

        self.assertEqual(res.skipped_count, 1)
        self.assertEqual(len(res.outcomes), 1)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.SKIPPED_CLAIM_UNAVAILABLE)
        self.mock_vector_store.delete_by_document_id.assert_not_called()

        # Record remains owned by the other claimant
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.claim_token, other_claim.claim_token)

    # -------------------------------------------------------------------------
    # 7. Ownership lost before Qdrant side effect
    # -------------------------------------------------------------------------
    def test_ownership_lost_before_vector_deletion_aborts(self):
        """Verify that if ownership check fails before Qdrant deletion, execution aborts with CLAIM_OWNERSHIP_LOST."""
        rec = self.repo.create_record("lease_expired.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Fail")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        # Mock verify_reconciliation_claim to return False (simulating lease expiry or reclamation)
        with patch.object(self.repo, "verify_reconciliation_claim", return_value=False):
            res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.failure_count, 1)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.CLAIM_OWNERSHIP_LOST)
        self.mock_vector_store.delete_by_document_id.assert_not_called()

    # -------------------------------------------------------------------------
    # 8. Vector deletion failure triggers claim release and preserves diagnostics
    # -------------------------------------------------------------------------
    def test_vector_deletion_failure_releases_claim_preserving_retryability(self):
        """Verify vector deletion exception calls release_reconciliation_claim and preserves retryability."""
        rec = self.repo.create_record("del_err.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ORIG_ERR", "Original failure")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.delete_by_document_id.side_effect = VectorStoreError("Write timeout")

        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.failure_count, 1)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DELETION_FAILED)
        self.assertIn("Write timeout", outcome.error_message)

        # Claim was released back to 'failed'
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertIsNone(persisted.claim_token)
        self.assertEqual(persisted.error_code, "RECONCILIATION_VECTOR_DELETION_FAILED")

        # Immediately retryable: another claim can be acquired
        reclaimed = self.repo.claim_for_reconciliation(rec.document_id)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.status, DocumentStatus.RECONCILING)

    # -------------------------------------------------------------------------
    # 9. Finalization conflict records failure without overwriting state
    # -------------------------------------------------------------------------
    def test_finalization_conflict_records_failure_without_overwriting(self):
        """Verify finalization state conflict (superseded token) reports FINALIZATION_FAILED without overwriting."""
        rec = self.repo.create_record("fin_conflict.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Fail")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.delete_by_document_id.return_value = 5

        # Mock finalize_reconciliation to raise DocumentStateConflictError
        with patch.object(
            self.repo,
            "finalize_reconciliation",
            side_effect=DocumentStateConflictError("Claim token mismatch"),
        ):
            res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.failure_count, 1)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.FINALIZATION_FAILED)
        self.assertEqual(outcome.error_code, "FINALIZATION_CONFLICT")

    # -------------------------------------------------------------------------
    # 10. Idempotent retry on already-cleaned record
    # -------------------------------------------------------------------------
    def test_idempotent_retry_on_already_cleaned_record(self):
        """Verify reconciling a document whose vectors were already 0 succeeds and retains row in failed status."""
        rec = self.repo.create_record("idempotent.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        # Vector store returns 0 points deleted (already cleaned)
        self.mock_vector_store.delete_by_document_id.return_value = 0

        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.success_count, 1)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DELETION_SUCCEEDED)
        self.assertEqual(outcome.vectors_deleted, 0)

        # Row remains present in failed status with chunks_stored = 0
        persisted = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertEqual(persisted.chunks_stored, 0)

    # -------------------------------------------------------------------------
    # 11. Missing collection treated as failure, not successful zero cleanup
    # -------------------------------------------------------------------------
    def test_missing_collection_treated_as_failure(self):
        """Verify VectorStoreCollectionNotFoundError is treated as failure and releases claim."""
        rec = self.repo.create_record("missing_col.pdf", "nonexistent_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.mock_vector_store.delete_by_document_id.side_effect = VectorStoreCollectionNotFoundError(
            "Collection 'nonexistent_collection' does not exist."
        )

        res = self.service.reconcile_failed_ingestion(dry_run=False, min_age_seconds=60)

        self.assertEqual(res.failure_count, 1)
        self.assertEqual(res.success_count, 0)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.DELETION_FAILED)
        self.assertIn("nonexistent_collection", outcome.error_message)

    # -------------------------------------------------------------------------
    # 12. Factory helper function instantiation
    # -------------------------------------------------------------------------
    def test_get_reconciliation_service_factory(self):
        """Verify get_reconciliation_service properly instantiates service with provided dependencies."""
        svc = get_reconciliation_service(
            registry_repo=self.repo,
            vector_store=self.mock_vector_store,
            settings=self.settings,
        )
        self.assertIsInstance(svc, ReconciliationService)
        self.assertEqual(svc.registry_repo, self.repo)
        self.assertEqual(svc.vector_store, self.mock_vector_store)

    # -------------------------------------------------------------------------
    # 13. Step 21H-K-B: Post-claim ineligibility triggers safe release
    # -------------------------------------------------------------------------
    def test_post_claim_unexpected_ineligibility_triggers_safe_release(self):
        """Verify that if a claimed record is unexpectedly ineligible post-claim, claim is released and Qdrant is not touched."""
        rec = self.repo.create_record("post_claim_ineligible.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        fake_token = "rec_defensive_test_token"
        # Simulate a race where claim_for_reconciliation returns a record whose previous_status is 'deletion_failed'
        ineligible_claimed_record = DocumentRecord(
            document_id=rec.document_id,
            document_name="post_claim_ineligible.pdf",
            status=DocumentStatus.RECONCILING,
            collection_name="test_enterprise_collection",
            created_at=rec.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            provenance_eligible=True,
            claim_token=fake_token,
            previous_status=DocumentStatus.DELETION_FAILED.value,  # Unexpected non-failed status
        )

        with patch.object(self.repo, "claim_for_reconciliation", return_value=ineligible_claimed_record):
            with patch.object(self.repo, "release_reconciliation_claim") as mock_release:
                res = self.service.reconcile_failed_ingestion(
                    dry_run=False,
                    target_document_id=rec.document_id,
                )

                # Release called with exact document ID and claim token
                mock_release.assert_called_once_with(
                    document_id=rec.document_id,
                    claim_token=fake_token,
                    error_code="INELIGIBLE_RECONCILIATION",
                    error_message="Record ineligible for failed ingestion cleanup.",
                )

        # Zero Qdrant calls
        self.mock_vector_store.count_by_document_id.assert_not_called()
        self.mock_vector_store.delete_by_document_id.assert_not_called()

        self.assertEqual(res.skipped_count, 1)
        self.assertEqual(res.success_count, 0)
        outcome = res.outcomes[0]
        self.assertEqual(outcome.status, ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE)
        self.assertEqual(outcome.error_code, "INELIGIBLE_POST_CLAIM")

    # -------------------------------------------------------------------------
    # 14. Step 21H-K-B: Release failure during post-claim ineligibility does not report success
    # -------------------------------------------------------------------------
    def test_post_claim_ineligibility_release_failure_does_not_report_success(self):
        """Verify that if releasing an ineligible claim encounters an error, it does not become a success."""
        rec = self.repo.create_record("release_fail.pdf", "test_enterprise_collection", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        fake_token = "rec_failing_release_token"
        ineligible_claimed_record = DocumentRecord(
            document_id=rec.document_id,
            document_name="release_fail.pdf",
            status=DocumentStatus.RECONCILING,
            collection_name="test_enterprise_collection",
            created_at=rec.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            provenance_eligible=False,  # Unexpectedly ineligible provenance
            claim_token=fake_token,
            previous_status=DocumentStatus.FAILED.value,
        )

        with patch.object(self.repo, "claim_for_reconciliation", return_value=ineligible_claimed_record):
            with patch.object(
                self.repo,
                "release_reconciliation_claim",
                side_effect=DocumentStateConflictError("Simulated release conflict"),
            ) as mock_release:
                res = self.service.reconcile_failed_ingestion(
                    dry_run=False,
                    target_document_id=rec.document_id,
                )
                mock_release.assert_called_once()

        self.mock_vector_store.delete_by_document_id.assert_not_called()
        self.assertEqual(res.success_count, 0)
        self.assertEqual(res.skipped_count, 1)
        self.assertEqual(res.outcomes[0].status, ReconciliationOutcomeStatus.SKIPPED_INELIGIBLE)


if __name__ == "__main__":
    unittest.main()
