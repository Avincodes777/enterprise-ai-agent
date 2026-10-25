"""Unit tests for the persistent SQLite DocumentRegistryRepository."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch



# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.api.dependencies import (
    get_document_registry_dependency,
    get_shared_document_registry,
    reset_document_registry_cache,
)
from app.config import Settings
from app.database import (
    DEFAULT_INGESTION_LEASE_SECONDS,
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


class TestDocumentRegistryRepository(unittest.TestCase):
    """Unit test suite for SQLite DocumentRegistryRepository persistence and lifecycle."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_registry.db")
        self.repo = DocumentRegistryRepository(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()
        reset_document_registry_cache()

    # -------------------------------------------------------------------------
    # 1. Creating a processing record
    # -------------------------------------------------------------------------
    def test_create_processing_record(self):
        """Verify new document is created with status 'processing' and valid timestamps."""
        record = self.repo.create_record(
            document_name="architecture_overview.pdf",
            collection_name="enterprise_knowledge",
        )
        self.assertIsInstance(record, DocumentRecord)
        self.assertTrue(record.document_id.startswith("doc_"))
        self.assertEqual(record.document_name, "architecture_overview.pdf")
        self.assertEqual(record.status, DocumentStatus.PROCESSING)
        self.assertEqual(record.collection_name, "enterprise_knowledge")
        self.assertIsNotNone(record.created_at)
        self.assertIsNotNone(record.updated_at)
        self.assertIsNone(record.pages_processed)
        self.assertIsNone(record.chunks_created)
        self.assertIsNone(record.chunks_stored)
        self.assertIsNone(record.error_code)
        self.assertIsNone(record.error_message)

    # -------------------------------------------------------------------------
    # 2. Persisting records across repository/database reconnection
    # -------------------------------------------------------------------------
    def test_persistence_across_reconnection(self):
        """Verify data persists when repository instance is discarded and re-opened."""
        record1 = self.repo.create_record(
            document_name="policy_v1.pdf",
            collection_name="enterprise_knowledge",
        )
        doc_id = record1.document_id

        # Update to succeeded
        self.repo.update_status_succeeded(
            document_id=doc_id,
            pages_processed=10,
            chunks_created=25,
            chunks_stored=25,
        )

        # Discard repository instance and instantiate a new repository pointing to the same file
        reconnected_repo = DocumentRegistryRepository(db_path=self.db_path)
        persisted_record = reconnected_repo.get_record(doc_id)

        self.assertIsNotNone(persisted_record)
        self.assertEqual(persisted_record.document_id, doc_id)
        self.assertEqual(persisted_record.document_name, "policy_v1.pdf")
        self.assertEqual(persisted_record.status, DocumentStatus.SUCCEEDED)
        self.assertEqual(persisted_record.pages_processed, 10)
        self.assertEqual(persisted_record.chunks_created, 25)
        self.assertEqual(persisted_record.chunks_stored, 25)

    # -------------------------------------------------------------------------
    # 3. Updating record to succeeded
    # -------------------------------------------------------------------------
    def test_update_status_succeeded(self):
        """Verify status transitions from processing to succeeded with metrics."""
        record = self.repo.create_record(
            document_name="handbook.pdf",
            collection_name="enterprise_knowledge",
        )
        updated = self.repo.update_status_succeeded(
            document_id=record.document_id,
            pages_processed=4,
            chunks_created=16,
            chunks_stored=16,
            collection_name="custom_knowledge",
        )
        self.assertEqual(updated.status, DocumentStatus.SUCCEEDED)
        self.assertEqual(updated.pages_processed, 4)
        self.assertEqual(updated.chunks_created, 16)
        self.assertEqual(updated.chunks_stored, 16)
        self.assertEqual(updated.collection_name, "custom_knowledge")
        self.assertIsNone(updated.error_code)
        self.assertIsNone(updated.error_message)

    # -------------------------------------------------------------------------
    # 4. Updating record to failed
    # -------------------------------------------------------------------------
    def test_update_status_failed(self):
        """Verify status transitions to failed with error categorization and sanitized message."""
        record = self.repo.create_record(
            document_name="corrupt.pdf",
            collection_name="enterprise_knowledge",
        )
        failed = self.repo.update_status_failed(
            document_id=record.document_id,
            error_code="MALFORMED_PDF",
            error_message="Failed to parse PDF document header.",
        )
        self.assertEqual(failed.status, DocumentStatus.FAILED)
        self.assertEqual(failed.error_code, "MALFORMED_PDF")
        self.assertEqual(failed.error_message, "Failed to parse PDF document header.")
        self.assertIsNone(failed.pages_processed)

    # -------------------------------------------------------------------------
    # 5. Unique document IDs for separate ingestion attempts
    # -------------------------------------------------------------------------
    def test_unique_document_ids_generated(self):
        """Verify separate ingestion attempts receive distinct, stable identifiers."""
        r1 = self.repo.create_record("file.pdf", "col1")
        r2 = self.repo.create_record("file.pdf", "col1")
        r3 = self.repo.create_record("file.pdf", "col1")

        self.assertNotEqual(r1.document_id, r2.document_id)
        self.assertNotEqual(r2.document_id, r3.document_id)
        self.assertNotEqual(r1.document_id, r3.document_id)
        self.assertEqual(self.repo.count_records(), 3)

    # -------------------------------------------------------------------------
    # 6. Listing records ordered newest first
    # -------------------------------------------------------------------------
    def test_list_records_ordered_newest_first(self):
        """Verify list_records returns records in descending creation order."""
        r1 = self.repo.create_record("first.pdf", "col")
        r2 = self.repo.create_record("second.pdf", "col")
        r3 = self.repo.create_record("third.pdf", "col")

        records = self.repo.list_records(limit=10, offset=0)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].document_id, r3.document_id)
        self.assertEqual(records[1].document_id, r2.document_id)
        self.assertEqual(records[2].document_id, r1.document_id)

    # -------------------------------------------------------------------------
    # 7. Pagination limit and offset work correctly
    # -------------------------------------------------------------------------
    def test_pagination_limit_and_offset(self):
        """Verify pagination limits and offsets accurately slice result sets."""
        for i in range(5):
            self.repo.create_record(f"doc_{i}.pdf", "col")

        page1 = self.repo.list_records(limit=2, offset=0)
        self.assertEqual(len(page1), 2)

        page2 = self.repo.list_records(limit=2, offset=2)
        self.assertEqual(len(page2), 2)
        self.assertNotEqual(page1[0].document_id, page2[0].document_id)

        page3 = self.repo.list_records(limit=2, offset=4)
        self.assertEqual(len(page3), 1)

    # -------------------------------------------------------------------------
    # 8. Empty registries return empty list and 0 count
    # -------------------------------------------------------------------------
    def test_empty_registry_returns_empty_list_and_zero_count(self):
        """Verify empty database returns empty list and 0 count without error."""
        self.assertEqual(self.repo.count_records(), 0)
        self.assertEqual(self.repo.list_records(), [])
        self.assertIsNone(self.repo.get_record("nonexistent_id"))

    # -------------------------------------------------------------------------
    # 9. Updating non-existent document raises DocumentNotFoundError
    # -------------------------------------------------------------------------
    def test_updating_nonexistent_document_raises_not_found(self):
        """Verify update operations on absent document IDs raise DocumentNotFoundError."""
        with self.assertRaises(DocumentNotFoundError):
            self.repo.update_status_succeeded("ghost_doc_id", 1, 1, 1)

        with self.assertRaises(DocumentNotFoundError):
            self.repo.update_status_failed("ghost_doc_id", "ERR", "msg")

    # -------------------------------------------------------------------------
    # 10. In-memory connection support
    # -------------------------------------------------------------------------
    def test_in_memory_connection_support(self):
        """Verify repository operates cleanly when provided an existing connection."""
        mem_conn = sqlite3.connect(":memory:")
        repo = DocumentRegistryRepository(connection=mem_conn)
        rec = repo.create_record("in_memory.pdf", "col")
        self.assertEqual(repo.count_records(), 1)
        self.assertEqual(repo.get_record(rec.document_id).document_name, "in_memory.pdf")
        mem_conn.close()

    # -------------------------------------------------------------------------
    # 11. Dependency and cache lifecycle
    # -------------------------------------------------------------------------
    def test_dependency_and_cache_lifecycle(self):
        """Verify get_shared_document_registry caches singleton and reset clears it."""
        reset_document_registry_cache()
        reg1 = get_shared_document_registry()
        reg2 = get_shared_document_registry()
        self.assertIs(reg1, reg2)

        reset_document_registry_cache()
        reg3 = get_shared_document_registry()
        self.assertIsNot(reg1, reg3)
        reset_document_registry_cache()

    # -------------------------------------------------------------------------
    # 12. Provenance eligibility flag behavior
    # -------------------------------------------------------------------------
    def test_provenance_eligible_flag_behavior(self):
        """Verify records default to provenance_eligible=False unless explicitly set."""
        r_default = self.repo.create_record("default.pdf", "col")
        self.assertFalse(r_default.provenance_eligible)

        r_eligible = self.repo.create_record(
            "eligible.pdf", "col", provenance_eligible=True
        )
        self.assertTrue(r_eligible.provenance_eligible)

    # -------------------------------------------------------------------------
    # 13. Idempotent migration for legacy database tables
    # -------------------------------------------------------------------------
    def test_legacy_database_migration_defaults_to_false(self):
        """Verify pre-existing tables without provenance_eligible receive column with default 0."""
        legacy_db = os.path.join(self.temp_dir.name, "legacy.db")
        conn = sqlite3.connect(legacy_db)
        # Create table with old schema (pre-Step 21D)
        conn.executescript("""
        CREATE TABLE document_registry (
            document_id TEXT PRIMARY KEY,
            document_name TEXT NOT NULL,
            status TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pages_processed INTEGER,
            chunks_created INTEGER,
            chunks_stored INTEGER,
            error_code TEXT,
            error_message TEXT
        );
        INSERT INTO document_registry VALUES (
            'legacy_doc_1', 'old_file.pdf', 'succeeded', 'knowledge',
            '2026-01-01T00:00:00', '2026-01-01T00:00:00', 5, 10, 10, NULL, NULL
        );
        """)
        conn.commit()
        conn.close()

        # Connect with repository which runs _ensure_tables migration
        legacy_repo = DocumentRegistryRepository(db_path=legacy_db)
        record = legacy_repo.get_record("legacy_doc_1")
        self.assertIsNotNone(record)
        self.assertFalse(record.provenance_eligible)
        self.assertEqual(record.status, DocumentStatus.SUCCEEDED)

    # -------------------------------------------------------------------------
    # 14. Claim for deletion lifecycle and concurrency guards
    # -------------------------------------------------------------------------
    def test_claim_for_deletion_lifecycle_and_guards(self):
        """Verify atomic claiming rules: eligible succeeded/failed can be claimed; processing, legacy, or active deleting cannot."""
        # Case A: Legacy record cannot be claimed
        r_legacy = self.repo.create_record("legacy.pdf", "col", provenance_eligible=False)
        self.repo.update_status_succeeded(r_legacy.document_id, 1, 1, 1)
        self.assertIsNone(self.repo.claim_for_deletion(r_legacy.document_id))

        # Case B: Processing record cannot be claimed
        r_proc = self.repo.create_record("proc.pdf", "col", provenance_eligible=True)
        self.assertIsNone(self.repo.claim_for_deletion(r_proc.document_id))

        # Case C: Eligible succeeded record CAN be claimed
        r_succ = self.repo.create_record("succ.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(r_succ.document_id, 2, 4, 4)
        claimed = self.repo.claim_for_deletion(r_succ.document_id)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, DocumentStatus.DELETING)

        # Case D: Already claimed record CANNOT be claimed concurrently by another request
        second_claim = self.repo.claim_for_deletion(r_succ.document_id)
        self.assertIsNone(second_claim)

        # Case E: Eligible failed record CAN be claimed
        r_fail = self.repo.create_record("fail.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(r_fail.document_id, "ERR", "msg")
        claimed_fail = self.repo.claim_for_deletion(r_fail.document_id)
        self.assertIsNotNone(claimed_fail)
        self.assertEqual(claimed_fail.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 15. Delete record requires expected deletion status
    # -------------------------------------------------------------------------
    def test_delete_record_expected_status_guard(self):
        """Verify delete_record only deletes rows that are in DELETING status."""
        rec = self.repo.create_record("guard.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        # Cannot delete directly when status is SUCCEEDED
        self.assertFalse(self.repo.delete_record(rec.document_id, expected_status=DocumentStatus.DELETING))
        self.assertIsNotNone(self.repo.get_record(rec.document_id))

        # Claim into DELETING
        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(claimed)

        # Now delete succeeds
        self.assertTrue(self.repo.delete_record(rec.document_id, expected_status=DocumentStatus.DELETING))
        self.assertIsNone(self.repo.get_record(rec.document_id))

    # -------------------------------------------------------------------------
    # 16. Update status deletion failed enables safe retry
    # -------------------------------------------------------------------------
    def test_update_status_deletion_failed_allows_retry(self):
        """Verify failed deletion transitions to DELETION_FAILED and can be re-claimed."""
        rec = self.repo.create_record("retry.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        # Claim for deletion
        self.repo.claim_for_deletion(rec.document_id)

        # Simulate vector store failure -> update to deletion_failed
        failed_rec = self.repo.update_status_deletion_failed(
            document_id=rec.document_id,
            error_code="VECTOR_STORE_UNAVAILABLE",
            error_message="Qdrant connection refused",
        )
        self.assertIsNotNone(failed_rec)
        self.assertEqual(failed_rec.status, DocumentStatus.DELETION_FAILED)
        self.assertEqual(failed_rec.error_code, "VECTOR_STORE_UNAVAILABLE")

        # Retry: can be re-claimed
        re_claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(re_claimed)
        self.assertEqual(re_claimed.status, DocumentStatus.DELETING)


    # -------------------------------------------------------------------------
    # 17. Abandoned deletion claim lease recovery
    # -------------------------------------------------------------------------
    def test_stale_deleting_claim_can_be_recovered(self):
        """Verify an abandoned record in 'deleting' status older than the lease timeout can be reclaimed."""
        rec = self.repo.create_record("abandoned.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        # Claim into deleting
        self.repo.claim_for_deletion(rec.document_id)

        # Fresh claim is rejected with default timeout
        self.assertIsNone(self.repo.claim_for_deletion(rec.document_id))

        # Manually backdate updated_at to simulate an abandoned claim (e.g. 10 minutes ago)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        # Stale claim can now be recovered with default 300s timeout
        recovered = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, DocumentStatus.DELETING)

        # Immediately following claim is fresh again and must be rejected
        immediate_second = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
        self.assertIsNone(immediate_second)

    # -------------------------------------------------------------------------
    # 18. Stale claim with provenance_eligible=False is rejected
    # -------------------------------------------------------------------------
    def test_stale_deleting_claim_with_provenance_ineligible_rejected(self):
        """Verify an abandoned claim with provenance_eligible=0 is rejected even when stale."""
        rec = self.repo.create_record("legacy_abandoned.pdf", "col", provenance_eligible=False)

        # Manually force into deleting with stale updated_at
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        # Reclaim must be rejected because provenance_eligible == 0
        self.assertIsNone(self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0))

    # -------------------------------------------------------------------------
    # 19. Stale claim with processing status is rejected
    # -------------------------------------------------------------------------
    def test_stale_processing_record_cannot_be_claimed_for_deletion(self):
        """Verify that a document in 'processing' status is never claimed, even if updated_at is stale."""
        rec = self.repo.create_record("stale_proc.pdf", "col", provenance_eligible=True)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        self.assertIsNone(self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0))

    # -------------------------------------------------------------------------
    # 20. Concurrent claims on same document: exactly one succeeds
    # -------------------------------------------------------------------------
    def test_concurrent_claim_attempts_atomic_single_winner(self):
        """Verify that concurrent claim attempts on the same document result in exactly one winner."""
        rec = self.repo.create_record("race.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        import threading

        results = []

        def worker():
            res = self.repo.claim_for_deletion(rec.document_id)
            results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful_claims = [r for r in results if r is not None]
        self.assertEqual(len(successful_claims), 1)
        self.assertEqual(successful_claims[0].status, DocumentStatus.DELETING)


    # -------------------------------------------------------------------------
    # 21. Deletion lease boundary: 299 seconds old cannot be reclaimed
    # -------------------------------------------------------------------------
    def test_deletion_lease_boundary_299_seconds_rejected(self):
        """Verify that a deleting record approximately 299 seconds old cannot be reclaimed."""
        rec = self.repo.create_record("lease_299.pdf", "col", provenance_eligible=True)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            # 299 seconds old (inside lease window)
            t_299 = (fixed_now - timedelta(seconds=299)).isoformat()
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_299, rec.document_id),
                )

            # Must be rejected because 299s < 300s timeout
            claimed = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
            self.assertIsNone(claimed)

    # -------------------------------------------------------------------------
    # 22. Deletion lease boundary: exactly 300 seconds old cannot be reclaimed
    # -------------------------------------------------------------------------
    def test_deletion_lease_boundary_exact_300_seconds_rejected(self):
        """Verify that a deleting record exactly 300 seconds old cannot be reclaimed under strict comparison."""
        rec = self.repo.create_record("lease_300.pdf", "col", provenance_eligible=True)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            # Exactly 300 seconds old (on the boundary)
            t_300 = (fixed_now - timedelta(seconds=300)).isoformat()
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_300, rec.document_id),
                )

            # Strict '<' comparison means datetime(updated_at) == datetime(cutoff), evaluating to False
            claimed = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
            self.assertIsNone(claimed)

    # -------------------------------------------------------------------------
    # 23. Deletion lease boundary: 301 seconds old can be reclaimed
    # -------------------------------------------------------------------------
    def test_deletion_lease_boundary_301_seconds_reclaimed(self):
        """Verify that a deleting record approximately 301 seconds old can be reclaimed."""
        rec = self.repo.create_record("lease_301.pdf", "col", provenance_eligible=True)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            # 301 seconds old (expired past 300s lease window)
            t_301 = (fixed_now - timedelta(seconds=301)).isoformat()
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_301, rec.document_id),
                )

            # Expired lease is successfully claimed
            claimed = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 24. Deletion lease: UTC timestamp ending in 'Z' format
    # -------------------------------------------------------------------------
    def test_deletion_lease_timestamp_trailing_z_format(self):
        """Verify that timestamps formatted with trailing 'Z' are normalized and compared correctly."""
        rec = self.repo.create_record("lease_z.pdf", "col", provenance_eligible=True)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now

            # Fresh claim with trailing Z (299s old) -> rejected
            t_z_fresh = (fixed_now - timedelta(seconds=299)).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_z_fresh, rec.document_id),
                )
            self.assertIsNone(self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0))

            # Stale claim with trailing Z (301s old) -> reclaimed
            t_z_stale = (fixed_now - timedelta(seconds=301)).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_z_stale, rec.document_id),
                )
            claimed = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 25. Deletion lease: legacy timestamp without explicit timezone offset
    # -------------------------------------------------------------------------
    def test_deletion_lease_legacy_timestamp_no_offset(self):
        """Verify that legacy timestamps without timezone offset behave consistently as UTC."""
        rec = self.repo.create_record("lease_no_offset.pdf", "col", provenance_eligible=True)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now

            # Fresh claim without offset (299s old) -> rejected
            t_no_tz_fresh = (fixed_now - timedelta(seconds=299)).strftime("%Y-%m-%d %H:%M:%S")
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_no_tz_fresh, rec.document_id),
                )
            self.assertIsNone(self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0))

            # Stale claim without offset (301s old) -> reclaimed
            t_no_tz_stale = (fixed_now - timedelta(seconds=301)).strftime("%Y-%m-%d %H:%M:%S")
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_no_tz_stale, rec.document_id),
                )
            claimed = self.repo.claim_for_deletion(rec.document_id, claim_timeout_seconds=300.0)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 26. Boundary recovery strictly enforces provenance_eligible=1
    # -------------------------------------------------------------------------
    def test_deletion_lease_boundary_requires_provenance_eligible(self):
        """Verify that expired boundary claims on provenance_eligible=0 records are rejected."""
        rec = self.repo.create_record("lease_unverified.pdf", "col", provenance_eligible=False)
        fixed_now = datetime(2026, 10, 21, 12, 10, 0, 0, tzinfo=timezone.utc)

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now

            # 301 seconds old, but provenance_eligible is False
            t_301 = (fixed_now - timedelta(seconds=301)).isoformat()
            with self.repo._get_connection() as conn:
                conn.execute(
                    "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                    (t_301, rec.document_id),
                )

            # Rejected despite lease expiration
    # -------------------------------------------------------------------------
    # 27. Step 21G-B: Success finalization rejected when status is not processing
    # -------------------------------------------------------------------------
    def test_update_status_succeeded_rejected_when_not_processing(self):
        """Verify update_status_succeeded raises DocumentStateConflictError if status is not processing."""
        rec = self.repo.create_record("conflict_doc.pdf", "col")
        # Move document to failed
        self.repo.update_status_failed(rec.document_id, "ERROR_CODE", "Something failed")
        rec_failed = self.repo.get_record(rec.document_id)
        self.assertEqual(rec_failed.status, DocumentStatus.FAILED)

        # Attempt to transition already-failed document to succeeded
        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.update_status_succeeded(
                document_id=rec.document_id,
                pages_processed=1,
                chunks_created=2,
                chunks_stored=2,
            )
        self.assertIn("failed", str(ctx.exception))
        self.assertIn("expected 'processing'", str(ctx.exception))

        # Status must remain FAILED and not overwritten
        rec_check = self.repo.get_record(rec.document_id)
        self.assertEqual(rec_check.status, DocumentStatus.FAILED)
        self.assertEqual(rec_check.error_code, "ERROR_CODE")

    # -------------------------------------------------------------------------
    # 28. Step 21G-B: Failure finalization cannot overwrite a newer state
    # -------------------------------------------------------------------------
    def test_update_status_failed_cannot_overwrite_newer_state(self):
        """Verify update_status_failed raises DocumentStateConflictError if status is not processing."""
        rec = self.repo.create_record("success_first.pdf", "col")
        # Ingestion finishes normally
        self.repo.update_status_succeeded(
            document_id=rec.document_id,
            pages_processed=3,
            chunks_created=9,
            chunks_stored=9,
        )
        rec_succeeded = self.repo.get_record(rec.document_id)
        self.assertEqual(rec_succeeded.status, DocumentStatus.SUCCEEDED)

        # Late or erroneous failure update arrives
        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.update_status_failed(
                document_id=rec.document_id,
                error_code="LATE_ERROR",
                error_message="Worker thought it timed out",
            )
        self.assertIn("succeeded", str(ctx.exception))
        self.assertIn("expected 'processing'", str(ctx.exception))

        # Record must remain SUCCEEDED with chunks intact
        rec_after = self.repo.get_record(rec.document_id)
        self.assertEqual(rec_after.status, DocumentStatus.SUCCEEDED)
        self.assertEqual(rec_after.chunks_stored, 9)
        self.assertIsNone(rec_after.error_code)

    # -------------------------------------------------------------------------
    # 29. Step 21G-B: Conflicting or repeated finalizations cannot both succeed
    # -------------------------------------------------------------------------
    def test_conflicting_or_repeated_finalizations_cannot_both_succeed(self):
        """Verify that two conflicting finalization attempts cannot both perform state transitions."""
        rec = self.repo.create_record("race_doc.pdf", "col")
        self.assertEqual(rec.status, DocumentStatus.PROCESSING)

        # First finalization succeeds
        first_success = self.repo.update_status_succeeded(
            document_id=rec.document_id,
            pages_processed=2,
            chunks_created=5,
            chunks_stored=5,
        )
        self.assertEqual(first_success.status, DocumentStatus.SUCCEEDED)

        # Repeated finalization raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError):
            self.repo.update_status_succeeded(
                document_id=rec.document_id,
                pages_processed=2,
                chunks_created=5,
                chunks_stored=5,
            )

        # Failure update also raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError):
            self.repo.update_status_failed(
                document_id=rec.document_id,
                error_code="FAIL_ATTEMPT",
                error_message="Late fail",
            )

        # Final record state is unchanged
        final_rec = self.repo.get_record(rec.document_id)
        self.assertEqual(final_rec.status, DocumentStatus.SUCCEEDED)

    # -------------------------------------------------------------------------
    # 30. Step 21H-B: Stale, provenance-eligible processing record can be claimed
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_stale_record_succeeds(self):
        """Verify an abandoned record in 'processing' status older than lease is transitioned to 'failed'."""
        rec = self.repo.create_record("stale_proc.pdf", "col", provenance_eligible=True)
        self.assertEqual(rec.status, DocumentStatus.PROCESSING)

        # Backdate updated_at past the 900s default lease window
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        claimed = self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=DEFAULT_INGESTION_LEASE_SECONDS)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.document_id, rec.document_id)
        self.assertEqual(claimed.status, DocumentStatus.FAILED)
        self.assertEqual(claimed.error_code, "INGESTION_ABANDONED")
        self.assertIn("timed out and was reclaimed", claimed.error_message)
        self.assertGreater(claimed.updated_at, stale_time)

        # Persistence check
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertEqual(persisted.error_code, "INGESTION_ABANDONED")

    # -------------------------------------------------------------------------
    # 31. Step 21H-B: Fresh processing record cannot be claimed
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_fresh_record_rejected(self):
        """Verify a newly created 'processing' record within the lease window cannot be claimed."""
        rec = self.repo.create_record("fresh_proc.pdf", "col", provenance_eligible=True)

        claimed = self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=900.0)
        self.assertIsNone(claimed)

        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.PROCESSING)

    # -------------------------------------------------------------------------
    # 32. Step 21H-B: Record exactly at lease boundary is rejected
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_exact_boundary_rejected(self):
        """Verify that a record at the exact cutoff boundary is not claimed (strictly older required)."""
        rec = self.repo.create_record("boundary_proc.pdf", "col", provenance_eligible=True)

        now_utc = datetime.now(timezone.utc)
        boundary_iso = (now_utc - timedelta(seconds=900.0)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (boundary_iso, rec.document_id),
            )

        # Patch datetime.now to return exact now_utc so cutoff equals boundary_iso exactly
        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = now_utc
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            claimed = self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=900.0)
            self.assertIsNone(claimed)

        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.PROCESSING)

    # -------------------------------------------------------------------------
    # 33. Step 21H-B: Stale record with provenance_eligible=False cannot be claimed
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_provenance_ineligible_rejected(self):
        """Verify an abandoned record with provenance_eligible=0 cannot be claimed."""
        rec = self.repo.create_record("legacy_proc.pdf", "col", provenance_eligible=False)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        claimed = self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=900.0)
        self.assertIsNone(claimed)

        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.PROCESSING)

    # -------------------------------------------------------------------------
    # 34. Step 21H-B: Non-processing records cannot be claimed
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_non_processing_status_rejected(self):
        """Verify that records in succeeded, failed, or deleting status cannot be claimed even if stale."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()

        # 1. Succeeded record
        rec_succ = self.repo.create_record("succ.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_succ.document_id, 1, 1, 1)
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_succ.document_id))
        self.assertIsNone(self.repo.claim_abandoned_ingestion(rec_succ.document_id, lease_seconds=900.0))

        # 2. Failed record
        rec_fail = self.repo.create_record("fail.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_fail.document_id, "PREV_ERR", "Msg")
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_fail.document_id))
        self.assertIsNone(self.repo.claim_abandoned_ingestion(rec_fail.document_id, lease_seconds=900.0))

        # 3. Deleting record
        rec_del = self.repo.create_record("del.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_del.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec_del.document_id)
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_del.document_id))
        self.assertIsNone(self.repo.claim_abandoned_ingestion(rec_del.document_id, lease_seconds=900.0))

    # -------------------------------------------------------------------------
    # 35. Step 21H-B: Repeated and concurrent claims cannot both succeed
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_repeated_and_concurrent_claims(self):
        """Verify that an abandoned record cannot be claimed multiple times or concurrently."""
        rec = self.repo.create_record("concurrent_claim.pdf", "col", provenance_eligible=True)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=1200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        import threading

        results = []

        def worker():
            res = self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=900.0)
            results.append(res)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread succeeds with a DocumentRecord; all others return None
        successful_claims = [r for r in results if r is not None]
        rejected_claims = [r for r in results if r is None]

        self.assertEqual(len(successful_claims), 1)
        self.assertEqual(len(rejected_claims), 4)

        # Immediately following sequential claim also returns None
        self.assertIsNone(self.repo.claim_abandoned_ingestion(rec.document_id, lease_seconds=900.0))

    # -------------------------------------------------------------------------
    # 36. Step 21H-B: Invalid arguments rejected for claim_abandoned_ingestion
    # -------------------------------------------------------------------------
    def test_claim_abandoned_ingestion_invalid_arguments_rejected(self):
        """Verify invalid document IDs or non-positive lease durations raise DocumentRegistryError."""
        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_abandoned_ingestion("")

        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_abandoned_ingestion("   ")

        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_abandoned_ingestion("doc_1", lease_seconds=0)

        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_abandoned_ingestion("doc_1", lease_seconds=-5.0)

        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_abandoned_ingestion("doc_1", lease_seconds=None)  # type: ignore[arg-type]

        # Non-existent document returns None
        self.assertIsNone(self.repo.claim_abandoned_ingestion("doc_nonexistent_123", lease_seconds=900.0))

    # -------------------------------------------------------------------------
    # 37. Step 21H-B: update_status_deletion_failed guarded by deleting status
    # -------------------------------------------------------------------------
    def test_update_status_deletion_failed_requires_deleting_status(self):
        """Verify update_status_deletion_failed rejects records not in 'deleting' and never overwrites state."""
        rec = self.repo.create_record("not_deleting.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 2, 4, 4)

        # Document is SUCCEEDED, not DELETING -> raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.update_status_deletion_failed(
                document_id=rec.document_id,
                error_code="DEL_ERR",
                error_message="Should fail",
            )
        self.assertIn("succeeded", str(ctx.exception))
        self.assertIn("expected 'deleting'", str(ctx.exception))

        # Record must remain SUCCEEDED
        rec_after = self.repo.get_record(rec.document_id)
        self.assertEqual(rec_after.status, DocumentStatus.SUCCEEDED)

    # -------------------------------------------------------------------------
    # 38. Step 21H-B: update_status_deletion_failed missing document raises DocumentNotFoundError
    # -------------------------------------------------------------------------
    def test_update_status_deletion_failed_missing_or_invalid_id(self):
        """Verify update_status_deletion_failed raises DocumentNotFoundError on missing ID and DocumentRegistryError on invalid ID."""
        with self.assertRaises(DocumentNotFoundError):
            self.repo.update_status_deletion_failed(
                document_id="doc_missing_404",
                error_code="ERR",
                error_message="Msg",
            )

        with self.assertRaises(DocumentRegistryError):
            self.repo.update_status_deletion_failed(
                document_id="",
                error_code="ERR",
                error_message="Msg",
            )

    # -------------------------------------------------------------------------
    # 39. Step 21H-D: heartbeat_ingestion successfully renews updated_at for processing record
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_processing_document_renews_timestamp(self):
        """Verify that heartbeat_ingestion updates updated_at while retaining status and other metadata."""
        rec = self.repo.create_record("heartbeat_proc.pdf", "col", provenance_eligible=True)
        self.assertEqual(rec.status, DocumentStatus.PROCESSING)

        earlier_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (earlier_time, rec.document_id),
            )

        renewed = self.repo.heartbeat_ingestion(rec.document_id)
        self.assertIsNotNone(renewed)
        self.assertEqual(renewed.document_id, rec.document_id)
        self.assertEqual(renewed.status, DocumentStatus.PROCESSING)
        self.assertEqual(renewed.document_name, "heartbeat_proc.pdf")
        self.assertEqual(renewed.collection_name, "col")
        self.assertEqual(renewed.created_at, rec.created_at)
        self.assertTrue(renewed.provenance_eligible)
        self.assertGreater(renewed.updated_at, earlier_time)

        # Confirm persisted record in database matches
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.PROCESSING)
        self.assertEqual(persisted.updated_at, renewed.updated_at)

    # -------------------------------------------------------------------------
    # 40. Step 21H-D: heartbeat_ingestion raises conflict on failed state
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_failed_state_raises_conflict(self):
        """Verify that heartbeat_ingestion raises DocumentStateConflictError if document is in failed state."""
        rec = self.repo.create_record("fail_doc.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "PREV_ERR", "Prev msg")

        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.heartbeat_ingestion(rec.document_id)

        self.assertIn("failed", str(ctx.exception))
        self.assertIn("expected 'processing'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 41. Step 21H-D: heartbeat_ingestion raises conflict on succeeded state
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_succeeded_state_raises_conflict(self):
        """Verify that heartbeat_ingestion raises DocumentStateConflictError if document is in succeeded state."""
        rec = self.repo.create_record("succ_doc.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 3, 5, 5)

        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.heartbeat_ingestion(rec.document_id)

        self.assertIn("succeeded", str(ctx.exception))
        self.assertIn("expected 'processing'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 42. Step 21H-D: heartbeat_ingestion raises conflict on deleting state
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_deleting_state_raises_conflict(self):
        """Verify that heartbeat_ingestion raises DocumentStateConflictError if document is in deleting state."""
        rec = self.repo.create_record("del_doc.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec.document_id)

        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.heartbeat_ingestion(rec.document_id)

        self.assertIn("deleting", str(ctx.exception))
        self.assertIn("expected 'processing'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 43. Step 21H-D: heartbeat_ingestion missing and invalid document IDs
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_missing_and_invalid_id(self):
        """Verify heartbeat_ingestion raises DocumentNotFoundError for missing ID and DocumentRegistryError for invalid ID."""
        with self.assertRaises(DocumentNotFoundError):
            self.repo.heartbeat_ingestion("doc_nonexistent_xyz")

        with self.assertRaises(DocumentRegistryError):
            self.repo.heartbeat_ingestion("")

        with self.assertRaises(DocumentRegistryError):
            self.repo.heartbeat_ingestion("   ")

        with self.assertRaises(DocumentRegistryError):
            self.repo.heartbeat_ingestion(None)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # 44. Step 21H-D: rejected heartbeat does not change document state or timestamp
    # -------------------------------------------------------------------------
    def test_heartbeat_ingestion_rejected_does_not_modify_record(self):
        """Verify that a rejected heartbeat update performs no writes and leaves existing state and timestamp intact."""
        rec = self.repo.create_record("frozen.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ABANDONED_PREV", "Frozen message")

        before = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(before)

        with self.assertRaises(DocumentStateConflictError):
            self.repo.heartbeat_ingestion(rec.document_id)

        after = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(after)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.updated_at, before.updated_at)
        self.assertEqual(after.error_code, before.error_code)
        self.assertEqual(after.error_message, before.error_message)

    # -------------------------------------------------------------------------
    # 45. Step 21H-G: Old failed, provenance-eligible records included
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_includes_old_failed_provenance_eligible(self):
        """Verify old failed, provenance-eligible records are returned as reconciliation candidates."""
        rec = self.repo.create_record("failed_cand.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "INGESTION_ERROR", "Worker crashed")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].document_id, rec.document_id)
        self.assertEqual(candidates[0].status, DocumentStatus.FAILED)
        self.assertTrue(candidates[0].provenance_eligible)

    # -------------------------------------------------------------------------
    # 46. Step 21H-G: Old deletion_failed, provenance-eligible records included
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_includes_old_deletion_failed_provenance_eligible(self):
        """Verify old deletion_failed, provenance-eligible records are returned as reconciliation candidates."""
        rec = self.repo.create_record("del_failed_cand.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec.document_id)
        self.repo.update_status_deletion_failed(rec.document_id, "DEL_FAIL", "Qdrant error")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].document_id, rec.document_id)
        self.assertEqual(candidates[0].status, DocumentStatus.DELETION_FAILED)
        self.assertTrue(candidates[0].provenance_eligible)

    # -------------------------------------------------------------------------
    # 47. Step 21H-G: Active and succeeded records are excluded
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_excludes_active_and_succeeded(self):
        """Verify processing, succeeded, and deleting records are strictly excluded even if old."""
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=500)).isoformat()

        # 1. Processing
        rec_proc = self.repo.create_record("proc.pdf", "col", provenance_eligible=True)
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_proc.document_id))

        # 2. Succeeded
        rec_succ = self.repo.create_record("succ.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_succ.document_id, 2, 2, 2)
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_succ.document_id))

        # 3. Deleting
        rec_del = self.repo.create_record("del.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_del.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec_del.document_id)
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec_del.document_id))

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 0)

    # -------------------------------------------------------------------------
    # 48. Step 21H-G: Ineligible provenance records are excluded
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_excludes_ineligible_provenance(self):
        """Verify legacy/ineligible records (provenance_eligible = 0) are excluded even if old and failed."""
        rec = self.repo.create_record("legacy_failed.pdf", "col", provenance_eligible=False)
        self.repo.update_status_failed(rec.document_id, "OLD_ERR", "Legacy failure")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=500)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 0)

    # -------------------------------------------------------------------------
    # 49. Step 21H-G: Records younger than grace period are excluded
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_excludes_young_records(self):
        """Verify failed records younger than min_age_seconds are not included."""
        rec = self.repo.create_record("recent_failed.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Recent failure")

        # updated_at is within the last 30 seconds
        recent_time = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (recent_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 0)

    # -------------------------------------------------------------------------
    # 50. Step 21H-G: Conservative boundary behavior
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_boundary_behavior_conservative(self):
        """Verify records exactly at the cutoff boundary are excluded (strict < check prevents premature inclusion)."""
        base_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

        rec_exact = self.repo.create_record("exact_boundary.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_exact.document_id, "ERR", "Boundary")

        rec_older = self.repo.create_record("older_boundary.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_older.document_id, "ERR", "Older")

        # Set exact cutoff: exactly 120s before base_time
        exact_time = (base_time - timedelta(seconds=120)).isoformat()
        # Set 1 second older than cutoff: 121s before base_time
        older_time = (base_time - timedelta(seconds=121)).isoformat()

        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (exact_time, rec_exact.document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (older_time, rec_older.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)

        # Only the record strictly older than 120s is returned; the exact boundary record is excluded
        candidate_ids = [c.document_id for c in candidates]
        self.assertIn(rec_older.document_id, candidate_ids)
        self.assertNotIn(rec_exact.document_id, candidate_ids)

    # -------------------------------------------------------------------------
    # 51. Step 21H-G: Limit respected and deterministic ordering
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_limit_and_deterministic_ordering(self):
        """Verify limit parameter caps returned count and records are ordered oldest first with document_id tie-breaker."""
        created_records = []
        base_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

        # Create 4 candidates with deterministic timestamps
        for i in range(4):
            rec = self.repo.create_record(f"cand_{i}.pdf", "col", provenance_eligible=True)
            self.repo.update_status_failed(rec.document_id, "ERR", f"Fail {i}")
            created_records.append(rec)

        # Timestamps: rec 0 -> 400s old, rec 1 -> 300s old, rec 2 & 3 -> 200s old (tie)
        t_400 = (base_time - timedelta(seconds=400)).isoformat()
        t_300 = (base_time - timedelta(seconds=300)).isoformat()
        t_200 = (base_time - timedelta(seconds=200)).isoformat()

        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_400, created_records[0].document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_300, created_records[1].document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_200, created_records[2].document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_200, created_records[3].document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Test limit = 2
            cand_limit2 = self.repo.list_reconciliation_candidates(min_age_seconds=120, limit=2)
            self.assertEqual(len(cand_limit2), 2)
            self.assertEqual(cand_limit2[0].document_id, created_records[0].document_id)
            self.assertEqual(cand_limit2[1].document_id, created_records[1].document_id)

            # Test all 4 with tie-breaker
            cand_all = self.repo.list_reconciliation_candidates(min_age_seconds=120, limit=10)
            self.assertEqual(len(cand_all), 4)
            self.assertEqual(cand_all[0].document_id, created_records[0].document_id)
            self.assertEqual(cand_all[1].document_id, created_records[1].document_id)

            # Verify tie-breaker: rec 2 and rec 3 ordered by document_id ASC
            expected_tie = sorted([created_records[2].document_id, created_records[3].document_id])
            self.assertEqual(cand_all[2].document_id, expected_tie[0])
            self.assertEqual(cand_all[3].document_id, expected_tie[1])

    # -------------------------------------------------------------------------
    # 52. Step 21H-G: Invalid arguments rejected
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_invalid_arguments_rejected(self):
        """Verify invalid min_age_seconds and limit arguments raise DocumentRegistryError."""
        # Booleans must be rejected even though bool subclasses int
        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(min_age_seconds=True)  # type: ignore[arg-type]

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(min_age_seconds=False)  # type: ignore[arg-type]

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit=True)  # type: ignore[arg-type]

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit=False)  # type: ignore[arg-type]

        # Negative min_age_seconds
        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(min_age_seconds=-1)

        # NaN / Inf min_age_seconds
        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(min_age_seconds=float("nan"))

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(min_age_seconds=float("inf"))

        # Non-positive or non-integer limit
        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit=0)

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit=-10)

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit=5.5)  # type: ignore[arg-type]

        with self.assertRaises(DocumentRegistryError):
            self.repo.list_reconciliation_candidates(limit="50")  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # 53. Step 21H-G: Read-only guarantee (zero mutation)
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_read_only_zero_mutation(self):
        """Verify calling list_reconciliation_candidates performs zero mutations on registry records."""
        rec = self.repo.create_record("no_mutation.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "PREV_ERR", "Prev message")

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        before = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(before)

        # Call discovery
        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=120)
        self.assertEqual(len(candidates), 1)

        after = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(after)

        # Field-by-field verification of immutability
        self.assertEqual(after.document_id, before.document_id)
        self.assertEqual(after.document_name, before.document_name)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.collection_name, before.collection_name)
        self.assertEqual(after.created_at, before.created_at)
        self.assertEqual(after.updated_at, before.updated_at)
        self.assertEqual(after.pages_processed, before.pages_processed)
        self.assertEqual(after.chunks_created, before.chunks_created)
        self.assertEqual(after.chunks_stored, before.chunks_stored)
        self.assertEqual(after.error_code, before.error_code)
        self.assertEqual(after.error_message, before.error_message)
        self.assertEqual(after.provenance_eligible, before.provenance_eligible)

    # -------------------------------------------------------------------------
    # 54. Step 21H-J: Claim eligible failed record
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_eligible_failed_record(self):
        """Verify claiming an eligible failed record returns reconciling status and unguessable claim token."""
        rec = self.repo.create_record("fail_cand.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "PREV_ERR", "Prev message")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.document_id, rec.document_id)
        self.assertEqual(claimed.status, DocumentStatus.RECONCILING)
        self.assertIsNotNone(claimed.claim_token)
        self.assertTrue(claimed.claim_token.startswith("rec_"))
        self.assertEqual(claimed.previous_status, "failed")

        # Persisted state matches
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.RECONCILING)
        self.assertEqual(persisted.claim_token, claimed.claim_token)
        self.assertEqual(persisted.previous_status, "failed")

    # -------------------------------------------------------------------------
    # 55. Step 21H-J: Claim eligible deletion_failed record
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_eligible_deletion_failed_record(self):
        """Verify claiming an eligible deletion_failed record records pre-claim status properly."""
        rec = self.repo.create_record("del_fail_cand.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec.document_id)
        self.repo.update_status_deletion_failed(rec.document_id, "DEL_ERR", "Qdrant down")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.document_id, rec.document_id)
        self.assertEqual(claimed.status, DocumentStatus.RECONCILING)
        self.assertIsNotNone(claimed.claim_token)
        self.assertEqual(claimed.previous_status, "deletion_failed")

    # -------------------------------------------------------------------------
    # 56. Step 21H-J: Exclude ineligible statuses and unverified provenance
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_excludes_ineligible_records(self):
        """Verify processing, succeeded, deleting, and provenance_eligible=False records cannot be claimed."""
        # 1. Processing
        rec_proc = self.repo.create_record("proc.pdf", "col", provenance_eligible=True)
        self.assertIsNone(self.repo.claim_for_reconciliation(rec_proc.document_id))

        # 2. Succeeded
        rec_succ = self.repo.create_record("succ.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_succ.document_id, 2, 2, 2)
        self.assertIsNone(self.repo.claim_for_reconciliation(rec_succ.document_id))

        # 3. Deleting
        rec_del = self.repo.create_record("del.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec_del.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec_del.document_id)
        self.assertIsNone(self.repo.claim_for_reconciliation(rec_del.document_id))

        # 4. Provenance ineligible failed
        rec_legacy = self.repo.create_record("legacy.pdf", "col", provenance_eligible=False)
        self.repo.update_status_failed(rec_legacy.document_id, "ERR", "Msg")
        self.assertIsNone(self.repo.claim_for_reconciliation(rec_legacy.document_id))

    # -------------------------------------------------------------------------
    # 57. Step 21H-J: Reject invalid arguments (including booleans)
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_invalid_arguments_rejected(self):
        """Verify invalid document_id or lease_seconds arguments raise DocumentRegistryError."""
        # Invalid document IDs
        for invalid_id in ["", "   ", None]:
            with self.subTest(invalid_id=invalid_id):
                with self.assertRaises(DocumentRegistryError):
                    self.repo.claim_for_reconciliation(invalid_id)  # type: ignore[arg-type]

        # Booleans rejected for numeric lease_seconds
        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_for_reconciliation("doc_1", lease_seconds=True)  # type: ignore[arg-type]

        with self.assertRaises(DocumentRegistryError):
            self.repo.claim_for_reconciliation("doc_1", lease_seconds=False)  # type: ignore[arg-type]

        # Non-positive or non-finite lease_seconds
        for invalid_lease in [0, -10, float("nan"), float("inf"), "60"]:
            with self.subTest(invalid_lease=invalid_lease):
                with self.assertRaises(DocumentRegistryError):
                    self.repo.claim_for_reconciliation("doc_1", lease_seconds=invalid_lease)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    # 58. Step 21H-J: Concurrent claimants mutual exclusion
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_concurrent_claimants(self):
        """Verify exactly one concurrent claimant succeeds in obtaining ownership of a fresh claim generation."""
        rec = self.repo.create_record("concurrent_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Msg")

        import threading

        claims = []

        def worker():
            res = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
            claims.append(res)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successful = [c for c in claims if c is not None]
        rejected = [c for c in claims if c is None]
        self.assertEqual(len(successful), 1)
        self.assertEqual(len(rejected), 4)

        # Subsequent sequential attempt also returns None (active lease)
        self.assertIsNone(self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0))

    # -------------------------------------------------------------------------
    # 59. Step 21H-J: Reclaiming an expired reconciliation lease
    # -------------------------------------------------------------------------
    def test_claim_for_reconciliation_reclaims_expired_lease(self):
        """Verify that an expired reconciliation claim can be reclaimed with a new token and preserved previous_status."""
        rec = self.repo.create_record("lease_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Msg")

        first_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(first_claim)
        token1 = first_claim.claim_token

        # Artificially age the claim beyond the 60s lease window
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        # Second claimant reclaims
        second_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(second_claim)
        self.assertEqual(second_claim.status, DocumentStatus.RECONCILING)
        self.assertNotEqual(second_claim.claim_token, token1)
        self.assertEqual(second_claim.previous_status, "failed")

    # -------------------------------------------------------------------------
    # 60. Step 21H-J: Stale owner cannot finalize after claim reclamation
    # -------------------------------------------------------------------------
    def test_stale_owner_cannot_finalize_after_claim_reclamation(self):
        """Verify that a stale claim owner cannot finalize after another claimant reclaims the record."""
        rec = self.repo.create_record("stale_fin.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Msg")

        # Worker 1 claims
        w1_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(w1_claim)
        w1_token = w1_claim.claim_token

        # Lease expires and Worker 2 reclaims
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        w2_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(w2_claim)
        w2_token = w2_claim.claim_token

        # Worker 1 attempts to finalize with old token -> raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.finalize_reconciliation(rec.document_id, claim_token=w1_token, chunks_stored=0)
        self.assertIn("claim token mismatch", str(ctx.exception))

        # Row remains in RECONCILING owned by Worker 2
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.RECONCILING)
        self.assertEqual(persisted.claim_token, w2_token)

    # -------------------------------------------------------------------------
    # 61. Step 21H-J: Stale owner cannot release after claim reclamation
    # -------------------------------------------------------------------------
    def test_stale_owner_cannot_release_after_claim_reclamation(self):
        """Verify that a stale claim owner cannot release after another claimant reclaims the record."""
        rec = self.repo.create_record("stale_rel.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Msg")

        w1_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        w1_token = w1_claim.claim_token

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        w2_claim = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        w2_token = w2_claim.claim_token

        # Worker 1 attempts to release with old token -> raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError) as ctx:
            self.repo.release_reconciliation_claim(
                rec.document_id,
                claim_token=w1_token,
                error_code="STALE_ERR",
                error_message="Should fail",
            )
        self.assertIn("claim token mismatch", str(ctx.exception))

        # Row remains in RECONCILING owned by Worker 2
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.RECONCILING)
        self.assertEqual(persisted.claim_token, w2_token)

    # -------------------------------------------------------------------------
    # 62. Step 21H-J: Current owner finalizes failed-ingestion cleanup retaining row
    # -------------------------------------------------------------------------
    def test_current_owner_finalizes_failed_ingestion_retaining_row(self):
        """Verify current owner finalizes failed ingestion cleanup, returning to 'failed' and clearing claim_token."""
        rec = self.repo.create_record("clean_fin.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "INGESTION_ERROR", "Original error")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        finalized = self.repo.finalize_reconciliation(
            document_id=rec.document_id,
            claim_token=claimed.claim_token,
            chunks_stored=0,
        )

        self.assertIsNotNone(finalized)
        self.assertEqual(finalized.status, DocumentStatus.FAILED)
        self.assertIsNone(finalized.claim_token)
        self.assertEqual(finalized.chunks_stored, 0)

        # Row is verified to still exist in database (not deleted)
        persisted = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, DocumentStatus.FAILED)
        self.assertIsNone(persisted.claim_token)

    # -------------------------------------------------------------------------
    # 63. Step 21H-J: Failure release preserves retryability and status semantics
    # -------------------------------------------------------------------------
    def test_failure_release_preserves_retryability_and_status(self):
        """Verify release_reconciliation_claim restores pre-claim status, updates error, and enables immediate retry."""
        rec = self.repo.create_record("rel_test.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        self.repo.claim_for_deletion(rec.document_id)
        self.repo.update_status_deletion_failed(rec.document_id, "DEL_FAIL", "Qdrant timeout")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        released = self.repo.release_reconciliation_claim(
            document_id=rec.document_id,
            claim_token=claimed.claim_token,
            error_code="RECONCILIATION_FAILED",
            error_message="Vector store connection dropped during sweep",
        )

        self.assertIsNotNone(released)
        self.assertEqual(released.status, DocumentStatus.DELETION_FAILED)
        self.assertIsNone(released.claim_token)
        self.assertEqual(released.error_code, "RECONCILIATION_FAILED")
        self.assertEqual(released.error_message, "Vector store connection dropped during sweep")

        # Immediately retryable: another claim can be taken immediately
        reclaimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.status, DocumentStatus.RECONCILING)

    # -------------------------------------------------------------------------
    # 64. Step 21H-J: Existing registry databases migrate without data loss
    # -------------------------------------------------------------------------
    def test_existing_database_migration_preserves_data(self):
        """Verify opening an un-migrated database adds claim_token and previous_status columns without discarding rows."""
        legacy_dir = tempfile.TemporaryDirectory()
        legacy_db_path = os.path.join(legacy_dir.name, "legacy_registry.db")

        # Manually create legacy schema without claim_token and previous_status
        conn = sqlite3.connect(legacy_db_path)
        conn.execute("""
        CREATE TABLE document_registry (
            document_id TEXT PRIMARY KEY,
            document_name TEXT NOT NULL,
            status TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pages_processed INTEGER,
            chunks_created INTEGER,
            chunks_stored INTEGER,
            error_code TEXT,
            error_message TEXT,
            provenance_eligible INTEGER NOT NULL DEFAULT 0
        );
        """)
        conn.execute("""
        INSERT INTO document_registry (
            document_id, document_name, status, collection_name, created_at, updated_at, provenance_eligible
        ) VALUES (
            'doc_legacy_1', 'old.pdf', 'failed', 'col', '2026-10-25T12:00:00+00:00', '2026-10-25T12:00:00+00:00', 1
        );
        """)
        conn.commit()
        conn.close()

        # Instantiate DocumentRegistryRepository on legacy database
        legacy_repo = DocumentRegistryRepository(db_path=legacy_db_path)
        rec = legacy_repo.get_record("doc_legacy_1")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.document_id, "doc_legacy_1")
        self.assertEqual(rec.document_name, "old.pdf")
        self.assertIsNone(rec.claim_token)
        self.assertIsNone(rec.previous_status)

        # Claiming work on migrated database succeeds
        claimed = legacy_repo.claim_for_reconciliation("doc_legacy_1", lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, DocumentStatus.RECONCILING)
        self.assertIsNotNone(claimed.claim_token)
        self.assertEqual(claimed.previous_status, "failed")

        legacy_dir.cleanup()

    # -------------------------------------------------------------------------
    # 65. Step 21H-J: RECONCILING status blocks user deletion
    # -------------------------------------------------------------------------
    def test_reconciling_status_blocks_user_deletion_claim(self):
        """Verify that claim_for_deletion returns None when a document is in RECONCILING status."""
        rec = self.repo.create_record("active_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Msg")

        # Claim for reconciliation
        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)

        # User deletion claim must fail (return None)
        user_claim = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNone(user_claim)

        # Record remains in RECONCILING status with claim intact
        persisted = self.repo.get_record(rec.document_id)
        self.assertEqual(persisted.status, DocumentStatus.RECONCILING)
        self.assertEqual(persisted.claim_token, claimed.claim_token)

    # -------------------------------------------------------------------------
    # 66. Step 21H-J-B: Fresh reconciliation claim is excluded from candidates
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_excludes_fresh_reconciling_claim(self):
        """Verify that an active, fresh reconciling claim is excluded from candidate discovery."""
        rec = self.repo.create_record("fresh_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, DocumentStatus.RECONCILING)

        # Immediate candidate discovery query with min_age_seconds=60 excludes fresh claim
        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=60)
        self.assertEqual(len(candidates), 0)

        # Query with default min_age_seconds (120) also excludes fresh claim
        default_candidates = self.repo.list_reconciliation_candidates()
        self.assertEqual(len(default_candidates), 0)

    # -------------------------------------------------------------------------
    # 67. Step 21H-J-B: Expired reconciliation claim is discovered as candidate
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_discovers_expired_reconciling_claim(self):
        """Verify that a reconciling claim older than min_age_seconds is returned as a candidate."""
        rec = self.repo.create_record("expired_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)

        # Age the claim beyond the 60s lease window (e.g. 70s ago)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=60)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].document_id, rec.document_id)
        self.assertEqual(candidates[0].status, DocumentStatus.RECONCILING)
        self.assertEqual(candidates[0].claim_token, claimed.claim_token)
        self.assertEqual(candidates[0].previous_status, "failed")
        self.assertTrue(candidates[0].provenance_eligible)

    # -------------------------------------------------------------------------
    # 68. Step 21H-J-B: Ineligible expired claim remains excluded
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_excludes_ineligible_expired_reconciling_claim(self):
        """Verify that an expired reconciling claim with provenance_eligible=0 is never returned."""
        rec = self.repo.create_record("ineligible_rec.pdf", "col", provenance_eligible=False)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()

        with self.repo._get_connection() as conn:
            conn.execute(
                """
                UPDATE document_registry
                SET status = 'reconciling', claim_token = 'rec_fake_token', previous_status = 'failed', updated_at = ?
                WHERE document_id = ?;
                """,
                (stale_time, rec.document_id),
            )

        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=60)
        self.assertEqual(len(candidates), 0)

    # -------------------------------------------------------------------------
    # 69. Step 21H-J-B: Candidate discovery remains read-only on reconciling records
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_read_only_on_reconciling_records(self):
        """Verify list_reconciliation_candidates performs zero mutations on reconciling records."""
        rec = self.repo.create_record("readonly_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Initial failure")
        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        before = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(before)
        self.assertEqual(before.status, DocumentStatus.RECONCILING)

        # Call discovery query
        candidates = self.repo.list_reconciliation_candidates(min_age_seconds=60)
        self.assertEqual(len(candidates), 1)

        after = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(after)

        # Field-by-field verification of complete immutability
        self.assertEqual(after.document_id, before.document_id)
        self.assertEqual(after.document_name, before.document_name)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.collection_name, before.collection_name)
        self.assertEqual(after.created_at, before.created_at)
        self.assertEqual(after.updated_at, before.updated_at)
        self.assertEqual(after.pages_processed, before.pages_processed)
        self.assertEqual(after.chunks_created, before.chunks_created)
        self.assertEqual(after.chunks_stored, before.chunks_stored)
        self.assertEqual(after.error_code, before.error_code)
        self.assertEqual(after.error_message, before.error_message)
        self.assertEqual(after.provenance_eligible, before.provenance_eligible)
        self.assertEqual(after.claim_token, before.claim_token)
        self.assertEqual(after.previous_status, before.previous_status)

    # -------------------------------------------------------------------------
    # 70. Step 21H-J-B: Boundary behavior for reconciling claim discovery
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_reconciling_boundary_behavior(self):
        """Verify strict < cutoff excludes reconciling records at the exact boundary or younger."""
        base_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Younger than threshold (59s old vs 60s min age) -> excluded
        rec_younger = self.repo.create_record("younger_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_younger.document_id, "ERR", "Younger")
        self.repo.claim_for_reconciliation(rec_younger.document_id, lease_seconds=60.0)

        # 2. Exact boundary (exactly 60s old vs 60s min age) -> excluded by strict <
        rec_exact = self.repo.create_record("exact_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_exact.document_id, "ERR", "Exact")
        self.repo.claim_for_reconciliation(rec_exact.document_id, lease_seconds=60.0)

        # 3. Older than threshold (61s old vs 60s min age) -> discovered
        rec_older = self.repo.create_record("older_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec_older.document_id, "ERR", "Older")
        self.repo.claim_for_reconciliation(rec_older.document_id, lease_seconds=60.0)

        t_59 = (base_time - timedelta(seconds=59)).isoformat()
        t_60 = (base_time - timedelta(seconds=60)).isoformat()
        t_61 = (base_time - timedelta(seconds=61)).isoformat()

        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_59, rec_younger.document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_60, rec_exact.document_id))
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_61, rec_older.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            candidates = self.repo.list_reconciliation_candidates(min_age_seconds=60)

        candidate_ids = [c.document_id for c in candidates]
        self.assertIn(rec_older.document_id, candidate_ids)
        self.assertNotIn(rec_exact.document_id, candidate_ids)
        self.assertNotIn(rec_younger.document_id, candidate_ids)

    # -------------------------------------------------------------------------
    # 71. Step 21H-J-B: Semantic relationship between default lease and min age
    # -------------------------------------------------------------------------
    def test_reconciliation_candidates_default_parameters_semantics(self):
        """Verify default parameters: DEFAULT_RECONCILIATION_LEASE_SECONDS (60s) vs DEFAULT_RECONCILIATION_MIN_AGE_SECONDS (120s)."""
        base_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

        # Verify configured constant defaults
        self.assertEqual(DEFAULT_RECONCILIATION_LEASE_SECONDS, 60.0)
        self.assertEqual(DEFAULT_RECONCILIATION_MIN_AGE_SECONDS, 120)

        rec = self.repo.create_record("defaults_rec.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Default test")
        self.repo.claim_for_reconciliation(rec.document_id)

        # Age 1: 80s old (past the 60s lease, but within the 120s discovery grace period)
        t_80 = (base_time - timedelta(seconds=80)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_80, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Default discovery (min_age_seconds=120) excludes 80s-old record
            default_cands = self.repo.list_reconciliation_candidates()
            self.assertEqual(len(default_cands), 0)

            # Explicit lease-aligned discovery (min_age_seconds=DEFAULT_RECONCILIATION_LEASE_SECONDS) discovers it
            lease_aligned_cands = self.repo.list_reconciliation_candidates(
                min_age_seconds=int(DEFAULT_RECONCILIATION_LEASE_SECONDS)
            )
            self.assertEqual(len(lease_aligned_cands), 1)
            self.assertEqual(lease_aligned_cands[0].document_id, rec.document_id)

        # Age 2: 130s old (past both the 60s lease and the 120s discovery grace period)
        t_130 = (base_time - timedelta(seconds=130)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_130, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Now discovered by default parameters
            default_cands_after = self.repo.list_reconciliation_candidates()
            self.assertEqual(len(default_cands_after), 1)
            self.assertEqual(default_cands_after[0].document_id, rec.document_id)

    # -------------------------------------------------------------------------
    # 72. Step 21H-K: verify_reconciliation_claim validates active ownership
    # -------------------------------------------------------------------------
    def test_verify_reconciliation_claim_active_and_expired(self):
        """Verify verify_reconciliation_claim returns True for active claim, False for expired or mismatch."""
        rec = self.repo.create_record("verify_claim.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Fail")

        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        token = claimed.claim_token

        # Active claim with correct token returns True
        self.assertTrue(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

        # Token mismatch returns False
        self.assertFalse(self.repo.verify_reconciliation_claim(rec.document_id, "rec_wrong_token", lease_seconds=60.0))

        # Non-existent document ID returns False
        self.assertFalse(self.repo.verify_reconciliation_claim("non_existent_doc", token, lease_seconds=60.0))

        # Expired lease returns False
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (stale_time, rec.document_id))

        self.assertFalse(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

    # -------------------------------------------------------------------------
    # 73. Step 21H-K: verify_reconciliation_claim validates arguments
    # -------------------------------------------------------------------------
    def test_verify_reconciliation_claim_invalid_arguments_rejected(self):
        """Verify invalid document_id, claim_token, or lease_seconds raise DocumentRegistryError."""
        with self.assertRaises(DocumentRegistryError):
            self.repo.verify_reconciliation_claim("", "rec_token")

        with self.assertRaises(DocumentRegistryError):
            self.repo.verify_reconciliation_claim("doc_1", "")

        with self.assertRaises(DocumentRegistryError):
            self.repo.verify_reconciliation_claim("doc_1", "rec_token", lease_seconds=True)

        with self.assertRaises(DocumentRegistryError):
            self.repo.verify_reconciliation_claim("doc_1", "rec_token", lease_seconds=-10)

        with self.assertRaises(DocumentRegistryError):
            self.repo.verify_reconciliation_claim("doc_1", "rec_token", lease_seconds=float("nan"))

    # -------------------------------------------------------------------------
    # 74. Step 21H-K-B: verify_reconciliation_claim controlled-clock boundary behavior
    # -------------------------------------------------------------------------
    def test_verify_reconciliation_claim_lease_boundary(self):
        """Verify lease boundary behavior of verify_reconciliation_claim with deterministic clock mocking.

        SQL Predicate: datetime(updated_at) >= datetime(cutoff)
        Where cutoff = now - timedelta(seconds=float(lease_seconds)).
        SQLite's datetime() floor-truncates fractional seconds to whole seconds (YYYY-MM-DD HH:MM:SS).
        """
        base_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
        rec = self.repo.create_record("boundary_test.pdf", "col", provenance_eligible=True)
        self.repo.update_status_failed(rec.document_id, "ERR", "Boundary test")
        claimed = self.repo.claim_for_reconciliation(rec.document_id, lease_seconds=60.0)
        self.assertIsNotNone(claimed)
        token = claimed.claim_token

        # 1. Before expiry (59s old vs 60s lease): datetime(updated_at) is 11:59:01, cutoff is 11:59:00 -> TRUE
        t_before = (base_time - timedelta(seconds=59)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_before, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            self.assertTrue(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

        # 2. Exactly at boundary (60s old vs 60s lease): datetime(updated_at) is 11:59:00, cutoff is 11:59:00 -> TRUE (inclusive >= predicate)
        t_exact = (base_time - timedelta(seconds=60)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_exact, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            self.assertTrue(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

        # 3. After expiry (61s old vs 60s lease): datetime(updated_at) is 11:58:59, cutoff is 11:59:00 -> FALSE
        t_after = (base_time - timedelta(seconds=61)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_after, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            self.assertFalse(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

        # 4. Subsecond behavior: demonstrate floor truncation in SQLite datetime()
        # updated_at = 11:59:00.800 (integer second 11:59:00)
        # cutoff = 11:59:00.200 (now = 12:00:00.200, elapsed = 59.4s < 60s)
        # Both truncate to 11:59:00; '11:59:00' >= '11:59:00' -> TRUE
        t_subsecond_valid = "2026-10-25T11:59:00.800000+00:00"
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_subsecond_valid, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 10, 25, 12, 0, 0, 200000, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            self.assertTrue(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

        # updated_at = 11:58:59.900 (integer second 11:58:59)
        # cutoff = 11:59:00.000 (now = 12:00:00.000, elapsed = 60.1s > 60s)
        # '11:58:59' >= '11:59:00' -> FALSE
        t_subsecond_expired = "2026-10-25T11:58:59.900000+00:00"
        with self.repo._get_connection() as conn:
            conn.execute("UPDATE document_registry SET updated_at = ? WHERE document_id = ?;", (t_subsecond_expired, rec.document_id))

        with patch("app.database.registry.datetime") as mock_dt:
            mock_dt.now.return_value = base_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            self.assertFalse(self.repo.verify_reconciliation_claim(rec.document_id, token, lease_seconds=60.0))

    # -------------------------------------------------------------------------
    # 57. Deletion claim attempt token generation
    # -------------------------------------------------------------------------
    def test_claim_for_deletion_generates_attempt_token(self):
        """Verify claim_for_deletion generates unique attempt tokens prefixed with del_."""
        rec = self.repo.create_record("token_test.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 2, 2)

        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, DocumentStatus.DELETING)
        self.assertTrue(claimed.claim_token.startswith("del_"))

        # Advance lease to allow reclaim
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, rec.document_id),
            )

        reclaimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(reclaimed)
        self.assertTrue(reclaimed.claim_token.startswith("del_"))
        self.assertNotEqual(claimed.claim_token, reclaimed.claim_token)

    # -------------------------------------------------------------------------
    # 58. Finalize deletion transitions to durable tombstone and preserves metadata
    # -------------------------------------------------------------------------
    def test_finalize_deletion_transitions_to_durable_tombstone(self):
        """Verify finalize_deletion sets status to DELETED, chunks_stored to 0, clears claim_token, and retains row."""
        rec = self.repo.create_record("tombstone_unit.pdf", "test_col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, pages_processed=5, chunks_created=15, chunks_stored=15)

        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(claimed)
        token = claimed.claim_token

        finalized = self.repo.finalize_deletion(rec.document_id, claim_token=token)
        self.assertEqual(finalized.status, DocumentStatus.DELETED)
        self.assertEqual(finalized.chunks_stored, 0)
        self.assertIsNone(finalized.claim_token)
        self.assertEqual(finalized.document_id, rec.document_id)
        self.assertEqual(finalized.document_name, "tombstone_unit.pdf")
        self.assertEqual(finalized.collection_name, "test_col")
        self.assertTrue(finalized.provenance_eligible)
        self.assertEqual(finalized.pages_processed, 5)
        self.assertEqual(finalized.chunks_created, 15)

        # Confirm directly in SQLite
        row = self.repo.get_record(rec.document_id)
        self.assertIsNotNone(row)
        self.assertEqual(row.status, DocumentStatus.DELETED)
        self.assertEqual(row.chunks_stored, 0)
        self.assertIsNone(row.claim_token)

    # -------------------------------------------------------------------------
    # 59. Finalize deletion rejects stale token
    # -------------------------------------------------------------------------
    def test_finalize_deletion_rejects_stale_token(self):
        """Verify finalize_deletion raises DocumentStateConflictError when claim_token does not match."""
        rec = self.repo.create_record("stale_finalize.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(claimed)
        stale_token = "del_stale_nonexistent_token"

        with self.assertRaises(DocumentStateConflictError):
            self.repo.finalize_deletion(rec.document_id, claim_token=stale_token)

        # Record remains in DELETING
        current = self.repo.get_record(rec.document_id)
        self.assertEqual(current.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 60. Finalize deletion rejects non-deleting status
    # -------------------------------------------------------------------------
    def test_finalize_deletion_rejects_non_deleting_status(self):
        """Verify finalize_deletion raises DocumentStateConflictError if document is not in DELETING status."""
        rec = self.repo.create_record("not_deleting.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        with self.assertRaises(DocumentStateConflictError):
            self.repo.finalize_deletion(rec.document_id, claim_token="del_dummy")

    # -------------------------------------------------------------------------
    # 61. Finalize deletion missing document raises DocumentNotFoundError
    # -------------------------------------------------------------------------
    def test_finalize_deletion_missing_document_raises_not_found(self):
        """Verify finalize_deletion raises DocumentNotFoundError if document does not exist."""
        with self.assertRaises(DocumentNotFoundError):
            self.repo.finalize_deletion("missing_doc_xyz", claim_token="del_dummy")

    # -------------------------------------------------------------------------
    # 62. Finalize deletion invalid arguments raise DocumentRegistryError
    # -------------------------------------------------------------------------
    def test_finalize_deletion_invalid_arguments_raise_registry_error(self):
        """Verify finalize_deletion raises DocumentRegistryError for empty document_id or claim_token."""
        with self.assertRaises(DocumentRegistryError):
            self.repo.finalize_deletion("", claim_token="del_token")
        with self.assertRaises(DocumentRegistryError):
            self.repo.finalize_deletion("doc_id", claim_token="")

    # -------------------------------------------------------------------------
    # 63. Already-deleted document cannot be claimed for deletion
    # -------------------------------------------------------------------------
    def test_already_deleted_document_cannot_be_claimed(self):
        """Verify claim_for_deletion returns None when document is already in DELETED status."""
        rec = self.repo.create_record("already_del.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)
        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.repo.finalize_deletion(rec.document_id, claim_token=claimed.claim_token)

        # Attempt to claim tombstone
        second_claim = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNone(second_claim)

    # -------------------------------------------------------------------------
    # 64. update_status_deletion_failed guards against stale claim token
    # -------------------------------------------------------------------------
    def test_update_status_deletion_failed_with_claim_token_guard(self):
        """Verify update_status_deletion_failed succeeds with valid token and rejects stale token."""
        rec = self.repo.create_record("fail_guard.pdf", "col", provenance_eligible=True)
        self.repo.update_status_succeeded(rec.document_id, 1, 1, 1)

        claimed = self.repo.claim_for_deletion(rec.document_id)
        self.assertIsNotNone(claimed)

        # Stale token raises DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError):
            self.repo.update_status_deletion_failed(
                rec.document_id,
                error_code="ERR",
                error_message="msg",
                claim_token="del_stale_wrong_token",
            )

        # Valid token succeeds and clears claim_token
        updated = self.repo.update_status_deletion_failed(
            rec.document_id,
            error_code="ERR_OK",
            error_message="Valid failure",
            claim_token=claimed.claim_token,
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, DocumentStatus.DELETION_FAILED)
        self.assertIsNone(updated.claim_token)


if __name__ == "__main__":
    unittest.main()




