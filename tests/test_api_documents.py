"""API tests for the GET /api/documents document registry listing endpoint."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock


# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_registry_dependency,
    reset_document_registry_cache,
)
from app.database import (
    DocumentRecord,
    DocumentRegistryError,
    DocumentRegistryRepository,
    DocumentStatus,
)
from app.main import app


class TestAPIDocumentsEndpoint(unittest.TestCase):
    """Test suite covering GET /api/documents pagination, ordering, validation, and error safety."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_api_docs.db")
        self.repo = DocumentRegistryRepository(db_path=self.db_path)

        # Inject repository dependency
        app.dependency_overrides[get_document_registry_dependency] = lambda: self.repo
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_document_registry_cache()
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Empty registry returns empty list with total=0
    # -------------------------------------------------------------------------
    def test_empty_registry_returns_empty_list(self):
        """Verify empty database returns empty documents list and total count of 0."""
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["documents"], [])
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["limit"], 50)
        self.assertEqual(data["offset"], 0)

    # -------------------------------------------------------------------------
    # 2. Listing documents returns records newest first
    # -------------------------------------------------------------------------
    def test_listing_documents_returns_newest_first(self):
        """Verify returned documents are ordered by creation time descending."""
        doc1 = self.repo.create_record("first.pdf", "col")
        doc2 = self.repo.create_record("second.pdf", "col")
        doc3 = self.repo.create_record("third.pdf", "col")

        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["documents"]), 3)

        doc_ids = [d["document_id"] for d in data["documents"]]
        self.assertEqual(doc_ids, [doc3.document_id, doc2.document_id, doc1.document_id])

    # -------------------------------------------------------------------------
    # 3. Document payload contains required metadata and no raw bytes/vectors
    # -------------------------------------------------------------------------
    def test_document_payload_attributes(self):
        """Verify document fields contain complete metadata and no sensitive/internal content."""
        doc = self.repo.create_record("audit_report.pdf", "col")
        self.repo.update_status_succeeded(
            document_id=doc.document_id,
            pages_processed=12,
            chunks_created=36,
            chunks_stored=36,
        )

        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        item = response.json()["documents"][0]

        self.assertEqual(item["document_id"], doc.document_id)
        self.assertEqual(item["document_name"], "audit_report.pdf")
        self.assertEqual(item["status"], "succeeded")
        self.assertEqual(item["collection_name"], "col")
        self.assertEqual(item["pages_processed"], 12)
        self.assertEqual(item["chunks_created"], 36)
        self.assertEqual(item["chunks_stored"], 36)
        self.assertIsNone(item["error_code"])
        self.assertIsNone(item["error_message"])

        # Explicitly verify no internal raw binary or vector fields exist
        self.assertNotIn("file_bytes", item)
        self.assertNotIn("vector", item)
        self.assertNotIn("embedding", item)

    # -------------------------------------------------------------------------
    # 4. Failed document exposes safe failure information
    # -------------------------------------------------------------------------
    def test_failed_document_exposes_safe_error_info(self):
        """Verify failed records display categorized error_code and sanitized summary."""
        doc = self.repo.create_record("malformed.pdf", "col")
        self.repo.update_status_failed(
            document_id=doc.document_id,
            error_code="MALFORMED_PDF",
            error_message="Failed to parse PDF document header.",
        )

        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        item = response.json()["documents"][0]

        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_code"], "MALFORMED_PDF")
        self.assertEqual(item["error_message"], "Failed to parse PDF document header.")

    # -------------------------------------------------------------------------
    # 5. Pagination limit and offset work correctly
    # -------------------------------------------------------------------------
    def test_pagination_limit_and_offset(self):
        """Verify limit and offset query parameters slice results properly."""
        for i in range(10):
            self.repo.create_record(f"doc_{i}.pdf", "col")

        # Page 1: limit=3, offset=0
        res1 = self.client.get("/api/documents?limit=3&offset=0")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["total"], 10)
        self.assertEqual(data1["limit"], 3)
        self.assertEqual(data1["offset"], 0)
        self.assertEqual(len(data1["documents"]), 3)

        # Page 2: limit=3, offset=3
        res2 = self.client.get("/api/documents?limit=3&offset=3")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(len(data2["documents"]), 3)
        self.assertNotEqual(
            data1["documents"][0]["document_id"], data2["documents"][0]["document_id"]
        )

    # -------------------------------------------------------------------------
    # 6. Invalid pagination parameters are rejected with HTTP 422
    # -------------------------------------------------------------------------
    def test_invalid_pagination_parameters_rejected(self):
        """Verify non-positive limit, excessive limit, and negative offset return 422."""
        # limit <= 0
        res_zero_limit = self.client.get("/api/documents?limit=0")
        self.assertEqual(res_zero_limit.status_code, 422)

        res_neg_limit = self.client.get("/api/documents?limit=-5")
        self.assertEqual(res_neg_limit.status_code, 422)

        # limit > 100
        res_huge_limit = self.client.get("/api/documents?limit=101")
        self.assertEqual(res_huge_limit.status_code, 422)

        # offset < 0
        res_neg_offset = self.client.get("/api/documents?offset=-1")
        self.assertEqual(res_neg_offset.status_code, 422)

    # -------------------------------------------------------------------------
    # 7. Internal database errors are not leaked to API clients
    # -------------------------------------------------------------------------
    def test_internal_database_errors_not_leaked(self):
        """Verify unexpected database exceptions result in safe HTTP 500 without leaking details."""
        mock_broken_repo = MagicMock(spec=DocumentRegistryRepository)
        mock_broken_repo.count_records.side_effect = sqlite3.OperationalError(
            "no such table: secret_internal_table at /var/data/private.db"
        )

        app.dependency_overrides[get_document_registry_dependency] = lambda: mock_broken_repo

        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]

        # Detail must be sanitized and free from internal error leaks
        self.assertEqual(detail, "An error occurred while retrieving registered documents.")
        self.assertNotIn("secret_internal_table", detail)
        self.assertNotIn("sqlite", detail.lower())
        self.assertNotIn("/var/data", detail)

    # -------------------------------------------------------------------------
    # 8. Retrieve existing document returns HTTP 200 and data
    # -------------------------------------------------------------------------
    def test_get_existing_document_by_id_returns_200_and_data(self):
        """Verify GET /api/documents/{document_id} returns 200 and correct document payload."""
        rec = self.repo.create_record("contract_2026.pdf", "legal_knowledge")
        self.repo.update_status_succeeded(
            document_id=rec.document_id,
            pages_processed=8,
            chunks_created=24,
            chunks_stored=24,
            collection_name="legal_knowledge",
        )

        response = self.client.get(f"/api/documents/{rec.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["document_id"], rec.document_id)
        self.assertEqual(data["document_name"], "contract_2026.pdf")
        self.assertEqual(data["status"], "succeeded")
        self.assertEqual(data["collection_name"], "legal_knowledge")
        self.assertEqual(data["pages_processed"], 8)
        self.assertEqual(data["chunks_created"], 24)
        self.assertEqual(data["chunks_stored"], 24)
        self.assertIsNone(data["error_code"])
        self.assertIsNone(data["error_message"])
        self.assertIsNotNone(data["created_at"])
        self.assertIsNotNone(data["updated_at"])

    # -------------------------------------------------------------------------
    # 9. Missing document ID returns HTTP 404
    # -------------------------------------------------------------------------
    def test_get_missing_document_by_id_returns_404(self):
        """Verify GET /api/documents/{document_id} returns 404 when ID does not exist."""
        response = self.client.get("/api/documents/non_existent_doc_12345")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("non_existent_doc_12345", data["detail"])
        self.assertIn("not found", data["detail"].lower())

    # -------------------------------------------------------------------------
    # 10. Document in 'processing' status can be retrieved
    # -------------------------------------------------------------------------
    def test_get_document_in_processing_status(self):
        """Verify a document currently in 'processing' state returns 200 with processing status."""
        rec = self.repo.create_record("in_flight.pdf", "default_collection")

        response = self.client.get(f"/api/documents/{rec.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["document_id"], rec.document_id)
        self.assertEqual(data["document_name"], "in_flight.pdf")
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["collection_name"], "default_collection")
        self.assertIsNone(data["pages_processed"])
        self.assertIsNone(data["chunks_created"])
        self.assertIsNone(data["chunks_stored"])
        self.assertIsNone(data["error_code"])
        self.assertIsNone(data["error_message"])

    # -------------------------------------------------------------------------
    # 11. Document in 'failed' status can be retrieved
    # -------------------------------------------------------------------------
    def test_get_document_in_failed_status(self):
        """Verify a document with 'failed' state returns 200 with categorized safe error info."""
        rec = self.repo.create_record("broken_file.pdf", "default_collection")
        self.repo.update_status_failed(
            document_id=rec.document_id,
            error_code="MALFORMED_PDF",
            error_message="Failed to parse PDF document header.",
        )

        response = self.client.get(f"/api/documents/{rec.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["document_id"], rec.document_id)
        self.assertEqual(data["document_name"], "broken_file.pdf")
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["error_code"], "MALFORMED_PDF")
        self.assertEqual(data["error_message"], "Failed to parse PDF document header.")
        self.assertIsNone(data["pages_processed"])
        self.assertIsNone(data["chunks_stored"])

    # -------------------------------------------------------------------------
    # 12. Unexpected repository/registry error returns safe HTTP 500
    # -------------------------------------------------------------------------
    def test_get_document_internal_database_error_returns_safe_500(self):
        """Verify unexpected database exceptions during retrieval return 500 without leaking details."""
        mock_broken_repo = MagicMock(spec=DocumentRegistryRepository)
        mock_broken_repo.get_record.side_effect = sqlite3.OperationalError(
            "disk I/O error at /var/private/enterprise_docs.db with SELECT password"
        )

        app.dependency_overrides[get_document_registry_dependency] = lambda: mock_broken_repo

        response = self.client.get("/api/documents/doc_error_test")
        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]

        self.assertEqual(detail, "An error occurred while retrieving the document.")
        self.assertNotIn("disk I/O", detail)
        self.assertNotIn("password", detail)
        self.assertNotIn("sqlite", detail.lower())
        self.assertNotIn("/var/private", detail)

    # -------------------------------------------------------------------------
    # 13. Dynamic route does not interfere with list endpoint or pagination
    # -------------------------------------------------------------------------
    def test_get_document_does_not_interfere_with_list_endpoint(self):
        """Verify GET /api/documents and pagination continue functioning alongside GET /api/documents/{id}."""
        d1 = self.repo.create_record("doc1.pdf", "col")
        d2 = self.repo.create_record("doc2.pdf", "col")

        # Static list route works
        list_res = self.client.get("/api/documents?limit=1&offset=0")
        self.assertEqual(list_res.status_code, 200)
        list_data = list_res.json()
        self.assertEqual(list_data["total"], 2)
        self.assertEqual(len(list_data["documents"]), 1)
        self.assertEqual(list_data["documents"][0]["document_id"], d2.document_id)

        # Dynamic detail route works
        detail_res = self.client.get(f"/api/documents/{d1.document_id}")
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(detail_res.json()["document_id"], d1.document_id)



from app.api.dependencies import (
    get_document_registry_dependency,
    get_vector_store_dependency,
    reset_document_registry_cache,
    reset_vector_store_cache,
)
from app.services.vector_store import (
    BaseVectorStoreService,
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)


class TestAPIDocumentsDeleteEndpoint(unittest.TestCase):
    """Test suite covering DELETE /api/documents/{document_id} safe deletion and coordination."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_api_delete_docs.db")
        self.repo = DocumentRegistryRepository(db_path=self.db_path)

        # Mock vector store
        self.mock_vector_store = MagicMock(spec=BaseVectorStoreService)
        self.mock_vector_store.delete_by_document_id.return_value = 5

        # Inject dependencies
        app.dependency_overrides[get_document_registry_dependency] = lambda: self.repo
        app.dependency_overrides[get_vector_store_dependency] = lambda: self.mock_vector_store
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_document_registry_cache()
        reset_vector_store_cache()
        self.temp_dir.cleanup()

    # 1. Eligible document is deleted successfully
    def test_eligible_succeeded_document_deleted_successfully(self):
        """Verify an eligible document in 'succeeded' status is deleted from Qdrant and SQLite."""
        doc = self.repo.create_record(
            "report.pdf", "enterprise_knowledge", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=2, chunks_created=5, chunks_stored=5)

        self.mock_vector_store.delete_by_document_id.return_value = 5

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["document_id"], doc.document_id)
        self.assertTrue(data["deleted"])
        self.assertEqual(data["vectors_deleted"], 5)
        self.assertIn("successfully deleted", data["message"])

        # Verify Qdrant deletion called with exact document_id and collection_name
        self.mock_vector_store.delete_by_document_id.assert_called_once_with(
            document_id=doc.document_id,
            collection_name="enterprise_knowledge",
        )

        # Verify SQLite record is retained as durable tombstone with status 'deleted'
        record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.DELETED)
        self.assertEqual(record.chunks_stored, 0)
        self.assertTrue(record.provenance_eligible)
        self.assertEqual(record.collection_name, "enterprise_knowledge")

    # 2. Only target document vectors removed & 3. Other documents remain intact
    def test_only_target_document_vectors_removed_other_docs_intact(self):
        """Verify only the specified document is targeted and other registry records remain untouched."""
        doc1 = self.repo.create_record("doc1.pdf", "col_a", provenance_eligible=True)
        self.repo.update_status_succeeded(doc1.document_id, pages_processed=1, chunks_created=3, chunks_stored=3)

        doc2 = self.repo.create_record("doc2.pdf", "col_a", provenance_eligible=True)
        self.repo.update_status_succeeded(doc2.document_id, pages_processed=1, chunks_created=4, chunks_stored=4)

        # Delete doc1
        response = self.client.delete(f"/api/documents/{doc1.document_id}")
        self.assertEqual(response.status_code, 200)

        # Verify vector store called strictly with doc1
        self.mock_vector_store.delete_by_document_id.assert_called_once_with(
            document_id=doc1.document_id,
            collection_name="col_a",
        )

        # doc1 retained with status 'deleted', doc2 still exists in SQLite with status 'succeeded'
        rec1 = self.repo.get_record(doc1.document_id)
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1.status, DocumentStatus.DELETED)
        remaining = self.repo.get_record(doc2.document_id)
        self.assertIsNotNone(remaining)
        self.assertEqual(remaining.document_id, doc2.document_id)
        self.assertEqual(remaining.status, DocumentStatus.SUCCEEDED)

    # 4. Legacy/unverified record returns 409 and remains untouched
    def test_legacy_unverified_record_rejected_with_409(self):
        """Verify legacy document without provenance eligibility is rejected with HTTP 409."""
        doc = self.repo.create_record(
            "legacy.pdf", "col", provenance_eligible=False
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("legacy unverified provenance", response.json()["detail"])

        # Qdrant must NOT be called
        self.mock_vector_store.delete_by_document_id.assert_not_called()

        # Record must NOT be deleted or modified
        record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.SUCCEEDED)

    # 5. Missing document ID returns 404
    def test_missing_document_id_returns_404(self):
        """Verify deleting a non-existent document ID returns HTTP 404."""
        response = self.client.delete("/api/documents/non_existent_id_999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())
        self.mock_vector_store.delete_by_document_id.assert_not_called()

    # 6. Processing document returns 409
    def test_processing_document_rejected_with_409(self):
        """Verify document currently in 'processing' status cannot be deleted (HTTP 409)."""
        doc = self.repo.create_record(
            "active.pdf", "col", provenance_eligible=True
        )
        # Status defaults to PROCESSING

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("currently being processed", response.json()["detail"])

        # Vectors and SQLite record preserved
        self.mock_vector_store.delete_by_document_id.assert_not_called()
        record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.PROCESSING)

    # 7. Failed but eligible document can be deleted safely
    def test_failed_eligible_document_can_be_deleted(self):
        """Verify document in 'failed' status with provenance eligibility can be deleted."""
        doc = self.repo.create_record(
            "failed.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_failed(doc.document_id, error_code="ERR", error_message="Failed parsing")

        self.mock_vector_store.delete_by_document_id.return_value = 0

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)
        self.mock_vector_store.delete_by_document_id.assert_called_once_with(
            document_id=doc.document_id,
            collection_name="col",
        )
        rec = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.DELETED)

    # 8. Qdrant failure preserves registry record and produces safe error (503)
    def test_qdrant_failure_preserves_registry_record_as_deletion_failed(self):
        """Verify vector store connection error preserves SQLite record in deletion_failed and returns 503."""
        doc = self.repo.create_record(
            "doc.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        self.mock_vector_store.delete_by_document_id.side_effect = VectorStoreConnectionError(
            "Failed to connect to Qdrant at 127.0.0.1:6333: Connection refused"
        )

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertIn("currently unavailable", detail)
        # Internal socket/host details must not leak
        self.assertNotIn("127.0.0.1", detail)
        self.assertNotIn("Connection refused", detail)

        # Record must NOT be deleted, but preserved in 'deletion_failed' state
        record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.DELETION_FAILED)
        self.assertEqual(record.error_code, "VECTOR_STORE_UNAVAILABLE")

    # 9. SQLite finalization failure after vector deletion is reported safely and can be retried
    def test_sqlite_failure_after_vector_deletion_reported_safely_and_retryable(self):
        """Verify unexpected failure to finalize SQLite record after vector deletion returns 500 and is retryable."""
        doc = self.repo.create_record(
            "doc.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        # Mock finalize_deletion raising unexpected exception
        original_finalize_deletion = self.repo.finalize_deletion
        self.repo.finalize_deletion = MagicMock(side_effect=RuntimeError("Simulated database write error"))

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 500)
        self.assertIn("finalizing document removal", response.json()["detail"])

        # Document is marked DELETION_FAILED
        record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.DELETION_FAILED)

        # Restore original finalize_deletion and retry deletion
        self.repo.finalize_deletion = original_finalize_deletion
        self.mock_vector_store.delete_by_document_id.return_value = 0

        retry_response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(retry_response.status_code, 200)
        final_record = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(final_record)
        self.assertEqual(final_record.status, DocumentStatus.DELETED)

    # 10. Repeated deletion requests do not delete other documents or report false success
    def test_repeated_deletion_requests_handled_safely(self):
        """Verify deleting an already deleted document returns 409 Conflict and preserves tombstone."""
        doc = self.repo.create_record(
            "repeat.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        # First request succeeds
        res1 = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(res1.status_code, 200)

        # Second request returns 409 Conflict
        res2 = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(res2.status_code, 409)
        self.assertIn("already been deleted", res2.json()["detail"].lower())

        # Tombstone record is preserved in SQLite
        rec = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.DELETED)

    # 11. Concurrent deletion attempts cannot both claim the same record
    def test_concurrent_deletion_claims_rejected(self):
        """Verify that when a record is already in 'deleting' state, another request gets 409."""
        doc = self.repo.create_record(
            "concurrent.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        # Manually claim it
        claimed = self.repo.claim_for_deletion(doc.document_id)
        self.assertIsNotNone(claimed)

        # Now simulate a concurrent HTTP DELETE arriving while first is deleting vectors
        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("already in progress", response.json()["detail"])

    # 12. Existing list and single-document GET endpoints remain compatible
    def test_get_and_list_endpoints_include_provenance_eligible(self):
        """Verify list and detail endpoints correctly return provenance_eligible flag."""
        doc_eligible = self.repo.create_record(
            "new.pdf", "col", provenance_eligible=True
        )
        doc_legacy = self.repo.create_record(
            "old.pdf", "col", provenance_eligible=False
        )

        # Test single GET
        res_single = self.client.get(f"/api/documents/{doc_eligible.document_id}")
        self.assertEqual(res_single.status_code, 200)
        self.assertTrue(res_single.json()["provenance_eligible"])

        res_legacy = self.client.get(f"/api/documents/{doc_legacy.document_id}")
        self.assertEqual(res_legacy.status_code, 200)
        self.assertFalse(res_legacy.json()["provenance_eligible"])

        # Test list GET
        res_list = self.client.get("/api/documents")
        self.assertEqual(res_list.status_code, 200)
        items = {d["document_id"]: d["provenance_eligible"] for d in res_list.json()["documents"]}
        self.assertTrue(items[doc_eligible.document_id])
        self.assertFalse(items[doc_legacy.document_id])

    # 13. Invalid empty/whitespace document_id rejected
    def test_invalid_document_id_returns_422(self):
        """Verify empty or whitespace-only document_id returns HTTP 422."""
        response = self.client.delete("/api/documents/%20%20")
        self.assertEqual(response.status_code, 422)

    # 14. Non-existent collection handled gracefully
    def test_collection_not_found_handled_gracefully(self):
        """Verify that if Qdrant collection is missing, deletion finishes safely with 0 vectors."""
        doc = self.repo.create_record(
            "ghost.pdf", "missing_col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        self.mock_vector_store.delete_by_document_id.side_effect = VectorStoreCollectionNotFoundError(
            "Collection 'missing_col' does not exist."
        )

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["deleted"])
        self.assertEqual(data["vectors_deleted"], 0)
        rec = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.DELETED)

    # 15. Stale deleting claim is safely recovered via API
    def test_stale_deleting_claim_recovered_via_api(self):
        """Verify that a document stranded in 'deleting' past the lease timeout can be recovered via DELETE endpoint."""
        doc = self.repo.create_record(
            "stranded.pdf", "enterprise_knowledge", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=2, chunks_created=3, chunks_stored=3)

        # Force state to 'deleting' with updated_at 10 minutes ago (> 300s lease)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                (stale_time, doc.document_id),
            )

        self.mock_vector_store.delete_by_document_id.return_value = 3

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_id"], doc.document_id)
        self.assertTrue(data["deleted"])
        self.assertEqual(data["vectors_deleted"], 3)

        # Vector store called with exact document ID
        self.mock_vector_store.delete_by_document_id.assert_called_once_with(
            document_id=doc.document_id,
            collection_name="enterprise_knowledge",
        )
        # SQLite record retained as tombstone with status 'deleted'
        rec = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.DELETED)

    # 16. Fresh deleting claim is rejected with HTTP 409
    def test_fresh_deleting_claim_rejected_with_409(self):
        """Verify that a document in 'deleting' with a recent claim (< 300s lease) is rejected with 409."""
        doc = self.repo.create_record(
            "fresh_active.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=1, chunks_created=2, chunks_stored=2)

        # Claim into deleting now (fresh lease)
        claimed = self.repo.claim_for_deletion(doc.document_id)
        self.assertIsNotNone(claimed)

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("already in progress", response.json()["detail"])
        self.mock_vector_store.delete_by_document_id.assert_not_called()

    # 17. Stale deleting claim with provenance_eligible=False is rejected
    def test_stale_deleting_claim_unverified_rejected_with_409(self):
        """Verify that an abandoned claim on an unverified document is still rejected with 409."""
        doc = self.repo.create_record(
            "stale_legacy.pdf", "col", provenance_eligible=False
        )
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                (stale_time, doc.document_id),
            )

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("legacy unverified provenance", response.json()["detail"])
        self.mock_vector_store.delete_by_document_id.assert_not_called()

    # 18. Recovery when vectors were already deleted during prior attempt
    def test_recovery_when_vectors_already_deleted_prior_attempt(self):
        """Verify that when Qdrant vectors were already deleted during a crashed run, recovery finishes registry cleanup."""
        doc = self.repo.create_record(
            "partially_cleaned.pdf", "col", provenance_eligible=True
        )
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET status = 'deleting', updated_at = ? WHERE document_id = ?;",
                (stale_time, doc.document_id),
            )

        # Vectors already 0 (deleted during prior crashed execution)
        self.mock_vector_store.delete_by_document_id.return_value = 0

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["deleted"])
        self.assertEqual(data["vectors_deleted"], 0)

        # SQLite record retained as tombstone with status 'deleted'
        rec = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.DELETED)

    # 19. Deletion rejected when document is reconciling
    def test_deletion_rejected_when_document_is_reconciling(self):
        """Verify DELETE /api/documents/{document_id} rejects active reconciliation with 409 Conflict."""
        doc = self.repo.create_record(
            "reconciling_doc.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_failed(doc.document_id, "ingest_err", "Ingestion failed")
        claimed = self.repo.claim_for_reconciliation(doc.document_id)
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, DocumentStatus.RECONCILING)

        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 409)
        self.assertIn("reconciliation is currently in progress", response.json()["detail"])
        self.mock_vector_store.delete_by_document_id.assert_not_called()

        # Document record remains in reconciling status
        current = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(current)
        self.assertEqual(current.status, DocumentStatus.RECONCILING)

    # 20. Stale deletion attempt cannot finalize over newer deletion claim
    def test_stale_deletion_attempt_cannot_finalize_over_newer_claim(self):
        """Verify that if a deletion attempt's lease expires and another claim takes over, the stale attempt fails finalization with 409."""
        doc = self.repo.create_record(
            "lease_race.pdf", "col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=2, chunks_created=4, chunks_stored=4)

        # Worker 1 claims document
        claim1 = self.repo.claim_for_deletion(doc.document_id)
        self.assertIsNotNone(claim1)
        token1 = claim1.claim_token

        # Time passes (> 300s), lease expires, Worker 2 claims document with new token
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        with self.repo._get_connection() as conn:
            conn.execute(
                "UPDATE document_registry SET updated_at = ? WHERE document_id = ?;",
                (stale_time, doc.document_id),
            )
        claim2 = self.repo.claim_for_deletion(doc.document_id)
        self.assertIsNotNone(claim2)
        token2 = claim2.claim_token
        self.assertNotEqual(token1, token2)

        # Worker 1 now attempts to finalize using stale token1 -> DocumentStateConflictError
        from app.database import DocumentStateConflictError
        with self.assertRaises(DocumentStateConflictError):
            self.repo.finalize_deletion(doc.document_id, claim_token=token1)

        # Record is still in DELETING owned by token2
        current = self.repo.get_record(doc.document_id)
        self.assertEqual(current.status, DocumentStatus.DELETING)
        self.assertEqual(current.claim_token, token2)

        # Worker 2 successfully finalizes
        finalized = self.repo.finalize_deletion(doc.document_id, claim_token=token2)
        self.assertEqual(finalized.status, DocumentStatus.DELETED)
        self.assertIsNone(finalized.claim_token)

    # 21. Tombstone retains exact identity and provenance metadata
    def test_tombstone_retains_exact_identity_and_provenance_metadata(self):
        """Verify durable tombstone retains exact document_id, document_name, collection, and provenance_eligible."""
        doc = self.repo.create_record(
            "durable_tombstone.pdf", "target_col", provenance_eligible=True
        )
        self.repo.update_status_succeeded(doc.document_id, pages_processed=3, chunks_created=9, chunks_stored=9)

        # Delete document via API
        self.mock_vector_store.delete_by_document_id.return_value = 9
        response = self.client.delete(f"/api/documents/{doc.document_id}")
        self.assertEqual(response.status_code, 200)

        # Inspect retained tombstone
        tombstone = self.repo.get_record(doc.document_id)
        self.assertIsNotNone(tombstone)
        self.assertEqual(tombstone.document_id, doc.document_id)
        self.assertEqual(tombstone.document_name, "durable_tombstone.pdf")
        self.assertEqual(tombstone.collection_name, "target_col")
        self.assertEqual(tombstone.status, DocumentStatus.DELETED)
        self.assertTrue(tombstone.provenance_eligible)
        self.assertEqual(tombstone.chunks_stored, 0)
        self.assertEqual(tombstone.pages_processed, 3)
        self.assertEqual(tombstone.chunks_created, 9)
        self.assertIsNone(tombstone.claim_token)


if __name__ == "__main__":
    unittest.main()



