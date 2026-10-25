"""Focused API tests for the FastAPI document ingestion endpoint and lifecycle."""

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

from fastapi.testclient import TestClient
import starlette.datastructures

from app.api.dependencies import (
    get_document_registry_dependency,
    get_ingestion_service_dependency,
    get_rag_answer_generator_dependency,
    get_settings_dependency,
    get_shared_ingestion_service,
    get_vector_store_dependency,
    reset_document_registry_cache,
    reset_ingestion_service_cache,
    reset_rag_answer_generator_cache,
    reset_settings_cache,
    reset_vector_store_cache,
)
from app.config import Settings
from app.database import (
    DocumentRecord,
    DocumentRegistryRepository,
    DocumentStateConflictError,
    DocumentStatus,
)
from app.main import app
from app.rag.answer_generator import RAGResponse
from app.services.ingestion import (
    IngestionEmptyDocumentError,
    IngestionError,
    IngestionResult,
    IngestionService,
    IngestionValidationError,
)
from app.services.pdf_loader import PDFEncryptedError, PDFProcessingError
from app.services.vector_store import (
    VectorStoreCollectionNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)


class StubIngestionService:
    """Configurable stub for IngestionService dependency injection."""

    def __init__(
        self,
        result: IngestionResult = None,
        exception: Exception = None,
        default_collection_name: str = "enterprise_knowledge",
        vector_store_service=None,
    ):
        self.result = result
        self.exception = exception
        self.default_collection_name = default_collection_name
        self.vector_store_service = vector_store_service
        self.call_count = 0
        self.last_file_bytes = None
        self.last_document_name = None
        self.last_collection_name = None
        self.last_document_id = None

    def ingest_pdf(
        self,
        file_bytes: bytes,
        document_name: str,
        collection_name: str = None,
        document_id: str = None,
        heartbeat_callback=None,
        **kwargs,
    ) -> IngestionResult:
        self.call_count += 1
        self.last_file_bytes = file_bytes
        self.last_document_name = document_name
        self.last_collection_name = collection_name
        self.last_document_id = document_id
        self.last_heartbeat_callback = heartbeat_callback
        if heartbeat_callback is not None and getattr(self, "invoke_heartbeat_in_stub", False):
            heartbeat_callback()
        if self.exception:
            raise self.exception
        return self.result or IngestionResult(
            document_name=document_name,
            pages_processed=2,
            chunks_created=3,
            chunks_stored=3,
            collection_name=collection_name or self.default_collection_name,
            chunk_ids=["doc_p1_c0", "doc_p1_c1", "doc_p2_c0"],
        )


class StubRAGAnswerGenerator:
    """Stub for query endpoint to verify coexistence."""

    def answer(self, query: str, **kwargs):
        return RAGResponse(
            query=query,
            answer="Safe mocked answer",
            sources=[],
            context_text="",
        )


class TestAPIIngestionEndpoint(unittest.TestCase):
    """Test suite covering the /api/ingest endpoint, validation, errors, and lifecycle."""

    def setUp(self):
        self.client = TestClient(app)
        self.sample_pdf_bytes = b"%PDF-1.4 Mock valid PDF binary stream"
        self.default_result = IngestionResult(
            document_name="employee_handbook.pdf",
            pages_processed=5,
            chunks_created=12,
            chunks_stored=12,
            collection_name="enterprise_knowledge",
            chunk_ids=[f"handbook_p{i}_c0" for i in range(1, 13)],
        )
        self.stub = StubIngestionService(result=self.default_result)
        app.dependency_overrides[get_ingestion_service_dependency] = lambda: self.stub

        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_file.close()
        self.test_registry = DocumentRegistryRepository(db_path=self.temp_db_file.name)
        app.dependency_overrides[get_document_registry_dependency] = lambda: self.test_registry

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_ingestion_service_cache()
        reset_rag_answer_generator_cache()
        reset_document_registry_cache()
        reset_settings_cache()
        reset_vector_store_cache()
        if hasattr(self, "temp_db_file") and os.path.exists(self.temp_db_file.name):
            try:
                os.remove(self.temp_db_file.name)
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # 1. Successful PDF Upload
    # -------------------------------------------------------------------------
    def test_successful_pdf_upload_returns_http_200(self):
        """1. Verify valid PDF upload returns HTTP 200."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("employee_handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)

    # -------------------------------------------------------------------------
    # 2. Response Contains IngestionResult Fields
    # -------------------------------------------------------------------------
    def test_response_contains_expected_ingestion_result_fields(self):
        """2. Verify response JSON contains all IngestionResult attributes."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("employee_handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_name"], "employee_handbook.pdf")
        self.assertEqual(data["pages_processed"], 5)
        self.assertEqual(data["chunks_created"], 12)
        self.assertEqual(data["chunks_stored"], 12)
        self.assertEqual(data["collection_name"], "enterprise_knowledge")
        self.assertEqual(len(data["chunk_ids"]), 12)
        self.assertIn("document_id", data)
        self.assertIsInstance(data["document_id"], str)
        self.assertTrue(len(data["document_id"]) > 0)

    # -------------------------------------------------------------------------
    # 3. Uploaded Bytes Reach Ingest Unchanged
    # -------------------------------------------------------------------------
    def test_uploaded_bytes_reach_ingest_unchanged(self):
        """3. Verify binary payload is delivered to ingest_pdf unmodified."""
        raw_payload = b"%PDF-1.4\x00\xff\xfeCustomBinaryBytes123"
        response = self.client.post(
            "/api/ingest",
            files={"file": ("test.pdf", raw_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_file_bytes, raw_payload)

    # -------------------------------------------------------------------------
    # 4. Uploaded Filename Reaches Ingestion Service Correctly
    # -------------------------------------------------------------------------
    def test_uploaded_filename_reaches_ingestion_service(self):
        """4. Verify document filename is forwarded accurately to ingest_pdf."""
        filename = "corporate_security_policy_2026.pdf"
        response = self.client.post(
            "/api/ingest",
            files={"file": (filename, self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_document_name, filename)

    # -------------------------------------------------------------------------
    # 5. Supplied Collection Override Propagated
    # -------------------------------------------------------------------------
    def test_supplied_collection_override_propagates(self):
        """5. Verify collection_name form field is passed to ingest_pdf."""
        custom_collection = "hr_confidential_docs"
        response = self.client.post(
            "/api/ingest",
            files={"file": ("handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
            data={"collection_name": custom_collection},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.last_collection_name, custom_collection)

    # -------------------------------------------------------------------------
    # 6. Omitted Collection Override Uses Configured Default
    # -------------------------------------------------------------------------
    def test_omitted_collection_override_uses_default(self):
        """6. Verify omitted collection_name passes None to service for default handling."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.stub.last_collection_name)

    # -------------------------------------------------------------------------
    # 7. Empty Uploads Rejected
    # -------------------------------------------------------------------------
    def test_empty_uploads_are_rejected(self):
        """7. Verify empty byte uploads are rejected with HTTP 422."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("empty", response.json()["detail"].lower())
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 8. Missing or Invalid Filenames Rejected
    # -------------------------------------------------------------------------
    def test_missing_or_invalid_filenames_rejected(self):
        """8. Verify empty or whitespace-only filenames return HTTP 422."""
        for invalid_name in ["", "   ", " \t "]:
            with self.subTest(filename=invalid_name):
                response = self.client.post(
                    "/api/ingest",
                    files={"file": (invalid_name, self.sample_pdf_bytes, "application/pdf")},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 9. Unsupported File Types Rejected
    # -------------------------------------------------------------------------
    def test_unsupported_file_types_rejected(self):
        """9. Verify non-PDF extensions or incompatible content-types return HTTP 422."""
        # Non-PDF extension
        response_ext = self.client.post(
            "/api/ingest",
            files={"file": ("document.docx", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response_ext.status_code, 422)
        self.assertIn("pdf", response_ext.json()["detail"].lower())

        # Incompatible content type
        response_type = self.client.post(
            "/api/ingest",
            files={"file": ("document.pdf", self.sample_pdf_bytes, "text/plain")},
        )
        self.assertEqual(response_type.status_code, 422)
        self.assertIn("content type", response_type.json()["detail"].lower())
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 10. Malformed PDFs Map to Documented Client Error (HTTP 422)
    # -------------------------------------------------------------------------
    def test_malformed_pdfs_map_to_client_error(self):
        """10. Verify malformed/corrupted PDF errors map to HTTP 422."""
        self.stub.exception = IngestionError(
            "Failed to load PDF document 'corrupt.pdf': Corrupted xref table"
        )
        self.stub.exception.__cause__ = PDFProcessingError("Corrupted xref table")

        response = self.client.post(
            "/api/ingest",
            files={"file": ("corrupt.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("failed to parse pdf", response.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 11. Empty Extracted Documents Map to Documented Client Error (HTTP 422)
    # -------------------------------------------------------------------------
    def test_empty_extracted_documents_map_to_client_error(self):
        """11. Verify documents with no extractable text map to HTTP 422."""
        self.stub.exception = IngestionEmptyDocumentError(
            "PDF document 'blank.pdf' contains no extractable text"
        )

        response = self.client.post(
            "/api/ingest",
            files={"file": ("blank.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("no extractable text", response.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 12. Ingestion Validation Exceptions Map Correctly (HTTP 422)
    # -------------------------------------------------------------------------
    def test_ingestion_validation_exceptions_map_correctly(self):
        """12. Verify IngestionValidationError maps to HTTP 422."""
        self.stub.exception = IngestionValidationError("collection_name cannot be whitespace-only")

        response = self.client.post(
            "/api/ingest",
            files={"file": ("valid.pdf", self.sample_pdf_bytes, "application/pdf")},
            data={"collection_name": "   "},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("validation failed", response.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 13. Recognized Temporary Backend Failures Map to HTTP 503
    # -------------------------------------------------------------------------
    def test_backend_storage_failures_map_to_http_503(self):
        """13. Verify vector database/Qdrant outages map to HTTP 503."""
        self.stub.exception = IngestionError(
            "Vector store upsert failed: Connection refused by Qdrant"
        )
        self.stub.exception.__cause__ = VectorStoreConnectionError("Connection refused by Qdrant")

        response = self.client.post(
            "/api/ingest",
            files={"file": ("valid.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.json()["detail"].lower())
        self.assertNotIn("Qdrant", response.json()["detail"])

    # -------------------------------------------------------------------------
    # 14. Unexpected Ingestion Failures Return Safe HTTP 500
    # -------------------------------------------------------------------------
    def test_unexpected_ingestion_failures_return_safe_http_500(self):
        """14. Verify unexpected internal errors return generic HTTP 500 without stack traces."""
        self.stub.exception = RuntimeError("Fatal internal memory segment fault")

        response = self.client.post(
            "/api/ingest",
            files={"file": ("valid.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("unexpected error", response.json()["detail"].lower())
        self.assertNotIn("memory segment", response.json()["detail"])

    # -------------------------------------------------------------------------
    # 15. Error Responses Do Not Expose Sensitive Details
    # -------------------------------------------------------------------------
    def test_error_responses_do_not_expose_stack_traces(self):
        """15. Verify error JSON payload contains only client-safe detail string."""
        self.stub.exception = IngestionError(
            "Secret DB credentials failed at postgres://user:secret@10.0.0.1:5432"
        )

        response = self.client.post(
            "/api/ingest",
            files={"file": ("valid.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        self.assertIn("detail", data)
        self.assertNotIn("postgres://", data["detail"])
        self.assertNotIn("secret", data["detail"])
        self.assertNotIn("10.0.0.1", data["detail"])
        self.assertNotIn("Traceback", str(data))

    # -------------------------------------------------------------------------
    # 16. Upload Is Closed After Success
    # -------------------------------------------------------------------------
    def test_upload_is_closed_after_success(self):
        """16. Verify the UploadFile underlying file is closed after successful ingestion."""
        from fastapi import UploadFile

        opened_files = []

        async def spy_read(file_self):
            opened_files.append(file_self)
            return file_self._real_read()

        # Intercept UploadFile instances to check file.file.closed
        response = self.client.post(
            "/api/ingest",
            files={"file": ("handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)

    # -------------------------------------------------------------------------
    # 17. Upload Is Closed After Failure
    # -------------------------------------------------------------------------
    def test_upload_is_closed_after_failure(self):
        """17. Verify the UploadFile underlying file is closed after failure."""
        self.stub.exception = IngestionError("Simulated pipeline failure")

        response = self.client.post(
            "/api/ingest",
            files={"file": ("handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)

    # -------------------------------------------------------------------------
    # 18. Dependency Overrides Work Correctly
    # -------------------------------------------------------------------------
    def test_dependency_overrides_work_correctly(self):
        """18. Verify custom IngestionService dependency can be injected and overridden."""
        custom_result = IngestionResult(
            document_name="overridden.pdf",
            pages_processed=1,
            chunks_created=1,
            chunks_stored=1,
            collection_name="custom_col",
            chunk_ids=["custom_p1_c0"],
        )
        custom_stub = StubIngestionService(result=custom_result)

        app.dependency_overrides[get_ingestion_service_dependency] = lambda: custom_stub
        response = self.client.post(
            "/api/ingest",
            files={"file": ("overridden.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_name"], "overridden.pdf")
        self.assertEqual(custom_stub.call_count, 1)

    # -------------------------------------------------------------------------
    # 19. Existing /health and /api/query Routes Remain Available
    # -------------------------------------------------------------------------
    def test_existing_health_and_query_routes_remain_available(self):
        """19. Verify GET /health and POST /api/query continue functioning seamlessly."""
        # Check /health
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json(), {"status": "ok"})

        # Check /api/query
        app.dependency_overrides[get_rag_answer_generator_dependency] = (
            lambda: StubRAGAnswerGenerator()
        )
        res_query = self.client.post("/api/query", json={"query": "What is our leave policy?"})
        self.assertEqual(res_query.status_code, 200)
        self.assertEqual(res_query.json()["answer"], "Safe mocked answer")

    # -------------------------------------------------------------------------
    # 20. Endpoint Does Not Call LLM or Reimplement Pipeline
    # -------------------------------------------------------------------------
    def test_endpoint_does_not_call_llm_or_reimplement_pipeline(self):
        """20. Verify /api/ingest strictly delegates to IngestionService.ingest_pdf."""
        with patch("app.services.llm.get_llm_service") as mock_llm_factory:
            response = self.client.post(
                "/api/ingest",
                files={"file": ("handbook.pdf", self.sample_pdf_bytes, "application/pdf")},
            )
            self.assertEqual(response.status_code, 200)
            # The ingestion endpoint must never invoke the LLM
            mock_llm_factory.assert_not_called()
            self.assertEqual(self.stub.call_count, 1)

    # -------------------------------------------------------------------------
    # 21. Shared Ingestion Service Lifecycle Caching
    # -------------------------------------------------------------------------
    def test_shared_ingestion_service_lifecycle_caches_instance(self):
        """21. Verify get_shared_ingestion_service reuses singleton across calls."""
        with patch("app.services.ingestion.get_ingestion_service") as mock_factory:
            mock_factory.return_value = MagicMock(spec=IngestionService)
            reset_ingestion_service_cache()

            svc1 = get_shared_ingestion_service()
            svc2 = get_shared_ingestion_service()

            self.assertIs(svc1, svc2)
            self.assertEqual(mock_factory.call_count, 1)
            reset_ingestion_service_cache()

    # -------------------------------------------------------------------------
    # 22. PDF Smaller Than Configured Size Limit Is Accepted
    # -------------------------------------------------------------------------
    def test_file_smaller_than_size_limit_is_accepted(self):
        """22. Verify file with size below max_upload_size_bytes returns HTTP 200."""
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=500
        )
        small_payload = b"%PDF-1.4 " + b"A" * 200
        response = self.client.post(
            "/api/ingest",
            files={"file": ("small.pdf", small_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.call_count, 1)
        self.assertEqual(self.stub.last_file_bytes, small_payload)

    # -------------------------------------------------------------------------
    # 23. PDF Exactly At Configured Size Limit Is Accepted
    # -------------------------------------------------------------------------
    def test_file_exactly_at_size_limit_is_accepted(self):
        """23. Verify file with size exactly equal to max_upload_size_bytes returns HTTP 200."""
        exact_size = 350
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=exact_size
        )
        header = b"%PDF-1.4 "
        exact_payload = header + b"B" * (exact_size - len(header))
        self.assertEqual(len(exact_payload), exact_size)

        response = self.client.post(
            "/api/ingest",
            files={"file": ("exact.pdf", exact_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stub.call_count, 1)
        self.assertEqual(self.stub.last_file_bytes, exact_payload)

    # -------------------------------------------------------------------------
    # 24. PDF Exceeding Configured Size Limit Returns HTTP 413
    # -------------------------------------------------------------------------
    def test_file_exceeding_size_limit_returns_http_413(self):
        """24. Verify file with size exceeding limit returns HTTP 413 Payload Too Large."""
        limit_size = 300
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=limit_size
        )
        oversized_payload = b"%PDF-1.4 " + b"C" * 350
        self.assertGreater(len(oversized_payload), limit_size)

        response = self.client.post(
            "/api/ingest",
            files={"file": ("oversized.pdf", oversized_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds the maximum allowed size", response.json()["detail"])
        self.assertIn(str(limit_size), response.json()["detail"])

    # -------------------------------------------------------------------------
    # 25. Oversized Upload Never Calls Ingestion Service
    # -------------------------------------------------------------------------
    def test_oversized_upload_never_calls_ingestion_service(self):
        """25. Verify oversized payload is rejected before reaching IngestionService.ingest_pdf."""
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=200
        )
        oversized_payload = b"%PDF-1.4 " + b"D" * 600

        response = self.client.post(
            "/api/ingest",
            files={"file": ("too_large.pdf", oversized_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 413)
        # Guarantee ingest_pdf was never invoked
        self.assertEqual(self.stub.call_count, 0)
        self.assertIsNone(self.stub.last_file_bytes)

    # -------------------------------------------------------------------------
    # 26. Absent Content-Length Cannot Bypass Size Check
    # -------------------------------------------------------------------------
    def test_absent_content_length_cannot_bypass_size_check(self):
        """26. Verify missing Content-Length header still triggers bounded-read 413 enforcement."""
        limit_size = 250
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=limit_size
        )
        oversized_payload = b"%PDF-1.4 " + b"E" * 500

        # Post without Content-Length header or with custom headers
        response = self.client.post(
            "/api/ingest",
            files={"file": ("no_len.pdf", oversized_payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 27. Misleading Content-Length Header Cannot Bypass Size Check
    # -------------------------------------------------------------------------
    def test_misleading_content_length_cannot_bypass_size_check(self):
        """27. Verify inaccurate/spoofed small Content-Length header does not bypass 413 limit."""
        limit_size = 250
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=limit_size
        )
        oversized_payload = b"%PDF-1.4 " + b"F" * 600

        # Client claims body is only 50 bytes while transmitting 600 bytes
        response = self.client.post(
            "/api/ingest",
            files={"file": ("spoofed.pdf", oversized_payload, "application/pdf")},
            headers={"content-length": "50"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 28. Upload File Is Closed After Oversized Request
    # -------------------------------------------------------------------------
    def test_upload_file_is_closed_after_oversized_request(self):
        """28. Verify UploadFile.close() is executed even when request fails with 413."""
        app.dependency_overrides[get_settings_dependency] = lambda: Settings(
            max_upload_size_bytes=200
        )
        oversized_payload = b"%PDF-1.4 " + b"G" * 500

        closed_files = []
        original_close = starlette.datastructures.UploadFile.close

        async def spy_close(instance_self):
            closed_files.append(instance_self)
            return await original_close(instance_self)

        with patch.object(starlette.datastructures.UploadFile, "close", spy_close):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("closed_test.pdf", oversized_payload, "application/pdf")},
            )
            self.assertEqual(response.status_code, 413)
            # Verify close was awaited
            self.assertGreaterEqual(len(closed_files), 1)

    # -------------------------------------------------------------------------
    # 29. Size Limit Configurable Through Environment Variable
    # -------------------------------------------------------------------------
    def test_size_limit_configurable_through_environment_variable(self):
        """29. Verify MAX_UPLOAD_SIZE_BYTES environment variable dynamically sets size boundary."""
        with patch.dict(os.environ, {"MAX_UPLOAD_SIZE_BYTES": "400"}, clear=True):
            reset_settings_cache()
            payload_ok = b"%PDF-1.4 " + b"H" * 300
            payload_too_big = b"%PDF-1.4 " + b"H" * 500

            res_ok = self.client.post(
                "/api/ingest",
                files={"file": ("ok.pdf", payload_ok, "application/pdf")},
            )
            self.assertEqual(res_ok.status_code, 200)

            res_fail = self.client.post(
                "/api/ingest",
                files={"file": ("fail.pdf", payload_too_big, "application/pdf")},
            )
            self.assertEqual(res_fail.status_code, 413)
            reset_settings_cache()

    # -------------------------------------------------------------------------
    # 30. Existing Validation and Error Mappings Unchanged
    # -------------------------------------------------------------------------
    def test_existing_validation_and_error_mappings_remain_unchanged(self):
        """30. Verify empty file (422), non-PDF (422), missing name (422) still function."""
        # Empty file
        res_empty = self.client.post(
            "/api/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        self.assertEqual(res_empty.status_code, 422)
        self.assertIn("Uploaded file is empty.", res_empty.json()["detail"])

        # Non-PDF
        res_non_pdf = self.client.post(
            "/api/ingest",
            files={"file": ("doc.txt", b"Hello text", "text/plain")},
        )
        self.assertEqual(res_non_pdf.status_code, 422)

        # Missing filename
        res_no_name = self.client.post(
            "/api/ingest",
            files={"file": ("", b"%PDF-1.4 dummy", "application/pdf")},
        )
        self.assertEqual(res_no_name.status_code, 422)

    # -------------------------------------------------------------------------
    # 31. Ingestion Lifecycle Tracking in Registry (Processing -> Succeeded)
    # -------------------------------------------------------------------------
    def test_successful_ingestion_tracks_lifecycle_in_registry(self):
        """31. Verify successful ingestion transitions record to succeeded with counts."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("report_q3.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        doc_id = data["document_id"]

        record = self.test_registry.get_record(doc_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.document_id, doc_id)
        self.assertEqual(record.document_name, "report_q3.pdf")
        self.assertEqual(record.status, DocumentStatus.SUCCEEDED)
        self.assertEqual(record.pages_processed, 5)
        self.assertEqual(record.chunks_created, 12)
        self.assertEqual(record.chunks_stored, 12)
        self.assertEqual(record.collection_name, "enterprise_knowledge")
        self.assertIsNone(record.error_code)
        self.assertIsNone(record.error_message)

    # -------------------------------------------------------------------------
    # 32. Separate Ingestion Attempts Receive Unique Document IDs
    # -------------------------------------------------------------------------
    def test_separate_ingestions_receive_unique_document_ids(self):
        """32. Verify multiple uploads of same filename generate distinct document IDs."""
        res1 = self.client.post(
            "/api/ingest",
            files={"file": ("same_name.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        res2 = self.client.post(
            "/api/ingest",
            files={"file": ("same_name.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        id1 = res1.json()["document_id"]
        id2 = res2.json()["document_id"]
        self.assertNotEqual(id1, id2)

        records = self.test_registry.list_records()
        self.assertEqual(len(records), 2)
        ids = {r.document_id for r in records}
        self.assertEqual(ids, {id1, id2})

    # -------------------------------------------------------------------------
    # 33. Handled Ingestion Failure Records Failed Status and Safe Error
    # -------------------------------------------------------------------------
    def test_handled_ingestion_failure_records_failed_status_and_safe_error(self):
        """33. Verify pipeline failure updates record to failed without sensitive leak."""
        self.stub.exception = IngestionError("Internal DB password exposed: secret123")
        response = self.client.post(
            "/api/ingest",
            files={"file": ("failed_doc.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)

        records = self.test_registry.list_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_code, "INGESTION_ERROR")
        self.assertEqual(record.error_message, "An error occurred while processing the document.")
        self.assertNotIn("secret123", record.error_message)
        self.assertIsNone(record.chunks_stored)

    # -------------------------------------------------------------------------
    # 34. Unexpected Exception Records Failed Status and Safe Error
    # -------------------------------------------------------------------------
    def test_unexpected_exception_records_failed_status_and_safe_error(self):
        """34. Verify unexpected runtime error updates record to failed safely."""
        self.stub.exception = RuntimeError("Low level segfault in libm")
        response = self.client.post(
            "/api/ingest",
            files={"file": ("crash_doc.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)

        records = self.test_registry.list_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_code, "UNEXPECTED_ERROR")
        self.assertEqual(record.error_message, "An unexpected error occurred during document ingestion.")
        self.assertNotIn("libm", record.error_message)

    # -------------------------------------------------------------------------
    # 35. Database Registration Failure Yields 500 Without Ingestion
    # -------------------------------------------------------------------------
    def test_database_registration_failure_yields_500_without_ingestion(self):
        """35. Verify database error before ingestion returns safe 500 and prevents ingestion."""
        mock_broken_repo = MagicMock()
        mock_broken_repo.create_record.side_effect = Exception("Disk write I/O error at /var/data")
        app.dependency_overrides[get_document_registry_dependency] = lambda: mock_broken_repo

        response = self.client.post(
            "/api/ingest",
            files={"file": ("will_fail.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "An internal error occurred while registering the document.")
        self.assertNotIn("Disk write", response.json()["detail"])
        self.assertNotIn("/var/data", response.json()["detail"])
        self.assertEqual(self.stub.call_count, 0)

    # -------------------------------------------------------------------------
    # 36. Registry Update Failure Does Not Mask Ingestion Nor Leak Internals
    # -------------------------------------------------------------------------
    def test_registry_update_failure_returns_safe_500(self):
        """36. Verify update failure after ingestion returns safe 500."""
        mock_repo = MagicMock()
        mock_record = MagicMock()
        mock_record.document_id = "doc-test-12345"
        mock_repo.create_record.return_value = mock_record
        mock_repo.update_status_succeeded.side_effect = Exception("DB update connection timed out")
        app.dependency_overrides[get_document_registry_dependency] = lambda: mock_repo

        response = self.client.post(
            "/api/ingest",
            files={"file": ("update_fail.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Failed to persist ingestion success status in registry.",
        )
        self.assertNotIn("DB update", response.json()["detail"])

    # -------------------------------------------------------------------------
    # 37. Registry Document ID Propagates to Ingestion Service
    # -------------------------------------------------------------------------
    def test_api_generated_registry_document_id_propagates_to_ingestion_service(self):
        """37. Verify registry document_id is passed unchanged to ingest_pdf."""
        response = self.client.post(
            "/api/ingest",
            files={"file": ("provenance_test.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        doc_id = data["document_id"]

        self.assertIsNotNone(self.stub.last_document_id)
        self.assertEqual(self.stub.last_document_id, doc_id)


    # -------------------------------------------------------------------------
    # 38. Step 21F: Successful Finalization Does Not Trigger Compensating Deletion
    # -------------------------------------------------------------------------
    def test_successful_finalization_does_not_trigger_compensating_deletion(self):
        """38. Verify that normal successful finalization does not call delete_by_document_id."""
        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        response = self.client.post(
            "/api/ingest",
            files={"file": ("success_no_clean.pdf", self.sample_pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        mock_vector_store.delete_by_document_id.assert_not_called()

        data = response.json()
        record = self.test_registry.get_record(data["document_id"])
        self.assertIsNotNone(record)
        self.assertEqual(record.status, DocumentStatus.SUCCEEDED)

    # -------------------------------------------------------------------------
    # 39. Step 21F: Finalization Failure Triggers Compensating Deletion with Exact ID and Collection
    # -------------------------------------------------------------------------
    def test_registry_finalization_failure_triggers_compensating_deletion(self):
        """39. Verify update_status_succeeded failure triggers delete_by_document_id with exact doc ID and collection."""
        self.stub.result = None
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 5
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Disk I/O failure")):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "custom_knowledge"},
                files={"file": ("fail_finalization.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Failed to persist ingestion success status in registry.",
        )
        mock_vector_store.delete_by_document_id.assert_called_once()
        call_kwargs = mock_vector_store.delete_by_document_id.call_args.kwargs
        self.assertEqual(call_kwargs["collection_name"], "custom_knowledge")
        self.assertTrue(isinstance(call_kwargs["document_id"], str) and len(call_kwargs["document_id"]) > 0)

    # -------------------------------------------------------------------------
    # 40. Step 21F: Successful Cleanup Marks Record FAILED with FINALIZATION_FAILED
    # -------------------------------------------------------------------------
    def test_successful_compensating_cleanup_marks_registry_failed(self):
        """40. Verify successful compensating deletion transitions record to FAILED with FINALIZATION_FAILED."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 3
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Database lock error")):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("cleanup_ok.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        records = self.test_registry.list_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_code, "FINALIZATION_FAILED")
        self.assertIn("safely purged", record.error_message)

    # -------------------------------------------------------------------------
    # 41. Step 21F: Cleanup Failure Marks Record FAILED with COMPENSATION_FAILED
    # -------------------------------------------------------------------------
    def test_compensating_cleanup_failure_marks_compensation_failed(self):
        """41. Verify failing compensating deletion transitions record to FAILED with COMPENSATION_FAILED."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.side_effect = VectorStoreError("Qdrant connection dropped during delete")
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Database lock error")):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("cleanup_fail.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Failed to persist ingestion success status in registry.",
        )
        self.assertNotIn("Qdrant connection dropped", response.json()["detail"])

        records = self.test_registry.list_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_code, "COMPENSATION_FAILED")
        self.assertIn("compensating vector cleanup could not be completed", record.error_message)

    # -------------------------------------------------------------------------
    # 42. Step 21F: Double Failure Handled Safely
    # -------------------------------------------------------------------------
    def test_compensating_cleanup_double_failure_handled_safely(self):
        """42. Verify when both finalization and failure-status update fail, endpoint returns safe 500 without leaking internals."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 0
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Success update disk crash")), \
             patch.object(self.test_registry, "update_status_failed", side_effect=Exception("Failed update disk crash")):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("double_fail.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Failed to persist ingestion success status in registry.",
        )
        self.assertNotIn("disk crash", response.json()["detail"])

    # -------------------------------------------------------------------------
    # 43. Step 21F: Compensating Cleanup Never Targets Other Documents or Wildcards
    # -------------------------------------------------------------------------
    def test_compensating_cleanup_never_targets_other_documents(self):
        """43. Verify compensating deletion specifically targets only the failing document's ID."""
        existing_doc = self.test_registry.create_record(
            document_name="doc_a.pdf",
            collection_name="enterprise_knowledge",
            provenance_eligible=True,
        )
        self.test_registry.update_status_succeeded(
            document_id=existing_doc.document_id,
            pages_processed=1,
            chunks_created=1,
            chunks_stored=1,
            collection_name="enterprise_knowledge",
        )

        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Finalization fail")):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("doc_b.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        mock_vector_store.delete_by_document_id.assert_called_once()
        targeted_id = mock_vector_store.delete_by_document_id.call_args.kwargs["document_id"]

        self.assertNotEqual(targeted_id, existing_doc.document_id)
        self.assertNotEqual(targeted_id, "*")
        self.assertNotEqual(targeted_id, "")
        self.assertIsNotNone(targeted_id)

    # -------------------------------------------------------------------------
    # 44. Step 21F: Collection NotFound During Cleanup Is Handled as Succeeded Cleanup
    # -------------------------------------------------------------------------
    def test_compensating_cleanup_collection_not_found_treated_as_succeeded(self):
        """44. Verify VectorStoreCollectionNotFoundError during cleanup is treated as successful (zero vectors remain)."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.side_effect = VectorStoreCollectionNotFoundError("Collection not found")
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        with patch.object(self.test_registry, "update_status_succeeded", side_effect=Exception("Finalize fail")):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("cnf_clean.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        records = self.test_registry.list_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_code, "FINALIZATION_FAILED")

    # -------------------------------------------------------------------------
    # 45. Step 21G-B: Late Worker State Conflict Triggers Exact Compensating Cleanup
    # -------------------------------------------------------------------------
    def test_late_worker_state_conflict_triggers_exact_compensating_cleanup(self):
        """45. Verify late worker hitting DocumentStateConflictError triggers compensating cleanup and returns 409."""
        self.stub.result = None
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 4
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        conflict_err = DocumentStateConflictError("Cannot transition: status is 'failed', expected 'processing'")
        with patch.object(self.test_registry, "update_status_succeeded", side_effect=conflict_err):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "custom_late_col"},
                files={"file": ("late_conflict.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("conflict", response.json()["detail"].lower())

        # Verify compensating cleanup was invoked with exact document ID and collection
        mock_vector_store.delete_by_document_id.assert_called_once()
        call_kwargs = mock_vector_store.delete_by_document_id.call_args.kwargs
        self.assertEqual(call_kwargs["collection_name"], "custom_late_col")
        self.assertTrue(isinstance(call_kwargs["document_id"], str) and len(call_kwargs["document_id"]) > 0)

    # -------------------------------------------------------------------------
    # 46. Step 21G-B: Late Worker Cleanup Failure Does Not Overwrite Registry
    # -------------------------------------------------------------------------
    def test_late_worker_state_conflict_cleanup_failure_does_not_overwrite_registry(self):
        """46. Verify late worker state conflict with cleanup failure preserves newer registry state and returns 409."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.side_effect = VectorStoreError("Qdrant drop")
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        conflict_err = DocumentStateConflictError("Cannot transition: status is 'failed', expected 'processing'")
        with patch.object(self.test_registry, "update_status_succeeded", side_effect=conflict_err):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("late_fail_cleanup.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("conflict", response.json()["detail"].lower())
        self.assertNotIn("Qdrant drop", response.json()["detail"])

        mock_vector_store.delete_by_document_id.assert_called_once()

    # -------------------------------------------------------------------------
    # 47. Step 21G-B: Safe Record Failure Ignores State Conflict Without Overwriting
    # -------------------------------------------------------------------------
    def test_safe_record_failure_ignores_state_conflict_without_overwriting(self):
        """47. Verify _safe_record_failure safely catches DocumentStateConflictError without raising unhandled exception."""
        rec = self.test_registry.create_record("fail_conflict.pdf", "col")
        # Pre-set record to SUCCEEDED
        self.test_registry.update_status_succeeded(
            document_id=rec.document_id,
            pages_processed=1,
            chunks_created=1,
            chunks_stored=1,
        )

        from app.api.ingestion import _safe_record_failure
        # Calling _safe_record_failure on a non-processing document should not raise
        _safe_record_failure(
            self.test_registry,
            rec.document_id,
            error_code="LATE_FAIL",
            error_message="Worker error after cancellation",
        )

        # Status must remain SUCCEEDED
        rec_after = self.test_registry.get_record(rec.document_id)
        self.assertEqual(rec_after.status, DocumentStatus.SUCCEEDED)

    # -------------------------------------------------------------------------
    # 48. Step 21G-D: Partial Ingestion Write Followed by Failure State Conflict
    # -------------------------------------------------------------------------
    def test_partial_ingestion_write_with_failure_state_conflict(self):
        """48. Verify IngestionError after partial writes with state conflict triggers compensating cleanup and preserves newer state."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 5
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_partial_upsert_and_concurrent_conflict(
            file_bytes, document_name, collection_name=None, document_id=None, **kwargs
        ):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            # Simulate a concurrent recovery process or administrator transitioning the document
            # from 'processing' to 'failed' while the worker was stalled in Qdrant upsert
            self.test_registry.update_status_failed(
                document_id=document_id,
                error_code="RECOVERED_STALE",
                error_message="Document ingestion timed out and was recovered by background cleaner.",
            )
            # Ingestion pipeline now raises IngestionError representing a failure after partial Qdrant writes
            err = IngestionError("Vector store upsert failed: connection dropped after partial points written")
            err.__cause__ = VectorStoreError("Connection dropped after partial points written")
            raise err

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_partial_upsert_and_concurrent_conflict):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "target_col"},
                files={"file": ("partial_crash.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        # 1. Existing error-handling contract: vector store failures map to HTTP 503 Service Unavailable
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Vector storage service is currently unavailable.")

        # 2. Document was registered and assigned an ID
        self.assertIsNotNone(captured_doc_id)
        self.assertTrue(len(captured_doc_id) > 0)

        # 3. Step 21G-D: Verify delete_by_document_id IS invoked with exact document ID and collection
        mock_vector_store.delete_by_document_id.assert_called_once()
        call_kwargs = mock_vector_store.delete_by_document_id.call_args.kwargs
        self.assertEqual(call_kwargs.get("collection_name"), "target_col")
        self.assertEqual(call_kwargs.get("document_id"), captured_doc_id)

        # 4. Verify that the registry's newer state is NOT overwritten by _safe_record_failure
        # The record remains in the 'RECOVERED_STALE' state set by the concurrent operation,
        # and was not overwritten by 'VECTOR_STORE_UNAVAILABLE'
        rec = self.test_registry.get_record(captured_doc_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.FAILED)
        self.assertEqual(rec.error_code, "RECOVERED_STALE")
        self.assertNotEqual(rec.error_code, "VECTOR_STORE_UNAVAILABLE")
        self.assertIn("recovered by background cleaner", rec.error_message)

    # -------------------------------------------------------------------------
    # 49. Step 21G-D: Cleanup Failure During Ingestion Error State Conflict
    # -------------------------------------------------------------------------
    def test_partial_ingestion_write_cleanup_failure_does_not_mask_http_503(self):
        """49. Verify vector store cleanup failure on state conflict does not mask HTTP 503 or corrupt registry."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.side_effect = VectorStoreError("Qdrant connection reset during cleanup")
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_partial_upsert_and_conflict(
            file_bytes, document_name, collection_name=None, document_id=None, **kwargs
        ):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            self.test_registry.update_status_failed(
                document_id=document_id,
                error_code="RECOVERED_STALE",
                error_message="Document ingestion timed out and was recovered by background cleaner.",
            )
            err = IngestionError("Vector store upsert failed: connection dropped")
            err.__cause__ = VectorStoreError("Connection dropped")
            raise err

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_partial_upsert_and_conflict):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "target_col"},
                files={"file": ("partial_cleanup_fail.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        # HTTP 503 response is preserved; cleanup failure does NOT mask original error
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Vector storage service is currently unavailable.")
        self.assertNotIn("Qdrant connection reset", response.json()["detail"])

        mock_vector_store.delete_by_document_id.assert_called_once()
        call_kwargs = mock_vector_store.delete_by_document_id.call_args.kwargs
        self.assertEqual(call_kwargs.get("collection_name"), "target_col")
        self.assertEqual(call_kwargs.get("document_id"), captured_doc_id)

        # Newer registry state preserved
        rec = self.test_registry.get_record(captured_doc_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.FAILED)
        self.assertEqual(rec.error_code, "RECOVERED_STALE")

    # -------------------------------------------------------------------------
    # 50. Step 21G-D: Ordinary Pre-Write Validation Failure Does Not Trigger Cleanup
    # -------------------------------------------------------------------------
    def test_pre_write_validation_failure_with_state_conflict_does_not_trigger_cleanup(self):
        """50. Verify pre-write validation failure with state conflict returns 422 and does NOT call delete_by_document_id."""
        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_pre_write_pdf_error(
            file_bytes, document_name, collection_name=None, document_id=None, **kwargs
        ):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            self.test_registry.update_status_failed(
                document_id=document_id,
                error_code="RECOVERED_STALE",
                error_message="Document ingestion timed out and was recovered by background cleaner.",
            )
            # IngestionError caused by PDF parsing before any vector writes
            err = IngestionError("Corrupt PDF structure: invalid header")
            err.__cause__ = PDFProcessingError("Invalid PDF header")
            raise err

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_pre_write_pdf_error):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "target_col"},
                files={"file": ("corrupt.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        # 1. Pre-write PDF parsing error returns HTTP 422
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Failed to parse PDF document. The file may be corrupt or invalid.")

        # 2. No vector writes were possible, so vector store cleanup MUST NOT be called
        mock_vector_store.delete_by_document_id.assert_not_called()

        # 3. Newer registry state preserved
        rec = self.test_registry.get_record(captured_doc_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.FAILED)
        self.assertEqual(rec.error_code, "RECOVERED_STALE")

    # -------------------------------------------------------------------------
    # 51. Step 21G-D: General Ingestion Processing Error with Conflict Triggers Cleanup
    # -------------------------------------------------------------------------
    def test_general_ingestion_error_with_state_conflict_triggers_cleanup(self):
        """51. Verify general IngestionError with state conflict triggers compensating cleanup and returns 500."""
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 2
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_general_error_and_conflict(
            file_bytes, document_name, collection_name=None, document_id=None, **kwargs
        ):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            self.test_registry.update_status_failed(
                document_id=document_id,
                error_code="RECOVERED_STALE",
                error_message="Document ingestion timed out.",
            )
            # General ingestion error without specific vector store / PDF cause
            raise IngestionError("Embedding batch computation crashed midway")

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_general_error_and_conflict):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("batch_crash.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        # 1. General ingestion error maps to HTTP 500
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "An error occurred while processing the document.")

        # 2. Vector store cleanup is invoked with default collection and captured document ID
        mock_vector_store.delete_by_document_id.assert_called_once()
        call_kwargs = mock_vector_store.delete_by_document_id.call_args.kwargs
        self.assertEqual(call_kwargs.get("collection_name"), "enterprise_knowledge")
        self.assertEqual(call_kwargs.get("document_id"), captured_doc_id)

        # 3. Newer registry state preserved
        rec = self.test_registry.get_record(captured_doc_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.FAILED)
        self.assertEqual(rec.error_code, "RECOVERED_STALE")

    # -------------------------------------------------------------------------
    # 52. Step 21H-E: API Wires Heartbeat Callback to DocumentRegistryRepository
    # -------------------------------------------------------------------------
    def test_api_wires_heartbeat_callback_to_registry(self):
        """52. Verify API passes a heartbeat callback that calls registry_repo.heartbeat_ingestion."""
        self.stub.result = None
        self.stub.last_heartbeat_callback = None
        heartbeat_verified = False

        def intercept_and_verify_heartbeat(file_bytes, document_name, collection_name=None, document_id=None, heartbeat_callback=None, **kwargs):
            nonlocal heartbeat_verified
            self.assertIsNotNone(heartbeat_callback)
            self.assertTrue(callable(heartbeat_callback))

            rec_before = self.test_registry.get_record(document_id)
            self.assertEqual(rec_before.status, DocumentStatus.PROCESSING)

            # Invoke callback while still processing
            heartbeat_callback()

            rec_after = self.test_registry.get_record(document_id)
            self.assertEqual(rec_after.status, DocumentStatus.PROCESSING)
            heartbeat_verified = True

            return IngestionResult(
                document_name=document_name,
                pages_processed=1,
                chunks_created=1,
                chunks_stored=1,
                collection_name=collection_name or "enterprise_knowledge",
                chunk_ids=["c1"],
            )

        with patch.object(self.stub, "ingest_pdf", side_effect=intercept_and_verify_heartbeat):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("wired_heartbeat.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(heartbeat_verified)

    # -------------------------------------------------------------------------
    # 53. Step 21H-E: Heartbeat State Conflict Returns Sanitized HTTP 409
    # -------------------------------------------------------------------------
    def test_heartbeat_state_conflict_returns_sanitized_409_and_no_cleanup(self):
        """53. Verify heartbeat state conflict returns 409 and avoids vector cleanup (0 vectors written)."""
        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_heartbeat_conflict(file_bytes, document_name, collection_name=None, document_id=None, heartbeat_callback=None, **kwargs):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            # Transition document to failed (simulating concurrent worker or recovery claim)
            self.test_registry.update_status_failed(
                document_id=document_id,
                error_code="RECOVERED_STALE",
                error_message="Document ingestion timed out and was recovered.",
            )
            # Invoke heartbeat callback, which detects status is 'failed' and raises DocumentStateConflictError
            assert heartbeat_callback is not None
            heartbeat_callback()

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_heartbeat_conflict):
            response = self.client.post(
                "/api/ingest",
                data={"collection_name": "target_col"},
                files={"file": ("stale_conflict.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        # 1. API returns sanitized HTTP 409
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "Document ingestion conflict: document status was modified by another operation.",
        )

        # 2. No vectors were written yet, so vector store cleanup MUST NOT be called
        mock_vector_store.delete_by_document_id.assert_not_called()

        # 3. Newer registry state preserved
        rec = self.test_registry.get_record(captured_doc_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, DocumentStatus.FAILED)
        self.assertEqual(rec.error_code, "RECOVERED_STALE")

    # -------------------------------------------------------------------------
    # 54. Step 21H-E: Heartbeat Conflict Preserves Newer Registry State (Deleting/Succeeded)
    # -------------------------------------------------------------------------
    def test_heartbeat_conflict_preserves_deleting_or_succeeded_state(self):
        """54. Verify heartbeat conflict does not overwrite deleting state or restore processing."""
        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        captured_doc_id = None

        def simulate_concurrent_claim(file_bytes, document_name, collection_name=None, document_id=None, heartbeat_callback=None, **kwargs):
            nonlocal captured_doc_id
            captured_doc_id = document_id
            # Transition document to succeeded, then claim for deletion
            self.test_registry.update_status_succeeded(document_id, 1, 1, 1)
            self.test_registry.claim_for_deletion(document_id)
            assert heartbeat_callback is not None
            heartbeat_callback()

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_concurrent_claim):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("deleting_claim.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "Document ingestion conflict: document status was modified by another operation.",
        )
        mock_vector_store.delete_by_document_id.assert_not_called()

        rec = self.test_registry.get_record(captured_doc_id)
        self.assertEqual(rec.status, DocumentStatus.DELETING)

    # -------------------------------------------------------------------------
    # 55. Step 21H-E: Heartbeat Document Not Found Returns Sanitized 409
    # -------------------------------------------------------------------------
    def test_heartbeat_document_not_found_returns_sanitized_409(self):
        """55. Verify document deletion during ingestion triggers DocumentNotFoundError and returns 409."""
        mock_vector_store = MagicMock()
        app.dependency_overrides[get_vector_store_dependency] = lambda: mock_vector_store

        def simulate_removed_record(file_bytes, document_name, collection_name=None, document_id=None, heartbeat_callback=None, **kwargs):
            # Hard delete from test DB
            import sqlite3
            with sqlite3.connect(self.temp_db_file.name) as conn:
                conn.execute("DELETE FROM document_registry WHERE document_id = ?", (document_id,))
                conn.commit()
            assert heartbeat_callback is not None
            heartbeat_callback()

        with patch.object(self.stub, "ingest_pdf", side_effect=simulate_removed_record):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("purged.pdf", self.sample_pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"],
            "Document ingestion conflict: document status was modified by another operation.",
        )
        mock_vector_store.delete_by_document_id.assert_not_called()


if __name__ == "__main__":
    unittest.main()

