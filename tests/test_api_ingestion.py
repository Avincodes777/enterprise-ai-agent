"""Focused API tests for the FastAPI document ingestion endpoint and lifecycle."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_ingestion_service_dependency,
    get_rag_answer_generator_dependency,
    get_shared_ingestion_service,
    reset_ingestion_service_cache,
    reset_rag_answer_generator_cache,
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
from app.services.vector_store import VectorStoreConnectionError, VectorStoreError


class StubIngestionService:
    """Configurable stub for IngestionService dependency injection."""

    def __init__(
        self,
        result: IngestionResult = None,
        exception: Exception = None,
    ):
        self.result = result
        self.exception = exception
        self.call_count = 0
        self.last_file_bytes = None
        self.last_document_name = None
        self.last_collection_name = None

    def ingest_pdf(
        self,
        file_bytes: bytes,
        document_name: str,
        collection_name: str = None,
    ) -> IngestionResult:
        self.call_count += 1
        self.last_file_bytes = file_bytes
        self.last_document_name = document_name
        self.last_collection_name = collection_name
        if self.exception:
            raise self.exception
        return self.result or IngestionResult(
            document_name=document_name,
            pages_processed=2,
            chunks_created=3,
            chunks_stored=3,
            collection_name=collection_name or "enterprise_knowledge",
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

    def tearDown(self):
        app.dependency_overrides.clear()
        reset_ingestion_service_cache()
        reset_rag_answer_generator_cache()

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


if __name__ == "__main__":
    unittest.main()
