"""Unit tests for the provider-independent document IngestionService."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import Settings
from app.services import (
    BaseEmbeddingService,
    BaseVectorStoreService,
    DocumentChunk,
    IngestionEmptyDocumentError,
    IngestionError,
    IngestionResult,
    IngestionService,
    IngestionValidationError,
    MockEmbeddingService,
    MockVectorStoreService,
    PDFDocument,
    PDFLoader,
    PDFPage,
    PDFProcessingError,
    TextCleaner,
    TextChunker,
    VectorPoint,
    VectorStoreError,
    get_ingestion_service,
)


class TestIngestionService(unittest.TestCase):
    """Test suite for IngestionService validation, flow, error handling, and lifecycle."""

    def setUp(self):
        # Sample document fixture
        self.sample_bytes = b"%PDF-1.4 Mock PDF content for enterprise testing"
        self.doc_name = "security_policy.pdf"

        # Mock PDFLoader
        self.mock_loader = MagicMock(spec=PDFLoader)
        self.mock_pages = [
            PDFPage(
                page_number=1,
                content="Page 1: Multi-factor authentication is required.",
                char_count=48,
                metadata={"source": self.doc_name, "page": 1, "department": "IT"},
            ),
            PDFPage(
                page_number=2,
                content="Page 2: Passwords must be at least 16 characters.",
                char_count=49,
                metadata={"source": self.doc_name, "page": 2, "department": "IT"},
            ),
        ]
        self.mock_doc = PDFDocument(
            filename=self.doc_name,
            total_pages=2,
            pages=self.mock_pages,
            doc_metadata={"title": "Security Policy 2026"},
        )
        self.mock_loader.load_from_bytes.return_value = self.mock_doc

        # Real TextCleaner and TextChunker for deterministic processing
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=500, overlap=100)

        # Mock Embedding Service
        self.mock_embedder = MagicMock(spec=BaseEmbeddingService)
        self.mock_embedder.dimension = 384
        self.mock_embedder.model_name = "mock-embedder"
        # 2 chunks => 2 vectors of length 384
        self.sample_vectors = [[0.05] * 384, [0.10] * 384]
        self.mock_embedder.embed_documents.return_value = self.sample_vectors

        # Mock Vector Store Service
        self.mock_vector_store = MagicMock(spec=BaseVectorStoreService)
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.create_collection.return_value = True
        self.mock_vector_store.upsert_vectors.return_value = 2

        # Configure service under test with injected dependencies
        self.service = IngestionService(
            pdf_loader=self.mock_loader,
            text_cleaner=self.text_cleaner,
            chunker=self.chunker,
            embedding_service=self.mock_embedder,
            vector_store_service=self.mock_vector_store,
            default_collection_name="enterprise_knowledge",
        )

    # -------------------------------------------------------------------------
    # 1. Successful PDF Ingestion
    # -------------------------------------------------------------------------
    def test_successful_pdf_ingestion(self):
        """1. Verify standard PDF ingestion runs end-to-end and returns IngestionResult."""
        result = self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertIsInstance(result, IngestionResult)
        self.assertEqual(result.document_name, self.doc_name)
        self.assertEqual(result.pages_processed, 2)
        self.assertEqual(result.chunks_created, 2)
        self.assertEqual(result.chunks_stored, 2)
        self.assertEqual(result.collection_name, "enterprise_knowledge")
        self.assertEqual(len(result.chunk_ids), 2)
        self.mock_vector_store.upsert_vectors.assert_called_once()

    # -------------------------------------------------------------------------
    # 2. Invalid Empty Bytes Rejection
    # -------------------------------------------------------------------------
    def test_invalid_empty_bytes_rejected(self):
        """2. Verify empty or non-bytes input raises IngestionValidationError."""
        for invalid_bytes in [b"", None, "", "not-bytes", []]:
            with self.subTest(invalid_bytes=invalid_bytes):
                with self.assertRaises(IngestionValidationError):
                    self.service.ingest_pdf(invalid_bytes, self.doc_name)  # type: ignore

    # -------------------------------------------------------------------------
    # 3. Invalid Document Name Rejection
    # -------------------------------------------------------------------------
    def test_invalid_document_name_rejected(self):
        """3. Verify empty, whitespace, or non-string document name raises IngestionValidationError."""
        for invalid_name in ["", "   ", "\t\n", None, 12345]:
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(IngestionValidationError):
                    self.service.ingest_pdf(self.sample_bytes, invalid_name)  # type: ignore

    # -------------------------------------------------------------------------
    # 4. PDF Loader Failures Handled Correctly
    # -------------------------------------------------------------------------
    def test_pdf_loader_failures_handled_correctly(self):
        """4. Verify PDF parsing errors are caught and re-raised as IngestionError."""
        self.mock_loader.load_from_bytes.side_effect = PDFProcessingError("Corrupt xref table")

        with self.assertRaises(IngestionError) as ctx:
            self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertIn("Failed to load PDF document", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, PDFProcessingError)

    # -------------------------------------------------------------------------
    # 5. Empty Extracted Text Rejection
    # -------------------------------------------------------------------------
    def test_empty_extracted_text_rejected(self):
        """5. Verify a PDF with blank pages raises IngestionEmptyDocumentError."""
        blank_doc = PDFDocument(
            filename=self.doc_name,
            total_pages=2,
            pages=[
                PDFPage(page_number=1, content="   ", char_count=3, metadata={}),
                PDFPage(page_number=2, content="", char_count=0, metadata={}),
            ],
            doc_metadata={},
        )
        self.mock_loader.load_from_bytes.return_value = blank_doc

        with self.assertRaises(IngestionEmptyDocumentError):
            self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.mock_vector_store.upsert_vectors.assert_not_called()

    # -------------------------------------------------------------------------
    # 6. Chunker Output Passed to Embedding
    # -------------------------------------------------------------------------
    def test_chunker_output_passed_to_embedding(self):
        """6. Verify exact chunk texts generated by chunker are passed to embed_documents."""
        self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.mock_embedder.embed_documents.assert_called_once()
        passed_texts = self.mock_embedder.embed_documents.call_args[0][0]
        self.assertEqual(len(passed_texts), 2)
        self.assertIn("Multi-factor authentication", passed_texts[0])
        self.assertIn("Passwords must be at least", passed_texts[1])

    # -------------------------------------------------------------------------
    # 7. Embedding Count Mismatch Rejection
    # -------------------------------------------------------------------------
    def test_embedding_count_mismatch_rejected(self):
        """7. Verify mismatch between chunk count and embedding count raises IngestionError."""
        # 2 chunks generated, but embedder returns only 1 vector
        self.mock_embedder.embed_documents.return_value = [[0.1] * 384]

        with self.assertRaises(IngestionError) as ctx:
            self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertIn("Embedding count mismatch", str(ctx.exception))
        self.mock_vector_store.upsert_vectors.assert_not_called()

    # -------------------------------------------------------------------------
    # 8. Correct VectorPoint Construction
    # -------------------------------------------------------------------------
    def test_correct_vector_point_construction(self):
        """8. Verify VectorPoint objects are correctly constructed from chunks and vectors."""
        self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        upsert_call_args = self.mock_vector_store.upsert_vectors.call_args
        points = upsert_call_args[1].get("points") or upsert_call_args[0][0]

        self.assertEqual(len(points), 2)
        for idx, pt in enumerate(points):
            self.assertIsInstance(pt, VectorPoint)
            self.assertEqual(pt.document_name, self.doc_name)
            self.assertEqual(pt.page_number, idx + 1)
            self.assertEqual(pt.vector, self.sample_vectors[idx])
            self.assertTrue(pt.chunk_id.startswith("security_policy_p"))

    # -------------------------------------------------------------------------
    # 9. Metadata Preservation
    # -------------------------------------------------------------------------
    def test_metadata_preservation(self):
        """9. Verify custom page and document metadata are preserved in VectorPoint."""
        self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        points = self.mock_vector_store.upsert_vectors.call_args[1].get(
            "points"
        ) or self.mock_vector_store.upsert_vectors.call_args[0][0]

        self.assertEqual(points[0].metadata.get("department"), "IT")
        self.assertEqual(points[0].metadata.get("title"), "Security Policy 2026")
        self.assertEqual(points[0].metadata.get("page_number"), 1)
        self.assertEqual(points[0].metadata.get("chunk_index"), 0)

    # -------------------------------------------------------------------------
    # 10. Collection Creation When Missing
    # -------------------------------------------------------------------------
    def test_collection_creation_when_missing(self):
        """10. Verify create_collection is called when target collection does not exist."""
        self.mock_vector_store.collection_exists.return_value = False

        self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.mock_vector_store.collection_exists.assert_called_with("enterprise_knowledge")
        self.mock_vector_store.create_collection.assert_called_once_with(
            collection_name="enterprise_knowledge",
            dimension=384,
        )

    # -------------------------------------------------------------------------
    # 11. Upsert Receives Expected Points
    # -------------------------------------------------------------------------
    def test_upsert_receives_expected_points(self):
        """11. Verify upsert_vectors receives the expected points list and collection name."""
        self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.mock_vector_store.upsert_vectors.assert_called_once()
        args, kwargs = self.mock_vector_store.upsert_vectors.call_args
        target_col = kwargs.get("collection_name") or args[1]
        self.assertEqual(target_col, "enterprise_knowledge")

    # -------------------------------------------------------------------------
    # 12. Ingestion Result Fields Are Correct
    # -------------------------------------------------------------------------
    def test_ingestion_result_fields_are_correct(self):
        """12. Verify all attributes in IngestionResult contain accurate values."""
        result = self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertEqual(result.document_name, self.doc_name)
        self.assertEqual(result.pages_processed, 2)
        self.assertEqual(result.chunks_created, 2)
        self.assertEqual(result.chunks_stored, 2)
        self.assertEqual(result.collection_name, "enterprise_knowledge")
        self.assertEqual(
            result.chunk_ids,
            ["security_policy_p1_c0", "security_policy_p2_c0"],
        )

    # -------------------------------------------------------------------------
    # 13. Collection Override Propagation
    # -------------------------------------------------------------------------
    def test_collection_override_propagation(self):
        """13. Verify optional collection_name argument overrides default collection."""
        custom_col = "custom_hr_collection"
        self.mock_vector_store.collection_exists.return_value = False

        result = self.service.ingest_pdf(
            self.sample_bytes, self.doc_name, collection_name=custom_col
        )

        self.assertEqual(result.collection_name, custom_col)
        self.mock_vector_store.collection_exists.assert_called_with(custom_col)
        self.mock_vector_store.create_collection.assert_called_once_with(
            collection_name=custom_col,
            dimension=384,
        )
        _, kwargs = self.mock_vector_store.upsert_vectors.call_args
        self.assertEqual(kwargs.get("collection_name"), custom_col)

    # -------------------------------------------------------------------------
    # 14. Storage Failures Not Reported As Success
    # -------------------------------------------------------------------------
    def test_storage_failures_not_reported_as_success(self):
        """14. Verify vector store upsert errors raise IngestionError rather than returning success."""
        self.mock_vector_store.upsert_vectors.side_effect = VectorStoreError(
            "Connection refused by Qdrant"
        )

        with self.assertRaises(IngestionError) as ctx:
            self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertIn("Vector store upsert failed", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, VectorStoreError)

    # -------------------------------------------------------------------------
    # 15. Dependency Injection Works
    # -------------------------------------------------------------------------
    def test_dependency_injection_works(self):
        """15. Verify constructor injection and factory function correctly inject components."""
        custom_loader = MagicMock(spec=PDFLoader)
        custom_cleaner = MagicMock(spec=TextCleaner)
        custom_chunker = MagicMock(spec=TextChunker)
        custom_embedder = MagicMock(spec=BaseEmbeddingService)
        custom_store = MagicMock(spec=BaseVectorStoreService)

        custom_service = IngestionService(
            pdf_loader=custom_loader,
            text_cleaner=custom_cleaner,
            chunker=custom_chunker,
            embedding_service=custom_embedder,
            vector_store_service=custom_store,
            default_collection_name="custom_col",
        )

        self.assertIs(custom_service.pdf_loader, custom_loader)
        self.assertIs(custom_service.text_cleaner, custom_cleaner)
        self.assertIs(custom_service.chunker, custom_chunker)
        self.assertIs(custom_service.embedding_service, custom_embedder)
        self.assertIs(custom_service.vector_store_service, custom_store)
        self.assertEqual(custom_service.default_collection_name, "custom_col")

        # Test factory function
        factory_service = get_ingestion_service(
            pdf_loader=custom_loader,
            embedding_service=custom_embedder,
            vector_store_service=custom_store,
        )
        self.assertIs(factory_service.pdf_loader, custom_loader)
        self.assertIs(factory_service.embedding_service, custom_embedder)
        self.assertIs(factory_service.vector_store_service, custom_store)

    # -------------------------------------------------------------------------
    # 16. Package Exports Work
    # -------------------------------------------------------------------------
    def test_package_exports_work(self):
        """16. Verify public symbols are exposed from app.services."""
        import app.services as services

        exported_classes = [
            "IngestionService",
            "IngestionResult",
            "IngestionError",
            "IngestionValidationError",
            "IngestionEmptyDocumentError",
            "get_ingestion_service",
        ]
        for name in exported_classes:
            self.assertTrue(
                hasattr(services, name), f"app.services is missing exported symbol: {name}"
            )

    # -------------------------------------------------------------------------
    # 17. Repeated Deterministic Input Preserves Chunk IDs
    # -------------------------------------------------------------------------
    def test_repeated_deterministic_input_preserves_chunk_ids(self):
        """17. Verify identical document input processed twice produces identical chunk IDs."""
        result_1 = self.service.ingest_pdf(self.sample_bytes, self.doc_name)
        result_2 = self.service.ingest_pdf(self.sample_bytes, self.doc_name)

        self.assertEqual(result_1.chunk_ids, result_2.chunk_ids)
        self.assertEqual(
            result_1.chunk_ids,
            ["security_policy_p1_c0", "security_policy_p2_c0"],
        )


if __name__ == "__main__":
    unittest.main()
