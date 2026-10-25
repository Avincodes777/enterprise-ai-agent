"""Live end-to-end integration tests for document ingestion and retrieval.

Proves that a PDF uploaded through POST /api/ingest can subsequently be
retrieved through POST /api/query using the real FastEmbed embedding implementation
and live Qdrant vector store, with grounded context returned to the LLM layer.

Maintains strict collection isolation and preserves unit-test independence.
"""

import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_ingestion_service_dependency,
    get_rag_answer_generator_dependency,
    reset_ingestion_service_cache,
    reset_rag_answer_generator_cache,
)
from app.config import get_settings
from app.main import app
from app.rag.answer_generator import RAGAnswerGenerator
from app.rag.context_builder import RAGContextBuilder
from app.services import (
    FastEmbedEmbeddingService,
    IngestionService,
    MockLLMService,
    PDFLoader,
    QdrantVectorStoreService,
    TextCleaner,
    TextChunker,
    VectorRetriever,
)


def create_in_memory_pdf(text_lines: list[str]) -> bytes:
    """
    Construct a structurally valid, self-contained PDF entirely in memory with extractable text.

    Calculates exact byte offsets for xref entries to avoid parser warnings or errors.
    """
    def escape_pdf_str(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream_lines = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for i, line in enumerate(text_lines):
        escaped = escape_pdf_str(line)
        if i > 0:
            stream_lines.append(f"T* ({escaped}) Tj")
        else:
            stream_lines.append(f"({escaped}) Tj")
    stream_lines.append("ET")
    stream_data = "\n".join(stream_lines).encode("latin1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        b"4 0 obj\n<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)

    xref_offset = out.tell()
    out.write(b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n")
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))

    out.write(b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root 1 0 R >>\n")
    out.write(b"startxref\n" + str(xref_offset).encode("ascii") + b"\n%%EOF\n")
    return out.getvalue()


class TestIngestionQueryE2ELive(unittest.TestCase):
    """
    End-to-end integration tests verifying PDF upload -> live ingestion -> live retrieval.

    Uses real FastEmbed embeddings and live Qdrant vector database.
    Avoids external OpenAI API calls by utilizing MockLLMService.
    """

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.qdrant_url = cls.settings.qdrant_url or "http://localhost:6333"

        # Check live Qdrant health using real QdrantVectorStoreService
        cls.vector_store_service = QdrantVectorStoreService(
            url=cls.qdrant_url,
            collection_name=cls.settings.qdrant_collection_name,
            api_key=cls.settings.qdrant_api_key,
            timeout=cls.settings.qdrant_timeout_seconds,
            dimension=cls.settings.embedding_dimension,
            settings=cls.settings,
        )

        # Real FastEmbed embedding service
        cls.embedding_service = FastEmbedEmbeddingService(settings=cls.settings)

    def setUp(self):
        if not self.vector_store_service.health_check():
            self.skipTest(f"Live Qdrant instance is not reachable at {self.qdrant_url}")

        # Unique dedicated test collection to prevent any collision
        self.test_collection_name = f"test_e2e_ingest_{uuid.uuid4().hex[:12]}"
        self._test_collection_created = False

        # Build real IngestionService wired to real FastEmbed & live Qdrant for this test collection
        self.real_ingestion_service = IngestionService(
            pdf_loader=PDFLoader(),
            text_cleaner=TextCleaner(),
            chunker=TextChunker(),
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            default_collection_name=self.test_collection_name,
            settings=self.settings,
        )

        # Build real Retriever and ContextBuilder for this test collection
        self.real_retriever = VectorRetriever(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            default_top_k=5,
            default_collection_name=self.test_collection_name,
            settings=self.settings,
        )
        self.real_context_builder = RAGContextBuilder(
            retriever=self.real_retriever,
            default_top_k=5,
            default_collection_name=self.test_collection_name,
        )

        # Mock LLM service to prevent external OpenAI API calls while verifying grounding prompt
        self.mock_llm_service = MockLLMService(settings=self.settings)

        self.real_answer_generator = RAGAnswerGenerator(
            context_builder=self.real_context_builder,
            llm_service=self.mock_llm_service,
        )

        # Apply FastAPI dependency overrides
        app.dependency_overrides[get_ingestion_service_dependency] = lambda: self.real_ingestion_service
        app.dependency_overrides[get_rag_answer_generator_dependency] = lambda: self.real_answer_generator

        self.client = TestClient(app)

    def tearDown(self):
        try:
            # Reset FastAPI dependency overrides and cached instances
            app.dependency_overrides.clear()
            reset_ingestion_service_cache()
            reset_rag_answer_generator_cache()
        finally:
            # Strictly clean up the test-owned collection
            self._cleanup_test_collection()

    def _cleanup_test_collection(self):
        """Clean up only the test-owned collection, ensuring no production collections are affected."""
        if not getattr(self, "test_collection_name", None):
            return

        # Verification guard: Ensure collection is explicitly a test-owned collection
        if not self.test_collection_name.startswith("test_e2e_ingest_"):
            raise RuntimeError(
                f"Refusing to delete unsafe collection name: '{self.test_collection_name}'"
            )
        if self.test_collection_name == "enterprise_knowledge" or self.test_collection_name == self.settings.qdrant_collection_name:
            raise RuntimeError(
                "Refusing to delete production collection."
            )

        try:
            if self.vector_store_service.collection_exists(self.test_collection_name):
                deleted = self.vector_store_service.delete_collection(self.test_collection_name)
                if not deleted:
                    raise RuntimeError(
                        f"Failed to delete test collection '{self.test_collection_name}': delete returned False."
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Error cleaning up test collection '{self.test_collection_name}': {exc}"
            ) from exc

    def test_end_to_end_ingestion_and_retrieval_flow(self):
        """
        Prove the complete ingestion and retrieval loop across live infrastructure:
        1. Generate in-memory PDF with distinctive factual content (Orchid Mercury Protocol).
        2. POST /api/ingest with multipart form data into the dedicated test collection.
        3. Verify HTTP 200 and structured IngestionResult details (pages, chunks, chunk_ids).
        4. POST /api/query targeting the dedicated test collection.
        5. Verify HTTP 200 and RAGResponse with retrieved sources.
        6. Verify retrieved source text, document provenance, page/chunk indices, and similarity score.
        7. Verify Mock LLM generated answer received the grounded context and OpenAI was not called.
        """
        # Step 1: Construct in-memory PDF with distinctive factual content
        distinctive_lines = [
            "Confidential Enterprise Technical Memorandum",
            "Project name: Orchid Mercury Protocol",
            "Backup interval: every 17 hours",
            "Recovery location: Northstar Archive",
            "Primary failover cluster: Sector-9 Datacenter",
        ]
        pdf_bytes = create_in_memory_pdf(distinctive_lines)
        doc_filename = "orchid_mercury_memorandum.pdf"

        # Track that this test collection is being populated
        self._test_collection_created = True

        # Safety patch: Ensure OpenAI client is never constructed or called
        with patch("app.services.llm.OpenAIChatService._get_client") as mock_openai_client:
            mock_openai_client.side_effect = AssertionError("OpenAI API client was unexpectedly invoked!")

            # Step 2: Ingest PDF via POST /api/ingest HTTP route
            upload_files = {
                "file": (doc_filename, pdf_bytes, "application/pdf")
            }
            form_data = {
                "collection_name": self.test_collection_name
            }

            ingest_response = self.client.post(
                "/api/ingest",
                files=upload_files,
                data=form_data,
            )

            # Step 3: Verify ingestion HTTP response and payload
            self.assertEqual(
                ingest_response.status_code,
                200,
                f"Ingestion failed with status {ingest_response.status_code}: {ingest_response.text}",
            )
            ingest_data = ingest_response.json()

            self.assertEqual(ingest_data["document_name"], doc_filename)
            self.assertGreaterEqual(ingest_data["pages_processed"], 1)
            self.assertGreaterEqual(ingest_data["chunks_created"], 1)
            self.assertGreaterEqual(ingest_data["chunks_stored"], 1)
            self.assertEqual(ingest_data["collection_name"], self.test_collection_name)
            self.assertTrue(len(ingest_data["chunk_ids"]) >= 1)

            # Verify collection point count in live Qdrant
            stored_count = self.vector_store_service.count(self.test_collection_name)
            self.assertEqual(stored_count, ingest_data["chunks_stored"])

            # Step 4: Query knowledge base via POST /api/query HTTP route
            query_text = "What is the backup interval and recovery location for Orchid Mercury Protocol?"
            query_payload = {
                "query": query_text,
                "collection_name": self.test_collection_name,
                "top_k": 3,
            }

            query_response = self.client.post("/api/query", json=query_payload)

            # Step 5: Verify query HTTP response
            self.assertEqual(
                query_response.status_code,
                200,
                f"Query failed with status {query_response.status_code}: {query_response.text}",
            )
            query_data = query_response.json()

            # Step 6: Verify retrieval returned relevant source chunks with provenance
            self.assertEqual(query_data["query"], query_text)
            self.assertIn("sources", query_data)
            sources = query_data["sources"]
            self.assertGreaterEqual(
                len(sources),
                1,
                "Query response contains no retrieved sources.",
            )

            # Inspect top retrieved source
            top_source = sources[0]
            self.assertEqual(top_source["document_name"], doc_filename)
            self.assertEqual(top_source["page_number"], 1)
            self.assertEqual(top_source["chunk_index"], 0)
            self.assertIn("orchid_mercury_memorandum", top_source["chunk_id"])

            # Verify distinctive content matches the uploaded PDF
            self.assertIn("Orchid Mercury Protocol", top_source["text"])
            self.assertIn("every 17 hours", top_source["text"])
            self.assertIn("Northstar Archive", top_source["text"])

            # Verify similarity score is positive and reflects cosine similarity
            self.assertIsInstance(top_source["score"], float)
            self.assertGreater(top_source["score"], 0.5)

            # Verify context_text was formatted with document metadata and content
            self.assertIn("Orchid Mercury Protocol", query_data["context_text"])
            self.assertIn(doc_filename, query_data["context_text"])

            # Step 7: Verify LLM execution without external OpenAI calls
            self.assertGreaterEqual(self.mock_llm_service.call_count, 1)
            self.assertIn("Orchid Mercury Protocol", self.mock_llm_service.last_prompt)
            self.assertIn(query_text, self.mock_llm_service.last_prompt)

            # Answer was produced by mock LLM using context
            self.assertIn("[Mock LLM Output", query_data["answer"])
            self.assertIn("Orchid Mercury Protocol", query_data["answer"])

            # Confirm OpenAI was never called
            mock_openai_client.assert_not_called()

    def test_query_isolation_between_collections(self):
        """
        Verify that documents ingested into the dedicated test collection
        are not visible from an empty alternate collection.
        """
        pdf_bytes = create_in_memory_pdf([
            "Project name: Orchid Mercury Protocol",
            "Backup interval: every 17 hours",
            "Recovery location: Northstar Archive",
        ])

        # Ingest into test collection
        ingest_res = self.client.post(
            "/api/ingest",
            files={"file": ("orchid.pdf", pdf_bytes, "application/pdf")},
            data={"collection_name": self.test_collection_name},
        )
        self.assertEqual(ingest_res.status_code, 200)

        # Create empty secondary collection
        empty_col = f"{self.test_collection_name}_empty"
        try:
            self.vector_store_service.create_collection(
                collection_name=empty_col,
                dimension=self.settings.embedding_dimension,
            )

            # Query the empty secondary collection
            query_res = self.client.post(
                "/api/query",
                json={
                    "query": "What is the backup interval for Orchid Mercury Protocol?",
                    "collection_name": empty_col,
                },
            )
            self.assertEqual(query_res.status_code, 200)
            query_data = query_res.json()
            # Empty collection must return 0 sources
            self.assertEqual(len(query_data["sources"]), 0)
            self.assertEqual(
                query_data["answer"],
                RAGAnswerGenerator.EMPTY_RETRIEVAL_ANSWER,
            )
        finally:
            if self.vector_store_service.collection_exists(empty_col):
                self.vector_store_service.delete_collection(empty_col)


if __name__ == "__main__":
    unittest.main()
