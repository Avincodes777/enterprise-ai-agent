import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure backend root is on sys.path for test runners
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.config import get_settings
from app.services import (
    FastEmbedEmbeddingService,
    MockVectorStoreService,
    PDFDocument,
    PDFLoader,
    TextChunker,
    VectorPoint,
    VectorSearchResult,
    chunk_document,
    clean_document,
    load_pdf_from_bytes,
)


def generate_sample_test_pdf() -> bytes:
    """
    Generate a 100% standard-compliant 3-page test PDF in pure bytes
    with exact cross-reference table offsets (no external builder dependencies).
    """
    body = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3>>\nendobj\n",
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources <</Font <</F1 9 0 R>>>>>>\nendobj\n",
        b"4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R /Resources <</Font <</F1 9 0 R>>>>>>\nendobj\n",
        b"5 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 8 0 R /Resources <</Font <</F1 9 0 R>>>>>>\nendobj\n",
        b"6 0 obj\n<</Length 95>>\nstream\nBT /F1 12 Tf 72 700 Td (Enterprise AI Agent - Page 1 Policy Overview. Standard benefits.) Tj ET\nendstream\nendobj\n",
        b"7 0 obj\n<</Length 102>>\nstream\nBT /F1 12 Tf 72 700 Td (Enterprise AI Agent - Page 2 Security Standards. MFA is mandatory.) Tj ET\nendstream\nendobj\n",
        b"8 0 obj\n<</Length 0>>\nstream\nendstream\nendobj\n",
        b"9 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n",
    ]

    offsets = []
    current_offset = len(body[0])
    for part in body[1:]:
        offsets.append(current_offset)
        current_offset += len(part)

    xref_offset = current_offset
    xref = [b"xref\n0 10\n0000000000 65535 f \n"]
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = f"trailer\n<</Size 10 /Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF".encode(
        "ascii"
    )

    return body[0] + b"".join(body[1:]) + b"".join(xref) + trailer


class TestIngestionPipelineIntegration(unittest.TestCase):
    """Integration test suite for the document ingestion pipeline:
    PDF -> PDFLoader -> PDFPage -> TextCleaner -> TextChunker -> DocumentChunks
    """

    def setUp(self):
        # Create a temporary real PDF file on disk
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "enterprise_policy.pdf"
        self.pdf_path.write_bytes(generate_sample_test_pdf())
        self.initial_hash = hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_execution(self):
        """
        Verify end-to-end flow:
        1. Load PDF from disk
        2. Extract pages (including empty/blank page resilience)
        3. Clean extracted page text
        4. Chunk cleaned pages
        5. Verify text, page numbers, deterministic chunk IDs
        6. Verify source PDF was not altered
        """
        # Step 1 & 2: Load PDF & Extract pages
        loader = PDFLoader()
        raw_doc = loader.load(self.pdf_path)

        self.assertIsInstance(raw_doc, PDFDocument)
        self.assertEqual(raw_doc.total_pages, 3)
        self.assertEqual(len(raw_doc.pages), 3)

        # Step 3: Clean text
        cleaned_doc = clean_document(raw_doc)
        self.assertEqual(cleaned_doc.total_pages, 3)
        self.assertEqual(len(cleaned_doc.pages), 3)

        page_1 = cleaned_doc.pages[0]
        page_2 = cleaned_doc.pages[1]
        page_3_empty = cleaned_doc.pages[2]

        self.assertEqual(page_1.page_number, 1)
        self.assertIn("Page 1 Policy Overview", page_1.content)

        self.assertEqual(page_2.page_number, 2)
        self.assertIn("Page 2 Security Standards", page_2.content)

        self.assertEqual(page_3_empty.page_number, 3)
        self.assertEqual(page_3_empty.content, "")

        # Step 4: Chunk document
        chunker = TextChunker(chunk_size=500, overlap=100)
        chunks = chunker.chunk_document(cleaned_doc)

        # Step 5: Verify chunks
        # Page 1 -> 1 chunk, Page 2 -> 1 chunk, Page 3 (empty) -> 0 chunks => 2 total chunks
        self.assertEqual(len(chunks), 2)

        # Verify Chunk 1
        chunk_1 = chunks[0]
        self.assertEqual(chunk_1.chunk_id, "enterprise_policy_p1_c0")
        self.assertEqual(chunk_1.page_number, 1)
        self.assertIn("Page 1 Policy Overview", chunk_1.text)
        self.assertEqual(chunk_1.metadata["document_name"], "enterprise_policy.pdf")
        self.assertEqual(chunk_1.metadata["page_number"], 1)
        self.assertEqual(chunk_1.metadata["chunk_index"], 0)

        # Verify Chunk 2
        chunk_2 = chunks[1]
        self.assertEqual(chunk_2.chunk_id, "enterprise_policy_p2_c0")
        self.assertEqual(chunk_2.page_number, 2)
        self.assertIn("Page 2 Security Standards", chunk_2.text)
        self.assertEqual(chunk_2.metadata["document_name"], "enterprise_policy.pdf")
        self.assertEqual(chunk_2.metadata["page_number"], 2)
        self.assertEqual(chunk_2.metadata["chunk_index"], 0)

        # Step 6: Verify deterministic chunk IDs on re-run
        re_chunks = chunker.chunk_document(cleaned_doc)
        self.assertEqual([c.chunk_id for c in chunks], [c.chunk_id for c in re_chunks])

        # Step 7: Verify source PDF file immutability (not modified by pipeline)
        final_hash = hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()
        self.assertEqual(self.initial_hash, final_hash)


class TestIngestionToVectorStoreIntegration(unittest.TestCase):
    """End-to-end integration tests proving:
    PDF bytes -> PDFLoader -> TextCleaner -> TextChunker -> FastEmbed -> VectorPoint -> MockVectorStoreService -> Search
    """

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.embedding_service = FastEmbedEmbeddingService(settings=cls.settings)

    def setUp(self):
        self.pdf_bytes = generate_sample_test_pdf()
        self.vector_store = MockVectorStoreService(
            collection_name=self.settings.qdrant_collection_name,
            dimension=self.settings.embedding_dimension,
            settings=self.settings,
        )

    def test_complete_ingestion_to_vector_store_pipeline(self):
        """
        Verify end-to-end ingestion pipeline with real FastEmbed embeddings:
        1. Load PDF from in-memory bytes
        2. Clean text content
        3. Chunk into discrete DocumentChunks
        4. Embed chunks with FastEmbed
        5. Convert to VectorPoint instances with provenance metadata
        6. Create collection and upsert into MockVectorStoreService
        7. Verify point count == chunk count
        8. Embed query and search collection
        9. Verify relevant results, metadata preservation, and ranking
        """
        # 1. Load PDF
        raw_doc = load_pdf_from_bytes(self.pdf_bytes, filename="enterprise_policy.pdf")
        self.assertEqual(raw_doc.total_pages, 3)

        # 2. Clean Text
        cleaned_doc = clean_document(raw_doc)
        self.assertEqual(len(cleaned_doc.pages), 3)

        # 3. Chunk
        chunker = TextChunker(chunk_size=500, overlap=100)
        chunks = chunker.chunk_document(cleaned_doc)
        self.assertEqual(len(chunks), 2)

        # 4. Generate Embeddings using real FastEmbed
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_documents(chunk_texts)
        self.assertEqual(len(embeddings), len(chunks))

        # 5. Verify embedding dimensions match configuration
        for vec in embeddings:
            self.assertEqual(len(vec), self.settings.embedding_dimension)
            self.assertEqual(len(vec), 384)

        # 6. Map to VectorPoints
        points = [
            VectorPoint(
                chunk_id=chunk.chunk_id,
                vector=vec,
                document_name=chunk.metadata.get("document_name", "enterprise_policy.pdf"),
                page_number=chunk.page_number,
                chunk_index=chunk.metadata.get("chunk_index", idx),
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, embeddings))
        ]
        self.assertEqual(len(points), 2)

        # 7. Create Collection & Upsert
        col_name = self.settings.qdrant_collection_name
        self.vector_store.create_collection(
            collection_name=col_name,
            dimension=self.settings.embedding_dimension,
        )
        upsert_count = self.vector_store.upsert_vectors(points, collection_name=col_name)
        self.assertEqual(upsert_count, 2)

        # 8. Verify Stored Count
        stored_count = self.vector_store.count(collection_name=col_name)
        self.assertEqual(stored_count, 2)

        # 9. Embed Search Query
        query_text = "What are the security requirements and is MFA mandatory?"
        query_vec = self.embedding_service.embed_query(query_text)
        self.assertEqual(len(query_vec), self.settings.embedding_dimension)

        # 10. Perform Cosine Similarity Search
        results = self.vector_store.search(
            query_vector=query_vec,
            top_k=2,
            collection_name=col_name,
        )

        # 11. Verify Search Results
        self.assertEqual(len(results), 2)
        top_result = results[0]
        self.assertIsInstance(top_result, VectorSearchResult)

        # Top result must be Page 2 (Security Standards / MFA)
        self.assertEqual(top_result.chunk_id, "enterprise_policy_p2_c0")
        self.assertEqual(top_result.document_name, "enterprise_policy.pdf")
        self.assertEqual(top_result.page_number, 2)
        self.assertEqual(top_result.chunk_index, 0)
        self.assertIn("MFA is mandatory", top_result.text)
        self.assertTrue(top_result.score > 0.0)

        # Verify score ordering (descending)
        self.assertGreaterEqual(results[0].score, results[1].score)

        # Verify metadata dictionary preservation
        self.assertEqual(top_result.metadata["document_name"], "enterprise_policy.pdf")
        self.assertEqual(top_result.metadata["page_number"], 2)
        self.assertEqual(top_result.metadata["chunk_index"], 0)

    def test_query_semantic_disambiguation_between_pages(self):
        """Verify semantic query precision correctly discriminates between distinct pages."""
        raw_doc = load_pdf_from_bytes(self.pdf_bytes, filename="enterprise_policy.pdf")
        cleaned_doc = clean_document(raw_doc)
        chunks = chunk_document(cleaned_doc, chunk_size=500, overlap=100)

        embeddings = self.embedding_service.embed_documents([c.text for c in chunks])
        points = [
            VectorPoint(
                chunk_id=c.chunk_id,
                vector=v,
                document_name=c.metadata.get("document_name", "enterprise_policy.pdf"),
                page_number=c.page_number,
                chunk_index=c.metadata.get("chunk_index", idx),
                text=c.text,
                metadata=c.metadata,
            )
            for idx, (c, v) in enumerate(zip(chunks, embeddings))
        ]

        col_name = "disambiguation_col"
        self.vector_store.create_collection(collection_name=col_name)
        self.vector_store.upsert_vectors(points, collection_name=col_name)

        # Query A: targeting Page 1 (Policy Overview & Standard benefits)
        query_a_vec = self.embedding_service.embed_query("employee policy overview and benefits")
        res_a = self.vector_store.search(query_a_vec, top_k=1, collection_name=col_name)
        self.assertEqual(res_a[0].chunk_id, "enterprise_policy_p1_c0")
        self.assertEqual(res_a[0].page_number, 1)

        # Query B: targeting Page 2 (Security Standards & MFA)
        query_b_vec = self.embedding_service.embed_query("information security guidelines and multi factor authentication")
        res_b = self.vector_store.search(query_b_vec, top_k=1, collection_name=col_name)
        self.assertEqual(res_b[0].chunk_id, "enterprise_policy_p2_c0")
        self.assertEqual(res_b[0].page_number, 2)


if __name__ == "__main__":
    unittest.main()

