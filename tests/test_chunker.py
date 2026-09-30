import unittest
import sys
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.services.chunker import (
    TextChunker,
    DocumentChunk,
    chunk_text,
    chunk_page,
    chunk_document,
)
from app.services.text_cleaner import clean_page, clean_text
from app.services.pdf_loader import PDFPage, PDFDocument


class TestTextChunker(unittest.TestCase):
    """Unit tests for the deterministic page-aware document chunker."""

    def setUp(self):
        self.chunker = TextChunker(chunk_size=1000, overlap=200)

    def test_empty_input_handling(self):
        """Verify that empty, None, and whitespace-only inputs return an empty chunk list."""
        self.assertEqual(self.chunker.chunk_text(None), [])
        self.assertEqual(self.chunker.chunk_text(""), [])
        self.assertEqual(self.chunker.chunk_text("    \n\n\t  "), [])

    def test_text_shorter_than_chunk_size(self):
        """Verify text shorter than chunk_size produces exactly 1 chunk."""
        short_text = "Enterprise AI Agent is designed for automated knowledge retrieval."
        chunks = self.chunker.chunk_text(short_text, page_number=1, doc_id="handbook.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "handbook_p1_c0")
        self.assertEqual(chunks[0].text, short_text)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].char_count, len(short_text))
        self.assertEqual(chunks[0].metadata["document_name"], "handbook.pdf")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_text_larger_than_chunk_size(self):
        """Verify long text is split into multiple contiguous chunks."""
        # Create a text of 2500 characters
        chunker = TextChunker(chunk_size=100, overlap=20)
        long_text = "A" * 250
        chunks = chunker.chunk_text(long_text, page_number=2, doc_id="policy.pdf")

        # Step = 100 - 20 = 80.
        # Chunks: [0:100], [80:180], [160:250] -> 3 chunks
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "policy_p2_c0")
        self.assertEqual(chunks[1].chunk_id, "policy_p2_c1")
        self.assertEqual(chunks[2].chunk_id, "policy_p2_c2")

        for chunk in chunks:
            self.assertEqual(chunk.page_number, 2)
            self.assertTrue(chunk.char_count <= 100)

    def test_overlap_behavior(self):
        """Verify that consecutive chunks share the expected overlapping characters."""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "0123456789ABCDEFGHIJ0123456789"  # length 31
        chunks = chunker.chunk_text(text, page_number=1, doc_id="test.pdf")

        # Step = 20 - 5 = 15
        # Chunk 0: text[0:20] -> "0123456789ABCDEFGHIJ"
        # Chunk 1: text[15:31] -> "FGHIJ0123456789"
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "0123456789ABCDEFGHIJ")
        self.assertEqual(chunks[1].text, "FGHIJ0123456789")
        # Overlap substring is "FGHIJ" (length 5)
        self.assertTrue(chunks[0].text.endswith("FGHIJ"))
        self.assertTrue(chunks[1].text.startswith("FGHIJ"))

    def test_invalid_parameters_validation(self):
        """Verify validation errors on invalid chunk_size and overlap."""
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0, overlap=0)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=-10, overlap=0)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, overlap=-5)

        with self.assertRaises(ValueError):
            # overlap equal to chunk_size
            TextChunker(chunk_size=100, overlap=100)

        with self.assertRaises(ValueError):
            # overlap greater than chunk_size
            TextChunker(chunk_size=100, overlap=150)

    def test_page_number_preservation(self):
        """Verify chunks retain their respective page numbers across multiple pages."""
        doc = PDFDocument(
            filename="annual_report.pdf",
            total_pages=3,
            pages=[
                PDFPage(page_number=1, content="Page 1 summary.", char_count=15),
                PDFPage(page_number=2, content="Page 2 financials.", char_count=18),
                PDFPage(page_number=3, content="Page 3 outlook.", char_count=15),
            ],
            doc_metadata={"title": "Annual Report 2026"},
        )
        chunks = chunk_document(doc, chunk_size=500, overlap=100)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].chunk_id, "annual_report_p1_c0")
        self.assertEqual(chunks[1].page_number, 2)
        self.assertEqual(chunks[1].chunk_id, "annual_report_p2_c0")
        self.assertEqual(chunks[2].page_number, 3)
        self.assertEqual(chunks[2].chunk_id, "annual_report_p3_c0")

    def test_deterministic_chunk_ids(self):
        """Verify that chunking produces consistent deterministic IDs across multiple runs."""
        sample = "Sample text for chunk ID consistency."
        chunks_run_1 = self.chunker.chunk_text(sample, page_number=5, doc_id="company_policy.pdf")
        chunks_run_2 = self.chunker.chunk_text(sample, page_number=5, doc_id="company_policy.pdf")

        self.assertEqual(len(chunks_run_1), len(chunks_run_2))
        for c1, c2 in zip(chunks_run_1, chunks_run_2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.chunk_id, "company_policy_p5_c0")

    def test_pipeline_integration_page_cleaner_chunker(self):
        """
        Integration test:
        PDFPage / extracted text -> cleaner -> chunker -> chunks
        """
        raw_content = (
            "   Section 4.1: Security Policies.   \r\n\r\n\r\n\r\n"
            "Employees must use multi-factor authentication (MFA)   on all corporate systems.   \n\n"
            "Passwords must be at least 16 characters in length.   "
        )
        raw_page = PDFPage(
            page_number=4,
            content=raw_content,
            char_count=len(raw_content),
            metadata={"source": "security_handbook.pdf", "page": 4},
        )

        # Step 1: Clean page
        cleaned_page = clean_page(raw_page)
        self.assertNotIn("\r\n", cleaned_page.content)
        self.assertNotIn("\n\n\n", cleaned_page.content)

        # Step 2: Chunk page
        chunker = TextChunker(chunk_size=80, overlap=20)
        chunks = chunker.chunk_page(cleaned_page)

        # Assertions on chunks
        self.assertTrue(len(chunks) >= 2)
        self.assertEqual(chunks[0].page_number, 4)
        self.assertEqual(chunks[0].chunk_id, "security_handbook_p4_c0")
        self.assertEqual(chunks[1].chunk_id, "security_handbook_p4_c1")
        self.assertEqual(chunks[0].metadata["document_name"], "security_handbook.pdf")
        self.assertEqual(chunks[0].metadata["page_number"], 4)
        self.assertTrue("Security Policies." in chunks[0].text)


if __name__ == "__main__":
    unittest.main()
