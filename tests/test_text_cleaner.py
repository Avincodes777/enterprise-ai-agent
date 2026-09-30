import unittest
import sys
from pathlib import Path

# Add backend to sys.path so app package can be imported
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.services.text_cleaner import (
    clean_text,
    TextCleaner,
    clean_page,
    clean_document,
)
from app.services.pdf_loader import PDFPage, PDFDocument


class TestTextCleaner(unittest.TestCase):
    """Unit tests for the deterministic text cleaning service."""

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_empty_and_none_input_handling(self):
        """Verify that None, empty string, and whitespace-only strings safely return empty string."""
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text("\n\n\t  \r\n"), "")
        self.assertEqual(self.cleaner.clean(None), "")

    def test_whitespace_normalization(self):
        """Verify that excessive horizontal spaces and tabs within lines are normalized to single spaces."""
        raw_text = "Enterprise    AI\tagent   architecture\t\ttesting."
        expected = "Enterprise AI agent architecture testing."
        self.assertEqual(clean_text(raw_text), expected)

    def test_newline_and_line_ending_normalization(self):
        """Verify CRLF (\r\n) and CR (\r) line endings are normalized to LF (\n)."""
        raw_text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        expected = "Line 1\nLine 2\nLine 3\nLine 4"
        self.assertEqual(clean_text(raw_text), expected)

    def test_excessive_blank_line_reduction(self):
        """Verify that 3 or more consecutive newlines are reduced to 2 newlines (paragraph boundary)."""
        raw_text = "Paragraph 1.\n\n\n\n\nParagraph 2.\n\n\nParagraph 3."
        expected = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        self.assertEqual(clean_text(raw_text), expected)

    def test_line_boundary_whitespace_removal(self):
        """Verify leading and trailing spaces per line are stripped while preserving line breaks."""
        raw_text = "   Leading spaces\nTrailing spaces   \n   Both sides   "
        expected = "Leading spaces\nTrailing spaces\nBoth sides"
        self.assertEqual(clean_text(raw_text), expected)

    def test_preservation_of_meaningful_text_and_punctuation(self):
        """Verify that semantic content, punctuation, numbers, and bullet points remain intact."""
        raw_text = (
            "   Section 1.2: Enterprise Policy (2026)\n\n"
            "   - Requirement A: Employees must comply with ISO/IEC 27001 standard.\n"
            "   - Requirement B: Contact support@enterprise.com for questions!   "
        )
        expected = (
            "Section 1.2: Enterprise Policy (2026)\n\n"
            "- Requirement A: Employees must comply with ISO/IEC 27001 standard.\n"
            "- Requirement B: Contact support@enterprise.com for questions!"
        )
        self.assertEqual(clean_text(raw_text), expected)

    def test_clean_page_integration(self):
        """Verify cleaning a PDFPage updates content and char_count while preserving metadata and page_number."""
        page = PDFPage(
            page_number=3,
            content="   Page 3 text with   excessive spaces.   \n\n\n\nNext paragraph.   ",
            char_count=65,
            metadata={"source": "policy.pdf", "page": 3},
        )
        cleaned = clean_page(page)

        self.assertEqual(cleaned.page_number, 3)
        self.assertEqual(
            cleaned.content,
            "Page 3 text with excessive spaces.\n\nNext paragraph.",
        )
        self.assertEqual(cleaned.char_count, len(cleaned.content))
        self.assertEqual(cleaned.metadata, {"source": "policy.pdf", "page": 3})

    def test_clean_document_integration(self):
        """Verify cleaning a PDFDocument processes all pages and retains document-level metadata."""
        doc = PDFDocument(
            filename="policy.pdf",
            total_pages=2,
            pages=[
                PDFPage(page_number=1, content="  Page 1   content  ", char_count=20),
                PDFPage(page_number=2, content="  Page 2   content  ", char_count=20),
            ],
            doc_metadata={"title": "Policy Document"},
        )
        cleaned_doc = clean_document(doc)

        self.assertEqual(cleaned_doc.filename, "policy.pdf")
        self.assertEqual(cleaned_doc.total_pages, 2)
        self.assertEqual(len(cleaned_doc.pages), 2)
        self.assertEqual(cleaned_doc.pages[0].content, "Page 1 content")
        self.assertEqual(cleaned_doc.pages[1].content, "Page 2 content")
        self.assertEqual(cleaned_doc.doc_metadata, {"title": "Policy Document"})


if __name__ == "__main__":
    unittest.main()
