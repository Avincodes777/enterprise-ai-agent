import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .pdf_loader import PDFDocument, PDFPage


def clean_text(text: Optional[str]) -> str:
    """
    Clean extracted text deterministically while preserving semantic content,
    punctuation, and meaningful paragraph structures.

    Transformations applied:
    1. Safely handle None or empty input.
    2. Normalize line endings (CRLF and CR -> LF).
    3. Normalize excessive horizontal whitespace (multiple spaces/tabs -> single space).
    4. Remove unnecessary whitespace at line boundaries (strip each line).
    5. Reduce excessive blank lines (3+ newlines -> 2 newlines) to preserve paragraph structure.
    6. Strip outer leading/trailing whitespace.
    """
    if text is None:
        return ""

    if not text:
        return ""

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize excessive horizontal spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Strip whitespace at line boundaries
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 4. Reduce excessive consecutive blank lines (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


class TextCleaner:
    """Service for deterministic text cleaning in document processing."""

    def clean(self, text: Optional[str]) -> str:
        """Clean a raw text string."""
        return clean_text(text)

    def clean_page(self, page: "PDFPage") -> "PDFPage":
        """Clean a PDFPage's text content while preserving page structure and metadata."""
        from .pdf_loader import PDFPage

        cleaned_content = self.clean(page.content)
        return PDFPage(
            page_number=page.page_number,
            content=cleaned_content,
            char_count=len(cleaned_content),
            metadata=dict(page.metadata),
        )

    def clean_document(self, doc: "PDFDocument") -> "PDFDocument":
        """Clean all pages of a PDFDocument, returning a new PDFDocument instance."""
        from .pdf_loader import PDFDocument

        cleaned_pages = [self.clean_page(p) for p in doc.pages]
        return PDFDocument(
            filename=doc.filename,
            total_pages=doc.total_pages,
            pages=cleaned_pages,
            doc_metadata=dict(doc.doc_metadata),
        )


def clean_page(page: "PDFPage") -> "PDFPage":
    """Convenience helper to clean a PDFPage."""
    return TextCleaner().clean_page(page)


def clean_document(doc: "PDFDocument") -> "PDFDocument":
    """Convenience helper to clean a PDFDocument."""
    return TextCleaner().clean_document(doc)
