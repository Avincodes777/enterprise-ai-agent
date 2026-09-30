import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Base exception raised for errors during PDF processing."""
    pass


class PDFEncryptedError(PDFProcessingError):
    """Raised when an encrypted PDF cannot be decrypted."""
    pass


class PDFPage(BaseModel):
    """Represents a single extracted page from a PDF document."""
    page_number: int = Field(..., description="1-indexed page number")
    content: str = Field(..., description="Extracted text content of the page")
    char_count: int = Field(..., description="Total characters in the page content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata specific to this page")


class PDFDocument(BaseModel):
    """Represents a fully loaded PDF document with all pages and metadata."""
    filename: Optional[str] = Field(None, description="Original filename or file path")
    total_pages: int = Field(..., description="Total count of pages extracted")
    pages: List[PDFPage] = Field(default_factory=list, description="Extracted pages")
    doc_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata (title, author, etc.)")

    @property
    def full_text(self) -> str:
        """Returns the concatenated text across all pages."""
        return "\n\n".join(page.content for page in self.pages if page.content.strip())


class PDFLoader:
    """Enterprise-ready PDF loading and parsing service."""

    def __init__(self, strip_whitespace: bool = True, password: Optional[str] = None):
        """
        Initialize the PDF loader.

        Args:
            strip_whitespace: Whether to strip leading/trailing whitespace per page.
            password: Optional password for decrypting protected PDFs.
        """
        self.strip_whitespace = strip_whitespace
        self.password = password

    def _ensure_pypdf(self) -> None:
        if PdfReader is None:
            raise PDFProcessingError(
                "The 'pypdf' package is required for PDFLoader. Install it via `pip install pypdf`."
            )

    def _parse_reader(self, reader: "PdfReader", filename: Optional[str] = None) -> PDFDocument:
        self._ensure_pypdf()

        if reader.is_encrypted:
            if self.password:
                decrypt_success = reader.decrypt(self.password)
                if not decrypt_success:
                    raise PDFEncryptedError("Failed to decrypt PDF with the provided password.")
            else:
                try:
                    # Attempt decrypting with empty password
                    reader.decrypt("")
                except Exception as exc:
                    raise PDFEncryptedError(
                        "PDF is encrypted and requires a password to open."
                    ) from exc

        total_pages = len(reader.pages)
        pages: List[PDFPage] = []

        # Extract document metadata
        doc_metadata: Dict[str, Any] = {}
        if reader.metadata:
            for key, val in reader.metadata.items():
                clean_key = key.lstrip("/").lower()
                doc_metadata[clean_key] = str(val) if val is not None else None

        for index, page in enumerate(reader.pages):
            page_num = index + 1
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num} in {filename or 'PDF stream'}: {e}")
                page_text = ""

            if self.strip_whitespace:
                page_text = page_text.strip()

            pages.append(
                PDFPage(
                    page_number=page_num,
                    content=page_text,
                    char_count=len(page_text),
                    metadata={
                        "source": filename or "in-memory-pdf",
                        "page": page_num,
                        "total_pages": total_pages,
                    },
                )
            )

        return PDFDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages,
            doc_metadata=doc_metadata,
        )

    def load(self, file_path: Union[str, Path]) -> PDFDocument:
        """
        Load and parse a PDF file from a local filesystem path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFDocument containing extracted pages and metadata.
        """
        self._ensure_pypdf()
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            with open(path_obj, "rb") as f:
                reader = PdfReader(f)
                return self._parse_reader(reader, filename=path_obj.name)
        except PDFProcessingError:
            raise
        except Exception as exc:
            raise PDFProcessingError(f"Failed to read PDF '{file_path}': {exc}") from exc

    def load_from_bytes(self, content: bytes, filename: Optional[str] = None) -> PDFDocument:
        """
        Load and parse a PDF document from raw bytes or memory buffer.

        Args:
            content: Raw byte content of the PDF.
            filename: Optional filename for reference metadata.

        Returns:
            PDFDocument containing extracted pages and metadata.
        """
        self._ensure_pypdf()

        if not content:
            raise ValueError("Provided PDF byte content is empty.")

        try:
            stream = io.BytesIO(content)
            reader = PdfReader(stream)
            return self._parse_reader(reader, filename=filename)
        except PDFProcessingError:
            raise
        except Exception as exc:
            raise PDFProcessingError(f"Failed to parse PDF bytes: {exc}") from exc

    def extract_text(self, file_input: Union[str, Path, bytes]) -> str:
        """
        Extract concatenated full text from a PDF file path or raw bytes.

        Args:
            file_input: File path (str/Path) or raw bytes.

        Returns:
            Extracted full text string.
        """
        if isinstance(file_input, (str, Path)):
            doc = self.load(file_input)
        elif isinstance(file_input, bytes):
            doc = self.load_from_bytes(file_input)
        else:
            raise TypeError("Expected file_input to be a str, Path, or bytes.")
        return doc.full_text


# Convenience helper functions
def load_pdf(
    file_path: Union[str, Path],
    strip_whitespace: bool = True,
    password: Optional[str] = None,
) -> PDFDocument:
    """Helper function to load a PDF from a file path."""
    loader = PDFLoader(strip_whitespace=strip_whitespace, password=password)
    return loader.load(file_path)


def load_pdf_from_bytes(
    content: bytes,
    filename: Optional[str] = None,
    strip_whitespace: bool = True,
    password: Optional[str] = None,
) -> PDFDocument:
    """Helper function to load a PDF from bytes in memory."""
    loader = PDFLoader(strip_whitespace=strip_whitespace, password=password)
    return loader.load_from_bytes(content, filename=filename)


def extract_pdf_text(
    file_input: Union[str, Path, bytes],
    strip_whitespace: bool = True,
    password: Optional[str] = None,
) -> str:
    """Helper function to extract all text from a PDF."""
    loader = PDFLoader(strip_whitespace=strip_whitespace, password=password)
    return loader.extract_text(file_input)
