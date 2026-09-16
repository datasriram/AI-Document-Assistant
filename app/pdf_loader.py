from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFPageText:
    page_number: int
    text: str


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot be read or does not contain extractable text."""


def extract_text_from_pdf(pdf_path: Path) -> list[PDFPageText]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf raises several parser-specific errors.
        raise PDFExtractionError("The uploaded file is not a readable PDF.") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PDFExtractionError("Encrypted PDFs are not supported.") from exc

    pages: list[PDFPageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PDFPageText(page_number=index, text=text))

    if not pages:
        raise PDFExtractionError("No selectable text was found in the PDF.")

    return pages
