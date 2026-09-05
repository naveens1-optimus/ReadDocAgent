"""Renders a PDF page to an image so the vision model can read it."""

from __future__ import annotations

import pymupdf

from infrastructure.utilities.logging_config import get_logger

__all__ = ["MEDIA_TYPES", "media_type_for", "render_first_page", "page_count"]

logger = get_logger(__name__)

#: Image media types by file extension.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heif": "image/heif",
}


def media_type_for(extension: str) -> str:
    """Return the image media type for a file extension."""
    return MEDIA_TYPES.get(extension.lower(), "image/png")


def page_count(pdf_bytes: bytes) -> int:
    """Number of pages in a PDF."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        return document.page_count


def render_first_page(pdf_bytes: bytes, dpi: int = 150) -> bytes:
    """Render page 1 of a PDF to PNG bytes.

    Only the first page is rendered: it carries the letterhead, title and
    layout that identify the document type, and sending every page would cost
    far more for no classification benefit.

    150 DPI is enough for the model to read headings without producing an
    image large enough to slow the request down.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count == 0:
            raise ValueError("PDF has no pages")
        pixmap = document[0].get_pixmap(dpi=dpi)
        png = pixmap.tobytes("png")

    logger.debug("Rendered PDF page 1 to %s bytes of PNG at %s DPI", len(png), dpi)
    return png
