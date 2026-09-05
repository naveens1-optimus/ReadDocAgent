"""Tests for the PDF rendering tool."""

from __future__ import annotations

import pymupdf
import pytest

from infrastructure.tools.document_render_tool import (
    media_type_for,
    page_count,
    render_first_page,
)


def make_pdf(pages: int = 1, text: str = "INVOICE") -> bytes:
    """Build a small real PDF in memory."""
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"{text} page {index + 1}")
    data: bytes = document.tobytes()
    document.close()
    return data


class TestRenderFirstPage:
    def test_produces_png_bytes(self) -> None:
        png = render_first_page(make_pdf())
        assert png.startswith(b"\x89PNG\r\n\x1a\n")

    def test_renders_only_the_first_page(self) -> None:
        """Later pages add cost without helping classification."""
        one = render_first_page(make_pdf(pages=1))
        three = render_first_page(make_pdf(pages=3))
        # Same first page content, so the same rendered size.
        assert abs(len(one) - len(three)) < len(one) * 0.5

    def test_higher_dpi_produces_a_larger_image(self) -> None:
        assert len(render_first_page(make_pdf(), dpi=200)) > len(
            render_first_page(make_pdf(), dpi=72)
        )

    def test_rejects_non_pdf_bytes(self) -> None:
        with pytest.raises(Exception):
            render_first_page(b"this is not a pdf")


class TestPageCount:
    @pytest.mark.parametrize("pages", [1, 2, 5])
    def test_counts_pages(self, pages: int) -> None:
        assert page_count(make_pdf(pages=pages)) == pages


class TestMediaTypeFor:
    @pytest.mark.parametrize(
        ("extension", "expected"),
        [
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".JPG", "image/jpeg"),
            (".tiff", "image/tiff"),
        ],
    )
    def test_known_extensions(self, extension: str, expected: str) -> None:
        assert media_type_for(extension) == expected

    def test_unknown_extension_defaults_to_png(self) -> None:
        assert media_type_for(".xyz") == "image/png"
