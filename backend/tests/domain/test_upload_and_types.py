"""Tests for the document type enum and the upload request schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.enum.document_type import DocumentType
from domain.schema.human_review_request import HumanReviewRequest
from domain.schema.upload_file_request import UploadFileRequest


class TestDocumentTypeFromString:
    @pytest.mark.parametrize(
        "raw", ["invoice", "INVOICE", " Invoice ", "Invoice.", '"invoice"']
    )
    def test_accepts_dirty_model_output(self, raw: str) -> None:
        assert DocumentType.from_string(raw) is DocumentType.INVOICE

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("cv", DocumentType.RESUME),
            ("curriculum vitae", DocumentType.RESUME),
            ("bill", DocumentType.INVOICE),
            ("agreement", DocumentType.CONTRACT),
            ("nda", DocumentType.CONTRACT),
            ("passport", DocumentType.ID_CARD),
            ("drivers-license", DocumentType.ID_CARD),
        ],
    )
    def test_maps_aliases(self, raw: str, expected: DocumentType) -> None:
        assert DocumentType.from_string(raw) is expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "a photo of a cat"])
    def test_unknown_becomes_unsupported(self, raw: str | None) -> None:
        """A surprising answer must take the fallback path, not crash."""
        assert DocumentType.from_string(raw) is DocumentType.UNSUPPORTED

    def test_serialises_as_plain_string(self) -> None:
        assert DocumentType.INVOICE == "invoice"


class TestUploadFileRequest:
    @pytest.mark.parametrize(
        "name", ["a.pdf", "a.PDF", "scan.png", "photo.jpeg", "fax.tiff"]
    )
    def test_accepts_supported_types(self, name: str) -> None:
        assert UploadFileRequest(file_name=name, size_bytes=10).file_name == name

    @pytest.mark.parametrize("name", ["notes.txt", "app.exe", "archive.zip", "noext"])
    def test_rejects_unsupported_types(self, name: str) -> None:
        with pytest.raises(ValidationError):
            UploadFileRequest(file_name=name, size_bytes=10)

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("../../etc/passwd.pdf", "passwd.pdf"),
            ("a/b/c.pdf", "c.pdf"),
            ("C:\\Users\\x\\invoice.pdf", "invoice.pdf"),
        ],
    )
    def test_strips_directory_parts(self, given: str, expected: str) -> None:
        """The name is used to build a blob path, so it must be a basename."""
        assert UploadFileRequest(file_name=given, size_bytes=10).file_name == expected

    def test_rejects_empty_file(self) -> None:
        with pytest.raises(ValidationError):
            UploadFileRequest(file_name="a.pdf", size_bytes=0)

    def test_extension_and_is_pdf(self) -> None:
        pdf = UploadFileRequest(file_name="a.PDF", size_bytes=10)
        png = UploadFileRequest(file_name="a.png", size_bytes=10)
        assert pdf.extension == ".pdf"
        assert pdf.is_pdf is True
        assert png.is_pdf is False

    def test_size_limit(self) -> None:
        upload = UploadFileRequest(file_name="a.pdf", size_bytes=30 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds|over the"):
            upload.ensure_within_size_limit(20 * 1024 * 1024)

    def test_within_size_limit_passes(self) -> None:
        upload = UploadFileRequest(file_name="a.pdf", size_bytes=1024)
        upload.ensure_within_size_limit(20 * 1024 * 1024)  # must not raise


class TestHumanReviewRequest:
    def test_resume_payload_shape(self) -> None:
        review = HumanReviewRequest(
            approved=True,
            document_type=DocumentType.RECEIPT,
            reviewer="naveen",
            note="corrected",
        )
        assert review.to_resume_payload() == {
            "approved": True,
            "document_type": "receipt",
            "reviewer": "naveen",
            "note": "corrected",
        }

    def test_correction_is_optional(self) -> None:
        payload = HumanReviewRequest(approved=False).to_resume_payload()
        assert payload["approved"] is False
        assert payload["document_type"] is None

    def test_rejects_unknown_document_type(self) -> None:
        with pytest.raises(ValidationError):
            HumanReviewRequest(approved=True, document_type="spaceship")
