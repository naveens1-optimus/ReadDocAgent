"""In-memory stand-ins for the Azure services, so tests never call Azure."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from application.interface.blob_storage_service import IBlobStorageService
from application.interface.document_analysis_service import IDocumentAnalysisService
from application.interface.language_model_service import ILanguageModelService, SchemaT

__all__ = ["FakeBlobStorage", "FakeLanguageModel", "FakeAnalysisService"]


class FakeBlobStorage(IBlobStorageService):
    """Records uploads in memory."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.fail_with = fail_with

    def upload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.uploads.append(
            {
                "container": container,
                "blob_name": blob_name,
                "size": len(data),
                "content_type": content_type,
            }
        )
        return f"https://fake.blob.core.windows.net/{container}/{blob_name}"


class FakeLanguageModel(ILanguageModelService):
    """Returns a canned answer, and records how it was called."""

    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        fail_on_image: Exception | None = None,
    ) -> None:
        self.answer = answer or {
            "document_type": "invoice",
            "confidence": 0.94,
            "reasoning": "Has an invoice number and totals.",
        }
        self.fail_on_image = fail_on_image
        self.calls: list[dict[str, Any]] = []

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        image: bytes | None = None,
        image_media_type: str = "image/png",
    ) -> SchemaT:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "had_image": image is not None,
                "image_media_type": image_media_type if image else None,
            }
        )
        if image is not None and self.fail_on_image is not None:
            raise self.fail_on_image
        return schema.model_validate(self.answer)

    @property
    def used_vision(self) -> bool:
        """Whether any call included an image."""
        return any(call["had_image"] for call in self.calls)


class FakeAnalysisService(IDocumentAnalysisService):
    """Returns canned document text."""

    def __init__(self, text: str = "INVOICE #123 Total due: $500.00") -> None:
        self.text = text
        self.call_count = 0

    def extract_text(self, data: bytes) -> str:
        self.call_count += 1
        return self.text


class Verdict(BaseModel):
    """Matches the classifier's internal schema, for direct schema tests."""

    document_type: str
    confidence: float
    reasoning: str
