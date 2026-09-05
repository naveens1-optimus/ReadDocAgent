"""In-memory stand-ins for the Azure services, so tests never call Azure."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from application.interface.blob_storage_service import IBlobStorageService
from application.interface.document_analysis_service import IDocumentAnalysisService
from application.interface.entity_store import IEntityStore
from domain.entity.extraction_result import ExtractedField, ExtractionResult
from application.interface.language_model_service import ILanguageModelService, SchemaT

__all__ = [
    "FakeBlobStorage",
    "FakeLanguageModel",
    "FakeAnalysisService",
    "FakeEntityStore",
]


class FakeBlobStorage(IBlobStorageService):
    """Records uploads in memory."""

    def __init__(
        self,
        fail_with: Exception | None = None,
        contents: bytes = b"stored document bytes",
    ) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.downloads: list[str] = []
        self.fail_with = fail_with
        self.contents = contents

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

    def download(self, container: str, blob_name: str) -> bytes:
        self.downloads.append(blob_name)
        return self.contents


class FakeLanguageModel(ILanguageModelService):
    """Returns a canned answer, and records how it was called."""

    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        fail_on_image: Exception | None = None,
        enrichment: dict[str, Any] | None = None,
        structured: dict[str, Any] | None = None,
        fail_enrichment: Exception | None = None,
    ) -> None:
        self.answer = answer or {
            "document_type": "invoice",
            "confidence": 0.94,
            "reasoning": "Has an invoice number and totals.",
        }
        self.enrichment = enrichment or {
            "summary": "An invoice from Acme for 500.00 USD.",
            "enriched": [],
        }
        #: What the structuring path returns for contracts and resumes.
        self.structured = structured or {}
        self.fail_on_image = fail_on_image
        self.fail_enrichment = fail_enrichment
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

        fields = set(schema.model_fields)

        # The agents ask for different shapes; answer whichever was asked for.
        if {"summary", "enriched"} <= fields:
            if self.fail_enrichment is not None:
                raise self.fail_enrichment
            return schema.model_validate(self.enrichment)
        if {"document_type", "confidence", "reasoning"} <= fields:
            return schema.model_validate(self.answer)
        # The structuring path (contracts, resumes).
        return schema.model_validate(self.structured)

    @property
    def used_vision(self) -> bool:
        """Whether any call included an image."""
        return any(call["had_image"] for call in self.calls)


class FakeAnalysisService(IDocumentAnalysisService):
    """Returns canned document text and canned extracted fields."""

    def __init__(
        self,
        text: str = "INVOICE #123 Total due: $500.00",
        fields: list[ExtractedField] | None = None,
        fail_extract: Exception | None = None,
    ) -> None:
        self.text = text
        self.call_count = 0
        # A clean document by default: every score is above the 0.80
        # threshold, so validation passes and the run flows through. Tests
        # that want the correction loop pass their own low-scoring fields.
        self.fields = fields if fields is not None else [
            ExtractedField(name="InvoiceId", value="INV-123", confidence=0.97),
            ExtractedField(name="InvoiceTotal", value=500.0, confidence=0.91),
            ExtractedField(name="VendorName", value="Acme", confidence=None),
        ]
        self.fail_extract = fail_extract
        self.extract_calls: list[str] = []
        self.text_calls: list[str] = []

    def extract_text(self, data: bytes) -> str:
        return self.extract_text_with("prebuilt-read", data)

    def extract_text_with(self, model_id: str, data: bytes) -> str:
        self.call_count += 1
        self.text_calls.append(model_id)
        return self.text

    def extract_fields(self, model_id: str, data: bytes) -> ExtractionResult:
        if self.fail_extract is not None:
            raise self.fail_extract
        self.extract_calls.append(model_id)
        return ExtractionResult(
            model_id=model_id, fields=list(self.fields), page_count=1
        )


class Verdict(BaseModel):
    """Matches the classifier's internal schema, for direct schema tests."""

    document_type: str
    confidence: float
    reasoning: str


class FakeEntityStore(IEntityStore):
    """Records saved entities in memory."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.saved: dict[str, dict[str, Any]] = {}
        self.fail_with = fail_with

    def save(
        self, document_id: str, document_type: str, entity: dict[str, Any]
    ) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.saved[document_id] = {
            "document_type": document_type,
            "entity": entity,
        }
        return document_id

    def get(self, document_id: str) -> dict[str, Any] | None:
        return self.saved.get(document_id)
