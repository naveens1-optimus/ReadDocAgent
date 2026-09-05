"""Azure Document Intelligence adapter (key-based auth)."""

from __future__ import annotations

import datetime
import io
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langsmith import traceable

from application.interface.document_analysis_service import IDocumentAnalysisService
from domain.entity.extraction_result import (
    BoundingRegion,
    ExtractedField,
    ExtractionResult,
)
from domain.schema.settings import DocumentIntelligenceSettings, RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger
from infrastructure.utilities.retry import azure_retry

__all__ = ["AzureDocumentIntelligenceService", "PREBUILT_READ"]

logger = get_logger(__name__)

#: Cheapest prebuilt model: text and page content only.
PREBUILT_READ = "prebuilt-read"


class AzureDocumentIntelligenceService(IDocumentAnalysisService):
    """Analyses documents with Azure Document Intelligence prebuilt models."""

    def __init__(
        self,
        settings: DocumentIntelligenceSettings,
        retry_policy: RetryPolicySettings,
    ) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.endpoint,
            credential=AzureKeyCredential(settings.api_key.get_secret_value()),
        )
        self._analyze_with_retry = azure_retry(retry_policy)(self._analyze_once)

    @traceable(name="document_intelligence_read", run_type="tool")
    def extract_text(self, data: bytes) -> str:
        """Return the document's text using ``prebuilt-read``."""
        result = self._analyze_with_retry(PREBUILT_READ, data)
        text = result.content or ""
        page_count = len(result.pages or [])
        logger.info(
            "Read %s characters across %s page(s) with %s",
            len(text),
            page_count,
            PREBUILT_READ,
            extra={"model_id": PREBUILT_READ, "page_count": page_count},
        )
        return text

    @traceable(name="document_intelligence_extract", run_type="tool")
    def extract_fields(self, model_id: str, data: bytes) -> ExtractionResult:
        """Extract structured fields with the given prebuilt model."""
        result = self._analyze_with_retry(model_id, data)

        documents = getattr(result, "documents", None) or []
        raw_fields = (getattr(documents[0], "fields", None) or {}) if documents else {}

        fields = [
            ExtractedField(
                name=name,
                value=_plain_value(field),
                confidence=_get(field, "confidence"),
                content=_get(field, "content"),
                bounding_regions=_regions(field),
            )
            for name, field in raw_fields.items()
        ]
        page_count = len(getattr(result, "pages", None) or [])

        logger.info(
            "Extracted %s field(s) across %s page(s) with %s",
            len(fields),
            page_count,
            model_id,
            extra={"model_id": model_id, "field_count": len(fields)},
        )
        return ExtractionResult(
            model_id=model_id, fields=fields, page_count=page_count
        )

    def _analyze_once(self, model_id: str, data: bytes):  # type: ignore[no-untyped-def]
        """Run one analysis. Returns the SDK's ``AnalyzeResult``.

        The document is wrapped in a stream because the SDK's ``body``
        parameter expects a file-like object.
        """
        poller = self._client.begin_analyze_document(model_id, body=io.BytesIO(data))
        return poller.result()


# ---------------------------------------------------------------------------
# Mapping Document Intelligence fields to plain JSON
# ---------------------------------------------------------------------------

#: Field attributes holding a simple scalar, checked in order.
_SCALAR_ATTRS = (
    "value_string",
    "value_number",
    "value_integer",
    "value_boolean",
    "value_date",
    "value_time",
    "value_phone_number",
    "value_country_region",
    "value_selection_mark",
    "value_signature",
)


def _get(field: Any, name: str) -> Any:
    """Read an attribute from an SDK model or an equivalent dict.

    The SDK returns objects with snake_case attributes, while recorded
    responses and tests use the service's camelCase JSON. Accepting both keeps
    the mapping testable without constructing SDK models.
    """
    if isinstance(field, dict):
        head, *rest = name.split("_")
        camel = head + "".join(part.title() for part in rest)
        return field.get(name, field.get(camel))
    return getattr(field, name, None)


def _json_safe(value: Any) -> Any:
    """Render dates and times as ISO strings so the value survives JSON."""
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return value


def _plain_value(field: Any) -> Any:
    """Convert one Document Intelligence field into plain JSON.

    Arrays and objects recurse, so an invoice's line items come out as a list
    of dicts a reviewer can read and edit.
    """
    array = _get(field, "value_array")
    if array is not None:
        return [_plain_value(item) for item in array]

    obj = _get(field, "value_object")
    if obj is not None:
        return {name: _plain_value(item) for name, item in obj.items()}

    currency = _get(field, "value_currency")
    if currency is not None:
        return {
            "amount": _get(currency, "amount"),
            "currency": _get(currency, "currency_code"),
        }

    # Addresses come back as a component object; the raw text is more useful
    # to a reviewer than a nested structure.
    if _get(field, "value_address") is not None:
        return _get(field, "content")

    for attr in _SCALAR_ATTRS:
        value = _get(field, attr)
        if value is not None:
            return _json_safe(value)

    return _get(field, "content")


def _regions(field: Any) -> list[BoundingRegion]:
    """Parse a field's bounding regions, skipping any that are unusable."""
    raw = _get(field, "bounding_regions") or []
    parsed = (BoundingRegion.from_azure(region) for region in raw)
    return [region for region in parsed if region is not None]
