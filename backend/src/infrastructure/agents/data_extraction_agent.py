"""Agent 2: extracts structured fields using a type-specific prebuilt model.

Two paths, decided by the model the document type maps to:

* ``prebuilt-invoice`` / ``prebuilt-receipt`` return **named fields with
  confidence scores**, which are used directly.
* ``prebuilt-layout`` / ``prebuilt-read`` return only text, so Azure OpenAI
  structures it into the entity schema for that type. There are no per-field
  confidence scores on this path, and the fields say so by reporting ``None``
  rather than inventing a number.
"""

from __future__ import annotations

import json
from typing import Any

from langsmith import traceable
from pydantic import create_model

from application.interface.document_analysis_service import IDocumentAnalysisService
from application.interface.language_model_service import ILanguageModelService
from domain.entity.extraction_result import ExtractedField, ExtractionResult
from domain.entity.registry import entity_for
from domain.enum.document_type import DocumentType
from infrastructure.utilities.logging_config import get_logger

__all__ = ["DataExtractionAgent"]

logger = get_logger(__name__)

STRUCTURING_PROMPT = """\
You read a document and fill in a structured record.

Use only what the document actually says. Leave a field out rather than
guessing at it -- a missing field is handled downstream, an invented one is
not. Keep dates in ISO 8601 (YYYY-MM-DD) where the document makes the date
clear.
"""

#: Text beyond this is dropped before the model sees it. Long contracts
#: otherwise blow past the context window; the fields we want are near the
#: start, and a truncated read is better than a failed one.
MAX_TEXT_CHARS = 40_000


class DataExtractionAgent:
    """Runs the Document Intelligence model that matches the document type."""

    def __init__(
        self,
        analysis: IDocumentAnalysisService,
        language_model: ILanguageModelService,
    ) -> None:
        self._analysis = analysis
        self._language_model = language_model

    @traceable(name="data_extraction", run_type="chain")
    def extract(self, document_type: DocumentType, data: bytes) -> ExtractionResult:
        """Extract fields for a document of the given type.

        Raises:
            ValueError: If the type has no extraction model. Callers should
                check ``document_type.is_extractable`` first.
        """
        model = document_type.extraction_model
        if model is None:
            raise ValueError(
                f"No extraction model for document type {document_type.value!r}"
            )

        logger.info("Extracting %s with %s", document_type.value, model.value)

        if model.returns_typed_fields:
            return self._analysis.extract_fields(model.value, data)
        return self._structure_text(document_type, model.value, data)

    def _structure_text(
        self, document_type: DocumentType, model_id: str, data: bytes
    ) -> ExtractionResult:
        """Read the document as text, then have Azure OpenAI fill the schema."""
        entity_type = entity_for(document_type)
        if entity_type is None:
            raise ValueError(
                f"No entity schema to structure {document_type.value!r} into"
            )

        text = self._analysis.extract_text_with(model_id, data)
        truncated = text[:MAX_TEXT_CHARS]
        if len(text) > MAX_TEXT_CHARS:
            logger.warning(
                "Document text truncated from %s to %s characters for structuring",
                len(text),
                MAX_TEXT_CHARS,
            )

        # Every field optional: the model should omit what it cannot find, and
        # the validation agent decides what is actually required.
        lenient = create_model(  # type: ignore[call-overload]
            f"Draft{entity_type.__name__}",
            **{
                name: (field.annotation | None, None)
                for name, field in entity_type.model_fields.items()
            },
        )
        lenient.__doc__ = f"Fields to read out of a {document_type.value}."

        answer = self._language_model.structured_completion(
            system_prompt=STRUCTURING_PROMPT,
            user_prompt=(
                f"Document type: {document_type.value}\n\n"
                f"Document text:\n{truncated}"
            ),
            schema=lenient,
        )

        fields = [
            ExtractedField(name=name, value=_json_safe(value))
            for name, value in answer.model_dump(mode="json").items()
            if value not in (None, [], {}, "")
        ]
        logger.info(
            "Structured %s field(s) from %s characters of %s text",
            len(fields),
            len(truncated),
            model_id,
        )
        return ExtractionResult(model_id=model_id, fields=fields, page_count=0)


def _json_safe(value: Any) -> Any:
    """Keep values JSON-serialisable, since they are checkpointed and stored."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return json.loads(json.dumps(value, default=str))
    return str(value)
