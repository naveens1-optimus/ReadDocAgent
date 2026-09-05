"""Fields extracted from a document, each with its own confidence score."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["ExtractedField", "ExtractionResult"]


class ExtractedField(BaseModel):
    """One field Document Intelligence found, with its own confidence."""

    name: str
    value: Any = None

    #: Document Intelligence's confidence in this field alone. ``None`` when
    #: the model does not report one for the field -- not every field type
    #: carries a score, so absent is different from low.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    #: The raw text as it appeared on the page, before Azure normalised it.
    content: str | None = None

    #: True once a reviewer changed the value at the approval step.
    edited: bool = False


class ExtractionResult(BaseModel):
    """Everything the extraction step produced for one document."""

    model_id: str = Field(description="Prebuilt model used, e.g. prebuilt-invoice.")
    fields: list[ExtractedField] = Field(default_factory=list)
    page_count: int = 0

    def to_json(self) -> dict[str, Any]:
        """The plain ``{name: value}`` JSON -- what a reviewer edits and what
        is ultimately saved."""
        return {field.name: field.value for field in self.fields}

    def apply_edits(self, edited: dict[str, Any]) -> ExtractionResult:
        """Return a copy with the reviewer's JSON applied.

        Fields the reviewer changed are marked ``edited`` so the saved output
        distinguishes an extracted value from a corrected one. Fields absent
        from ``edited`` were deleted by the reviewer; new keys are added with
        no confidence, since nothing extracted them.
        """
        kept = [
            field.model_copy(
                update={"value": edited[field.name], "edited": True}
            )
            if field.name in edited and edited[field.name] != field.value
            else field.model_copy()
            for field in self.fields
            if field.name in edited
        ]
        known = {field.name for field in self.fields}
        added = [
            ExtractedField(name=name, value=value, edited=True)
            for name, value in edited.items()
            if name not in known
        ]
        return self.model_copy(update={"fields": kept + added})

    @property
    def edited_field_names(self) -> list[str]:
        """Names of fields a reviewer changed or added."""
        return [field.name for field in self.fields if field.edited]

    def low_confidence(self, threshold: float) -> list[str]:
        """Names of scored fields below ``threshold``."""
        return [
            field.name
            for field in self.fields
            if field.confidence is not None and field.confidence < threshold
        ]
