"""The processing report -- Agent 4's deliverable.

Mirrors the four sections the specification asks for:

1. classification result and confidence
2. extraction summary with field-level confidence scores
3. validation results, passed and failed
4. anything flagged for human review
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from domain.entity.classification_result import ClassificationResult
from domain.entity.extraction_result import ExtractionResult
from domain.entity.validation_result import ValidationCheck, ValidationResult

__all__ = ["ProcessingReport"]


class _Classification(BaseModel):
    """Section 1."""

    document_type: str
    confidence: float
    method: str = Field(description="vision or text fallback")
    reasoning: str | None = None
    approved_by: str | None = None
    corrected: bool = False


class _FieldSummary(BaseModel):
    """One row of the field-level confidence table."""

    name: str
    value: Any = None
    confidence: float | None = None
    low_confidence: bool = False
    edited_by_reviewer: bool = False
    pages: list[int] = Field(default_factory=list)
    #: ``(min_x, min_y, max_x, max_y)`` of the field's first region.
    bounding_box: tuple[float, float, float, float] | None = None


class _Extraction(BaseModel):
    """Section 2."""

    model_id: str
    page_count: int = 0
    field_count: int = 0
    average_confidence: float | None = None
    fields: list[_FieldSummary] = Field(default_factory=list)


class _Validation(BaseModel):
    """Section 3."""

    passed: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    checks: list[ValidationCheck] = Field(default_factory=list)


class _Review(BaseModel):
    """Section 4."""

    required: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    fields_edited: list[str] = Field(default_factory=list)
    reviewers: list[str] = Field(default_factory=list)


class ProcessingReport(BaseModel):
    """The full report for one document run."""

    document_id: str
    file_name: str
    status: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    classification: _Classification | None = None
    extraction: _Extraction | None = None
    validation: _Validation | None = None
    review: _Review = Field(default_factory=_Review)

    summary: str | None = Field(default=None, description="From enrichment.")
    entity_id: str | None = Field(
        default=None, description="Id of the entity stored in Cosmos DB."
    )
    error: str | None = None

    @classmethod
    def build(
        cls,
        *,
        document_id: str,
        file_name: str,
        status: str,
        classification: ClassificationResult | None,
        extraction: ExtractionResult | None,
        validation: ValidationResult | None,
        confidence_threshold: float,
        reviewers: list[str] | None = None,
        approval_reviewer: str | None = None,
        corrected_type: bool = False,
        entity_id: str | None = None,
        error: str | None = None,
    ) -> ProcessingReport:
        """Assemble the report from what the pipeline produced."""
        report = cls(
            document_id=document_id,
            file_name=file_name,
            status=status,
            entity_id=entity_id,
            error=error,
        )

        if classification is not None:
            report.classification = _Classification(
                document_type=classification.document_type.value,
                confidence=classification.confidence,
                method=(
                    "text fallback" if classification.used_fallback else "vision"
                ),
                reasoning=classification.reasoning,
                approved_by=approval_reviewer,
                corrected=corrected_type,
            )

        if extraction is not None:
            report.extraction = _build_extraction(extraction, confidence_threshold)

        if validation is not None:
            report.validation = _Validation(
                passed=validation.passed,
                total_checks=len(validation.checks),
                passed_checks=sum(1 for c in validation.checks if c.passed),
                checks=list(validation.checks),
            )
            report.summary = validation.summary
            report.review = _Review(
                required=bool(
                    validation.missing_fields or validation.low_confidence_fields
                ),
                missing_fields=list(validation.missing_fields),
                low_confidence_fields=list(validation.low_confidence_fields),
                fields_edited=(
                    extraction.edited_field_names if extraction else []
                ),
                reviewers=reviewers or [],
            )

        return report


def _build_extraction(
    extraction: ExtractionResult, threshold: float
) -> _Extraction:
    """Build the field-level confidence table, weakest fields first."""
    rows = [
        _FieldSummary(
            name=field.name,
            value=field.value,
            confidence=field.confidence,
            low_confidence=(
                field.confidence is not None and field.confidence < threshold
            ),
            edited_by_reviewer=field.edited,
            pages=field.pages,
            bounding_box=(
                field.bounding_regions[0].bounding_box
                if field.bounding_regions
                else None
            ),
        )
        for field in extraction.fields
    ]
    # A reviewer reads the top of this table, so put the doubtful values there.
    rows.sort(key=lambda row: (row.confidence is None, row.confidence or 0.0))

    scored = [row.confidence for row in rows if row.confidence is not None]
    return _Extraction(
        model_id=extraction.model_id,
        page_count=extraction.page_count,
        field_count=len(rows),
        average_confidence=(sum(scored) / len(scored) if scored else None),
        fields=rows,
    )
