"""Response schema for the document endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from domain.entity.extraction_result import ExtractedField
from domain.enum.document_type import DocumentType
from domain.enum.processing_status import ProcessingStatus

__all__ = ["ProcessDocumentResponse"]


class ProcessDocumentResponse(BaseModel):
    """State of one document run."""

    #: The identifier for this run. Use it to poll status or submit an
    #: approval.
    document_id: str

    file_name: str
    status: ProcessingStatus

    document_type: DocumentType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    used_fallback: bool = False

    #: Set when the graph paused at either checkpoint. ``approval_request``
    #: carries what the reviewer needs; its ``stage`` says which checkpoint --
    #: "classification" or "extraction".
    awaiting_approval: bool = False
    approval_request: dict[str, Any] | None = None

    #: Which prebuilt model extracted the fields, once extraction has run.
    extraction_model: str | None = None

    #: The extracted fields, each with its own confidence score. After the
    #: extraction review these hold the reviewer's finalised values.
    fields: list[ExtractedField] = Field(default_factory=list)

    #: The finalised ``{name: value}`` JSON -- the same data as ``fields``,
    #: flattened for consumers that only want the values.
    data: dict[str, Any] = Field(default_factory=dict)

    #: The validated entity. Present only once every check has passed --
    #: while validation is still failing there is no entity, and nothing has
    #: been stored.
    entity: dict[str, Any] | None = None

    #: Short document summary, from enrichment.
    summary: str | None = None

    #: Required schema fields the extraction could not supply.
    missing_fields: list[str] = Field(default_factory=list)

    #: Fields whose extraction confidence was below the threshold.
    low_confidence_fields: list[str] = Field(default_factory=list)

    #: Validation checks, passed and failed.
    validation_checks: list[dict[str, Any]] = Field(default_factory=list)

    error: str | None = None

    # Blob URLs and the audit trail are deliberately not returned. Both are
    # still recorded -- the audit trail in the graph state, and both in the
    # result JSON written to Blob Storage -- but the API response stays a
    # summary of the outcome rather than a dump of the run.

    @classmethod
    def from_state(
        cls,
        state: Any,
        approval_request: dict[str, Any] | None = None,
    ) -> ProcessDocumentResponse:
        """Build a response from the graph's ``DocumentState``."""
        classification = state.classification
        extraction = state.extraction
        validation = state.validation
        return cls(
            document_id=state.document_id,
            file_name=state.file_name,
            status=_paused_status(approval_request) or state.status,
            document_type=classification.document_type if classification else None,
            confidence=classification.confidence if classification else None,
            reasoning=classification.reasoning if classification else None,
            used_fallback=classification.used_fallback if classification else False,
            awaiting_approval=approval_request is not None,
            approval_request=approval_request,
            extraction_model=extraction.model_id if extraction else None,
            fields=list(extraction.fields) if extraction else [],
            data=extraction.to_json() if extraction else {},
            entity=(
                validation.entity
                if validation is not None and validation.passed
                else None
            ),
            summary=validation.summary if validation else None,
            missing_fields=list(validation.missing_fields) if validation else [],
            low_confidence_fields=(
                list(validation.low_confidence_fields) if validation else []
            ),
            validation_checks=(
                [check.model_dump(mode="json") for check in validation.checks]
                if validation
                else []
            ),
            error=state.error,
        )


def _paused_status(approval_request: dict[str, Any] | None) -> ProcessingStatus | None:
    """Map a pending interrupt to the status that names which gate it is.

    Returns ``None`` when the run is not paused, so the caller falls back to
    the state's own status.
    """
    if approval_request is None:
        return None
    stage = approval_request.get("stage")
    if stage == "extraction":
        return ProcessingStatus.AWAITING_EXTRACTION_REVIEW
    if stage == "data_correction":
        return ProcessingStatus.AWAITING_DATA_CORRECTION
    return ProcessingStatus.AWAITING_APPROVAL
