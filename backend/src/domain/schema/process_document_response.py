"""Response schema for the document endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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

    #: Set when the graph paused at the approval checkpoint. Contains what the
    #: reviewer needs to decide.
    awaiting_approval: bool = False
    approval_request: dict[str, Any] | None = None

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
        return cls(
            document_id=state.document_id,
            file_name=state.file_name,
            status=(
                ProcessingStatus.AWAITING_APPROVAL
                if approval_request is not None
                else state.status
            ),
            document_type=classification.document_type if classification else None,
            confidence=classification.confidence if classification else None,
            reasoning=classification.reasoning if classification else None,
            used_fallback=classification.used_fallback if classification else False,
            awaiting_approval=approval_request is not None,
            approval_request=approval_request,
            error=state.error,
        )
