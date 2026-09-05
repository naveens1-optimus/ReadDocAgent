"""Response schema for the document endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from domain.entity.audit_entry import AuditEntry
from domain.enum.document_type import DocumentType
from domain.enum.processing_status import ProcessingStatus

__all__ = ["ProcessDocumentResponse"]


class ProcessDocumentResponse(BaseModel):
    """State of one document run."""

    document_id: str
    file_name: str
    status: ProcessingStatus

    #: LangGraph thread id. Keep it: this is the key used to resume a paused
    #: run and to poll its status.
    thread_id: str

    document_type: DocumentType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    used_fallback: bool = False

    #: Set when the graph paused at the approval checkpoint. Contains what the
    #: reviewer needs to decide.
    awaiting_approval: bool = False
    approval_request: dict[str, Any] | None = None

    input_blob_url: str | None = None
    output_blob_url: str | None = None

    audit_trail: list[AuditEntry] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_state(
        cls,
        state: Any,
        thread_id: str,
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
            thread_id=thread_id,
            document_type=classification.document_type if classification else None,
            confidence=classification.confidence if classification else None,
            reasoning=classification.reasoning if classification else None,
            used_fallback=classification.used_fallback if classification else False,
            awaiting_approval=approval_request is not None,
            approval_request=approval_request,
            input_blob_url=state.input_blob_url,
            output_blob_url=state.output_blob_url,
            audit_trail=state.audit_trail,
            error=state.error,
        )
