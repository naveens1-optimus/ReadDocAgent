"""The LangGraph state schema.

One object travels through every node. Nodes return a dict of the fields they
changed and LangGraph merges it in; ``audit_trail`` uses an additive reducer
so each node appends to the history instead of overwriting it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from domain.entity.audit_entry import AuditEntry
from domain.entity.classification_result import ClassificationResult
from domain.entity.extraction_result import ExtractionResult
from domain.entity.validation_result import ValidationResult
from domain.enum.processing_status import ProcessingStatus

__all__ = ["ApprovalDecision", "ExtractionReview", "DocumentState"]


class ApprovalDecision(BaseModel):
    """What the human reviewer decided at the approval checkpoint."""

    approved: bool
    reviewer: str | None = None
    note: str | None = None
    #: Set when the reviewer corrected the classifier's answer.
    corrected_document_type: str | None = None
    #: True when confidence was high enough to skip asking a human.
    auto_approved: bool = False


class ExtractionReview(BaseModel):
    """What the reviewer decided about the extracted fields."""

    approved: bool
    reviewer: str | None = None
    note: str | None = None
    #: Fields the reviewer changed or added, for the audit trail.
    edited_fields: list[str] = Field(default_factory=list)


class DocumentState(BaseModel):
    """State for one document moving through the graph."""

    # --- Input ---
    document_id: str
    file_name: str
    content_type: str | None = None
    extension: str = ""
    is_pdf: bool = False

    #: Raw file bytes. The classify node clears this once it is done, so the
    #: bytes are not carried in every later checkpoint.
    file_bytes: bytes = b""

    # --- Produced by the nodes ---
    input_blob_url: str | None = None
    #: Blob path of the stored document, so the extract node can re-read it
    #: without the bytes being carried through the graph.
    input_blob_name: str | None = None
    classification: ClassificationResult | None = None
    approval: ApprovalDecision | None = None

    #: What Document Intelligence extracted, each field with its confidence.
    #: After the review step, values are the reviewer's finalised ones.
    extraction: ExtractionResult | None = None
    extraction_review: ExtractionReview | None = None

    #: What validation found, and the enrichment that followed it.
    validation: ValidationResult | None = None

    #: Values a reviewer supplied for required fields the extraction missed.
    completed_fields: dict[str, Any] = Field(default_factory=dict)

    #: Set when a reviewer chose to continue despite fields still missing,
    #: so the graph does not ask again in a loop.
    skip_completion: bool = False

    #: Id of the entity written to Cosmos DB.
    entity_id: str | None = None

    output_blob_url: str | None = None

    status: ProcessingStatus = ProcessingStatus.CLASSIFIED
    error: str | None = None

    #: Appended to by every node. The reducer is what makes the trail
    #: complete rather than last-write-wins.
    audit_trail: Annotated[list[AuditEntry], operator.add] = Field(
        default_factory=list
    )

    def summary(self) -> dict[str, Any]:
        """Compact view of the run, for logs and API responses."""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "status": self.status.value,
            "document_type": (
                self.classification.document_type.value
                if self.classification
                else None
            ),
            "confidence": (
                self.classification.confidence if self.classification else None
            ),
            "steps": len(self.audit_trail),
        }
