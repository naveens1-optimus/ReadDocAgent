"""Reviewer's answer at the human-in-the-loop approval checkpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from domain.enum.document_type import DocumentType

__all__ = ["HumanReviewRequest"]


class HumanReviewRequest(BaseModel):
    """What a reviewer sends back to resume a paused run."""

    approved: bool = Field(description="Accept the classification and continue.")
    document_type: DocumentType | None = Field(
        default=None,
        description="Set to correct the classifier's answer while approving.",
    )
    #: The finalised JSON, sent at the extraction review step. Whatever is
    #: here is what gets saved -- the reviewer may change values, drop fields
    #: or add new ones. Omitted at the classification step.
    fields: dict[str, Any] | None = Field(
        default=None, description="Finalised {field: value} JSON."
    )

    #: At the field-completion gate: continue even though required fields are
    #: still missing. Without it a reviewer who cannot supply a value would be
    #: asked the same question forever.
    skip: bool = Field(default=False)

    reviewer: str | None = Field(default=None, max_length=256)
    note: str | None = Field(default=None, max_length=2000)

    def to_resume_payload(self) -> dict[str, Any]:
        """Render the value handed to LangGraph's ``Command(resume=...)``.

        A plain dict, because the checkpointer stores whatever it is given and
        keeping it schema-free makes saved checkpoints easier to read back.
        """
        return {
            "approved": self.approved,
            "document_type": (
                self.document_type.value if self.document_type else None
            ),
            "fields": self.fields,
            "skip": self.skip,
            "reviewer": self.reviewer,
            "note": self.note,
        }
