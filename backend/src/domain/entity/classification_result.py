"""Result produced by the document classification agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from domain.enum.document_type import DocumentType

__all__ = ["ClassificationResult"]


class ClassificationResult(BaseModel):
    """What the classifier decided about a document."""

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = Field(
        default=None, description="Short justification from the model."
    )
    used_fallback: bool = Field(
        default=False,
        description="True when vision failed and the text fallback ran.",
    )

    def __str__(self) -> str:
        route = "text fallback" if self.used_fallback else "vision"
        return f"{self.document_type.value} ({self.confidence:.2f}, via {route})"
