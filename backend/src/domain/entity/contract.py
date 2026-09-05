"""The canonical contract entity.

Contracts have no dedicated Azure prebuilt model, so ``prebuilt-layout``
recovers the text and structure and Azure OpenAI fills this schema in. That
means the field names here are also what the model is asked to produce.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from domain.entity.common import EntityModel

__all__ = ["Signature", "Contract"]


class Signature(EntityModel):
    """A signature block."""

    signatory_name: str | None = None
    party: str | None = Field(default=None, description="Which party they signed for.")
    signed_date: date | None = None


class Contract(EntityModel):
    """A contract, as the pipeline stores it."""

    # --- Required ---
    title: str
    parties: list[str] = Field(
        min_length=1, description="Names of the parties bound by the agreement."
    )

    # --- Optional ---
    effective_date: date | None = None
    expiration_date: date | None = None
    contract_type: str | None = Field(
        default=None, description="e.g. NDA, MSA, lease, employment."
    )
    key_clauses: list[str] = Field(
        default_factory=list,
        description="Short summaries of the clauses that matter.",
    )
    signatures: list[Signature] = Field(default_factory=list)
    governing_law: str | None = None
