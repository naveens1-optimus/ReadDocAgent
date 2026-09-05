"""Output of the validation and enrichment agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["CheckCategory", "ValidationCheck", "ValidationResult"]


#: The three dimensions the specification asks Agent 3 to check.
CheckCategory = str  # "completeness" | "consistency" | "confidence"


class ValidationCheck(BaseModel):
    """The outcome of one named rule.

    Passing checks are kept as well as failing ones -- the processing report
    has to show what was verified, not just what went wrong.
    """

    name: str
    category: CheckCategory
    passed: bool
    message: str
    field: str | None = None

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.message}"


class ValidationResult(BaseModel):
    """Everything validation found, plus the enrichment that followed it."""

    checks: list[ValidationCheck] = Field(default_factory=list)

    #: Required schema fields the extraction could not supply. These are what
    #: the human-in-the-loop step asks a reviewer to fill in.
    missing_fields: list[str] = Field(default_factory=list)

    #: Fields whose Document Intelligence confidence was below the threshold.
    low_confidence_fields: list[str] = Field(default_factory=list)

    #: The validated entity, once every required field is present.
    entity: dict[str, Any] | None = None

    # --- Enrichment (Azure OpenAI) ---
    summary: str | None = Field(default=None, description="Brief document summary.")
    enriched_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Values normalised or inferred during enrichment.",
    )
    enrichment_error: str | None = Field(
        default=None,
        description=(
            "Set when enrichment failed. It is non-fatal: losing a summary is "
            "not worth failing an otherwise good extraction."
        ),
    )

    @property
    def is_complete(self) -> bool:
        """Whether every required field is present."""
        return not self.missing_fields

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[ValidationCheck]:
        """Only the failures."""
        return [check for check in self.checks if not check.passed]

    def summary_line(self) -> str:
        """One line for the logs."""
        passed = sum(1 for check in self.checks if check.passed)
        return (
            f"{passed}/{len(self.checks)} checks passed, "
            f"{len(self.missing_fields)} field(s) missing, "
            f"{len(self.low_confidence_fields)} low confidence"
        )
