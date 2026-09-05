"""Pipeline status values."""

from __future__ import annotations

from enum import Enum

__all__ = ["ProcessingStatus", "AgentName"]


class ProcessingStatus(str, Enum):
    """Where a document has got to in the pipeline."""

    CLASSIFIED = "classified"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXTRACTED = "extracted"
    AWAITING_EXTRACTION_REVIEW = "awaiting_extraction_review"
    VALIDATED = "validated"
    AWAITING_DATA_CORRECTION = "awaiting_data_correction"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether the run is finished and will not continue on its own."""
        return self in {
            ProcessingStatus.COMPLETED,
            ProcessingStatus.REJECTED,
            ProcessingStatus.FAILED,
        }


class AgentName(str, Enum):
    """Graph node names. Used as both node ids and audit-trail labels."""

    CLASSIFIER = "classify"
    HUMAN_APPROVAL = "human_approval"
    EXTRACTOR = "extract"
    EXTRACTION_REVIEW = "extraction_review"
    VALIDATOR = "validate"
    DATA_CORRECTION = "correct_data"
    ENRICHER = "enrich"
    #: Agent 4: builds the report, stores the entity and the output.
    SAVE = "save"
    ERROR_HANDLER = "error_handler"
