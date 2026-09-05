"""The LangGraph pipeline.

    START -> classify -> human_approval -> save -> END
                 |
                 +-----> error_handler -> END

``classify`` asks the classifier agent what the document is. ``human_approval``
is the human-in-the-loop checkpoint: it pauses the graph with ``interrupt()``
when the classifier is not confident enough, and resumes when a reviewer
answers. ``save`` writes the result to Blob Storage.

Extraction becomes a fourth node after ``human_approval``, once the approved
document type is known.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_azure_cosmosdb import CosmosDBSaverSync
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from application.interface.blob_storage_service import IBlobStorageService
from domain.entity.audit_entry import AuditEntry
from domain.enum.document_type import DocumentType
from domain.enum.processing_status import AgentName, ProcessingStatus
from domain.schema.settings import CosmosDbSettings, Settings
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.utilities.logging_config import get_logger
from infrastructure.workflows.state import ApprovalDecision, DocumentState

__all__ = ["DocumentProcessingWorkflow", "build_checkpointer"]

logger = get_logger(__name__)


def build_checkpointer(settings: CosmosDbSettings) -> BaseCheckpointSaver:
    """Open the Cosmos DB checkpointer used to pause and resume runs.

    Persistent rather than in-memory because approval arrives on a *later*
    HTTP request -- often against a different process -- so the paused state
    has to outlive the request that created it.

    Uses the synchronous saver to match the rest of the stack, and passes the
    account key explicitly so key-based auth is used rather than the
    ``DefaultAzureCredential`` fallback. The database and container are
    created on first use.
    """
    return CosmosDBSaverSync(
        database_name=settings.database_name,
        container_name=settings.container_name,
        endpoint=settings.endpoint,
        key=settings.key.get_secret_value(),
    )


class DocumentProcessingWorkflow:
    """Builds and owns the compiled graph."""

    def __init__(
        self,
        classifier: DocumentClassificationAgent,
        storage: IBlobStorageService,
        settings: Settings,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._classifier = classifier
        self._storage = storage
        self._settings = settings
        self.graph = self._build(checkpointer)

    def _build(self, checkpointer: BaseCheckpointSaver) -> Any:
        builder = StateGraph(DocumentState)

        builder.add_node(AgentName.CLASSIFIER.value, self._classify)
        builder.add_node(AgentName.HUMAN_APPROVAL.value, self._human_approval)
        builder.add_node(AgentName.SAVE.value, self._save)
        builder.add_node(AgentName.ERROR_HANDLER.value, self._handle_error)

        builder.add_edge(START, AgentName.CLASSIFIER.value)
        builder.add_conditional_edges(
            AgentName.CLASSIFIER.value,
            self._route_after_classify,
            {
                "approval": AgentName.HUMAN_APPROVAL.value,
                "error": AgentName.ERROR_HANDLER.value,
            },
        )
        builder.add_edge(AgentName.HUMAN_APPROVAL.value, AgentName.SAVE.value)
        builder.add_edge(AgentName.SAVE.value, END)
        builder.add_edge(AgentName.ERROR_HANDLER.value, END)

        return builder.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _classify(self, state: DocumentState) -> dict[str, Any]:
        """Ask the classifier agent what this document is."""
        try:
            result = self._classifier.classify(state.file_bytes, state.extension)
        except Exception as exc:  # noqa: BLE001 - routed to the error node
            logger.exception("Classification failed for %s", state.document_id)
            return {
                "error": str(exc),
                "status": ProcessingStatus.FAILED,
                "file_bytes": b"",
                "audit_trail": [
                    AuditEntry.failure(AgentName.CLASSIFIER, "classify", exc)
                ],
            }

        return {
            "classification": result,
            "status": ProcessingStatus.CLASSIFIED,
            # Drop the file bytes now they have been used, so later
            # checkpoints stay small.
            "file_bytes": b"",
            "audit_trail": [
                AuditEntry.record(
                    AgentName.CLASSIFIER,
                    "classify",
                    document_type=result.document_type.value,
                    confidence=result.confidence,
                    used_fallback=result.used_fallback,
                )
            ],
        }

    def _human_approval(self, state: DocumentState) -> dict[str, Any]:
        """Human-in-the-loop checkpoint.

        Auto-approves a confident, recognised classification so a reviewer is
        only interrupted when it matters. Anything below the threshold, or an
        unrecognised type, pauses the graph until someone answers.

        To review *every* document instead, delete the auto-approve branch.
        """
        classification = state.classification
        assert classification is not None  # guaranteed by the routing edge

        threshold = self._settings.app.confidence_threshold
        confident = (
            classification.confidence >= threshold
            and classification.document_type is not DocumentType.UNSUPPORTED
        )

        if confident:
            decision = ApprovalDecision(approved=True, auto_approved=True)
            logger.info(
                "Auto-approved %s (confidence %.2f >= %.2f)",
                classification.document_type.value,
                classification.confidence,
                threshold,
            )
            return self._approval_update(state, decision)

        # Pauses here. The value below is what the reviewer is shown; the
        # value they send back becomes the return value of interrupt().
        logger.info(
            "Pausing for human approval: %s at confidence %.2f (threshold %.2f)",
            classification.document_type.value,
            classification.confidence,
            threshold,
        )
        answer = interrupt(
            {
                "question": "Is this classification correct?",
                "document_id": state.document_id,
                "file_name": state.file_name,
                "document_type": classification.document_type.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "threshold": threshold,
                "options": [document.value for document in DocumentType],
            }
        )

        answer = answer or {}
        decision = ApprovalDecision(
            approved=bool(answer.get("approved", False)),
            reviewer=answer.get("reviewer"),
            note=answer.get("note"),
            corrected_document_type=answer.get("document_type"),
            auto_approved=False,
        )
        return self._approval_update(state, decision)

    def _approval_update(
        self, state: DocumentState, decision: ApprovalDecision
    ) -> dict[str, Any]:
        """Apply an approval decision, including any correction."""
        update: dict[str, Any] = {
            "approval": decision,
            "status": (
                ProcessingStatus.APPROVED
                if decision.approved
                else ProcessingStatus.REJECTED
            ),
        }

        # A reviewer may fix the type as well as approving it.
        if decision.corrected_document_type and state.classification is not None:
            corrected = DocumentType.from_string(decision.corrected_document_type)
            if corrected is not state.classification.document_type:
                update["classification"] = state.classification.model_copy(
                    update={"document_type": corrected, "confidence": 1.0}
                )
                logger.info(
                    "Reviewer corrected %s -> %s",
                    state.classification.document_type.value,
                    corrected.value,
                )

        update["audit_trail"] = [
            AuditEntry.record(
                AgentName.HUMAN_APPROVAL,
                "auto_approve" if decision.auto_approved else "human_review",
                approved=decision.approved,
                reviewer=decision.reviewer,
                corrected_document_type=decision.corrected_document_type,
            )
        ]
        return update

    def _save(self, state: DocumentState) -> dict[str, Any]:
        """Write the classification result to Blob Storage."""
        payload = {
            "document_id": state.document_id,
            "file_name": state.file_name,
            "input_blob_url": state.input_blob_url,
            "classification": (
                state.classification.model_dump(mode="json")
                if state.classification
                else None
            ),
            "approval": (
                state.approval.model_dump(mode="json") if state.approval else None
            ),
            "audit_trail": [
                entry.model_dump(mode="json") for entry in state.audit_trail
            ],
        }
        data = json.dumps(payload, indent=2).encode("utf-8")

        try:
            url = self._storage.upload(
                container=self._settings.blob_storage.output_container,
                blob_name=f"{state.document_id}/classification.json",
                data=data,
                content_type="application/json",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, run still ends
            logger.exception("Failed to save result for %s", state.document_id)
            return {
                "error": str(exc),
                "status": ProcessingStatus.FAILED,
                "audit_trail": [AuditEntry.failure(AgentName.SAVE, "save", exc)],
            }

        approved = state.approval is not None and state.approval.approved
        return {
            "output_blob_url": url,
            "status": (
                ProcessingStatus.COMPLETED if approved else ProcessingStatus.REJECTED
            ),
            "audit_trail": [
                AuditEntry.record(AgentName.SAVE, "save", output_blob_url=url)
            ],
        }

    def _handle_error(self, state: DocumentState) -> dict[str, Any]:
        """Terminal node for a failed run: log it and record the outcome."""
        logger.error(
            "Run %s failed: %s", state.document_id, state.error,
            extra={"document_id": state.document_id},
        )
        return {
            "status": ProcessingStatus.FAILED,
            "audit_trail": [
                AuditEntry.record(
                    AgentName.ERROR_HANDLER, "handle_error", error=state.error
                )
            ],
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_classify(state: DocumentState) -> str:
        """Send failed classifications to the error node."""
        if state.error or state.classification is None:
            return "error"
        return "approval"
