"""Use cases for running and resuming the document pipeline."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command

from application.interface.blob_storage_service import IBlobStorageService
from domain.schema.human_review_request import HumanReviewRequest
from domain.schema.process_document_response import ProcessDocumentResponse
from domain.schema.settings import Settings
from domain.schema.upload_file_request import UploadFileRequest
from infrastructure.utilities.logging_config import correlation_id_scope, get_logger
from infrastructure.workflows.document_processing_workflow import (
    DocumentProcessingWorkflow,
)
from infrastructure.workflows.state import DocumentState

__all__ = ["ProcessDocumentHandler", "UnknownDocumentError"]

logger = get_logger(__name__)


class UnknownDocumentError(LookupError):
    """Raised when a document id does not match any run."""


class ProcessDocumentHandler:
    """Stores an uploaded document and runs it through the graph."""

    def __init__(
        self,
        workflow: DocumentProcessingWorkflow,
        storage: IBlobStorageService,
        settings: Settings,
    ) -> None:
        self._workflow = workflow
        self._storage = storage
        self._settings = settings

    def handle(
        self,
        upload: UploadFileRequest,
        data: bytes,
        session_id: str | None = None,
    ) -> ProcessDocumentResponse:
        """Store the document, then run the graph until it finishes or pauses.

        Args:
            upload: Validated upload metadata.
            data: The raw file bytes.
            session_id: The caller's session. A new one is generated when
                omitted, which is how a client gets its first session id.

        Raises:
            ValueError: If the file exceeds the configured size limit.
        """
        upload.ensure_within_size_limit(self._settings.app.max_upload_size_bytes)

        session_id = session_id or new_session_id()
        document_id = uuid4().hex

        with correlation_id_scope(document_id):
            logger.info(
                "Processing %s (%s bytes) in session %s",
                upload.file_name,
                upload.size_bytes,
                session_id,
            )

            input_blob_url = self._storage.upload(
                container=self._settings.blob_storage.input_container,
                blob_name=f"{session_id}/{document_id}/{upload.file_name}",
                data=data,
                content_type=upload.content_type,
            )

            initial_state = DocumentState(
                document_id=document_id,
                session_id=session_id,
                file_name=upload.file_name,
                content_type=upload.content_type,
                extension=upload.extension,
                is_pdf=upload.is_pdf,
                file_bytes=data,
                input_blob_url=input_blob_url,
            )

            result = self._workflow.graph.invoke(
                initial_state, config=_run_config(document_id)
            )
            return _to_response(result, document_id)

    def resume(
        self, document_id: str, review: HumanReviewRequest
    ) -> ProcessDocumentResponse:
        """Resume a paused run with the reviewer's decision.

        Raises:
            UnknownDocumentError: If no run exists for ``document_id``.
        """
        config = _run_config(document_id)
        self._require_existing_run(config, document_id)

        with correlation_id_scope(document_id):
            logger.info(
                "Resuming %s (approved=%s, reviewer=%s)",
                document_id,
                review.approved,
                review.reviewer,
            )
            result = self._workflow.graph.invoke(
                Command(resume=review.to_resume_payload()), config=config
            )
            return _to_response(result, document_id)

    def get_status(self, document_id: str) -> ProcessDocumentResponse:
        """Return the current state of a run.

        Raises:
            UnknownDocumentError: If no run exists for ``document_id``.
        """
        config = _run_config(document_id)
        snapshot = self._require_existing_run(config, document_id)

        state = DocumentState.model_validate(snapshot.values)
        return ProcessDocumentResponse.from_state(state, _pending_approval(snapshot))

    def _require_existing_run(self, config: dict[str, Any], document_id: str) -> Any:
        """Return the state snapshot, or raise if the document is unknown."""
        snapshot = self._workflow.graph.get_state(config)
        if not snapshot.values:
            raise UnknownDocumentError(f"No run found for document {document_id!r}")
        return snapshot


def new_session_id() -> str:
    """Create an identifier for a new user session."""
    return uuid4().hex


def _run_config(document_id: str) -> dict[str, Any]:
    """Config telling the checkpointer which run to read or resume.

    Each document is its own LangGraph thread, keyed on the document id. A
    session may contain several documents, and giving them one shared thread
    would mean the second upload inherited the first's state -- the audit
    trail uses an additive reducer -- and could clobber an approval still
    pending on the first.
    """
    return {"configurable": {"thread_id": document_id}}


def _pending_approval(snapshot: Any) -> dict[str, Any] | None:
    """Return the interrupt payload if the run is paused, else None."""
    for task in getattr(snapshot, "tasks", ()) or ():
        for task_interrupt in getattr(task, "interrupts", ()) or ():
            return task_interrupt.value
    return None


def _to_response(result: Any, document_id: str) -> ProcessDocumentResponse:
    """Turn the value returned by ``graph.invoke`` into a response.

    When the graph pauses, the result carries the interrupt alongside the
    state, so the payload is pulled out and reported as a pending approval.
    """
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    approval_request = interrupts[0].value if interrupts else None

    state = DocumentState.model_validate(result)
    response = ProcessDocumentResponse.from_state(state, approval_request)
    logger.info("Document %s: %s", document_id, response.status.value)
    return response
