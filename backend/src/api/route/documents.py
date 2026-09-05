"""Document endpoints: upload, approve, status.

All three are ``def`` rather than ``async def`` so FastAPI runs them in a
worker thread -- the Azure SDK clients underneath are synchronous.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from application.handler.process_document_handler import (
    ProcessDocumentHandler,
    UnknownDocumentError,
)
from domain.schema.human_review_request import HumanReviewRequest
from domain.schema.process_document_response import ProcessDocumentResponse
from domain.schema.upload_file_request import UploadFileRequest
from infrastructure.utilities.logging_config import get_logger

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_handler(request: Request) -> ProcessDocumentHandler:
    """Resolve the handler, or fail clearly if the app is not configured."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Azure services are not configured. See /health/ready for the "
                "missing setting."
            ),
        )
    return container.resolve(ProcessDocumentHandler)


@router.post(
    "",
    response_model=ProcessDocumentResponse,
    summary="Upload and classify a document",
    description=(
        "Stores the document in Blob Storage and runs the LangGraph pipeline: "
        "classify, then the human approval checkpoint, then save.\n\n"
        "Omit `session_id` on the first upload and the API generates one; send "
        "it back on later uploads to keep them in the same session.\n\n"
        "If the classifier is confident the run completes immediately. If not, "
        "the response has `awaiting_approval: true` with an `approval_request` "
        "payload -- send the decision to "
        "`POST /documents/{document_id}/approval` to continue."
    ),
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(
        default=None, description="Omit on the first upload; the API returns one."
    ),
) -> ProcessDocumentResponse:
    """Accept a PDF or image and run it through the pipeline."""
    handler = _get_handler(request)
    data = file.file.read()

    try:
        upload = UploadFileRequest(
            file_name=file.filename or "",
            content_type=file.content_type,
            size_bytes=len(data),
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_first_error(exc)
        ) from exc

    try:
        return handler.handle(upload, data, session_id=session_id or None)
    except ValueError as exc:  # size limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc


@router.post(
    "/{document_id}/approval",
    response_model=ProcessDocumentResponse,
    summary="Approve or reject a paused classification",
    description=(
        "Resumes a run paused at the human-in-the-loop checkpoint. Set "
        "`document_type` to correct the classifier while approving."
    ),
)
def submit_approval(
    request: Request, document_id: str, review: HumanReviewRequest
) -> ProcessDocumentResponse:
    """Resume a paused run with a reviewer's decision."""
    handler = _get_handler(request)
    try:
        return handler.resume(document_id, review)
    except UnknownDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get(
    "/{document_id}",
    response_model=ProcessDocumentResponse,
    summary="Get the status of a document",
)
def get_document(request: Request, document_id: str) -> ProcessDocumentResponse:
    """Return the current state of a run, including any pending approval."""
    handler = _get_handler(request)
    try:
        return handler.get_status(document_id)
    except UnknownDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


def _first_error(exc: ValidationError) -> str:
    """Return the first validation message, without Pydantic's prefix."""
    errors = exc.errors()
    if not errors:
        return "Invalid upload."
    message = str(errors[0].get("msg", "Invalid upload."))
    return message.removeprefix("Value error, ")
