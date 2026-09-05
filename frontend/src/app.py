"""Streamlit UI for the document processing pipeline.

Run the API first, then:

    streamlit run frontend/src/app.py

Talks to three endpoints:

    GET  /health/ready              can the backend do work?
    POST /documents                 upload and run the graph
    POST /documents/{id}/approval   resume a run paused for approval

Requests are made from Streamlit's own Python process, not the browser, so no
CORS configuration is needed on the API.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

#: Generous, because a cold classify does a Blob upload plus an Azure OpenAI
#: vision call, and can fall back to Document Intelligence on top of that.
REQUEST_TIMEOUT_SECONDS = 120

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "heif"]

STATUS_COLOURS = {
    "completed": "green",
    "approved": "green",
    "awaiting_approval": "orange",
    "classified": "blue",
    "rejected": "red",
    "failed": "red",
}


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def api_error_message(response: requests.Response) -> str:
    """Pull the API's error message out of a failed response."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    return str(
        body.get("message") or body.get("detail") or f"HTTP {response.status_code}"
    )


def check_readiness(api_url: str) -> tuple[bool, list[dict[str, Any]]]:
    """Return whether the backend is ready, plus its component list."""
    response = requests.get(
        f"{api_url}/health/ready", timeout=REQUEST_TIMEOUT_SECONDS
    )
    body = response.json()
    return bool(body.get("ready")), body.get("components", [])


def upload_document(api_url: str, name: str, data: bytes, content_type: str) -> dict:
    """Upload a document and run it through the graph."""
    response = requests.post(
        f"{api_url}/documents",
        files={"file": (name, data, content_type)},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))
    return response.json()


def submit_approval(
    api_url: str,
    document_id: str,
    approved: bool,
    document_type: str | None,
    reviewer: str | None,
    note: str | None,
) -> dict:
    """Resume a paused run with the reviewer's decision."""
    response = requests.post(
        f"{api_url}/documents/{document_id}/approval",
        json={
            "approved": approved,
            "document_type": document_type,
            "reviewer": reviewer or None,
            "note": note or None,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))
    return response.json()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Draw the sidebar and return the API base URL."""
    st.sidebar.header("Backend")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")

    try:
        ready, components = check_readiness(api_url)
    except requests.RequestException as exc:
        st.sidebar.error("Cannot reach the API.")
        st.sidebar.caption(str(exc))
        return api_url

    if ready:
        st.sidebar.success("Ready")
    else:
        st.sidebar.error("Not ready")
        for component in components:
            if not component.get("ready"):
                st.sidebar.caption(
                    f"**{component['name']}** — {component.get('detail') or 'not ready'}"
                )
        st.sidebar.info(
            "Fill in `backend/src/.env` with your Azure credentials, "
            "then restart the API."
        )

    return api_url


def render_approval(api_url: str, result: dict) -> None:
    """Draw the human-in-the-loop approval form for a paused run."""
    request = result.get("approval_request") or {}

    st.warning("The classifier was not confident enough, so the run paused here.")

    left, right = st.columns(2)
    left.metric("Classified as", str(request.get("document_type", "unknown")))
    right.metric(
        "Confidence",
        f"{float(request.get('confidence') or 0):.0%}",
        delta=f"threshold {float(request.get('threshold') or 0):.0%}",
        delta_color="off",
    )

    if request.get("reasoning"):
        st.caption(f"Model's reasoning: _{request['reasoning']}_")

    options: list[str] = list(request.get("options") or [])
    current = request.get("document_type")
    index = options.index(current) if current in options else 0

    with st.form("approval_form"):
        document_type = st.selectbox(
            "Document type",
            options,
            index=index,
            help="Change this to correct the classifier while approving.",
        )
        reviewer = st.text_input("Your name (optional)")
        note = st.text_input("Note (optional)")

        approve_column, reject_column = st.columns(2)
        approved = approve_column.form_submit_button(
            "Approve", type="primary", use_container_width=True
        )
        rejected = reject_column.form_submit_button(
            "Reject", use_container_width=True
        )

    if not (approved or rejected):
        return

    with st.spinner("Resuming the pipeline..."):
        try:
            st.session_state.result = submit_approval(
                api_url,
                result["document_id"],
                approved=approved,
                document_type=document_type,
                reviewer=reviewer,
                note=note,
            )
        except (RuntimeError, requests.RequestException) as exc:
            st.error(str(exc))
            return
    st.rerun()


def render_result(result: dict) -> None:
    """Draw the outcome of a run."""
    status = str(result.get("status", "unknown"))
    colour = STATUS_COLOURS.get(status, "gray")
    st.markdown(f"### Result &nbsp; :{colour}[{status.replace('_', ' ')}]")

    confidence = float(result.get("confidence") or 0)
    columns = st.columns(3)
    columns[0].metric("Document type", str(result.get("document_type") or "—"))
    columns[1].metric("Confidence", f"{confidence:.0%}")
    columns[2].metric(
        "Route", "text fallback" if result.get("used_fallback") else "vision"
    )

    st.progress(min(max(confidence, 0.0), 1.0))

    if result.get("reasoning"):
        st.caption(f"_{result['reasoning']}_")

    if result.get("error"):
        st.error(result["error"])

    st.caption(f"Document id: `{result.get('document_id', '')}`")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="Document Processing", page_icon="📄")
    st.title("Intelligent Document Processing")
    st.caption(
        "Upload a document. It is stored in Blob Storage, classified by the "
        "LangGraph pipeline, and paused for your approval when the classifier "
        "is unsure."
    )

    api_url = render_sidebar()

    uploaded = st.file_uploader(
        "Choose a PDF or image", type=SUPPORTED_TYPES, accept_multiple_files=False
    )

    if st.button("Upload & classify", type="primary", disabled=uploaded is None):
        with st.spinner("Storing and classifying..."):
            try:
                st.session_state.result = upload_document(
                    api_url,
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                )
            except (RuntimeError, requests.RequestException) as exc:
                st.session_state.result = None
                st.error(str(exc))

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    if result.get("awaiting_approval"):
        render_approval(api_url, result)
        st.divider()
    render_result(result)


main()
