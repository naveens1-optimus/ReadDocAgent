"""Tests for the document upload, approval and status endpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from api.app import create_app
from application.handler.process_document_handler import ProcessDocumentHandler
from infrastructure.agents.data_extraction_agent import DataExtractionAgent
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.config.settings import build_settings
from infrastructure.di_container import DIContainer
from infrastructure.workflows.document_processing_workflow import (
    DocumentProcessingWorkflow,
)
from tests.fakes import FakeAnalysisService, FakeBlobStorage, FakeLanguageModel

PNG_BYTES = b"\x89PNG\r\n\x1a\n fake image bytes"

CONFIDENT = {
    "document_type": "invoice",
    "confidence": 0.94,
    "reasoning": "Has an invoice number and totals.",
}
UNSURE = {
    "document_type": "invoice",
    "confidence": 0.41,
    "reasoning": "Layout is unclear.",
}


@dataclass
class Harness:
    """A running app whose Azure services are fakes."""

    client: TestClient
    storage: FakeBlobStorage
    language_model: FakeLanguageModel


@pytest.fixture
def make_harness(valid_env: dict[str, str]):
    """Return a factory that builds an app wired to fakes."""
    clients: list[TestClient] = []

    def _build(answer: dict | None = None) -> Harness:
        app = create_app()
        client = TestClient(app)
        client.__enter__()
        clients.append(client)

        settings = build_settings(load_dotenv_file=False)
        language_model = FakeLanguageModel(answer or CONFIDENT)
        storage = FakeBlobStorage()
        analysis = FakeAnalysisService()
        workflow = DocumentProcessingWorkflow(
            classifier=DocumentClassificationAgent(language_model, analysis),
            extractor=DataExtractionAgent(analysis),
            storage=storage,
            settings=settings,
            checkpointer=SqliteSaver(
                sqlite3.connect(":memory:", check_same_thread=False)
            ),
        )

        container = DIContainer()
        container.register(
            ProcessDocumentHandler,
            ProcessDocumentHandler(workflow, storage, settings),
        )
        # Replace the real Azure-backed container built during startup.
        app.state.container = container

        return Harness(client=client, storage=storage, language_model=language_model)

    yield _build

    for client in clients:
        client.__exit__(None, None, None)


def upload(client: TestClient, name: str = "invoice.png", data: bytes = PNG_BYTES):
    return client.post(
        "/documents", files={"file": (name, data, "image/png")}
    )


class TestUpload:
    def test_confident_document_runs_on_to_extraction_review(
        self, make_harness
    ) -> None:
        """Confidence skips the classification gate, never the extraction one."""
        harness = make_harness(CONFIDENT)

        response = upload(harness.client)

        assert response.status_code == 200
        body = response.json()
        assert body["document_type"] == "invoice"
        assert body["confidence"] == pytest.approx(0.94)
        assert body["document_id"]
        # Auto-approved the classification, then paused on the data.
        assert body["status"] == "awaiting_extraction_review"
        assert body["awaiting_approval"] is True
        assert body["approval_request"]["stage"] == "extraction"

    def test_stores_the_input_document(self, make_harness) -> None:
        harness = make_harness(CONFIDENT)

        upload(harness.client)

        containers = [item["container"] for item in harness.storage.uploads]
        assert "idp-input" in containers
        # Nothing saved yet: the reviewer has not signed off the data.
        assert "idp-output" not in containers

    def test_response_omits_internals(self, make_harness) -> None:
        """Blob URLs and the audit trail stay server-side.

        Both are still recorded -- the trail in graph state, and both in the
        saved result JSON -- but the response is a summary, not a run dump.
        """
        harness = make_harness(CONFIDENT)

        body = upload(harness.client).json()

        for field in ("audit_trail", "input_blob_url", "output_blob_url"):
            assert field not in body

    def test_rejects_unsupported_file_type(self, make_harness) -> None:
        harness = make_harness()

        response = harness.client.post(
            "/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        assert response.status_code == 400
        assert "unsupported file type" in response.json()["message"].lower()

    def test_rejects_empty_file(self, make_harness) -> None:
        harness = make_harness()

        response = harness.client.post(
            "/documents", files={"file": ("empty.pdf", b"", "application/pdf")}
        )

        assert response.status_code == 400

    def test_returns_503_when_not_configured(
        self, clean_env: None
    ) -> None:
        """Without credentials the container is never built."""
        with TestClient(create_app()) as client:
            response = client.post(
                "/documents", files={"file": ("a.png", PNG_BYTES, "image/png")}
            )

        assert response.status_code == 503
        assert "not configured" in response.json()["message"].lower()


class TestApprovalCheckpoint:
    def test_low_confidence_pauses_and_reports_what_to_review(
        self, make_harness
    ) -> None:
        harness = make_harness(UNSURE)

        body = upload(harness.client).json()

        assert body["status"] == "awaiting_approval"
        assert body["awaiting_approval"] is True
        request = body["approval_request"]
        assert request["document_type"] == "invoice"
        assert request["confidence"] == pytest.approx(0.41)
        assert request["threshold"] == pytest.approx(0.80)

    def test_nothing_saved_before_approval(self, make_harness) -> None:
        harness = make_harness(UNSURE)

        upload(harness.client)

        containers = [item["container"] for item in harness.storage.uploads]
        assert "idp-output" not in containers

    def test_approving_completes_the_run(self, make_harness) -> None:
        harness = make_harness(UNSURE)
        document_id = upload(harness.client).json()["document_id"]

        response = harness.client.post(
            f"/documents/{document_id}/approval",
            json={"approved": True, "reviewer": "naveen"},
        )

        assert response.status_code == 200
        body = response.json()
        # Approving the classification advances to the extraction gate.
        assert body["status"] == "awaiting_extraction_review"
        assert body["approval_request"]["stage"] == "extraction"

    def test_rejecting_marks_the_run_rejected(self, make_harness) -> None:
        harness = make_harness(UNSURE)
        document_id = upload(harness.client).json()["document_id"]

        body = harness.client.post(
            f"/documents/{document_id}/approval",
            json={"approved": False, "note": "not an invoice"},
        ).json()

        assert body["status"] == "rejected"

    def test_reviewer_can_correct_the_type(self, make_harness) -> None:
        harness = make_harness(UNSURE)
        document_id = upload(harness.client).json()["document_id"]

        body = harness.client.post(
            f"/documents/{document_id}/approval",
            json={"approved": True, "document_type": "receipt"},
        ).json()

        assert body["document_type"] == "receipt"
        # The correction picks the extraction model: receipt, not invoice.
        assert body["extraction_model"] == "prebuilt-receipt"

    def test_unknown_document_returns_404(self, make_harness) -> None:
        harness = make_harness()

        response = harness.client.post(
            "/documents/no-such-document/approval", json={"approved": True}
        )

        assert response.status_code == 404

    def test_invalid_correction_type_is_rejected(self, make_harness) -> None:
        harness = make_harness(UNSURE)
        document_id = upload(harness.client).json()["document_id"]

        response = harness.client.post(
            f"/documents/{document_id}/approval",
            json={"approved": True, "document_type": "spaceship"},
        )

        assert response.status_code == 422


class TestStatus:
    def test_reports_a_run_paused_on_its_data(self, make_harness) -> None:
        harness = make_harness(CONFIDENT)
        document_id = upload(harness.client).json()["document_id"]

        body = harness.client.get(f"/documents/{document_id}").json()

        assert body["status"] == "awaiting_extraction_review"
        assert body["document_type"] == "invoice"
        assert body["approval_request"]["stage"] == "extraction"

    def test_reports_a_completed_run(self, make_harness) -> None:
        harness = make_harness(CONFIDENT)
        document_id = upload(harness.client).json()["document_id"]
        harness.client.post(
            f"/documents/{document_id}/approval", json={"approved": True}
        )

        body = harness.client.get(f"/documents/{document_id}").json()

        assert body["status"] == "completed"
        assert body["document_type"] == "invoice"

    def test_reports_a_paused_run_with_its_approval_request(
        self, make_harness
    ) -> None:
        harness = make_harness(UNSURE)
        document_id = upload(harness.client).json()["document_id"]

        body = harness.client.get(f"/documents/{document_id}").json()

        assert body["status"] == "awaiting_approval"
        assert body["approval_request"]["confidence"] == pytest.approx(0.41)

    def test_unknown_document_returns_404(self, make_harness) -> None:
        harness = make_harness()

        assert harness.client.get("/documents/nope").status_code == 404


class TestDocumentsAreIndependent:
    """Each upload is its own run, keyed on its own document id."""

    def test_uploads_get_distinct_document_ids(self, make_harness) -> None:
        harness = make_harness(CONFIDENT)

        first = upload(harness.client).json()
        second = upload(harness.client, name="other.png").json()

        assert first["document_id"] != second["document_id"]

    def test_blobs_are_grouped_under_the_document(self, make_harness) -> None:
        harness = make_harness(CONFIDENT)

        body = upload(harness.client).json()

        for item in harness.storage.uploads:
            assert item["blob_name"].startswith(f"{body['document_id']}/")

    def test_two_pending_approvals_do_not_interfere(self, make_harness) -> None:
        """Resolving one paused document must leave the other untouched."""
        harness = make_harness(UNSURE)

        first = upload(harness.client).json()
        second = upload(harness.client, name="second.png").json()
        assert first["awaiting_approval"] is True
        assert second["awaiting_approval"] is True

        done = harness.client.post(
            f"/documents/{first['document_id']}/approval", json={"approved": True}
        ).json()
        assert done["status"] == "awaiting_extraction_review"

        other = harness.client.get(f"/documents/{second['document_id']}").json()
        assert other["status"] == "awaiting_approval"

    def test_response_has_no_session_field(self, make_harness) -> None:
        """document_id is the only identifier the API exposes."""
        harness = make_harness(CONFIDENT)

        body = upload(harness.client).json()

        assert "session_id" not in body
        assert "thread_id" not in body
