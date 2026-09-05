"""Tests for the extraction step and its always-on review gate."""

from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from domain.enum.processing_status import AgentName, ProcessingStatus
from domain.schema.settings import Settings
from infrastructure.agents.data_extraction_agent import DataExtractionAgent
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.config.settings import build_settings
from infrastructure.workflows.document_processing_workflow import (
    DocumentProcessingWorkflow,
)
from infrastructure.workflows.state import DocumentState
from tests.fakes import FakeAnalysisService, FakeBlobStorage, FakeLanguageModel

PNG_BYTES = b"\x89PNG\r\n\x1a\n fake image bytes"


@pytest.fixture
def settings(valid_env: dict[str, str]) -> Settings:
    return build_settings(load_dotenv_file=False)


def make_workflow(
    settings: Settings,
    language_model: FakeLanguageModel | None = None,
    storage: FakeBlobStorage | None = None,
    analysis: FakeAnalysisService | None = None,
) -> tuple[DocumentProcessingWorkflow, FakeBlobStorage, FakeAnalysisService]:
    language_model = language_model or FakeLanguageModel()
    storage = storage or FakeBlobStorage()
    analysis = analysis or FakeAnalysisService()

    workflow = DocumentProcessingWorkflow(
        classifier=DocumentClassificationAgent(language_model, analysis),
        extractor=DataExtractionAgent(analysis),
        storage=storage,
        settings=settings,
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
    )
    return workflow, storage, analysis


def state(**overrides) -> DocumentState:
    base = DocumentState(
        document_id="doc-1",
        file_name="invoice.png",
        content_type="image/png",
        extension=".png",
        file_bytes=PNG_BYTES,
        input_blob_url="https://fake/in/invoice.png",
        input_blob_name="doc-1/invoice.png",
    )
    return base.model_copy(update=overrides)


def config(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def model_saying(document_type: str, confidence: float = 0.95) -> FakeLanguageModel:
    return FakeLanguageModel(
        {
            "document_type": document_type,
            "confidence": confidence,
            "reasoning": "Test.",
        }
    )


class TestModelRouting:
    """Each document type is extracted with its own prebuilt model."""

    @pytest.mark.parametrize(
        ("document_type", "expected_model"),
        [("invoice", "prebuilt-invoice"), ("receipt", "prebuilt-receipt")],
    )
    def test_routes_to_the_matching_model(
        self, settings: Settings, document_type: str, expected_model: str
    ) -> None:
        workflow, _, analysis = make_workflow(
            settings, language_model=model_saying(document_type)
        )

        workflow.graph.invoke(state(), config(f"r-{document_type}"))

        assert analysis.extract_calls == [expected_model]

    @pytest.mark.parametrize("document_type", ["contract", "resume", "id_card"])
    def test_unsupported_types_skip_extraction(
        self, settings: Settings, document_type: str
    ) -> None:
        """No model yet, so the run completes without extracting."""
        workflow, storage, analysis = make_workflow(
            settings, language_model=model_saying(document_type)
        )

        result = workflow.graph.invoke(state(), config(f"s-{document_type}"))

        assert "__interrupt__" not in result
        final = DocumentState.model_validate(result)
        assert final.status is ProcessingStatus.COMPLETED
        assert final.extraction is None
        assert analysis.extract_calls == []
        assert len(storage.uploads) == 1

    def test_reads_the_document_back_from_storage(self, settings: Settings) -> None:
        """The bytes are cleared after classify, so extract re-reads them."""
        workflow, storage, _ = make_workflow(settings)

        workflow.graph.invoke(state(), config("download"))

        assert storage.downloads == ["doc-1/invoice.png"]

    def test_a_correction_picks_the_corrected_model(
        self, settings: Settings
    ) -> None:
        """Approving with a corrected type must extract as that type."""
        workflow, _, analysis = make_workflow(
            settings, language_model=model_saying("invoice", confidence=0.2)
        )
        run = config("corrected")
        workflow.graph.invoke(state(), run)

        workflow.graph.invoke(
            Command(resume={"approved": True, "document_type": "receipt"}), run
        )

        assert analysis.extract_calls == ["prebuilt-receipt"]


class TestFieldConfidence:
    """Every field carries its own score, where Azure reports one."""

    def test_each_field_keeps_its_own_confidence(self, settings: Settings) -> None:
        workflow, _, _ = make_workflow(settings)

        result = workflow.graph.invoke(state(), config("conf"))

        fields = result["__interrupt__"][0].value["fields"]
        assert {f["name"]: f["confidence"] for f in fields} == {
            "InvoiceId": 0.97,
            "InvoiceTotal": 0.62,
            # None, not 0 -- Document Intelligence reports no score for this
            # field, which is different from reporting a low one.
            "VendorName": None,
        }

    def test_interrupt_carries_the_editable_json(self, settings: Settings) -> None:
        workflow, _, _ = make_workflow(settings)

        result = workflow.graph.invoke(state(), config("json"))

        payload = result["__interrupt__"][0].value
        assert payload["stage"] == "extraction"
        assert payload["model_id"] == "prebuilt-invoice"
        assert payload["json"] == {
            "InvoiceId": "INV-123",
            "InvoiceTotal": 500.0,
            "VendorName": "Acme",
        }


class TestReviewGate:
    """Extraction always pauses, whatever the classifier thought."""

    def test_pauses_even_at_full_confidence(self, settings: Settings) -> None:
        workflow, storage, _ = make_workflow(
            settings, language_model=model_saying("invoice", confidence=1.0)
        )

        result = workflow.graph.invoke(state(), config("always"))

        assert result["__interrupt__"][0].value["stage"] == "extraction"
        assert storage.uploads == [], "nothing saved before sign-off"

    def test_rejected_classification_never_extracts(
        self, settings: Settings
    ) -> None:
        workflow, _, analysis = make_workflow(
            settings, language_model=model_saying("invoice", confidence=0.2)
        )
        run = config("rejected")
        workflow.graph.invoke(state(), run)

        workflow.graph.invoke(Command(resume={"approved": False}), run)

        assert analysis.extract_calls == []

    def test_extraction_failure_routes_to_the_error_node(
        self, settings: Settings
    ) -> None:
        workflow, storage, _ = make_workflow(
            settings,
            analysis=FakeAnalysisService(fail_extract=RuntimeError("DI unavailable")),
        )

        result = workflow.graph.invoke(state(), config("boom"))

        final = DocumentState.model_validate(result)
        assert final.status is ProcessingStatus.FAILED
        assert "DI unavailable" in (final.error or "")
        assert final.audit_trail[-1].agent is AgentName.ERROR_HANDLER
        assert storage.uploads == []


class TestReviewerEdits:
    """Whatever JSON the reviewer sends back is what gets saved."""

    @staticmethod
    def _run_to_review(settings: Settings):
        workflow, storage, _ = make_workflow(settings)
        run = config("edit")
        workflow.graph.invoke(state(), run)
        return workflow, storage, run

    def test_edited_values_are_kept_and_marked(self, settings: Settings) -> None:
        workflow, _, run = self._run_to_review(settings)

        result = workflow.graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "reviewer": "naveen",
                    "fields": {
                        "InvoiceId": "INV-123",
                        "InvoiceTotal": 495.0,      # corrected
                        "VendorName": "Acme Ltd",   # corrected
                    },
                }
            ),
            run,
        )

        final = DocumentState.model_validate(result)
        assert final.extraction is not None
        assert final.extraction.to_json()["InvoiceTotal"] == 495.0
        assert set(final.extraction.edited_field_names) == {
            "InvoiceTotal",
            "VendorName",
        }
        assert final.extraction_review is not None
        assert final.extraction_review.reviewer == "naveen"
        assert final.status is ProcessingStatus.COMPLETED

    def test_a_corrected_field_keeps_its_original_confidence(
        self, settings: Settings
    ) -> None:
        """The score describes what Azure read, not what the human typed."""
        workflow, _, run = self._run_to_review(settings)

        result = workflow.graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "fields": {
                        "InvoiceId": "INV-123",
                        "InvoiceTotal": 495.0,
                        "VendorName": "Acme",
                    },
                }
            ),
            run,
        )

        final = DocumentState.model_validate(result)
        assert final.extraction is not None
        total = next(
            f for f in final.extraction.fields if f.name == "InvoiceTotal"
        )
        assert total.confidence == 0.62
        assert total.edited is True

    def test_unchanged_fields_are_not_marked_edited(
        self, settings: Settings
    ) -> None:
        workflow, _, run = self._run_to_review(settings)

        result = workflow.graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "fields": {"InvoiceId": "INV-123", "InvoiceTotal": 500.0},
                }
            ),
            run,
        )

        final = DocumentState.model_validate(result)
        assert final.extraction is not None
        assert final.extraction.edited_field_names == []
        # VendorName was dropped by the reviewer, so it is gone.
        assert "VendorName" not in final.extraction.to_json()

    def test_reviewer_can_add_a_field(self, settings: Settings) -> None:
        workflow, _, run = self._run_to_review(settings)

        result = workflow.graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "fields": {
                        "InvoiceId": "INV-123",
                        "InvoiceTotal": 500.0,
                        "VendorName": "Acme",
                        "PurchaseOrder": "PO-9",
                    },
                }
            ),
            run,
        )

        final = DocumentState.model_validate(result)
        assert final.extraction is not None
        added = next(
            f for f in final.extraction.fields if f.name == "PurchaseOrder"
        )
        assert added.value == "PO-9"
        assert added.edited is True
        # Nothing extracted it, so it carries no confidence.
        assert added.confidence is None

    def test_approving_without_edits_saves_what_was_extracted(
        self, settings: Settings
    ) -> None:
        workflow, _, run = self._run_to_review(settings)

        result = workflow.graph.invoke(Command(resume={"approved": True}), run)

        final = DocumentState.model_validate(result)
        assert final.extraction is not None
        assert final.extraction.to_json() == {
            "InvoiceId": "INV-123",
            "InvoiceTotal": 500.0,
            "VendorName": "Acme",
        }
        assert final.extraction.edited_field_names == []

    def test_rejecting_the_data_marks_the_run_rejected(
        self, settings: Settings
    ) -> None:
        workflow, storage, run = self._run_to_review(settings)

        result = workflow.graph.invoke(
            Command(resume={"approved": False, "note": "garbled scan"}), run
        )

        final = DocumentState.model_validate(result)
        assert final.status is ProcessingStatus.REJECTED
        # Still saved, so the rejection is recorded rather than lost.
        assert len(storage.uploads) == 1
