"""Tests for the LangGraph pipeline: classify -> approval -> save."""

from __future__ import annotations

import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from domain.enum.document_type import DocumentType
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
) -> tuple[DocumentProcessingWorkflow, FakeLanguageModel, FakeBlobStorage]:
    """Build a workflow backed by fakes and an in-memory checkpointer."""
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
    return workflow, language_model, storage


def initial_state(document_id: str = "doc-1") -> DocumentState:
    return DocumentState(
        document_id=document_id,
        file_name="invoice.png",
        content_type="image/png",
        extension=".png",
        file_bytes=PNG_BYTES,
        input_blob_url="https://fake/in/invoice.png",
    )


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def approve(workflow, config: dict, **answer) -> dict:
    """Resume whichever checkpoint is currently open."""
    return workflow.graph.invoke(
        Command(resume={"approved": True, **answer}), config=config
    )


class TestConfidentClassification:
    """High confidence auto-approves and runs straight through."""

    def test_auto_approves_then_pauses_on_the_data(
        self, settings: Settings
    ) -> None:
        """Confidence skips the classification gate, never the extraction one."""
        workflow, _, storage = make_workflow(settings)

        result = workflow.graph.invoke(initial_state(), config=config_for("t1"))

        assert result["__interrupt__"][0].value["stage"] == "extraction"
        state = DocumentState.model_validate(result)
        assert state.classification is not None
        assert state.classification.document_type is DocumentType.INVOICE
        assert state.approval is not None
        assert state.approval.auto_approved is True
        assert storage.uploads == [], "must not save before the data is signed off"

    def test_completes_after_the_extraction_is_approved(
        self, settings: Settings
    ) -> None:
        workflow, _, storage = make_workflow(settings)
        config = config_for("t1")
        workflow.graph.invoke(initial_state(), config=config)

        result = approve(workflow, config, reviewer="naveen")

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        assert len(storage.uploads) == 1

    def test_saves_result_json_to_output_container(
        self, settings: Settings
    ) -> None:
        workflow, _, storage = make_workflow(settings)
        config = config_for("t1")
        workflow.graph.invoke(initial_state(), config=config)

        result = approve(workflow, config)

        upload = storage.uploads[0]
        assert upload["container"] == settings.blob_storage.output_container
        assert upload["blob_name"] == "doc-1/result.json"
        assert upload["content_type"] == "application/json"
        assert DocumentState.model_validate(result).output_blob_url is not None

    def test_audit_trail_records_every_node(self, settings: Settings) -> None:
        """The trail is what makes the run auditable after the fact."""
        workflow, _, _ = make_workflow(settings)
        config = config_for("t1")
        workflow.graph.invoke(initial_state(), config=config)

        result = approve(workflow, config)

        agents = [entry.agent for entry in DocumentState.model_validate(result).audit_trail]
        assert agents == [
            AgentName.CLASSIFIER,
            AgentName.HUMAN_APPROVAL,
            AgentName.EXTRACTOR,
            AgentName.EXTRACTION_REVIEW,
            AgentName.SAVE,
        ]

    def test_file_bytes_dropped_after_classification(
        self, settings: Settings
    ) -> None:
        """Keeps later checkpoints small."""
        workflow, _, _ = make_workflow(settings)

        result = workflow.graph.invoke(initial_state(), config=config_for("t1"))

        assert DocumentState.model_validate(result).file_bytes == b""


class TestHumanApprovalCheckpoint:
    """Low confidence pauses the graph until a human answers."""

    @pytest.fixture
    def low_confidence(self) -> FakeLanguageModel:
        return FakeLanguageModel(
            {
                "document_type": "invoice",
                "confidence": 0.42,
                "reasoning": "Layout is unclear.",
            }
        )

    def test_pauses_below_threshold(
        self, settings: Settings, low_confidence: FakeLanguageModel
    ) -> None:
        workflow, _, storage = make_workflow(settings, language_model=low_confidence)

        result = workflow.graph.invoke(initial_state(), config=config_for("t2"))

        assert "__interrupt__" in result
        assert storage.uploads == [], "must not save before approval"

    def test_interrupt_payload_has_what_a_reviewer_needs(
        self, settings: Settings, low_confidence: FakeLanguageModel
    ) -> None:
        workflow, _, _ = make_workflow(settings, language_model=low_confidence)

        result = workflow.graph.invoke(initial_state(), config=config_for("t2"))

        payload = result["__interrupt__"][0].value
        assert payload["document_id"] == "doc-1"
        assert payload["document_type"] == "invoice"
        assert payload["confidence"] == pytest.approx(0.42)
        assert payload["reasoning"] == "Layout is unclear."
        assert payload["threshold"] == pytest.approx(0.80)
        assert "invoice" in payload["options"]

    def test_resume_with_approval_completes_the_run(
        self, settings: Settings, low_confidence: FakeLanguageModel
    ) -> None:
        workflow, _, storage = make_workflow(settings, language_model=low_confidence)
        config = config_for("t2")
        workflow.graph.invoke(initial_state(), config=config)

        result = workflow.graph.invoke(
            Command(resume={"approved": True, "reviewer": "naveen"}), config=config
        )

        # Classification approved; now paused on the extracted data.
        assert result["__interrupt__"][0].value["stage"] == "extraction"
        state = DocumentState.model_validate(result)
        assert state.approval is not None
        assert state.approval.approved is True
        assert state.approval.auto_approved is False
        assert state.approval.reviewer == "naveen"

        result = approve(workflow, config)
        assert DocumentState.model_validate(result).status is (
            ProcessingStatus.COMPLETED
        )
        assert len(storage.uploads) == 1

    def test_resume_with_rejection_marks_run_rejected(
        self, settings: Settings, low_confidence: FakeLanguageModel
    ) -> None:
        workflow, _, storage = make_workflow(settings, language_model=low_confidence)
        config = config_for("t2")
        workflow.graph.invoke(initial_state(), config=config)

        result = workflow.graph.invoke(
            Command(resume={"approved": False, "note": "not an invoice"}),
            config=config,
        )

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.REJECTED
        # Still saved, so the rejection is recorded rather than lost.
        assert len(storage.uploads) == 1

    def test_reviewer_can_correct_the_document_type(
        self, settings: Settings, low_confidence: FakeLanguageModel
    ) -> None:
        workflow, _, _ = make_workflow(settings, language_model=low_confidence)
        config = config_for("t2")
        workflow.graph.invoke(initial_state(), config=config)

        result = workflow.graph.invoke(
            Command(resume={"approved": True, "document_type": "receipt"}),
            config=config,
        )

        state = DocumentState.model_validate(result)
        assert state.classification is not None
        assert state.classification.document_type is DocumentType.RECEIPT
        assert state.classification.confidence == pytest.approx(1.0)

    def test_unsupported_type_always_pauses(self, settings: Settings) -> None:
        """Even a confident 'unsupported' needs a human to look."""
        confident_unsupported = FakeLanguageModel(
            {
                "document_type": "unsupported",
                "confidence": 0.99,
                "reasoning": "A photo of a cat.",
            }
        )
        workflow, _, _ = make_workflow(
            settings, language_model=confident_unsupported
        )

        result = workflow.graph.invoke(initial_state(), config=config_for("t3"))

        assert "__interrupt__" in result

    def test_state_survives_a_new_checkpointer_connection(
        self, settings: Settings, low_confidence: FakeLanguageModel, tmp_path
    ) -> None:
        """Approval arrives on a later request, so the pause must be durable."""
        database = str(tmp_path / "checkpoints.sqlite")
        config = config_for("t4")

        first_analysis = FakeAnalysisService()
        first = DocumentProcessingWorkflow(
            classifier=DocumentClassificationAgent(
                low_confidence, first_analysis
            ),
            extractor=DataExtractionAgent(first_analysis),
            storage=FakeBlobStorage(),
            settings=settings,
            checkpointer=SqliteSaver(
                sqlite3.connect(database, check_same_thread=False)
            ),
        )
        first.graph.invoke(initial_state(), config=config)

        # A separate workflow object, as a later HTTP request would build.
        storage = FakeBlobStorage()
        second_analysis = FakeAnalysisService()
        second = DocumentProcessingWorkflow(
            classifier=DocumentClassificationAgent(
                low_confidence, second_analysis
            ),
            extractor=DataExtractionAgent(second_analysis),
            storage=storage,
            settings=settings,
            checkpointer=SqliteSaver(
                sqlite3.connect(database, check_same_thread=False)
            ),
        )
        # Both gates answered through the second connection.
        second.graph.invoke(Command(resume={"approved": True}), config=config)
        result = second.graph.invoke(
            Command(resume={"approved": True}), config=config
        )

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        assert state.classification is not None
        assert len(storage.uploads) == 1


class TestVisionFallback:
    def test_falls_back_to_text_when_vision_fails(
        self, settings: Settings
    ) -> None:
        language_model = FakeLanguageModel(
            fail_on_image=RuntimeError("vision deployment unavailable")
        )
        analysis = FakeAnalysisService("INVOICE #123 Total: $500")
        workflow, _, _ = make_workflow(
            settings, language_model=language_model, analysis=analysis
        )

        config = config_for("t5")
        workflow.graph.invoke(initial_state(), config=config)
        result = approve(workflow, config)

        state = DocumentState.model_validate(result)
        assert state.classification is not None
        assert state.classification.used_fallback is True
        assert analysis.call_count == 1
        assert state.status is ProcessingStatus.COMPLETED


class TestErrorPath:
    def test_classification_failure_routes_to_error_node(
        self, settings: Settings
    ) -> None:
        """Both paths failing must end in the error node, not an exception."""
        language_model = FakeLanguageModel(
            fail_on_image=RuntimeError("vision down")
        )
        analysis = FakeAnalysisService()
        analysis.extract_text = _raise  # type: ignore[method-assign]

        workflow, _, storage = make_workflow(
            settings, language_model=language_model, analysis=analysis
        )

        result = workflow.graph.invoke(initial_state(), config=config_for("t6"))

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.FAILED
        assert state.error is not None
        assert storage.uploads == []
        assert state.audit_trail[-1].agent is AgentName.ERROR_HANDLER

    def test_save_failure_is_recorded(self, settings: Settings) -> None:
        storage = FakeBlobStorage(fail_with=RuntimeError("blob container gone"))
        workflow, _, _ = make_workflow(settings, storage=storage)

        config = config_for("t7")
        workflow.graph.invoke(initial_state(), config=config)
        result = approve(workflow, config)

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.FAILED
        assert "blob container gone" in (state.error or "")


class TestStreaming:
    def test_graph_can_be_streamed(self, settings: Settings) -> None:
        """The specification asks for invoke() and stream() to both work."""
        workflow, _, _ = make_workflow(settings)

        config = config_for("t8")
        steps = list(workflow.graph.stream(initial_state(), config=config))
        steps += list(
            workflow.graph.stream(Command(resume={"approved": True}), config=config)
        )

        node_names = [name for step in steps for name in step]
        assert AgentName.CLASSIFIER.value in node_names
        assert AgentName.EXTRACTOR.value in node_names
        assert AgentName.EXTRACTION_REVIEW.value in node_names
        assert AgentName.SAVE.value in node_names


class TestSavedPayload:
    def test_saved_json_contains_the_full_record(
        self, settings: Settings
    ) -> None:
        captured: dict[str, bytes] = {}

        class CapturingStorage(FakeBlobStorage):
            def upload(self, container, blob_name, data, content_type=None):  # type: ignore[override]
                captured["data"] = data
                return super().upload(container, blob_name, data, content_type)

        workflow, _, _ = make_workflow(settings, storage=CapturingStorage())
        config = config_for("t9")
        workflow.graph.invoke(initial_state(), config=config)
        approve(workflow, config, reviewer="naveen")

        payload = json.loads(captured["data"])
        assert payload["document_id"] == "doc-1"
        assert payload["classification"]["document_type"] == "invoice"
        assert payload["approval"]["approved"] is True
        # The finalised JSON the reviewer signed off.
        assert payload["data"]["InvoiceId"] == "INV-123"
        # ...alongside the per-field confidence it came with.
        scores = {f["name"]: f["confidence"] for f in payload["extraction"]["fields"]}
        assert scores["InvoiceId"] == 0.97
        assert scores["VendorName"] is None
        assert payload["extraction_review"]["reviewer"] == "naveen"
        # classify, approval, extract, review, save
        assert len(payload["audit_trail"]) == 4


def _raise(_data: bytes) -> str:
    raise RuntimeError("document intelligence unavailable")
