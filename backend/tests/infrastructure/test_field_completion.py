"""Tests for the missing-required-field checkpoint and the full pipeline."""

from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from domain.entity.extraction_result import ExtractedField
from domain.enum.processing_status import AgentName, ProcessingStatus
from domain.schema.settings import Settings
from infrastructure.agents.data_extraction_agent import DataExtractionAgent
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.agents.validation_agent import ValidationAgent
from infrastructure.config.settings import build_settings
from infrastructure.workflows.document_processing_workflow import (
    DocumentProcessingWorkflow,
)
from infrastructure.workflows.state import DocumentState
from tests.fakes import (
    FakeAnalysisService,
    FakeBlobStorage,
    FakeEntityStore,
    FakeLanguageModel,
)

PNG = b"\x89PNG\r\n\x1a\n fake"

#: A complete invoice extraction: all three required fields present.
COMPLETE = [
    ExtractedField(name="VendorName", value="Acme", confidence=0.92),
    ExtractedField(name="InvoiceId", value="INV-1", confidence=0.98),
    ExtractedField(name="InvoiceTotal", value=495.0, confidence=0.44),
]

#: Same document with the vendor unreadable -- the case that needs a human.
MISSING_VENDOR = [
    ExtractedField(name="InvoiceId", value="INV-1", confidence=0.98),
    ExtractedField(name="InvoiceTotal", value=495.0, confidence=0.44),
]


@pytest.fixture
def settings(valid_env: dict[str, str]) -> Settings:
    return build_settings(load_dotenv_file=False)


def build(settings: Settings, fields: list[ExtractedField]):
    """A workflow whose extraction returns ``fields``."""
    language_model = FakeLanguageModel()
    analysis = FakeAnalysisService(fields=fields)
    storage = FakeBlobStorage()
    entities = FakeEntityStore()

    workflow = DocumentProcessingWorkflow(
        classifier=DocumentClassificationAgent(language_model, analysis),
        extractor=DataExtractionAgent(analysis, language_model),
        validator=ValidationAgent(language_model),
        storage=storage,
        entities=entities,
        settings=settings,
        checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)),
    )
    return workflow, storage, entities


def start(workflow, run: dict):
    """Upload through to the extraction review pause."""
    return workflow.graph.invoke(
        DocumentState(
            document_id="doc-1",
            file_name="invoice.png",
            extension=".png",
            file_bytes=PNG,
            input_blob_name="doc-1/invoice.png",
        ),
        run,
    )


def config(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


class TestCompleteDataFlowsStraightThrough:
    def test_no_completion_gate_when_nothing_is_missing(
        self, settings: Settings
    ) -> None:
        workflow, _, entities = build(settings, COMPLETE)
        run = config("ok")
        start(workflow, run)

        result = workflow.graph.invoke(Command(resume={"approved": True}), run)

        assert "__interrupt__" not in result
        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        assert state.validation is not None
        assert state.validation.is_complete is True
        assert entities.saved["doc-1"]["entity"]["vendor_name"] == "Acme"

    def test_entity_is_stored_against_the_document_id(
        self, settings: Settings
    ) -> None:
        workflow, _, entities = build(settings, COMPLETE)
        run = config("store")
        start(workflow, run)

        workflow.graph.invoke(Command(resume={"approved": True}), run)

        stored = entities.saved["doc-1"]
        assert stored["document_type"] == "invoice"
        assert stored["entity"]["invoice_id"] == "INV-1"


class TestMissingFieldCheckpoint:
    """A required field the extraction could not read pauses for a human."""

    def test_pauses_and_names_the_missing_field(self, settings: Settings) -> None:
        workflow, storage, _ = build(settings, MISSING_VENDOR)
        run = config("missing")
        start(workflow, run)

        result = workflow.graph.invoke(Command(resume={"approved": True}), run)

        payload = result["__interrupt__"][0].value
        assert payload["stage"] == "field_completion"
        assert payload["missing_fields"] == ["vendor_name"]
        # Nothing stored while the data is still incomplete.
        assert storage.uploads == []

    def test_interrupt_carries_context_for_the_reviewer(
        self, settings: Settings
    ) -> None:
        workflow, _, _ = build(settings, MISSING_VENDOR)
        run = config("context")
        start(workflow, run)

        result = workflow.graph.invoke(Command(resume={"approved": True}), run)

        payload = result["__interrupt__"][0].value
        assert payload["document_type"] == "invoice"
        # The data as it stands, so the reviewer is not working blind.
        assert payload["current"]["InvoiceId"] == "INV-1"
        # Low-confidence fields are surfaced at the same time.
        assert "InvoiceTotal" in payload["low_confidence_fields"]

    def test_supplying_the_value_completes_the_run(
        self, settings: Settings
    ) -> None:
        workflow, _, entities = build(settings, MISSING_VENDOR)
        run = config("supply")
        start(workflow, run)
        workflow.graph.invoke(Command(resume={"approved": True}), run)

        result = workflow.graph.invoke(
            Command(
                resume={"fields": {"VendorName": "Acme Ltd"}, "reviewer": "naveen"}
            ),
            run,
        )

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        assert state.validation is not None
        assert state.validation.is_complete is True
        assert entities.saved["doc-1"]["entity"]["vendor_name"] == "Acme Ltd"

    def test_supplied_values_are_revalidated_not_trusted(
        self, settings: Settings
    ) -> None:
        """The loop goes back through validation, so a still-empty answer
        is caught rather than waved through."""
        workflow, _, _ = build(settings, MISSING_VENDOR)
        run = config("revalidate")
        start(workflow, run)
        workflow.graph.invoke(Command(resume={"approved": True}), run)

        # A reviewer who supplies nothing useful is asked again.
        result = workflow.graph.invoke(Command(resume={"fields": {}}), run)

        assert result["__interrupt__"][0].value["stage"] == "field_completion"

    def test_reviewer_can_choose_to_continue_regardless(
        self, settings: Settings
    ) -> None:
        """Without this the graph would ask, be refused, and ask forever."""
        workflow, storage, entities = build(settings, MISSING_VENDOR)
        run = config("skip")
        start(workflow, run)
        workflow.graph.invoke(Command(resume={"approved": True}), run)

        result = workflow.graph.invoke(
            Command(resume={"skip": True, "reviewer": "naveen"}), run
        )

        assert "__interrupt__" not in result
        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        # The output is still written...
        assert len(storage.uploads) == 2
        # ...but no entity, because it never validated.
        assert entities.saved == {}

    def test_audit_trail_records_the_whole_pipeline(
        self, settings: Settings
    ) -> None:
        workflow, _, _ = build(settings, MISSING_VENDOR)
        run = config("audit")
        start(workflow, run)
        workflow.graph.invoke(Command(resume={"approved": True}), run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"VendorName": "Acme Ltd"}}), run
        )

        agents = [e.agent for e in DocumentState.model_validate(result).audit_trail]
        assert agents == [
            AgentName.CLASSIFIER,
            AgentName.HUMAN_APPROVAL,
            AgentName.EXTRACTOR,
            AgentName.EXTRACTION_REVIEW,
            AgentName.VALIDATOR,        # found the gap
            AgentName.FIELD_COMPLETION, # asked a human
            AgentName.VALIDATOR,        # checked the answer
            AgentName.ENRICHER,
            AgentName.SAVE,
        ]


class TestReportSections:
    """The processing report carries the four sections the spec requires."""

    def test_report_has_all_four_sections(self, settings: Settings) -> None:
        import json

        captured: dict[str, bytes] = {}

        class Capturing(FakeBlobStorage):
            def upload(self, container, blob_name, data, content_type=None):  # type: ignore[override]
                captured[blob_name] = data
                return super().upload(container, blob_name, data, content_type)

        language_model = FakeLanguageModel()
        analysis = FakeAnalysisService(fields=COMPLETE)
        workflow = DocumentProcessingWorkflow(
            classifier=DocumentClassificationAgent(language_model, analysis),
            extractor=DataExtractionAgent(analysis, language_model),
            validator=ValidationAgent(language_model),
            storage=Capturing(),
            entities=FakeEntityStore(),
            settings=settings,
            checkpointer=SqliteSaver(
                sqlite3.connect(":memory:", check_same_thread=False)
            ),
        )
        run = config("report")
        start(workflow, run)
        workflow.graph.invoke(
            Command(resume={"approved": True, "reviewer": "naveen"}), run
        )

        report = json.loads(captured["doc-1/report.json"])

        # 1: classification result and confidence
        assert report["classification"]["document_type"] == "invoice"
        assert report["classification"]["confidence"] == pytest.approx(0.94)

        # 2: extraction summary with field-level confidence
        assert report["extraction"]["model_id"] == "prebuilt-invoice"
        assert report["extraction"]["field_count"] == 3
        scores = {f["name"]: f["confidence"] for f in report["extraction"]["fields"]}
        assert scores["InvoiceTotal"] == pytest.approx(0.44)
        # Weakest first, so a reviewer reads the doubtful values at the top.
        assert report["extraction"]["fields"][0]["name"] == "InvoiceTotal"

        # 3: validation results, passed and failed
        assert report["validation"]["total_checks"] > 0
        assert isinstance(report["validation"]["passed"], bool)

        # 4: anything flagged for review
        assert report["review"]["low_confidence_fields"] == ["InvoiceTotal"]
        assert report["entity_id"] == "doc-1"
