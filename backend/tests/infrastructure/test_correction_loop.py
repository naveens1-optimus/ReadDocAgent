"""Tests for the validate -> correct -> validate loop.

Nothing is stored until validation passes. A run that needs work stays paused
in the checkpoint, and each correction is merged into the extraction so the
next pass validates the corrected data rather than the original.

Note what approval changes: signing off the extraction marks every field
human-verified at full confidence, so Azure's own score stops being a
blocker. What a person eyeballing values *cannot* reliably catch -- money
that does not add up, a required field that is absent -- still stops the run.
"""

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

#: Everything present, but Azure was not confident about the total. The
#: reviewer sees that score at the review gate and approves anyway.
LOW_CONFIDENCE_TOTAL = [
    ExtractedField(name="VendorName", value="Acme", confidence=0.92),
    ExtractedField(name="InvoiceId", value="INV-1", confidence=0.98),
    ExtractedField(name="InvoiceTotal", value=495.0, confidence=0.44),
]

#: Money that does not add up: subtotal 450 + no tax, but the total says 600.
BAD_ARITHMETIC = [
    ExtractedField(name="VendorName", value="Acme", confidence=0.92),
    ExtractedField(name="InvoiceId", value="INV-1", confidence=0.98),
    ExtractedField(name="InvoiceTotal", value=600.0, confidence=0.95),
    ExtractedField(name="SubTotal", value=450.0, confidence=0.95),
    ExtractedField(name="TotalTax", value=0.0, confidence=0.95),
]

#: A required field the extraction could not read at all.
MISSING_VENDOR = [
    ExtractedField(name="InvoiceId", value="INV-1", confidence=0.98),
    ExtractedField(name="InvoiceTotal", value=495.0, confidence=0.95),
]


@pytest.fixture
def settings(valid_env: dict[str, str]) -> Settings:
    return build_settings(load_dotenv_file=False)


def build(settings: Settings, fields: list[ExtractedField]):
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


def run_to_validation(workflow, run: dict):
    """Upload, then approve the extraction, landing at validation."""
    workflow.graph.invoke(
        DocumentState(
            document_id="doc-1",
            file_name="invoice.png",
            extension=".png",
            file_bytes=PNG,
            input_blob_name="doc-1/invoice.png",
        ),
        run,
    )
    return workflow.graph.invoke(Command(resume={"approved": True}), run)


def config(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


class TestApprovalIsASignOff:
    """Approving the extraction verifies every value it showed."""

    def test_approval_raises_every_field_to_full_confidence(
        self, settings: Settings
    ) -> None:
        workflow, _, _ = build(settings, LOW_CONFIDENCE_TOTAL)

        result = run_to_validation(workflow, config("full"))

        state = DocumentState.model_validate(result)
        assert state.extraction is not None
        total = next(f for f in state.extraction.fields if f.name == "InvoiceTotal")
        assert total.confidence == 1.0
        assert total.verified is True
        # The machine's own score is kept, so the report stays honest about
        # how well the extraction actually did.
        assert total.original_confidence == 0.44

    def test_low_confidence_alone_no_longer_blocks(
        self, settings: Settings
    ) -> None:
        """The reviewer saw the score and approved, so it is not a blocker."""
        workflow, _, entities = build(settings, LOW_CONFIDENCE_TOTAL)

        result = run_to_validation(workflow, config("low"))

        assert "__interrupt__" not in result
        assert DocumentState.model_validate(result).status is (
            ProcessingStatus.COMPLETED
        )
        assert entities.saved["doc-1"]["entity"]["invoice_total"] == "495.0"


class TestNothingIsSavedWhileValidationFails:
    """The regression: a failing check used to be reported, then saved anyway."""

    def test_bad_arithmetic_blocks_the_save(self, settings: Settings) -> None:
        workflow, storage, entities = build(settings, BAD_ARITHMETIC)

        result = run_to_validation(workflow, config("maths"))

        payload = result["__interrupt__"][0].value
        assert payload["stage"] == "data_correction"
        names = [check["name"] for check in payload["failed_checks"]]
        assert "subtotal_plus_tax_matches_total" in names
        assert storage.uploads == [], "must not save while a check is failing"
        assert entities.saved == {}

    def test_a_missing_required_field_blocks_the_save(
        self, settings: Settings
    ) -> None:
        workflow, storage, entities = build(settings, MISSING_VENDOR)

        result = run_to_validation(workflow, config("missing"))

        payload = result["__interrupt__"][0].value
        assert payload["stage"] == "data_correction"
        assert payload["missing_fields"] == ["vendor_name"]
        assert storage.uploads == []
        assert entities.saved == {}

    def test_the_reviewer_is_told_what_failed(self, settings: Settings) -> None:
        workflow, _, _ = build(settings, BAD_ARITHMETIC)

        result = run_to_validation(workflow, config("why"))

        payload = result["__interrupt__"][0].value
        assert any("600" in check["message"] for check in payload["failed_checks"])
        # And the current data, so they can see what to change.
        assert payload["current"]["InvoiceTotal"] == 600.0

    def test_no_entity_is_reported_while_validation_fails(
        self, settings: Settings
    ) -> None:
        """The response must not show an entity that was never stored."""
        from domain.schema.process_document_response import ProcessDocumentResponse

        workflow, _, entities = build(settings, BAD_ARITHMETIC)
        result = run_to_validation(workflow, config("noentity"))

        state = DocumentState.model_validate(result)
        response = ProcessDocumentResponse.from_state(
            state, result["__interrupt__"][0].value
        )

        assert response.entity is None
        assert entities.saved == {}


class TestCorrectionsGoIntoTheCheckpoint:
    """Edits are merged into the extraction, not held alongside it."""

    def test_correction_is_merged_and_revalidated(self, settings: Settings) -> None:
        workflow, _, entities = build(settings, BAD_ARITHMETIC)
        run = config("merge")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 450.0}, "reviewer": "naveen"}),
            run,
        )

        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        # The correction lives in the extraction itself.
        assert state.extraction is not None
        assert state.extraction.to_json()["InvoiceTotal"] == 450.0
        assert "InvoiceTotal" in state.extraction.edited_field_names
        # ...and it is the corrected value that was stored.
        assert entities.saved["doc-1"]["entity"]["invoice_total"] == "450.0"

    def test_untouched_fields_survive_a_correction(
        self, settings: Settings
    ) -> None:
        """Sending only the changed field must not drop the others."""
        workflow, _, entities = build(settings, BAD_ARITHMETIC)
        run = config("keep")
        run_to_validation(workflow, run)

        workflow.graph.invoke(Command(resume={"fields": {"InvoiceTotal": 450.0}}), run)

        stored = entities.saved["doc-1"]["entity"]
        assert stored["vendor_name"] == "Acme"
        assert stored["invoice_id"] == "INV-1"

    def test_supplying_a_missing_field_completes_the_run(
        self, settings: Settings
    ) -> None:
        workflow, _, entities = build(settings, MISSING_VENDOR)
        run = config("supply")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"VendorName": "Acme Ltd"}}), run
        )

        assert "__interrupt__" not in result
        assert DocumentState.model_validate(result).status is (
            ProcessingStatus.COMPLETED
        )
        assert entities.saved["doc-1"]["entity"]["vendor_name"] == "Acme Ltd"

    def test_a_correction_is_also_a_verification(self, settings: Settings) -> None:
        workflow, _, _ = build(settings, BAD_ARITHMETIC)
        run = config("verify")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 450.0}}), run
        )

        state = DocumentState.model_validate(result)
        assert state.extraction is not None
        total = next(f for f in state.extraction.fields if f.name == "InvoiceTotal")
        assert total.confidence == 1.0
        assert total.verified is True
        assert total.edited is True

    def test_resubmitting_an_unchanged_value_still_verifies_it(
        self, settings: Settings
    ) -> None:
        """A flagged value that happens to be correct must be clearable.

        If verification required a change, the only way out would be to alter
        the value to something wrong and back again.
        """
        workflow, _, _ = build(settings, BAD_ARITHMETIC)
        run = config("confirm")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"SubTotal": 450.0}}), run
        )

        state = DocumentState.model_validate(result)
        assert state.extraction is not None
        subtotal = next(f for f in state.extraction.fields if f.name == "SubTotal")
        assert subtotal.verified is True
        assert subtotal.edited is False, "unchanged, so not an edit"


class TestLoopingUntilItPasses:
    def test_a_correction_that_does_not_fix_it_asks_again(
        self, settings: Settings
    ) -> None:
        workflow, storage, _ = build(settings, BAD_ARITHMETIC)
        run = config("again")
        run_to_validation(workflow, run)

        # Still wrong: 500 != 450.
        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 500.0}}), run
        )

        assert result["__interrupt__"][0].value["stage"] == "data_correction"
        assert storage.uploads == []

    def test_it_loops_until_the_data_is_right(self, settings: Settings) -> None:
        workflow, _, entities = build(settings, BAD_ARITHMETIC)
        run = config("loop")
        run_to_validation(workflow, run)

        for wrong in (500.0, 475.0):
            result = workflow.graph.invoke(
                Command(resume={"fields": {"InvoiceTotal": wrong}}), run
            )
            assert result["__interrupt__"][0].value["stage"] == "data_correction"

        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 450.0}}), run
        )

        assert "__interrupt__" not in result
        assert DocumentState.model_validate(result).status is (
            ProcessingStatus.COMPLETED
        )
        assert entities.saved["doc-1"]["entity"]["invoice_total"] == "450.0"

    def test_each_pass_starts_from_the_latest_data(
        self, settings: Settings
    ) -> None:
        """The reviewer sees their previous correction, not the original."""
        workflow, _, _ = build(settings, BAD_ARITHMETIC)
        run = config("latest")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 500.0}}), run
        )

        assert result["__interrupt__"][0].value["current"]["InvoiceTotal"] == 500.0

    def test_audit_trail_records_every_attempt(self, settings: Settings) -> None:
        workflow, _, _ = build(settings, BAD_ARITHMETIC)
        run = config("trail")
        run_to_validation(workflow, run)
        workflow.graph.invoke(Command(resume={"fields": {"InvoiceTotal": 500.0}}), run)
        result = workflow.graph.invoke(
            Command(resume={"fields": {"InvoiceTotal": 450.0}}), run
        )

        agents = [e.agent for e in DocumentState.model_validate(result).audit_trail]
        assert agents == [
            AgentName.CLASSIFIER,
            AgentName.HUMAN_APPROVAL,
            AgentName.EXTRACTOR,
            AgentName.EXTRACTION_REVIEW,
            AgentName.VALIDATOR,        # failed
            AgentName.DATA_CORRECTION,  # first attempt
            AgentName.VALIDATOR,        # still failed
            AgentName.DATA_CORRECTION,  # second attempt
            AgentName.VALIDATOR,        # passed
            AgentName.ENRICHER,
            AgentName.SAVE,
        ]


class TestAbandoning:
    """An escape for data that genuinely cannot be corrected."""

    def test_abandoning_ends_the_run_without_storing_an_entity(
        self, settings: Settings
    ) -> None:
        workflow, storage, entities = build(settings, BAD_ARITHMETIC)
        run = config("abandon")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(resume={"abandon": True, "reviewer": "naveen"}), run
        )

        assert "__interrupt__" not in result
        state = DocumentState.model_validate(result)
        assert state.status is ProcessingStatus.COMPLETED
        # The run is on record...
        assert len(storage.uploads) == 2
        # ...but the data never validated, so no entity was stored.
        assert entities.saved == {}
        assert state.validation is not None
        assert state.validation.passed is False


class TestDatesTypedByAReviewer:
    """A corrected date must not be rejected for its format."""

    @pytest.mark.parametrize(
        ("typed", "expected"),
        [
            ("2003-01-01", "2003-01-01"),
            ("01/01/2003", "2003-01-01"),
            ("1 Jan 2003", "2003-01-01"),
        ],
    )
    def test_common_date_formats_are_accepted(
        self, settings: Settings, typed: str, expected: str
    ) -> None:
        workflow, _, entities = build(settings, MISSING_VENDOR)
        run = config(f"date-{typed}")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(
                resume={
                    "fields": {"VendorName": "Acme", "InvoiceDate": typed}
                }
            ),
            run,
        )

        assert "__interrupt__" not in result
        assert entities.saved["doc-1"]["entity"]["invoice_date"] == expected

    def test_an_unparseable_date_is_reported_not_swallowed(
        self, settings: Settings
    ) -> None:
        workflow, _, _ = build(settings, MISSING_VENDOR)
        run = config("baddate")
        run_to_validation(workflow, run)

        result = workflow.graph.invoke(
            Command(
                resume={"fields": {"VendorName": "Acme", "InvoiceDate": "yesterday"}}
            ),
            run,
        )

        payload = result["__interrupt__"][0].value
        assert any(
            "InvoiceDate" in check["message"] or check["field"] == "invoice_date"
            for check in payload["failed_checks"]
        )
