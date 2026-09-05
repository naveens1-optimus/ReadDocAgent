"""The LangGraph pipeline -- four agents, three human checkpoints.

    classify -> human_approval -> extract -> extraction_review -> validate
                                                                     ^  |
                                        correct_data <--(failed)-----+  |
                                             |                          |
                                             +--------------------------+
                                                                     (passed)
                                                                        |
                                            save <-- enrich <-----------+

Agents:

1. ``classify``            what the document is (Azure OpenAI vision).
2. ``extract``             its fields (the Document Intelligence model for
                           that type).
3. ``validate`` + ``enrich``  checks the data against the entity schema, then
                           standardises and summarises it with Azure OpenAI.
4. ``save``                stores the entity, the output and the report.

Human checkpoints, each gating something different:

* ``human_approval``    confirms *what the document is*. Auto-approves a
                        confident, recognised classification, so a reviewer is
                        only interrupted when the answer is doubtful.
* ``extraction_review`` confirms *the data*. Always pauses -- the reviewer is
                        signing off the values that get saved, and may edit
                        them first.
* ``correct_data``      opens whenever validation fails -- a missing required
                        field, money that does not add up, or a value Azure
                        was not confident about. The reviewer's corrections
                        are merged into the extraction and validated again,
                        looping until they pass. Nothing is stored until they
                        do, so a document that needs work simply stays paused
                        in the checkpoint.

                        A field a human edits is treated as verified, so its
                        original confidence score no longer counts against
                        it -- otherwise a low-confidence field could never be
                        cleared and the loop would never end.

A failure in any node routes to ``error_handler``.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from langchain_azure_cosmosdb import CosmosDBSaverSync
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from application.interface.blob_storage_service import IBlobStorageService
from application.interface.entity_store import IEntityStore
from domain.entity.audit_entry import AuditEntry
from domain.entity.processing_report import ProcessingReport
from domain.enum.document_type import DocumentType
from domain.enum.processing_status import AgentName, ProcessingStatus
from domain.schema.settings import CosmosDbSettings, Settings
from infrastructure.agents.data_extraction_agent import DataExtractionAgent
from infrastructure.agents.validation_agent import ValidationAgent
from infrastructure.agents.document_classification_agent import (
    DocumentClassificationAgent,
)
from infrastructure.utilities.logging_config import get_logger
from infrastructure.workflows.state import (
    ApprovalDecision,
    DocumentState,
    ExtractionReview,
)

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
    try:
        return CosmosDBSaverSync(
            database_name=settings.database_name,
            container_name=settings.container_name,
            endpoint=settings.endpoint,
            key=settings.key.get_secret_value(),
        )
    except Exception as exc:
        # The SDK reports "An unexpected error occurred during CosmosClient
        # initialization", which does not say what to fix. Name the endpoint
        # and the settings involved, and keep the underlying cause.
        cause = exc.__cause__ or exc
        raise RuntimeError(
            f"Could not connect to Cosmos DB at {settings.endpoint} "
            f"(database {settings.database_name!r}, container "
            f"{settings.container_name!r}). Check AZURE_COSMOS_ENDPOINT and "
            f"AZURE_COSMOS_KEY in backend/src/.env. Cause: {cause}"
        ) from exc


class DocumentProcessingWorkflow:
    """Builds and owns the compiled graph."""

    def __init__(
        self,
        classifier: DocumentClassificationAgent,
        extractor: DataExtractionAgent,
        validator: ValidationAgent,
        storage: IBlobStorageService,
        entities: IEntityStore,
        settings: Settings,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._classifier = classifier
        self._extractor = extractor
        self._validator = validator
        self._storage = storage
        self._entities = entities
        self._settings = settings
        self.graph = self._build(checkpointer)

    def _build(self, checkpointer: BaseCheckpointSaver) -> Any:
        builder = StateGraph(DocumentState)

        builder.add_node(AgentName.CLASSIFIER.value, self._classify)
        builder.add_node(AgentName.HUMAN_APPROVAL.value, self._human_approval)
        builder.add_node(AgentName.EXTRACTOR.value, self._extract)
        builder.add_node(AgentName.EXTRACTION_REVIEW.value, self._extraction_review)
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
        builder.add_conditional_edges(
            AgentName.HUMAN_APPROVAL.value,
            self._route_after_approval,
            {
                "extract": AgentName.EXTRACTOR.value,
                "save": AgentName.SAVE.value,
            },
        )
        builder.add_conditional_edges(
            AgentName.EXTRACTOR.value,
            self._route_after_extract,
            {
                "review": AgentName.EXTRACTION_REVIEW.value,
                "error": AgentName.ERROR_HANDLER.value,
            },
        )
        builder.add_node(AgentName.VALIDATOR.value, self._validate)
        builder.add_node(AgentName.DATA_CORRECTION.value, self._correct_data)
        builder.add_node(AgentName.ENRICHER.value, self._enrich)

        builder.add_conditional_edges(
            AgentName.EXTRACTION_REVIEW.value,
            self._route_after_extraction_review,
            {
                "validate": AgentName.VALIDATOR.value,
                "save": AgentName.SAVE.value,
            },
        )
        builder.add_conditional_edges(
            AgentName.VALIDATOR.value,
            self._route_after_validate,
            {
                "correct": AgentName.DATA_CORRECTION.value,
                "enrich": AgentName.ENRICHER.value,
            },
        )
        # Back to validation. The loop is the point: corrections are checked
        # like any other data, and nothing is stored until they pass.
        builder.add_edge(AgentName.DATA_CORRECTION.value, AgentName.VALIDATOR.value)
        builder.add_edge(AgentName.ENRICHER.value, AgentName.SAVE.value)
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
                "stage": "classification",
                "question": "Is this classification correct?",
                "document_id": state.document_id,
                "file_name": state.file_name,
                "document_type": classification.document_type.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "threshold": threshold,
                "options": DocumentType.values(),
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

    def _extract(self, state: DocumentState) -> dict[str, Any]:
        """Extract fields with the prebuilt model for the approved type.

        Re-reads the document from Blob Storage rather than carrying its bytes
        through the graph, so the checkpoint written while waiting for
        approval stays small.
        """
        assert state.classification is not None  # guaranteed by the routing
        document_type = state.classification.document_type

        try:
            data = self._storage.download(
                self._settings.blob_storage.input_container,
                state.input_blob_name or "",
            )
            result = self._extractor.extract(document_type, data)
        except Exception as exc:  # noqa: BLE001 - routed to the error node
            logger.exception("Extraction failed for %s", state.document_id)
            return {
                "error": str(exc),
                "status": ProcessingStatus.FAILED,
                "audit_trail": [
                    AuditEntry.failure(AgentName.EXTRACTOR, "extract", exc)
                ],
            }

        return {
            "extraction": result,
            "status": ProcessingStatus.EXTRACTED,
            "audit_trail": [
                AuditEntry.record(
                    AgentName.EXTRACTOR,
                    "extract",
                    model_id=result.model_id,
                    field_count=len(result.fields),
                )
            ],
        }

    def _extraction_review(self, state: DocumentState) -> dict[str, Any]:
        """Human-in-the-loop checkpoint for the extracted fields.

        Always pauses -- unlike the classification gate there is no confidence
        shortcut, because the reviewer is signing off the data itself. The
        JSON they send back is what gets saved.
        """
        extraction = state.extraction
        assert extraction is not None  # guaranteed by the routing edge

        answer = interrupt(
            {
                "stage": "extraction",
                "document_id": state.document_id,
                "file_name": state.file_name,
                "document_type": (
                    state.classification.document_type.value
                    if state.classification
                    else None
                ),
                "model_id": extraction.model_id,
                # Per-field confidence, so the reviewer can see which values
                # to check most closely.
                "fields": [field.model_dump(mode="json") for field in extraction.fields],
                # The editable JSON.
                "json": extraction.to_json(),
            }
        )

        answer = answer or {}
        edited = answer.get("fields")
        final = (
            extraction.apply_edits(edited)
            if isinstance(edited, dict)
            else extraction
        )
        approved = bool(answer.get("approved", False))

        # Approving is a sign-off on every value shown, so each field becomes
        # human-verified at full confidence. Azure's own scores are kept in
        # original_confidence for the report.
        if approved:
            final = final.mark_verified()

        review = ExtractionReview(
            approved=approved,
            reviewer=answer.get("reviewer"),
            note=answer.get("note"),
            edited_fields=final.edited_field_names,
        )
        logger.info(
            "Extraction %s by %s (%s field(s) edited)",
            "approved" if approved else "rejected",
            review.reviewer or "unknown",
            len(review.edited_fields),
        )
        return {
            "extraction": final,
            "extraction_review": review,
            "status": (
                ProcessingStatus.APPROVED if approved else ProcessingStatus.REJECTED
            ),
            "audit_trail": [
                AuditEntry.record(
                    AgentName.EXTRACTION_REVIEW,
                    "review_extraction",
                    approved=approved,
                    reviewer=review.reviewer,
                    edited_fields=review.edited_fields,
                )
            ],
        }

    def _validate(self, state: DocumentState) -> dict[str, Any]:
        """Check the approved data against the entity schema for its type."""
        assert state.classification is not None
        assert state.extraction is not None

        # The extraction is the single source of truth: corrections were
        # merged back into it, so this is always the current data.
        data = state.extraction.to_json()
        confidences = state.extraction.confidences()

        try:
            result = self._validator.validate(
                document_type=state.classification.document_type,
                data=data,
                confidences=confidences,
                confidence_threshold=self._settings.app.confidence_threshold,
                arithmetic_tolerance=Decimal(
                    str(self._settings.app.arithmetic_tolerance)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - routed to the error node
            logger.exception("Validation failed for %s", state.document_id)
            return {
                "error": str(exc),
                "status": ProcessingStatus.FAILED,
                "audit_trail": [
                    AuditEntry.failure(AgentName.VALIDATOR, "validate", exc)
                ],
            }

        return {
            "validation": result,
            "status": ProcessingStatus.VALIDATED,
            "audit_trail": [
                AuditEntry.record(
                    AgentName.VALIDATOR,
                    "validate",
                    passed=result.passed,
                    missing_fields=result.missing_fields,
                    low_confidence=result.low_confidence_fields,
                )
            ],
        }

    def _correct_data(self, state: DocumentState) -> dict[str, Any]:
        """Show the reviewer what failed validation and take their corrections.

        The corrections are merged **into the extraction**, so the checkpoint
        carries the corrected data forward and the next validation pass reads
        it rather than the original extracted values.
        """
        validation = state.validation
        extraction = state.extraction
        assert validation is not None
        assert extraction is not None

        answer = interrupt(
            {
                "stage": "data_correction",
                "document_id": state.document_id,
                "file_name": state.file_name,
                "document_type": (
                    state.classification.document_type.value
                    if state.classification
                    else None
                ),
                # Why it came back, so the reviewer knows what to fix.
                "failed_checks": [
                    {
                        "name": check.name,
                        "category": check.category,
                        "message": check.message,
                        "field": check.field,
                    }
                    for check in validation.failed_checks
                ],
                "missing_fields": validation.missing_fields,
                "low_confidence_fields": validation.low_confidence_fields,
                # The data as it currently stands, including earlier
                # corrections, so each pass starts from the latest values.
                "fields": [
                    field.model_dump(mode="json") for field in extraction.fields
                ],
                "current": extraction.to_json(),
            }
        )

        answer = answer or {}
        edits = answer.get("fields") or {}
        abandon = bool(answer.get("abandon") or answer.get("skip"))

        corrected = extraction.merge_edits(edits) if edits else extraction
        logger.info(
            "Reviewer corrected %s field(s)%s",
            len(edits),
            " and abandoned the correction" if abandon else "",
        )
        return {
            # Merged back in, so validation re-runs against the corrections.
            "extraction": corrected,
            "abandon_correction": abandon,
            "status": ProcessingStatus.AWAITING_DATA_CORRECTION,
            "audit_trail": [
                AuditEntry.record(
                    AgentName.DATA_CORRECTION,
                    "correct_data",
                    corrected=sorted(edits),
                    reviewer=answer.get("reviewer"),
                    abandoned=abandon,
                )
            ],
        }

    def _enrich(self, state: DocumentState) -> dict[str, Any]:
        """Standardise and summarise the validated data with Azure OpenAI."""
        validation = state.validation
        assert validation is not None
        assert state.classification is not None

        data = validation.entity or (
            state.extraction.to_json() if state.extraction else {}
        )
        summary, enriched, error = self._validator.enrich(
            state.classification.document_type, data
        )

        updated = validation.model_copy(
            update={
                "summary": summary,
                "enriched_fields": enriched,
                "enrichment_error": error,
                "entity": {**data, **enriched} if validation.entity else None,
            }
        )
        return {
            "validation": updated,
            "audit_trail": [
                AuditEntry.record(
                    AgentName.ENRICHER,
                    "enrich",
                    enriched=sorted(enriched),
                    summary_generated=summary is not None,
                    error=error,
                )
            ],
        }

    def _save(self, state: DocumentState) -> dict[str, Any]:
        """Agent 4: store the entity, the output and the processing report.

        Persisting happens here rather than earlier so nothing is written
        until the run has actually been signed off.
        """
        # Both gates must pass. A reviewer who rejects the extracted data
        # rejects the run, even though the classification was approved
        # earlier to get there.
        approved = state.approval is not None and state.approval.approved
        if state.extraction_review is not None:
            approved = approved and state.extraction_review.approved

        entity_id: str | None = None
        validation = state.validation

        # The validated entity goes to Cosmos, keyed on the document id.
        # Only when validation actually passed: a run that reached save by
        # being abandoned still has failing checks, and storing that data
        # would defeat the point of validating it.
        if (
            approved
            and validation is not None
            and validation.entity is not None
            and validation.passed
        ):
            try:
                entity_id = self._entities.save(
                    document_id=state.document_id,
                    document_type=(
                        state.classification.document_type.value
                        if state.classification
                        else "unknown"
                    ),
                    entity=validation.entity,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, run still ends
                logger.exception("Failed to store entity for %s", state.document_id)
                return {
                    "error": str(exc),
                    "status": ProcessingStatus.FAILED,
                    "audit_trail": [
                        AuditEntry.failure(AgentName.SAVE, "store_entity", exc)
                    ],
                }

        report = ProcessingReport.build(
            document_id=state.document_id,
            file_name=state.file_name,
            status=(
                ProcessingStatus.COMPLETED.value
                if approved
                else ProcessingStatus.REJECTED.value
            ),
            classification=state.classification,
            extraction=state.extraction,
            validation=validation,
            confidence_threshold=self._settings.app.confidence_threshold,
            reviewers=_reviewers(state),
            approval_reviewer=(
                state.approval.reviewer if state.approval else None
            ),
            corrected_type=bool(
                state.approval and state.approval.corrected_document_type
            ),
            entity_id=entity_id,
            error=state.error,
        )

        payload = {
            "document_id": state.document_id,
            "file_name": state.file_name,
            "input_blob_url": state.input_blob_url,
            # The final, validated and enriched entity -- what consumers read.
            "entity": validation.entity if validation else None,
            "summary": validation.summary if validation else None,
            # The same fields with their per-field confidence and spatial
            # data, so the output records how the values were arrived at.
            "extraction": (
                state.extraction.model_dump(mode="json") if state.extraction else None
            ),
            "report": report.model_dump(mode="json"),
            "audit_trail": [
                entry.model_dump(mode="json") for entry in state.audit_trail
            ],
        }

        try:
            url = self._storage.upload(
                container=self._settings.blob_storage.output_container,
                blob_name=f"{state.document_id}/result.json",
                data=json.dumps(payload, indent=2, default=str).encode("utf-8"),
                content_type="application/json",
            )
            self._storage.upload(
                container=self._settings.blob_storage.output_container,
                blob_name=f"{state.document_id}/report.json",
                data=json.dumps(
                    report.model_dump(mode="json"), indent=2, default=str
                ).encode("utf-8"),
                content_type="application/json",
            )
        except Exception as exc:  # noqa: BLE001 - recorded, run still ends
            logger.exception("Failed to save result for %s", state.document_id)
            return {
                "error": str(exc),
                "status": ProcessingStatus.FAILED,
                "audit_trail": [AuditEntry.failure(AgentName.SAVE, "save", exc)],
            }

        return {
            "output_blob_url": url,
            "entity_id": entity_id,
            "status": (
                ProcessingStatus.COMPLETED if approved else ProcessingStatus.REJECTED
            ),
            "audit_trail": [
                AuditEntry.record(
                    AgentName.SAVE,
                    "save",
                    output_blob_url=url,
                    entity_id=entity_id,
                )
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

    @staticmethod
    def _route_after_approval(state: DocumentState) -> str:
        """Extract only when an approved type has a model for it.

        A rejected document, or an approved type with no extraction path yet,
        goes straight to save so the outcome is still recorded.
        """
        approved = state.approval is not None and state.approval.approved
        if not approved or state.classification is None:
            return "save"
        return (
            "extract" if state.classification.document_type.is_extractable else "save"
        )

    @staticmethod
    def _route_after_extract(state: DocumentState) -> str:
        """Send failed extractions to the error node."""
        if state.error or state.extraction is None:
            return "error"
        return "review"

    @staticmethod
    def _route_after_extraction_review(state: DocumentState) -> str:
        """Validate approved data; a rejected run goes straight to save."""
        review = state.extraction_review
        if review is None or not review.approved:
            return "save"
        return "validate"

    @staticmethod
    def _route_after_validate(state: DocumentState) -> str:
        """Nothing is stored until validation passes.

        *Any* failing check sends the run back to a human -- a missing
        required field, money that does not add up, or a value Azure was not
        confident about. Previously only missing fields did, so a document
        with a failed arithmetic or confidence check was saved anyway, which
        defeated the point of validating it.

        The reviewer can abandon a document that genuinely cannot be fixed;
        that ends the run without storing an entity.
        """
        validation = state.validation
        if validation is None or state.error:
            return "enrich"
        if not validation.passed and not state.abandon_correction:
            return "correct"
        return "enrich"


def _reviewers(state: DocumentState) -> list[str]:
    """Everyone who took part in reviewing this run, in order, de-duplicated."""
    names = [
        state.approval.reviewer if state.approval else None,
        state.extraction_review.reviewer if state.extraction_review else None,
    ]
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen
