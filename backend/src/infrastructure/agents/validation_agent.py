"""Agent 3: validates extracted data against the entity schema, then enriches it.

Validation has three parts, matching the specification:

* **completeness** -- every required field of the entity schema is present.
* **consistency**  -- the money adds up (line items -> subtotal + tax -> total).
* **confidence**   -- no field scored below the configured threshold.

Enrichment then uses Azure OpenAI to standardise dates, currency codes and
addresses, infer what it reasonably can, and write a short summary. Enrichment
is deliberately non-fatal: losing a summary is not a reason to fail an
otherwise good extraction.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

from application.interface.language_model_service import ILanguageModelService
from domain.entity.common import EntityModel, to_decimal
from domain.entity.registry import entity_for, required_fields_of
from domain.entity.validation_result import ValidationCheck, ValidationResult
from domain.enum.document_type import DocumentType
from infrastructure.utilities.logging_config import get_logger

__all__ = ["ValidationAgent"]

logger = get_logger(__name__)

ENRICHMENT_PROMPT = """\
You clean up data extracted from a business document.

Return:
- summary: one or two sentences describing the document.
- enriched: a list of {field, value} pairs for ONLY the fields you actually
  changed or inferred. Leave out anything you did not touch.

Rules for `enriched`:
- Dates become ISO 8601 (YYYY-MM-DD).
- Currency becomes an ISO 4217 code (USD, EUR, GBP, INR...).
- Addresses become a single tidy line.
- You may infer a field only when the document's own data clearly supports it
  -- for example a city and state from a postal code. Never invent a value.
- Do not change amounts, identifiers or names.
"""


class _EnrichedField(BaseModel):
    """One value the model normalised or inferred."""

    field: str = Field(description="Name of the field, exactly as given.")
    value: str = Field(description="The cleaned-up value.")


class _Enrichment(BaseModel):
    """What the model is asked to return.

    ``enriched`` is a *list* rather than a dict because OpenAI's structured
    output rejects free-form objects -- it requires ``additionalProperties:
    false``, which a ``dict[str, Any]`` cannot express. Asking for a list of
    named pairs keeps the schema valid, and the agent turns it back into a
    mapping.
    """

    summary: str = Field(description="One or two sentences.")
    enriched: list[_EnrichedField] = Field(
        default_factory=list, description="Only the fields changed or inferred."
    )


class ValidationAgent:
    """Validates an extraction against its schema and enriches the result."""

    def __init__(self, language_model: ILanguageModelService) -> None:
        self._language_model = language_model

    @traceable(name="validation", run_type="chain")
    def validate(
        self,
        document_type: DocumentType,
        data: dict[str, Any],
        confidences: dict[str, float | None],
        confidence_threshold: float,
        arithmetic_tolerance: Decimal,
    ) -> ValidationResult:
        """Check the extracted data against the entity schema.

        Args:
            document_type: Decides which schema applies.
            data: The reviewer-approved ``{field: value}`` JSON.
            confidences: Per-field Document Intelligence scores.
            confidence_threshold: Fields below this are flagged.
            arithmetic_tolerance: Allowed slack when comparing money totals.
        """
        result = ValidationResult()
        entity_type = entity_for(document_type)

        if entity_type is None:
            result.checks.append(
                ValidationCheck(
                    name="schema_known",
                    category="completeness",
                    passed=False,
                    message=f"No entity schema for {document_type.value!r}.",
                )
            )
            return result

        entity = self._build_entity(entity_type, data, result)
        if entity is not None:
            result.entity = entity.model_dump(mode="json")
            self._check_arithmetic(entity, result, arithmetic_tolerance)

        self._check_confidence(confidences, confidence_threshold, result)
        logger.info("Validation: %s", result.summary_line())
        return result

    # ------------------------------------------------------------------
    # Completeness
    # ------------------------------------------------------------------

    def _build_entity(
        self,
        entity_type: type[EntityModel],
        data: dict[str, Any],
        result: ValidationResult,
    ) -> EntityModel | None:
        """Validate the data against the schema.

        Pydantic *is* the completeness check: constructing the entity either
        succeeds, or reports exactly which required fields are missing.
        """
        try:
            entity = entity_type.model_validate(data)
        except ValidationError as exc:
            result.missing_fields = _missing_from(exc, entity_type)
            for field in result.missing_fields:
                result.checks.append(
                    ValidationCheck(
                        name="required_field",
                        category="completeness",
                        passed=False,
                        message=f"Required field {field!r} is missing.",
                        field=field,
                    )
                )
            # Errors that are not simply absent fields still matter.
            for problem in _other_problems(exc, result.missing_fields):
                result.checks.append(
                    ValidationCheck(
                        name="field_valid",
                        category="completeness",
                        passed=False,
                        message=problem,
                    )
                )
            return None

        result.checks.append(
            ValidationCheck(
                name="required_fields",
                category="completeness",
                passed=True,
                message=(
                    f"All {len(required_fields_of(entity_type))} required field(s) "
                    f"present."
                ),
            )
        )
        return entity

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def _check_arithmetic(
        self, entity: EntityModel, result: ValidationResult, tolerance: Decimal
    ) -> None:
        """Cross-check the money, where the entity has money to check."""
        total = _decimal_attr(entity, "invoice_total") or _decimal_attr(entity, "total")
        if total is None:
            return

        subtotal = _decimal_attr(entity, "subtotal")
        tax = _decimal_attr(entity, "total_tax") or Decimal("0")
        line_total = _decimal_attr(entity, "line_item_total") or _decimal_attr(
            entity, "item_total"
        )

        if line_total is not None and subtotal is not None:
            _compare(
                result,
                name="line_items_match_subtotal",
                expected=subtotal,
                actual=line_total,
                tolerance=tolerance,
                label="line items",
                against="subtotal",
            )

        if subtotal is not None:
            _compare(
                result,
                name="subtotal_plus_tax_matches_total",
                expected=total,
                actual=subtotal + tax,
                tolerance=tolerance,
                label="subtotal + tax",
                against="total",
            )
        elif line_total is not None:
            _compare(
                result,
                name="line_items_match_total",
                expected=total,
                actual=line_total + tax,
                tolerance=tolerance,
                label="line items + tax",
                against="total",
            )

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _check_confidence(
        confidences: dict[str, float | None],
        threshold: float,
        result: ValidationResult,
    ) -> None:
        """Flag fields Document Intelligence was not confident about.

        A field with no score is not flagged: Azure reports none for some
        field types, and treating that as low would flag every document.
        """
        low = sorted(
            name
            for name, score in confidences.items()
            if score is not None and score < threshold
        )
        result.low_confidence_fields = low
        result.checks.append(
            ValidationCheck(
                name="confidence_threshold",
                category="confidence",
                passed=not low,
                message=(
                    f"{len(low)} field(s) below {threshold:.2f}: {', '.join(low)}"
                    if low
                    else f"All scored fields at or above {threshold:.2f}."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    @traceable(name="enrichment", run_type="chain")
    def enrich(
        self, document_type: DocumentType, data: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any], str | None]:
        """Standardise and summarise via Azure OpenAI.

        Returns:
            ``(summary, enriched_fields, error)``. On failure the first two
            are empty and ``error`` explains why -- the caller carries on with
            un-enriched data rather than failing the run.
        """
        try:
            answer = self._language_model.structured_completion(
                system_prompt=ENRICHMENT_PROMPT,
                user_prompt=(
                    f"Document type: {document_type.value}\n"
                    f"Extracted data:\n{json.dumps(data, indent=2, default=str)}"
                ),
                schema=_Enrichment,
            )
        except Exception as exc:  # noqa: BLE001 - enrichment is non-fatal
            logger.warning("Enrichment failed: %s", exc)
            return None, {}, str(exc)

        enriched = {item.field: item.value for item in answer.enriched}
        logger.info("Enrichment changed %s field(s)", len(enriched))
        return answer.summary, enriched, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _missing_from(exc: ValidationError, entity_type: type[EntityModel]) -> list[str]:
    """Required field names Pydantic reported as absent."""
    required = set(required_fields_of(entity_type))
    missing = [
        str(error["loc"][0])
        for error in exc.errors()
        if error["type"] == "missing" and error["loc"]
    ]
    # An alias miss reports the alias; map it back to the field name.
    by_alias = {
        (field.alias or name): name
        for name, field in entity_type.model_fields.items()
    }
    resolved = [by_alias.get(name, name) for name in missing]
    return sorted({name for name in resolved if name in required})


def _other_problems(exc: ValidationError, already_reported: list[str]) -> list[str]:
    """Validation errors that are not simply a missing field."""
    return [
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
        if error["type"] != "missing"
        and not (error["loc"] and str(error["loc"][0]) in already_reported)
    ]


def _decimal_attr(entity: EntityModel, name: str) -> Decimal | None:
    """Read a Decimal attribute if the entity has one."""
    return to_decimal(getattr(entity, name, None))


def _compare(
    result: ValidationResult,
    *,
    name: str,
    expected: Decimal,
    actual: Decimal,
    tolerance: Decimal,
    label: str,
    against: str,
) -> None:
    """Record whether two money figures agree within tolerance."""
    difference = abs(expected - actual)
    passed = difference <= tolerance
    result.checks.append(
        ValidationCheck(
            name=name,
            category="consistency",
            passed=passed,
            message=(
                f"{label} ({actual}) matches {against} ({expected})."
                if passed
                else f"{label} is {actual} but {against} is {expected}, "
                f"a difference of {difference}."
            ),
        )
    )
