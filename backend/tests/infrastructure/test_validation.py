"""Tests for Agent 3: validation against the entity schema, and enrichment."""

from __future__ import annotations

from decimal import Decimal

import pytest

from domain.entity.registry import entity_for, required_fields_of
from domain.enum.document_type import DocumentType
from infrastructure.agents.validation_agent import ValidationAgent
from tests.fakes import FakeLanguageModel

TOLERANCE = Decimal("0.01")

# A well-formed invoice: 2 line items -> subtotal 450 + tax 45 = total 495.
GOOD_INVOICE = {
    "VendorName": "Acme Supplies Ltd",
    "InvoiceId": "INV-2026-0042",
    "InvoiceTotal": {"amount": 495.0, "currency": "USD"},
    "SubTotal": {"amount": 450.0, "currency": "USD"},
    "TotalTax": {"amount": 45.0, "currency": "USD"},
    "InvoiceDate": "2026-03-09",
    "PaymentTerm": "Net 30",
    "Items": [
        {"Description": "Widget", "Amount": {"amount": 300.0, "currency": "USD"}},
        {"Description": "Install", "Amount": {"amount": 150.0, "currency": "USD"}},
    ],
}


@pytest.fixture
def agent() -> ValidationAgent:
    return ValidationAgent(FakeLanguageModel())


def validate(agent: ValidationAgent, data: dict, confidences: dict | None = None,
             threshold: float = 0.80, document_type: DocumentType = DocumentType.INVOICE):
    return agent.validate(
        document_type=document_type,
        data=data,
        confidences=confidences or {},
        confidence_threshold=threshold,
        arithmetic_tolerance=TOLERANCE,
    )


class TestSchemaIsTheSourceOfTruth:
    """Required fields come from the entity model, not a separate list."""

    def test_invoice_required_fields(self) -> None:
        entity = entity_for(DocumentType.INVOICE)
        assert entity is not None
        assert set(required_fields_of(entity)) == {
            "vendor_name",
            "invoice_id",
            "invoice_total",
        }

    def test_receipt_required_fields(self) -> None:
        entity = entity_for(DocumentType.RECEIPT)
        assert entity is not None
        assert set(required_fields_of(entity)) == {"merchant_name", "total"}

    def test_id_card_has_no_schema(self) -> None:
        assert entity_for(DocumentType.ID_CARD) is None


class TestCompleteness:
    def test_a_good_invoice_validates(self, agent: ValidationAgent) -> None:
        result = validate(agent, GOOD_INVOICE)

        assert result.missing_fields == []
        assert result.is_complete is True
        assert result.entity is not None
        assert result.entity["vendor_name"] == "Acme Supplies Ltd"
        assert result.entity["invoice_total"] == "495.0"

    @pytest.mark.parametrize(
        ("dropped", "expected"),
        [
            ("VendorName", "vendor_name"),
            ("InvoiceId", "invoice_id"),
            ("InvoiceTotal", "invoice_total"),
        ],
    )
    def test_missing_required_field_is_named(
        self, agent: ValidationAgent, dropped: str, expected: str
    ) -> None:
        data = {k: v for k, v in GOOD_INVOICE.items() if k != dropped}

        result = validate(agent, data)

        assert result.missing_fields == [expected]
        assert result.is_complete is False
        assert result.entity is None

    def test_several_missing_fields_are_all_reported(
        self, agent: ValidationAgent
    ) -> None:
        result = validate(agent, {"InvoiceId": "INV-1"})

        assert set(result.missing_fields) == {"vendor_name", "invoice_total"}

    def test_optional_fields_may_be_absent(self, agent: ValidationAgent) -> None:
        minimal = {
            "VendorName": "Acme",
            "InvoiceId": "INV-1",
            "InvoiceTotal": 100.0,
        }

        result = validate(agent, minimal)

        assert result.is_complete is True

    def test_unknown_document_type_reports_no_schema(
        self, agent: ValidationAgent
    ) -> None:
        result = validate(agent, {}, document_type=DocumentType.ID_CARD)

        assert result.passed is False
        assert "No entity schema" in result.failed_checks[0].message


class TestConsistency:
    """The arithmetic cross-check the specification asks for."""

    def test_totals_that_add_up_pass(self, agent: ValidationAgent) -> None:
        result = validate(agent, GOOD_INVOICE)

        consistency = [c for c in result.checks if c.category == "consistency"]
        assert consistency, "arithmetic should have been checked"
        assert all(check.passed for check in consistency)

    def test_line_items_not_matching_subtotal_fails(
        self, agent: ValidationAgent
    ) -> None:
        data = {**GOOD_INVOICE, "SubTotal": {"amount": 400.0, "currency": "USD"}}

        result = validate(agent, data)

        failed = [c.name for c in result.failed_checks]
        assert "line_items_match_subtotal" in failed

    def test_subtotal_plus_tax_not_matching_total_fails(
        self, agent: ValidationAgent
    ) -> None:
        data = {**GOOD_INVOICE, "InvoiceTotal": {"amount": 600.0, "currency": "USD"}}

        result = validate(agent, data)

        failed = [c.name for c in result.failed_checks]
        assert "subtotal_plus_tax_matches_total" in failed

    def test_small_rounding_differences_are_tolerated(
        self, agent: ValidationAgent
    ) -> None:
        """A cent of rounding must not fail a correct invoice."""
        data = {**GOOD_INVOICE, "InvoiceTotal": {"amount": 495.01, "currency": "USD"}}

        result = validate(agent, data)

        assert "subtotal_plus_tax_matches_total" not in [
            c.name for c in result.failed_checks
        ]

    def test_decimal_arithmetic_not_float(self, agent: ValidationAgent) -> None:
        """0.1 + 0.2 must equal 0.3 here, which it would not in float."""
        data = {
            "VendorName": "A",
            "InvoiceId": "1",
            "InvoiceTotal": 0.3,
            "SubTotal": 0.1,
            "TotalTax": 0.2,
        }

        result = validate(agent, data)

        assert "subtotal_plus_tax_matches_total" not in [
            c.name for c in result.failed_checks
        ]

    def test_falls_back_to_line_items_when_no_subtotal(
        self, agent: ValidationAgent
    ) -> None:
        data = {k: v for k, v in GOOD_INVOICE.items() if k != "SubTotal"}

        result = validate(agent, data)

        assert "line_items_match_total" in [c.name for c in result.checks]

    def test_no_arithmetic_check_without_a_total(
        self, agent: ValidationAgent
    ) -> None:
        result = validate(
            agent,
            {"VendorName": "A", "InvoiceId": "1", "InvoiceTotal": 100.0},
        )

        assert [c for c in result.checks if c.category == "consistency"] == []


class TestConfidence:
    def test_low_scoring_fields_are_flagged(self, agent: ValidationAgent) -> None:
        result = validate(
            agent,
            GOOD_INVOICE,
            confidences={"InvoiceTotal": 0.44, "VendorName": 0.97},
            threshold=0.80,
        )

        assert result.low_confidence_fields == ["InvoiceTotal"]
        assert "confidence_threshold" in [c.name for c in result.failed_checks]

    def test_unscored_fields_are_not_flagged(self, agent: ValidationAgent) -> None:
        """Azure reports no score for some fields; that is not a low score."""
        result = validate(
            agent, GOOD_INVOICE, confidences={"Items": None, "VendorName": 0.97}
        )

        assert result.low_confidence_fields == []

    def test_all_confident_passes(self, agent: ValidationAgent) -> None:
        result = validate(
            agent, GOOD_INVOICE, confidences={"VendorName": 0.97, "InvoiceId": 0.99}
        )

        check = next(c for c in result.checks if c.name == "confidence_threshold")
        assert check.passed is True


class TestReceipts:
    def test_a_good_receipt_validates(self, agent: ValidationAgent) -> None:
        result = validate(
            agent,
            {
                "MerchantName": "Corner Shop",
                "Total": {"amount": 12.50, "currency": "GBP"},
                "Subtotal": {"amount": 10.0, "currency": "GBP"},
                "TotalTax": {"amount": 2.50, "currency": "GBP"},
                "TransactionDate": "2026-03-09",
            },
            document_type=DocumentType.RECEIPT,
        )

        assert result.is_complete is True
        assert result.entity is not None
        assert result.entity["currency"] == "GBP"
        assert all(
            c.passed for c in result.checks if c.category == "consistency"
        )

    def test_missing_merchant_is_reported(self, agent: ValidationAgent) -> None:
        result = validate(
            agent, {"Total": 10.0}, document_type=DocumentType.RECEIPT
        )

        assert result.missing_fields == ["merchant_name"]


class TestEnrichment:
    def test_returns_summary_and_changed_fields(self) -> None:
        agent = ValidationAgent(
            FakeLanguageModel(
                enrichment={
                    "summary": "Invoice from Acme for 495.00 USD.",
                    # A list of pairs, because OpenAI's structured output
                    # rejects free-form objects.
                    "enriched": [
                        {"field": "invoice_date", "value": "2026-03-09"},
                        {"field": "currency", "value": "USD"},
                    ],
                }
            )
        )

        summary, enriched, error = agent.enrich(DocumentType.INVOICE, GOOD_INVOICE)

        assert summary == "Invoice from Acme for 495.00 USD."
        assert enriched == {"invoice_date": "2026-03-09", "currency": "USD"}
        assert error is None

    def test_failure_is_non_fatal(self) -> None:
        """Losing a summary must not fail an otherwise good extraction."""
        agent = ValidationAgent(
            FakeLanguageModel(fail_enrichment=RuntimeError("OpenAI unavailable"))
        )

        summary, enriched, error = agent.enrich(DocumentType.INVOICE, GOOD_INVOICE)

        assert summary is None
        assert enriched == {}
        assert error is not None and "OpenAI unavailable" in error


class TestSchemasAreOpenAiCompatible:
    """Guards against a schema OpenAI's structured output will reject.

    A free-form ``dict[str, Any]`` produces a JSON schema without
    ``additionalProperties: false``, which Azure OpenAI rejects with a 400.
    Because enrichment is non-fatal, that failure is silent -- the run
    completes with no summary and nothing in the logs to explain it. These
    tests catch it at build time instead.
    """

    @staticmethod
    def _open_objects(schema: dict, path: str = "") -> list[str]:
        """Paths of object schemas that permit arbitrary extra properties."""
        problems: list[str] = []
        if isinstance(schema, dict):
            if schema.get("type") == "object" and "additionalProperties" not in schema:
                if not schema.get("properties"):
                    problems.append(path or "<root>")
            for key, value in schema.items():
                if isinstance(value, (dict, list)):
                    problems.extend(
                        TestSchemasAreOpenAiCompatible._open_objects(
                            value, f"{path}.{key}" if path else key
                        )
                    )
        elif isinstance(schema, list):
            for index, item in enumerate(schema):
                problems.extend(
                    TestSchemasAreOpenAiCompatible._open_objects(
                        item, f"{path}[{index}]"
                    )
                )
        return problems

    def test_enrichment_schema_has_no_free_form_objects(self) -> None:
        from infrastructure.agents.validation_agent import _Enrichment

        assert self._open_objects(_Enrichment.model_json_schema()) == []

    def test_enriched_is_a_list_of_named_pairs(self) -> None:
        from infrastructure.agents.validation_agent import _Enrichment

        enriched = _Enrichment.model_json_schema()["properties"]["enriched"]
        assert enriched["type"] == "array"

    def test_classifier_schema_has_no_free_form_objects(self) -> None:
        from infrastructure.agents.document_classification_agent import _Verdict

        assert self._open_objects(_Verdict.model_json_schema()) == []
