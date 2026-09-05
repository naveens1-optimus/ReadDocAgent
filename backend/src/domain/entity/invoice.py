"""The canonical invoice entity.

This is the schema extracted data is validated against. Fields declared
without a default are **required**: if the extraction cannot supply one, the
validation agent pauses and asks a human to fill it in.

Aliases are Azure's ``prebuilt-invoice`` field names, so an extraction maps
straight onto the entity.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import Field, computed_field, field_validator, model_validator

from domain.entity.common import EntityModel, currency_of, to_decimal

__all__ = ["LineItem", "Invoice"]


class LineItem(EntityModel):
    """One line on an invoice. Every part is optional -- real invoices vary."""

    description: str | None = Field(default=None, alias="Description")
    quantity: Decimal | None = Field(default=None, alias="Quantity")
    unit_price: Decimal | None = Field(default=None, alias="UnitPrice")
    amount: Decimal | None = Field(default=None, alias="Amount")
    product_code: str | None = Field(default=None, alias="ProductCode")

    _money = field_validator(
        "quantity", "unit_price", "amount", mode="before"
    )(to_decimal)


class Invoice(EntityModel):
    """An invoice, as the pipeline stores it."""

    # --- Required ---
    vendor_name: str = Field(alias="VendorName")
    invoice_id: str = Field(alias="InvoiceId")
    invoice_total: Decimal = Field(alias="InvoiceTotal")

    # --- Optional ---
    customer_name: str | None = Field(default=None, alias="CustomerName")
    invoice_date: date | None = Field(default=None, alias="InvoiceDate")
    due_date: date | None = Field(default=None, alias="DueDate")
    purchase_order: str | None = Field(default=None, alias="PurchaseOrder")
    subtotal: Decimal | None = Field(default=None, alias="SubTotal")
    total_tax: Decimal | None = Field(default=None, alias="TotalTax")
    amount_due: Decimal | None = Field(default=None, alias="AmountDue")
    payment_terms: str | None = Field(default=None, alias="PaymentTerm")
    vendor_address: str | None = Field(default=None, alias="VendorAddress")
    customer_address: str | None = Field(default=None, alias="CustomerAddress")
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list, alias="Items")

    _money = field_validator(
        "invoice_total", "subtotal", "total_tax", "amount_due", mode="before"
    )(to_decimal)

    @model_validator(mode="before")
    @classmethod
    def _infer_currency(cls, data: Any) -> Any:
        """Take the currency from whichever money field carries one.

        Azure reports the code on individual money fields rather than on the
        document, so without this the invoice would have no currency at all.
        """
        if not isinstance(data, dict) or data.get("currency"):
            return data
        for key in ("InvoiceTotal", "SubTotal", "AmountDue", "TotalTax"):
            code = currency_of(data.get(key))
            if code:
                data = {**data, "currency": code}
                break
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def line_item_total(self) -> Decimal | None:
        """Sum of the line amounts, or ``None`` when there are none.

        The left-hand side of the arithmetic cross-check.
        """
        amounts = [item.amount for item in self.line_items if item.amount is not None]
        return sum(amounts, Decimal("0")) if amounts else None
