"""The canonical receipt entity.

Aliases are Azure's ``prebuilt-receipt`` field names. Fields without a default
are required.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import Field, computed_field, field_validator, model_validator

from domain.entity.common import (
    EntityModel,
    currency_of,
    to_date,
    to_decimal,
)

__all__ = ["ReceiptItem", "Receipt"]


class ReceiptItem(EntityModel):
    """One purchased item."""

    description: str | None = Field(default=None, alias="Description")
    quantity: Decimal | None = Field(default=None, alias="Quantity")
    price: Decimal | None = Field(default=None, alias="Price")
    total_price: Decimal | None = Field(default=None, alias="TotalPrice")

    _money = field_validator(
        "quantity", "price", "total_price", mode="before"
    )(to_decimal)


class Receipt(EntityModel):
    """A receipt, as the pipeline stores it."""

    # --- Required ---
    merchant_name: str = Field(alias="MerchantName")
    total: Decimal = Field(alias="Total")

    # --- Optional ---
    transaction_date: date | None = Field(default=None, alias="TransactionDate")
    merchant_address: str | None = Field(default=None, alias="MerchantAddress")
    merchant_phone: str | None = Field(default=None, alias="MerchantPhoneNumber")
    subtotal: Decimal | None = Field(default=None, alias="Subtotal")
    total_tax: Decimal | None = Field(default=None, alias="TotalTax")
    tip: Decimal | None = Field(default=None, alias="Tip")
    currency: str | None = None
    items: list[ReceiptItem] = Field(default_factory=list, alias="Items")

    _money = field_validator(
        "total", "subtotal", "total_tax", "tip", mode="before"
    )(to_decimal)
    _dates = field_validator("transaction_date", mode="before")(to_date)

    @model_validator(mode="before")
    @classmethod
    def _infer_currency(cls, data: Any) -> Any:
        """Take the currency from whichever money field carries one."""
        if not isinstance(data, dict) or data.get("currency"):
            return data
        for key in ("Total", "Subtotal", "TotalTax"):
            code = currency_of(data.get(key))
            if code:
                data = {**data, "currency": code}
                break
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def item_total(self) -> Decimal | None:
        """Sum of the item totals, falling back to price times quantity.

        ``prebuilt-receipt`` often omits the extended total on faint or
        hand-written receipts, so deriving it keeps the arithmetic check
        useful rather than silently skipped.
        """
        running: Decimal | None = None
        for item in self.items:
            if item.total_price is not None:
                line = item.total_price
            elif item.price is not None and item.quantity is not None:
                line = item.price * item.quantity
            else:
                continue
            running = line if running is None else running + line
        return running
