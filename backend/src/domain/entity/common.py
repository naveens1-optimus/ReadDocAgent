"""Shared pieces of the document entities.

Money uses :class:`~decimal.Decimal`, never ``float``. The validation agent
cross-checks line items against totals, and binary floating point makes that
unreliable -- ``0.1 + 0.2 != 0.3`` would fail perfectly correct invoices.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["EntityModel", "to_decimal", "currency_of", "to_date", "DAY_FIRST"]

#: How to read a slash date whose meaning is ambiguous -- 09/03/2026 is the
#: 9th of March if day-first, the 3rd of September if month-first. Set to
#: False for US-style documents.
#:
#: This only affects values typed by hand: Azure always returns ISO 8601, and
#: the UI sends ISO from its date picker.
DAY_FIRST = True

#: Date formats accepted from a reviewer, tried in order. ISO comes first
#: because it is the only unambiguous one.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)

#: The ambiguous pair, ordered by the DAY_FIRST preference.
_SLASH_FORMATS: tuple[str, ...] = ("%d/%m/%Y", "%m/%d/%Y")
_DASH_FORMATS: tuple[str, ...] = ("%d-%m-%Y", "%m-%d-%Y")


class EntityModel(BaseModel):
    """Base for the document entities.

    ``populate_by_name`` lets a model be built either from Azure's field names
    (``VendorName``) via alias, or from our own (``vendor_name``) -- the
    former when validating an extraction, the latter when reading one back.

    ``extra="ignore"`` because Azure returns fields we do not model, and
    dropping them is better than failing an otherwise good extraction.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def to_decimal(value: Any) -> Decimal | None:
    """Coerce a Document Intelligence money value into a ``Decimal``.

    Handles the shapes a money field actually arrives in: Azure's
    ``{"amount": 495.0, "currency": "USD"}``, a bare number, or a string a
    reviewer typed such as ``"$1,234.56"``.
    """
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, dict):
        return to_decimal(value.get("amount"))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if not isinstance(value, str):
        return None

    cleaned = re.sub(r"[^\d.\-]", "", value.strip())
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def currency_of(value: Any) -> str | None:
    """Pull the currency code out of an Azure money value, if it has one."""
    if isinstance(value, dict):
        code = value.get("currency") or value.get("currency_code")
        return str(code).upper() if code else None
    return None


def to_date(value: Any) -> Any:
    """Parse a date from the forms Azure or a reviewer might supply.

    Azure returns ISO 8601, which Pydantic already handles. This exists for
    values a person typed: a reviewer correcting a date writes "01/01/2003",
    and rejecting that with "invalid character in year" is not a useful
    answer.

    An unrecognised string is returned unchanged so Pydantic reports the
    problem in its own terms, rather than this silently turning a typo into
    ``None``.
    """
    if value is None or isinstance(value, (date, datetime)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    ordered = (
        *_DATE_FORMATS,
        *(_SLASH_FORMATS if DAY_FIRST else _SLASH_FORMATS[::-1]),
        *(_DASH_FORMATS if DAY_FIRST else _DASH_FORMATS[::-1]),
    )
    for fmt in ordered:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return value
