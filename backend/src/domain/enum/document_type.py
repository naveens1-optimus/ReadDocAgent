"""The document types the classifier can recognise."""

from __future__ import annotations

from enum import Enum

__all__ = ["DocumentType"]

# Common ways a language model names each type, mapped to the canonical value.
_ALIASES = {
    "bill": "invoice",
    "tax_invoice": "invoice",
    "sales_receipt": "receipt",
    "agreement": "contract",
    "nda": "contract",
    "cv": "resume",
    "curriculum_vitae": "resume",
    "id": "id_card",
    "identity_card": "id_card",
    "passport": "id_card",
    "drivers_license": "id_card",
}


class DocumentType(str, Enum):
    """A document class. Inherits from ``str`` so it serialises as plain JSON."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    RESUME = "resume"
    ID_CARD = "id_card"
    UNSUPPORTED = "unsupported"

    @classmethod
    def from_string(cls, raw: str | None) -> DocumentType:
        """Convert model output into a known type.

        The model is asked to reply with one of these values, but can still
        return "Invoice." or "CV". Anything unrecognised becomes UNSUPPORTED
        so a surprising answer takes the fallback path instead of crashing.
        """
        if not raw:
            return cls.UNSUPPORTED

        text = raw.strip().lower().replace("-", "_").replace(" ", "_")
        text = text.strip("_.\"'")
        text = _ALIASES.get(text, text)

        try:
            return cls(text)
        except ValueError:
            return cls.UNSUPPORTED
