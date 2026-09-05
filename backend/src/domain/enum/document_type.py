"""The document types the classifier can recognise.

This module is the single source of truth for the taxonomy: the values, what
each one means, and how a model's free-form answer maps back onto them. The
classifier's prompt and response schema are generated from here, so adding a
type is a one-place change.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping

__all__ = ["DocumentType", "PrebuiltModel", "EXTRACTION_MODEL_BY_TYPE"]


class PrebuiltModel(str, Enum):
    """Azure Document Intelligence prebuilt model ids."""

    INVOICE = "prebuilt-invoice"
    RECEIPT = "prebuilt-receipt"
    #: Text plus tables and spatial structure; used for contracts.
    LAYOUT = "prebuilt-layout"
    #: Text only, the cheapest model; used for resumes.
    READ = "prebuilt-read"

    @property
    def returns_typed_fields(self) -> bool:
        """Whether the model returns named fields with confidence scores.

        Only the document-specific models do. Layout and read return text,
        which Azure OpenAI then has to structure -- and that path has no
        per-field confidence to report.
        """
        return self in {PrebuiltModel.INVOICE, PrebuiltModel.RECEIPT}


class DocumentType(str, Enum):
    """A document class. Inherits from ``str`` so it serialises as plain JSON."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    RESUME = "resume"
    ID_CARD = "id_card"
    UNSUPPORTED = "unsupported"

    @property
    def description(self) -> str:
        """One-line definition, used to tell the model what this type means."""
        return _DESCRIPTIONS[self]

    @property
    def extraction_model(self) -> PrebuiltModel | None:
        """The prebuilt model that extracts fields from this type.

        ``None`` for types with no extraction path yet. Those still classify
        and are still stored -- they just skip the extraction step.
        """
        return EXTRACTION_MODEL_BY_TYPE.get(self)

    @property
    def is_extractable(self) -> bool:
        """Whether structured extraction is supported for this type."""
        return self.extraction_model is not None

    @classmethod
    def values(cls) -> list[str]:
        """Every value, e.g. for the reviewer's dropdown."""
        return [document_type.value for document_type in cls]

    @classmethod
    def as_value_list(cls) -> str:
        """Comma-separated values, for a schema field description."""
        return ", ".join(cls.values())

    @classmethod
    def as_prompt_list(cls) -> str:
        """Bullet list of every type and its definition, for a prompt."""
        return "\n".join(
            f"- {document_type.value}: {document_type.description}"
            for document_type in cls
        )

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


#: What each type means. Written for a language model to act on, so each entry
#: names the features that distinguish the type rather than restating the name.
#: Every member must appear here; a test enforces that.
_DESCRIPTIONS: Mapping[DocumentType, str] = MappingProxyType(
    {
        DocumentType.INVOICE: (
            "a request for payment, with an invoice number and totals"
        ),
        DocumentType.RECEIPT: (
            "proof of a completed purchase, from a merchant"
        ),
        DocumentType.CONTRACT: (
            "an agreement between parties, with clauses and signature blocks"
        ),
        DocumentType.RESUME: (
            "a person's CV, listing experience, education and skills"
        ),
        DocumentType.ID_CARD: "a passport, driving licence or identity card",
        DocumentType.UNSUPPORTED: "anything else",
    }
)

#: Which prebuilt model extracts each document type. Only the types listed
#: here have an extraction path; add a line to support another.
EXTRACTION_MODEL_BY_TYPE: Mapping[DocumentType, PrebuiltModel] = MappingProxyType(
    {
        DocumentType.INVOICE: PrebuiltModel.INVOICE,
        DocumentType.RECEIPT: PrebuiltModel.RECEIPT,
        DocumentType.CONTRACT: PrebuiltModel.LAYOUT,
        DocumentType.RESUME: PrebuiltModel.READ,
    }
)

#: Common ways a language model names each type, mapped to the canonical value.
_ALIASES: Mapping[str, str] = MappingProxyType(
    {
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
)
