"""Which entity schema each document type is validated against."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from domain.entity.common import EntityModel
from domain.entity.contract import Contract
from domain.entity.invoice import Invoice
from domain.entity.receipt import Receipt
from domain.entity.resume import Resume
from domain.enum.document_type import DocumentType

__all__ = ["ENTITY_BY_TYPE", "entity_for", "required_fields_of", "field_names_of"]


#: Read-only so a caller cannot mutate shared state at runtime.
ENTITY_BY_TYPE: Mapping[DocumentType, type[EntityModel]] = MappingProxyType(
    {
        DocumentType.INVOICE: Invoice,
        DocumentType.RECEIPT: Receipt,
        DocumentType.CONTRACT: Contract,
        DocumentType.RESUME: Resume,
    }
)


def entity_for(document_type: DocumentType) -> type[EntityModel] | None:
    """The entity schema for a type, or ``None`` if it has none."""
    return ENTITY_BY_TYPE.get(document_type)


def required_fields_of(entity: type[EntityModel]) -> list[str]:
    """Field names the schema requires.

    Derived from the model rather than listed separately, so the schema is the
    single source of truth: giving a field a default makes it optional, and
    nothing else needs changing.
    """
    return [
        name for name, field in entity.model_fields.items() if field.is_required()
    ]


def field_names_of(entity: type[EntityModel]) -> list[str]:
    """Every field name the schema defines, required or not."""
    return list(entity.model_fields)
