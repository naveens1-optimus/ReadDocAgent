"""Port for the validated-entity store."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["IEntityStore"]


class IEntityStore(ABC):
    """Stores validated business entities, keyed on document id."""

    @abstractmethod
    def save(
        self, document_id: str, document_type: str, entity: dict[str, Any]
    ) -> str:
        """Store an entity and return its id.

        Upserts, so re-running a document replaces its entity rather than
        creating a duplicate.
        """

    @abstractmethod
    def get(self, document_id: str) -> dict[str, Any] | None:
        """Return a stored entity, or ``None`` when there is none."""
