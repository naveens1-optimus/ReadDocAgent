"""Port for object storage."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["IBlobStorageService"]


class IBlobStorageService(ABC):
    """Stores documents and processing output."""

    @abstractmethod
    def upload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> str:
        """Upload bytes and return the blob URL. Creates the container if needed."""
