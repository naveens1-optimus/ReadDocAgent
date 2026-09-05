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

    @abstractmethod
    def download(self, container: str, blob_name: str) -> bytes:
        """Download a blob's bytes.

        Extraction re-reads the stored document rather than carrying its
        bytes through the graph, so a paused run's checkpoint stays small.
        """
