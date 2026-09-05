"""Port for Azure Document Intelligence."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["IDocumentAnalysisService"]


class IDocumentAnalysisService(ABC):
    """Reads documents with Azure Document Intelligence prebuilt models."""

    @abstractmethod
    def extract_text(self, data: bytes) -> str:
        """Return the document's text using the ``prebuilt-read`` model.

        Used as the classifier's fallback when the vision path cannot be used.
        """
