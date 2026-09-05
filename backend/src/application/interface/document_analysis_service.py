"""Port for Azure Document Intelligence."""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entity.extraction_result import ExtractionResult

__all__ = ["IDocumentAnalysisService"]


class IDocumentAnalysisService(ABC):
    """Reads documents with Azure Document Intelligence prebuilt models."""

    @abstractmethod
    def extract_text(self, data: bytes) -> str:
        """Return the document's text using the ``prebuilt-read`` model.

        Used as the classifier's fallback when the vision path cannot be used.
        """

    @abstractmethod
    def extract_fields(self, model_id: str, data: bytes) -> ExtractionResult:
        """Extract structured fields using the given prebuilt model.

        Args:
            model_id: Prebuilt model to run, e.g. ``prebuilt-invoice``.
            data: The document bytes.

        Returns:
            The fields found, each carrying its own confidence score where
            Document Intelligence reports one.
        """
