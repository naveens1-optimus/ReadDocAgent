"""Agent 2: extracts structured fields using a type-specific prebuilt model."""

from __future__ import annotations

from langsmith import traceable

from application.interface.document_analysis_service import IDocumentAnalysisService
from domain.entity.extraction_result import ExtractionResult
from domain.enum.document_type import DocumentType
from infrastructure.utilities.logging_config import get_logger

__all__ = ["DataExtractionAgent"]

logger = get_logger(__name__)


class DataExtractionAgent:
    """Runs the Document Intelligence model that matches the document type.

    Which model to use is the document type's own business
    (``DocumentType.extraction_model``), so supporting a new type is a line in
    that mapping rather than a change here.
    """

    def __init__(self, analysis: IDocumentAnalysisService) -> None:
        self._analysis = analysis

    @traceable(name="data_extraction", run_type="chain")
    def extract(self, document_type: DocumentType, data: bytes) -> ExtractionResult:
        """Extract fields for a document of the given type.

        Raises:
            ValueError: If the type has no extraction model. Callers should
                check ``document_type.is_extractable`` first.
        """
        model = document_type.extraction_model
        if model is None:
            raise ValueError(
                f"No extraction model for document type {document_type.value!r}"
            )

        logger.info("Extracting %s with %s", document_type.value, model.value)
        return self._analysis.extract_fields(model.value, data)
