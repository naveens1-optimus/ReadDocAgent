"""Azure Document Intelligence adapter (key-based auth)."""

from __future__ import annotations

import io

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langsmith import traceable

from application.interface.document_analysis_service import IDocumentAnalysisService
from domain.schema.settings import DocumentIntelligenceSettings, RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger
from infrastructure.utilities.retry import azure_retry

__all__ = ["AzureDocumentIntelligenceService", "PREBUILT_READ"]

logger = get_logger(__name__)

#: Cheapest prebuilt model: text and page content only.
PREBUILT_READ = "prebuilt-read"


class AzureDocumentIntelligenceService(IDocumentAnalysisService):
    """Analyses documents with Azure Document Intelligence prebuilt models."""

    def __init__(
        self,
        settings: DocumentIntelligenceSettings,
        retry_policy: RetryPolicySettings,
    ) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.endpoint,
            credential=AzureKeyCredential(settings.api_key.get_secret_value()),
        )
        self._analyze_with_retry = azure_retry(retry_policy)(self._analyze_once)

    @traceable(name="document_intelligence_read", run_type="tool")
    def extract_text(self, data: bytes) -> str:
        """Return the document's text using ``prebuilt-read``."""
        result = self._analyze_with_retry(PREBUILT_READ, data)
        text = result.content or ""
        page_count = len(result.pages or [])
        logger.info(
            "Read %s characters across %s page(s) with %s",
            len(text),
            page_count,
            PREBUILT_READ,
            extra={"model_id": PREBUILT_READ, "page_count": page_count},
        )
        return text

    def _analyze_once(self, model_id: str, data: bytes):  # type: ignore[no-untyped-def]
        """Run one analysis. Returns the SDK's ``AnalyzeResult``.

        The document is wrapped in a stream because the SDK's ``body``
        parameter expects a file-like object.
        """
        poller = self._client.begin_analyze_document(model_id, body=io.BytesIO(data))
        return poller.result()
