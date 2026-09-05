"""Agent 1 -- classifies a document into a known type.

Primary path: render the first page and ask the Azure OpenAI vision model.
Fallback: if the vision path fails, read the text with Azure Document
Intelligence (``prebuilt-read``) and classify that instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from application.interface.document_analysis_service import IDocumentAnalysisService
from application.interface.language_model_service import ILanguageModelService
from domain.entity.classification_result import ClassificationResult
from domain.enum.document_type import DocumentType
from infrastructure.tools.document_render_tool import media_type_for, render_first_page
from infrastructure.utilities.logging_config import get_logger

__all__ = ["DocumentClassificationAgent"]

logger = get_logger(__name__)

#: Characters of extracted text sent to the model in the fallback path. The
#: opening of a document is what identifies it, and a cap keeps the prompt
#: small and cheap.
_TEXT_EXCERPT_LIMIT = 4000

SYSTEM_PROMPT = """\
You classify business documents. Reply with exactly one document_type from:

- invoice: a request for payment, with an invoice number and totals
- receipt: proof of a completed purchase, from a merchant
- contract: an agreement between parties, with clauses and signature blocks
- resume: a person's CV, listing experience, education and skills
- id_card: a passport, driving licence or identity card
- unsupported: anything else

Set confidence to how certain you are, from 0.0 to 1.0. Use a value below 0.5
when the document is unclear or does not fit any category. Keep reasoning to
one short sentence."""

VISION_PROMPT = "Classify this document."

TEXT_PROMPT = """\
Classify the document from this extracted text:

---
{text}
---"""


class _Verdict(BaseModel):
    """Structured answer requested from the model."""

    document_type: str = Field(
        description="One of: invoice, receipt, contract, resume, id_card, unsupported"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One short sentence.")


class DocumentClassificationAgent:
    """Decides what kind of document was uploaded."""

    def __init__(
        self,
        language_model: ILanguageModelService,
        analysis_service: IDocumentAnalysisService,
    ) -> None:
        self._language_model = language_model
        self._analysis_service = analysis_service

    def classify(self, data: bytes, extension: str) -> ClassificationResult:
        """Classify a document, falling back to text if vision fails.

        Args:
            data: The raw file bytes.
            extension: Lowercased file extension, e.g. ".pdf".
        """
        try:
            return self._classify_with_vision(data, extension)
        except Exception as exc:  # noqa: BLE001 - any vision failure is recoverable
            logger.warning(
                "Vision classification failed (%s); falling back to text.", exc
            )
            return self._classify_with_text(data)

    def _classify_with_vision(
        self, data: bytes, extension: str
    ) -> ClassificationResult:
        if extension == ".pdf":
            image = render_first_page(data)
            media_type = "image/png"
        else:
            image = data
            media_type = media_type_for(extension)

        verdict = self._language_model.structured_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=VISION_PROMPT,
            schema=_Verdict,
            image=image,
            image_media_type=media_type,
        )
        return self._to_result(verdict, used_fallback=False)

    def _classify_with_text(self, data: bytes) -> ClassificationResult:
        text = self._analysis_service.extract_text(data)
        if not text.strip():
            logger.warning("No text could be extracted; classifying as unsupported.")
            return ClassificationResult(
                document_type=DocumentType.UNSUPPORTED,
                confidence=0.0,
                reasoning="Neither the vision model nor text extraction could read the document.",
                used_fallback=True,
            )

        verdict = self._language_model.structured_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=TEXT_PROMPT.format(text=text[:_TEXT_EXCERPT_LIMIT]),
            schema=_Verdict,
        )
        return self._to_result(verdict, used_fallback=True)

    @staticmethod
    def _to_result(verdict: _Verdict, used_fallback: bool) -> ClassificationResult:
        result = ClassificationResult(
            document_type=DocumentType.from_string(verdict.document_type),
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            used_fallback=used_fallback,
        )
        logger.info(
            "Classified document as %s",
            result,
            extra={
                "document_type": result.document_type.value,
                "confidence": result.confidence,
                "used_fallback": used_fallback,
            },
        )
        return result
