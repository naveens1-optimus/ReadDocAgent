"""Azure OpenAI adapter (key-based auth)."""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from application.interface.language_model_service import ILanguageModelService, SchemaT
from domain.schema.settings import AzureOpenAISettings, RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger

__all__ = ["AzureOpenAIService", "build_messages"]

logger = get_logger(__name__)


def build_messages(
    system_prompt: str,
    user_prompt: str,
    image: bytes | None = None,
    image_media_type: str = "image/png",
) -> list[BaseMessage]:
    """Build the message list, attaching an image as a base64 data URL.

    Separate from the client so it can be tested without calling Azure.
    """
    if image is None:
        return [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    encoded = base64.b64encode(image).decode("ascii")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_media_type};base64,{encoded}"},
        },
    ]
    return [SystemMessage(system_prompt), HumanMessage(content=content)]


class AzureOpenAIService(ILanguageModelService):
    """Calls Azure OpenAI and returns responses validated against a schema."""

    def __init__(
        self, settings: AzureOpenAISettings, retry_policy: RetryPolicySettings
    ) -> None:
        self._chat = self._build_client(settings, settings.chat_deployment, retry_policy)
        self._vision = (
            self._chat
            if settings.vision_deployment == settings.chat_deployment
            else self._build_client(
                settings, settings.vision_deployment, retry_policy
            )
        )

    @staticmethod
    def _build_client(
        settings: AzureOpenAISettings,
        deployment: str,
        retry_policy: RetryPolicySettings,
    ) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=settings.endpoint,
            azure_deployment=deployment,
            api_version=settings.api_version,
            api_key=settings.api_key.get_secret_value(),
            # Deterministic: the same document should classify the same way.
            temperature=0.0,
            # The OpenAI SDK does its own exponential backoff on 429/5xx.
            max_retries=retry_policy.max_attempts - 1,
        )

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        image: bytes | None = None,
        image_media_type: str = "image/png",
    ) -> SchemaT:
        """Ask the model for an answer matching ``schema``."""
        client = self._vision if image is not None else self._chat
        messages = build_messages(
            system_prompt, user_prompt, image, image_media_type
        )

        logger.debug(
            "Calling Azure OpenAI (%s, schema=%s)",
            "vision" if image is not None else "chat",
            schema.__name__,
        )
        result = client.with_structured_output(schema).invoke(messages)
        return schema.model_validate(result)
