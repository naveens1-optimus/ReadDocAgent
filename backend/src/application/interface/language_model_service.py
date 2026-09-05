"""Port for Azure OpenAI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

__all__ = ["ILanguageModelService", "SchemaT"]

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ILanguageModelService(ABC):
    """Asks Azure OpenAI for an answer that matches a Pydantic schema."""

    @abstractmethod
    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        image: bytes | None = None,
        image_media_type: str = "image/png",
    ) -> SchemaT:
        """Return a model response validated against ``schema``.

        When ``image`` is given the vision deployment is used, otherwise the
        chat deployment.
        """
