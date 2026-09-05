"""LangSmith tracing setup.

LangChain and LangGraph pick tracing up from environment variables, so this
copies the validated settings into the environment before any graph is built.
Every graph run, node and Azure OpenAI call is then traced automatically.
"""

from __future__ import annotations

import os

from domain.schema.settings import LangSmithSettings
from infrastructure.utilities.logging_config import get_logger

__all__ = ["configure_langsmith"]

logger = get_logger(__name__)


def configure_langsmith(settings: LangSmithSettings) -> bool:
    """Enable or disable tracing. Returns whether it is on."""
    if not settings.tracing_enabled or settings.api_key is None:
        # Set explicitly rather than left unset, so a stray LANGSMITH_TRACING
        # in the shell cannot switch tracing on with no key behind it.
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.project
    os.environ["LANGSMITH_ENDPOINT"] = settings.endpoint

    logger.info("LangSmith tracing enabled (project %r)", settings.project)
    return True
