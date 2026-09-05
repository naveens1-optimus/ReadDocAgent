"""LangSmith tracing setup.

LangChain and LangGraph pick tracing up from environment variables, so this
copies the validated settings into the environment before any graph is built.
Every graph run, node and Azure OpenAI call is then traced automatically.
"""

from __future__ import annotations

import os

from langsmith import utils as langsmith_utils

from domain.schema.settings import LangSmithSettings
from infrastructure.utilities.logging_config import get_logger

__all__ = ["configure_langsmith", "tracing_is_active"]

logger = get_logger(__name__)


def _apply_env_changes() -> None:
    """Make environment changes visible to LangSmith.

    ``langsmith.utils.get_env_var`` is ``lru_cache``d, so if anything reads
    ``LANGSMITH_TRACING`` before we set it, the cached value sticks and
    tracing silently never turns on -- the classic "configured but no traces
    appear" failure. Clearing the cache makes the setting take effect
    regardless of import order.
    """
    langsmith_utils.get_env_var.cache_clear()


def configure_langsmith(settings: LangSmithSettings) -> bool:
    """Enable or disable tracing. Returns whether it is on."""
    if not settings.tracing_enabled or settings.api_key is None:
        # Set explicitly rather than left unset, so a stray LANGSMITH_TRACING
        # in the shell cannot switch tracing on with no key behind it.
        os.environ["LANGSMITH_TRACING"] = "false"
        _apply_env_changes()
        logger.info("LangSmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.project
    os.environ["LANGSMITH_ENDPOINT"] = settings.endpoint
    _apply_env_changes()

    # Confirm with LangSmith rather than assuming. Silently-not-tracing is the
    # failure mode worth catching: the app looks instrumented but records
    # nothing.
    if not tracing_is_active():
        logger.warning(
            "LangSmith was configured but reports tracing as OFF; no traces "
            "will be recorded for project %r.",
            settings.project,
        )
        return False

    logger.info(
        "LangSmith tracing ENABLED -- project %r at %s",
        settings.project,
        settings.endpoint,
    )
    return True


def tracing_is_active() -> bool:
    """Whether LangSmith would record a run right now.

    Reports what LangSmith itself thinks, rather than what we asked for, so a
    misconfiguration shows up instead of being assumed away.
    """
    return bool(langsmith_utils.tracing_is_enabled())
