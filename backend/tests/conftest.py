"""Shared pytest fixtures.

The important one here is :func:`clean_env`, which strips every variable the
application reads. Without it, tests would pick up whatever sits in a
developer's real ``backend/src/.env`` and pass or fail depending on the
machine they run on.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

#: Every environment variable the application consults. Kept in one place so
#: that adding a setting without isolating it in tests is an obvious omission.
APP_ENV_VARS: tuple[str, ...] = (
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_VISION_DEPLOYMENT",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_INPUT_CONTAINER",
    "AZURE_STORAGE_OUTPUT_CONTAINER",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "APP_ENV",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "CONFIDENCE_THRESHOLD",
    "ARITHMETIC_TOLERANCE",
    "CHECKPOINT_DB_PATH",
    "MAX_UPLOAD_SIZE_MB",
    "AZURE_MAX_ATTEMPTS",
    "AZURE_RETRY_BASE_DELAY_SECONDS",
    "AZURE_RETRY_MAX_DELAY_SECONDS",
)

#: The smallest environment that yields a valid ``Settings`` aggregate.
MINIMAL_VALID_ENV: dict[str, str] = {
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "https://test-di.cognitiveservices.azure.com",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY": "test-di-key",
    "AZURE_OPENAI_ENDPOINT": "https://test-aoai.openai.azure.com",
    "AZURE_OPENAI_KEY": "test-aoai-key",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o-test",
    "AZURE_STORAGE_CONNECTION_STRING": (
        "DefaultEndpointsProtocol=https;AccountName=test;"
        "AccountKey=dGVzdC1hY2NvdW50LWtleQ==;EndpointSuffix=core.windows.net"
    ),
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove all application environment variables for the duration of a test."""
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def valid_env(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Populate the minimal set of variables needed for valid settings."""
    for name, value in MINIMAL_VALID_ENV.items():
        monkeypatch.setenv(name, value)
    yield dict(MINIMAL_VALID_ENV)


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Clear the cached settings singleton around every test.

    ``get_settings`` is ``lru_cache``d, so without this a value resolved by
    one test would bleed into the next.
    """
    from infrastructure.config.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def unset_env(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that unsets a named variable."""

    def _unset(name: str) -> None:
        monkeypatch.delenv(name, raising=False)

    return _unset


def pytest_configure(config: pytest.Config) -> None:
    """Guard against a stray dotenv file influencing the suite.

    ``load_env`` deliberately does not override real environment variables,
    but a ``.env`` could still supply a value a test expects to be absent.
    Setting this flag lets loader tests opt out of dotenv loading entirely.
    """
    os.environ.setdefault("PYTEST_RUNNING", "1")
