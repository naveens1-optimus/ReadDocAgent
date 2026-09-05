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
    "AZURE_COSMOS_ENDPOINT",
    "AZURE_COSMOS_KEY",
    "AZURE_COSMOS_DATABASE",
    "AZURE_COSMOS_CONTAINER",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "APP_ENV",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "CONFIDENCE_THRESHOLD",
    "ARITHMETIC_TOLERANCE",
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
    "AZURE_COSMOS_ENDPOINT": "https://test-cosmos.documents.azure.com:443/",
    "AZURE_COSMOS_KEY": "test-cosmos-key",
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


@pytest.fixture(autouse=True)
def _ignore_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop the developer's own ``backend/src/.env`` reaching the tests.

    ``build_settings`` calls ``load_env()``, which would read that file and
    repopulate the very variables ``clean_env`` just removed -- making tests
    pass or fail depending on whether the local .env happens to be filled in.
    Neutralising the load here means settings come only from what a test sets.
    """
    monkeypatch.setattr(
        "infrastructure.config.settings.load_env", lambda *args, **kwargs: None
    )
    yield


@pytest.fixture(autouse=True)
def _offline_checkpointer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap the Cosmos DB checkpointer for an in-memory SQLite one.

    ``CosmosDBSaverSync`` connects and creates its database in its
    constructor, so building the services would otherwise make a real network
    call during app startup. The suite must run offline and fast, and the
    checkpointer is injected, so substituting it here changes nothing about
    what the tests actually exercise.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    monkeypatch.setattr(
        "infrastructure.services.service_registry.build_checkpointer",
        lambda _settings: SqliteSaver(
            sqlite3.connect(":memory:", check_same_thread=False)
        ),
    )
    yield


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
