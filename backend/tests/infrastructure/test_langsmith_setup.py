"""Tests for LangSmith tracing setup."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from langsmith import utils as langsmith_utils
from pydantic import SecretStr

from domain.schema.settings import LangSmithSettings
from infrastructure.utilities.langsmith_setup import (
    configure_langsmith,
    tracing_is_active,
)


@pytest.fixture(autouse=True)
def _clean_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate LangSmith environment variables and its cached lookups."""
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    langsmith_utils.get_env_var.cache_clear()
    yield
    langsmith_utils.get_env_var.cache_clear()


def enabled_settings(**overrides: object) -> LangSmithSettings:
    defaults: dict[str, object] = {
        "tracing_enabled": True,
        "api_key": SecretStr("lsv2_test_key"),
        "project": "test-project",
    }
    return LangSmithSettings(**(defaults | overrides))


class TestTracingEnabled:
    def test_sets_environment_variables(self) -> None:
        assert configure_langsmith(enabled_settings()) is True

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test_key"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
        assert os.environ["LANGSMITH_ENDPOINT"]

    def test_langsmith_actually_reports_tracing_on(self) -> None:
        """The point of the exercise: LangSmith itself must agree."""
        configure_langsmith(enabled_settings())
        assert tracing_is_active() is True

    def test_takes_effect_even_after_a_disabled_read(self) -> None:
        """Regression: ``get_env_var`` is lru_cached.

        If anything reads LANGSMITH_TRACING while it is unset, the cached
        "off" would stick and tracing would silently never turn on -- the app
        looks instrumented but records nothing. configure_langsmith clears
        that cache.
        """
        assert tracing_is_active() is False  # poison the cache with "off"

        configure_langsmith(enabled_settings())

        assert tracing_is_active() is True

    def test_custom_endpoint_is_applied(self) -> None:
        configure_langsmith(
            enabled_settings(endpoint="https://eu.api.smith.langchain.com")
        )
        assert os.environ["LANGSMITH_ENDPOINT"] == (
            "https://eu.api.smith.langchain.com"
        )


class TestTracingDisabled:
    def test_returns_false_and_disables(self) -> None:
        assert configure_langsmith(LangSmithSettings(tracing_enabled=False)) is False
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert tracing_is_active() is False

    def test_disables_explicitly_over_a_stray_shell_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leftover shell variable must not enable tracing with no key."""
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        langsmith_utils.get_env_var.cache_clear()

        assert configure_langsmith(LangSmithSettings(tracing_enabled=False)) is False
        assert tracing_is_active() is False

    def test_turns_tracing_back_off_after_it_was_on(self) -> None:
        configure_langsmith(enabled_settings())
        assert tracing_is_active() is True

        configure_langsmith(LangSmithSettings(tracing_enabled=False))
        assert tracing_is_active() is False


class TestSettingsValidation:
    def test_enabling_without_a_key_is_rejected_at_config_time(self) -> None:
        """Caught by the settings model, before anything can silently no-op."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
            LangSmithSettings(tracing_enabled=True)
