"""Tests for :mod:`infrastructure.config.settings` -- environment to models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from infrastructure.config.settings import (
    build_settings,
    get_settings,
    reset_settings_cache,
)
from infrastructure.utilities.load_env import MissingEnvironmentVariableError


def _build() -> object:
    """Build settings without consulting a developer's local dotenv file."""
    return build_settings(load_dotenv_file=False)


class TestRequiredVariables:
    """Every required variable must fail loudly and by name."""

    @pytest.mark.parametrize(
        "missing",
        [
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "AZURE_DOCUMENT_INTELLIGENCE_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_STORAGE_CONNECTION_STRING",
        ],
    )
    def test_missing_required_variable_raises_named_error(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        monkeypatch.delenv(missing, raising=False)
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            _build()
        assert exc_info.value.variable_name == missing

    def test_minimal_valid_environment_builds(self, valid_env: dict[str, str]) -> None:
        settings = _build()
        assert settings.document_intelligence.endpoint == (
            "https://test-di.cognitiveservices.azure.com"
        )
        assert settings.azure_openai.chat_deployment == "gpt-4o-test"


class TestDefaults:
    """Optional variables fall back to documented defaults."""

    def test_applies_documented_defaults(self, valid_env: dict[str, str]) -> None:
        settings = _build()
        assert settings.app.environment == "local"
        assert settings.app.log_level == "INFO"
        assert settings.app.log_format == "text"
        assert settings.app.confidence_threshold == pytest.approx(0.80)
        assert settings.app.arithmetic_tolerance == pytest.approx(0.01)
        assert settings.app.max_upload_size_mb == 20
        assert settings.blob_storage.input_container == "idp-input"
        assert settings.blob_storage.output_container == "idp-output"
        assert settings.document_intelligence.api_version == "2024-11-30"
        assert settings.azure_openai.api_version == "2024-10-21"
        assert settings.retry.max_attempts == 4
        assert settings.langsmith.tracing_enabled is False

    def test_vision_deployment_defaults_to_chat_deployment(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One vision-capable deployment can serve both roles."""
        monkeypatch.delenv("AZURE_OPENAI_VISION_DEPLOYMENT", raising=False)
        settings = _build()
        assert settings.azure_openai.vision_deployment == "gpt-4o-test"

    def test_explicit_vision_deployment_is_respected(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4o-vision")
        settings = _build()
        assert settings.azure_openai.vision_deployment == "gpt-4o-vision"
        assert settings.azure_openai.chat_deployment == "gpt-4o-test"


class TestOverrides:
    """Values supplied in the environment reach the right model field."""

    def test_overrides_are_applied_and_coerced(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("LOG_FORMAT", "JSON")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.65")
        monkeypatch.setenv("ARITHMETIC_TOLERANCE", "0.05")
        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "50")
        monkeypatch.setenv("AZURE_MAX_ATTEMPTS", "6")
        monkeypatch.setenv("AZURE_RETRY_BASE_DELAY_SECONDS", "0.5")
        monkeypatch.setenv("AZURE_RETRY_MAX_DELAY_SECONDS", "60")

        settings = _build()

        assert settings.app.environment == "production"
        assert settings.app.is_production is True
        # Case is normalised, so `debug` / `JSON` in .env are accepted.
        assert settings.app.log_level == "DEBUG"
        assert settings.app.log_format == "json"
        assert settings.app.confidence_threshold == pytest.approx(0.65)
        assert settings.app.arithmetic_tolerance == pytest.approx(0.05)
        assert settings.app.max_upload_size_mb == 50
        assert settings.retry.max_attempts == 6
        assert settings.retry.base_delay_seconds == pytest.approx(0.5)
        assert settings.retry.max_delay_seconds == pytest.approx(60.0)

    def test_container_names_are_overridable(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_STORAGE_INPUT_CONTAINER", "custom-in")
        monkeypatch.setenv("AZURE_STORAGE_OUTPUT_CONTAINER", "custom-out")
        settings = _build()
        assert settings.blob_storage.input_container == "custom-in"
        assert settings.blob_storage.output_container == "custom-out"

    def test_invalid_value_raises_validation_error(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Out-of-range values are rejected at startup, not silently clamped."""
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")
        with pytest.raises(ValidationError):
            _build()


class TestLangSmithLoading:
    def test_tracing_enabled_requires_key(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
            _build()

    def test_tracing_enabled_with_key(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
        settings = _build()
        assert settings.langsmith.tracing_enabled is True
        assert settings.langsmith.project == "my-project"


class TestSettingsCaching:
    """``get_settings`` is a cached singleton."""

    def test_returns_same_instance(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "infrastructure.config.settings.load_env", lambda *a, **k: None
        )
        reset_settings_cache()
        assert get_settings() is get_settings()

    def test_cache_reset_allows_reresolution(
        self, valid_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "infrastructure.config.settings.load_env", lambda *a, **k: None
        )
        reset_settings_cache()
        first = get_settings()

        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.55")
        assert get_settings() is first, "cache should hold until explicitly reset"

        reset_settings_cache()
        assert get_settings().app.confidence_threshold == pytest.approx(0.55)


class TestNoCredentialLeak:
    def test_summary_of_loaded_settings_is_clean(
        self, valid_env: dict[str, str]
    ) -> None:
        """Guards the startup banner against leaking the account key."""
        settings = _build()
        summary = settings.summary()
        assert "test-di-key" not in summary
        assert "test-aoai-key" not in summary
        assert "dGVzdC1hY2NvdW50LWtleQ==" not in summary
