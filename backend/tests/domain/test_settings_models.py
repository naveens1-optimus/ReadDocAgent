"""Tests for the pure configuration models in :mod:`domain.schema.settings`."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from domain.schema.settings import (
    AppSettings,
    AzureOpenAISettings,
    BlobStorageSettings,
    DocumentIntelligenceSettings,
    LangSmithSettings,
    RetryPolicySettings,
    Settings,
)


def _document_intelligence(**overrides: object) -> DocumentIntelligenceSettings:
    defaults: dict[str, object] = {
        "endpoint": "https://x.cognitiveservices.azure.com",
        "api_key": "key",
    }
    return DocumentIntelligenceSettings(**(defaults | overrides))


def _azure_openai(**overrides: object) -> AzureOpenAISettings:
    defaults: dict[str, object] = {
        "endpoint": "https://x.openai.azure.com",
        "api_key": "key",
        "chat_deployment": "gpt-4o",
        "vision_deployment": "gpt-4o",
    }
    return AzureOpenAISettings(**(defaults | overrides))


def _blob_storage(**overrides: object) -> BlobStorageSettings:
    defaults: dict[str, object] = {"connection_string": "conn"}
    return BlobStorageSettings(**(defaults | overrides))


class TestEndpointNormalisation:
    """Endpoint validation shared by the Azure service models."""

    def test_strips_trailing_slash(self) -> None:
        """The SDKs append paths, so a trailing slash would double-slash URLs."""
        settings = _document_intelligence(
            endpoint="https://x.cognitiveservices.azure.com/"
        )
        assert settings.endpoint == "https://x.cognitiveservices.azure.com"

    def test_strips_multiple_trailing_slashes(self) -> None:
        settings = _document_intelligence(
            endpoint="https://x.cognitiveservices.azure.com///"
        )
        assert settings.endpoint == "https://x.cognitiveservices.azure.com"

    @pytest.mark.parametrize(
        "bad", ["", "   ", "x.cognitiveservices.azure.com", "ftp://x.com"]
    )
    def test_rejects_non_http_url(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _document_intelligence(endpoint=bad)

    def test_accepts_http_for_local_emulators(self) -> None:
        settings = _document_intelligence(endpoint="http://localhost:8080")
        assert settings.endpoint == "http://localhost:8080"


class TestSecretHandling:
    """Credentials must never render in plain text."""

    def test_api_key_is_masked_in_repr(self) -> None:
        settings = _document_intelligence(api_key="super-secret-value")
        assert "super-secret-value" not in repr(settings)
        assert "super-secret-value" not in str(settings)

    def test_api_key_is_retrievable(self) -> None:
        settings = _document_intelligence(api_key="super-secret-value")
        assert settings.api_key.get_secret_value() == "super-secret-value"

    def test_connection_string_is_masked(self) -> None:
        settings = _blob_storage(connection_string="AccountKey=SECRET==")
        assert "SECRET" not in repr(settings)


class TestModelStrictness:
    """Immutability and unknown-field rejection."""

    def test_models_are_frozen(self) -> None:
        settings = _document_intelligence()
        with pytest.raises(ValidationError):
            settings.endpoint = "https://other.azure.com"  # type: ignore[misc]

    def test_unknown_field_is_rejected(self) -> None:
        """A typo'd settings key must fail loudly, not be silently ignored."""
        with pytest.raises(ValidationError):
            _document_intelligence(api_kye="typo")


class TestAzureOpenAISettings:
    def test_deployments_are_required(self) -> None:
        with pytest.raises(ValidationError):
            AzureOpenAISettings(
                endpoint="https://x.openai.azure.com",
                api_key="k",
                chat_deployment="   ",
                vision_deployment="gpt-4o",
            )

    def test_deployments_may_differ(self) -> None:
        settings = _azure_openai(chat_deployment="gpt-4o-mini", vision_deployment="gpt-4o")
        assert settings.chat_deployment == "gpt-4o-mini"
        assert settings.vision_deployment == "gpt-4o"


class TestBlobStorageSettings:
    """Container naming is validated up front so Agent 4 cannot fail late."""

    def test_defaults(self) -> None:
        settings = _blob_storage()
        assert settings.input_container == "idp-input"
        assert settings.output_container == "idp-output"

    def test_lowercases_container_name(self) -> None:
        assert _blob_storage(input_container="IDP-Input").input_container == "idp-input"

    @pytest.mark.parametrize(
        "bad",
        [
            "ab",                    # too short
            "a" * 64,                # too long
            "-leading",              # must start alphanumeric
            "trailing-",             # must end alphanumeric
            "double--hyphen",        # no consecutive hyphens
            "under_score",           # invalid character
            "has space",             # invalid character
        ],
    )
    def test_rejects_invalid_container_names(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _blob_storage(input_container=bad)

    @pytest.mark.parametrize("good", ["abc", "idp-input", "a1-b2-c3", "a" * 63])
    def test_accepts_valid_container_names(self, good: str) -> None:
        assert _blob_storage(input_container=good).input_container == good


class TestLangSmithSettings:
    def test_disabled_by_default_without_key(self) -> None:
        settings = LangSmithSettings()
        assert settings.tracing_enabled is False
        assert settings.api_key is None

    def test_enabled_requires_api_key(self) -> None:
        """Tracing on with no key would silently record nothing."""
        with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
            LangSmithSettings(tracing_enabled=True)

    def test_enabled_with_key_is_valid(self) -> None:
        settings = LangSmithSettings(tracing_enabled=True, api_key=SecretStr("ls-key"))
        assert settings.tracing_enabled is True

    def test_key_without_tracing_is_allowed(self) -> None:
        """Keeping a key on hand while tracing is off is a normal state."""
        settings = LangSmithSettings(tracing_enabled=False, api_key=SecretStr("ls-key"))
        assert settings.tracing_enabled is False


class TestRetryPolicySettings:
    def test_defaults(self) -> None:
        policy = RetryPolicySettings()
        assert policy.max_attempts == 4
        assert policy.base_delay_seconds == pytest.approx(1.0)
        assert policy.max_delay_seconds == pytest.approx(30.0)

    @pytest.mark.parametrize("attempts", [0, -1, 11])
    def test_rejects_out_of_range_attempts(self, attempts: int) -> None:
        with pytest.raises(ValidationError):
            RetryPolicySettings(max_attempts=attempts)

    def test_max_delay_must_not_be_below_base(self) -> None:
        with pytest.raises(ValidationError, match="max_delay_seconds"):
            RetryPolicySettings(base_delay_seconds=10.0, max_delay_seconds=5.0)

    def test_single_attempt_disables_retry(self) -> None:
        assert RetryPolicySettings(max_attempts=1).max_attempts == 1


class TestAppSettings:
    def test_defaults(self) -> None:
        app = AppSettings()
        assert app.environment == "local"
        assert app.confidence_threshold == pytest.approx(0.80)
        assert app.is_production is False

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_confidence_threshold_bounded_to_unit_interval(
        self, threshold: float
    ) -> None:
        """Confidence is a 0-1 ratio; anything else is a misconfiguration."""
        with pytest.raises(ValidationError):
            AppSettings(confidence_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_accepts_boundary_thresholds(self, threshold: float) -> None:
        assert AppSettings(confidence_threshold=threshold).confidence_threshold == (
            pytest.approx(threshold)
        )

    def test_upload_size_converted_to_bytes(self) -> None:
        assert AppSettings(max_upload_size_mb=20).max_upload_size_bytes == 20 * 1024 * 1024

    def test_is_production_flag(self) -> None:
        assert AppSettings(environment="production").is_production is True

    def test_rejects_unknown_environment(self) -> None:
        with pytest.raises(ValidationError):
            AppSettings(environment="prd")

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(ValidationError):
            AppSettings(log_level="TRACE")


class TestSettingsAggregate:
    """The aggregate root and its credential-free summary."""

    @staticmethod
    def _build() -> Settings:
        return Settings(
            app=AppSettings(),
            document_intelligence=_document_intelligence(api_key="di-secret"),
            azure_openai=_azure_openai(api_key="aoai-secret"),
            blob_storage=_blob_storage(connection_string="AccountKey=blob-secret=="),
            langsmith=LangSmithSettings(
                tracing_enabled=True, api_key=SecretStr("ls-secret")
            ),
            retry=RetryPolicySettings(),
        )

    def test_summary_is_human_readable(self) -> None:
        summary = self._build().summary()
        assert "doc intelligence" in summary
        assert "gpt-4o" in summary
        assert "langsmith tracing    : on" in summary

    @pytest.mark.parametrize(
        "secret", ["di-secret", "aoai-secret", "blob-secret", "ls-secret"]
    )
    def test_summary_leaks_no_credentials(self, secret: str) -> None:
        """The summary is logged at startup, so it must be credential-free."""
        settings = self._build()
        assert secret not in settings.summary()
        assert secret not in repr(settings)

    def test_round_trips_through_serialisation(self) -> None:
        settings = self._build()
        restored = Settings.model_validate(settings.model_dump())
        assert restored.document_intelligence.endpoint == (
            settings.document_intelligence.endpoint
        )
        assert restored.app.confidence_threshold == settings.app.confidence_threshold
