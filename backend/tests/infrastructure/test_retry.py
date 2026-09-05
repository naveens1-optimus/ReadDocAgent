"""Tests for the Azure retry policy."""

from __future__ import annotations

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)

from domain.schema.settings import RetryPolicySettings
from infrastructure.utilities.retry import azure_retry, is_transient_error


def http_error(status_code: int) -> HttpResponseError:
    error = HttpResponseError(message=f"status {status_code}")
    error.status_code = status_code
    return error


class TestIsTransientError:
    @pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
    def test_retryable_status_codes(self, status_code: int) -> None:
        assert is_transient_error(http_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 422])
    def test_client_errors_are_not_retried(self, status_code: int) -> None:
        """Retrying a bad key or bad request only wastes time and quota."""
        assert is_transient_error(http_error(status_code)) is False

    def test_network_errors_are_retried(self) -> None:
        assert is_transient_error(ServiceRequestError(message="no route")) is True
        assert is_transient_error(ServiceResponseError(message="reset")) is True

    def test_unrelated_exceptions_are_not_retried(self) -> None:
        assert is_transient_error(ValueError("bad input")) is False
        assert is_transient_error(ClientAuthenticationError(message="bad key")) is False


class TestAzureRetry:
    @pytest.fixture
    def fast_policy(self) -> RetryPolicySettings:
        """Tiny delays so the tests stay fast."""
        return RetryPolicySettings(
            max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.01
        )

    def test_retries_until_success(self, fast_policy: RetryPolicySettings) -> None:
        calls = {"count": 0}

        @azure_retry(fast_policy)
        def flaky() -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise http_error(503)
            return "ok"

        assert flaky() == "ok"
        assert calls["count"] == 3

    def test_gives_up_after_max_attempts(
        self, fast_policy: RetryPolicySettings
    ) -> None:
        calls = {"count": 0}

        @azure_retry(fast_policy)
        def always_fails() -> None:
            calls["count"] += 1
            raise http_error(500)

        with pytest.raises(HttpResponseError):
            always_fails()
        assert calls["count"] == 3

    def test_reraises_the_original_azure_error(
        self, fast_policy: RetryPolicySettings
    ) -> None:
        """Callers should see the Azure error, not tenacity's wrapper."""

        @azure_retry(fast_policy)
        def always_fails() -> None:
            raise http_error(429)

        with pytest.raises(HttpResponseError) as exc_info:
            always_fails()
        assert exc_info.value.status_code == 429

    def test_does_not_retry_client_errors(
        self, fast_policy: RetryPolicySettings
    ) -> None:
        calls = {"count": 0}

        @azure_retry(fast_policy)
        def bad_request() -> None:
            calls["count"] += 1
            raise http_error(400)

        with pytest.raises(HttpResponseError):
            bad_request()
        assert calls["count"] == 1, "a 400 will fail the same way every time"

    def test_single_attempt_policy_disables_retry(self) -> None:
        policy = RetryPolicySettings(max_attempts=1, base_delay_seconds=0.001)
        calls = {"count": 0}

        @azure_retry(policy)
        def always_fails() -> None:
            calls["count"] += 1
            raise http_error(503)

        with pytest.raises(HttpResponseError):
            always_fails()
        assert calls["count"] == 1
