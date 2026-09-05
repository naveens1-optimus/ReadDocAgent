"""Retry policy for Azure API calls: exponential backoff with jitter."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from domain.schema.settings import RetryPolicySettings
from infrastructure.utilities.logging_config import get_logger

__all__ = ["is_transient_error", "azure_retry"]

logger = get_logger(__name__)

#: Status codes worth retrying: throttling, request timeout and server errors.
#: A 4xx such as 401 or 400 means the call will fail again the same way, so
#: retrying only wastes time and quota.
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_transient_error(error: BaseException) -> bool:
    """Whether an exception is worth retrying."""
    if isinstance(error, (ServiceRequestError, ServiceResponseError)):
        return True  # network-level failure, never reached the service
    if isinstance(error, HttpResponseError):
        return error.status_code in _RETRYABLE_STATUS_CODES
    return False


def azure_retry(policy: RetryPolicySettings) -> Callable[..., Any]:
    """Build a retry decorator from the configured policy.

    ``reraise=True`` so callers see the original Azure exception rather than
    tenacity's ``RetryError`` wrapper.
    """
    return retry(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential_jitter(
            initial=policy.base_delay_seconds,
            max=policy.max_delay_seconds,
        ),
        retry=retry_if_exception(is_transient_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
