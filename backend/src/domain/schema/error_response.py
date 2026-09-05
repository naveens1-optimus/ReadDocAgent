"""Error response schema.

A single error shape for every failure the API returns, so clients can parse
one structure regardless of which handler produced it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ErrorResponse"]


class ErrorResponse(BaseModel):
    """Body returned by the API's exception handlers."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(
        description="Short machine-readable code, e.g. 'validation_error'."
    )
    message: str = Field(description="Human-readable explanation.")

    #: Echoed from the request's correlation ID so a client can quote it when
    #: reporting a problem, and it can be matched against the server logs.
    request_id: str | None = Field(default=None)

    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured context, e.g. field errors."
    )
