"""Health and readiness response schemas.

Liveness and readiness are deliberately separate concerns:

* **Liveness** (:class:`HealthResponse`) -- the process is running and can
  serve HTTP. It does not depend on configuration, so it stays available even
  when the app is misconfigured, which is exactly when it is most needed for
  diagnosis.
* **Readiness** (:class:`ReadinessResponse`) -- configuration resolved and the
  app can actually do work. An orchestrator should withhold traffic until this
  passes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ComponentStatus", "HealthResponse", "ReadinessResponse"]


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentStatus(_ResponseModel):
    """Readiness of one dependency."""

    name: str
    ready: bool
    detail: str | None = Field(
        default=None, description="Why it is not ready, when it is not."
    )


class HealthResponse(_ResponseModel):
    """Liveness probe response. Always ``200`` while the process is serving."""

    status: Literal["ok"] = "ok"
    service: str = "doc-int-agent-system"
    version: str
    environment: str | None = Field(
        default=None, description="Absent when configuration failed to load."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReadinessResponse(_ResponseModel):
    """Readiness probe response. ``200`` when ready, ``503`` when not."""

    ready: bool
    service: str = "doc-int-agent-system"
    version: str
    components: list[ComponentStatus] = Field(default_factory=list)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def not_ready(self) -> list[ComponentStatus]:
        """Components blocking readiness."""
        return [component for component in self.components if not component.ready]
