"""One recorded step in the pipeline's audit trail."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from domain.enum.processing_status import AgentName

__all__ = ["AuditEntry"]


class AuditEntry(BaseModel):
    """What one node did. Entries are appended, never modified."""

    agent: AgentName
    action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def record(
        cls, agent: AgentName, action: str, **details: Any
    ) -> AuditEntry:
        """Record a successful step."""
        return cls(agent=agent, action=action, details=details)

    @classmethod
    def failure(
        cls, agent: AgentName, action: str, error: Exception | str
    ) -> AuditEntry:
        """Record a failed step."""
        return cls(
            agent=agent,
            action=action,
            error=str(error) or type(error).__name__,
            details={"error_type": type(error).__name__}
            if isinstance(error, Exception)
            else {},
        )

    def __str__(self) -> str:
        stamp = self.timestamp.strftime("%H:%M:%S")
        suffix = f" error={self.error}" if self.error else ""
        return f"{stamp} | {self.agent.value} | {self.action}{suffix}"
