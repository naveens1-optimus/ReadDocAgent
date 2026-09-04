"""Application logging configuration.

Provides a single ``configure_logging`` entry point plus a correlation-ID
mechanism so that every log line emitted while processing one document can be
tied back to that document -- across all four agents, the retry wrapper and
the API layer.

The correlation ID lives in a :class:`~contextvars.ContextVar`, which means it
propagates correctly through ``async`` code (FastAPI request handlers) and is
isolated per task, unlike a module-level global.

Two output formats are supported, selected by ``LOG_FORMAT``:

* ``text`` -- human-readable, for local development.
* ``json`` -- one JSON object per line, for log aggregators.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

__all__ = [
    "LogFormat",
    "configure_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "correlation_id_scope",
]

LogFormat = Literal["text", "json"]

#: Placeholder used when no correlation ID has been set.
_NO_CORRELATION_ID = "-"

#: Per-task correlation identifier, injected into every log record.
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=_NO_CORRELATION_ID
)

#: Guards against duplicate handler installation (which would double-log).
_configured: bool = False

#: Third-party loggers that are far too chatty at DEBUG/INFO. Azure's HTTP
#: policy logger in particular dumps full request/response headers, which can
#: include credential material -- so it is pinned to WARNING.
_NOISY_LOGGERS: dict[str, int] = {
    "azure": logging.WARNING,
    "azure.core.pipeline.policies.http_logging_policy": logging.WARNING,
    "azure.identity": logging.WARNING,
    "urllib3": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "openai": logging.WARNING,
    "PIL": logging.WARNING,
}

#: Attributes present on every ``LogRecord``; anything else was supplied by the
#: caller via ``extra=`` and is therefore worth emitting in JSON output.
_STANDARD_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "stacklevel", "thread", "threadName",
        "taskName", "correlation_id",
    }
)


def get_correlation_id() -> str:
    """Return the current correlation ID, or ``"-"`` when none is set."""
    return _correlation_id.get()


def set_correlation_id(value: str | None) -> contextvars.Token[str]:
    """Set the correlation ID for the current context.

    Returns:
        A token that can be passed to ``ContextVar.reset`` to restore the
        previous value. Prefer :func:`correlation_id_scope` for scoped use.
    """
    return _correlation_id.set(value or _NO_CORRELATION_ID)


@contextmanager
def correlation_id_scope(value: str | None) -> Iterator[str]:
    """Bind a correlation ID for the duration of a ``with`` block.

    Example::

        with correlation_id_scope(document_id):
            logger.info("Classifying document")   # tagged with document_id
    """
    token = set_correlation_id(value)
    try:
        yield get_correlation_id()
    finally:
        _correlation_id.reset(token)


class _CorrelationIdFilter(logging.Filter):
    """Attach the current correlation ID to every record.

    Implemented as a filter rather than baked into the formatter so that both
    formatters -- and any handler added later -- see the attribute.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = get_correlation_id()
        return True


class _JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object.

    Any keyword passed through ``extra=`` is promoted to a top-level field,
    which keeps structured context (document type, confidence, model ID)
    queryable in a log aggregator.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", _NO_CORRELATION_ID),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_formatter(log_format: LogFormat) -> logging.Formatter:
    """Return the formatter matching the requested output format."""
    if log_format == "json":
        return _JsonFormatter()
    return logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(correlation_id)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging(
    level: str | int = "INFO",
    log_format: LogFormat = "text",
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for the application.

    Safe to call more than once: subsequent calls are ignored unless ``force``
    is set. This matters because both the FastAPI lifespan and the standalone
    demo scripts call it, and re-running it would attach a second handler and
    duplicate every line.

    Args:
        level: Log level name or numeric value.
        log_format: ``"text"`` or ``"json"``.
        force: Tear down existing handlers and reconfigure.

    Returns:
        The configured root logger.
    """
    global _configured

    root = logging.getLogger()

    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # stdout, not stderr: application logs are not errors, and this keeps
    # container log collection simple.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_build_formatter(log_format))
    handler.addFilter(_CorrelationIdFilter())

    resolved_level = (
        logging.getLevelName(level.upper()) if isinstance(level, str) else level
    )
    if not isinstance(resolved_level, int):
        # getLevelName returns "Level <name>" for unknown names.
        resolved_level = logging.INFO

    root.addHandler(handler)
    root.setLevel(resolved_level)

    for logger_name, noisy_level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(noisy_level)

    _configured = True
    root.info(
        "Logging configured (level=%s, format=%s)",
        logging.getLevelName(resolved_level),
        log_format,
    )
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Thin wrapper for a consistent import."""
    return logging.getLogger(name)
