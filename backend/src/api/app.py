"""FastAPI application factory.

Assembles the HTTP layer: lifespan-managed configuration, correlation-ID and
request-logging middleware, uniform error handling, and the route modules.

Run it with either::

    python main.py
    uvicorn api.app:create_app --factory --reload

**Startup is deliberately tolerant of missing configuration.** If settings fail
to resolve, the app still starts: liveness (``/health``) stays green so the
process is diagnosable, while readiness (``/health/ready``) returns 503 naming
the variable that is missing. Hard-failing at startup would make the server
impossible to inspect at exactly the moment something is wrong with its
configuration.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.route import health
from domain.schema.error_response import ErrorResponse
from infrastructure.config.settings import get_settings
from infrastructure.utilities.logging_config import (
    configure_logging,
    get_logger,
    set_correlation_id,
)
from version import __version__

__all__ = ["create_app", "CorrelationIdMiddleware"]

logger = get_logger(__name__)

#: Request header carrying a client-supplied correlation ID, and the response
#: header it is echoed back on.
REQUEST_ID_HEADER = "X-Request-ID"

#: Paths excluded from per-request logging, to keep probe traffic from
#: drowning out real requests.
_QUIET_PATHS = frozenset({"/health", "/health/ready", "/favicon.ico"})


class CorrelationIdMiddleware:
    """Assign every request a correlation ID and log its outcome.

    Written as a pure ASGI middleware rather than a ``BaseHTTPMiddleware``
    subclass on purpose: ``BaseHTTPMiddleware`` dispatches through a separate
    task, which has historically broken :class:`~contextvars.ContextVar`
    propagation into the endpoint. A pure ASGI middleware runs in the same
    task, so the correlation ID set here is visible to every log line the
    handler emits.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex

        # Expose it on request.state for handlers and error responses.
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        token = set_correlation_id(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        path = scope.get("path", "")
        method = scope.get("method", "")

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            if path not in _QUIET_PATHS:
                logger.info(
                    "%s %s -> %s (%.1fms)",
                    method,
                    path,
                    status_code,
                    duration_ms,
                    extra={
                        "http_method": method,
                        "http_path": path,
                        "http_status": status_code,
                        "duration_ms": round(duration_ms, 2),
                    },
                )
            set_correlation_id(None)
            del token


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Resolve configuration on startup and record the outcome on app state.

    Configuration failure is captured rather than raised; see the module
    docstring for why.
    """
    # Bootstrap logging with defaults so that a configuration failure is
    # itself logged properly.
    configure_logging(force=True)

    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - recorded and surfaced by readiness
        app.state.settings = None
        app.state.settings_error = str(exc)
        logger.error(
            "Configuration failed to load; the server is running but NOT ready. %s",
            exc,
        )
    else:
        # Re-configure with the operator's chosen level and format.
        configure_logging(
            level=settings.app.log_level,
            log_format=settings.app.log_format,
            force=True,
        )
        app.state.settings = settings
        app.state.settings_error = None
        logger.info(
            "Configuration loaded\n%s",
            settings.summary(),
        )

    logger.info("Application startup complete (version %s)", __version__)
    yield
    logger.info("Application shutdown")


def _error_response(
    request: Request,
    *,
    status_code: int,
    error: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Render an :class:`ErrorResponse` as a JSON response."""
    body = ErrorResponse(
        error=error,
        message=message,
        request_id=getattr(request.state, "request_id", None),
        details=details or {},
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
    )


def _register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every failure returns the same JSON shape."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        return _error_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error="validation_error",
            message="The request payload failed validation.",
            # mode="json" on the model dump cannot reach these, so coerce any
            # non-serialisable values (e.g. exception instances) to strings.
            details={"errors": _jsonable_errors(exc.errors())},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            error=f"http_{exc.status_code}",
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback, but return a generic message: internal
        # details must not leak to the client.
        logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="internal_error",
            message="An unexpected error occurred. Quote the request_id when reporting this.",
        )


def _jsonable_errors(errors: list[Any]) -> list[dict[str, Any]]:
    """Coerce Pydantic error dicts into JSON-serialisable form."""
    cleaned: list[dict[str, Any]] = []
    for error in errors:
        if not isinstance(error, dict):
            cleaned.append({"detail": str(error)})
            continue
        cleaned.append(
            {
                key: (value if isinstance(value, (str, int, float, bool, type(None), list)) else str(value))
                for key, value in error.items()
                if key != "url"
            }
        )
    return cleaned


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Agent Document Processing API",
        description=(
            "Intelligent Document Processing pipeline built on LangGraph and "
            "Azure Document Intelligence. Classifies uploaded documents, "
            "extracts structured data, validates and enriches it, and stores "
            "the result."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(CorrelationIdMiddleware)
    _register_exception_handlers(app)

    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        """Send browsers to the interactive documentation."""
        return RedirectResponse(url="/docs")

    return app
