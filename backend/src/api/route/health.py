"""Health and readiness endpoints.

``/health`` answers "is the process alive?" and ``/health/ready`` answers "can
it actually do work?". Keeping them apart means a misconfigured deployment is
still reachable for diagnosis while an orchestrator correctly withholds
traffic from it.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from domain.schema.health_response import (
    ComponentStatus,
    HealthResponse,
    ReadinessResponse,
)
from infrastructure.utilities.logging_config import get_logger
from version import __version__

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 whenever the process is serving HTTP. Deliberately "
        "independent of configuration, so it stays available when the app is "
        "misconfigured."
    ),
)
async def health(request: Request) -> HealthResponse:
    """Report that the process is alive."""
    settings = getattr(request.app.state, "settings", None)
    return HealthResponse(
        version=__version__,
        environment=settings.app.environment if settings is not None else None,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Returns 200 when configuration has resolved and the service can do "
        "work, or 503 naming the component that is not ready."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "One or more components are not ready.",
        }
    },
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Report whether the service can serve real traffic.

    Currently the only gate is configuration. As Azure adapters are added they
    register further components here, so readiness reflects the whole
    dependency set rather than just the process.
    """
    settings = getattr(request.app.state, "settings", None)
    settings_error = getattr(request.app.state, "settings_error", None)

    components = [
        ComponentStatus(
            name="configuration",
            ready=settings is not None,
            detail=settings_error,
        )
    ]

    result = ReadinessResponse(
        ready=all(component.ready for component in components),
        version=__version__,
        components=components,
    )

    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning(
            "Readiness check failed: %s",
            ", ".join(component.name for component in result.not_ready),
        )

    return result
