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

    Two gates: configuration resolved, and the Azure clients built. Both are
    local checks -- a probe must stay fast, so it deliberately does not call
    Azure. Whether the credentials actually work is proved by the first real
    upload.
    """
    state = request.app.state
    settings = getattr(state, "settings", None)
    container = getattr(state, "container", None)

    components = [
        ComponentStatus(
            name="configuration",
            ready=settings is not None,
            detail=getattr(state, "settings_error", None),
        ),
        ComponentStatus(
            name="azure_services",
            ready=container is not None,
            detail=getattr(state, "services_error", None)
            or (None if container is not None else "not built"),
        ),
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
