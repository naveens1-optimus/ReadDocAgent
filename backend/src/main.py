"""Development entry point for the API server.

    cd backend/src
    python main.py

Equivalent to ``uvicorn api.app:create_app --factory``, but reads host, port
and reload settings from the environment so there is one obvious way to start
the server locally.

The app is passed to uvicorn as an import string with ``factory=True`` rather
than as an instance, because ``reload=True`` requires uvicorn to be able to
re-import the application in a fresh worker process.
"""

from __future__ import annotations

import uvicorn

from infrastructure.utilities.load_env import (
    get_env,
    get_env_bool,
    get_env_int,
    load_env,
)
from infrastructure.utilities.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Start the development server."""
    load_env()
    configure_logging(
        level=get_env("LOG_LEVEL", "INFO") or "INFO",
        log_format=(get_env("LOG_FORMAT", "text") or "text").lower(),  # type: ignore[arg-type]
    )

    host = get_env("API_HOST", "127.0.0.1") or "127.0.0.1"
    port = get_env_int("API_PORT", 8000)
    reload_enabled = get_env_bool("API_RELOAD", True)

    logger.info("Starting API server on http://%s:%s (reload=%s)", host, port, reload_enabled)
    logger.info("Interactive docs: http://%s:%s/docs", host, port)

    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload_enabled,
        log_config=None,  # keep our own logging configuration
    )


if __name__ == "__main__":
    main()
