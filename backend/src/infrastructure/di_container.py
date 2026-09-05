"""A small service container.

Services are built once at startup and looked up by their interface, which
keeps the API layer unaware of the concrete Azure classes and lets tests
register fakes instead.
"""

from __future__ import annotations

from typing import Any, TypeVar

from infrastructure.utilities.logging_config import get_logger

__all__ = ["DIContainer"]

logger = get_logger(__name__)

T = TypeVar("T")


class DIContainer:
    """Holds one instance per registered type."""

    def __init__(self) -> None:
        self._services: dict[type, Any] = {}

    def register(self, key: type[T], instance: T) -> None:
        """Register an instance under a type, usually its interface."""
        self._services[key] = instance
        logger.debug("Registered %s", key.__name__)

    def resolve(self, key: type[T]) -> T:
        """Return the instance registered for a type.

        Raises:
            KeyError: If nothing is registered for it.
        """
        try:
            return self._services[key]
        except KeyError:
            raise KeyError(
                f"No service registered for {key.__name__}. "
                "Was register_services() called at startup?"
            ) from None

    def __contains__(self, key: type) -> bool:
        return key in self._services
