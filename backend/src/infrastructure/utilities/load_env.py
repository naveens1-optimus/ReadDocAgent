"""Environment-variable loading utilities.

Single gateway through which the application reads environment configuration.
Keeping this in one place is what makes the "no hardcoded credentials" rule
enforceable: nothing else in the codebase calls ``os.environ`` for a secret.

Typical use::

    load_env()                                  # once, at process start
    key = load_required_env("AZURE_OPENAI_KEY")  # raises if absent
    level = get_env("LOG_LEVEL", "INFO")         # optional, with default
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "MissingEnvironmentVariableError",
    "load_env",
    "get_env",
    "load_required_env",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
]

logger = logging.getLogger(__name__)

#: Name of the dotenv file searched for, relative to ``backend/src``.
_ENV_FILENAME = ".env"

#: Values (lowercased) accepted as boolean true / false.
_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off"})


class MissingEnvironmentVariableError(RuntimeError):
    """Raised when a required environment variable is absent or blank.

    Carries an actionable message naming the variable and pointing at the
    template, so a misconfigured deployment fails loudly at startup rather
    than surfacing later as a confusing Azure authentication error.
    """

    def __init__(self, name: str) -> None:
        self.variable_name = name
        super().__init__(
            f"Required environment variable {name!r} is not set (or is empty). "
            f"Add it to backend/src/.env -- see .env.example for the expected "
            f"format. Never hardcode this value in source."
        )


def _default_env_path() -> Path:
    """Return the conventional ``backend/src/.env`` path.

    Resolved relative to this file (``infrastructure/utilities/load_env.py``),
    so it works regardless of the process working directory.
    """
    return Path(__file__).resolve().parents[2] / _ENV_FILENAME


def load_env(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load variables from a dotenv file into ``os.environ``.

    Args:
        path: Explicit dotenv file. Defaults to ``backend/src/.env``.
        override: When ``True``, dotenv values replace variables already
            present in the environment. Left ``False`` by default so that
            real deployment environments (App Service, container env vars,
            CI secrets) always win over a stray local file.

    Returns:
        The path loaded, or ``None`` when no dotenv file exists. A missing
        file is not an error -- production supplies configuration directly
        through the environment.
    """
    env_path = Path(path) if path is not None else _default_env_path()

    if not env_path.is_file():
        logger.info(
            "No dotenv file at %s; reading configuration from the process "
            "environment only.",
            env_path,
        )
        return None

    load_dotenv(dotenv_path=env_path, override=override)
    logger.info("Loaded environment configuration from %s", env_path)
    return env_path


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an optional environment variable.

    Surrounding whitespace is stripped and a blank value is treated as
    absent, so a leftover ``KEY=`` line in ``.env`` behaves like "not set"
    instead of silently supplying an empty credential.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def load_required_env(name: str) -> str:
    """Return a required environment variable.

    Raises:
        MissingEnvironmentVariableError: If unset or blank.
    """
    value = get_env(name)
    if value is None:
        raise MissingEnvironmentVariableError(name)
    return value


def get_env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean environment variable.

    Accepts ``1/true/t/yes/y/on`` and ``0/false/f/no/n/off``, case-insensitive.
    An unrecognised value logs a warning and falls back to ``default`` rather
    than crashing, since these flags are non-critical toggles.
    """
    value = get_env(name)
    if value is None:
        return default

    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False

    logger.warning(
        "Environment variable %s has unrecognised boolean value %r; "
        "falling back to %s.",
        name,
        value,
        default,
    )
    return default


def get_env_int(name: str, default: int) -> int:
    """Return an integer environment variable, falling back on a bad value."""
    value = get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Environment variable %s has non-integer value %r; falling back to %s.",
            name,
            value,
            default,
        )
        return default


def get_env_float(name: str, default: float) -> float:
    """Return a float environment variable, falling back on a bad value."""
    value = get_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Environment variable %s has non-float value %r; falling back to %s.",
            name,
            value,
            default,
        )
        return default
