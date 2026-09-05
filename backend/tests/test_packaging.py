"""Keeps pyproject.toml and the requirements files from drifting apart.

``pyproject.toml`` at the repo root is the canonical dependency list. The two
requirements files exist so ``pip install -r`` keeps working:

* ``backend/src/requirements.txt`` -- what the API needs.
* ``frontend/requirements.txt``    -- what the Streamlit UI needs.

Three lists of pins is a real risk: one gets bumped and the others do not, and
the difference only shows up as a confusing runtime error. This asserts they
agree.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
BACKEND_REQUIREMENTS = ROOT / "backend" / "src" / "requirements.txt"
FRONTEND_REQUIREMENTS = ROOT / "frontend" / "requirements.txt"


def _pyproject_pins() -> set[str]:
    """Every pin declared in pyproject, runtime and extras alike."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    pins = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        pins.extend(extra)
    return {pin.strip() for pin in pins}


def _requirements_pins(*paths: Path) -> set[str]:
    """Every pin listed in the given files, ignoring comments and blanks."""
    pins: set[str] = set()
    for path in paths:
        pins |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return pins


def test_dependency_lists_agree() -> None:
    """pyproject must pin exactly what the two requirements files do."""
    declared = _pyproject_pins()
    installed = _requirements_pins(BACKEND_REQUIREMENTS, FRONTEND_REQUIREMENTS)

    only_in_pyproject = declared - installed
    only_in_requirements = installed - declared

    assert not only_in_pyproject, (
        f"in pyproject.toml but in neither requirements file: "
        f"{sorted(only_in_pyproject)}"
    )
    assert not only_in_requirements, (
        f"in a requirements file but not pyproject.toml: "
        f"{sorted(only_in_requirements)}"
    )


def test_frontend_and_backend_do_not_pin_the_same_package_differently() -> None:
    """A shared package pinned two ways would install unpredictably."""
    def by_name(path: Path) -> dict[str, str]:
        return {
            line.split("==")[0].strip(): line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }

    backend = by_name(BACKEND_REQUIREMENTS)
    frontend = by_name(FRONTEND_REQUIREMENTS)

    conflicts = {
        name: (backend[name], frontend[name])
        for name in set(backend) & set(frontend)
        if backend[name] != frontend[name]
    }
    assert not conflicts, f"conflicting pins: {conflicts}"


def test_python_version_floor_is_declared() -> None:
    """The project targets 3.11+; the README tells people to use it."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11"
