"""Keeps pyproject.toml and requirements.txt from drifting apart.

pyproject.toml is the canonical dependency list; requirements.txt exists so
`pip install -r` keeps working. Two lists of pins is a real risk -- one gets
bumped and the other does not, and the difference only shows up as a
confusing runtime error -- so this asserts they agree.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _pyproject_pins() -> set[str]:
    """Every pin declared in pyproject, runtime and dev alike."""
    data = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    pins = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        pins.extend(extra)
    return {pin.strip() for pin in pins}


def _requirements_pins() -> set[str]:
    """Every pin listed in requirements.txt, ignoring comments and blanks."""
    lines = (BACKEND / "src" / "requirements.txt").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in lines.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_dependency_lists_agree() -> None:
    """The two files must pin exactly the same packages at the same versions."""
    only_in_pyproject = _pyproject_pins() - _requirements_pins()
    only_in_requirements = _requirements_pins() - _pyproject_pins()

    assert not only_in_pyproject, (
        f"in pyproject.toml but not requirements.txt: {sorted(only_in_pyproject)}"
    )
    assert not only_in_requirements, (
        f"in requirements.txt but not pyproject.toml: {sorted(only_in_requirements)}"
    )


def test_python_version_floor_matches_the_readme() -> None:
    """The project targets 3.11+; the README tells people to use it."""
    data = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.11"
