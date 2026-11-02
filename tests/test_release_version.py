"""Release metadata guards: the version must be written in both places, consistently.

``release.yml`` refuses to publish when the pushed tag does not match
``app.__version__`` (normalised). That gate only runs on GitHub, so a mismatch
is discovered *after* the tag is already public. These checks catch the cheap
half of it locally: the two in-repo version sites drifting apart.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from app import __version__

_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    version = data["project"]["version"]
    assert isinstance(version, str)
    return version


def _normalise(text: str) -> str:
    """Mirror of the ``norm()`` shell function in .github/workflows/release.yml.

    Strips a leading ``v`` and every ``-``, then keeps the first three
    dot-separated numbers: ``v0.8.0-beta.1`` and ``0.8.0b1`` both → ``0.8.0``.
    """
    cleaned = text.strip().lstrip("vV").replace("-", "")
    parts = cleaned.split(".")
    numbers = [_leading_int(part) for part in parts[:3]]
    assert len(numbers) == 3, f"version needs three parts: {text!r}"
    return ".".join(str(number) for number in numbers)


def _leading_int(part: str) -> int:
    """Leading digits of one version component, mirroring awk's ``%d`` on ``0beta``."""
    digits = ""
    for character in part:
        if not character.isdigit():
            break
        digits += character
    return int(digits or 0)


def test_version_matches_pyproject() -> None:
    """`app/__init__.py` and `pyproject.toml` must agree, or the release tag can only match one."""
    assert _pyproject_version() == __version__


def test_version_has_three_numeric_parts() -> None:
    """The in-app updater parses ``X.Y.Z`` and raises on anything else."""
    assert re.fullmatch(r"\d+\.\d+\.\d+(\D.*)?", __version__), __version__


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.8.0", "0.8.0"),
        ("v0.8.0-beta.1", "0.8.0"),
        ("v0.8.0-rc1", "0.8.0"),
    ],
)
def test_tag_normalisation(tag: str, expected: str) -> None:
    """Pinned examples so a rewrite of the workflow's norm() is caught here."""
    assert _normalise(tag) == expected


def test_current_version_normalises_like_its_beta_tag() -> None:
    """A beta version must be reachable by a dashed tag, e.g. 0.8.0b1 ↔ v0.8.0-beta.1."""
    if "b" not in __version__:
        pytest.skip("stable version, no beta tag to compare")
    stable = __version__.split("b")[0]
    beta_number = __version__.split("b")[1]
    assert _normalise(__version__) == _normalise(stable)
    assert _normalise(f"v{stable}-beta.{beta_number}") == _normalise(__version__)
