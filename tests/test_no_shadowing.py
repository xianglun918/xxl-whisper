"""Guard: no class may give an attribute the same name as one of its methods.

``self._tray = Tray(...)`` silently shadowed the helper method ``_tray``, and
every menu click then raised ``TypeError: 'Tray' object is not callable``. Python
does not warn, the traceback points at the menu rather than at the clash, and no
unit test instantiates the GUI app — so the check has to be static.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_shadowing", REPO / "scripts" / "check_shadowing.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_attribute_shadows_a_method() -> None:
    checker = _checker()
    offenders: dict[str, set[str]] = {}
    for path in sorted((REPO / "app").glob("*.py")):
        for cls, names in checker.shadowing(path).items():
            offenders[f"{path.name}:{cls}"] = names
    assert not offenders, f"attribute/method name clashes: {offenders}"


def test_the_checker_actually_detects_a_clash(tmp_path: Path) -> None:
    """The check must fail on a known-bad sample, or it proves nothing."""
    checker = _checker()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "class Demo:\n"
        "    def __init__(self):\n"
        "        self._thing = 1\n"
        "    def _thing(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    assert checker.shadowing(sample) == {"Demo": {"_thing"}}
