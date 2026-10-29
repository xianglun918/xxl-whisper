"""Find instance attributes that shadow methods of the same class.

`self._tray = Tray(...)` silently overwrote the helper method `_tray`, so every
menu click raised ``TypeError: 'Tray' object is not callable``. Python does not
warn about this, and the symptom points at the menu rather than at the name
clash, so it is worth a standing check.

Run directly for a report, or via pytest (``test_no_attribute_shadows_a_method``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def shadowing(path: Path) -> dict[str, set[str]]:
    """Return {class_name: {attribute names that clash with a method}}."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        attributes = {
            target.attr
            for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            for target in item.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        clashes = methods & attributes
        if clashes:
            found[node.name] = clashes
    return found


def main() -> int:
    """Report every class whose attributes shadow its own methods."""
    problems = 0
    for path in sorted((REPO / "app").glob("*.py")):
        for cls, names in shadowing(path).items():
            problems += 1
            print(f"{path.relative_to(REPO)}: {cls} shadows {sorted(names)}")
    if not problems:
        print("no attribute/method shadowing found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
