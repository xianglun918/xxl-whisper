"""Fail loudly when a lazily imported module would be missing from the frozen app.

PyInstaller finds imports by *reading the source*. Anything fetched through
``importlib.import_module("x")`` (used deliberately here for pystray, pyobjc and
uiautomation, so the module stays importable on a headless runner) is invisible
to it — unless the name is also listed in the build script. That is exactly how
0.8.0-beta.3 shipped an exe that died with:

    app\\tray.py, line 57: 系统托盘不可用：No module named 'pystray'

The tray was lazily imported, the name was never added to ``build.bat``, and
nothing verified the two lists agreed. Run this from the repo root:

    python scripts/check_bundled_imports.py     # exit 1 + a list on a gap
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
WIN_BUILD = ROOT / "build.bat"
MAC_SPEC = ROOT / "xxl-whisper-mac.spec"

_CALLS: frozenset[str] = frozenset({"import_module", "importlib.import_module", "__import__"})
#: Spec helpers that pull a whole package (and its submodules) into the bundle.
_COLLECTORS: frozenset[str] = frozenset({"collect_all", "collect_submodules"})

#: Modules only one platform's code path can reach, so only that build needs them.
#: Anything not listed here is assumed to be needed by both builds.
PLATFORM_SPECIFIC: dict[str, str] = {
    "AppKit": "darwin",
    "ApplicationServices": "darwin",
    "Foundation": "darwin",
    "PyObjCTools": "darwin",
    "Quartz": "darwin",
    "ServiceManagement": "darwin",
    "objc": "darwin",
    "soundcard": "win32",
    "uiautomation": "win32",
}


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _call_name(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    return None


def dynamic_imports(app_dir: Path = APP_DIR) -> dict[str, list[str]]:
    """Top-level modules imported only through ``import_module``/``__import__``."""
    found: dict[str, list[str]] = {}
    for path in sorted(app_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file is another test's job
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if _call_name(node.func) not in _CALLS:
                continue
            argument = node.args[0]
            if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                continue
            module = argument.value.strip().split(".")[0]
            if not module or module in sys.stdlib_module_names:
                continue
            where = str(path.relative_to(ROOT.parent)) if ROOT in path.parents else path.name
            found.setdefault(module, []).append(where)
    return found


def windows_bundle(build_file: Path = WIN_BUILD) -> set[str]:
    """``--hidden-import`` / ``--collect-all`` / ``--collect-submodules`` values."""
    text = build_file.read_text(encoding="utf-8")
    modules: set[str] = set()
    for match in re.finditer(r"--hidden-import[= ]([\w.]+)", text):
        modules.add(match.group(1).split(".")[0])
    for match in re.finditer(r"--collect-(?:all|submodules)[= ]([\w.]+)", text):
        modules.add(match.group(1).split(".")[0])
    return modules


def macos_bundle(spec_file: Path = MAC_SPEC) -> set[str]:
    """String literals the spec feeds to ``hiddenimports`` or ``collect_all``."""
    try:
        tree = ast.parse(spec_file.read_text(encoding="utf-8"), filename=str(spec_file))
    except SyntaxError:
        # PyInstaller specs are Python files; a spec we cannot parse is a real
        # problem, but reporting it as "nothing bundled" gives a clearer message.
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        modules |= _spec_strings(node)
    return modules


def _literal_strings(node: ast.AST) -> set[str]:
    """Top-level package names in one literal (str or list/tuple of strs)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value.split(".")[0]}
    if isinstance(node, ast.List | ast.Tuple):
        return {name for element in node.elts for name in _literal_strings(element)}
    return set()


def _spec_strings(node: ast.AST) -> set[str]:
    """What one spec node contributes to PyInstaller's collection."""
    found: set[str] = set()
    targets = getattr(node, "targets", []) if isinstance(node, ast.Assign) else []
    if any(isinstance(target, ast.Name) and target.id == "hiddenimports" for target in targets):
        found |= _literal_strings(node.value)  # type: ignore[attr-defined]
    if isinstance(node, ast.For):
        found |= _literal_strings(node.iter)
    if isinstance(node, ast.Call) and _call_name(node.func) in _COLLECTORS:
        for argument in node.args:
            found |= _literal_strings(argument)
    return found


def check(
    *,
    app_dir: Path = APP_DIR,
    build_file: Path = WIN_BUILD,
    spec_file: Path = MAC_SPEC,
) -> list[str]:
    """Return one human-readable line per module a frozen build would be missing."""
    windows = windows_bundle(build_file)
    macos = macos_bundle(spec_file)
    problems: list[str] = []
    for module, places in sorted(dynamic_imports(app_dir).items()):
        platform = PLATFORM_SPECIFIC.get(module)
        targets = ("win32", "darwin") if platform is None else (platform,)
        for target in targets:
            bundled = windows if target == "win32" else macos
            if module in bundled:
                continue
            script = "build.bat" if target == "win32" else "xxl-whisper-mac.spec"
            problems.append(
                f"{module}: 由 {'、'.join(sorted(set(places)))} 动态导入，"
                f"但 {script} 里没有 hidden-import/collect —— "
                f"打包后运行会 No module named '{module}'"
            )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("打包缺口（冻结后的程序会缺模块）：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {len(dynamic_imports())} 个动态导入模块在两份打包脚本里都有交代")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
