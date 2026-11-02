"""The guard that would have caught the 0.8.0-beta.3 tray crash before shipping.

`app/tray.py` fetches pystray through ``importlib.import_module`` so the module
stays importable on a headless runner. PyInstaller cannot see that, and nothing
compared the dynamic imports against the two build scripts — so the released exe
died on start-up with "系统托盘不可用：No module named 'pystray'".

These tests pin the checker itself (including that it can fail), so the next
lazily imported module cannot slip out of a build unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.check_bundled_imports import (
    check,
    dynamic_imports,
    macos_bundle,
    unresolved_imports,
    windows_bundle,
)

REPO = Path(__file__).resolve().parent.parent


def test_repo_is_clean() -> None:
    """Every lazily imported module is listed in build.bat / the mac spec."""
    assert check() == []


def test_pystray_and_uiautomation_are_actually_bundled() -> None:
    """Both are importlib-only; a regression here is an exe that dies on start."""
    assert "pystray" in windows_bundle()
    assert "pystray" in macos_bundle()
    assert "uiautomation" in windows_bundle()


def test_detects_the_modules_we_expect(tmp_path: Path) -> None:
    found = dynamic_imports()
    assert "pystray" in found
    assert "uiautomation" in found
    # stdlib modules fetched the same way must not pollute the report
    assert not {"json", "os", "sys"} & found.keys()


def _fixture(tmp_path: Path, *, build: str, spec: str, app_source: str) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "mod.py").write_text(app_source, encoding="utf-8")
    (tmp_path / "build.bat").write_text(build, encoding="utf-8")
    (tmp_path / "xxl-whisper-mac.spec").write_text(spec, encoding="utf-8")
    return tmp_path


def test_flags_a_dynamic_import_missing_from_both_builds(tmp_path: Path) -> None:
    """The exact shape of the beta.3 bug: dynamic import, no hidden-import."""
    root = _fixture(
        tmp_path,
        build="pyinstaller --hidden-import=sounddevice run.py\n",
        spec="hiddenimports = []\n",
        app_source='import importlib\nm = importlib.import_module("pystray")\n',
    )

    problems = check(
        app_dir=root / "app",
        build_file=root / "build.bat",
        spec_file=root / "xxl-whisper-mac.spec",
    )

    assert len(problems) == 2, problems  # one per missing build script
    assert all("pystray" in problem for problem in problems)


def test_platform_specific_module_only_needs_its_own_build(tmp_path: Path) -> None:
    """Quartz is macOS-only: the Windows build must not be blamed for it."""
    root = _fixture(
        tmp_path,
        build="pyinstaller --hidden-import=sounddevice run.py\n",
        spec='for package in ("Quartz",):\n    pass\n',
        app_source='import importlib\nAppKit = importlib.import_module("Quartz")\n',
    )

    assert (
        check(
            app_dir=root / "app",
            build_file=root / "build.bat",
            spec_file=root / "xxl-whisper-mac.spec",
        )
        == []
    )


def test_passes_when_both_builds_list_the_module(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        build="pyinstaller --hidden-import=pystray run.py\n",
        spec='hiddenimports = ["pystray"]\n',
        app_source='import importlib\nm = importlib.import_module("pystray")\n',
    )

    assert (
        check(
            app_dir=root / "app",
            build_file=root / "build.bat",
            spec_file=root / "xxl-whisper-mac.spec",
        )
        == []
    )


def test_stdlib_dynamic_imports_are_ignored(tmp_path: Path) -> None:
    root = _fixture(
        tmp_path,
        build="pyinstaller run.py\n",
        spec="hiddenimports = []\n",
        app_source='import importlib\nm = importlib.import_module("json")\n',
    )

    assert "json" not in sys.stdlib_module_names or (
        check(
            app_dir=root / "app",
            build_file=root / "build.bat",
            spec_file=root / "xxl-whisper-mac.spec",
        )
        == []
    )


def test_an_alias_import_is_still_checked(tmp_path: Path) -> None:
    """`from importlib import import_module as im` must not slip past the checker."""
    root = _fixture(
        tmp_path,
        build="pyinstaller run.py\n",
        spec="hiddenimports = []\n",
        app_source=(
            "from importlib import import_module as im\n"
            'm = im("pystray")\n'
        ),
    )

    problems = check(
        app_dir=root / "app",
        build_file=root / "build.bat",
        spec_file=root / "xxl-whisper-mac.spec",
    )

    assert any("pystray" in problem for problem in problems), problems


def test_a_non_literal_import_fails_instead_of_being_ignored(tmp_path: Path) -> None:
    """看不懂的动态导入必须失败（fail-closed），否则检查器就是个安慰剂。"""
    root = _fixture(
        tmp_path,
        build="pyinstaller --hidden-import=pystray run.py\n",
        spec='hiddenimports = ["pystray"]\n',
        app_source=(
            "import importlib\n"
            "NAMES = ('pystray',)\n"
            "for name in NAMES:\n"
            "    importlib.import_module(name)\n"
        ),
    )

    unresolved = unresolved_imports(root / "app")

    assert len(unresolved) == 1
    assert "不是字符串字面量" in unresolved[0]
    assert check(
        app_dir=root / "app",
        build_file=root / "build.bat",
        spec_file=root / "xxl-whisper-mac.spec",
    ) == unresolved


def test_spec_augassign_and_append_are_collected(tmp_path: Path) -> None:
    """`hiddenimports += [...]` / `.append("x")` 也是 spec 里常见写法。"""
    spec_text = (
        'hiddenimports = ["sounddevice"]\n'
        'hiddenimports += ["AppKit"]\n'
        'hiddenimports.append("Quartz")\n'
    )
    root = _fixture(
        tmp_path,
        build="pyinstaller run.py\n",
        spec=spec_text,
        app_source="import importlib\n",
    )

    modules = macos_bundle(root / "xxl-whisper-mac.spec")

    assert {"sounddevice", "AppKit", "Quartz"} <= modules
