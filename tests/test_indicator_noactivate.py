"""Regression: the indicator bar must never steal keyboard focus.

2026-08-31 user report: recognition worked, paste executed, text landed
nowhere — the Tk indicator window took focus on show and swallowed Ctrl+V.
The window now carries WS_EX_NOACTIVATE and is shown via SW_SHOWNOACTIVATE.

Runs in a subprocess with its own foreground window, because in-process
foreground checks are polluted by the test runner's console.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_indicator_show_keeps_foreground_window() -> None:
    probe = Path(__file__).parents[1] / "scripts" / "probe_focus.py"
    result = subprocess.run(  # noqa: S603 — fixed interpreter + repo script
        [sys.executable, "-X", "utf8", str(probe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"indicator broken\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "VERDICT: OK" in result.stdout
