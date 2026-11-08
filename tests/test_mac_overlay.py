"""macOS native subtitle overlay: interface parity + no-tkinter guarantee."""

import subprocess
import sys
from pathlib import Path

import pytest

from app import mac_overlay, overlay

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only (AppKit overlay)")


def test_reuses_shared_overlay_logic() -> None:
    """The native bar must reuse the platform-independent stacking/expiry model."""
    assert mac_overlay.OverlayModel is overlay.OverlayModel
    assert mac_overlay.build_style is overlay.build_style


def test_facade_matches_the_tk_overlay_interface() -> None:
    """app.py only uses this surface, so mac_overlay must expose all of it."""
    required = {
        "available",
        "hwnd",
        "push",
        "push_provisional",
        "clear_provisional",
        "clear",
        "quit",
    }
    tk_names = {name for name in dir(overlay.Overlay) if not name.startswith("_")}
    mac_names = {name for name in dir(mac_overlay.Overlay) if not name.startswith("_")}
    assert required <= tk_names
    assert required <= mac_names


def test_does_not_import_tkinter() -> None:
    """Hard constraint: the macOS path must never pull in tkinter."""
    code = "import sys; import app.mac_overlay; print('tkinter' in sys.modules)"
    result = subprocess.run(  # noqa: S603 - fixed interpreter + fixed -c code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_fade_scales_rgb_towards_black() -> None:
    """Older rows dim towards the background; the newest stays full contrast."""
    rgba = (0.5, 0.25, 0.125, 1.0)
    assert mac_overlay._fade(rgba, 0.5) == (0.25, 0.125, 0.0625, 1.0)
    assert mac_overlay._fade(rgba, 0.0) == (0.0, 0.0, 0.0, 1.0)
    assert mac_overlay._fade(rgba, 1.0) == rgba


def test_line_height_scales_with_font_size() -> None:
    assert mac_overlay._line_height(15.0) == pytest.approx(1.6 * 15.0 + 8.0)


def test_bar_width_respects_ratio_and_minimum() -> None:
    assert mac_overlay._bar_width((0, 0, 1000, 800), 0.72) == pytest.approx(720.0)
    assert mac_overlay._bar_width((0, 0, 100, 800), 0.72) == pytest.approx(240.0)
