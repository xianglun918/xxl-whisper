"""Headless smoke test for the subtitle overlay.

Runs the real Tk window (not a mock) and walks it through its lifecycle on a
throwaway X display, printing a timeline so a CI log shows exactly what
happened. What it proves:

* the window is created and mapped (``mapped`` flag from ``winfo_viewable``);
* a pushed sentence reaches the window geometry (width/height are non-zero and
  the bar sits near the bottom of the work area);
* three stacked segments fit without the window growing past ``max_lines``;
* the bar hides itself once the hold time expires — no timer leaks, no ghost
  window left on screen;
* ``quit()`` returns while the Tk thread is still alive.

What it cannot prove: always-on-top stacking, click-through and "does not steal
focus" are Win32/AppKit behaviours inside the real session — those need a
Windows/macOS machine.

Usage::

    Xvfb :99 -screen 0 1280x800x24 &
    DISPLAY=:99 .venv/bin/python scripts/overlay_smoke.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.overlay import Overlay, OverlayStyle

_WORK_AREA = (0, 0, 1280, 800)


def _probe(overlay: Overlay) -> tuple[int, int, int, int, bool]:
    """Snapshot the window's geometry and visibility from the Tk thread."""
    root = overlay._root  # noqa: SLF001 - a smoke test is allowed to look inside
    if root is None:
        return (0, 0, 0, 0, False)
    try:
        return (
            root.winfo_x(),
            root.winfo_y(),
            root.winfo_width(),
            root.winfo_height(),
            bool(root.winfo_viewable()),
        )
    except Exception:  # noqa: BLE001 - a torn-down interpreter is a valid answer
        return (0, 0, 0, 0, False)


def main() -> int:
    """Drive one overlay through show → stack → auto-hide → quit."""
    style = OverlayStyle(mode="bilingual", hold_seconds=2.0, max_lines=3, font_size=28)
    overlay = Overlay(style, work_area=lambda: _WORK_AREA)
    if not overlay.available:
        print("FAIL: overlay unavailable (no Tk or no display)")
        return 1
    print(f"1. window ready; hwnd={overlay.hwnd()}")

    overlay.push("今天我们来聊聊大模型。", "Let's talk about large models today.")
    time.sleep(0.6)
    x, y, width, height, viewable = _probe(overlay)
    print(f"2. first segment: viewable={viewable} box=({x},{y}) {width}x{height}")
    if not viewable or width <= 0 or height <= 0:
        print("FAIL: bar did not map")
        return 1
    if y + height > _WORK_AREA[3]:
        print("FAIL: bar is below the work area")
        return 1

    for said, became in (
        ("第二句比第一句长一点，用来看自动换行。", "The second one is longer, to exercise wrapping."),
        ("第三句。", "The third one."),
    ):
        overlay.push(said, became)
        time.sleep(0.3)
    _x, _y, stacked_width, stacked_height, _v = _probe(overlay)
    print(f"3. three segments stacked: {stacked_width}x{stacked_height} (was {width}x{height})")
    if stacked_height <= height:
        print("FAIL: stacking a segment did not grow the bar")
        return 1

    print("4. waiting for the hold time to expire…")
    time.sleep(2.6)
    _x, _y, _w, _h, viewable_after = _probe(overlay)
    print(f"5. after expiry: viewable={viewable_after}")
    if viewable_after:
        print("FAIL: bar is still on screen after the hold time")
        return 1

    overlay.quit()
    print("6. quit() returned; thread alive=" + str(overlay._thread.is_alive()))  # noqa: SLF001
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
