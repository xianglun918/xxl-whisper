"""Smoke test: the provisional (in-progress) subtitle line under Xvfb.

Renders two finished segments plus a live preview through the real Tk window, then
grabs a screenshot so the layout can be eyeballed. Run with::

    Xvfb :99 -screen 0 1280x800x24 &
    DISPLAY=:99 python scripts/partial_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.overlay import Overlay, OverlayStyle

OUT = Path(os.environ.get("SMOKE_OUT", "/tmp/partial_smoke.png"))


def main() -> int:
    """Build the bar, push two finished lines and one live preview."""
    style = OverlayStyle(
        mode="target",
        max_lines=4,
        hold_seconds=60.0,
        font_size=32,
        opacity=0.9,
        click_through=True,
    )
    overlay = Overlay(style)
    if not overlay.available:
        print("FAIL: overlay unavailable (no Tk / no DISPLAY)")
        return 1

    overlay.push("第一句已经说完了。", "The first sentence is done.")
    overlay.push("第二句也说完了。", "So is the second one.")
    time.sleep(0.4)
    overlay.push_provisional("第三句还在说，这里是实时预览", "The third sentence is still")
    time.sleep(0.6)

    subprocess.run(  # noqa: S603 - fixed argv, local debugging tool
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-video_size", "1280x800",
         "-i", os.environ.get("DISPLAY", ":99"), "-frames:v", "1", str(OUT)],
        check=False,
    )
    print(f"screenshot: {OUT}")

    overlay.clear_provisional()
    time.sleep(0.3)
    overlay.quit()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
