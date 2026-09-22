# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_focus.py
# Verifies: bar visible, bottom-centre, no focus theft (incl. live breath).

import ctypes
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.indicator import Indicator

from app import winio

_user32 = ctypes.WinDLL("user32")

# Visual tolerances for "close enough to the bottom centre" (px).
_CENTER_TOLERANCE_PX = 200
_BOTTOM_BAND_PX = 160
_MIN_BAR_WIDTH_PX = 60


class _RECT(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


root = tk.Tk()
root.title("FOCUSPROBE")
root.geometry("300x120+120+120")
root.deiconify()
root.update()
time.sleep(0.3)

before = winio.foreground_window_title()
indicator = Indicator()
time.sleep(0.3)
indicator.show("● 正在听…")
indicator.update("识别中…")
time.sleep(0.4)
after = winio.foreground_window_title()

# Live mode: the pulse loop must not steal focus or hide the bar.
indicator.listen("● 聆听中")
time.sleep(0.5)
after_live = winio.foreground_window_title()

rect = _RECT()
_user32.GetWindowRect(indicator.hwnd(), ctypes.byref(rect))
visible = bool(_user32.IsWindowVisible(indicator.hwnd()))
left, _top, right, bottom = winio.active_monitor_work_area()
width = rect.right - rect.left
height = rect.bottom - rect.top
bar_center_x = (rect.left + rect.right) // 2
screen_center_x = (left + right) // 2
centered_x = abs(bar_center_x - screen_center_x) < _CENTER_TOLERANCE_PX
near_bottom = (bottom - rect.bottom) < _BOTTOM_BAND_PX

print(f"before={before!r}")
print(f"after={after!r}")
print(f"after_live={after_live!r}")
print(f"visible={visible} rect=({rect.left},{rect.top},{rect.right},{rect.bottom})")
print(f"centered_x={centered_x} near_bottom={near_bottom}")

ok = (
    before == after == after_live
    and visible
    and centered_x
    and near_bottom
    and width > _MIN_BAR_WIDTH_PX
)
print("VERDICT:", "OK" if ok else "BROKEN")
indicator.quit()
root.destroy()
sys.exit(0 if ok else 1)
