# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_focus.py
# Verifies: bar visible, bottom-center of the active monitor, no focus theft.

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

# Visual tolerances for "close enough to center/bottom" (px).
_CENTER_TOLERANCE_PX = 200
_BOTTOM_BAND_PX = 300
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

rect = _RECT()
_user32.GetWindowRect(indicator.hwnd(), ctypes.byref(rect))
visible = bool(_user32.IsWindowVisible(indicator.hwnd()))
screen_h = root.winfo_screenheight()
screen_w = root.winfo_screenwidth()
width = rect.right - rect.left
centered = abs((rect.left + width // 2) - screen_w // 2) < _CENTER_TOLERANCE_PX
near_bottom = screen_h - rect.bottom < _BOTTOM_BAND_PX

print(f"before={before!r}")
print(f"after={after!r}")
print(f"visible={visible} rect=({rect.left},{rect.top},{rect.right},{rect.bottom})")
print(f"centered={centered} near_bottom={near_bottom}")

ok = before == after and visible and centered and near_bottom and width > _MIN_BAR_WIDTH_PX
print("VERDICT:", "OK" if ok else "BROKEN")
indicator.quit()
root.destroy()
sys.exit(0 if ok else 1)
