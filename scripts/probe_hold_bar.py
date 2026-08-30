# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_hold_bar.py
# Regression: a stale Flash auto-hide timer must NOT hide a newer hold bar.

import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.indicator import Indicator

_user32 = ctypes.WinDLL("user32")

# previous utterance's flash with a 400ms auto-hide…
indicator = Indicator()
indicator.flash("已粘贴（本机拦截键盘注入）", 400)
# …user immediately starts the next hold
indicator.show("● 正在听…")
time.sleep(1.0)

visible = bool(_user32.IsWindowVisible(indicator.hwnd()))
print(f"bar visible after stale timer fired: {visible}")
print("VERDICT:", "OK" if visible else "KILLED_BY_STALE_TIMER")
indicator.quit()
sys.exit(0 if visible else 1)
