# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_emit_chain.py
# E2E: exactly the app's _emit_text decision chain against a live EDIT control.

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import winio

_user32 = ctypes.WinDLL("user32")
_kernel32 = ctypes.WinDLL("kernel32")
_user32.CreateWindowExW.restype = ctypes.c_void_p


class _MSG(ctypes.Structure):
    _fields_ = (
        ("hwnd", ctypes.c_void_p),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    )


def pump(seconds: float) -> None:
    msg = _MSG()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))
        time.sleep(0.01)


hwnd = _user32.CreateWindowExW(
    0, "EDIT", "CHAINPROBE", 0x00CF0000 | 0x10000000,
    200, 200, 420, 90, None, None, None, None,
)
pump(0.3)
fg = _user32.GetForegroundWindow()
our_tid = _kernel32.GetCurrentThreadId()
fg_tid = _user32.GetWindowThreadProcessId(fg, None)
if fg_tid != our_tid:
    _user32.AttachThreadInput(our_tid, fg_tid, True)
_user32.SetForegroundWindow(hwnd)
_user32.SetFocus(hwnd)
if fg_tid != our_tid:
    _user32.AttachThreadInput(our_tid, fg_tid, False)
pump(0.3)

alive = winio.keyboard_injection_alive()
print("injection alive:", alive)
channel = "keys"
winio.set_clipboard_text("CHAIN_OK")
if alive:
    winio.paste_text("CHAIN_OK", restore_clipboard=False, delay_ms=0)
else:
    posted = winio.post_wm_paste_to_focus()
    channel = f"wm_paste(posted={posted})"
pump(0.4)
buf = ctypes.create_unicode_buffer(256)
_user32.SendMessageW(hwnd, 13, 255, buf)
print(f"channel={channel} content={buf.value!r}")
_user32.DestroyWindow(hwnd)
sys.exit(0 if "CHAIN_OK" in buf.value else 1)
