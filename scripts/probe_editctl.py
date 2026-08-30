# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_editctl.py
# Decides: does SendInput (keys / unicode / clipboard+Ctrl+V) work AT ALL
# in this session? Target = raw Win32 EDIT control with a real message pump.

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
_user32.SetFocus.argtypes = (ctypes.c_void_p,)
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowThreadProcessId.argtypes = (ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))

_WS_OVERLAPPEDWINDOW = 0x00CF0000
_WS_VISIBLE = 0x10000000
_WS_CHILD = 0x40000000
_WS_BORDER = 0x00800000
_WM_GETTEXT = 13


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
    0, "EDIT", "EDITPROBE", _WS_OVERLAPPEDWINDOW | _WS_VISIBLE,
    200, 200, 420, 90, None, None, None, None,
)
if not hwnd:
    sys.exit(f"CreateWindowExW failed: {ctypes.get_last_error()}")
edit = hwnd  # EDIT as top-level window: it IS the control

fg = _user32.GetForegroundWindow()
fg_tid = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
our_tid = _kernel32.GetCurrentThreadId()
if fg_tid and fg_tid != our_tid:
    _user32.AttachThreadInput(our_tid, fg_tid, True)
_user32.SetForegroundWindow(hwnd)
_user32.SetFocus(edit)
if fg_tid and fg_tid != our_tid:
    _user32.AttachThreadInput(our_tid, fg_tid, False)
pump(0.3)
print("foreground:", repr(winio.foreground_window_title()))


def content() -> str:
    buf = ctypes.create_unicode_buffer(512)
    _user32.SendMessageW(edit, _WM_GETTEXT, 511, buf)
    return buf.value


winio.tap_key(0x41)  # 'A'
pump(0.3)
print("after tap A:", repr(content()))

winio.type_text("bc")
pump(0.3)
print("after type bc:", repr(content()))

winio.paste_text("PASTED", restore_clipboard=False, delay_ms=0)
pump(0.4)
print("after paste:", repr(content()))

_user32.DestroyWindow(hwnd)
