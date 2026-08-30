# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_notepad.py
# Real-notepad injection target: letter key, unicode typing, clipboard paste.

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import winio

_user32 = ctypes.WinDLL("user32")
_kernel32 = ctypes.WinDLL("kernel32")
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowThreadProcessId.argtypes = (ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
_user32.FindWindowW.restype = ctypes.c_void_p
_user32.FindWindowExW.restype = ctypes.c_void_p
_user32.SendMessageW.restype = ctypes.c_ssize_t


def force_foreground(hwnd: int, focus_hwnd: int) -> None:
    fg = _user32.GetForegroundWindow()
    fg_tid = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
    our_tid = _kernel32.GetCurrentThreadId()
    if fg_tid and fg_tid != our_tid:
        _user32.AttachThreadInput(our_tid, fg_tid, True)
    _user32.SetForegroundWindow(hwnd)
    _user32.SetFocus(focus_hwnd)
    if fg_tid and fg_tid != our_tid:
        _user32.AttachThreadInput(our_tid, fg_tid, False)


def edit_text(edit: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    _user32.SendMessageW(edit, 13, 511, buf)  # WM_GETTEXT
    return buf.value


subprocess.Popen(["notepad.exe"])  # noqa: S607 — system notepad, PATH lookup intended
time.sleep(1.5)
hwnd = _user32.FindWindowW("Notepad", None)
if not hwnd:
    print("classic notepad window not found")
    sys.exit(2)
edit = _user32.FindWindowExW(hwnd, None, "Edit", None)
print(f"hwnd={hwnd} edit={edit} title={winio.foreground_window_title()!r}")
force_foreground(hwnd, edit)
time.sleep(0.3)
print("foreground:", repr(winio.foreground_window_title()))

winio.tap_key(0x41)  # letter A
time.sleep(0.3)
print("after A:", repr(edit_text(edit)))

winio.type_text("bc")
time.sleep(0.3)
print("after type_text bc:", repr(edit_text(edit)))

winio.paste_text("PASTED", restore_clipboard=False, delay_ms=0)
time.sleep(0.4)
print("after paste:", repr(edit_text(edit)))

_user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
sys.exit(0)
