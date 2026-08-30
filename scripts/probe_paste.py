# ─── How to run ───
# .venv\Scripts\python.exe scripts\probe_paste.py
# Verifies clipboard+Ctrl+V delivery into a REAL foreground editable window.

import ctypes
import sys
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import winio

_user32 = ctypes.WinDLL("user32")
_kernel32 = ctypes.WinDLL("kernel32")
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_user32.GetWindowThreadProcessId.argtypes = (ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))


def force_foreground(hwnd: int, focus_hwnd: int) -> None:
    """AttachThreadInput dance so a background process may take foreground."""
    fg = _user32.GetForegroundWindow()
    fg_tid = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
    our_tid = _kernel32.GetCurrentThreadId()
    if fg_tid and fg_tid != our_tid:
        _user32.AttachThreadInput(our_tid, fg_tid, True)
    _user32.SetForegroundWindow(hwnd)
    _user32.SetFocus(focus_hwnd)
    if fg_tid and fg_tid != our_tid:
        _user32.AttachThreadInput(our_tid, fg_tid, False)


root = tk.Tk()
root.title("PASTEPROBE")
events: list[str] = []
entry = tk.Entry(root, width=60)
entry.pack(padx=20, pady=20)
root.bind("<KeyPress>", lambda e: events.append(f"key:{e.keysym}"))
root.bind("<Control-KeyPress>", lambda e: events.append(f"ctrlkey:{e.keysym}"))
root.geometry("420x80+200+200")
root.update()
time.sleep(0.3)

frame_hwnd = _user32.GetParent(root.winfo_id())
force_foreground(frame_hwnd, entry.winfo_id())
root.update()
time.sleep(0.4)
def pump(seconds: float) -> None:
    """Pump the Win32/Tk message queues so injected input gets processed."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update()
        time.sleep(0.02)


print("foreground:", repr(winio.foreground_window_title()))

events.clear()
entry.delete(0, "end")
root.update()
winio.paste_text("CLIPPASTE", restore_clipboard=False, delay_ms=0)
pump(0.5)
print(f"paste -> {entry.get()!r} events={events}")

events.clear()
entry.delete(0, "end")
root.update()
winio.type_text("TYPED456")
pump(0.5)
print(f"type -> {entry.get()!r} events={events}")

results = [entry.get() == "TYPED456"]
root.destroy()
sys.exit(0 if all(results) else 1)
