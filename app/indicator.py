"""Bottom-of-screen status bar ("正在听… / 识别中… / 下载中…") on a Tk thread.

Tk must own exactly one thread; every other thread talks to it through a
command queue polled by ``after``. The window is borderless and topmost.
"""

import ctypes
import queue
import threading
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from functools import partial
from typing import assert_never

from app import winio

_user32 = ctypes.WinDLL("user32")
_GWL_EXSTYLE: int = -20
_WS_EX_TOPMOST: int = 0x00000008
_WS_EX_TOOLWINDOW: int = 0x00000080
_WS_EX_NOACTIVATE: int = 0x08000000
_SWP_NOACTIVATE: int = 0x0010
_SWP_SHOWWINDOW: int = 0x0040
_HWND_TOPMOST = ctypes.c_void_p(-1)
_user32.SetWindowPos.restype = wintypes.BOOL
_user32.SetWindowPos.argtypes = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
)

_POLL_MS: int = 60


@dataclass(frozen=True, slots=True)
class Show:
    text: str


@dataclass(frozen=True, slots=True)
class Update:
    text: str


@dataclass(frozen=True, slots=True)
class Progress:
    pct: float  # 0.0 .. 1.0
    text: str


@dataclass(frozen=True, slots=True)
class Hide:
    pass


@dataclass(frozen=True, slots=True)
class Flash:
    """Show a message, then auto-hide after the given milliseconds."""

    text: str
    ms: int


@dataclass(frozen=True, slots=True)
class Quit:
    pass


type Command = Show | Update | Progress | Hide | Flash | Quit

_BG: str = "#101418"
_FG: str = "#e8eaed"
_ACCENT: str = "#7dd3fc"


class Indicator:
    """Thread-safe façade over the Tk status bar."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Command] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="indicator")
        self._started = threading.Event()
        self._generation = 0  # Tk-thread-only: invalidates stale auto-hide timers
        self._hwnd = 0
        self._thread.start()
        self._started.wait(timeout=5)

    def show(self, text: str) -> None:
        self._queue.put(Show(text=text))

    def update(self, text: str) -> None:
        self._queue.put(Update(text=text))

    def progress(self, pct: float, text: str) -> None:
        self._queue.put(Progress(pct=pct, text=text))

    def hide(self) -> None:
        self._queue.put(Hide())

    def flash(self, text: str, ms: int = 1500) -> None:
        self._queue.put(Flash(text=text, ms=ms))

    def quit(self) -> None:
        self._queue.put(Quit())
        self._thread.join(timeout=2)

    def hwnd(self) -> int:
        """Top-level Win32 handle of the bar (observability/tests)."""
        return self._hwnd

    # -- Tk thread ---------------------------------------------------------

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        # Harden BEFORE anything can map or activate the window: NOACTIVATE
        # blocks focus theft (paste would land on this bar otherwise),
        # TOOLWINDOW keeps it out of the taskbar, TOPMOST replaces Tk's own
        # "-topmost" attribute (whose SetWindowPos path can activate).
        self._root.update_idletasks()
        self._hwnd = _user32.GetParent(self._root.winfo_id()) or self._root.winfo_id()
        style = _user32.GetWindowLongW(self._hwnd, _GWL_EXSTYLE)
        _user32.SetWindowLongW(
            self._hwnd,
            _GWL_EXSTYLE,
            style | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW | _WS_EX_TOPMOST,
        )
        self._label = tk.Label(
            self._root,
            text="",
            font=("Microsoft YaHei UI", 11),
            fg=_FG,
            bg=_BG,
            padx=18,
            pady=7,
        )
        self._label.pack(anchor="center")
        self._root.withdraw()
        self._started.set()
        self._root.after(_POLL_MS, self._drain)
        self._root.mainloop()

    def _drain(self) -> None:
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                break
            match command:
                case Show(text=text):
                    self._generation += 1  # invalidate all pending auto-hide timers
                    self._label.configure(text=text, fg=_FG)
                    self._place()
                case Update(text=text):
                    self._label.configure(text=text)
                    if not self._root.winfo_viewable():
                        self._place()
                case Progress(pct=pct, text=text):
                    self._generation += 1
                    self._label.configure(text=f"{text} {pct * 100:.0f}%", fg=_ACCENT)
                    self._place()
                case Hide():
                    self._root.withdraw()
                case Flash(text=text, ms=ms):
                    self._generation += 1
                    generation = self._generation
                    self._label.configure(text=text, fg=_FG)
                    self._place()
                    self._root.after(ms, partial(self._hide_if_stale, generation))
                case Quit():
                    self._root.destroy()
                    return
                case unreachable:
                    assert_never(unreachable)
        self._root.after(_POLL_MS, self._drain)

    def _hide_if_stale(self, generation: int) -> None:
        """Auto-hide only when nothing newer has been shown since scheduling."""
        if generation == self._generation and self._root.winfo_viewable():
            self._root.withdraw()

    def _place(self) -> None:
        """Show on the ACTIVE monitor, bottom-center, pinned topmost.

        Re-asserts HWND_TOPMOST on every show: borderless-fullscreen apps
        (terminal F11 etc.) otherwise stack above a bar whose z-order was
        never re-asserted. Never activates (focus stays on the user's app).
        """
        self._root.update_idletasks()
        width = self._root.winfo_reqwidth()
        height = self._root.winfo_reqheight()
        left, _top, right, bottom = winio.active_monitor_work_area()
        x = left + max((right - left) - width, 0) // 2
        y = bottom - 96
        _user32.SetWindowPos(
            self._hwnd,
            _HWND_TOPMOST,
            x,
            y,
            width,
            height,
            _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
        )
