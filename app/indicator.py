"""Right-edge status bar ("正在听… / 识别中… / 下载中…") on a Tk thread.

Tk must own exactly one thread; every other thread talks to it through a
command queue polled by ``after``. The window is borderless and topmost,
vertically centred against the active monitor's right edge. A sticky "live"
mode (continuous dictation) keeps the bar visible and pulses a cheap accent
shade from a Tk ``after`` loop.
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
_PULSE_MS: int = 350  # ~2.9 toggles/s: subtle, cheap, easy on the eye
_RIGHT_MARGIN: int = 24  # px between the pill and the work-area right edge


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
class Listen:
    """Enter sticky listening mode: stay visible with a repeating pulse."""

    text: str


@dataclass(frozen=True, slots=True)
class FlashListen:
    """Flash a message, then switch to sticky listening mode."""

    flash_text: str
    listen_text: str
    ms: int


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


type Command = Show | Update | Progress | Listen | FlashListen | Hide | Flash | Quit

_BG: str = "#101418"
_FG: str = "#e8eaed"
_ACCENT: str = "#7dd3fc"
_ACCENT_DIM: str = "#3d6b82"  # second pulse shade


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

    def listen(self, text: str) -> None:
        """Show a sticky, softly pulsing bar (continuous dictation)."""
        self._queue.put(Listen(text=text))

    def flash_listen(self, flash_text: str, listen_text: str, ms: int = 1500) -> None:
        """Flash a confirmation, then return to sticky listening mode."""
        self._queue.put(FlashListen(flash_text=flash_text, listen_text=listen_text, ms=ms))

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
        # Tk-thread-only live-mode state (never touched off this thread).
        self._live = False
        self._pulse_on = False
        self._pulse_job: str | None = None
        self._started.set()
        self._root.after(_POLL_MS, self._drain)
        self._root.mainloop()

    def _drain(self) -> None:
        """Tk-thread poll: apply every queued command, then reschedule."""
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                break
            if self._apply(command):
                return
        self._root.after(_POLL_MS, self._drain)

    def _apply(self, command: Command) -> bool:
        """Handle one command; returns True when the loop must stop (Quit)."""
        match command:
            case Show(text=text):
                self._generation += 1  # invalidate all pending auto-hide timers
                self._stop_pulse()
                self._label.configure(text=text, fg=_FG)
                self._place()
            case Update(text=text):
                self._update_text(text)
            case Progress(pct=pct, text=text):
                self._generation += 1
                self._stop_pulse()
                self._label.configure(text=f"{text} {pct * 100:.0f}%", fg=_ACCENT)
                self._place()
            case Listen(text=text):
                self._generation += 1
                self._start_live(text)
            case FlashListen(flash_text=flash_text, listen_text=listen_text, ms=ms):
                self._flash_listen(flash_text, listen_text, ms)
            case Hide():
                self._generation += 1
                self._stop_pulse()
                self._root.withdraw()
            case Flash(text=text, ms=ms):
                self._flash(text, ms)
            case Quit():
                self._stop_pulse()
                self._root.destroy()
                return True
            case unreachable:
                assert_never(unreachable)
        return False

    def _update_text(self, text: str) -> None:
        """Live mode owns the fg (the pulse), so refresh text without re-placing."""
        self._label.configure(text=text)
        if not self._root.winfo_viewable():
            self._place()

    def _flash_listen(self, flash_text: str, listen_text: str, ms: int) -> None:
        """Flash a message, then resume sticky listening unless superseded."""
        self._generation += 1
        generation = self._generation
        self._stop_pulse()
        self._label.configure(text=flash_text, fg=_FG)
        self._place()
        self._root.after(ms, partial(self._resume_listen, generation, listen_text))

    def _flash(self, text: str, ms: int) -> None:
        """Show a message, then auto-hide unless something newer appeared."""
        self._generation += 1
        generation = self._generation
        self._stop_pulse()
        self._label.configure(text=text, fg=_FG)
        self._place()
        self._root.after(ms, partial(self._hide_if_stale, generation))

    def _hide_if_stale(self, generation: int) -> None:
        """Auto-hide only when nothing newer has been shown since scheduling."""
        if generation == self._generation and self._root.winfo_viewable():
            self._root.withdraw()

    # -- live pulse (Tk thread only) ---------------------------------------

    def _start_live(self, text: str) -> None:
        """Enter sticky listening mode and start the accent pulse."""
        self._live = True
        self._pulse_on = True
        self._label.configure(text=text, fg=_ACCENT)
        self._place()
        self._schedule_pulse()

    def _resume_listen(self, generation: int, text: str) -> None:
        """Return to sticky listening after a flash, unless superseded."""
        if generation == self._generation:
            self._start_live(text)

    def _schedule_pulse(self) -> None:
        """(Re)arm the single pending pulse callback."""
        if self._pulse_job is not None:
            self._root.after_cancel(self._pulse_job)
        self._pulse_job = self._root.after(_PULSE_MS, self._pulse_tick)

    def _pulse_tick(self) -> None:
        """Toggle the accent shade; the entire animation is this one line."""
        self._pulse_job = None
        if not self._live:
            return
        self._pulse_on = not self._pulse_on
        self._label.configure(fg=_ACCENT if self._pulse_on else _ACCENT_DIM)
        self._pulse_job = self._root.after(_PULSE_MS, self._pulse_tick)

    def _stop_pulse(self) -> None:
        """Leave live mode and cancel any pending pulse callback (no leaks)."""
        self._live = False
        job = self._pulse_job
        self._pulse_job = None
        if job is not None:
            self._root.after_cancel(job)

    def _place(self) -> None:
        """Show on the ACTIVE monitor's right edge, vertically centred.

        Follows the foreground window's monitor (multi-monitor: the bar must
        appear on the screen the user is looking at), sizes explicitly in the
        same SetWindowPos call (a withdrawn Tk window has no reliable OS-side
        size), and bypasses Tk's deiconify path (focus theft). Suppresses
        itself entirely while an exclusive-fullscreen game or presentation
        owns the screen — never fight the game for the display.
        """
        if winio.exclusive_fullscreen_owner_active():
            self._root.withdraw()
            return
        self._root.update_idletasks()
        width = self._root.winfo_reqwidth()
        height = self._root.winfo_reqheight()
        left, top, right, bottom = winio.active_monitor_work_area()
        x = max(right - width - _RIGHT_MARGIN, left)
        y = top + max((bottom - top) - height, 0) // 2
        _user32.SetWindowPos(
            self._hwnd,
            _HWND_TOPMOST,
            x,
            y,
            width,
            height,
            _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
        )
