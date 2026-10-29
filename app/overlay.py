"""Bottom-of-screen subtitle overlay for translation output (上屏字幕).

This is the second output channel next to cursor injection: instead of (or in
addition to) typing the text into the focused app, each finished utterance is
pushed to a borderless, click-through, always-on-top bar pinned near the bottom
of the active screen, where it stays readable for a few seconds and then fades
away. It is useful for meetings and videos, where the cursor is not the target.

Design notes:

* **Pure logic first.** :class:`OverlayModel` owns the stacking and expiry
  rules and imports nothing platform-specific, so those rules are unit-tested
  with a fake clock instead of a real window.
* **One Tk thread, like the status bar.** :class:`Overlay` mirrors
  :mod:`app.indicator`: a command queue drained by ``after`` on the thread that
  owns the Tk interpreter. Other threads only ever enqueue commands.
* **Imports stay lazy.** ``tkinter`` and the Win32 style calls are imported
  inside the Tk thread, so importing this module (and the whole app) on a
  headless box, or where ``python3-tk`` is missing, never explodes — the
  overlay simply reports itself unavailable and the app carries on.
* **The window must never steal focus.** Pasted text has to land in the user's
  app, so the window is created ``overrideredirect`` and hardened with
  ``WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST`` before it is mapped;
  ``WS_EX_TRANSPARENT | WS_EX_LAYERED`` adds click-through on request.
* **Sentence granularity is honest.** The NMT engine is not streaming (one
  encoder/decoder pass per finished utterance), so the bar shows one finished
  sentence at a time — not word-by-word live captions.
"""

import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

__all__ = [
    "Line",
    "Overlay",
    "OverlayModel",
    "OverlayStyle",
    "Segment",
    "build_style",
]

_POLL_MS: int = 60
_TICK_MS: int = 250
_BG: str = "#101418"
_FG: str = "#f2f4f6"
_SOURCE_FG: str = "#9aa4ae"
#: In-progress line: brighter than the grey source rows, dimmer than a finished
#: translation, so the eye reads it as "not final yet".
_PARTIAL_FG: str = "#cfd6dd"
_MIN_WIDTH: int = 240
_FONT_FAMILIES: dict[str, str] = {
    "win32": "Microsoft YaHei UI",
    "darwin": "PingFang SC",
}
_DEFAULT_FONT_SIZE: int = 32

#: What the bar shows for one segment: only the translation, only what was
#: said, or both (source in a smaller grey line above the translation).
type OverlayMode = Literal["target", "source", "bilingual"]

#: Where the text goes: only to the bar, or to the bar *and* the cursor.
type OverlayOutput = Literal["type", "screen", "both"]


@dataclass(frozen=True, slots=True)
class OverlayStyle:
    """Everything about how the bar looks and how long a line lives."""

    mode: OverlayMode = "target"
    font_size: int = _DEFAULT_FONT_SIZE
    opacity: float = 0.85
    width_ratio: float = 0.72
    bottom_margin: int = 80
    hold_seconds: float = 4.0
    max_lines: int = 3
    click_through: bool = True


@dataclass(frozen=True, slots=True)
class Segment:
    """One finished utterance: what was said, what it became, when it showed."""

    source: str
    target: str
    shown_at: float


@dataclass(frozen=True, slots=True)
class Line:
    """One rendered row; ``kind`` picks the font (grey source vs bright target)."""

    kind: Literal["source", "target", "provisional"]
    text: str


class OverlayModel:
    """Stacking, de-duplication and expiry rules for the subtitle bar."""

    def __init__(
        self,
        style: OverlayStyle,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._style = style
        self._clock = clock
        self._segments: list[Segment] = []
        #: The in-progress utterance: replaced in place, never stacked, never expired.
        self._provisional: tuple[str, str] | None = None

    @property
    def style(self) -> OverlayStyle:
        """The style the model renders with."""
        return self._style

    def push(self, source: str, target: str) -> bool:
        """Add one utterance; returns True when the bar must be re-rendered.

        An immediate repeat of the previous utterance (the decoder re-emitting
        the same sentence) refreshes its hold time instead of stacking a
        duplicate line. Old segments are dropped once ``max_lines`` is
        exceeded, so the bar can never grow without bound.
        """
        said = source.strip()
        became = target.strip()
        if not said and not became:
            return False
        if (
            self._segments
            and self._segments[-1].source == said
            and self._segments[-1].target == became
        ):
            self._segments[-1] = Segment(source=said, target=became, shown_at=self._clock())
            return True
        self._segments.append(Segment(source=said, target=became, shown_at=self._clock()))
        overflow = len(self._segments) - max(1, self._style.max_lines)
        if overflow > 0:
            del self._segments[:overflow]
        return True

    def push_provisional(self, source: str, target: str) -> bool:
        """Show the utterance still being spoken; replaced in place each refresh.

        Provisional text is *not* a finished segment: it never enters
        ``_segments``, so it cannot be counted against ``max_lines`` or pushed
        out by the hold timer. It disappears the moment the real sentence lands.
        """
        said = source.strip()
        became = target.strip()
        if not said and not became:
            return False
        current = (said, became)
        if current == self._provisional:
            return False
        self._provisional = current
        return True

    def clear_provisional(self) -> bool:
        """Drop the in-progress line; True when there was one to drop."""
        if self._provisional is None:
            return False
        self._provisional = None
        return True

    def expire(self) -> bool:
        """Drop segments past their hold time; True when something went away."""
        now = self._clock()
        hold = self._style.hold_seconds
        kept = [seg for seg in self._segments if seg.shown_at + hold > now]
        if len(kept) == len(self._segments):
            return False
        self._segments = kept
        return True

    def clear(self) -> None:
        """Forget every segment and the provisional line."""
        self._segments.clear()
        self._provisional = None

    def lines(self) -> list[Line]:
        """Rows to draw, newest last, already filtered by the style's mode."""
        rows: list[Line] = []
        for seg in self._segments:
            if self._style.mode in ("source", "bilingual") and seg.source:
                rows.append(Line(kind="source", text=seg.source))
            if self._style.mode in ("target", "bilingual"):
                text = seg.target or seg.source  # failed MT must not blank the bar
                if text:
                    rows.append(Line(kind="target", text=text))
        if self._provisional is not None:
            said, became = self._provisional
            if self._style.mode in ("source", "bilingual") and said:
                rows.append(Line(kind="provisional", text=said))
            if self._style.mode in ("target", "bilingual"):
                text = became or said
                if text:
                    rows.append(Line(kind="provisional", text=text))
        return rows

    def is_empty(self) -> bool:
        """Return True when nothing is left to show."""
        return not self._segments and self._provisional is None


@dataclass(frozen=True, slots=True)
class _Push:
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class _PushPartial:
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class _ClearPartial:
    pass


@dataclass(frozen=True, slots=True)
class _Clear:
    pass


@dataclass(frozen=True, slots=True)
class _Quit:
    pass


type _Command = _Push | _PushPartial | _ClearPartial | _Clear | _Quit


class Overlay:
    """Thread-safe facade over the subtitle window (no-op if Tk is missing)."""

    def __init__(
        self,
        style: OverlayStyle,
        *,
        work_area: Callable[[], tuple[int, int, int, int]] | None = None,
        fullscreen_owner: Callable[[], bool] | None = None,
    ) -> None:
        self._queue: queue.Queue[_Command] = queue.Queue()
        self._model = OverlayModel(style)
        self._work_area = work_area or _fallback_work_area
        self._fullscreen_owner = fullscreen_owner or (lambda: False)
        self._started = threading.Event()
        self._failed = False
        self._hwnd = 0
        self._root: Any = None
        self._frame: Any = None
        self._labels: list[Any] = []
        self._tick_job: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="overlay")
        self._thread.start()
        self._started.wait(timeout=5)

    @property
    def available(self) -> bool:
        """False when the window could not be created (no Tk, no display)."""
        return not self._failed

    def hwnd(self) -> int:
        """Top-level Win32 handle of the bar (0 on other platforms)."""
        return self._hwnd

    def push(self, source: str, target: str) -> None:
        """Show ``source``/``target`` as the newest subtitle segment."""
        self._queue.put(_Push(source=source, target=target))

    def push_provisional(self, source: str, target: str) -> None:
        """Queue the in-progress line (replaces the previous one in place)."""
        self._queue.put(_PushPartial(source=source, target=target))

    def clear_provisional(self) -> None:
        """Queue removal of the in-progress line."""
        self._queue.put(_ClearPartial())

    def clear(self) -> None:
        """Hide the bar and forget everything on it."""
        self._queue.put(_Clear())

    def quit(self) -> None:
        """Stop the Tk thread (idempotent, safe during teardown)."""
        self._queue.put(_Quit())
        self._thread.join(timeout=2)

    # -- Tk thread ---------------------------------------------------------

    def _run(self) -> None:
        """Own the Tk interpreter: build the window, then drain the queue."""
        try:
            import tkinter as tk  # noqa: PLC0415 - lazy so headless boxes can import us
        except Exception as exc:  # noqa: BLE001 - any import failure means "no bar"
            log.warning("overlay unavailable: tkinter import failed (%s)", exc)
            self._failed = True
            self._started.set()
            return
        try:
            self._root = root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.configure(bg=_BG)
            if self._model.style.opacity < 1.0:
                root.attributes("-alpha", max(0.2, min(1.0, self._model.style.opacity)))
            self._win32_harden()
            root.attributes("-topmost", True)
            self._frame = tk.Frame(root, bg=_BG)
            self._frame.pack(fill="both", expand=True)
            self._labels = [
                tk.Label(
                    self._frame,
                    text="",
                    bg=_BG,
                    fg=_FG,
                    justify="center",
                    anchor="center",
                )
                for _ in range(max(2, self._model.style.max_lines * 2))
            ]
        except Exception:
            log.exception("overlay unavailable: Tk init failed")
            self._failed = True
            self._started.set()
            return
        self._started.set()
        root.after(_POLL_MS, self._drain)
        root.mainloop()

    def _win32_harden(self) -> None:
        """Apply the non-activating, topmost, optionally click-through styles."""
        if sys.platform != "win32":
            self._apply_mac_ignores_mouse()
            return
        import ctypes  # noqa: PLC0415 - only needed on Windows
        from ctypes import wintypes  # noqa: PLC0415

        user32 = ctypes.WinDLL("user32")
        root = self._root
        assert root is not None  # noqa: S101 - set by _run before this call
        root.update_idletasks()
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        self._hwnd = hwnd
        gwl_exstyle = -20
        style = user32.GetWindowLongW(hwnd, gwl_exstyle)
        style |= 0x00000008  # WS_EX_TOPMOST
        style |= 0x00000080  # WS_EX_TOOLWINDOW: out of the taskbar
        style |= 0x08000000  # WS_EX_NOACTIVATE: paste must land in the user's app
        if self._model.style.click_through:
            style |= 0x00000020  # WS_EX_TRANSPARENT: clicks pass through
            style |= 0x00080000  # WS_EX_LAYERED: required alongside TRANSPARENT
        user32.SetWindowLongW(hwnd, gwl_exstyle, style)
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )

    def _apply_mac_ignores_mouse(self) -> None:
        """Ask AppKit to ignore mouse events when click-through is on (best effort)."""
        if sys.platform != "darwin" or not self._model.style.click_through:
            return
        try:
            import importlib  # noqa: PLC0415

            appkit = importlib.import_module("AppKit")
            assert self._root is not None  # noqa: S101
            ns_window = appkit.NSApp.windows().lastObject()
            ns_window.setIgnoresMouseEvents_(True)
            ns_window.setLevel_(appkit.NSFloatingWindowLevel)
        except Exception:
            log.debug("overlay: AppKit click-through unavailable", exc_info=True)

    def _drain(self) -> None:
        """Apply every queued command, then reschedule the poll."""
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, _Quit):
                self._cancel_tick()
                root = self._root
                if root is not None:
                    root.quit()
                    root.destroy()
                    # Drop every reference to the interpreter ON the thread that
                    # created it: letting the last one die on another thread
                    # makes Tcl abort with "Tcl_AsyncDelete: async handler
                    # deleted by the wrong thread" at shutdown.
                    self._labels = []
                    self._frame = None
                    self._root = None
                return
            self._apply(command)
        root = self._root
        if root is not None:
            root.after(_POLL_MS, self._drain)

    def _apply(self, command: _Command) -> None:
        """Handle one non-quit command."""
        if isinstance(command, _Push):
            cleared = self._model.clear_provisional()
            pushed = self._model.push(command.source, command.target)
            if cleared or pushed:
                self._render()
        elif isinstance(command, _PushPartial):
            if self._model.push_provisional(command.source, command.target):
                self._render()
        elif isinstance(command, _ClearPartial):
            if self._model.clear_provisional():
                self._render()
        elif isinstance(command, _Clear):
            self._model.clear()
            self._hide()

    def _render(self) -> None:
        """Draw the model's current rows and re-place the window."""
        root = self._root
        if root is None or self._failed:
            return
        if self._fullscreen_owner():
            self._hide()  # never fight a game or a presentation for the screen
            return
        rows = self._model.lines()
        if not rows:
            self._hide()
            return
        self._layout(rows)
        self._place()
        self._schedule_tick()

    def _layout(self, rows: list[Line]) -> None:
        """Assign the visible rows to the label pool (no per-frame widget churn)."""
        root = self._root
        assert root is not None  # noqa: S101
        left, _top, right, _bottom = self._work_area()
        width = right - left if right > left else root.winfo_screenwidth()
        wraplength = max(_MIN_WIDTH, int(width * self._model.style.width_ratio)) - 48
        family = _FONT_FAMILIES.get(sys.platform, "TkDefaultFont")
        size = self._model.style.font_size
        for index, label in enumerate(self._labels):
            if index >= len(rows):
                label.pack_forget()
                continue
            row = rows[index]
            # Older segments fade towards the background so the eye lands on the
            # newest line first; the newest row always renders at full contrast.
            brightness = 0.42 + 0.58 * (index + 1) / len(rows)
            if row.kind == "target":
                base = _FG
            elif row.kind == "provisional":
                base = _PARTIAL_FG
                brightness *= 0.9  # provisional text is a preview, not the record
            else:
                base = _SOURCE_FG
            label.configure(
                text=row.text,
                fg=_fade(base, brightness),
                # The preview is the line the user is reading right now, so it
                # gets the subtitle font; only its colour marks it as not final.
                font=(family, size, "bold")
                if row.kind != "source"
                else (family, max(11, size // 2)),
                wraplength=wraplength,
            )
            label.pack(
                fill="x",
                padx=24,
                pady=(14 if index == 0 else (10 if row.kind == "source" else 2), 8),
            )

    def _place(self) -> None:
        """Pin the bar bottom-centre of the active monitor without activating it."""
        root = self._root
        assert root is not None  # noqa: S101
        root.update_idletasks()
        width = max(root.winfo_reqwidth(), _MIN_WIDTH)
        height = root.winfo_reqheight()
        left, top, right, bottom = self._work_area()
        if right <= left or bottom <= top:
            left, top = 0, 0
            right, bottom = root.winfo_screenwidth(), root.winfo_screenheight()
        x = left + max((right - left) - width, 0) // 2
        y = max(top, bottom - self._model.style.bottom_margin - height)
        if sys.platform == "win32" and self._hwnd:
            import ctypes  # noqa: PLC0415

            ctypes.windll.user32.SetWindowPos(
                ctypes.c_void_p(self._hwnd),
                ctypes.c_void_p(-1),  # HWND_TOPMOST
                x,
                y,
                width,
                height,
                0x0010 | 0x0040,  # SWP_NOACTIVATE | SWP_SHOWWINDOW
            )
            return
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.deiconify()

    def _hide(self) -> None:
        """Withdraw the bar and stop the expiry tick (restarted by a new push)."""
        self._cancel_tick()
        root = self._root
        if root is not None:
            root.withdraw()

    def _schedule_tick(self) -> None:
        """(Re)arm the single pending expiry callback."""
        if self._tick_job is not None:
            return
        root = self._root
        if root is not None:
            self._tick_job = root.after(_TICK_MS, self._tick)

    def _cancel_tick(self) -> None:
        job = self._tick_job
        self._tick_job = None
        root = self._root
        if job is not None and root is not None:
            root.after_cancel(job)

    def _tick(self) -> None:
        """Expire finished segments; re-render, or hide once the bar is empty."""
        self._tick_job = None
        if self._model.expire():
            self._render()
            return
        if not self._model.is_empty():
            self._schedule_tick()


def _fallback_work_area() -> tuple[int, int, int, int]:
    """Empty rectangle: callers without platform info get the whole screen."""
    return (0, 0, 0, 0)


def _fade(color: str, factor: float) -> str:
    """Scale a ``#rrggbb`` colour towards the background (0.0 = invisible)."""
    clamped = max(0.0, min(1.0, factor))
    channels = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    return "#" + "".join(f"{int(channel * clamped):02x}" for channel in channels)


def build_style(config: object) -> OverlayStyle:
    """Build an :class:`OverlayStyle` from the flat app config.

    Kept here (rather than in ``app.config``) so the semantics of each field
    live next to the code that uses them; the config module stays a plain,
    total TOML parser.
    """
    return OverlayStyle(
        mode=config.overlay_mode,  # type: ignore[attr-defined]
        font_size=config.overlay_font_size,  # type: ignore[attr-defined]
        opacity=config.overlay_opacity,  # type: ignore[attr-defined]
        width_ratio=config.overlay_width_ratio,  # type: ignore[attr-defined]
        bottom_margin=config.overlay_bottom_margin,  # type: ignore[attr-defined]
        hold_seconds=config.overlay_hold_seconds,  # type: ignore[attr-defined]
        max_lines=config.overlay_max_lines,  # type: ignore[attr-defined]
        click_through=config.overlay_click_through,  # type: ignore[attr-defined]
    )
