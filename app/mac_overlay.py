"""macOS native subtitle overlay — non-activating AppKit NSPanel.

Mirrors :mod:`app.overlay` (the Windows Tk bar) but renders with AppKit so the
macOS path never imports ``tkinter``. The stacking/expiry rules are reused from
:mod:`app.overlay`'s :class:`OverlayModel`; only the window is native. Every
method is thread-safe: UI work is dispatched to the main thread via the same
``NSOperationQueue`` pattern as :mod:`app.mac_indicator`, where the tray's
NSApplication run loop services it. The panel is borderless, non-activating,
floating and click-through by default, so pasted text still lands in the target
app. pyobjc is imported with ``importlib`` (M1 constraint).
"""

import importlib
import logging
from collections.abc import Callable
from typing import Any

from app.overlay import Line, OverlayModel, OverlayStyle, build_style

AppKit = importlib.import_module("AppKit")
Foundation = importlib.import_module("Foundation")

log = logging.getLogger(__name__)

__all__ = ["Overlay", "build_style"]

_MIN_WIDTH: float = 240.0
_FONT_SIZE: float = 15.0
_SOURCE_SIZE_DELTA: float = 3.0
_RADIUS: float = 12.0
_SIDE_PAD: float = 18.0
_TOP_PAD: float = 6.0
_BOTTOM_PAD: float = 6.0
_LINE_HEIGHT_FACTOR: float = 1.6
_TICK_S: float = 0.25

#: Same palette as app.overlay (which itself matches app.mac_indicator), so the
#: subtitle bar and the "正在听…" pill read as one visual language.
_BG: tuple[float, float, float, float] = (0.063, 0.078, 0.094, 0.92)  # #101418
_FG: tuple[float, float, float, float] = (0.910, 0.918, 0.929, 1.0)  # #e8eaed
_SOURCE_FG: tuple[float, float, float, float] = (0.604, 0.643, 0.682, 1.0)  # #9aa4ae
_PARTIAL_FG: tuple[float, float, float, float] = (0.812, 0.839, 0.867, 1.0)  # #cfd6dd


class Overlay:
    """Thread-safe facade over the native AppKit subtitle panel."""

    def __init__(
        self,
        style: OverlayStyle,
        *,
        work_area: Callable[[], tuple[int, int, int, int]] | None = None,
        fullscreen_owner: Callable[[], bool] | None = None,
    ) -> None:
        self._model = OverlayModel(style)
        self._work_area = work_area or _fallback_work_area
        self._fullscreen_owner = fullscreen_owner or (lambda: False)
        self._panel: Any = None
        self._content: Any = None
        self._labels: list[Any] = []
        self._tick_timer: Any = None
        self._window_number = 0

    @property
    def available(self) -> bool:
        """Always True: AppKit is a hard macOS dependency."""
        return True

    def hwnd(self) -> int:
        """Native window number (0 until first shown); mirrors the Win32 bar."""
        return self._window_number

    def push(self, source: str, target: str) -> None:
        """Show ``source``/``target`` as the newest finished segment."""
        self._dispatch(lambda: self._apply_push(source, target))

    def push_provisional(self, source: str, target: str) -> None:
        """Queue the in-progress line (replaces the previous one in place)."""
        self._dispatch(lambda: self._apply_push_provisional(source, target))

    def clear_provisional(self) -> None:
        """Queue removal of the in-progress line."""
        self._dispatch(self._apply_clear_provisional)

    def clear(self) -> None:
        """Hide the bar and forget everything on it."""
        self._dispatch(self._apply_clear)

    def quit(self) -> None:
        """Hide the bar and release the window (idempotent)."""
        self._dispatch(self._apply_quit)

    # -- main thread -------------------------------------------------------

    def _dispatch(self, action: Callable[[], None]) -> None:
        """Run ``action`` on the main thread (the tray's run loop)."""
        AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(action)

    def _apply_push(self, source: str, target: str) -> None:
        cleared = self._model.clear_provisional()
        pushed = self._model.push(source, target)
        if cleared or pushed:
            self._render()

    def _apply_push_provisional(self, source: str, target: str) -> None:
        if self._model.push_provisional(source, target):
            self._render()

    def _apply_clear_provisional(self) -> None:
        if self._model.clear_provisional():
            self._render()

    def _apply_clear(self) -> None:
        self._model.clear()
        self._hide()

    def _apply_quit(self) -> None:
        self._cancel_tick()
        if self._panel is not None:
            self._panel.orderOut_(None)
        self._panel = None
        self._content = None
        self._labels = []

    def _render(self) -> None:
        """Draw the model's current rows and (re)place the window."""
        if self._fullscreen_owner():
            self._hide()  # never fight a game or a presentation for the screen
            return
        rows = self._model.lines()
        if not rows:
            self._hide()
            return
        self._ensure_panel()
        self._layout(rows)
        self._place()
        self._schedule_tick()

    def _ensure_panel(self) -> None:
        """Main-thread: create the non-activating panel on first use."""
        if self._panel is not None:
            return
        rect = Foundation.NSMakeRect(0.0, 0.0, _MIN_WIDTH, 60.0)
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(self._model.style.click_through)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )
        content = AppKit.NSView.alloc().initWithFrame_(rect)
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(_ns_color(_BG).CGColor())
        content.layer().setCornerRadius_(_RADIUS)
        labels = [_make_label() for _ in range(max(2, self._model.style.max_lines * 2))]
        for label in labels:
            content.addSubview_(label)
        panel.setContentView_(content)
        self._panel = panel
        self._content = content
        self._labels = labels
        self._window_number = int(panel.windowNumber())

    def _layout(self, rows: list[Line]) -> None:
        """Main-thread: assign the visible rows to the label pool."""
        content = self._content
        if content is None:
            return
        width = _bar_width(self._work_area(), self._model.style.width_ratio)
        line_height = _line_height(self._model.style.font_size)
        height = line_height * len(rows) + _TOP_PAD + _BOTTOM_PAD
        for index, label in enumerate(self._labels):
            if index >= len(rows):
                label.setHidden_(True)
                continue
            label.setHidden_(False)
            row = rows[index]
            size = (
                self._model.style.font_size
                if row.kind != "source"
                else max(11.0, self._model.style.font_size - _SOURCE_SIZE_DELTA)
            )
            brightness = 0.42 + 0.58 * (index + 1) / len(rows)
            if row.kind == "target":
                rgba = _FG
            elif row.kind == "provisional":
                rgba = _PARTIAL_FG
                brightness *= 0.9
            else:
                rgba = _SOURCE_FG
            label.setStringValue_(row.text)
            label.setFont_(AppKit.NSFont.systemFontOfSize_(size))
            label.setTextColor_(_ns_color(_fade(rgba, brightness)))
            y = _BOTTOM_PAD + (len(rows) - 1 - index) * line_height
            label.setFrame_(
                Foundation.NSMakeRect(_SIDE_PAD, y, width - 2 * _SIDE_PAD, line_height)
            )
        frame_size = Foundation.NSMakeSize(width, height)
        content.setFrameSize_(frame_size)
        self._panel.setContentSize_(frame_size)

    def _place(self) -> None:
        """Main-thread: bottom-centre of the active screen's work area."""
        screen = AppKit.NSScreen.mainScreen()
        if screen is None or self._panel is None:
            return
        frame = screen.visibleFrame()
        width = self._panel.frame().size.width
        x = frame.origin.x + (frame.size.width - width) / 2.0
        y = frame.origin.y + self._model.style.bottom_margin
        self._panel.setFrameOrigin_(Foundation.NSMakePoint(x, y))
        self._panel.orderFrontRegardless()

    def _hide(self) -> None:
        """Main-thread: order the panel out and stop the expiry tick."""
        self._cancel_tick()
        if self._panel is not None:
            self._panel.orderOut_(None)

    def _schedule_tick(self) -> None:
        """Main-thread: (re)arm the single pending expiry timer."""
        if self._tick_timer is not None:
            return
        self._tick_timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            _TICK_S, False, self._tick
        )

    def _cancel_tick(self) -> None:
        timer = self._tick_timer
        self._tick_timer = None
        if timer is not None:
            timer.invalidate()

    def _tick(self, _timer: object) -> None:
        """Main-thread: expire finished segments; re-render or hide."""
        self._tick_timer = None
        if self._model.expire():
            self._render()
            return
        if not self._model.is_empty():
            self._schedule_tick()


def _fallback_work_area() -> tuple[int, int, int, int]:
    """Empty rectangle: callers without platform info get the whole screen."""
    return (0, 0, 0, 0)


def _ns_color(rgba: tuple[float, float, float, float]) -> object:
    """Build a calibrated NSColor from an (r, g, b, a) tuple."""
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba)


def _fade(
    rgba: tuple[float, float, float, float], factor: float
) -> tuple[float, float, float, float]:
    """Scale the RGB channels towards black (0.0 = invisible); alpha kept."""
    clamped = max(0.0, min(1.0, factor))
    return (rgba[0] * clamped, rgba[1] * clamped, rgba[2] * clamped, rgba[3])


def _line_height(font_size: float) -> float:
    """Per-row height: the font's line height plus vertical breathing room."""
    return _LINE_HEIGHT_FACTOR * font_size + 8.0


def _bar_width(work_area: tuple[int, int, int, int], ratio: float) -> float:
    """Bar width from the active work area, clamped to a minimum."""
    left, _top, right, _bottom = work_area
    if right > left:
        work_width = float(right - left)
    else:
        screen = AppKit.NSScreen.mainScreen()
        work_width = float(screen.visibleFrame().size.width) if screen else 1920.0
    return max(_MIN_WIDTH, work_width * ratio)


def _make_label() -> object:
    """Build one centred, borderless single-line text field."""
    label = AppKit.NSTextField.alloc().initWithFrame_(
        Foundation.NSMakeRect(0.0, 0.0, 100.0, 20.0)
    )
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setAlignment_(AppKit.NSTextAlignmentCenter)
    label.setTextColor_(_ns_color(_FG))
    label.setFont_(AppKit.NSFont.systemFontOfSize_(_FONT_SIZE))
    return label
