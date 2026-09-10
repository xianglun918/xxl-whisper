"""macOS "正在听…" status bar — non-activating AppKit NSPanel (M4).

The bar must never take keyboard focus (pasted text has to land in the target
app), so the panel is borderless, non-activating and floating, shown with
``orderFrontRegardless()`` while mouse events are ignored. Every method is
thread-safe: UI work is dispatched to the main thread, where the tray's
NSApplication run loop services it. Placement mirrors the Windows bar
(bottom-centre of the active screen, suppressed while a fullscreen window owns
the display). pyobjc is imported with ``importlib`` (M1 constraint).
"""

import importlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from app import macio

AppKit = importlib.import_module("AppKit")
Foundation = importlib.import_module("Foundation")

log = logging.getLogger(__name__)

_WIDTH = 240.0
_HEIGHT = 44.0
_LABEL_HEIGHT = 20.0
_BOTTOM_MARGIN = 96.0
_RADIUS = 12.0
_FONT_SIZE = 15.0
_BG = (0.063, 0.078, 0.094, 0.92)
_FG = (0.91, 0.92, 0.93, 1.0)


class Indicator:
    """Thread-safe facade over the AppKit status panel."""

    def __init__(self) -> None:
        self._generation = 0
        self._panel: Any = None
        self._label: Any = None
        self._window_number = 0

    def show(self, text: str) -> None:
        """Show the bar with ``text``."""
        self._dispatch(lambda: self._render(text, flash_ms=None))

    def update(self, text: str) -> None:
        """Update the bar text."""
        self._dispatch(lambda: self._render(text, flash_ms=None))

    def progress(self, pct: float, text: str) -> None:
        """Show download progress."""
        self._dispatch(lambda: self._render(f"{text} {pct * 100:.0f}%", flash_ms=None))

    def hide(self) -> None:
        """Hide the bar."""
        self._dispatch(self._hide_now)

    def flash(self, text: str, ms: int = 1500) -> None:
        """Show a transient message that auto-hides."""
        self._dispatch(lambda: self._render(text, flash_ms=ms))

    def quit(self) -> None:
        """Hide the bar; the panel dies with the process."""
        self._dispatch(self._hide_now)

    def hwnd(self) -> int:
        """Native window number of the bar (observability/tests); 0 until shown."""
        return self._window_number

    def _dispatch(self, action: Callable[[], None]) -> None:
        """Run ``action`` on the main thread (the tray's run loop)."""
        AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(action)

    def _render(self, text: str, flash_ms: int | None) -> None:
        """Main-thread: show or update the panel with ``text``."""
        self._ensure_panel()
        self._generation += 1
        generation = self._generation
        self._label.setStringValue_(text)
        if macio.exclusive_fullscreen_owner_active():
            self._hide_now()
            return
        self._place()
        self._panel.orderFrontRegardless()
        if flash_ms is not None:
            timer = threading.Timer(flash_ms / 1000, lambda: self._hide_if_stale(generation))
            timer.daemon = True
            timer.start()

    def _hide_if_stale(self, generation: int) -> None:
        """Auto-hide only when nothing newer has been shown since scheduling."""

        def _check() -> None:
            if generation == self._generation:
                self._hide_now()

        self._dispatch(_check)

    def _hide_now(self) -> None:
        """Main-thread: order the panel out."""
        if self._panel is not None:
            self._panel.orderOut_(None)

    def _place(self) -> None:
        """Main-thread: bottom-centre of the active screen's work area."""
        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            return
        frame = screen.visibleFrame()
        x = frame.origin.x + (frame.size.width - _WIDTH) / 2
        y = frame.origin.y + _BOTTOM_MARGIN
        self._panel.setFrameOrigin_(Foundation.NSMakePoint(x, y))

    def _ensure_panel(self) -> None:
        """Main-thread: create the non-activating panel on first use."""
        if self._panel is not None:
            return
        rect = Foundation.NSMakeRect(0, 0, _WIDTH, _HEIGHT)
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
        panel.setIgnoresMouseEvents_(True)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )
        content = AppKit.NSView.alloc().initWithFrame_(rect)
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*_BG).CGColor()
        )
        content.layer().setCornerRadius_(_RADIUS)
        label = AppKit.NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(12, (_HEIGHT - _LABEL_HEIGHT) / 2, _WIDTH - 24, _LABEL_HEIGHT)
        )
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setAlignment_(AppKit.NSTextAlignmentCenter)
        label.setTextColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*_FG))
        label.setFont_(AppKit.NSFont.systemFontOfSize_(_FONT_SIZE))
        content.addSubview_(label)
        panel.setContentView_(content)
        self._panel = panel
        self._label = label
        self._window_number = int(panel.windowNumber())
