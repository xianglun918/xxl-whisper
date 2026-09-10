"""macOS "正在听…" status bar — skeleton, body lands in M4.

Mirrors the ``Indicator`` surface of ``app/indicator.py`` so shared code
reaches it via ``app.native.indicator``. M4 implements a non-activating AppKit
NSPanel: the Windows version relies on ``WS_EX_NOACTIVATE`` and Tk cannot
reproduce that on macOS (see the M1 findings in ``docs/macos-port-plan.md``).
``quit`` is a real no-op so shutdown never raises.
"""


class Indicator:
    """Owns the status bar; ``quit`` is safe, the rest are M4 placeholders."""

    def __init__(self) -> None:
        pass

    def show(self, text: str) -> None:
        """Show the bar with ``text``."""
        msg = f"mac_indicator.show({text!r}) lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)

    def update(self, text: str) -> None:
        """Update the bar text."""
        msg = f"mac_indicator.update({text!r}) lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)

    def progress(self, pct: float, text: str) -> None:
        """Show download progress."""
        msg = f"mac_indicator.progress({pct}, {text!r}) lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)

    def hide(self) -> None:
        """Hide the bar."""
        msg = "mac_indicator.hide lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)

    def flash(self, text: str, ms: int = 1500) -> None:
        """Show a transient message."""
        msg = f"mac_indicator.flash({text!r}, {ms}ms) lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)

    def quit(self) -> None:
        """No-op: there is no window loop to stop yet."""

    def hwnd(self) -> int:
        """Native window handle (observability/tests)."""
        msg = "mac_indicator.hwnd lands in M4 (AppKit NSPanel)"
        raise NotImplementedError(msg)
