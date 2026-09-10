"""macOS "正在听…" status bar.

Invisible until M4 ships the AppKit NSPanel: every method is a no-op so the
dictation loop runs without a bar. M4 implements the non-activating panel
(the Windows version relies on ``WS_EX_NOACTIVATE``; Tk cannot reproduce that
on macOS — see the M1 findings in ``docs/macos-port-plan.md``).
"""

import logging

log = logging.getLogger(__name__)


class Indicator:
    """Owns the status bar; invisible (no-op) until M4 ships the NSPanel."""

    def __init__(self) -> None:
        pass

    def show(self, text: str) -> None:
        """No-op until M4: the bar stays hidden."""
        log.debug("indicator.show(%r) ignored until M4", text)

    def update(self, text: str) -> None:
        """No-op until M4."""
        log.debug("indicator.update(%r) ignored until M4", text)

    def progress(self, pct: float, text: str) -> None:
        """No-op until M4."""
        log.debug("indicator.progress(%s, %r) ignored until M4", pct, text)

    def hide(self) -> None:
        """No-op until M4."""

    def flash(self, text: str, ms: int = 1500) -> None:
        """No-op until M4."""
        log.debug("indicator.flash(%r, %dms) ignored until M4", text, ms)

    def quit(self) -> None:
        """No-op: there is no window loop to stop yet."""

    def hwnd(self) -> int:
        """No native window handle until M4."""
        return 0
