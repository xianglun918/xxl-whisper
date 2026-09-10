"""macOS UI-Automation channel — permanently reports unavailable.

Windows writes into modern controls through the UIAutomation COM API; macOS has
no general text-append equivalent (AXUIElement cannot set arbitrary text), so
this channel is terminal-unavailable and M4's emit ladder registers no UIA
channel on macOS. Shared code reaches it via ``app.native.uia``.
"""

from dataclasses import dataclass


class UiaUnavailableError(Exception):
    """Raised when a UIA write is attempted on macOS."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class UiaTarget:
    """A focused control and whether UIA value writes are possible."""

    writable: bool


def probe_focused() -> UiaTarget:
    """Never writable on macOS."""
    return UiaTarget(writable=False)


def append_text(text: str) -> None:
    """Raise UiaUnavailableError: macOS has no UIA append channel."""
    raise UiaUnavailableError(reason=f"UIA is not available on macOS ({len(text)} chars dropped)")
