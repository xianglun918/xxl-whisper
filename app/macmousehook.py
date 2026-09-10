"""macOS mouse side-button hook — inert stand-in.

The Windows app supports X1/X2 side buttons as hotkeys; macOS drops that preset
(most Macs have no side buttons — see the plan's preset decision), so
``MouseHook`` mirrors the win lifecycle surface and never intercepts anything.
Shared code reaches it via ``app.native.mousehook``.
"""

from collections.abc import Callable


class MouseHookError(Exception):
    """Kept for surface parity with the Windows hook; never raised on macOS."""

    def __init__(self, code: int) -> None:
        super().__init__(f"mouse hook install failed ({code})")
        self.code = code


class MouseHook:
    """Inert stand-in for the Windows X1/X2 hook (macOS drops side buttons)."""

    def __init__(
        self,
        vk: int,
        on_transition: Callable[[bool], None],
        disarmed_prompt: Callable[[bool], None] | None = None,
    ) -> None:
        self._vk = vk
        self._on_transition = on_transition
        self._disarmed_prompt = disarmed_prompt
        self._armed = False

    def set_armed(self, armed: bool) -> None:
        """No-op: macOS never intercepts side buttons."""
        self._armed = armed

    def run(self) -> None:
        """No-op: there is no hook thread to pump."""

    def start_and_wait(self) -> None:
        """Return immediately so app startup proceeds."""

    def stop(self) -> None:
        """No-op: nothing to tear down."""

    def retarget(self, vk: int) -> None:
        """No-op: the observed button is irrelevant on macOS."""
        self._vk = vk
