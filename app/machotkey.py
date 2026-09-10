"""macOS global push-to-talk hook — skeleton, body lands in M3.

Mirrors the lifecycle surface of ``app/hotkey.py`` (``HotkeyHook``) so shared
code reaches it via ``app.native.hotkey``. M3 implements the listen-only
CGEventTap (right Cmd = keycode 0x36, ``flagsChanged``) with a CFRunLoop
thread and a tap watchdog — see the M1 findings in ``docs/macos-port-plan.md``.

The Windows hook suppresses the key and re-synthesizes taps; macOS deliberately
never suppresses (decision #5), so this class only observes transitions.
"""

from collections.abc import Callable


class HotkeyError(Exception):
    """Raised when the macOS event tap cannot be installed."""

    def __init__(self, code: int) -> None:
        super().__init__(f"event tap install failed ({code})")
        self.code = code


class HotkeyHook:
    """Daemon-thread event-tap hook; emits key transitions for one keycode."""

    def __init__(
        self,
        vk: int,
        on_transition: Callable[[bool], None],
        disarmed_prompt: Callable[[bool], None] | None = None,
    ) -> None:
        self._vk = vk
        self._on_transition = on_transition
        self._disarmed_prompt = disarmed_prompt

    def set_armed(self, armed: bool) -> None:
        """Arm/disarm transition delivery."""
        msg = f"machotkey.set_armed(armed={armed}) lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)

    def run(self) -> None:
        """Pump the CFRunLoop for the event tap."""
        msg = "machotkey.run lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)

    def start_and_wait(self) -> None:
        """Start the hook thread and block until the tap is installed."""
        msg = "machotkey.start_and_wait lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)

    def stop(self) -> None:
        """Stop the hook thread and tear the tap down."""
        msg = "machotkey.stop lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)

    def retarget(self, vk: int) -> None:
        """Switch the observed keycode."""
        msg = f"machotkey.retarget(vk={vk}) lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)

    def arm_capture(self, on_key: Callable[[int], None]) -> None:
        """One-shot capture of the next key for the hotkey picker."""
        msg = f"machotkey.arm_capture({on_key!r}) lands in M3 (listen-only tap)"
        raise NotImplementedError(msg)
