"""macOS global push-to-talk hook — listen-only CGEventTap (M3).

Observes one mac keycode via a listen-only event tap at the session level and
reports press/release transitions. macOS never suppresses the key (decision
#5), so a quick tap passes through natively and the app treats it as a click.

The mechanism is the one validated by ``scripts/probe_mac_tap.py`` (M1):
``flagsChanged`` for modifier keys, ``keyDown``/``keyUp`` for regular keys, a
CFRunLoop on the hook thread, and a watchdog that re-enables the tap when
macOS disables it. pyobjc is imported with ``importlib`` (M1 constraint).
"""

import importlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType

Quartz = importlib.import_module("Quartz")

log = logging.getLogger(__name__)

#: mac virtual keycodes for the hotkey presets (kVK_*). macOS has no Scroll
#: Lock and drops mouse side buttons, so neither appears here.
PRESET_KEYCODES: Mapping[str, int] = MappingProxyType(
    {
        "right_cmd": 0x36,
        "f2": 0x78,
        "f4": 0x76,
        "f6": 0x61,
        "f8": 0x64,
    }
)

#: Preset selected on a fresh macOS install.
DEFAULT_HOTKEY: str = "right_cmd"

#: No mouse side-button hotkeys on macOS.
MOUSE_KEYCODES: frozenset[int] = frozenset()

#: Modifier keycodes -> the CGEventFlags bit that means "this key is down".
_MODIFIER_FLAGS: Mapping[int, int] = MappingProxyType(
    {
        0x36: Quartz.kCGEventFlagMaskCommand,  # right Command
        0x37: Quartz.kCGEventFlagMaskCommand,  # left Command
        0x3C: Quartz.kCGEventFlagMaskShift,  # right Shift
        0x3D: Quartz.kCGEventFlagMaskAlternate,  # right Option
        0x3E: Quartz.kCGEventFlagMaskControl,  # right Control
    }
)

_VK_DISABLED: int = 0


class HotkeyError(Exception):
    """Raised when the macOS event tap cannot be installed."""

    def __init__(self, code: int) -> None:
        super().__init__(f"event tap install failed ({code})")
        self.code = code


class HotkeyHook:
    """Daemon-thread listen-only tap; emits transitions for one keycode.

    ``on_transition(pressed)`` runs on the hook thread — it must only enqueue.
    Auto-repeat and stray releases are filtered here, so the callback sees
    clean down/up pairs. ``vk=0`` disables interception (no key is observed).
    """

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
        self._is_down = False
        self._capture_cb: Callable[[int], None] | None = None
        self._install_error: HotkeyError | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._tap: object | None = None
        self._thread = threading.Thread(target=self.run, daemon=True, name="hotkey-hook")

    def set_armed(self, armed: bool) -> None:
        """Arm or disarm transition delivery.

        While disarmed (e.g. the model is still loading) transitions drive
        ``disarmed_prompt`` instead of the worker.
        """
        self._armed = armed

    def retarget(self, vk: int) -> None:
        """Switch the observed keycode."""
        self._vk = vk
        self._is_down = False

    def arm_capture(self, on_key: Callable[[int], None]) -> None:
        """Capture the next non-modifier key press for the hotkey picker."""
        self._capture_cb = on_key

    def _callback(self, _proxy: object, typ: int, event: object, _refcon: object) -> object:
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if self._capture_cb is not None and typ == Quartz.kCGEventKeyDown:
            capture = self._capture_cb
            self._capture_cb = None
            capture(keycode)
            return event
        if self._vk == _VK_DISABLED or keycode != self._vk:
            return event
        if typ == Quartz.kCGEventFlagsChanged:
            flag = _MODIFIER_FLAGS.get(self._vk)
            if flag is None:
                return event
            pressed = bool(Quartz.CGEventGetFlags(event) & flag)
        elif typ in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
            pressed = typ == Quartz.kCGEventKeyDown
        else:
            return event
        if pressed == self._is_down:
            return event
        self._is_down = pressed
        if self._armed:
            self._on_transition(pressed)
        elif self._disarmed_prompt is not None:
            self._disarmed_prompt(pressed)
        return event

    def _install(self, run_loop: object) -> tuple[object | None, object | None]:
        """Create the tap, wire it into ``run_loop``, enable it, return both."""
        mask = (
            (1 << Quartz.kCGEventFlagsChanged)
            | (1 << Quartz.kCGEventKeyDown)
            | (1 << Quartz.kCGEventKeyUp)
        )
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._callback,
            None,
        )
        if not tap:
            return None, None
        self._tap = tap
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(run_loop, source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)
        return tap, source

    def run(self) -> None:
        """Install the tap and pump its CFRunLoop until :meth:`stop`."""
        run_loop = Quartz.CFRunLoopGetCurrent()
        tap, source = self._install(run_loop)
        if tap is None:
            self._install_error = HotkeyError(code=0)
            self._ready.set()
            return
        self._ready.set()
        watchdog = time.monotonic()
        while not self._stop.is_set():
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.1, False)
            now = time.monotonic()
            if now - watchdog < 1.0:
                continue
            watchdog = now
            if Quartz.CGEventTapIsEnabled(tap):
                continue
            if Quartz.CGPreflightListenEventAccess():
                # The grant arrived after this tap was created; a tap created
                # without permission stays dead, so rebuild it — the permission
                # check happens at creation, which saves the user a restart.
                log.info("input monitoring granted; recreating the event tap")
                Quartz.CFRunLoopRemoveSource(run_loop, source, Quartz.kCFRunLoopDefaultMode)
                new_tap, new_source = self._install(run_loop)
                if new_tap is not None:
                    tap, source = new_tap, new_source
                continue
            log.warning("event tap disabled by the system; re-enabling")
            Quartz.CGEventTapEnable(tap, True)

    def start_and_wait(self) -> None:
        """Start the hook thread and block until the tap is installed."""
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._install_error is not None:
            raise self._install_error

    def stop(self) -> None:
        """Stop the hook thread."""
        self._stop.set()
