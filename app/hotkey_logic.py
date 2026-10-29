"""Pure hold/click discrimination for the push-to-talk hotkey.

The Win32 hook layer (app.hotkey) converts raw key transitions into
:class:`Press` / ::class:`Release` events with monotonic timestamps; this
module decides whether the user tapped (click) or held (record) the key.
No I/O, no clock access — timestamps are injected, so tests are deterministic.
"""

from dataclasses import dataclass
from typing import assert_never


def push_to_talk_active(*, continuous: bool) -> bool:
    """Return whether a hotkey press may start a push-to-talk recording.

    Push-to-talk and continuous dictation are mutually exclusive: continuous
    mode already owns the microphone, so a press must not start a recording.

    The key itself keeps working: while continuous mode runs the hooks stay
    *disarmed*, so CapsLock still toggles caps exactly as the OS normally does —
    only the "hold to talk" behaviour is suppressed. The disarmed "model
    loading" hint is silenced separately, since the app is not loading anything.
    """
    return not continuous


def loading_hint_visible(*, continuous: bool) -> bool:
    """Return whether the disarmed "model loading" pill may be shown at all.

    Disarmed means "the key is native": that is normal while the model is still
    loading, but during continuous dictation nothing is loading, so the pill
    would be a lie — and showing then hiding it would also wipe the sticky
    "listening" pill the continuous mode keeps on screen.
    """
    return not continuous


@dataclass(frozen=True, slots=True)
class Press:
    """Key transitioned to down at this monotonic timestamp (ms)."""

    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class Release:
    """Key transitioned to up at this monotonic timestamp (ms)."""

    timestamp_ms: int


type KeyEvent = Press | Release


@dataclass(frozen=True, slots=True)
class StartHold:
    """Key held down past nothing yet — recording should begin."""


@dataclass(frozen=True, slots=True)
class Click:
    """Key was tapped: re-synthesize the original key function (e.g. CapsLock toggle)."""


@dataclass(frozen=True, slots=True)
class EndHold:
    """Key released after a real hold: stop recording, transcribe, type."""

    duration_ms: int


type Action = StartHold | Click | EndHold


class HoldClickDetector:
    """Mutable accumulator turning key transitions into hold/click actions.

    Mutation is the documented purpose: this is a one-shot state machine fed
    by the keyboard hook thread. Feed order is guaranteed by the hook layer.
    """

    __slots__ = ("_down_at_ms", "_threshold_ms")

    def __init__(self, threshold_ms: int) -> None:
        self._threshold_ms = threshold_ms
        self._down_at_ms: int | None = None

    def feed(self, event: KeyEvent) -> Action | None:
        match event:
            case Press(timestamp_ms=ts):
                if self._down_at_ms is not None:
                    return None  # auto-repeat: already down
                self._down_at_ms = ts
                return StartHold()
            case Release(timestamp_ms=ts):
                down_at = self._down_at_ms
                if down_at is None:
                    return None  # stray release (e.g. key was down before hook start)
                self._down_at_ms = None
                duration = ts - down_at
                if duration < self._threshold_ms:
                    return Click()
                return EndHold(duration_ms=duration)
            case unreachable:
                assert_never(unreachable)
