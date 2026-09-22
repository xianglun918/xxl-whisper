"""Pure scheduling for the continuous decode path (priority + adaptivity).

Continuous dictation splits into a real-time VAD consumer and a slower decoder
so a heavy model (Fun-ASR-Nano) can never stall endpointing. This module holds
the pieces of that split that are pure or trivially testable — the priority
decision, the adaptive partial cadence, the bounded work item, and the
single-slot holder that lets stale partial snapshots be dropped — keeping the
threads themselves thin glue.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

#: A partial decode that takes at least the base interval to run is "slow": it
#: would compete with the final sentence for the CPU, so the next partial is
#: pushed out in proportion to its cost.
_SLOW_DECODE_BACKOFF: float = 2.0

#: Hard ceiling on the partial gap: past this, live partials are effectively
#: paused and the decoder gives the real-time path its CPU back.
_MAX_PARTIAL_INTERVAL_S: float = 4.0


class DecodeKind(StrEnum):
    """What the decoder should do on its next turn."""

    FINISHED = "finished"
    PARTIAL = "partial"
    IDLE = "idle"


@dataclass(frozen=True, slots=True)
class FinishedBatch:
    """Finished speech segments plus when the endpoint was detected.

    Carried across the VAD→decode queue so the decoder can log the real
    endpoint→emit latency the user feels, without the VAD thread ever decoding.
    """

    endpoint_ts: float
    segments: tuple[np.ndarray, ...]


def next_decode(*, has_finished: bool, partial_due: bool) -> DecodeKind:
    """Pick the next unit of work; a finished sentence always wins.

    Pure: the decoder drains every finished batch before it ever spends time on
    a partial, so a pending sentence can never wait behind partial work.
    """
    if has_finished:
        return DecodeKind.FINISHED
    if partial_due:
        return DecodeKind.PARTIAL
    return DecodeKind.IDLE


def partial_interval(base_s: float, last_decode_s: float | None) -> float:
    """Effective minimum gap between live-partial decodes.

    Fast decodes keep the configured cadence. A decode slower than the base
    interval stretches the gap to twice its duration (capped at
    :data:`_MAX_PARTIAL_INTERVAL_S`), so partials back off exactly when they
    would otherwise delay the finished sentence or saturate the CPU.
    """
    if last_decode_s is None or last_decode_s <= base_s:
        return base_s
    return min(last_decode_s * _SLOW_DECODE_BACKOFF, _MAX_PARTIAL_INTERVAL_S)


class LatestSlot[T]:
    """Single-slot holder: only the newest value matters.

    The VAD consumer publishes a fresh partial snapshot on every speech block;
    a slow decoder may not keep up, and only the latest snapshot is worth
    decoding, so each publish overwrites the previous value (it never queues).
    :meth:`publish` reports whether it dropped a stale pending value so the
    caller can log how many partials were skipped.
    """

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None

    def publish(self, value: T) -> bool:
        """Store *value* as the newest snapshot; True when a stale one was dropped."""
        with self._lock:
            replaced = self._value is not None
            self._value = value
            return replaced

    def take(self) -> T | None:
        """Return and clear the newest snapshot, or ``None`` when empty."""
        with self._lock:
            value = self._value
            self._value = None
            return value

    def clear(self) -> None:
        """Drop any pending snapshot (the endpoint made it stale)."""
        with self._lock:
            self._value = None
