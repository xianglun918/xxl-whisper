"""Pure scheduling for the continuous decode path (priority + adaptivity).

Continuous dictation splits into three roles so a heavy model (Fun-ASR-Nano)
can never stall endpointing or delay a finished sentence:

* the real-time VAD consumer (never decodes),
* the finished-sentence decoder (its own thread, always wins),
* the live-partial decoder (its own thread, never blocks the finished path).

This module holds the pieces of that split that are pure or trivially
testable — the adaptive partial cadence, the utterance-length ceiling, the
thread-safe publish gate shared by the VAD loop and the partial decoder, and
the single-slot holder that lets stale partial snapshots be dropped — keeping
the threads themselves thin glue.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

#: A partial decode that takes at least the base interval to run is "slow": it
#: would compete with the final sentence for the CPU, so the next partial is
#: pushed out in proportion to its cost.
_SLOW_DECODE_BACKOFF: float = 2.0

#: Hard ceiling on the partial gap: past this, live partials are effectively
#: paused and the decoder gives the real-time path its CPU back.
_MAX_PARTIAL_INTERVAL_S: float = 4.0

#: An in-progress utterance longer than this stops getting live partials. The
#: measured decode cost grows with the utterance (Fun-ASR-Nano: ~0.33 s at 2 s,
#: ~0.53 s at 4 s, ~0.97 s at 8 s), so past this point each preview competes
#: with the finished sentence for the same CPU. The trade-off: the pill keeps
#: its last preview until the endpoint instead of refreshing, and the inserted
#: text is still the full utterance — partials are skipped, never sliced, so
#: the "preview matches the insert" invariant is untouched. 6 s is the knee
#: where a preview still costs well under the endpoint's silence gap.
_PARTIAL_MAX_DURATION_S: float = 6.0


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


def within_partial_ceiling(
    duration_s: float, max_s: float = _PARTIAL_MAX_DURATION_S
) -> bool:
    """Whether an utterance of *duration_s* is still cheap enough to preview.

    Past :data:`_PARTIAL_MAX_DURATION_S` the preview is skipped so it can never
    starve the finished sentence's CPU; the endpoint still decodes the full
    utterance, so the preview is only ever *absent*, never wrong.
    """
    return duration_s <= max_s


@dataclass(slots=True)
class PartialCadence:
    """Thread-safe publish gate shared by the VAD loop and the partial decoder.

    The VAD loop asks :meth:`should_publish` *before* paying for a full
    utterance copy — the O(N) snapshot it used to take on every speech block —
    so the copy only happens when a partial is genuinely due. The partial
    thread brackets each decode with :meth:`mark_start` / :meth:`mark_end`, so
    a slow model stretches the cadence and an in-flight decode blocks a new
    publish. :meth:`reset` marks an endpoint so the next utterance starts from
    a fresh cadence and a fresh caption.
    """

    base_s: float
    _last_decode_s: float | None = None
    _last_partial_at: float = 0.0
    _in_flight: bool = False
    _generation: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def should_publish(self, now: float, *, paused: bool) -> bool:
        """Whether the VAD loop should snapshot and publish a partial now.

        False while paused, while a decode is in flight, and until the adaptive
        interval has elapsed since the last decode started.
        """
        with self._lock:
            if paused or self._in_flight:
                return False
            gap = partial_interval(self.base_s, self._last_decode_s)
            return now - self._last_partial_at >= gap

    def mark_start(self, now: float) -> None:
        """Record the start of a partial decode; blocks further publishes."""
        with self._lock:
            self._in_flight = True
            self._last_partial_at = now

    def mark_end(self, duration_s: float) -> None:
        """Record a completed partial decode so the cadence can back off."""
        with self._lock:
            self._in_flight = False
            self._last_decode_s = duration_s

    def reset(self, now: float) -> None:
        """Endpoint: start the next utterance's cadence from a clean slate."""
        with self._lock:
            self._in_flight = False
            self._last_decode_s = None
            self._last_partial_at = now
            self._generation += 1

    def generation(self) -> int:
        """Monotonic endpoint counter; the partial thread clears its caption on a bump."""
        with self._lock:
            return self._generation


@dataclass(frozen=True, slots=True)
class FinishedBatch:
    """Finished speech segments plus when the endpoint was detected.

    Carried across the VAD→decode queue so the decoder can log the real
    endpoint→emit latency the user feels, without the VAD thread ever decoding.
    """

    endpoint_ts: float
    segments: tuple[np.ndarray, ...]


class LatestSlot[T]:
    """Single-slot holder: only the newest value matters.

    The VAD consumer publishes a fresh partial snapshot only when the slot is
    empty (see :meth:`has_value`) so a slow decoder cannot make it copy the
    growing utterance repeatedly; when a publish does race, the single slot
    overwrites the previous value (it never queues). :meth:`publish` reports
    whether it dropped a stale pending value so the caller can log how many
    partials were skipped.
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

    def has_value(self) -> bool:
        """Whether a snapshot is pending (lets the VAD loop skip a redundant copy)."""
        with self._lock:
            return self._value is not None

    def clear(self) -> None:
        """Drop any pending snapshot (the endpoint made it stale)."""
        with self._lock:
            self._value = None
