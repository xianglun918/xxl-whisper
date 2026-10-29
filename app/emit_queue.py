"""Bounded FIFO between the decoder and the emitter (translate → type/overlay).

Why this exists: translation is the slow part (100–300 ms per hop, two hops for
日→中) and it used to run *inside* the decode thread. With someone talking
non-stop, every finished sentence then waited behind the previous sentence's
translation, so subtitles lagged further and further behind the speaker.

Splitting them means the decoder only does ASR (fast, keeps up with speech) and
hands each finished sentence to this queue; one emitter thread drains it in
order, so subtitles still arrive in the order they were spoken.

The queue is bounded and drops the OLDEST pending sentence when the emitter
falls behind. That is deliberate: for live subtitles a stale sentence is worse
than a missing one — the viewer is already past it.
"""

import queue
import time
from dataclasses import dataclass

__all__ = ["EmitQueue", "PendingEmit"]


@dataclass(frozen=True, slots=True)
class PendingEmit:
    """One finished sentence on its way to the cursor / subtitle bar."""

    text: str
    enqueued_at: float
    endpoint_ts: float | None = None


class EmitQueue:
    """Thread-safe bounded FIFO with drop-oldest pressure relief."""

    def __init__(self, maxsize: int = 64) -> None:
        self._queue: queue.Queue[PendingEmit] = queue.Queue(maxsize=maxsize)
        self._dropped = 0

    def submit(self, text: str, *, endpoint_ts: float | None = None) -> bool:
        """Queue one sentence; returns False when an older one had to go."""
        item = PendingEmit(text=text, enqueued_at=time.monotonic(), endpoint_ts=endpoint_ts)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass
        else:
            return True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._queue.task_done()
            self._dropped += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:  # another producer won the race; this one is lost
            self._dropped += 1
            return False
        return False

    def get(self, timeout: float = 0.2) -> PendingEmit | None:
        """Next sentence, or None when nothing arrived within *timeout*."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark the item returned by :meth:`get` as handled."""
        self._queue.task_done()

    def drain(self, limit: int = 64) -> list[PendingEmit]:
        """Remove and return everything pending (used on shutdown/route change)."""
        items: list[PendingEmit] = []
        while len(items) < limit:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        return items

    @property
    def dropped(self) -> int:
        """How many sentences were sacrificed to keep the queue shallow."""
        return self._dropped

    def __len__(self) -> int:
        """Return how many sentences are waiting to be translated."""
        return self._queue.qsize()
