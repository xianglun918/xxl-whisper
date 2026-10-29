"""The ASR→translator hand-off queue.

The behaviour that matters for live subtitles: order is preserved, and when
translation falls behind the OLDEST sentence is sacrificed (a stale line is
worse than a missing one — the speaker is already past it).
"""

from app.emit_queue import EmitQueue, PendingEmit


def _take(q: EmitQueue) -> PendingEmit:
    item = q.get(timeout=0.01)
    assert item is not None
    return item


def test_keeps_fifo_order() -> None:
    q = EmitQueue()
    for text in ("一", "二", "三"):
        q.submit(text)
    assert [_take(q).text for _ in range(3)] == ["一", "二", "三"]


def test_get_returns_none_when_idle() -> None:
    assert EmitQueue().get(timeout=0.01) is None


def test_full_queue_drops_the_oldest() -> None:
    q = EmitQueue(maxsize=2)
    assert q.submit("一") is True
    assert q.submit("二") is True
    assert q.submit("三") is False  # oldest ("一") had to go
    assert q.dropped == 1
    assert [_take(q).text for _ in range(2)] == ["二", "三"]


def test_dropped_counter_only_grows() -> None:
    q = EmitQueue(maxsize=1)
    q.submit("一")
    q.submit("二")
    q.submit("三")
    assert q.dropped == 2
    assert len(q) == 1


def test_endpoint_timestamp_is_carried_for_latency_logging() -> None:
    q = EmitQueue()
    q.submit("早", endpoint_ts=123.0)
    item = q.get(timeout=0.01)
    assert item is not None
    assert item.endpoint_ts == 123.0
    assert item.enqueued_at >= 0.0


def test_drain_empties_the_queue() -> None:
    q = EmitQueue()
    for text in ("一", "二"):
        q.submit(text)
    assert [item.text for item in q.drain()] == ["一", "二"]
    assert len(q) == 0


def test_emitter_never_blocks_the_producer() -> None:
    """A full queue must cost one drop, never a wait (the decoder is real-time)."""
    q = EmitQueue(maxsize=1)
    q.submit("一")
    for _ in range(50):
        q.submit("二")  # would hang forever if submit blocked
    assert len(q) == 1
