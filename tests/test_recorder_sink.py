"""Unit tests for the recorder's never-blocking sink forwarder (continuous mode)."""

import queue

import numpy as np
from app.recorder import _forward, _offer


def test_offer_enqueues_a_copy() -> None:
    sink: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
    block = np.arange(4, dtype=np.float32)
    assert _offer(sink, block) is True
    assert sink.qsize() == 1
    assert np.array_equal(sink.get_nowait(), block)


def test_offer_returns_false_when_full() -> None:
    sink: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    assert _offer(sink, np.zeros(1, dtype=np.float32)) is True
    assert _offer(sink, np.zeros(1, dtype=np.float32)) is False


def test_forward_drops_oldest_when_full() -> None:
    sink: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
    _forward(sink, np.full(1, 1.0, dtype=np.float32))
    _forward(sink, np.full(1, 2.0, dtype=np.float32))
    _forward(sink, np.full(1, 3.0, dtype=np.float32))  # full: the 1.0 block is dropped
    assert [float(sink.get_nowait()[0]) for _ in range(2)] == [2.0, 3.0]


def test_forward_never_raises_and_stays_bounded() -> None:
    sink: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    for value in range(5):
        _forward(sink, np.full(1, value, dtype=np.float32))
    assert sink.qsize() == 1
    assert float(sink.get_nowait()[0]) == 4.0
