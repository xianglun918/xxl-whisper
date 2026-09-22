"""Pure unit tests for the 512-sample VAD framing (no model, no I/O)."""

import numpy as np
from app.vad import WINDOW_SIZE, WindowFramer, split_full_windows


def test_split_exact_multiple_leaves_empty_remainder() -> None:
    samples = np.arange(1024, dtype=np.float32)
    windows, tail = split_full_windows(samples, 512)
    assert len(windows) == 2
    assert tail.size == 0
    assert np.array_equal(windows[0], samples[:512])
    assert np.array_equal(windows[1], samples[512:])


def test_split_odd_length_keeps_remainder() -> None:
    samples = np.arange(1000, dtype=np.float32)
    windows, tail = split_full_windows(samples, 512)
    assert len(windows) == 1
    assert windows[0].shape == (512,)
    assert np.array_equal(tail, samples[512:])


def test_split_shorter_than_window() -> None:
    samples = np.arange(300, dtype=np.float32)
    windows, tail = split_full_windows(samples, 512)
    assert windows == []
    assert np.array_equal(tail, samples)


def test_split_empty_input() -> None:
    windows, tail = split_full_windows(np.empty(0, dtype=np.float32), 512)
    assert windows == []
    assert tail.size == 0


def test_framer_accumulates_across_odd_blocks() -> None:
    framer = WindowFramer(WINDOW_SIZE)
    assert framer.push(np.zeros(300, dtype=np.float32)) == []  # tail 300
    assert [w.shape for w in framer.push(np.zeros(300, dtype=np.float32))] == [(512,)]  # tail 88
    assert [w.shape for w in framer.push(np.zeros(424, dtype=np.float32))] == [(512,)]  # tail 0


def test_framer_flush_pads_and_resets() -> None:
    framer = WindowFramer(WINDOW_SIZE)
    framer.push(np.ones(100, dtype=np.float32))
    tail = framer.flush()
    assert tail is not None
    assert tail.shape == (WINDOW_SIZE,)
    assert np.all(tail[:100] == 1.0)
    assert np.all(tail[100:] == 0.0)
    assert framer.flush() is None


def test_framer_preserves_every_sample_in_order() -> None:
    rng = np.random.default_rng(0)
    stream = rng.standard_normal(5000).astype(np.float32)
    framer = WindowFramer(WINDOW_SIZE)
    windows: list[np.ndarray] = []
    position = 0
    for size in (300, 700, 1, 512, 1000, 2487):
        windows.extend(framer.push(stream[position : position + size]))
        position += size
    tail = framer.flush()
    assert tail is not None
    rebuilt = np.concatenate([*windows, tail])
    assert all(window.shape == (WINDOW_SIZE,) for window in windows)
    assert np.array_equal(rebuilt[: stream.shape[0]], stream)
    assert np.all(rebuilt[stream.shape[0] :] == 0.0)


def test_framer_accepts_column_block() -> None:
    """PortAudio hands (frames, channels) blocks; the framer must flatten."""
    framer = WindowFramer(WINDOW_SIZE)
    windows = framer.push(np.zeros((600, 1), dtype=np.float32))
    assert [w.shape for w in windows] == [(512,)]
    assert framer.flush() is not None
