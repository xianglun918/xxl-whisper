"""System-audio capture: downmix, resampling and device naming.

Everything here is pure signal/logic work — the actual WASAPI loopback needs a
Windows box with a speaker, so the tests cover the parts that silently corrupt
speech when they are wrong: channel downmix, the streaming resampler's phase
across block boundaries, and the pseudo-device naming the tray relies on.
"""

import sys

import numpy as np
import pytest
from app.loopback import (
    DEVICE_PREFIX,
    LoopbackStream,
    LoopbackUnavailable,
    Resampler,
    display_name,
    list_loopback_devices,
    to_mono,
)


def test_stereo_is_averaged_to_mono() -> None:
    block = np.array([[1.0, 0.0], [0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    assert to_mono(block).tolist() == [0.5, 0.5, 0.0]


def test_mono_input_passes_through() -> None:
    block = np.array([0.25, -0.25], dtype=np.float32)
    assert to_mono(block).tolist() == [0.25, -0.25]


def test_zero_channels_yields_silence_not_a_crash() -> None:
    block = np.zeros((4, 0), dtype=np.float32)
    assert to_mono(block).shape == (4,)
    assert not to_mono(block).any()


def test_resampler_halves_the_length_for_half_the_rate() -> None:
    resampler = Resampler(32_000, 16_000)
    signal = np.sin(np.linspace(0, 8 * np.pi, 3200, dtype=np.float32))
    out = resampler.process(signal)
    assert abs(out.size - 1600) <= 2


def test_resampler_keeps_phase_across_blocks() -> None:
    """Two half-blocks must yield the same samples as one whole block.

    A resampler that restarts phase per call produces clicks at every block
    boundary — audible to the recogniser as garbled words.
    """
    signal = np.sin(np.linspace(0, 8 * np.pi, 4096, dtype=np.float32))
    whole = Resampler(48_000, 16_000).process(signal)
    splitter = Resampler(48_000, 16_000)
    first = splitter.process(signal[:2048])
    second = splitter.process(signal[2048:])
    joined = np.concatenate((first, second))
    assert abs(joined.size - whole.size) <= 2
    common = min(joined.size, whole.size)
    assert np.allclose(joined[:common], whole[:common], atol=1e-5)


def test_resampler_identity_rate_is_lossless() -> None:
    resampler = Resampler(16_000, 16_000)
    signal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.allclose(resampler.process(signal), signal)


def test_resampler_rejects_impossible_rates() -> None:
    with pytest.raises(LoopbackUnavailable):
        Resampler(0, 16_000)


def test_stream_emits_fixed_size_blocks() -> None:
    """Blocks reach the recogniser at a steady cadence, never partial ones."""
    blocks: list[np.ndarray] = []
    stream = LoopbackStream("", lambda data, _frames, _t, _s: blocks.append(data[:, 0]))
    stream._emit(np.ones(2000, dtype=np.float32))
    step = stream._block
    assert len(blocks) == 2000 // step
    assert all(block.size == step for block in blocks)


def test_stream_carries_the_remainder_to_the_next_block() -> None:
    blocks: list[np.ndarray] = []
    stream = LoopbackStream("", lambda data, _frames, _t, _s: blocks.append(data[:, 0]))
    step = stream._block
    stream._emit(np.ones(step // 2, dtype=np.float32))
    assert blocks == []
    stream._emit(np.ones(step, dtype=np.float32))
    assert len(blocks) == 1
    assert blocks[0].size == step


def test_device_names_are_prefixed_for_the_tray() -> None:
    assert display_name("Speakers (Realtek)") == DEVICE_PREFIX + "Speakers (Realtek)"


def test_listing_is_empty_off_windows() -> None:
    """macOS/Linux have no loopback API: the menu must simply stay unchanged."""
    if sys.platform == "win32":
        pytest.skip("Windows: the real list depends on the machine's speakers")
    assert list_loopback_devices() == []
