"""Silero VAD wrapper for continuous dictation: exact-window framing.

sherpa-onnx's :class:`~sherpa_onnx.VoiceActivityDetector` only segments speech
reliably when fed its native window (512 samples at 16 kHz) one block at a
time; handing it a whole utterance in a single call collapses to a near-empty
segment. The framing that guarantees this is a pure function
(:func:`split_full_windows`) so it is unit-tested without the model; the
model-backed :class:`Segmenter` only glues framing to sherpa-onnx and exposes
finished speech segments for decoding plus the in-progress segment for live
partials.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

log = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000
WINDOW_SIZE: int = 512


def split_full_windows(
    samples: np.ndarray, window: int = WINDOW_SIZE
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split *samples* into complete *window*-sized blocks plus a remainder.

    Pure: no state, no I/O. The complete windows come back in order and the
    trailing samples that did not fill a window are returned separately (empty
    when the input divides evenly).
    """
    count = samples.shape[0] // window
    windows = [samples[i * window : (i + 1) * window] for i in range(count)]
    return windows, samples[count * window :]


class WindowFramer:
    """Carries the partial tail across calls so every emitted window is exact.

    Mutation is the documented purpose: it is a one-shot accumulator fed by the
    continuous-capture thread. Only :meth:`push` and :meth:`flush` touch state.
    """

    __slots__ = ("_tail", "_window")

    def __init__(self, window: int = WINDOW_SIZE) -> None:
        self._window = window
        self._tail = np.empty(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Return the complete windows now available; buffer the remainder."""
        flat = block.reshape(-1).astype(np.float32, copy=False)
        combined = np.concatenate((self._tail, flat)) if self._tail.size else flat
        windows, self._tail = split_full_windows(combined, self._window)
        return windows

    def flush(self) -> np.ndarray | None:
        """Return the final short window padded to full size, then reset.

        Returns ``None`` when no samples are buffered.
        """
        tail = self._tail
        self._tail = np.empty(0, dtype=np.float32)
        if tail.size == 0:
            return None
        padded = np.zeros(self._window, dtype=np.float32)
        padded[: tail.size] = tail
        return padded


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Resolved VAD tuning (probability / seconds) built from the config."""

    model_path: Path
    threshold: float = 0.5
    min_speech_s: float = 0.25
    min_silence_s: float = 0.5
    max_speech_s: float = 20.0
    num_threads: int = 1


class Segmenter:
    """Streaming VAD: feed arbitrary blocks, drain finished speech segments."""

    def __init__(self, settings: VadSettings) -> None:
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(settings.model_path)
        config.silero_vad.threshold = settings.threshold
        config.silero_vad.min_speech_duration = settings.min_speech_s
        config.silero_vad.min_silence_duration = settings.min_silence_s
        config.silero_vad.max_speech_duration = settings.max_speech_s
        config.sample_rate = SAMPLE_RATE
        config.num_threads = settings.num_threads
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._framer = WindowFramer(int(config.silero_vad.window_size))
        log.info(
            "vad: threshold=%.2f min_speech=%.2fs min_silence=%.2fs max_speech=%.1fs",
            settings.threshold,
            settings.min_speech_s,
            settings.min_silence_s,
            settings.max_speech_s,
        )

    def accept(self, block: np.ndarray) -> None:
        """Feed one arbitrary-length float32 block; detect on exact windows."""
        for window in self._framer.push(block):
            self._vad.accept_waveform(window)

    def current_samples(self) -> np.ndarray | None:
        """Return the in-progress speech segment's samples so far, or ``None`` when idle.

        The VAD accumulates the current utterance itself — onset included — so
        this is the same audio that :meth:`drain` will hand back at the
        endpoint. Decoding it makes the live partial a faithful preview of the
        sentence that will be inserted, instead of a separately accumulated
        buffer that misses the speech onset.

        Returns an owned float32 copy (the VAD's segment list keeps growing
        until the endpoint) so the caller can publish it to another thread
        without racing the VAD.
        """
        segment = self._vad.current_segment
        if segment is None or not segment.samples:
            return None
        return np.array(segment.samples, dtype=np.float32, copy=True)

    def drain(self) -> list[np.ndarray]:
        """Speech segments finished since the last drain, in order."""
        segments: list[np.ndarray] = []
        while not self._vad.empty():
            segments.append(np.asarray(self._vad.front.samples, dtype=np.float32))
            self._vad.pop()
        return segments

    def flush(self) -> None:
        """End the stream: feed the padded tail, then close the VAD."""
        tail = self._framer.flush()
        if tail is not None:
            self._vad.accept_waveform(tail)
        self._vad.flush()
