"""Microphone capture: 16 kHz mono stream with gated buffering.

On Windows the stream stays open for the app's lifetime (opening a device costs
50-150 ms, which would eat the push-to-talk latency budget). On macOS an open
stream keeps the system microphone indicator lit, so the stream is opened per
hold instead (measured ~68 ms) and the indicator only appears while dictating.
Either way the audio callback only appends frames while the gate is open.

A *sink* (continuous dictation) turns the recorder into a pure forwarder: every
callback block goes straight into a bounded queue and is never buffered here, so
push-to-talk behavior is byte-for-byte unchanged while no sink is set.
"""

import contextlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from app import native

SAMPLE_RATE: int = 16_000
MIN_RECORD_MS: int = 300  # shorter captures are treated as accidental


@dataclass(frozen=True, slots=True)
class MicDevice:
    index: int
    name: str


def list_input_devices() -> list[MicDevice]:
    """All capture devices PortAudio can see, in index order."""
    return [
        MicDevice(index=i, name=str(dev["name"]))
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0
    ]


def _resolve_device(name: str) -> int | None:
    """Map a configured device name to its index; None = system default."""
    if not name:
        return None
    for device in list_input_devices():
        if device.name == name:
            return device.index
    return None  # configured mic vanished: fall back to default silently


def _offer(sink: queue.Queue[np.ndarray], block: np.ndarray) -> bool:
    """Enqueue a copy without blocking; False when the sink is full."""
    try:
        sink.put_nowait(block.copy())
    except queue.Full:
        return False
    return True


def _forward(sink: queue.Queue[np.ndarray], block: np.ndarray) -> None:
    """Hand one block to *sink* without ever blocking the audio callback.

    When the consumer falls behind, the oldest queued block is dropped so the
    stream keeps flowing and the callback stays real-time.
    """
    if not _offer(sink, block):
        with contextlib.suppress(queue.Empty):
            sink.get_nowait()  # drop the oldest block
        _offer(sink, block)


class Recorder:
    """Owns the InputStream; yields float32 mono samples on stop()."""

    def __init__(self, device_name: str, on_stream_error: Callable[[str], None]) -> None:
        self._device_name = device_name
        self._on_stream_error = on_stream_error
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._sink: queue.Queue[np.ndarray] | None = None
        self._stream: sd.InputStream | None = None
        if native.RECORDER_KEEP_OPEN:
            self._open_stream()

    def _open_stream(self) -> None:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=_resolve_device(self._device_name),
            callback=self._on_audio,
        )
        stream.start()
        self._stream = stream

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def start(self) -> None:
        if not native.RECORDER_KEEP_OPEN:
            self._open_stream()
        with self._lock:
            self._buffer = []
            self._recording = True

    def stop(self) -> np.ndarray | None:
        """Finish a capture; None when it was too short to be speech."""
        with self._lock:
            self._recording = False
            chunks = self._buffer
            self._buffer = []
        if not native.RECORDER_KEEP_OPEN:
            self._close_stream()
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0).reshape(-1)
        if audio.shape[0] < SAMPLE_RATE * MIN_RECORD_MS // 1000:
            return None
        return audio

    def close(self) -> None:
        self._close_stream()

    def start_sink(self, sink: queue.Queue[np.ndarray]) -> None:
        """Continuous capture: forward every callback block to *sink*.

        The sink owns its own drop policy (see :func:`_forward`), so the
        callback can never block; the hold buffer is left untouched.
        """
        if not native.RECORDER_KEEP_OPEN and self._stream is None:
            self._open_stream()
        with self._lock:
            self._sink = sink
            self._recording = False

    def stop_sink(self) -> None:
        """Stop forwarding blocks; push-to-talk state is left untouched."""
        with self._lock:
            self._sink = None
        if not native.RECORDER_KEEP_OPEN:
            self._close_stream()

    def _on_audio(self, indata: np.ndarray, _frames: int, _time: object, status: int) -> None:
        if status:
            self._on_stream_error(f"audio stream status: {status}")
        sink = self._sink
        if sink is not None:
            _forward(sink, indata)
            return
        if self._recording:
            with self._lock:
                self._buffer.append(indata.copy())
