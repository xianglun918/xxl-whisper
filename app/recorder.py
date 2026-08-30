"""Microphone capture: persistent 16 kHz mono stream with gated buffering.

The stream stays open for the app's lifetime (opening a device costs
50-150 ms, which would eat the push-to-talk latency budget); the audio
callback only appends frames while the recording gate is open.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

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


class Recorder:
    """Owns the InputStream; yields float32 mono samples on stop()."""

    def __init__(self, device_name: str, on_stream_error: Callable[[str], None]) -> None:
        self._on_stream_error = on_stream_error
        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=_resolve_device(device_name),
            callback=self._on_audio,
        )
        self._stream.start()

    def start(self) -> None:
        with self._lock:
            self._buffer = []
            self._recording = True

    def stop(self) -> np.ndarray | None:
        """Finish a capture; None when it was too short to be speech."""
        with self._lock:
            self._recording = False
            chunks = self._buffer
            self._buffer = []
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0).reshape(-1)
        if audio.shape[0] < SAMPLE_RATE * MIN_RECORD_MS // 1000:
            return None
        return audio

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _on_audio(self, indata: np.ndarray, _frames: int, _time: object, status: int) -> None:
        if status:
            self._on_stream_error(f"audio stream status: {status}")
        if self._recording:
            with self._lock:
                self._buffer.append(indata.copy())
