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
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import sounddevice as sd

from app import loopback, native

log = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000
MIN_RECORD_MS: int = 300  # shorter captures are treated as accidental

#: Hard cap on the push-to-talk buffer (2 minutes). Nothing should ever hold the
#: key that long; the cap exists so a capture nobody stopped cannot grow forever.
MAX_HOLD_S: float = 120.0
_MAX_HOLD_SAMPLES: int = int(SAMPLE_RATE * MAX_HOLD_S)


@dataclass(frozen=True, slots=True)
class MicDevice:
    index: int
    name: str


class _CaptureStream(Protocol):
    """What the recorder needs from a backend (PortAudio or loopback)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


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

    def __init__(
        self,
        device_name: str,
        on_stream_error: Callable[[str], None],
        *,
        source: str = "mic",
    ) -> None:
        self._device_name = device_name
        self._source = source
        self._on_stream_error = on_stream_error
        self._lock = threading.Lock()
        #: Serialises the open/close decisions against each other: the hook
        #: thread and the worker can both call start()/stop() for one hold, and
        #: "check → open → mark" is not atomic on its own.
        self._lifecycle = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._buffered_samples = 0
        self._sink: queue.Queue[np.ndarray] | None = None
        #: Monotonic count of blocks handed to a sink — proof the stream is live.
        self.blocks_delivered: int = 0
        self._stream: _CaptureStream | None = None
        if native.RECORDER_KEEP_OPEN:
            self._open_stream()

    def _open_stream(self) -> None:
        """Open the capture stream; a second call is a no-op.

        Idempotent on purpose: macOS opens a stream per hold, so a double open
        meant a leaked stream and a lit microphone indicator for every sentence.
        """
        if self._stream is not None:
            return
        if self._source == "system":
            # System audio: WASAPI loopback resampled to 16 kHz mono, pushed
            # through the same callback shape PortAudio uses.
            stream = cast(
                "_CaptureStream",
                loopback.LoopbackStream(
                    self._device_name,
                    self._on_audio,
                    on_error=self._on_stream_error,
                ),
            )
        else:
            stream = cast(
                "_CaptureStream",
                sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=_resolve_device(self._device_name),
                    callback=self._on_audio,
                ),
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
        """Begin a hold capture; safe to call twice for the same hold.

        The hook thread prewarms on key-down and the worker's ``StartHold`` then
        re-enters here. The guard must come *before* ``_open_stream()``: macOS
        opens a stream per hold, so opening before the guard left a second stream
        open for every sentence — a lit microphone indicator, doubled audio and
        a stream leak per utterance (``stop()`` only closes the last one).
        """
        with self._lifecycle:
            if self._capturing_hold():
                return
            if not native.RECORDER_KEEP_OPEN:
                self._open_stream()
            with self._lock:
                if self._recording:
                    return
                self._buffer = []
                self._buffered_samples = 0
                self._recording = True

    def _capturing_hold(self) -> bool:
        """Return True when a hold is already captured on a live stream."""
        return self._recording and (native.RECORDER_KEEP_OPEN or self._stream is not None)

    def _cap_buffer(self) -> None:
        """Keep the hold buffer bounded even if nobody stops the capture.

        The hook thread prewarms on key-down; if the worker then decides the
        press must be skipped (paused, model still loading) nothing would stop
        that capture, and the buffer grew for as long as the app stayed open —
        then the next hold shipped minutes of stale audio to the model.
        """
        if not self._recording:
            return
        dropped = False
        with self._lock:
            while (
                self._buffered_samples > _MAX_HOLD_SAMPLES and len(self._buffer) > 1
            ):
                oldest = self._buffer.pop(0)
                self._buffered_samples -= int(oldest.shape[0])
                dropped = True
        if dropped:
            log.warning("recorder: hold buffer over %.0f s, dropped the oldest audio",
                        _MAX_HOLD_SAMPLES / SAMPLE_RATE)

    def stop(self) -> np.ndarray | None:
        """Finish a capture; None when it was too short to be speech."""
        with self._lifecycle:
            with self._lock:
                self._recording = False
                chunks = self._buffer
                self._buffer = []
                self._buffered_samples = 0
            if not native.RECORDER_KEEP_OPEN:
                self._close_stream()
        if not chunks:
            return None
        audio = np.concatenate(chunks, axis=0).reshape(-1)
        if audio.shape[0] > _MAX_HOLD_SAMPLES:
            # Nobody holds a hotkey for two minutes: this capture was never
            # stopped (a skipped press, a lost key-up). Shipping it to the model
            # would paste minutes of stale audio, so drop it instead of "capping".
            log.warning(
                "recorder: dropped a runaway capture (%.0f s)", audio.shape[0] / SAMPLE_RATE
            )
            return None
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
            # Continuous capture owns the stream, so a hold that is still marked
            # active is stale: clear it or the next push-to-talk would skip
            # opening a stream and silently record nothing.
            self._recording = False
            self._buffer = []
            self._buffered_samples = 0

    @property
    def capturing(self) -> bool:
        """True when a stream exists and audio is being delivered to a sink."""
        return self._stream is not None and self._sink is not None

    def stop_sink(self) -> None:
        """Stop forwarding blocks; push-to-talk state is left untouched."""
        with self._lock:
            self._sink = None
        if not native.RECORDER_KEEP_OPEN:
            self._close_stream()

    def _on_audio(self, indata: np.ndarray, _frames: int, _time: object, status: int) -> None:
        if status:
            self._on_stream_error(f"audio stream status: {status}")
        self._cap_buffer()
        sink = self._sink
        if sink is not None:
            self.blocks_delivered += 1
            _forward(sink, indata)
            return
        if self._recording:
            with self._lock:
                self._buffer.append(indata.copy())
                self._buffered_samples += int(indata.shape[0])
