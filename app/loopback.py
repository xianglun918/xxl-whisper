"""System-audio capture ("what the speakers are playing") as a dictation source.

Watching a video, a meeting or a course means the interesting audio never
reaches the microphone. On Windows, WASAPI *loopback* hands us exactly that mix
without any driver or "Stereo Mix" device, which is what this module wraps.

Layout mirrors :mod:`app.recorder`: the class exposes ``start``/``stop``/
``close`` and pushes blocks through a PortAudio-shaped callback, so the
recorder can swap one backend for the other without knowing the difference.

Design notes:

* **Optional dependency, lazy import.** ``soundcard`` is imported inside the
  capture thread. Without it (Linux, or a stripped build) the feature reports
  itself unavailable and dictation keeps using the microphone.
* **Two conversion jobs in one place.** The loopback endpoint runs at the
  speaker's rate (usually 48 kHz stereo) while the recogniser wants 16 kHz mono,
  so :func:`to_mono` and :class:`Resampler` live here and are unit-tested with
  synthetic signals instead of real audio.
* **Silence is data.** WASAPI loopback only produces packets while something is
  playing. The recogniser's VAD needs the gaps too (otherwise one sentence
  glues onto the next), so blocks keep flowing at a steady cadence.
* **macOS has no loopback API.** Apple removed it; users route system audio
  through a virtual device (BlackHole) and pick it as a normal microphone, so
  this module stays Windows-only and reports "unavailable" elsewhere.
"""

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "DEVICE_PREFIX",
    "TARGET_RATE",
    "BlockCallback",
    "LoopbackDevice",
    "LoopbackStream",
    "LoopbackUnavailable",
    "Resampler",
    "available",
    "display_name",
    "list_loopback_devices",
    "to_mono",
]

#: Target rate of the recogniser (matches ``app.recorder.SAMPLE_RATE``).
TARGET_RATE: int = 16_000
#: Frames pulled per iteration at the capture rate (~64 ms at 48 kHz).
CAPTURE_BLOCK: int = 3_072
#: Device-name prefix the tray uses for these pseudo-devices.
DEVICE_PREFIX: str = "系统声音 · "

type BlockCallback = Callable[[np.ndarray, int, object, int], None]


class LoopbackUnavailable(Exception):  # noqa: N818 - reads better as a state than an error
    """Raised when system-audio capture cannot run on this machine."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class LoopbackDevice:
    """One capturable output endpoint (its speakers, seen from the inside)."""

    name: str
    device_id: str


def _soundcard() -> Any:  # noqa: ANN401 - third-party module
    """Import the optional dependency on first use."""
    if sys.platform != "win32":
        reason = "系统声音回环仅 Windows 支持（macOS 请用 BlackHole 虚拟声卡）"
        raise LoopbackUnavailable(reason)
    try:
        import soundcard  # noqa: PLC0415 - optional dependency, lazily imported
    except Exception as exc:
        reason = f"未安装 soundcard（{exc}）"
        raise LoopbackUnavailable(reason) from exc
    return soundcard


def available() -> bool:
    """Return True when this machine can capture the system mix directly."""
    try:
        _soundcard()
    except LoopbackUnavailable:
        return False
    return True


def list_loopback_devices() -> list[LoopbackDevice]:
    """List every loopback endpoint, named the way the tray shows it.

    Never raises: a missing dependency or a broken audio stack yields an empty
    list, so the microphone menu simply has no "system audio" section.
    """
    try:
        soundcard = _soundcard()
    except LoopbackUnavailable:
        return []
    try:
        speakers = soundcard.all_speakers()
    except Exception:
        log.debug("loopback: enumerate speakers failed", exc_info=True)
        return []
    return [
        LoopbackDevice(name=display_name(str(speaker.name)), device_id=str(speaker.name))
        for speaker in speakers
    ]


def display_name(device_id: str) -> str:
    """Return the tray label for a stored device id."""
    return f"{DEVICE_PREFIX}{device_id}"


def to_mono(block: np.ndarray) -> np.ndarray:
    """Downmix ``(frames, channels)`` float32 audio to a 1-D mono signal."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.shape[1] == 0:
        return np.zeros(block.shape[0], dtype=np.float32)
    return block.astype(np.float32, copy=False).mean(axis=1)


class Resampler:
    """Streaming linear resampler that keeps phase across blocks.

    Linear interpolation is enough here: the recogniser only needs speech to be
    intelligible at 16 kHz, and a windowed-sinc polyphase filter would cost more
    CPU per block than the ASR itself on a low-power machine.
    """

    def __init__(self, source_rate: int, target_rate: int = TARGET_RATE) -> None:
        if source_rate <= 0 or target_rate <= 0:
            reason = f"非法采样率 {source_rate} -> {target_rate}"
            raise LoopbackUnavailable(reason)
        self._ratio = source_rate / target_rate
        self._source_rate = source_rate
        self._target_rate = target_rate
        #: Source samples carried into the next call so interpolation is
        #: continuous across block boundaries.
        self._tail = np.zeros(0, dtype=np.float32)
        #: Fractional read position inside ``_tail`` for the next call.
        self._phase = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Resample one block; the output length follows the rate ratio."""
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)
        data = np.concatenate((self._tail, block.astype(np.float32, copy=False)))
        if self._source_rate == self._target_rate:
            self._tail = data[-1:]
            return data
        span = data.size - 1 - self._phase
        if span < 0:
            self._tail = data
            return np.zeros(0, dtype=np.float32)
        count = int(np.floor(span / self._ratio)) + 1
        positions = self._phase + np.arange(count, dtype=np.float64) * self._ratio
        out = np.interp(positions, np.arange(data.size, dtype=np.float64), data).astype(np.float32)
        next_position = positions[-1] + self._ratio  # first sample of the next call
        keep_from = min(int(np.floor(next_position)), data.size - 1)
        self._tail = data[keep_from:].copy()
        self._phase = next_position - keep_from
        return out


class LoopbackStream:
    """PortAudio-shaped capture of the default (or named) speaker output."""

    def __init__(
        self,
        device_name: str,
        callback: BlockCallback,
        *,
        rate: int = TARGET_RATE,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._device_name = device_name
        self._callback = callback
        self._on_error = on_error
        self._rate = rate
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending = np.zeros(0, dtype=np.float32)
        #: ~64 ms at the target rate: small enough for a responsive VAD.
        self._block = max(1, rate // 16)

    # -- PortAudio-compatible surface --------------------------------------

    def start(self) -> None:
        """Begin capturing on a background thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._thread = threading.Thread(target=self._run, daemon=True, name="loopback")
        self._thread.start()

    def stop(self) -> None:
        """Ask the capture thread to finish and wait for it briefly."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2)

    def close(self) -> None:
        """Alias for :meth:`stop` (the recorder calls both)."""
        self.stop()

    # -- capture thread ----------------------------------------------------

    def _pick_speaker(self, soundcard: Any) -> Any:  # noqa: ANN401 - third-party module
        """Resolve the configured speaker, falling back to the default one."""
        if self._device_name:
            try:
                return soundcard.get_speaker(self._device_name)
            except Exception:  # noqa: BLE001 - a renamed endpoint is not fatal
                log.info("loopback: %r not found, using the default speaker", self._device_name)
        return soundcard.default_speaker()

    def _run(self) -> None:
        try:
            soundcard = _soundcard()
            speaker = self._pick_speaker(soundcard)
            # ``include_loopback`` is what turns a speaker into a microphone.
            mic = soundcard.get_microphone(speaker.name, include_loopback=True)
            capture_rate = self._probe_rate(mic)
            resampler = Resampler(capture_rate, self._rate)
            log.info(
                "loopback: capturing %r at %d Hz -> %d Hz mono",
                speaker.name,
                capture_rate,
                self._rate,
            )
            with mic.recorder(samplerate=capture_rate) as recorder:
                while not self._stop.is_set():
                    chunk = recorder.record(numframes=CAPTURE_BLOCK)
                    self._emit(resampler.process(to_mono(chunk)))
        except Exception as exc:  # noqa: BLE001 - report once, then stay quiet
            log.warning("loopback capture stopped: %s", exc)
            if self._on_error is not None:
                self._on_error(f"系统声音采集已停止：{exc}")

    @staticmethod
    def _probe_rate(mic: Any) -> int:  # noqa: ANN401 - third-party object
        """Pick a capture rate the driver accepts (48 kHz, else 44.1 kHz).

        The loopback endpoint inherits the speaker's shared-mode format; almost
        every Windows box mixes at 48 kHz, but a few are set to 44.1 kHz and
        rejecting that would leave the user with a dead feature.
        """
        for rate in (48_000, 44_100):
            try:
                with mic.recorder(samplerate=rate):
                    return rate
            except Exception:  # try the next rate
                log.debug("loopback: rate %d rejected", rate, exc_info=True)
        return 48_000  # let the real open surface the error to the user

    def _emit(self, samples: np.ndarray) -> None:
        """Chop the resampled signal into fixed blocks and deliver them."""
        if samples.size == 0:
            return
        self._pending = (
            np.concatenate((self._pending, samples)) if self._pending.size else samples
        )
        while self._pending.size >= self._block:
            block = self._pending[: self._block]
            self._pending = self._pending[self._block :]
            self._callback(block.reshape(-1, 1), self._block, None, 0)
