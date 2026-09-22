"""Continuous dictation must survive a transient per-block/per-segment failure.

Regression guard: the continuous loop is a long-lived daemon thread with no
restart path, so a single escaped exception (clipboard contention, a UIA
element gone mid-focus-change, a decode hiccup) used to wedge the mode
permanently while the tray still reported it ON. The loop now logs and
continues; these tests pin that behaviour without a model or a display.
"""

import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from app.app import DictationApp
from app.decode_scheduler import FinishedBatch, LatestSlot
from app.emit import Channel
from app.partial import PartialBuffer


class _TransientError(RuntimeError):
    """Test-only failure injected to prove the loop survives it."""


class _FakeIndicator:
    """Minimal stand-in for the platform indicator facade."""

    def __init__(self) -> None:
        self.flashes: list[str] = []

    def level(self, _value: float) -> None:
        pass

    def flash_listen(self, flash_text: str, _listen_text: str, _ms: int) -> None:
        self.flashes.append(flash_text)


class _FakeRecognizer:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def transcribe(self, _samples: np.ndarray) -> str:
        return self._texts.pop(0)


@dataclass(slots=True)
class _ScriptedSegmenter:
    """Raises on the first block, then lets the loop exit on the second."""

    app: DictationApp
    seen: int = 0

    def accept(self, _block: np.ndarray) -> None:
        self.seen += 1
        if self.seen == 1:
            raise _TransientError
        self.app._continuous = False  # clean exit on the second block

    def is_speech_detected(self) -> bool:
        return False

    def drain(self) -> list[np.ndarray]:
        return []

    def flush(self) -> None:
        pass


def _bare_app() -> DictationApp:
    """A DictationApp with only the fields the continuous loop touches."""
    app = object.__new__(DictationApp)
    app._continuous = True
    app._stop_event = threading.Event()
    app._paused = False
    app._recognizer = None
    app._indicator = _FakeIndicator()
    app._decode_queue = queue.Queue()
    app._partial_slot = LatestSlot()
    app._vad_done = threading.Event()
    return app


def test_loop_survives_a_block_failure() -> None:
    app = _bare_app()
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(8, dtype=np.float32))
    blocks.put(np.zeros(8, dtype=np.float32))
    segmenter = _ScriptedSegmenter(app)

    app._continuous_loop(segmenter, blocks)

    assert segmenter.seen == 2  # reached block two => survived block one
    assert app._continuous is False


def test_insert_segments_isolates_a_failed_emit(monkeypatch) -> None:
    app = _bare_app()
    app._recognizer = _FakeRecognizer(["first", "second", "third"])
    app._config = SimpleNamespace(restore_clipboard=False, paste_delay_ms=0)
    calls: list[str] = []

    def fake_emit(text, settings, indicator):
        calls.append(text)
        if text == "second":
            raise _TransientError
        return Channel.KEYS

    monkeypatch.setattr("app.app.emit_text", fake_emit)
    segments = [np.zeros(8, dtype=np.float32) for _ in range(3)]

    emitted = app._insert_segments(segments)

    assert calls == ["first", "second", "third"]  # the third still ran
    assert emitted == 2  # only the successful deliveries counted


@dataclass(slots=True)
class _SpeechSegmenter:
    """Reports speech, then yields one finished segment on the first drain."""

    app: DictationApp
    drained: bool = False

    def accept(self, _block: np.ndarray) -> None:
        self.app._continuous = False  # exit after this one block

    def is_speech_detected(self) -> bool:
        return True

    def drain(self) -> list[np.ndarray]:
        if self.drained:
            return []
        self.drained = True
        return [np.ones(16, dtype=np.float32)]

    def flush(self) -> None:
        pass


def test_vad_loop_queues_finished_segments_without_decoding() -> None:
    """The VAD thread must never decode: it hands the sentence to the queue."""
    app = _bare_app()
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(8, dtype=np.float32))

    app._continuous_loop(_SpeechSegmenter(app), blocks)

    batch = app._decode_queue.get_nowait()
    assert len(batch.segments) == 1
    assert app._partial_slot.take() is None  # the endpoint cleared the snapshot


def test_decode_loop_emits_a_finished_sentence_before_a_partial(monkeypatch) -> None:
    """A pending sentence must never wait behind partial work."""
    app = _bare_app()
    app._config = SimpleNamespace(restore_clipboard=False, paste_delay_ms=0)
    app._recognizer = _FakeRecognizer(["finished"])
    emitted: list[str] = []

    def fake_emit(text: str, _settings: object, _indicator: object) -> Channel:
        emitted.append(text)
        return Channel.KEYS

    monkeypatch.setattr("app.app.emit_text", fake_emit)
    decode_queue: queue.Queue[FinishedBatch] = queue.Queue()
    decode_queue.put(
        FinishedBatch(
            endpoint_ts=time.monotonic(),
            segments=(np.ones(8, dtype=np.float32),),
        )
    )
    slot: LatestSlot[PartialBuffer] = LatestSlot()
    slot.publish(PartialBuffer().push(np.ones(8, dtype=np.float32)))
    vad_done = threading.Event()
    vad_done.set()  # nothing more will arrive; exit once the queue drains

    app._decode_loop(decode_queue, slot, vad_done)

    assert emitted == ["finished"]
    assert app._indicator.flashes == ["✓ 已上屏"]
    assert slot.take() is not None  # the partial was never decoded
