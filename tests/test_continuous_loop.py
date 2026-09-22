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
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from app.app import DictationApp
from app.decode_scheduler import FinishedBatch, LatestSlot, PartialCadence
from app.emit import Channel

_BASE_INTERVAL_S = 0.35


class _TransientError(RuntimeError):
    """Test-only failure injected to prove the loop survives it."""


class _FakeIndicator:
    """Minimal stand-in for the platform indicator facade."""

    def __init__(self) -> None:
        self.flashes: list[str] = []
        self.updates: list[str] = []

    def level(self, _value: float) -> None:
        pass

    def update(self, text: str) -> None:
        self.updates.append(text)

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

    def current_sample_count(self) -> int:
        return 0

    def current_samples(self) -> np.ndarray | None:
        return None

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
    app._partial_cadence = PartialCadence(_BASE_INTERVAL_S)
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

    def current_sample_count(self) -> int:
        return 8

    def current_samples(self) -> np.ndarray | None:
        return np.ones(8, dtype=np.float32)

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


@dataclass(slots=True)
class _GrowingSegmenter:
    """Reports a distinct in-progress snapshot, exiting on the second block."""

    app: DictationApp
    calls: int = 0
    snapshots: list[np.ndarray] = field(default_factory=list)

    def accept(self, _block: np.ndarray) -> None:
        self.calls += 1
        if self.calls >= 2:
            self.app._continuous = False  # exit after the second block

    def current_sample_count(self) -> int:
        return self.calls * 2

    def current_samples(self) -> np.ndarray | None:
        snapshot = np.full(self.calls * 2, float(self.calls), dtype=np.float32)
        self.snapshots.append(snapshot)
        return snapshot

    def drain(self) -> list[np.ndarray]:
        return []

    def flush(self) -> None:
        pass


def test_vad_loop_publishes_the_vads_current_samples_not_accumulated_blocks() -> None:
    """The live partial must be the VAD's own segment, not the raw fed blocks.

    Regression: the loop used to accumulate the blocks it fed while the VAD
    reported speech, which misses the onset (``is_speech_detected`` flips only
    after the VAD buffered it) and so decoded different audio from the insert.
    The published snapshot must be exactly a ``current_samples()`` return.
    """
    app = _bare_app()
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))  # every fed block is zeros
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _GrowingSegmenter(app)

    app._continuous_loop(segmenter, blocks)

    snapshot = app._partial_slot.take()
    assert snapshot is not None
    # A raw accumulated buffer would be zeros, so this pins the partial audio to
    # the VAD's own segment.
    assert any(np.array_equal(snapshot, produced) for produced in segmenter.snapshots)
    assert not np.array_equal(snapshot, np.zeros(2, dtype=np.float32))


def test_vad_loop_throttles_the_snapshot_to_one_pending_copy() -> None:
    """Only one snapshot may be pending: the growing utterance is not re-copied.

    Regression: the loop copied the whole utterance into the slot on every
    speech block (O(N^2) memcpy). Now the single-slot emptiness check gates the
    copy, so a second block while a snapshot is pending produces no new copy.
    """
    app = _bare_app()
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _GrowingSegmenter(app)

    app._continuous_loop(segmenter, blocks)

    assert len(segmenter.snapshots) == 1  # only the first block copied


@dataclass(slots=True)
class _CeilingSegmenter:
    """Reports an utterance past the preview ceiling and exits immediately."""

    app: DictationApp
    copied: int = 0

    def accept(self, _block: np.ndarray) -> None:
        self.app._continuous = False  # exit after this one block

    def current_sample_count(self) -> int:
        return 16_000 * 60  # 60 s, far past the 6 s partial ceiling

    def current_samples(self) -> np.ndarray | None:
        self.copied += 1
        return np.ones(8, dtype=np.float32)

    def drain(self) -> list[np.ndarray]:
        return []

    def flush(self) -> None:
        pass


def test_vad_loop_skips_the_snapshot_past_the_partial_ceiling() -> None:
    """A long utterance gets no live partial, so it never starves the final."""
    app = _bare_app()
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _CeilingSegmenter(app)

    app._continuous_loop(segmenter, blocks)

    assert segmenter.copied == 0  # the O(N) copy was never paid
    assert app._partial_slot.take() is None


def test_decode_loop_emits_finished_sentences_and_ignores_partials(monkeypatch) -> None:
    """The finished path owns its own thread and never touches partial work."""
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
    slot: LatestSlot[np.ndarray] = LatestSlot()
    slot.publish(np.ones(8, dtype=np.float32))
    vad_done = threading.Event()
    vad_done.set()  # nothing more will arrive; exit once the queue drains

    app._decode_loop(decode_queue, vad_done)

    assert emitted == ["finished"]
    assert app._indicator.flashes == ["✓ 已上屏"]
    assert slot.take() is not None  # the partial is the partial thread's job


class _StopRecognizer:
    """Returns fixed text and stops the partial loop after the first decode."""

    def __init__(self, app: DictationApp, text: str) -> None:
        self._app = app
        self._text = text

    def transcribe(self, _samples: np.ndarray) -> str:
        self._app._continuous = False
        return self._text


def test_partial_loop_decodes_the_published_snapshot() -> None:
    """The partial thread consumes the VAD snapshot and refreshes the pill."""
    app = _bare_app()
    app._recognizer = _StopRecognizer(app, "你好")
    slot: LatestSlot[np.ndarray] = LatestSlot()
    slot.publish(np.ones(8, dtype=np.float32))
    cadence = PartialCadence(_BASE_INTERVAL_S)

    app._partial_loop(slot, cadence)

    assert app._indicator.updates == ["● 你好"]
    assert slot.take() is None  # the snapshot was consumed


def test_show_partial_skips_while_the_recognizer_is_none() -> None:
    """The model-swap window leaves no recognizer; a partial must simply skip."""
    app = _bare_app()
    app._recognizer = None

    assert app._show_partial(np.ones(8, dtype=np.float32), "") is None
    assert app._indicator.updates == []
