"""Continuous dictation: the session's loop contracts and its start/stop seam.

Regression guard: the loops are long-lived daemon threads with no restart path,
so a single escaped exception (clipboard contention, a UIA element gone
mid-focus-change, a decode hiccup) used to wedge the mode permanently while the
tray still reported it ON. The loops log and continue; these tests pin that.

Two levels are covered:

* the **interface** — ``start()`` opens the sink and starts the workers,
  ``stop()`` joins them in order and is idempotent, ``running`` tracks both;
* the **loops** — driven synchronously with a scripted segmenter, which needs the
  session's internal collaborators to be in place. That wiring is the one place
  these tests reach inside, and it is five attributes instead of the twelve
  fields the old ``object.__new__(DictationApp)`` helper had to hand-set.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from app.continuous import ContinuousSession, SessionLimits, SessionPorts
from app.decode_scheduler import FinishedBatch, LatestSlot, PartialCadence

from app import continuous

_BASE_INTERVAL_S = 0.35
_LIMITS = SessionLimits(
    min_silence_ms=350,
    max_speech_ms=4000,
    threshold=0.5,
    min_speech_ms=250,
    num_threads=2,
)


class _TransientError(RuntimeError):
    """Test-only failure injected to prove the loop survives it."""


class _RecordingQueue:
    """Stand-in for ``EmitQueue`` that records what the decoder hands over."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def submit(self, text: str, **_kwargs: object) -> bool:
        """Record one queued sentence."""
        self.items.append(text)
        return True


class _FailingRecognizer:
    """Yields three segments; the middle decode raises."""

    def __init__(self) -> None:
        self._texts = iter(["first", "second", "third"])

    def transcribe(self, _samples: object) -> str:
        text = next(self._texts)
        if text == "second":
            raise _TransientError
        return text


class _FakeIndicator:
    """Minimal stand-in for the platform indicator facade."""

    def __init__(self) -> None:
        self.flashes: list[str] = []
        self.updates: list[str] = []
        self.levels = 0

    def level(self, _value: float) -> None:
        self.levels += 1

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
class _Host:
    """The app side of the ports: mutable state plus recording callbacks."""

    recognizer: Any = None
    subtitles: bool = False
    paused: bool = False
    blocks: int = 0
    confirmed: int = 0
    never_started: int = 0
    stopped: int = 0
    sinks: list[queue.Queue[np.ndarray]] = field(default_factory=list)
    partial_texts: list[str] = field(default_factory=list)
    indicator: _FakeIndicator = field(default_factory=_FakeIndicator)
    emit_queue: Any = field(default_factory=_RecordingQueue)
    stop_event: threading.Event = field(default_factory=threading.Event)

    def _confirmed(self) -> None:
        self.confirmed += 1

    def _never_started(self) -> None:
        self.never_started += 1

    def _stop_sink(self) -> None:
        self.stopped += 1

    def ports(self) -> SessionPorts:
        """Build the port bundle the session sees."""
        return SessionPorts(
            stop_event=self.stop_event,
            recognizer=lambda: self.recognizer,
            indicator=self.indicator,
            emit_queue=self.emit_queue,
            subtitles=lambda: self.subtitles,
            paused=lambda: self.paused,
            on_partial_text=self.partial_texts.append,
            on_audio_confirmed=self._confirmed,
            on_audio_never_started=self._never_started,
            start_sink=self.sinks.append,
            stop_sink=self._stop_sink,
            blocks_delivered=lambda: self.blocks,
        )


def _session(host: _Host) -> ContinuousSession:
    return ContinuousSession(
        vad_path=Path("/nonexistent/vad.onnx"), limits=_LIMITS, ports=host.ports()
    )


@dataclass(slots=True)
class _Loop:
    """A session wired for a synchronous loop run, plus the state it reads."""

    session: ContinuousSession
    decode_queue: queue.Queue[FinishedBatch]
    slot: LatestSlot[np.ndarray]
    cadence: PartialCadence
    vad_done: threading.Event


def _armed(host: _Host) -> _Loop:
    """Install the collaborators a loop reads, then return them for assertions."""
    decode_queue: queue.Queue[FinishedBatch] = queue.Queue()
    slot: LatestSlot[np.ndarray] = LatestSlot()
    cadence = PartialCadence(_BASE_INTERVAL_S)
    vad_done = threading.Event()
    session = _session(host)
    session._running = True
    session._decode_queue = decode_queue
    session._partial_slot = slot
    session._partial_cadence = cadence
    session._vad_done = vad_done
    return _Loop(
        session=session,
        decode_queue=decode_queue,
        slot=slot,
        cadence=cadence,
        vad_done=vad_done,
    )


# -- interface ---------------------------------------------------------------


class _IdleSegmenter:
    """Never reports speech: the loops idle until ``stop()`` sets running False."""

    def accept(self, _block: np.ndarray) -> None:
        pass

    def current_sample_count(self) -> int:
        return 0

    def current_samples(self) -> np.ndarray | None:
        return None

    def drain(self) -> list[np.ndarray]:
        return []

    def flush(self) -> None:
        pass


def test_start_opens_the_sink_and_stop_joins_the_workers(monkeypatch) -> None:
    """start()/stop() are the whole interface the app sees."""
    monkeypatch.setattr(continuous, "Segmenter", lambda _settings: _IdleSegmenter())
    host = _Host()
    session = _session(host)

    session.start()

    assert session.running
    assert len(host.sinks) == 1  # the capture sink was attached
    # Hold the Thread objects: stop() clears the attribute, so asserting on it
    # afterwards would be true whether or not anything was joined.
    workers = list(session._threads)
    assert len(workers) == 3
    session.stop()
    assert not session.running
    assert host.stopped == 1  # ... and detached on the way out
    assert not any(worker.is_alive() for worker in workers)  # every worker was joined
    assert session._threads == []


def test_stop_is_idempotent(monkeypatch) -> None:
    """A second stop (worker + watchdog both firing) must not raise."""
    monkeypatch.setattr(continuous, "Segmenter", lambda _settings: _IdleSegmenter())
    host = _Host()
    session = _session(host)
    session.start()

    session.stop()
    session.stop()

    assert host.stopped == 1


# -- VAD loop ----------------------------------------------------------------


@dataclass(slots=True)
class _ScriptedSegmenter:
    """Raises on the first block, then lets the loop exit on the second."""

    session: ContinuousSession
    seen: int = 0

    def accept(self, _block: np.ndarray) -> None:
        self.seen += 1
        if self.seen == 1:
            raise _TransientError
        self.session._running = False  # clean exit on the second block

    def current_sample_count(self) -> int:
        return 0

    def current_samples(self) -> np.ndarray | None:
        return None

    def drain(self) -> list[np.ndarray]:
        return []

    def flush(self) -> None:
        pass


def test_loop_survives_a_block_failure() -> None:
    host = _Host()
    loop = _armed(host)
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(8, dtype=np.float32))
    blocks.put(np.zeros(8, dtype=np.float32))
    segmenter = _ScriptedSegmenter(loop.session)

    loop.session._vad_loop(segmenter, blocks)  # type: ignore[arg-type]

    assert segmenter.seen == 2  # reached block two => survived block one
    assert not loop.session.running


def test_insert_segments_isolates_a_failed_decode() -> None:
    """One failed segment must not stop the rest reaching the emitter."""
    host = _Host(recognizer=_FailingRecognizer())
    loop = _armed(host)
    segments = [np.zeros(8, dtype=np.float32) for _ in range(3)]

    emitted = loop.session._insert_segments(segments)

    assert host.emit_queue.items == ["first", "third"]  # the third still ran
    assert emitted == 2  # only the successful decodes counted


@dataclass(slots=True)
class _SpeechSegmenter:
    """Reports speech, then yields one finished segment on the first drain."""

    session: ContinuousSession
    drained: bool = False

    def accept(self, _block: np.ndarray) -> None:
        self.session._running = False  # exit after this one block

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
    host = _Host()
    loop = _armed(host)
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(8, dtype=np.float32))

    loop.session._vad_loop(_SpeechSegmenter(loop.session), blocks)  # type: ignore[arg-type]

    batch = loop.decode_queue.get_nowait()
    assert len(batch.segments) == 1
    assert loop.slot.take() is None  # the endpoint cleared the snapshot


@dataclass(slots=True)
class _GrowingSegmenter:
    """Reports a distinct in-progress snapshot, exiting on the second block."""

    session: ContinuousSession
    calls: int = 0
    snapshots: list[np.ndarray] = field(default_factory=list)

    def accept(self, _block: np.ndarray) -> None:
        self.calls += 1
        if self.calls >= 2:
            self.session._running = False  # exit after the second block

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
    """
    host = _Host()
    loop = _armed(host)
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))  # every fed block is zeros
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _GrowingSegmenter(loop.session)

    loop.session._vad_loop(segmenter, blocks)  # type: ignore[arg-type]

    snapshot = loop.slot.take()
    assert snapshot is not None
    assert any(np.array_equal(snapshot, produced) for produced in segmenter.snapshots)
    assert not np.array_equal(snapshot, np.zeros(2, dtype=np.float32))


def test_vad_loop_throttles_the_snapshot_to_one_pending_copy() -> None:
    """Only one snapshot may be pending: the growing utterance is not re-copied.

    Regression: the loop copied the whole utterance into the slot on every
    speech block (O(N^2) memcpy). Now the single-slot emptiness check gates the
    copy, so a second block while a snapshot is pending produces no new copy.
    """
    host = _Host()
    loop = _armed(host)
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _GrowingSegmenter(loop.session)

    loop.session._vad_loop(segmenter, blocks)  # type: ignore[arg-type]

    assert len(segmenter.snapshots) == 1  # only the first block copied


@dataclass(slots=True)
class _CeilingSegmenter:
    """Reports an utterance past the preview ceiling and exits immediately."""

    session: ContinuousSession
    copied: int = 0

    def accept(self, _block: np.ndarray) -> None:
        self.session._running = False  # exit after this one block

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
    host = _Host()
    loop = _armed(host)
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    blocks.put(np.zeros(4, dtype=np.float32))
    segmenter = _CeilingSegmenter(loop.session)

    loop.session._vad_loop(segmenter, blocks)  # type: ignore[arg-type]

    assert segmenter.copied == 0  # the O(N) copy was never paid
    assert loop.slot.take() is None


def test_decode_loop_hands_finished_sentences_to_the_emitter() -> None:
    """The finished path owns its own thread and never touches partial work."""
    host = _Host(recognizer=_FakeRecognizer(["finished"]))
    loop = _armed(host)
    loop.decode_queue.put(
        FinishedBatch(
            endpoint_ts=time.monotonic(),
            segments=(np.ones(8, dtype=np.float32),),
        )
    )
    slot: LatestSlot[np.ndarray] = LatestSlot()
    slot.publish(np.ones(8, dtype=np.float32))
    vad_done = threading.Event()
    vad_done.set()  # nothing more will arrive; exit once the queue drains

    loop.session._decode_loop(loop.decode_queue, vad_done)

    assert host.emit_queue.items == ["finished"]
    assert slot.take() is not None  # the partial is the partial thread's job


class _StopRecognizer:
    """Returns fixed text and stops the partial loop after the first decode."""

    def __init__(self, session: ContinuousSession, text: str) -> None:
        self._session = session
        self._text = text

    def transcribe(self, _samples: np.ndarray) -> str:
        self._session._running = False
        return self._text


def test_partial_loop_decodes_the_published_snapshot() -> None:
    """The partial thread consumes the VAD snapshot and refreshes the pill."""
    host = _Host()
    loop = _armed(host)
    host.recognizer = _StopRecognizer(loop.session, "你好")
    loop.slot.publish(np.ones(8, dtype=np.float32))

    loop.session._partial_loop(loop.slot, loop.cadence)

    assert host.indicator.updates == ["● 你好"]
    assert loop.slot.take() is None  # the snapshot was consumed
    assert host.partial_texts == ["你好"]  # the host was offered the caption


def test_show_partial_skips_while_the_recognizer_is_none() -> None:
    """The model-swap window leaves no recognizer; a partial must simply skip."""
    host = _Host(recognizer=None)
    session = _session(host)

    assert session._show_partial(np.ones(8, dtype=np.float32), "") is None
    assert host.indicator.updates == []
