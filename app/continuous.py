"""The continuous-dictation session: one module for one live transcript.

Continuous dictation used to exist only as eleven attributes on ``DictationApp``
(three worker threads, two queues, a slot, a cadence, a done-flag, a segment
cap), so its interface *was* its implementation: every race fix had to be made
inside a 1,483-line class, and the tests had to reach past the interface
(``object.__new__(DictationApp)`` plus hand-set fields) to exercise the loop.

This module owns that state. The interface is :meth:`ContinuousSession.start`,
:meth:`ContinuousSession.stop` and :attr:`ContinuousSession.running`; behind it
sit the VAD loop, the finished-sentence decoder, the live partial, the audio
watchdog and their join order. What the session needs from its host arrives as
:class:`SessionPorts` — callables, not objects — so the session imports no Tk,
no tray and no config: the app stays the policy layer, the session is the
concurrency.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app import logsafe
from app.audio_flow import plan_recovery
from app.decode_scheduler import (
    FinishedBatch,
    LatestSlot,
    PartialCadence,
    within_partial_ceiling,
)
from app.partial import level_from_block, new_text_or_none, truncate_partial
from app.vad import SAMPLE_RATE, Segmenter, VadSettings

if TYPE_CHECKING:
    from pathlib import Path

    from app.asr import Recognizer

log = logging.getLogger(__name__)

#: Bounded continuous-capture queue: the audio callback drops the oldest block
#: when this fills, so a slow decode can never stall the microphone stream.
CONTINUOUS_QUEUE_MAX: int = 128

#: Bounded finished-sentence queue: the VAD thread drops the oldest batch when
#: the decoder lags, so a slow model can never make memory grow without limit.
DECODE_QUEUE_MAX: int = 16

#: How long the decode thread blocks for a finished batch before it checks
#: whether the VAD is done; small so an endpoint is picked up promptly.
DECODE_POLL_S: float = 0.05

#: Minimum gap between live partial decodes; a slower decode simply slips.
PARTIAL_INTERVAL_S: float = 0.35

#: How long :meth:`ContinuousSession.stop` waits for each worker to unwind.
JOIN_TIMEOUT_S: float = 2.0

#: Sticky pill shown while continuous mode is listening.
LISTEN_TEXT: str = "● 聆听中"
#: Confirmation flashed at each endpoint before returning to listening.
COMMIT_TEXT: str = "✓ 已上屏"
COMMIT_FLASH_MS: int = 900


class EmitSink(Protocol):
    """What the session needs from the delivery side: queue one sentence."""

    def submit(
        self, text: str, *, endpoint_ts: float | None = None, forced: bool = False
    ) -> bool:
        """Queue one finished sentence; False when an older one had to go."""
        ...


@dataclass(frozen=True, slots=True)
class SessionLimits:
    """What one session needs to build its VAD, decided by the config layer."""

    min_silence_ms: int
    max_speech_ms: int
    threshold: float
    min_speech_ms: int
    num_threads: int


@dataclass(frozen=True, slots=True)
class SessionPorts:
    """Everything the session needs from its host, as callables.

    The host owns policy (which device, which recognizer, where text goes, what
    the tray says); the session owns the concurrency. Passing callables rather
    than the host itself is what keeps the two from growing back together.
    """

    #: Shutdown flag owned by the app; the workers exit when it is set.
    stop_event: threading.Event
    #: Re-read per decode: the model may be swapped (or briefly None) mid-session.
    recognizer: Callable[[], Recognizer | None]
    #: The platform pill facade (``level`` / ``update`` / ``flash_listen``).
    indicator: Any
    #: Bounded ASR→translation sink; the session never translates or types.
    emit_queue: EmitSink
    #: True when the subtitle bar is the visible surface (suppresses pill noise).
    subtitles: Callable[[], bool]
    #: True while playback is paused: no decode, no emit.
    paused: Callable[[], bool]
    #: Hand an in-progress caption to the host (translation + provisional line).
    on_partial_text: Callable[[str], None]
    #: Audio is demonstrably flowing (the host re-renders the tray tick).
    on_audio_confirmed: Callable[[], None]
    #: Audio never started: the host stops the session, persists "off" and says so.
    on_audio_never_started: Callable[[], None]
    #: Attach the capture sink (also used to reopen a stalled stream).
    start_sink: Callable[[queue.Queue[np.ndarray]], None]
    #: Detach the capture sink.
    stop_sink: Callable[[], None]
    #: Blocks delivered so far, for the "is audio actually arriving" watchdog.
    blocks_delivered: Callable[[], int]


class ContinuousSession:
    """One live transcript: VAD feed → finished-sentence decode → live partial."""

    def __init__(
        self,
        *,
        vad_path: Path,
        limits: SessionLimits,
        ports: SessionPorts,
    ) -> None:
        self._vad_path = vad_path
        self._limits = limits
        self._ports = ports
        self._max_speech_s = limits.max_speech_ms / 1000
        self._running = False
        self._blocks: queue.Queue[np.ndarray] | None = None
        self._decode_queue: queue.Queue[FinishedBatch] | None = None
        self._partial_slot: LatestSlot[np.ndarray] | None = None
        self._partial_cadence: PartialCadence | None = None
        self._vad_done: threading.Event | None = None
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._running

    # -- interface -----------------------------------------------------------

    def start(self) -> None:
        """Open the sink and start the workers.

        Three workers, so no role can stall another: the VAD consumer stays
        real-time (never decodes), the finished-sentence decoder always wins,
        and the live partial runs on its own thread so an in-flight partial (up
        to ~1 s on Fun-ASR-Nano) can never delay an endpoint's text.
        """
        if self._running:
            return
        segmenter = Segmenter(
            VadSettings(
                model_path=self._vad_path,
                threshold=self._limits.threshold,
                min_speech_s=self._limits.min_speech_ms / 1000,
                min_silence_s=self._limits.min_silence_ms / 1000,
                max_speech_s=self._limits.max_speech_ms / 1000,
                num_threads=self._limits.num_threads,
            )
        )
        blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=CONTINUOUS_QUEUE_MAX)
        decode_queue: queue.Queue[FinishedBatch] = queue.Queue(maxsize=DECODE_QUEUE_MAX)
        slot: LatestSlot[np.ndarray] = LatestSlot()
        cadence = PartialCadence(PARTIAL_INTERVAL_S)
        vad_done = threading.Event()
        self._blocks = blocks
        self._decode_queue = decode_queue
        self._partial_slot = slot
        self._partial_cadence = cadence
        self._vad_done = vad_done
        self._ports.start_sink(blocks)
        self._running = True
        self._threads = [
            threading.Thread(
                target=self._vad_loop,
                args=(segmenter, blocks),
                daemon=True,
                name="continuous-vad",
            ),
            threading.Thread(
                target=self._decode_loop,
                args=(decode_queue, vad_done),
                daemon=True,
                name="continuous-decode",
            ),
            threading.Thread(
                target=self._partial_loop,
                args=(slot, cadence),
                daemon=True,
                name="continuous-partial",
            ),
        ]
        for thread in self._threads:
            thread.start()
        # A busy endpoint looks identical to a working one from the outside.
        threading.Thread(target=self._watch_audio, daemon=True, name="audio-watchdog").start()

    def stop(self) -> None:
        """Stop the workers, in the order the tail of audio needs.

        The VAD thread goes first: its final flush queues the last utterance for
        the decoder, which then drains it before exiting. Nothing here persists
        state or touches the tray — that is the host's policy, applied after.
        """
        if not self._running:
            return
        self._running = False
        self._ports.stop_sink()
        for thread in self._threads:
            thread.join(timeout=JOIN_TIMEOUT_S)
        self._threads = []
        self._blocks = None
        self._decode_queue = None
        self._partial_slot = None
        self._partial_cadence = None
        self._vad_done = None

    def reconnect(self) -> None:
        """Re-attach the capture sink after the host swapped its recorder."""
        if self._running and self._blocks is not None:
            self._ports.start_sink(self._blocks)

    # -- VAD thread ----------------------------------------------------------

    def _vad_loop(self, segmenter: Segmenter, blocks: queue.Queue[np.ndarray]) -> None:
        """Daemon thread: feed the VAD and publish partials; never decode.

        This is the responsiveness-critical path. It consumes audio blocks, runs
        the VAD, hands finished segments to the decode queue, and — only when a
        partial is actually due — snapshots the VAD's own in-progress segment
        (the exact audio that will become the finished sentence, onset
        included) into the single-slot holder. It performs no decoding at all,
        so a slow model (Fun-ASR-Nano) can neither stall the VAD feed nor delay
        endpointing.
        """
        decode_queue = self._decode_queue
        slot = self._partial_slot
        cadence = self._partial_cadence
        vad_done = self._vad_done
        skipped_partials = 0
        while self._running and not self._ports.stop_event.is_set():
            try:
                block = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            # A transient failure (clipboard contention, a UIA element that went
            # away mid-focus-change, a decode hiccup) must never kill this
            # long-lived thread: continuous dictation has no restart path, so an
            # escaped exception would silently wedge the mode until the user
            # toggles it off and on again.
            try:
                self._ports.indicator.level(level_from_block(block))
                segmenter.accept(block)
                samples = self._due_partial(segmenter, slot, cadence)
                if samples is not None:
                    self._publish_partial(slot, samples)
                elif slot is not None and slot.has_value():
                    skipped_partials += 1  # a snapshot was already pending
                segments = segmenter.drain()
                if segments:  # endpoint: the utterance is complete
                    self._finish_utterance(slot, decode_queue, segments)
            except Exception:
                log.exception("continuous: block processing failed; loop continues")
        self._flush_segments(segmenter, decode_queue)
        if vad_done is not None:
            vad_done.set()  # tells the decoder to drain the tail and exit
        log.info("continuous: vad loop exited (deferred %d pending partials)", skipped_partials)

    def _due_partial(
        self,
        segmenter: Segmenter,
        slot: LatestSlot[np.ndarray] | None,
        cadence: PartialCadence | None,
    ) -> np.ndarray | None:
        """Snapshot the in-progress utterance only when a partial is due.

        The VAD loop used to copy the whole growing utterance on every speech
        block (O(N²) memcpy). Here the cheap O(1) length gate, the single-slot
        emptiness check, and the shared adaptive cadence all run *before* the
        O(N) copy, so the copy is paid only when the pill is about to refresh.
        Returns ``None`` when no partial is due (paused, already pending, in
        flight, or past the preview ceiling).
        """
        if slot is None or cadence is None or self._ports.paused():
            return None
        if slot.has_value():  # a snapshot is already pending; do not re-copy
            return None
        duration_s = segmenter.current_sample_count() / SAMPLE_RATE
        if not within_partial_ceiling(duration_s):
            return None
        if not cadence.should_publish(time.monotonic(), paused=self._ports.paused()):
            return None
        return segmenter.current_samples()

    def _publish_partial(
        self, slot: LatestSlot[np.ndarray] | None, samples: np.ndarray
    ) -> bool:
        """Publish the newest in-progress samples; True when one was superseded."""
        if slot is None or self._ports.paused():
            return False
        return slot.publish(samples)

    def _finish_utterance(
        self,
        slot: LatestSlot[np.ndarray] | None,
        decode_queue: queue.Queue[FinishedBatch] | None,
        segments: Sequence[np.ndarray],
    ) -> None:
        """Endpoint: drop the stale snapshot and queue the sentence for decoding.

        Resets the partial cadence too, so the next utterance starts from a
        fresh caption and cadence (the partial thread clears its last caption on
        the generation bump).
        """
        if slot is not None:
            slot.clear()
        cadence = self._partial_cadence
        if cadence is not None:
            cadence.reset(time.monotonic())
        if decode_queue is not None and not self._ports.paused():
            _offer_finished(
                decode_queue,
                FinishedBatch(endpoint_ts=time.monotonic(), segments=tuple(segments)),
            )

    def _flush_segments(
        self,
        segmenter: Segmenter,
        decode_queue: queue.Queue[FinishedBatch] | None,
    ) -> None:
        """End the VAD stream and queue whatever tail it releases for decoding."""
        try:
            segmenter.flush()
            final = segmenter.drain()
        except Exception:
            log.exception("continuous: final flush failed")
            return
        if final:
            self._finish_utterance(None, decode_queue, final)

    # -- decode thread -------------------------------------------------------

    def _decode_loop(
        self,
        decode_queue: queue.Queue[FinishedBatch],
        vad_done: threading.Event,
    ) -> None:
        """Daemon thread: decode finished sentences, and only finished sentences.

        This thread never touches the live partial, so an endpoint's sentence is
        decoded the moment the VAD queues it, never waiting behind a partial
        decode already in flight.
        """
        finished_segments = 0
        while True:
            try:
                batch = decode_queue.get(timeout=DECODE_POLL_S)
            except queue.Empty:
                batch = None
            if batch is not None:
                emitted = self._insert_segments(
                    batch.segments, endpoint_ts=batch.endpoint_ts
                )
                finished_segments += len(batch.segments)
                if emitted and not self._ports.subtitles():
                    self._ports.indicator.flash_listen(
                        COMMIT_TEXT, LISTEN_TEXT, COMMIT_FLASH_MS
                    )
                continue
            if vad_done.is_set():
                break
        log.info("continuous: decode loop exited (finished segments=%d)", finished_segments)

    def _insert_segments(
        self,
        segments: Sequence[np.ndarray],
        *,
        endpoint_ts: float | None = None,
    ) -> int:
        """Decode each finished segment and hand it to the emit queue.

        Returns how many were handed over. Each segment is isolated: a single
        failed decode or emit is logged and skipped so the remaining sentences
        still land and the loop survives. ``endpoint_ts`` (when the VAD saw the
        pause) turns each success into an endpoint→emit latency line in
        ``app.log``.
        """
        emitted = 0
        for samples in segments:
            if self._ports.stop_event.is_set() or self._ports.paused():
                return emitted
            try:
                recognizer = self._ports.recognizer()
                if recognizer is None:
                    continue
                log.info("continuous: segment %.2f s", samples.shape[0] / SAMPLE_RATE)
                started = time.monotonic()
                text = recognizer.transcribe(samples)
                log.info("continuous: decode %.3f s", time.monotonic() - started)
                if not text:
                    continue
                log.info("continuous asr: %s", logsafe.describe_text(text))
                asr_done = time.monotonic()
                if endpoint_ts is not None:
                    log.info("continuous: endpoint→asr %.3f s", asr_done - endpoint_ts)
                seconds = samples.shape[0] / SAMPLE_RATE
                forced = seconds >= self._max_speech_s - 0.15
                if not self._ports.emit_queue.submit(
                    text, endpoint_ts=endpoint_ts, forced=forced
                ):
                    log.info("continuous: emit queue full, dropped an older sentence")
            except Exception:
                log.exception("continuous: segment decode failed; continuing")
                continue
            emitted += 1
        return emitted

    # -- partial thread ------------------------------------------------------

    def _partial_loop(
        self,
        slot: LatestSlot[np.ndarray],
        cadence: PartialCadence,
    ) -> None:
        """Daemon thread: refresh the live preview at the adaptive cadence.

        Separate from the finished-sentence decoder so a slow partial decode can
        never delay an endpoint's text. It decodes exactly what the VAD
        published — the full in-progress utterance — so the preview stays
        faithful to the sentence that will be inserted.
        """
        last_partial_text = ""
        seen_generation = cadence.generation()
        partials = 0
        while self._running and not self._ports.stop_event.is_set():
            if self._ports.paused():
                time.sleep(DECODE_POLL_S)
                continue
            generation = cadence.generation()
            if generation != seen_generation:
                seen_generation = generation
                last_partial_text = ""  # endpoint boundary: next utterance is fresh
            snapshot = slot.take()
            if snapshot is None:
                time.sleep(DECODE_POLL_S)
                continue
            started = time.monotonic()
            cadence.mark_start(started)
            try:
                shown = self._show_partial(snapshot, last_partial_text)
            except Exception:
                # A transient native decode failure must not wedge the cadence
                # (mark_end must still run) nor kill this long-lived thread.
                log.exception("continuous: partial decode failed; continuing")
                shown = None
            finally:
                duration = time.monotonic() - started
                cadence.mark_end(duration)
            log.info("continuous: partial decode %.3f s", duration)
            if shown is not None:
                last_partial_text = shown
                partials += 1
        log.info("continuous: partial loop exited (partials decoded=%d)", partials)

    def _show_partial(self, samples: np.ndarray, previous: str) -> str | None:
        """Decode the in-progress samples and refresh the preview; return the caption.

        *samples* is the VAD's own current segment — the audio that will become
        the finished sentence — so the caption is a faithful preview of the text
        that will be inserted. Returns the caption now shown, or ``None`` when
        there was nothing new to show (empty decode, or the same caption as the
        last frame — so the pill cannot flicker).
        """
        recognizer = self._ports.recognizer()
        if recognizer is None:
            return None
        text = recognizer.transcribe(samples)
        if not text:
            return None
        shown = new_text_or_none(previous, truncate_partial(text))
        if shown is None:
            return None
        # With the subtitle bar on, the pill would just duplicate the caption
        # one line higher — keep it quiet and let the bar do the talking.
        if not self._ports.subtitles():
            self._ports.indicator.update(f"● {shown}")
        self._ports.on_partial_text(text)
        return shown

    # -- watchdog ------------------------------------------------------------

    def _watch_audio(self) -> None:
        """Prove audio is actually arriving; recover once, then fail loudly.

        Symptom this fixes: the switch showed ON after a restart but nothing was
        transcribed until it was toggled off and on. The stream can fail to open
        while the audio endpoint is still initialising, and a silent stream looks
        exactly like a healthy one from the outside.
        """
        baseline = self._ports.blocks_delivered()
        reopens = 0
        while not self._ports.stop_event.is_set():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not self._running:
                    return
                verdict = plan_recovery(
                    blocks=self._ports.blocks_delivered(),
                    baseline=baseline,
                    reopens_done=reopens,
                )
                if verdict == "ok":
                    log.info(
                        "continuous: audio confirmed (%d blocks)",
                        self._ports.blocks_delivered() - baseline,
                    )
                    self._ports.on_audio_confirmed()
                    return
                time.sleep(0.25)
            verdict = plan_recovery(
                blocks=self._ports.blocks_delivered(),
                baseline=baseline,
                reopens_done=reopens,
            )
            if verdict == "ok":
                continue
            if verdict == "give-up":
                break
            reopens += 1
            log.warning("continuous: no audio after 5s — reopening the stream")
            with contextlib.suppress(Exception):
                self._ports.stop_sink()
                if self._blocks is not None:
                    self._ports.start_sink(self._blocks)
        # Still silent: do not leave a switch that lies about working.
        log.error("continuous: audio never started; turning the mode off")
        self._ports.on_audio_never_started()


def _offer_finished(sink: queue.Queue[FinishedBatch], batch: FinishedBatch) -> None:
    """Enqueue a finished batch, dropping the oldest when the decoder lags.

    The decode queue is bounded so a slow decoder can never make memory grow
    without limit; the newest sentence is the one the user most wants, so the
    oldest pending batch is sacrificed and the drop is logged.
    """
    try:
        sink.put_nowait(batch)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            dropped = sink.get_nowait()
            log.warning(
                "continuous: decode queue full; dropped %d finished segment(s)",
                len(dropped.segments),
            )
        with contextlib.suppress(queue.Full):
            sink.put_nowait(batch)
