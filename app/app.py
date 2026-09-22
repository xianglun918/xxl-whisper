"""Orchestrator: hotkey → recorder → recognizer → paste, driven by the tray.

Thread map: main thread runs the tray loop; the hook threads (keyboard and
mouse) pump Win32 messages; PortAudio runs its own callback thread; the ASR
worker owns the detector + recorder state machine so no locks are needed
around them. Tray-driven control actions live in app.controls.
"""

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import assert_never

import numpy as np

from app import __version__, native
from app.asr import Recognizer
from app.config import MOUSE_VKS, Config, config_path, hotkey_vk, models_root, save_config
from app.controls import Controls, ControlsDeps
from app.decode_scheduler import (
    DecodeKind,
    FinishedBatch,
    LatestSlot,
    next_decode,
    partial_interval,
)
from app.downloader import (
    DownloadError,
    ensure_model,
    ensure_vad_model,
    manual_download_guide,
)
from app.emit import EmitSettings, emit_text
from app.hotkey_logic import (
    Action,
    Click,
    EndHold,
    HoldClickDetector,
    Press,
    Release,
    StartHold,
)
from app.partial import level_from_block, new_text_or_none, truncate_partial
from app.recorder import Recorder
from app.tray import Tray, TrayCallbacks, TrayState
from app.update_flow import UpdateFlow
from app.vad import Segmenter, VadSettings

log = logging.getLogger(__name__)

_VK_DISABLED: int = 0
_VK_ESCAPE: int = 0x1B

#: Bounded continuous-capture queue: the audio callback drops the oldest block
#: when this fills, so a slow decode can never stall the microphone stream.
_CONTINUOUS_QUEUE_MAX: int = 128

#: Bounded finished-sentence queue: the VAD thread drops the oldest batch when
#: the decoder lags, so a slow model can never make memory grow without limit.
_DECODE_QUEUE_MAX: int = 16

#: How long the decode thread blocks for a finished batch before it considers a
#: live partial; small so an endpoint is picked up promptly.
_DECODE_POLL_S: float = 0.05

#: Sticky pill shown while continuous mode is listening.
_LISTEN_TEXT: str = "● 聆听中"
#: Confirmation flashed at each endpoint before returning to listening.
_COMMIT_TEXT: str = "✓ 已上屏"
_COMMIT_FLASH_MS: int = 900
#: Minimum gap between live partial decodes; a slower decode simply slips.
_PARTIAL_INTERVAL_S: float = 0.35


@dataclass(frozen=True, slots=True)
class KeyTransition:
    pressed: bool
    ts_ms: int


@dataclass(frozen=True, slots=True)
class SetPaused:
    paused: bool


@dataclass(frozen=True, slots=True)
class SetMic:
    name: str


@dataclass(frozen=True, slots=True)
class SetHotkey:
    key: str | int


@dataclass(frozen=True, slots=True)
class CaptureHotkey:
    pass


@dataclass(frozen=True, slots=True)
class SetModel:
    kind: str


@dataclass(frozen=True, slots=True)
class SetDisfluency:
    pass


@dataclass(frozen=True, slots=True)
class SetContinuous:
    pass


@dataclass(frozen=True, slots=True)
class InitModel:
    pass


type WorkerMsg = (
    KeyTransition
    | SetPaused
    | SetMic
    | SetHotkey
    | CaptureHotkey
    | SetModel
    | SetDisfluency
    | SetContinuous
    | InitModel
)


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


class DictationApp:
    """Owns every component; ``run()`` blocks until the user exits the tray."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[WorkerMsg] = queue.Queue()
        self._detector = HoldClickDetector(config.hold_threshold_ms)
        self._indicator = native.indicator.Indicator()
        self._recognizer: Recognizer | None = None
        self._recorder: Recorder | None = None
        self._hook: native.hotkey.HotkeyHook | None = None
        self._mouse_hook: native.mousehook.MouseHook | None = None
        self._paused = False
        self._ready = False
        self._skip_hold = False
        self._holding = False
        self._hold_confirm_timer: threading.Timer | None = None
        self._model_downloaded = False
        self._model_ready = False
        self._continuous = False
        self._continuous_thread: threading.Thread | None = None
        self._decode_thread: threading.Thread | None = None
        self._continuous_queue: queue.Queue[np.ndarray] | None = None
        self._decode_queue: queue.Queue[FinishedBatch] | None = None
        self._partial_slot: LatestSlot[np.ndarray] | None = None
        self._vad_done: threading.Event | None = None
        self._stop_event = threading.Event()
        self._tray = Tray(
            callbacks=TrayCallbacks(
                on_exit=self._request_exit,
                on_toggle_pause=lambda: self._queue.put(SetPaused(not self._paused)),
                on_select_mic=lambda name: self._queue.put(SetMic(name=name)),
                on_toggle_autostart=self._on_toggle_autostart,
                on_check_update=lambda: threading.Thread(
                    target=self._updates.manual_check, daemon=True
                ).start(),
                on_select_hotkey=lambda key: self._queue.put(SetHotkey(key=key)),
                on_capture_hotkey=lambda: self._queue.put(CaptureHotkey()),
                on_select_model=lambda kind: self._queue.put(SetModel(kind=kind)),
                on_show_diagnostics=self._show_diagnostics_deferred,
                on_toggle_disfluency=lambda: self._queue.put(SetDisfluency()),
                on_toggle_continuous=lambda: self._queue.put(SetContinuous()),
            ),
            state_provider=self._tray_state,
        )
        self._updates = UpdateFlow(tray=self._tray, current_version=__version__)
        self._controls = Controls(
            ControlsDeps(
                indicator=self._indicator,
                tray=self._tray,
                get_config=lambda: self._config,
                set_config=self._set_config,
                get_hooks=lambda: (self._hook, self._mouse_hook),
                get_recorder=lambda: self._recorder,
                rebuild_recorder=self._rebuild_recorder,
                models_root=models_root(),
                num_threads=lambda: self._config.num_threads,
                language=lambda: self._config.language,
            )
        )

    def run(self) -> None:
        self._recorder = Recorder(
            device_name=self._config.mic,
            on_stream_error=lambda msg: log.warning("%s", msg),
        )
        vk = hotkey_vk(self._config.hotkey)
        self._hook = native.hotkey.HotkeyHook(
            vk=_VK_DISABLED if vk in MOUSE_VKS else vk,
            on_transition=self._on_transition,
            disarmed_prompt=self._show_loading_prompt,
        )
        self._hook.start_and_wait()
        self._mouse_hook = native.mousehook.MouseHook(
            vk=vk if vk in MOUSE_VKS else _VK_DISABLED,
            on_transition=self._on_transition,
            disarmed_prompt=self._show_loading_prompt,
        )
        self._mouse_hook.start_and_wait()
        worker = threading.Thread(target=self._worker, daemon=True, name="asr-worker")
        worker.start()
        # Model download + load run on the worker so the tray is responsive
        # immediately and Exit can abort an in-flight download.
        self._queue.put(InitModel())
        self._ready = True
        self._updates.start_watcher(self._config.check_updates)
        log.info(
            "ready: hotkey=%s mic=%r model=%s (loading)",
            self._config.hotkey,
            self._config.mic,
            self._config.model,
        )
        threading.Thread(
            target=native.util.prompt_permissions, daemon=True, name="permissions-prompt"
        ).start()
        try:
            self._tray.run()
        finally:
            self._teardown()

    # -- wiring callbacks (foreign threads) ---------------------------------

    def _on_transition(self, pressed: bool) -> None:
        self._queue.put(
            KeyTransition(pressed=pressed, ts_ms=time.perf_counter_ns() // 1_000_000)
        )

    def _on_model_progress(self, filename: str, downloaded: int, total: int) -> None:
        self._model_downloaded = True
        pct = downloaded / total if total else 0.0
        self._indicator.progress(pct, f"下载模型 {filename}")

    def _on_vad_progress(self, filename: str, downloaded: int, total: int) -> None:
        pct = downloaded / total if total else 0.0
        self._indicator.progress(pct, f"下载 VAD {filename}")

    def _on_toggle_autostart(self) -> None:
        native.util.set_autostart(not native.util.autostart_enabled())
        self._tray.refresh_menu()

    def _request_exit(self) -> None:
        """Stop the tray loop immediately; run() then tears the process down.

        Runs on the tray menu thread. Stops the icon directly (rather than
        routing through the worker, which may be blocked downloading) so Exit
        always interrupts whatever the app is doing.
        """
        self._stop_event.set()
        self._tray.stop()

    def _tray_state(self) -> TrayState:
        return TrayState(
            ready=self._ready,
            paused=self._paused,
            autostart=native.util.autostart_enabled(),
            current_mic=self._config.mic,
            current_hotkey=self._config.hotkey,
            current_model=self._config.model,
            disfluency=self._config.disfluency,
            continuous=self._continuous,
        )

    def _set_config(self, config: Config) -> None:
        self._config = config
        save_config(config_path(), config)

    def _rebuild_recorder(self, name: str) -> Recorder:
        if self._recorder is not None:
            self._recorder.close()
        self._recorder = Recorder(
            device_name=name, on_stream_error=lambda msg: log.warning("%s", msg)
        )
        if self._continuous and self._continuous_queue is not None:
            self._recorder.start_sink(self._continuous_queue)  # keep the mode alive
        return self._recorder

    def _show_diagnostics(self) -> None:
        self._controls.show_diagnostics()

    def _show_diagnostics_deferred(self) -> None:
        """Show diagnostics after the tray menu closes.

        A modal MessageBox shown inline from the menu handler races the menu's
        own popup loop (a Windows reentrancy), leaving the dialog unresponsive
        to clicks. A short delay on a worker thread sidesteps it.
        """
        threading.Thread(target=self._delayed_diagnostics, daemon=True).start()

    def _delayed_diagnostics(self) -> None:
        time.sleep(0.2)
        self._controls.show_diagnostics()

    # -- worker thread -------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            message = self._queue.get()
            try:
                self._dispatch(message)
            except Exception:
                log.exception("worker failed on %r", message)
                self._indicator.update("出错了，详见日志")
                self._tray.notify("语音处理出错，详见日志")

    def _on_key_transition(self, pressed: bool, ts_ms: int) -> None:
        event = Press(timestamp_ms=ts_ms) if pressed else Release(timestamp_ms=ts_ms)
        action = self._detector.feed(event)
        if action is not None:
            self._handle(action)

    def _dispatch(self, message: WorkerMsg) -> None:
        match message:
            case KeyTransition(pressed=pressed, ts_ms=ts):
                self._on_key_transition(pressed=pressed, ts_ms=ts)
            case SetPaused(paused=paused):
                self._set_paused(paused)
            case SetMic(name=name):
                self._controls.swap_mic(name)
            case SetHotkey(key=key):
                self._controls.swap_hotkey(key)
                self._skip_hold = False
            case CaptureHotkey():
                self._controls.start_key_capture(self._on_captured_vk)
            case SetModel(kind=kind):
                self._swap_model(kind)
            case SetDisfluency():
                self._toggle_disfluency()
            case SetContinuous():
                self._toggle_continuous()
            case InitModel():
                self._init_model()
            case unreachable:
                assert_never(unreachable)

    def _set_paused(self, paused: bool) -> None:
        """Toggle pause; in continuous mode the live pill follows it."""
        self._paused = paused
        self._tray.refresh_menu()  # pystray caches the menu; re-render the check
        if self._continuous:
            if paused:
                self._indicator.hide()
            else:
                self._indicator.listen(_LISTEN_TEXT)
        log.info("paused=%s", paused)

    def _handle(self, action: Action) -> None:
        match action:
            case StartHold():
                if self._paused or self._recorder is None or not self._model_ready:
                    self._skip_hold = True
                    return
                self._skip_hold = False
                self._holding = True
                self._recorder.start()  # capture from the first instant
                self._arm_hold_confirm()  # only a hold past the threshold confirms
                log.info("press: buffering; hold not yet confirmed")
            case Click():
                self._cancel_hold_confirm()
                was_holding = self._holding
                self._holding = False
                if was_holding and self._recorder is not None:
                    self._recorder.stop()  # discard the click's buffer
                self._indicator.hide()
                self._skip_hold = False
                log.info("click: discarded buffer, passing the tap through")
                vk = hotkey_vk(self._config.hotkey)
                if vk in MOUSE_VKS:
                    native.io.tap_mouse_x(vk)
                else:
                    native.io.tap_key(vk)
            case EndHold(duration_ms=duration):
                self._cancel_hold_confirm()
                self._holding = False
                log.info("hold: ended after %d ms", duration)
                self._finish_hold()
            case unreachable:
                assert_never(unreachable)

    def _arm_hold_confirm(self) -> None:
        """Show the recording bar only once the press survives the threshold."""
        self._hold_confirm_timer = threading.Timer(
            self._config.hold_threshold_ms / 1000, self._on_hold_confirmed
        )
        self._hold_confirm_timer.daemon = True
        self._hold_confirm_timer.start()

    def _cancel_hold_confirm(self) -> None:
        if self._hold_confirm_timer is not None:
            self._hold_confirm_timer.cancel()
            self._hold_confirm_timer = None

    def _on_hold_confirmed(self) -> None:
        """Timer thread: a press held past the threshold is a real hold."""
        if self._holding:
            self._indicator.show("● 正在听…")

    def _finish_hold(self) -> None:
        if self._skip_hold:
            self._skip_hold = False
            return
        recognizer, recorder = self._recognizer, self._recorder
        if recognizer is None or recorder is None:
            return
        audio = recorder.stop()
        if audio is None:
            log.info("hold: capture missing or under 300 ms — nothing to decode")
            self._indicator.hide()
            return
        log.info(
            "hold: captured %d samples (%.1f s), rms=%.4f",
            audio.shape[0],
            audio.shape[0] / 16_000,
            float(np.sqrt(np.mean(audio**2))),
        )
        self._indicator.update("识别中…")
        text = recognizer.transcribe(audio)
        log.info("asr: %r", text)
        if text:
            emit_text(
                text,
                EmitSettings(
                    restore_clipboard=self._config.restore_clipboard,
                    paste_delay_ms=self._config.paste_delay_ms,
                ),
                self._indicator,
            )
        else:
            self._indicator.hide()

    def _swap_model(self, kind: str) -> None:
        # Disarm while the worker downloads (CapsLock stays native), re-arm after.
        if self._hook is not None:
            self._hook.set_armed(False)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
        recognizer = self._controls.swap_model(kind)
        self._set_hooks_armed()
        if recognizer is not None:
            self._recognizer = recognizer

    def _toggle_disfluency(self) -> None:
        # Rebuilding the recognizer (Fun-ASR-Nano) blocks the worker; disarm the
        # hooks first so CapsLock stays native instead of a dead key, then re-arm.
        if self._hook is not None:
            self._hook.set_armed(False)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
        recognizer = self._controls.toggle_disfluency()
        self._set_hooks_armed()
        if recognizer is not None:
            self._recognizer = recognizer

    def _init_model(self) -> None:
        """Download (if needed) and load the recognizer on the worker thread."""
        try:
            files = ensure_model(
                self._config.model,
                models_root(),
                self._on_model_progress,
                proxy=self._config.proxy,
            )
            self._recognizer = Recognizer(
                kind=self._config.model,
                model_dir=files.directory,
                num_threads=self._config.num_threads,
                language=self._config.language,
                disfluency=self._config.disfluency,
            )
        except DownloadError as exc:
            log.warning("model download failed: %s", exc)
            guide = manual_download_guide(self._config.model, models_root())
            native.util.show_info(f"模型自动下载失败：{exc.reason}\n\n{guide}")
            return
        self._model_ready = True
        self._set_hooks_armed()
        log.info("model ready: %s", self._config.model)
        if self._model_downloaded:
            self._indicator.flash("模型下载完成，可以开始使用了", 4000)
            self._tray.notify("模型下载完成，可以开始使用了", title="xxl-whisper")
        if self._config.continuous:  # honour the persisted mode from last run
            self._start_continuous()

    # -- continuous dictation -------------------------------------------------

    def _toggle_continuous(self) -> None:
        """Tray switch for always-on VAD-gated dictation (持续听写)."""
        if self._continuous:
            self._stop_continuous()
        else:
            self._start_continuous()
        self._tray.refresh_menu()  # pystray caches the menu; re-render the check

    def _start_continuous(self) -> None:
        """Lazily fetch the VAD model and start the always-on segment loop.

        Runs on the worker thread. A download failure is reported like a model
        download (manual guide) and re-raised so the worker's generic handler
        surfaces the "出错了" indicator.
        """
        try:
            vad_path = ensure_vad_model(
                models_root(), self._on_vad_progress, proxy=self._config.proxy
            )
        except DownloadError as exc:
            log.warning("VAD download failed: %s", exc)
            guide = manual_download_guide("vad", models_root())
            native.util.show_info(f"VAD 模型自动下载失败：{exc.reason}\n\n{guide}")
            raise
        segmenter = Segmenter(
            VadSettings(
                model_path=vad_path,
                threshold=self._config.vad_threshold,
                min_speech_s=self._config.vad_min_speech_ms / 1000,
                min_silence_s=self._config.vad_min_silence_ms / 1000,
                max_speech_s=self._config.vad_max_speech_ms / 1000,
                num_threads=self._config.num_threads,
            )
        )
        blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=_CONTINUOUS_QUEUE_MAX)
        self._continuous_queue = blocks
        self._decode_queue = queue.Queue(maxsize=_DECODE_QUEUE_MAX)
        self._partial_slot = LatestSlot()
        self._vad_done = threading.Event()
        if self._recorder is not None:
            self._recorder.start_sink(blocks)
        self._continuous = True
        self._set_hooks_armed()  # stays disarmed: continuous owns the microphone
        # Two threads: the VAD consumer stays real-time (never decodes), while
        # the decoder drains finished sentences and refreshes live partials.
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop,
            args=(segmenter, blocks),
            daemon=True,
            name="continuous-vad",
        )
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            args=(self._decode_queue, self._partial_slot, self._vad_done),
            daemon=True,
            name="continuous-decode",
        )
        self._continuous_thread.start()
        self._decode_thread.start()
        self._persist_continuous(enabled=True)
        self._indicator.flash_listen("持续听写已开启", _LISTEN_TEXT, 1500)
        log.info("continuous: on")

    def _stop_continuous(self) -> None:
        """Stop the segment loop, release the microphone, restore push-to-talk."""
        self._continuous = False
        if self._recorder is not None:
            self._recorder.stop_sink()
        # Join the VAD thread first: its final flush queues the last utterance
        # for the decoder, which then drains it before exiting.
        vad_thread = self._continuous_thread
        if vad_thread is not None:
            vad_thread.join(timeout=2)
        decode_thread = self._decode_thread
        if decode_thread is not None:
            decode_thread.join(timeout=2)
        self._continuous_thread = None
        self._decode_thread = None
        self._continuous_queue = None
        self._decode_queue = None
        self._partial_slot = None
        self._vad_done = None
        self._persist_continuous(enabled=False)
        self._set_hooks_armed()
        self._indicator.hide()  # leave live mode; the pulse loop is cancelled
        log.info("continuous: off")

    def _persist_continuous(self, *, enabled: bool) -> None:
        """Remember the mode so a restart resumes it (mirrors 语义顺滑)."""
        if self._config.continuous != enabled:
            self._set_config(replace(self._config, continuous=enabled))

    def _continuous_loop(
        self, segmenter: Segmenter, blocks: queue.Queue[np.ndarray]
    ) -> None:
        """Daemon thread: feed the VAD and publish partials; never decode.

        This is the responsiveness-critical path. It consumes audio blocks, runs
        the VAD, hands finished segments to the decode queue, and publishes the
        VAD's own in-progress segment (the exact audio that will become the
        finished sentence, onset included) into the single-slot holder so a
        stale snapshot is dropped — only the newest matters. It performs no
        decoding at all, so a slow model (Fun-ASR-Nano) can neither stall the
        VAD feed nor delay endpointing: the pause is detected on time and the
        sentence is emitted by the decode thread the moment it is decoded.
        """
        decode_queue = self._decode_queue
        slot = self._partial_slot
        vad_done = self._vad_done
        skipped_partials = 0
        while self._continuous and not self._stop_event.is_set():
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
                self._indicator.level(level_from_block(block))
                segmenter.accept(block)
                # Decode the VAD's own in-progress segment, never an ad-hoc
                # buffer: the VAD flips "speech detected" only after buffering
                # the onset, so accumulating raw blocks would drop the start of
                # the utterance and the partial would not match the insert.
                samples = segmenter.current_samples()
                if samples is not None and self._publish_partial(slot, samples):
                    skipped_partials += 1  # a stale snapshot was superseded
                segments = segmenter.drain()
                if segments:  # endpoint: the utterance is complete
                    self._finish_utterance(slot, decode_queue, segments)
            except Exception:
                log.exception("continuous: block processing failed; loop continues")
        self._flush_segments(segmenter, decode_queue)
        if vad_done is not None:
            vad_done.set()  # tells the decoder to drain the tail and exit
        log.info("continuous: vad loop exited (skipped %d stale partials)", skipped_partials)

    def _publish_partial(
        self, slot: LatestSlot[np.ndarray] | None, samples: np.ndarray
    ) -> bool:
        """Publish the newest in-progress samples; True when one was superseded."""
        if slot is None or self._paused:
            return False
        return slot.publish(samples)

    def _finish_utterance(
        self,
        slot: LatestSlot[np.ndarray] | None,
        decode_queue: queue.Queue[FinishedBatch] | None,
        segments: Sequence[np.ndarray],
    ) -> None:
        """Endpoint: drop the stale snapshot and queue the sentence for decoding."""
        if slot is not None:
            slot.clear()
        if decode_queue is not None and not self._paused:
            _offer_finished(
                decode_queue,
                FinishedBatch(endpoint_ts=time.monotonic(), segments=tuple(segments)),
            )

    def _flush_segments(
        self, segmenter: Segmenter, decode_queue: queue.Queue[FinishedBatch] | None
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

    def _decode_loop(
        self,
        decode_queue: queue.Queue[FinishedBatch],
        slot: LatestSlot[np.ndarray],
        vad_done: threading.Event,
    ) -> None:
        """Daemon thread: decode finished sentences first, partials only when idle.

        A finished sentence always wins: the loop drains the queue before it ever
        considers a partial, so an endpoint's text is never delayed behind
        partial work. Partials run only when the queue is empty and the adaptive
        interval elapsed; each decode's duration feeds that interval so a slow
        model backs the cadence off instead of saturating the CPU. The recognizer
        is re-read per decode (it may be swapped mid-stream).
        """
        last_decode_s: float | None = None
        last_partial_at = 0.0
        last_partial_text = ""
        partials = 0
        while True:
            try:
                batch = decode_queue.get(timeout=_DECODE_POLL_S)
            except queue.Empty:
                batch = None
            if batch is not None:
                started = time.monotonic()
                emitted = self._insert_segments(
                    batch.segments, endpoint_ts=batch.endpoint_ts
                )
                finished_at = time.monotonic()
                last_decode_s = finished_at - started
                last_partial_at = finished_at  # measure the next partial gap from here
                last_partial_text = ""  # endpoint boundary: next utterance is fresh
                if emitted:
                    self._indicator.flash_listen(
                        _COMMIT_TEXT, _LISTEN_TEXT, _COMMIT_FLASH_MS
                    )
                continue
            if vad_done.is_set():
                break
            snapshot = slot.take()
            if snapshot is None:
                continue
            now = time.monotonic()
            partial_due = not self._paused and (
                now - last_partial_at >= partial_interval(_PARTIAL_INTERVAL_S, last_decode_s)
            )
            # No finished batch is waiting (handled above), so the only choice
            # left is whether a partial is due; the pure decision keeps it clear.
            if next_decode(has_finished=False, partial_due=partial_due) is not DecodeKind.PARTIAL:
                continue
            last_partial_at = now
            started = time.monotonic()
            shown = self._show_partial(snapshot, last_partial_text)
            last_decode_s = time.monotonic() - started
            if shown is not None:
                last_partial_text = shown
                partials += 1
        log.info("continuous: decode loop exited (partials decoded=%d)", partials)

    def _show_partial(self, samples: np.ndarray, previous: str) -> str | None:
        """Decode the in-progress samples and refresh the pill; return the caption.

        Runs on the decode thread only, so a slow partial decode never blocks the
        VAD. *samples* is the VAD's own current segment — the audio that will
        become the finished sentence — so the caption is a faithful preview of
        the text that will be inserted. Returns the caption now shown, or
        ``None`` when there was nothing new to show (empty decode, or the same
        caption as the last frame — so the pill cannot flicker).
        """
        recognizer = self._recognizer
        if recognizer is None:
            return None
        text = recognizer.transcribe(samples)
        if not text:
            return None
        shown = new_text_or_none(previous, truncate_partial(text))
        if shown is None:
            return None
        self._indicator.update(f"● {shown}")
        return shown

    def _insert_segments(
        self,
        segments: Sequence[np.ndarray],
        *,
        endpoint_ts: float | None = None,
    ) -> int:
        """Decode and emit each finished segment; returns how many were emitted.

        Runs on the decode thread, which prioritises these finished sentences
        over live partials. Each segment is isolated: a single failed decode or
        emit is logged and skipped so the remaining sentences still land and the
        loop survives. ``endpoint_ts`` (when the VAD saw the pause) turns each
        successful emit into an endpoint→emit latency line in ``app.log``.
        """
        emitted = 0
        for samples in segments:
            if self._stop_event.is_set() or self._paused:
                return emitted
            try:
                recognizer = self._recognizer
                if recognizer is None:
                    continue
                log.info("continuous: segment %.2f s", samples.shape[0] / 16_000)
                started = time.monotonic()
                text = recognizer.transcribe(samples)
                log.info("continuous: decode %.3f s", time.monotonic() - started)
                if not text:
                    continue
                log.info("continuous asr: %r", text)
                channel = emit_text(
                    text,
                    EmitSettings(
                        restore_clipboard=self._config.restore_clipboard,
                        paste_delay_ms=self._config.paste_delay_ms,
                    ),
                    self._indicator,
                )
            except Exception:
                log.exception("continuous: segment decode/emit failed; continuing")
                continue
            if endpoint_ts is not None:
                log.info(
                    "continuous: endpoint→emit %.3f s via %s",
                    time.monotonic() - endpoint_ts,
                    channel,
                )
            else:
                log.info("continuous: inserted via %s", channel)
            emitted += 1
        return emitted

    def _set_hooks_armed(self) -> None:
        """Arm the hooks once the model is ready so dictation can suppress keys.

        While the model loads the hooks stay disarmed (CapsLock is native) so a
        click toggles caps even though the worker is busy decoding. Continuous
        dictation keeps them disarmed for as long as it owns the microphone.
        """
        armed = self._model_ready and not self._continuous
        if self._hook is not None:
            self._hook.set_armed(armed)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(armed)

    def _show_loading_prompt(self, show: bool) -> None:
        """Hook-thread callback: hint that dictation is not ready yet.

        Runs while the hooks are disarmed (model loading / switching) so a
        hold does not feel like a dead key. The indicator facade is thread-safe.
        """
        if show:
            self._indicator.show("模型加载中…请稍候")
        else:
            self._indicator.hide()

    def _on_captured_vk(self, vk: int) -> None:
        """Hook-thread callback: route the captured key through the worker."""
        if vk == _VK_ESCAPE:
            self._indicator.hide()
            return
        self._queue.put(SetHotkey(key=vk))

    def _teardown(self) -> None:
        log.info("shutting down")
        self._stop_event.set()
        self._cancel_hold_confirm()
        self._updates.stop()
        self._continuous = False
        if self._recorder is not None:
            self._recorder.stop_sink()
        if self._continuous_thread is not None:
            self._continuous_thread.join(timeout=2)
        if self._decode_thread is not None:
            self._decode_thread.join(timeout=2)
        if self._hook is not None:
            self._hook.set_armed(False)
            self._hook.stop()
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
            self._mouse_hook.stop()
        if self._recorder is not None:
            self._recorder.close()
        self._indicator.quit()
