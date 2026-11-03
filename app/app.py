"""Orchestrator: hotkey → recorder → recognizer → paste, driven by the tray.

Thread map: main thread runs the tray loop; the hook threads (keyboard and
mouse) pump Win32 messages; PortAudio runs its own callback thread; the ASR
worker owns the detector + recorder state machine so no locks are needed
around them. Tray-driven control actions live in app.controls.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, assert_never

import numpy as np

from app import __version__, logsafe, native
from app.asr import Recognizer
from app.config import (
    MOUSE_VKS,
    Config,
    config_path,
    hotkey_vk,
    models_root,
    parse_translate_route,
    route_of,
    save_config,
    segmenter_limits,
    take_pending_migration,
    with_overlay_mode,
)
from app.continuous import (
    COMMIT_FLASH_MS,
    COMMIT_TEXT,
    LISTEN_TEXT,
    ContinuousSession,
    SessionLimits,
    SessionPorts,
)
from app.controls import Controls, ControlsDeps
from app.downloader import (
    DownloadError,
    ensure_model,
    ensure_vad_model,
    manual_download_guide,
)
from app.emit import Channel, EmitSettings, emit_text
from app.emit_queue import EmitQueue, PendingEmit
from app.hotkey_logic import (
    Action,
    Click,
    EndHold,
    HoldClickDetector,
    Press,
    Release,
    StartHold,
    loading_hint_visible,
    push_to_talk_active,
)
from app.menu_spec import TrayCallbacks, TrayState
from app.partial import SubtitleGate
from app.recorder import Recorder
from app.translator import (
    TranslationError,
    Translator,
    compose_text,
    has_translatable_text,
    route_text,
)
from app.tray import Tray
from app.type_batcher import TypeBatcher
from app.update_flow import UpdateFlow, UpdateFlowDeps

log = logging.getLogger(__name__)

_VK_DISABLED: int = 0

#: Bounded finished-sentence queue between ASR and translation. Deep enough to
#: absorb a burst, shallow enough that stale subtitles are dropped, not queued.
_EMIT_QUEUE_MAX: int = 64


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
    kind: str


@dataclass(frozen=True, slots=True)
class SetContinuous:
    pass


@dataclass(frozen=True, slots=True)
class SetTranslate:
    pass


@dataclass(frozen=True, slots=True)
class SetTranslateRoute:
    route: str


@dataclass(frozen=True, slots=True)
class SetSystemAudio:
    device_id: str


@dataclass(frozen=True, slots=True)
class SetOverlayOutput:
    mode: str


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
    | SetTranslate
    | SetTranslateRoute
    | SetSystemAudio
    | SetOverlayOutput
    | InitModel
)


class DictationApp:
    """Owns every component; ``run()`` blocks until the user exits the tray."""

    @property
    def _continuous(self) -> bool:
        """True while a live transcript session owns the microphone.

        The session is the single source of truth: there is no second boolean to
        fall out of sync with the threads it describes.
        """
        session = self._session
        return session is not None and session.running

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[WorkerMsg] = queue.Queue()
        self._detector = HoldClickDetector(config.hold_threshold_ms)
        self._indicator = native.indicator.Indicator()
        #: Created lazily: a user who never turns the subtitles on never pays for
        #: a second Tk interpreter.
        self._overlay: Any | None = None
        self._subtitle_gate = SubtitleGate(
            min_interval_s=config.partial_subtitle_ms / 1000
        )
        self._recognizer: Recognizer | None = None
        self._translator: Translator | None = None
        self._recorder: Recorder | None = None
        self._hook: Any | None = None
        self._mouse_hook: Any | None = None
        self._paused = False
        self._ready = False
        self._skip_hold = False
        self._holding = False
        self._hold_confirm_timer: threading.Timer | None = None
        self._model_downloaded = False
        self._model_ready = False

        #: Finished sentences waiting for translation/typing. Keeps the decode
        #: thread free: ASR must never queue behind a slow translation.
        self._emit_queue = EmitQueue(maxsize=_EMIT_QUEUE_MAX)
        self._emit_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._started_at: float = time.monotonic()
        #: Glues force-cut ASR pieces back into whole sentences for typing.
        self._batcher = TypeBatcher()
        #: The live transcript, when 持续听写 is on. Threads, queues and their
        #: join order live inside it; the app only ever sees start/stop/running.
        self._session: ContinuousSession | None = None
        self._stop_event = threading.Event()
        self._tray = Tray(
            callbacks=TrayCallbacks(
                on_exit=self._request_exit,
                on_toggle_pause=lambda: self._tray_click(
                    "pause", lambda: self._queue.put(SetPaused(not self._paused))
                ),
                on_select_mic=lambda name: self._queue.put(SetMic(name=name)),
                on_toggle_autostart=self._on_toggle_autostart,
                on_check_update=lambda: threading.Thread(
                    target=self._updates.manual_check, daemon=True
                ).start(),
                on_select_hotkey=lambda key: self._queue.put(SetHotkey(key=key)),
                on_capture_hotkey=lambda: self._queue.put(CaptureHotkey()),
                on_select_model=lambda kind: self._queue.put(SetModel(kind=kind)),
                on_show_diagnostics=self._show_diagnostics_deferred,
                on_select_disfluency=lambda kind: self._tray_click(
                    f"disfluency:{kind}", lambda: self._queue.put(SetDisfluency(kind=kind))
                ),
                on_toggle_continuous=lambda: self._tray_click(
                    "continuous", lambda: self._queue.put(SetContinuous())
                ),
                on_toggle_translate=lambda: self._tray_click(
                    "translate", lambda: self._queue.put(SetTranslate())
                ),
                on_select_translate_route=self._on_tray_route,
                on_select_overlay_output=lambda mode: self._tray_click(
                    f"overlay-output:{mode}", lambda: self._queue.put(SetOverlayOutput(mode=mode))
                ),
                on_select_system_audio=lambda device_id: self._queue.put(
                    SetSystemAudio(device_id=device_id)
                ),
            ),
            state_provider=self._tray_state,
        )
        self._updates = UpdateFlow(
            UpdateFlowDeps(
                tray=self._tray,
                indicator=self._indicator,
                current_version=__version__,
                on_quit=self._request_exit,
                get_proxy=lambda: self._config.proxy,
            )
        )
        self._controls = Controls(
            ControlsDeps(
                indicator=self._indicator,
                tray=self._tray,
                get_config=lambda: self._config,
                set_config=self._set_config,
                get_hooks=lambda: (self._hook, self._mouse_hook),
                get_recorder=lambda: self._recorder,
                rebuild_recorder=self._rebuild_recorder,
                release_recognizer=self._release_recognizer,
                models_root=models_root(),
                num_threads=lambda: self._config.num_threads,
                language=lambda: self._config.language,
            )
        )

    def run(self) -> None:
        self._recorder = Recorder(
            device_name=self._config.mic,
            on_stream_error=lambda msg: log.warning("%s", msg),
            source=self._config.audio_source,
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
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="asr-worker"
        )
        self._worker_thread.start()
        self._emit_thread = threading.Thread(
            target=self._emit_loop, daemon=True, name="emit-worker"
        )
        self._emit_thread.start()
        # Model download + load run on the worker so the tray is responsive
        # immediately and Exit can abort an in-flight download.
        self._queue.put(InitModel())
        if self._config.overlay_enabled:
            self._ensure_overlay()  # honour the switch persisted from last run
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
        """Prewarm the capture on the hook thread, then let the worker decide.

        The worker may still be decoding the previous sentence (that decode used
        to run 0.2-3.5 s on the same thread), and every millisecond spent
        queueing here is speech that never reaches the model — the classic
        "第二句开头被吞掉". Only the stream start moves off the worker: the
        press/click/hold decision stays serialised on the worker.
        """
        self._prewarm_capture(pressed)
        self._queue.put(
            KeyTransition(pressed=pressed, ts_ms=time.perf_counter_ns() // 1_000_000)
        )

    def _prewarm_capture(self, pressed: bool) -> None:
        """Start recording the instant the hotkey goes down (hook thread)."""
        if not pressed or self._paused or self._continuous:
            return
        if self._recorder is None or not self._model_ready:
            return
        try:
            self._recorder.start()
        except Exception:
            log.exception("press: capture prewarm failed")

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
            translate=self._config.translate,
            translate_route=route_of(self._config) or "",
            translator_ready=self._translator is not None,
            audio_source=self._config.audio_source,
            current_system_audio=self._config.mic if self._config.audio_source == "system" else "",
            overlay=self._config.overlay_enabled,
            overlay_output=self._config.overlay_output,
        )

    def _set_config(self, config: Config) -> None:
        self._config = config
        save_config(config_path(), config)

    def _rebuild_recorder(self, name: str) -> Recorder:
        if self._recorder is not None:
            self._recorder.close()
        self._recorder = Recorder(
            device_name=name,
            on_stream_error=lambda msg: log.warning("%s", msg),
            source=self._config.audio_source,
        )
        session = self._session
        if session is not None:
            session.reconnect()  # keep the live transcript on the new device
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
        if not push_to_talk_active(continuous=self._continuous):
            # Continuous dictation owns the mic: the key is swallowed so it cannot
            # toggle CapsLock, but it must not start a recording either.
            return
        event = Press(timestamp_ms=ts_ms) if pressed else Release(timestamp_ms=ts_ms)
        action = self._detector.feed(event)
        if action is not None:
            self._handle(action)

    def _dispatch(self, message: WorkerMsg) -> None:  # noqa: C901, PLR0912 - flat message table
        if not isinstance(message, KeyTransition):
            log.info("worker: %s", type(message).__name__)
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
            case SetDisfluency(kind=kind):
                self._set_disfluency(kind)
            case SetContinuous():
                self._toggle_continuous()
            case SetTranslate():
                self._toggle_translate()
            case SetTranslateRoute(route=route):
                self._set_translate_route(route)
            case SetSystemAudio(device_id=device_id):
                self._controls.swap_system_audio(device_id)
            case SetOverlayOutput(mode=mode):
                self._set_overlay_output(mode)
            case InitModel():
                self._init_model()
            case unreachable:
                assert_never(unreachable)

    def _set_paused(self, paused: bool) -> None:
        """Toggle pause; in continuous mode the live pill follows it."""
        if paused:
            self._flush_typing()  # don't strand a half-typed sentence while paused
        self._paused = paused
        self._tray.refresh_menu()  # pystray caches the menu; re-render the check
        if self._continuous:
            if paused:
                self._indicator.hide()
            else:
                self._indicator.listen(LISTEN_TEXT)
        log.info("paused=%s", paused)

    def _handle(self, action: Action) -> None:
        match action:
            case StartHold():
                if self._paused or self._recorder is None or not self._model_ready:
                    self._skip_hold = True
                    # The hook thread prewarmed the capture on key-down; nobody
                    # else will stop it, and the buffer would keep growing (and
                    # the next hold would ship stale audio to the model).
                    if self._recorder is not None:
                        self._recorder.stop()
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
        if self._holding and not self._subtitles():
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
        if not self._subtitles():
            self._indicator.update("识别中…")
        text = recognizer.transcribe(audio)
        log.info("asr: %s", logsafe.describe_text(text))
        if text:
            self._emit(text)
        else:
            self._indicator.hide()

    def _release_recognizer(self) -> None:
        """Drop the recognizer so a new model never overlaps it in memory.

        Called by :class:`~app.controls.Controls` immediately before it builds
        the replacement, so the swap's peak is one model instead of two
        (~1.25 GB -> one). The continuous decode threads re-read
        ``self._recognizer`` per decode and skip while it is ``None``, so they
        simply idle across the short rebuild window; a decode already holding
        the old reference finishes on it and lets it go.
        """
        self._recognizer = None

    def _swap_model(self, kind: str) -> None:
        # Disarm while the worker downloads (CapsLock stays native), re-arm after.
        # The old recognizer is released by Controls just before the new build.
        if self._hook is not None:
            self._hook.set_armed(False)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
        recognizer: Recognizer | None = None
        try:
            recognizer = self._controls.swap_model(kind)
        except Exception:
            # Re-arming the hooks used to be skipped when the swap raised, which
            # left them disarmed for the rest of the session: push-to-talk stayed
            # dead and the only trace was one log line. Rebuild the current model
            # (the config was already switched) so dictation keeps working.
            log.exception("model switch failed for %r", kind)
            self._indicator.update("切换模型失败，正在恢复")
            recognizer = self._rebuild_after_failed_swap()
        finally:
            self._set_hooks_armed()
        if recognizer is not None:
            self._recognizer = recognizer

    def _rebuild_after_failed_swap(self) -> Recognizer | None:
        """Try to bring a recognizer back after a failed model switch."""
        rebuilt = self._controls.rebuild_recognizer()
        if rebuilt is None:
            self._indicator.update("模型不可用，请查看日志")
        else:
            self._indicator.flash("已恢复原模型", 1500)
        return rebuilt

    def _set_disfluency(self, kind: str) -> None:
        # Rebuilding the recognizer (Fun-ASR-Nano) blocks the worker; disarm the
        # hooks first so CapsLock stays native instead of a dead key, then re-arm.
        if self._hook is not None:
            self._hook.set_armed(False)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
        recognizer: Recognizer | None = None
        try:
            recognizer = self._controls.set_disfluency(kind)
        except Exception:
            log.exception("disfluency switch failed for %r", kind)
            recognizer = self._rebuild_after_failed_swap()
        finally:
            self._set_hooks_armed()
        if recognizer is not None:
            self._recognizer = recognizer

    def _init_model(self) -> None:
        """Download (if needed) and load the recognizer on the worker thread."""
        started = time.monotonic()
        # Say what is happening: loading a 1 GB Fun-ASR-Nano takes seconds and
        # used to show nothing at all, which read as "the app is dead".
        if not self._model_downloaded:
            self._indicator.show("正在加载识别模型…（首次约 3–10 秒）")
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
        elapsed = time.monotonic() - started
        log.info(
            "model ready: %s (%.1f s, startup total %.1f s)",
            self._config.model,
            elapsed,
            time.monotonic() - self._started_at,
        )
        if self._config.translate:
            # Lazy: the NMT model is only fetched/loaded when the user has
            # translation on, and a failure here must never block dictation.
            self._translator = self._controls.build_translator()
        if self._model_downloaded:
            self._indicator.flash("模型下载完成，可以开始使用了", 4000)
            self._tray.notify("模型下载完成，可以开始使用了", title="xxl-whisper")
        elif self._subtitles():
            self._indicator.hide()  # the subtitle bar owns the screen from here
        else:
            self._indicator.flash("就绪：可以开始说话了", 2000)
        # An upgraded config file is news the user needs once: which version it
        # came from, which keys were renamed, and where the old copy lives.
        migration = take_pending_migration()
        if migration is not None:
            self._indicator.flash(migration.describe(), 6000)
            self._tray.notify(migration.describe(), title="配置已升级")
        if self._config.continuous:  # honour the persisted mode from last run
            # Deferred: launching while Windows is still bringing up the audio
            # endpoint made the stream fail to open, leaving the mode "on" but
            # deaf until the user toggled it off and on again.
            threading.Timer(1.0, self._resume_continuous).start()

    # -- translation ------------------------------------------------------------

    def _toggle_translate(self) -> None:
        """Tray switch for local NMT: load the model on enable, drop it on disable."""
        enabled = self._controls.toggle_translate()
        self._translator = self._controls.build_translator() if enabled else None
        self._tray.refresh_menu()
        self._indicator.flash(
            "离线翻译：已开启" if enabled else "离线翻译：已关闭", 2000
        )
        log.info(
            "translate enabled=%s translator=%s",
            enabled,
            type(self._translator).__name__ if self._translator is not None else "none",
        )

    def _tray_click(self, name: str, action: Callable[[], None]) -> None:
        """Log a tray click before dispatching it.

        Named ``_tray_click`` and not ``_tray``: ``self._tray`` holds the Tray
        instance, and an instance attribute shadows a method of the same name —
        calling it then fails with "'Tray' object is not callable".

        The log is the only way to tell "the click never reached the app" (menu
        wiring) from "the click was applied but had no visible effect" (state).
        """
        log.info("tray: %s", name)
        self._ensure_worker_alive()
        action()

    def _ensure_worker_alive(self) -> None:
        """Restart the worker thread if it died.

        Every menu switch is dispatched through the worker queue, so a dead
        worker makes the whole menu look broken: clicks are accepted, logged,
        and never acted on. The loop catches per-message exceptions, but a failure
        at start-up (model init, native call) can still take the thread down.
        """
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            return
        log.error("worker thread is not running; restarting it")
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="asr-worker"
        )
        self._worker_thread.start()
        self._indicator.flash("后台线程已重启（上次异常退出，详见日志）", 4000)

    def _on_tray_route(self, route: str) -> None:
        """Tray thread: log the click, then hand it to the worker."""
        log.info("tray: route action fired %r", route)
        self._queue.put(SetTranslateRoute(route=route))

    def _set_translate_route(self, route: str) -> None:
        """Switch 中→英 / 英→中 / 日→中.

        Two-phase on purpose, because each direction is a different ~245 MB
        model:

        1. commit the choice and refresh the menu **immediately** — the tick mark
           must move on click, not minutes later when a download finishes;
        2. load/download the model on a background thread. Subtitle output keeps
           working meanwhile (untranslated), and if the model cannot be fetched
           the route is rolled back with a notification instead of silently
           disabling translation.
        """
        log.info("tray: translate route clicked %r (current=%s)", route, route_of(self._config))
        wanted = parse_translate_route(route)
        if wanted is None:
            self._indicator.flash(f"未知的翻译方向：{route}", 2500)
            log.warning("tray: unknown route %r", route)
            return
        previous = route_of(self._config)
        if wanted == previous:
            self._indicator.flash(f"翻译方向已是 {route_text(wanted)}", 1800)
            return
        self._flush_typing()  # let the in-flight sentence use the old translator
        previous_language = self._config.language
        if not self._controls.set_translate_route(wanted):
            return
        # A 日→… route pins the recogniser language. The *running* recognizer
        # keeps its old language until it is rebuilt, and SenseVoice decoding
        # Japanese as Chinese produces gibberish that the translator then
        # faithfully translates ("translation is completely wrong").
        if self._config.language != previous_language:
            rebuilt = self._controls.rebuild_recognizer()
            if rebuilt is not None:
                self._recognizer = rebuilt
                log.info(
                    "recognizer rebuilt for language=%s (route %s)",
                    self._config.language,
                    wanted,
                )
            else:
                log.warning("recognizer rebuild failed; language stays %s", previous_language)
        self._tray.refresh_menu()  # tick moves now, not after the download
        if not self._config.translate:
            self._indicator.flash(
                f"翻译方向记为 {route_text(wanted)}（翻译未开，开启后生效）", 3000
            )
            return
        self._translator = None  # no stale direction may translate meanwhile
        self._indicator.flash(f"正在加载 {route_text(wanted)} 模型…", 3000)
        threading.Thread(
            target=self._load_route_model,
            args=(wanted, previous),
            daemon=True,
            name="route-load",
        ).start()

    def _load_route_model(self, wanted: str, previous: str | None) -> None:
        """Background: fetch/load the new direction; roll back when it fails."""
        log.info("route-load: loading %s", wanted)
        translator: Translator | None = None
        try:
            translator = self._controls.build_translator(
                route=wanted, disable_on_failure=False
            )
        except Exception:
            log.exception("route-load: failed for %s", wanted)
        if translator is not None:
            self._translator = translator
            self._indicator.flash(f"翻译方向：{route_text(wanted)}", 2000)
            log.info("route-load: %s ready", wanted)
            return
        # Roll back so the tray never shows a direction the app cannot run.
        reason = f"{route_text(wanted)} 模型不可用，已回退"
        log.warning("route-load: %s unavailable; rolling back to %s", wanted, previous)
        if previous is not None:
            self._controls.set_translate_route(previous)
        self._tray.refresh_menu()
        self._tray.notify(f"{route_text(wanted)} 模型下载/加载失败，翻译方向已回退")
        self._indicator.flash(reason, 4000)

    # -- subtitle overlay ------------------------------------------------------

    def _overlay_style(self) -> Any:  # noqa: ANN401 - platform overlay style is dynamic
        """Map the flat config onto the overlay's window style."""
        return native.overlay.build_style(self._config)

    def _ensure_overlay(self) -> Any | None:  # noqa: ANN401 - see above
        """Start the subtitle window on first use; None when it cannot exist.

        A headless/no-Tk environment must not break dictation: the switch is
        turned back off in the config so the tray and the next start agree.
        """
        if not self._config.overlay_enabled:
            return None
        if self._overlay is None:
            overlay = native.overlay.Overlay(
                self._overlay_style(),
                work_area=native.io.active_monitor_work_area,
                fullscreen_owner=native.io.exclusive_fullscreen_owner_active,
            )
            if not overlay.available:
                self._config = replace(self._config, overlay_enabled=False)
                save_config(config_path(), self._config)
                self._tray.notify("上屏字幕不可用（缺少图形环境），已关闭")
                self._tray.refresh_menu()
                return None
            self._overlay = overlay
        return self._overlay

    def _set_overlay_output(self, mode: str) -> None:
        """Choose where text goes: 只上屏 / 上屏+打字 / 只打字.

        This single choice also decides whether the bar exists: "只打字" means no
        subtitle window. An earlier design had a separate "开启上屏字幕" switch
        next to these three, which let the menu contradict itself — you could be
        "上屏开启" and still pick "只打字（无字幕）".
        """
        if mode not in ("type", "screen", "both") or mode == self._config.overlay_output:
            return
        self._flush_typing()  # the sentence in flight belongs to the old mode
        wants_bar = mode != "type"
        self._config = with_overlay_mode(self._config, mode)
        save_config(config_path(), self._config)
        if wants_bar:
            self._ensure_overlay()
            if not self._config.translate:
                # Subtitles without translation are legitimate (pure dictation),
                # but a user switching to subtitles for a foreign film expects
                # translated lines — say it out loud instead of showing source.
                self._indicator.flash("已开字幕，但离线翻译未开启：目前显示原文", 4000)
                self._tray.notify("字幕已开，但离线翻译未开启，将显示原文", title="xxl-whisper")
            elif self._translator is None:
                self._translator = self._controls.build_translator()
        elif self._overlay is not None:
            self._overlay.clear()
        self._tray.refresh_menu()
        log.info(
            "overlay mode=%s bar=%s translate=%s translator=%s route=%s",
            mode,
            wants_bar,
            self._config.translate,
            type(self._translator).__name__ if self._translator is not None else "none",
            route_of(self._config),
        )

    def _present(self, text: str, translated: str) -> None:
        """Push one finished utterance to the subtitle bar (no-op when off)."""
        overlay = self._overlay
        if overlay is None or not self._config.overlay_enabled:
            return
        self._subtitle_gate.reset()  # the finished sentence replaces the preview
        overlay.push(text, translated)

    def _overlay_only(self) -> bool:
        """Return True when the user wants the subtitles instead of typing."""
        return self._config.overlay_enabled and self._config.overlay_output == "screen"

    def _subtitles(self) -> bool:
        """Return True when the subtitle bar is one of the outputs."""
        return self._config.overlay_enabled and self._config.overlay_output != "type"

    def _types(self) -> bool:
        """Return True when finished sentences are pasted into the cursor."""
        return not self._overlay_only()

    def _emit(self, text: str) -> Channel | None:
        """Deliver one utterance: translate once, then type and/or subtitle it.

        Returns the channel that typed the text, or ``None`` in subtitle-only
        mode (nothing was injected, by the user's explicit choice).
        """
        translated, attempted = self._translate(text)
        self._present(text, translated)
        if self._overlay_only():
            self._indicator.flash(COMMIT_TEXT, COMMIT_FLASH_MS)
            return None
        return emit_text(
            self._payload(text, translated, attempted=attempted),
            EmitSettings(
                restore_clipboard=self._config.restore_clipboard,
                paste_delay_ms=self._config.paste_delay_ms,
            ),
            self._indicator,
        )

    def _translate(self, text: str) -> tuple[str, bool]:
        """Run local MT once per utterance.

        Returns ``(translation, attempted)``: ``attempted`` is False when there
        was nothing to do (MT off, no model loaded, nothing translatable), which
        is what separates "silently skipped" from "tried and failed" for the
        caller's diagnostic flash.
        """
        translator = self._translator
        if translator is None or not self._config.translate or not has_translatable_text(text):
            return "", False
        if not self._subtitles():
            # The bar already shows the in-progress line; a sticky "翻译中…" pill
            # on top of it is just noise, and it never clears on its own.
            self._indicator.update("翻译中…")
        started = time.monotonic()
        try:
            translated = translator.translate(text)
        except TranslationError as exc:
            log.warning("translate failed: %s", exc.reason)
            translated = ""
        except Exception:
            log.exception("translate failed unexpectedly")
            translated = ""
        elapsed = time.monotonic() - started
        log.info(
            "translate: %.3f s (%s)",
            elapsed,
            logsafe.describe_pair(text, translated),
        )
        return translated, True

    def _payload(self, text: str, translated: str, *, attempted: bool) -> str:
        """Return what should be typed: source, translation, or both.

        Translation is best-effort by design: any model error falls back to the
        original text with a short pill message. Losing the translation is an
        annoyance; losing the user's words is not acceptable.
        """
        if translated.strip():
            return compose_text(text, translated, self._config.translate_mode)
        if attempted and not self._subtitles():
            self._indicator.flash("翻译失败，已上屏原文", 1800)
        return text

    # -- continuous dictation -------------------------------------------------

    def _toggle_continuous(self) -> None:
        """Tray switch for always-on VAD-gated dictation (持续听写)."""
        if self._continuous:
            self._stop_continuous()
        else:
            self._start_continuous()
            # The session owns its own audio watchdog: it reports "confirmed" or
            # "never started" through the ports, so nothing is left to verify here.
        self._tray.refresh_menu()  # pystray caches the menu; re-render the check

    def _start_continuous(self) -> None:
        """Fetch the VAD model, then hand the session its limits and ports.

        Runs on the worker thread. A download failure is reported like a model
        download (manual guide) and re-raised so the worker's generic handler
        surfaces the "出错了" indicator.
        """
        # One cadence for every continuous session: short segments are fine
        # because typing re-joins them (app.type_batcher) while subtitles want
        # them short. Push-to-talk never reaches this code.
        # Subtitle mode force-cuts every few seconds; typing-only keeps long,
        # complete utterances (a cut mid-word corrupts the typed sentence).
        min_silence_ms, max_speech_ms = segmenter_limits(self._config)
        limits = SessionLimits(
            min_silence_ms=min_silence_ms,
            max_speech_ms=max_speech_ms,
            threshold=self._config.vad_threshold,
            min_speech_ms=self._config.vad_min_speech_ms,
            num_threads=self._config.num_threads,
        )
        self._batcher = TypeBatcher(
            window_s=self._config.typing_merge_ms / 1000,
            max_chars=self._config.typing_merge_chars,
        )
        log.info(
            "continuous: mode=%s min_silence=%dms max_speech=%dms typing_merge=%dms/%dchars",
            "subtitles" if self._subtitles() else "typing",
            min_silence_ms,
            max_speech_ms,
            self._config.typing_merge_ms,
            self._config.typing_merge_chars,
        )
        self._cancel_hold_confirm()
        if self._holding and self._recorder is not None:
            self._recorder.stop()  # discard a hold that was in progress
            self._holding = False
        try:
            vad_path = ensure_vad_model(
                models_root(), self._on_vad_progress, proxy=self._config.proxy
            )
        except DownloadError as exc:
            log.warning("VAD download failed: %s", exc)
            guide = manual_download_guide("vad", models_root())
            native.util.show_info(f"VAD 模型自动下载失败：{exc.reason}\n\n{guide}")
            raise
        session = ContinuousSession(
            vad_path=vad_path, limits=limits, ports=self._session_ports()
        )
        session.start()
        self._session = session
        self._persist_continuous(enabled=True)
        self._set_hooks_armed()  # disarmed: the key stays native, presses ignored
        if self._subtitles():
            self._indicator.hide()  # the subtitle bar carries the state instead
        else:
            self._indicator.flash_listen("持续听写已开启", LISTEN_TEXT, 1500)
        log.info("continuous: on")

    def _stop_continuous(self) -> None:
        """Stop the session, then hand the microphone back to push-to-talk."""
        self._flush_typing()
        session = self._session
        self._session = None
        if session is not None:
            session.stop()
        self._persist_continuous(enabled=False)
        self._set_hooks_armed()
        self._indicator.hide()  # leave live mode; the pulse loop is cancelled
        log.info("continuous: off")

    def _resume_continuous(self) -> None:
        """Auto-start continuous dictation from the persisted mode."""
        if self._continuous or self._stop_event.is_set():
            return
        log.info("startup: resuming continuous dictation")
        try:
            self._start_continuous()
        except Exception:
            log.exception("startup: could not resume continuous dictation")
            self._persist_continuous(enabled=False)
            self._tray.refresh_menu()
            self._indicator.flash("持续听写启动失败（详见日志）", 4000)

    def _session_ports(self) -> SessionPorts:
        """Wire the session to the app without handing it the app itself."""
        return SessionPorts(
            stop_event=self._stop_event,
            recognizer=lambda: self._recognizer,
            indicator=self._indicator,
            emit_queue=self._emit_queue,
            subtitles=self._subtitles,
            paused=lambda: self._paused,
            on_partial_text=self._offer_partial_subtitle,
            on_audio_confirmed=self._tray.refresh_menu,
            on_audio_never_started=self._on_audio_never_started,
            start_sink=self._start_capture_sink,
            stop_sink=self._stop_capture_sink,
            blocks_delivered=self._blocks_delivered,
        )

    def _start_capture_sink(self, blocks: queue.Queue[np.ndarray]) -> None:
        if self._recorder is not None:
            self._recorder.start_sink(blocks)

    def _stop_capture_sink(self) -> None:
        if self._recorder is not None:
            self._recorder.stop_sink()

    def _blocks_delivered(self) -> int:
        recorder = self._recorder
        return 0 if recorder is None else recorder.blocks_delivered

    def _on_audio_never_started(self) -> None:
        """Close the switch so it never lies about working."""
        self._stop_continuous()
        self._tray.refresh_menu()
        self._tray.notify("持续听写没能启动（麦克风/系统声音未就绪），已自动关闭")
        self._indicator.flash("音频没起来，已关闭持续听写；可再点一次开启", 5000)

    def _persist_continuous(self, *, enabled: bool) -> None:
        """Remember the mode so a restart resumes it (mirrors 语义顺滑)."""
        if self._config.continuous != enabled:
            self._set_config(replace(self._config, continuous=enabled))

    def _offer_partial_subtitle(self, text: str) -> None:
        """Translate the still-being-spoken sentence and show it as a preview.

        Throttled by :class:`~app.partial.SubtitleGate` (change + growth + one
        refresh per interval): a partial decode arrives several times a second
        and each translation competes with the recogniser for the same CPU.
        """
        if not self._config.partial_subtitle:
            return
        overlay = self._overlay
        if overlay is None or not self._subtitles():
            return
        offered = self._subtitle_gate.offer(text, time.monotonic())
        if offered is None:
            return
        translated = ""
        if self._config.translate:
            translated, _attempted = self._translate(offered)
        overlay.push_provisional(offered, translated)
        log.info(
            "partial subtitle: %d chars translated=%s", len(offered), bool(translated)
        )

    def _emit_loop(self) -> None:
        """Daemon thread: subtitles per piece, typing per whole sentence.

        Runs MT and both outputs here so the decode thread never waits on them.
        The two channels deliberately keep different cadences: the subtitle bar
        is consumed while the speaker is still talking, so each piece goes up
        immediately; the cursor is a document, so force-cut pieces are glued back
        together by :class:`~app.type_batcher.TypeBatcher` and pasted once.
        """
        while not self._stop_event.is_set():
            item = self._emit_queue.get(timeout=0.2)
            if item is None:
                if self._batcher.should_flush():
                    self._type_batch()
                continue
            if item.flush:
                self._type_batch()
                self._emit_queue.task_done()
                continue
            try:
                self._serve(item)
            except Exception:  # one bad sentence must not stop the loop
                log.exception("emit failed for %s", logsafe.describe_text(item.text))
            finally:
                self._emit_queue.task_done()
        self._type_batch()  # never leave a half-sentence behind on shutdown

    def _serve(self, item: PendingEmit) -> None:
        """Translate and deliver one finished ASR piece."""
        translated = ""
        attempted = False
        if self._subtitles():
            if self._translator is None:
                log.info(
                    "subtitles: no translator (translate=%s route=%s) -> source only",
                    self._config.translate,
                    route_of(self._config),
                )
            translated, attempted = self._translate(item.text)
            self._present(item.text, translated)
        if self._types():
            ready = self._batcher.add(item.text, forced=item.forced)
            if ready is not None:
                # Reuse the subtitle translation when the batcher emitted exactly
                # this sentence; a merged batch covers other text and must be
                # translated on its own.
                # Only reuse when a translation really ran: with subtitles off
                # ``translated`` is "" and reusing it would silently skip the
                # translation this sentence needs for typing.
                cached = (
                    (translated, attempted) if ready == item.text and attempted else None
                )
                self._type_batch(ready, cached=cached)
        if item.endpoint_ts is not None:
            log.info(
                "continuous: endpoint→shown %.3f s (queued %.2f s, forced=%s)",
                time.monotonic() - item.endpoint_ts,
                item.enqueued_at - item.endpoint_ts,
                item.forced,
            )

    def _type_batch(
        self,
        text: str | None = None,
        *,
        cached: tuple[str, bool] | None = None,
    ) -> None:
        """Translate (if needed) and paste one whole sentence into the cursor.

        *cached* carries a translation already computed for this very sentence
        (the subtitle path does one), so the emitter does not run the NMT model
        twice per sentence.
        """
        sentence = text if text is not None else self._batcher.flush()
        if not sentence:
            return
        translated, attempted = cached if cached is not None else self._translate(sentence)
        try:
            emit_text(
                self._payload(sentence, translated, attempted=attempted),
                EmitSettings(
                    restore_clipboard=self._config.restore_clipboard,
                    paste_delay_ms=self._config.paste_delay_ms,
                ),
                self._indicator,
            )
        except Exception:  # one failed paste must not lose the rest
            log.exception("typing failed for %s", logsafe.describe_text(sentence))

    def _flush_typing(self) -> None:
        """Ask the emitter to type anything it is holding (safe from any thread)."""
        self._emit_queue.submit_flush()

    def _set_hooks_armed(self) -> None:
        """Arm the hooks once the model is ready so dictation can suppress keys.

        While the model loads the hooks stay disarmed (CapsLock is native) so a
        click toggles caps even though the worker is busy decoding.

        Continuous dictation stays *disarmed*: the key then behaves natively, so
        CapsLock still toggles caps while a live transcript is running. What must
        not happen is "hold to talk", and that is dropped a layer above in
        ``_on_key_transition``; the misleading "model loading" pill is silenced
        in ``_show_loading_prompt``.
        """
        armed = self._model_ready and not self._continuous
        if self._hook is not None:
            self._hook.set_armed(armed)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(armed)

    def _show_loading_prompt(self, show: bool) -> None:
        """Hook-thread callback: hint that dictation is not ready yet.

        Runs while the hooks are disarmed (model loading / switching) so a
        hold does not feel like a dead key. During continuous dictation nothing
        is loading — the key is simply native — so the pill must stay quiet.
        The indicator facade is thread-safe.
        """
        if not loading_hint_visible(continuous=self._continuous):
            return  # continuous: the key is simply native, leave the pill alone
        if show:
            self._indicator.show("模型加载中…请稍候")
        else:
            self._indicator.hide()

    def _on_captured_vk(self, vk: int) -> None:
        """Hook-thread callback: route the captured key through the worker."""
        if vk == native.ESCAPE_VK:
            self._indicator.hide()
            return
        self._queue.put(SetHotkey(key=vk))

    def _teardown(self) -> None:
        log.info("shutting down")
        self._flush_typing()
        self._stop_event.set()
        self._cancel_hold_confirm()
        self._updates.stop()
        session = self._session
        self._session = None
        if session is not None:
            session.stop()  # detaches the sink and joins the workers, in order
        if self._hook is not None:
            self._hook.set_armed(False)
            self._hook.stop()
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
            self._mouse_hook.stop()
        if self._recorder is not None:
            self._recorder.close()
        if len(self._emit_queue):
            log.info(
                "shutdown: %d finished sentence(s) still queued, %d dropped in total",
                len(self._emit_queue),
                self._emit_queue.dropped,
            )
        if self._overlay is not None:
            self._overlay.quit()
        self._indicator.quit()
