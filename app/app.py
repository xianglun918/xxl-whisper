"""Orchestrator: hotkey → recorder → recognizer → paste, driven by the tray.

Thread map: main thread runs the tray loop; the hook threads (keyboard and
mouse) pump Win32 messages; PortAudio runs its own callback thread; the ASR
worker owns the detector + recorder state machine so no locks are needed
around them. Tray-driven control actions live in app.controls.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from app import __version__, winio, winutil
from app.asr import Recognizer
from app.config import MOUSE_VKS, Config, config_path, hotkey_vk, models_root, save_config
from app.controls import Controls, ControlsDeps
from app.downloader import DownloadError, ensure_model, manual_download_guide
from app.emit import EmitSettings, emit_text
from app.hotkey import HotkeyHook
from app.hotkey_logic import (
    Action,
    Click,
    EndHold,
    HoldClickDetector,
    Press,
    Release,
    StartHold,
)
from app.indicator import Indicator
from app.mousehook import MouseHook
from app.recorder import Recorder
from app.tray import Tray, TrayCallbacks, TrayState
from app.update_flow import UpdateFlow

log = logging.getLogger(__name__)

_VK_DISABLED: int = 0
_VK_ESCAPE: int = 0x1B


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
    | InitModel
)


class DictationApp:
    """Owns every component; ``run()`` blocks until the user exits the tray."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[WorkerMsg] = queue.Queue()
        self._detector = HoldClickDetector(config.hold_threshold_ms)
        self._indicator = Indicator()
        self._recognizer: Recognizer | None = None
        self._recorder: Recorder | None = None
        self._hook: HotkeyHook | None = None
        self._mouse_hook: MouseHook | None = None
        self._paused = False
        self._ready = False
        self._skip_hold = False
        self._holding = False
        self._hold_confirm_timer: threading.Timer | None = None
        self._model_downloaded = False
        self._model_ready = False
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
        self._hook = HotkeyHook(
            vk=_VK_DISABLED if vk in MOUSE_VKS else vk,
            on_transition=self._on_transition,
        )
        self._hook.start_and_wait()
        self._mouse_hook = MouseHook(
            vk=vk if vk in MOUSE_VKS else _VK_DISABLED,
            on_transition=self._on_transition,
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

    def _on_toggle_autostart(self) -> None:
        winutil.set_autostart(not winutil.autostart_enabled())

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
            autostart=winutil.autostart_enabled(),
            current_mic=self._config.mic,
            current_hotkey=self._config.hotkey,
            current_model=self._config.model,
            disfluency=self._config.disfluency,
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

    def _dispatch(self, message: WorkerMsg) -> None:
        match message:
            case KeyTransition(pressed=pressed, ts_ms=ts):
                event = Press(timestamp_ms=ts) if pressed else Release(timestamp_ms=ts)
                action = self._detector.feed(event)
                if action is not None:
                    self._handle(action)
            case SetPaused(paused=paused):
                self._paused = paused
                log.info("paused=%s", paused)
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
            case InitModel():
                self._init_model()
            case unreachable:
                assert_never(unreachable)

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
                log.info("click: discarded buffer, toggling native key")
                vk = hotkey_vk(self._config.hotkey)
                if vk in MOUSE_VKS:
                    winio.tap_mouse_x(vk)
                else:
                    winio.tap_key(vk)
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
        recognizer = self._controls.toggle_disfluency()
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
            winutil.show_info(f"模型自动下载失败：{exc.reason}\n\n{guide}")
            return
        self._model_ready = True
        self._set_hooks_armed()
        log.info("model ready: %s", self._config.model)
        if self._model_downloaded:
            self._indicator.flash("模型下载完成，可以开始使用了", 4000)
            self._tray.notify("模型下载完成，可以开始使用了", title="xxl-whisper")

    def _set_hooks_armed(self) -> None:
        """Arm the hooks once the model is ready so dictation can suppress keys.

        While the model loads the hooks stay disarmed (CapsLock is native) so a
        click toggles caps even though the worker is busy decoding.
        """
        armed = self._model_ready
        if self._hook is not None:
            self._hook.set_armed(armed)
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(armed)

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
        if self._hook is not None:
            self._hook.set_armed(False)
            self._hook.stop()
        if self._mouse_hook is not None:
            self._mouse_hook.set_armed(False)
            self._mouse_hook.stop()
        if self._recorder is not None:
            self._recorder.close()
        self._indicator.quit()
