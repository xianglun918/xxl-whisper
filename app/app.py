"""Orchestrator: hotkey → recorder → recognizer → paste, driven by the tray.

Thread map: main thread runs the tray loop; the hook thread pumps Win32
messages; PortAudio runs its own callback thread; the ASR worker owns the
detector + recorder state machine so no locks are needed around them.
"""

import dataclasses
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from app import __version__, winio, winutil
from app.asr import Recognizer
from app.config import HOTKEY_VK, Config, config_path, model_dir, save_config
from app.downloader import ensure_model
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
from app.recorder import Recorder
from app.tray import Tray, TrayCallbacks, TrayState
from app.update_flow import UpdateFlow

log = logging.getLogger(__name__)


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
class Shutdown:
    pass


type WorkerMsg = KeyTransition | SetPaused | SetMic | Shutdown


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
        self._paused = False
        self._ready = False
        self._skip_hold = False
        self._stop_event = threading.Event()
        self._tray = Tray(
            callbacks=TrayCallbacks(
                on_exit=lambda: self._queue.put(Shutdown()),
                on_toggle_pause=lambda: self._queue.put(SetPaused(not self._paused)),
                on_select_mic=lambda name: self._queue.put(SetMic(name=name)),
                on_toggle_autostart=self._on_toggle_autostart,
                on_check_update=lambda: threading.Thread(
                    target=self._updates.manual_check, daemon=True
                ).start(),
            ),
            state_provider=self._tray_state,
        )
        self._updates = UpdateFlow(tray=self._tray, current_version=__version__)

    def run(self) -> None:
        files = ensure_model(model_dir(), progress=self._on_model_progress)
        self._recognizer = Recognizer(
            model=files.model,
            tokens=files.tokens,
            num_threads=self._config.num_threads,
            language=self._config.language,
        )
        self._recorder = Recorder(
            device_name=self._config.mic,
            on_stream_error=lambda msg: log.warning("%s", msg),
        )
        self._hook = HotkeyHook(
            vk=HOTKEY_VK[self._config.hotkey],
            on_transition=self._on_transition,
        )
        self._hook.start_and_wait()
        worker = threading.Thread(target=self._worker, daemon=True, name="asr-worker")
        worker.start()
        self._ready = True
        self._updates.start_watcher(self._config.check_updates)
        log.info("ready: hotkey=%s mic=%r", self._config.hotkey, self._config.mic)
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
        pct = downloaded / total if total else 0.0
        self._indicator.progress(pct, f"下载模型 {filename}")

    def _on_toggle_autostart(self) -> None:
        winutil.set_autostart(not winutil.autostart_enabled())

    def _tray_state(self) -> TrayState:
        return TrayState(
            ready=self._ready,
            paused=self._paused,
            autostart=winutil.autostart_enabled(),
            current_mic=self._config.mic,
        )

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
                self._swap_mic(name)
            case Shutdown():
                self._stop_event.set()
                self._tray.stop()
            case unreachable:
                assert_never(unreachable)

    def _handle(self, action: Action) -> None:
        match action:
            case StartHold():
                if self._paused or self._recorder is None:
                    self._skip_hold = True
                    return
                self._skip_hold = False
                self._recorder.start()
                self._indicator.show("● 正在听…")
                log.info("hold: recording started")
            case Click():
                log.info("click: passing through native key")
                winio.tap_key(HOTKEY_VK[self._config.hotkey])
            case EndHold(duration_ms=duration):
                log.info("hold: ended after %d ms", duration)
                self._finish_hold()
            case unreachable:
                assert_never(unreachable)

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
            self._emit_text(text)  # each branch settles the indicator itself
        else:
            self._indicator.hide()

    def _emit_text(self, text: str) -> None:
        log.info("emit: target window %r", winio.foreground_window_title())
        winio.set_clipboard_text(text)  # always staged: manual Ctrl+V also works
        if winio.keyboard_injection_alive():
            try:
                winio.paste_text(
                    text,
                    restore_clipboard=self._config.restore_clipboard,
                    delay_ms=self._config.paste_delay_ms,
                )
            except (winio.PasteError, OSError):
                log.warning("emit: keys path failed, trying WM_PASTE")
            else:
                log.info("emit: delivered via injected Ctrl+V")
                self._indicator.hide()
                return
        if winio.post_wm_paste_to_focus():
            log.info("emit: posted WM_PASTE to focused control")
            self._indicator.flash("已粘贴（本机拦截键盘注入）", 1200)
        else:
            log.warning("emit: no delivery channel — text is on the clipboard")
            self._indicator.flash("已复制到剪贴板，请手动 Ctrl+V", 2500)

    def _swap_mic(self, name: str) -> None:
        if self._recorder is None:
            return
        self._recorder.close()
        self._recorder = Recorder(
            device_name=name, on_stream_error=lambda msg: log.warning("%s", msg)
        )
        self._config = dataclasses.replace(self._config, mic=name)
        save_config(config_path(), self._config)
        log.info("mic switched to %r", name)

    def _teardown(self) -> None:
        log.info("shutting down")
        self._stop_event.set()
        self._updates.stop()
        if self._hook is not None:
            self._hook.stop()
        if self._recorder is not None:
            self._recorder.close()
        self._indicator.quit()
