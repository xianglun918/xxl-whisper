"""Tray-driven control actions: hotkey/mic/model switching and diagnostics.

These run on the worker thread (serialized through the app's message queue)
so the dictation pipeline never races a settings change. The module owns the
*what*; the app owns the *components* it mutates.
"""

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app import winio, winutil
from app.asr import Recognizer
from app.config import (
    MOUSE_VKS,
    Config,
    ModelKind,
    config_path,
    hotkey_vk,
    save_config,
)
from app.downloader import ensure_model
from app.emit import is_classic_control
from app.hotkey import HotkeyHook
from app.indicator import Indicator
from app.mousehook import MouseHook
from app.recorder import Recorder
from app.tray import Tray

log = logging.getLogger(__name__)

_VK_DISABLED: int = 0


@dataclass(frozen=True, slots=True)
class ControlsDeps:
    """Live dependencies the panel actions mutate through."""

    indicator: Indicator
    tray: Tray
    get_config: Callable[[], Config]
    set_config: Callable[[Config], None]
    get_hooks: Callable[[], tuple[HotkeyHook | None, MouseHook | None]]
    get_recorder: Callable[[], Recorder | None]
    rebuild_recorder: Callable[[str], Recorder]
    models_root: Path
    num_threads: Callable[[], int]
    language: Callable[[], str]


class Controls:
    """Executes tray-driven control actions; owns no global state of its own."""

    def __init__(self, deps: ControlsDeps) -> None:
        self._deps = deps

    # -- hotkey ---------------------------------------------------------------

    def swap_hotkey(self, key: str | int) -> None:
        """Retarget the live hooks to a new key and persist the choice."""
        config = self._deps.get_config()
        if key == config.hotkey:
            return
        try:
            vk = hotkey_vk(key)
        except (KeyError, TypeError):
            log.warning("hotkey %r is not a preset or VK integer; ignored", key)
            return
        kb_hook, mouse_hook = self._deps.get_hooks()
        if kb_hook is not None:
            kb_hook.retarget(_VK_DISABLED if vk in MOUSE_VKS else vk)
        if mouse_hook is not None:
            mouse_hook.retarget(vk if vk in MOUSE_VKS else _VK_DISABLED)
        self._deps.set_config(dataclasses.replace(config, hotkey=key))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        label = winio.key_name(vk) if isinstance(key, int) else key
        log.info("hotkey switched to %r (vk=0x%02X)", label, vk)
        self._deps.indicator.flash(f"热键已切换：{label}", 1200)

    def start_key_capture(self, on_captured: Callable[[int], None]) -> None:
        """Prompt the user to press an arbitrary key; ``on_captured`` gets its VK.

        ``on_captured`` runs on the hook thread — it must only enqueue.
        """
        kb_hook, _ = self._deps.get_hooks()
        if kb_hook is None:
            return
        self._deps.indicator.show("请按下新的热键（Esc 取消）…")
        kb_hook.arm_capture(on_captured)

    # -- mic ------------------------------------------------------------------

    def swap_mic(self, name: str) -> None:
        if self._deps.get_recorder() is None:
            return
        self._deps.rebuild_recorder(name)
        config = self._deps.get_config()
        self._deps.set_config(dataclasses.replace(config, mic=name))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        log.info("mic switched to %r", name)

    # -- model ----------------------------------------------------------------

    def swap_model(self, kind: str) -> Recognizer | None:
        """Download-if-needed and build the recognizer for another model.

        Returns the new recognizer (the caller assigns it), or None when the
        switch was a no-op or the kind is unknown. Download progress rides
        the indicator; dictation holds queue while the worker is busy.
        """
        if kind == self._deps.get_config().model:
            return None
        match kind:
            case "sensevoice" | "funasr_nano":
                kind_typed: ModelKind = kind
            case _:
                log.warning("unknown model kind %r", kind)
                return None
        log.info("switching model to %r", kind)
        self._deps.indicator.update("下载模型中…")

        def _progress(name: str, done: int, total: int) -> None:
            pct = done / total if total else 0.0
            self._deps.indicator.progress(pct, f"下载模型 {name}")

        files = ensure_model(kind, self._deps.models_root, _progress)
        recognizer = Recognizer(
            kind=kind_typed,
            model_dir=files.directory,
            num_threads=self._deps.num_threads(),
            language=self._deps.language(),
        )
        config = self._deps.get_config()
        self._deps.set_config(dataclasses.replace(config, model=kind))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        self._deps.indicator.flash("模型已切换", 1200)
        log.info("model switched to %r", kind)
        return recognizer

    # -- diagnostics ------------------------------------------------------------

    def show_diagnostics(self) -> None:
        """Show a human-readable input-channel health report."""
        alive = winio.keyboard_injection_alive()
        control_class = winio.focused_control_class()
        classic = bool(control_class) and is_classic_control(control_class)
        config = self._deps.get_config()
        hotkey_label = (
            winio.key_name(hotkey_vk(config.hotkey))
            if isinstance(config.hotkey, int)
            else config.hotkey
        )
        inject_status = (
            "正常"
            if alive
            else "被拦截（常驻软件的全局钩子吃掉了合成按键，已自动降级备用通道）"
        )
        classic_status = "是（可 WM_PASTE）" if classic else "否（将尝试 UIA/剪贴板提示）"
        lines = [
            f"键盘注入：{inject_status}",
            f"前台控件类：{control_class or '<未知>'}",
            f"经典控件：{classic_status}",
            f"当前热键：{hotkey_label}（vk={hotkey_vk(config.hotkey)}）",
            f"当前模型：{config.model}",
            f"麦克风：{config.mic or '系统默认'}",
        ]
        winutil.show_info("\n".join(lines), title="xxl-whisper 输入诊断")
