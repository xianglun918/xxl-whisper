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
from typing import cast

from app import native
from app.asr import Recognizer
from app.config import (
    MOUSE_VKS,
    AudioSource,
    Config,
    Disfluency,
    ModelKind,
    TranslateSource,
    TranslateTarget,
    config_path,
    hotkey_vk,
    language_for_route,
    route_of,
    save_config,
    translate_route,
)
from app.downloader import (
    TRANSLATE_ROUTE_MODELS,
    DownloadError,
    ensure_model,
    ensure_translate_model,
    manual_download_guide,
    translate_manual_guide,
)
from app.emit import is_classic_control
from app.recorder import Recorder
from app.translator import (
    ChainedTranslator,
    NmtSettings,
    TranslationError,
    Translator,
    load_translator,
    resolve_special_ids,
)
from app.tray import Tray

log = logging.getLogger(__name__)

_VK_DISABLED: int = 0


def disfluency_for_model(kind: str, current: Disfluency) -> Disfluency:
    """Disfluency defaults to smooth when switching to the LLM-decoder model."""
    return "smooth" if kind == "funasr_nano" else current


@dataclass(frozen=True, slots=True)
class ControlsDeps:
    """Live dependencies the panel actions mutate through."""

    indicator: native.indicator.Indicator
    tray: Tray
    get_config: Callable[[], Config]
    set_config: Callable[[Config], None]
    get_hooks: Callable[
        [], tuple[native.hotkey.HotkeyHook | None, native.mousehook.MouseHook | None]
    ]
    get_recorder: Callable[[], Recorder | None]
    rebuild_recorder: Callable[[str], Recorder]
    release_recognizer: Callable[[], None]
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
        label = native.io.key_name(vk) if isinstance(key, int) else key
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
        # Picking a real microphone also leaves system-audio mode.
        self._deps.set_config(dataclasses.replace(config, mic=name, audio_source="mic"))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        log.info("mic switched to %r", name)

    def swap_system_audio(self, device_id: str) -> None:
        """Capture the system mix instead of the microphone (Windows loopback)."""
        if self._deps.get_recorder() is None:
            return
        self._deps.rebuild_recorder(device_id)
        config = self._deps.get_config()
        self._deps.set_config(
            dataclasses.replace(
                config, mic=device_id, audio_source=cast("AudioSource", "system")
            )
        )
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        self._deps.indicator.flash("音频来源：系统声音（回环）", 2500)
        log.info("audio source -> system (%r)", device_id)

    # -- model ----------------------------------------------------------------

    def rebuild_recognizer(self) -> Recognizer | None:
        """Rebuild the current model with the current config (e.g. ``language``).

        Choosing a 日→… translation route pins ``language`` to ``ja``. Without a
        rebuild the *running* recognizer keeps its old language, so Japanese
        audio is decoded as Chinese — the translator then faithfully translates
        gibberish, which reads as "translation is completely wrong".
        """
        try:
            return self._build_recognizer(self._deps.get_config().model)
        except Exception:
            log.exception("recognizer rebuild failed")
            return None

    def swap_model(self, kind: str) -> Recognizer | None:
        """Download-if-needed and build the recognizer for another model.

        Returns the new recognizer (the caller assigns it), or None when the
        switch was a no-op or the kind is unknown. Download progress rides
        the indicator; dictation holds queue while the worker is busy.
        """
        if kind == self._deps.get_config().model:
            return None
        match kind:
            case "sensevoice" | "funasr_nano" | "whisper_turbo":
                kind_typed: ModelKind = kind
            case _:
                log.warning("unknown model kind %r", kind)
                return None
        log.info("switching model to %r", kind)
        self._deps.indicator.update("下载模型中…")

        def _progress(name: str, done: int, total: int) -> None:
            pct = done / total if total else 0.0
            self._deps.indicator.progress(pct, f"下载模型 {name}")

        try:
            ensure_model(
                kind,
                self._deps.models_root,
                _progress,
                proxy=self._deps.get_config().proxy,
            )
        except DownloadError as exc:
            log.warning("model download failed: %s", exc)
            self._deps.indicator.hide()
            guide = manual_download_guide(kind, self._deps.models_root)
            native.util.show_info(f"模型自动下载失败：{exc.reason}\n\n{guide}")
            return None
        config = self._deps.get_config()
        new_disfluency = disfluency_for_model(kind, config.disfluency)
        self._deps.set_config(
            dataclasses.replace(config, model=kind, disfluency=new_disfluency)
        )
        save_config(config_path(), self._deps.get_config())
        # Release the outgoing model *before* allocating the new one so the two
        # never overlap in memory (peak ~1.25 GB -> one model). The decode
        # threads re-read the recognizer and skip while it is None.
        self._deps.release_recognizer()
        recognizer = self._build_recognizer(kind_typed)
        self._deps.tray.refresh_menu()
        self._deps.indicator.flash("模型已切换", 1200)
        log.info("model switched to %r (disfluency=%s)", kind, new_disfluency)
        return recognizer

    def set_disfluency(self, kind: str) -> Recognizer | None:
        """Select verbatim/smooth; rebuild the recognizer when it matters.

        Only Fun-ASR-Nano honours the disfluency prompt; SenseVoice ignores it,
        so a rebuild is only needed for the LLM-decoder model.
        """
        config = self._deps.get_config()
        new = cast("Disfluency", kind)
        if new == config.disfluency:
            return None
        self._deps.set_config(dataclasses.replace(config, disfluency=new))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        if config.model == "funasr_nano":
            log.info("disfluency -> %s: rebuilding funasr_nano recognizer", new)
            self._deps.release_recognizer()  # drop the old model before rebuilding
            return self._build_recognizer(config.model)
        log.warning(
            "disfluency -> %s: model=%s cannot apply smoothing (only funasr_nano)",
            new,
            config.model,
        )
        return None

    # -- translation ------------------------------------------------------------

    def toggle_translate(self) -> bool:
        """Flip offline translation on/off, persist it, return the new state."""
        config = self._deps.get_config()
        enabled = not config.translate
        self._deps.set_config(dataclasses.replace(config, translate=enabled))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        log.info(
            "translate -> %s (target=%s mode=%s)",
            enabled,
            config.translate_target,
            config.translate_mode,
        )
        return enabled

    def set_translate_route(self, route: str) -> bool:
        """Switch translation direction ("zh-en" / "en-zh" / "ja-zh").

        Returns True when the route actually changed, so the caller knows it
        must rebuild the translator (each direction is a different model file,
        and 日→中 needs two).

        The recogniser language follows the route: choosing 日→中 pins
        ``language`` to ``ja`` (SenseVoice decodes Japanese as Chinese
        otherwise), and leaving it restores a sensible default. A user who has
        deliberately set ``auto`` keeps it.
        """
        config = self._deps.get_config()
        source_part, sep, target_part = route.partition("-")
        if not sep:
            log.warning("unknown translate route %r", route)
            return False
        new_route = translate_route(source_part, target_part)
        if new_route is None:
            log.warning("unsupported translate route %r", route)
            return False
        if new_route == route_of(config):
            return False
        source, target = new_route.split("-")
        language = language_for_route(config.language, new_route)
        _ = (source, target)  # parsed above for validation only
        self._deps.set_config(
            dataclasses.replace(
                config,
                translate_source=cast("TranslateSource", source),
                translate_target=cast("TranslateTarget", target),
                language=language,
            )
        )
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()
        log.info(
            "translate route -> %s (language=%s%s)",
            new_route,
            language,
            "; recognizer must be rebuilt" if language != config.language else "",
        )
        if source == "ja":
            self._deps.indicator.flash("已切到日→中：识别语言已设为 ja", 2500)
        elif target == "zh" and language == "auto":
            self._deps.indicator.flash("已切到英→中：识别语言已设为 auto", 2500)
        return True

    def _nmt_settings(self, model_dir: Path, pair: str) -> NmtSettings:
        """Per-pair settings: each Marian export has its own vocab and start id."""
        start, eos, unk = resolve_special_ids(model_dir)
        return NmtSettings(
            model_dir=model_dir,
            target=pair.split("-")[1],
            num_threads=self._deps.num_threads(),
            start_token_id=start,
            eos_token_id=eos,
            unk_token_id=unk,
        )

    def build_translator(
        self, route: str | None = None, *, disable_on_failure: bool = True
    ) -> Translator | None:
        """Download-if-needed and load the local NMT model(s) for the route.

        Runs on the worker thread (blocking download + model load). A failure
        disables translation rather than retrying on every utterance: a user who
        cannot fetch the model should get one clear dialog, not a stall on each
        sentence. Returns None when translation is off or could not start.

        日→中 needs two models (ja→en, then en→zh), so the loader walks the
        route's pairs and chains them; a single-hop route is just a chain of
        one and behaves exactly as before.
        """
        config = self._deps.get_config()
        if not config.translate:
            return None
        wanted = route or route_of(config)
        if wanted is None:
            log.warning(
                "translate route unsupported: %s→%s",
                config.translate_source,
                config.translate_target,
            )
            return None
        route = wanted
        pairs = TRANSLATE_ROUTE_MODELS[route]

        def _progress(name: str, done: int, total: int) -> None:
            pct = done / total if total else 0.0
            self._deps.indicator.progress(pct, f"下载翻译模型 {Path(name).name}")

        try:
            stages: list[Translator] = []
            for pair in pairs:
                files = ensure_translate_model(
                    pair,
                    self._deps.models_root,
                    _progress,
                    proxy=config.proxy,
                )
                stages.append(
                    load_translator(
                        self._nmt_settings(files.directory, pair)
                    )
                )
            translator: Translator = (
                stages[0] if len(stages) == 1 else ChainedTranslator(stages)
            )
        except (DownloadError, TranslationError) as exc:
            reason = getattr(exc, "reason", str(exc))
            log.warning("translator unavailable: %s", reason)
            self._deps.indicator.hide()
            guide = "\n\n".join(
                translate_manual_guide(pair, self._deps.models_root) for pair in pairs
            )
            native.util.show_info(
                f"翻译模型不可用：{reason}\n\n{guide}\n（翻译已关闭，语音听写不受影响）"
            )
            if disable_on_failure:
                self.disable_translate()
            return None
        hops = " → ".join(pair.split("-")[0] for pair in pairs) + f" → {pairs[-1].split('-')[1]}"
        self._deps.indicator.flash(f"翻译已开启（本地模型 {hops}）", 1800)
        log.info("translator ready: route=%s hops=%s", route, pairs)
        return translator

    def disable_translate(self) -> None:
        """Turn translation off and persist it (used after a load failure)."""
        config = self._deps.get_config()
        if not config.translate:
            return
        self._deps.set_config(dataclasses.replace(config, translate=False))
        save_config(config_path(), self._deps.get_config())
        self._deps.tray.refresh_menu()

    def _build_recognizer(self, kind: ModelKind) -> Recognizer:
        return Recognizer(
            kind=kind,
            model_dir=self._deps.models_root / kind,
            num_threads=self._deps.num_threads(),
            language=self._deps.language(),
            disfluency=self._deps.get_config().disfluency,
        )

    # -- diagnostics ------------------------------------------------------------

    def show_diagnostics(self) -> None:
        """Show a human-readable input-channel health report."""
        alive = native.io.keyboard_injection_alive()
        control_class = native.io.focused_control_class()
        classic = bool(control_class) and is_classic_control(control_class)
        config = self._deps.get_config()
        hotkey_label = (
            native.io.key_name(hotkey_vk(config.hotkey))
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
            *native.util.permission_report(),
        ]
        native.util.show_info("\n".join(lines), title="xxl-whisper 输入诊断")
