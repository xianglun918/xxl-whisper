"""Japanese → Chinese translation: route parsing and the two-hop chain.

The models themselves are exercised elsewhere (and only on a machine that has
the weights), so what is tested here is the wiring that is easy to get wrong:
which (source, target) pairs are legal, how a config written before
``translate_source`` existed migrates, and that a chain reports failure instead
of emitting a half-translated sentence.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
from app.translator import ChainedTranslator, TranslationError


class FakeTranslator:
    """Stand-in for one Marian hop, with a caller-visible vocabulary."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return self._mapping.get(text, "")


class BoomTranslator:
    """A hop that fails the way a broken model would."""

    def translate(self, text: str) -> str:
        reason = f"解码失败：{text}"
        raise TranslationError(reason)


def test_chain_runs_both_hops_in_order() -> None:
    first = FakeTranslator({"こんにちは": "hello"})
    second = FakeTranslator({"hello": "你好"})
    chain = ChainedTranslator([first, second])

    assert chain.translate("こんにちは") == "你好"
    assert first.calls == ["こんにちは"]
    assert second.calls == ["hello"]
    assert chain.hops == 2


def test_empty_hop_fails_the_whole_chain() -> None:
    """A hop with no output must not produce a half-translated sentence."""
    first = FakeTranslator({"こんにちは": "hello"})
    second = FakeTranslator({})  # knows nothing -> ""
    chain = ChainedTranslator([first, second])

    assert chain.translate("こんにちは") == ""
    assert second.calls == ["hello"]  # the second hop did run, once


def test_whitespace_only_hop_counts_as_empty() -> None:
    chain = ChainedTranslator([FakeTranslator({"あ": "   "}), FakeTranslator({})])
    assert chain.translate("あ") == ""


def test_single_hop_chain_passes_through() -> None:
    chain = ChainedTranslator([FakeTranslator({"你好": "hello"})])
    assert chain.translate("你好") == "hello"
    assert chain.hops == 1


def test_empty_chain_is_rejected() -> None:
    with pytest.raises(TranslationError):
        ChainedTranslator([])


def test_hop_failure_propagates() -> None:
    chain = ChainedTranslator([BoomTranslator()])
    with pytest.raises(TranslationError):
        chain.translate("あ")


# -- config side ---------------------------------------------------------------


def _load(tmp_path: Path, text: str):
    """Parse a config file, stubbing the platform layer out of the way.

    ``app.config`` reaches the platform through ``app.native`` (keycodes, data
    root), which cannot be imported off Windows/macOS. The stub is installed
    before the module is imported, and the module is re-imported afterwards so
    the stub is what it binds to.
    """
    fake_native = types.ModuleType("app.native")
    fake_native.hotkey = types.SimpleNamespace(  # type: ignore[attr-defined]
        PRESET_KEYCODES={"f2": 0x71}, MOUSE_KEYCODES=frozenset(), DEFAULT_HOTKEY="f2"
    )
    fake_native.data_root = lambda: tmp_path  # type: ignore[attr-defined]
    sys.modules["app.native"] = fake_native
    sys.modules.pop("app.config", None)
    config_mod = importlib.import_module("app.config")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return config_mod, path


def test_japanese_route_is_parsed(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, 'translate_source = "ja"\ntranslate_target = "zh"\n')
    config = config_mod.load_config(path)
    assert (config.translate_source, config.translate_target) == ("ja", "zh")
    assert config_mod.route_of(config) == "ja-zh"


def test_old_config_without_source_infers_en_to_zh(tmp_path: Path) -> None:
    """Configs written before this feature must keep working unchanged."""
    config_mod, path = _load(tmp_path, 'translate_target = "zh"\n')
    config = config_mod.load_config(path)
    assert config.translate_source == "en"
    assert config_mod.route_of(config) == "en-zh"


def test_old_config_without_source_infers_zh_to_en(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, 'translate_target = "en"\n')
    config = config_mod.load_config(path)
    assert config.translate_source == "zh"
    assert config_mod.route_of(config) == "zh-en"


def test_unsupported_pair_falls_back_instead_of_failing(tmp_path: Path) -> None:
    """A same-language pair has no model; the app must still start on a valid one."""
    config_mod, path = _load(tmp_path, 'translate_source = "zh"\ntranslate_target = "zh"\n')
    config = config_mod.load_config(path)
    assert config_mod.route_of(config) is not None  # coerced to a supported pair


def test_japanese_to_chinese_is_a_single_direct_model() -> None:
    """ja→zh is one model now: a community fine-tune we export and quantize.

    The English pivot was the reason Japanese came out worse than zh↔en; the
    direct model is 145 MB instead of 500 MB and about twice as fast.
    """
    from app.downloader import TRANSLATE_OWN_BASE, TRANSLATE_ROUTE_MODELS

    assert TRANSLATE_ROUTE_MODELS["ja-zh"] == ("ja-zh",)
    assert "ja-zh" in TRANSLATE_OWN_BASE  # hosted by us, not Helsinki


def test_japanese_to_english_is_a_single_hop(tmp_path: Path) -> None:
    """One model, no pivot: the best quality/speed/memory trade for Japanese."""
    config_mod, path = _load(
        tmp_path, 'translate_source = "ja"\ntranslate_target = "en"\n'
    )
    config = config_mod.load_config(path)
    assert config_mod.route_of(config) == "ja-en"

    from app.downloader import TRANSLATE_ROUTE_MODELS

    assert TRANSLATE_ROUTE_MODELS["ja-en"] == ("ja-en",)
    # ja→zh is one direct model now (the English pivot was the quality problem).
    assert TRANSLATE_ROUTE_MODELS["ja-zh"] == ("ja-zh",)


def test_japanese_route_roundtrips(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, 'translate_source = "ja"\ntranslate_target = "zh"\n')
    original = config_mod.load_config(path)
    config_mod.save_config(path, original)
    assert config_mod.load_config(path) == original


def test_audio_source_defaults_to_microphone(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, "")
    assert config_mod.load_config(path).audio_source == "mic"


def test_audio_source_accepts_system_and_roundtrips(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, 'audio_source = "system"\nmic = "Speakers"\n')
    original = config_mod.load_config(path)
    assert original.audio_source == "system"
    config_mod.save_config(path, original)
    assert config_mod.load_config(path) == original


def test_audio_source_rejects_unknown_values(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, 'audio_source = "bluetooth"\n')
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(path)


def test_typing_only_mode_means_no_subtitle_bar(tmp_path: Path) -> None:
    """One decision, not two: "只打字" cannot coexist with an enabled bar."""
    config_mod, path = _load(tmp_path, 'overlay_enabled = true\noverlay_output = "type"\n')
    assert config_mod.load_config(path).overlay_enabled is False


def test_switching_mode_turns_the_bar_on_and_off(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, "")
    config = config_mod.load_config(path)
    assert config.overlay_enabled is False

    screen = config_mod.with_overlay_mode(config, "screen")
    assert (screen.overlay_enabled, screen.overlay_output) == (True, "screen")

    both = config_mod.with_overlay_mode(screen, "both")
    assert (both.overlay_enabled, both.overlay_output) == (True, "both")

    typing = config_mod.with_overlay_mode(both, "type")
    assert (typing.overlay_enabled, typing.overlay_output) == (False, "type")


def test_typing_only_keeps_the_old_loose_segmentation(tmp_path: Path) -> None:
    """No bar means no force-cut: typing wants whole sentences, not 4 s pieces."""
    config_mod, path = _load(tmp_path, "")
    typing = config_mod.with_overlay_mode(config_mod.load_config(path), "type")
    assert config_mod.segmenter_limits(typing) == (500, 20_000)


def test_subtitle_modes_use_the_fast_segmentation(tmp_path: Path) -> None:
    config_mod, path = _load(tmp_path, "")
    config = config_mod.load_config(path)
    for mode in ("screen", "both"):
        assert config_mod.segmenter_limits(
            config_mod.with_overlay_mode(config, mode)
        ) == (350, 4_000)


def test_segmentation_limits_follow_the_config(tmp_path: Path) -> None:
    config_mod, path = _load(
        tmp_path,
        "live_max_speech_ms = 6000\ntyping_max_speech_ms = 30000\n"
        'typing_min_silence_ms = 700\noverlay_output = "screen"\n'
        "overlay_enabled = true\n",
    )
    config = config_mod.load_config(path)
    assert config_mod.segmenter_limits(config) == (350, 6_000)
    typing = config_mod.with_overlay_mode(config, "type")
    assert config_mod.segmenter_limits(typing) == (700, 30_000)
