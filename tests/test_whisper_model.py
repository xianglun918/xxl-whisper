"""Whisper language codes (and the model-table entry that goes with them).

Whisper takes ISO codes and uses ``""`` for auto-detect — unlike FunASR-Nano,
whose field is a natural-language prompt. Getting these two backwards is silent:
the model still runs, it just transcribes the wrong language.
"""

from __future__ import annotations

import sys
import types

_fake_native = types.ModuleType("app.native")
_fake_native.__dict__.update(
    {
        "data_root": lambda: None,
        "RECORDER_KEEP_OPEN": False,
        "hotkey": types.SimpleNamespace(
            PRESET_KEYCODES={}, MOUSE_KEYCODES={}, DEFAULT_HOTKEY="caps_lock"
        ),
        "util": types.SimpleNamespace(),
        "overlay": types.SimpleNamespace(),
    }
)
sys.modules.setdefault("app.native", _fake_native)

from app.asr import funasr_language, whisper_language  # noqa: E402


def test_whisper_uses_codes() -> None:
    assert whisper_language("ja") == "ja"
    assert whisper_language("zh") == "zh"
    assert whisper_language("en") == "en"


def test_whisper_auto_is_empty() -> None:
    assert whisper_language("auto") == ""


def test_whisper_and_funasr_disagree_on_purpose() -> None:
    """Same config value, two very different APIs — the reason both helpers exist."""
    assert whisper_language("ja") == "ja"
    assert funasr_language("ja", has_prompt=False) == "日语"


def test_whisper_model_files_are_registered() -> None:
    from app.downloader import _MODEL_FILES

    files = {dest for _url, dest, _size, _sha in _MODEL_FILES["whisper_turbo"]}
    assert files == {"turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"}
