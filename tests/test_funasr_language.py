"""FunASR-Nano's ``language`` field is a prompt, not a language code.

Field symptom: Japanese transcribed with SenseVoice translated acceptably, the
same audio on Fun-ASR-Nano translated badly. Cause: we passed ``language="ja"``,
which the model reads as a meaningless prompt fragment (it expects ``日语``), and
a non-empty ``language`` also *silently discards* our own ``user_prompt``.
"""

from __future__ import annotations

import sys
import types

# app.asr reaches the platform through app.native, which is macOS-only here.
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

from app.asr import funasr_language  # noqa: E402


def test_language_is_natural_language_not_a_code() -> None:
    assert funasr_language("ja", has_prompt=False) == "日语"
    assert funasr_language("zh", has_prompt=False) == "中文"
    assert funasr_language("en", has_prompt=False) == "英语"


def test_auto_leaves_the_field_empty() -> None:
    assert funasr_language("auto", has_prompt=False) == ""


def test_unknown_codes_do_not_leak_into_the_prompt() -> None:
    assert funasr_language("ko", has_prompt=False) == ""


def test_custom_prompt_requires_an_empty_language_field() -> None:
    """sherpa-onnx only honours user_prompt when hotwords and language are empty."""
    assert funasr_language("ja", has_prompt=True) == ""
    assert funasr_language("zh", has_prompt=True) == ""
