"""The recogniser language follows the translation route.

Field symptom: Japanese audio was "translated completely wrong". The ASR log
showed Chinese text (`看哇伊吧。`, `骗子。`) because the *running* SenseVoice was
still decoding in Chinese while the config already said `ja` — changing the route
updated the config but never rebuilt the recognizer.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config_module():
    """Import app.config with a stub platform layer (no AppKit on Linux)."""
    fake = types.ModuleType("app.native")
    fake.__dict__.update(
        {
            "data_root": lambda: ROOT / ".tmp-test",
            "RECORDER_KEEP_OPEN": False,
            "hotkey": types.SimpleNamespace(
                PRESET_KEYCODES={"caps_lock": 20},
                MOUSE_KEYCODES={},
                DEFAULT_HOTKEY="caps_lock",
            ),
            "util": types.SimpleNamespace(autostart_enabled=lambda: False),
            "overlay": types.SimpleNamespace(),
        }
    )
    sys.modules.setdefault("app.native", fake)
    sys.modules.pop("app.config", None)
    return importlib.import_module("app.config")


def test_japanese_route_pins_ja(config_module) -> None:
    assert config_module.language_for_route("zh", "ja-zh") == "ja"
    assert config_module.language_for_route("auto", "ja-en") == "ja"


def test_leaving_japanese_route_restores_a_default(config_module) -> None:
    assert config_module.language_for_route("ja", "zh-en") == "auto"
    assert config_module.language_for_route("ja", "en-zh") == "auto"


def test_non_japanese_routes_keep_the_users_choice(config_module) -> None:
    assert config_module.language_for_route("auto", "zh-en") == "auto"
    assert config_module.language_for_route("en", "en-zh") == "en"


def test_recognizer_rebuild_entry_point_exists() -> None:
    """The route switch must be able to rebuild the recogniser, not just config."""
    source = (ROOT / "app" / "controls.py").read_text(encoding="utf-8")
    assert "def rebuild_recognizer(" in source
    app_source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert "self._controls.rebuild_recognizer()" in app_source
