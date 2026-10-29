"""Every menu option must be a value the config actually accepts.

This is the guard against the class of bug that made the hotkey and filler-word
rows render without a tick: three hand-copied tables (menu label, menu key, config
literal) drifted apart, and nothing failed loudly — the app just showed a menu
whose choices were not the ones it honoured.

If a config vocabulary changes and ``app/options.py`` is not updated, this test
fails instead of a user discovering a dead row.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, get_args

from app.options import (
    DISFLUENCY_OPTIONS,
    HOTKEY_OPTIONS,
    MODEL_OPTIONS,
    OUTPUT_OPTIONS,
    ROUTE_OPTIONS,
    label_of,
)


def _values(alias: Any) -> set[str]:
    """Values of a ``type X = Literal[...]`` alias (PEP 695 wraps it)."""
    inner = getattr(alias, "__value__", alias)
    return set(get_args(inner))


def _config_module(tmp_path: Path):
    """Import app.config with the platform layer stubbed out (headless safe)."""
    fake = types.ModuleType("app.native")
    fake.hotkey = types.SimpleNamespace(
        PRESET_KEYCODES={
            "caps_lock": 0x14,
            "f2": 0x71,
            "f4": 0x73,
            "f6": 0x75,
            "f8": 0x77,
            "scroll_lock": 0x91,
            "mouse_x1": 0x05,
            "mouse_x2": 0x06,
        },
        MOUSE_KEYCODES=frozenset({0x05, 0x06}),
        DEFAULT_HOTKEY="caps_lock",
    )
    fake.data_root = lambda: tmp_path
    fake.io = types.SimpleNamespace()
    fake.overlay = types.SimpleNamespace()
    sys.modules["app.native"] = fake
    sys.modules.pop("app.config", None)
    return importlib.import_module("app.config")


def test_option_keys_are_legal_config_values(tmp_path: Path) -> None:
    config = _config_module(tmp_path)

    hotkeys = set(config.HOTKEY_VK)
    for option in HOTKEY_OPTIONS:
        assert option.key in hotkeys, f"hotkey {option.key!r} is not in HOTKEY_VK"

    models = _values(config.ModelKind)
    for option in MODEL_OPTIONS:
        assert option.key in models, f"model {option.key!r} is not a ModelKind"

    disfluency = _values(config.Disfluency)
    for option in DISFLUENCY_OPTIONS:
        assert option.key in disfluency, f"disfluency {option.key!r} is not allowed"

    routes = _values(config.TranslateRoute)
    for option in ROUTE_OPTIONS:
        assert option.key in routes, f"route {option.key!r} is not a TranslateRoute"

    outputs = _values(config.OverlayOutput)
    for option in OUTPUT_OPTIONS:
        assert option.key in outputs, f"output {option.key!r} is not an OverlayOutput"


def test_no_config_value_is_missing_from_the_menu(tmp_path: Path) -> None:
    """The other direction: a new config value must get a menu entry, not vanish."""
    config = _config_module(tmp_path)

    assert _values(config.ModelKind) == {option.key for option in MODEL_OPTIONS}
    assert _values(config.Disfluency) == {option.key for option in DISFLUENCY_OPTIONS}
    assert _values(config.TranslateRoute) == {option.key for option in ROUTE_OPTIONS}
    assert _values(config.OverlayOutput) == {option.key for option in OUTPUT_OPTIONS}
    # Hotkey presets are platform-owned; every one of them must be pickable.
    assert set(config.HOTKEY_VK) == {option.key for option in HOTKEY_OPTIONS}


def test_labels_are_unique_and_non_empty() -> None:
    for options in (
        HOTKEY_OPTIONS,
        MODEL_OPTIONS,
        DISFLUENCY_OPTIONS,
        ROUTE_OPTIONS,
        OUTPUT_OPTIONS,
    ):
        labels = [option.label for option in options]
        assert all(labels), "empty label"
        assert len(set(labels)) == len(labels), f"duplicate labels in {labels}"


def test_custom_hotkey_renders_readably() -> None:
    """A recorded key is a raw VK integer and must not print as a bare number."""
    assert label_of(HOTKEY_OPTIONS, 0x71) == "自定义按键（VK 0x71）"
    assert label_of(HOTKEY_OPTIONS, "caps_lock") == "CapsLock"


def test_every_route_has_a_label_and_a_model(tmp_path: Path) -> None:
    """The route vocabulary lives in four places; they must agree.

    A route present in ``config.TranslateRoute`` but missing from
    ``TRANSLATE_ROUTES`` raised ``KeyError`` inside the worker, so clicking the
    new 日→英 option failed with a traceback while every other test passed.
    """
    from app.downloader import TRANSLATE_ROUTE_MODELS
    from app.translator import TRANSLATE_ROUTES

    config = _config_module(tmp_path)
    vocabulary = _values(config.TranslateRoute)
    menu = {option.key for option in ROUTE_OPTIONS}

    assert vocabulary == menu == set(TRANSLATE_ROUTES) == set(TRANSLATE_ROUTE_MODELS), (
        vocabulary,
        menu,
        set(TRANSLATE_ROUTES),
        set(TRANSLATE_ROUTE_MODELS),
    )
    for route in vocabulary:
        assert TRANSLATE_ROUTES[route].strip(), f"{route} has an empty label"
        assert TRANSLATE_ROUTE_MODELS[route], f"{route} has no model chain"


def test_route_text_never_raises_for_unknown_routes() -> None:
    from app.translator import route_text

    assert route_text("ja-en")
    assert route_text("xx-yy") == "xx-yy"
