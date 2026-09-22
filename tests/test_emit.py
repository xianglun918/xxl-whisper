"""Emit channel selection logic.

The Windows-only channels (WM_PASTE/UIA) are asserted on win32 only; the mac
ladder lands in M4.
"""

import sys

import pytest
from app.emit import (
    Channel,
    DeliveryPath,
    EmitSettings,
    TargetProbe,
    channels_in_order,
    choose_delivery_path,
    emit_text,
    is_classic_control,
)

from app import native

_win_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only channel (WM_PASTE/UIA); mac ladder lands in M4",
)


def test_keys_win_when_injection_alive() -> None:
    channels = channels_in_order(TargetProbe(True, True, True))
    assert channels[0] is Channel.KEYS


@_win_only
def test_wm_paste_leads_for_classic_controls_when_keys_dead() -> None:
    channels = channels_in_order(TargetProbe(False, True, True))
    assert channels == (Channel.WM_PASTE, Channel.UIA, Channel.CLIPBOARD)


@_win_only
def test_uia_leads_for_modern_controls_when_keys_dead() -> None:
    channels = channels_in_order(TargetProbe(False, False, True))
    assert channels == (Channel.UIA, Channel.CLIPBOARD)


def test_clipboard_last_resort() -> None:
    channels = channels_in_order(TargetProbe(False, False, False))
    assert channels == (Channel.CLIPBOARD,)


def test_classic_control_detection() -> None:
    assert is_classic_control("Edit")
    assert is_classic_control("RICHEDIT50W")
    assert is_classic_control("ConsoleWindowClass")
    assert is_classic_control("CASCADIA_HOSTING_WINDOW_CLASS")
    assert not is_classic_control("Chrome_RenderWidgetHostHWND")
    assert not is_classic_control("")


class _FakeIndicator:
    def __init__(self) -> None:
        self.hidden = 0
        self.flashes: list[tuple[str, int]] = []

    def hide(self) -> None:
        self.hidden += 1

    def flash(self, text: str, ms: int = 1500) -> None:
        self.flashes.append((text, ms))


def _stub_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native.io, "foreground_window_title", lambda: "test-window")
    monkeypatch.setattr(native.io, "focused_control_hwnd", lambda: 0)
    monkeypatch.setattr(native.io, "focused_control_class", lambda: "Edit")


def test_choose_delivery_path_type_text_for_non_text_clipboard() -> None:
    assert choose_delivery_path(True) is DeliveryPath.TYPE_TEXT


def test_choose_delivery_path_clipboard_paste_for_text_clipboard() -> None:
    assert choose_delivery_path(False) is DeliveryPath.CLIPBOARD_PASTE


def test_emit_non_text_clipboard_types_without_touching_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: True)
    staged: list[str] = []
    typed: list[str] = []
    pasted: list[tuple[str, bool, int]] = []
    monkeypatch.setattr(native.io, "set_clipboard_text", staged.append)
    monkeypatch.setattr(native.io, "type_text", typed.append)

    def _record_paste(text: str, restore_clipboard: bool, delay_ms: int) -> None:
        pasted.append((text, restore_clipboard, delay_ms))

    monkeypatch.setattr(native.io, "paste_text", _record_paste)
    indicator = _FakeIndicator()

    channel = emit_text("你好", EmitSettings(restore_clipboard=True, paste_delay_ms=30), indicator)

    assert channel is Channel.KEYS
    assert typed == ["你好"]
    assert staged == []
    assert pasted == []
    assert indicator.hidden == 1


def test_emit_non_text_clipboard_dead_injection_leaves_clipboard_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: False)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: True)
    touched: list[str] = []
    monkeypatch.setattr(native.io, "set_clipboard_text", touched.append)
    monkeypatch.setattr(native.io, "type_text", touched.append)
    indicator = _FakeIndicator()

    channel = emit_text("hi", EmitSettings(restore_clipboard=True, paste_delay_ms=10), indicator)

    assert channel is Channel.CLIPBOARD
    assert touched == []
    assert indicator.flashes  # honest terminal message, not "copied to clipboard"


def test_emit_text_clipboard_paste_path_stages_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    staged: list[str] = []
    pasted: list[tuple[str, bool, int]] = []
    monkeypatch.setattr(native.io, "set_clipboard_text", staged.append)

    def _record_paste(text: str, restore_clipboard: bool, delay_ms: int) -> None:
        pasted.append((text, restore_clipboard, delay_ms))

    monkeypatch.setattr(native.io, "paste_text", _record_paste)
    indicator = _FakeIndicator()

    channel = emit_text("hello", EmitSettings(restore_clipboard=True, paste_delay_ms=42), indicator)

    assert channel is Channel.KEYS
    assert staged == ["hello"]
    assert pasted == [("hello", True, 42)]
    assert indicator.hidden == 1
