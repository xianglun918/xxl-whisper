"""Emit channel selection logic.

The Windows-only channels (WM_PASTE/UIA) are asserted on win32 only; the mac
ladder lands in M4.
"""

import sys

import pytest

from app.emit import Channel, TargetProbe, channels_in_order, is_classic_control

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
