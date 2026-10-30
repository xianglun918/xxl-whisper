"""Emit channel selection logic.

The Windows-only channels (WM_PASTE/UIA) are asserted on win32 only; the mac
ladder lands in M4.
"""

import sys
import time
from types import SimpleNamespace
from typing import cast

import pytest
from app.emit import (
    Channel,
    DeliveryPath,
    EmitSettings,
    ProbeCache,
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
    monkeypatch.setattr(native.io, "snapshot_clipboard", lambda: "OLD")
    monkeypatch.setattr(native.io, "restore_clipboard", lambda _snapshot: None)


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


def test_emit_text_snapshots_the_clipboard_before_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the snapshot must be taken *before* the stage.

    Taking it afterwards read our own dictated text back, so the "restore" put
    that text on the clipboard and destroyed whatever the user had copied —
    while the log claimed the clipboard had been restored.
    """
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "OLD")

    def _paste(text: str, *_args: object, **_kwargs: object) -> None:
        cast("list[str]", board["order"]).append("paste")

    monkeypatch.setattr(native.io, "paste_text", _paste)
    indicator = _FakeIndicator()

    channel = emit_text("hello", EmitSettings(restore_clipboard=True, paste_delay_ms=0), indicator)

    assert channel is Channel.KEYS
    assert board["order"][0] == "snapshot"  # before the stage
    assert board["order"][1] == "stage:hello"
    assert "paste" in cast("list[str]", board["order"])
    # The user's original comes back; a restore right after the stage would have
    # written "hello" back instead.
    assert board["restored"] == ["OLD"]
    assert board["content"] == "OLD"
    assert indicator.hidden == 1


def test_restore_disabled_never_touches_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``restore_clipboard = false`` must not snapshot or restore anything."""
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    events: list[str] = []
    monkeypatch.setattr(native.io, "snapshot_clipboard", lambda: events.append("snapshot"))
    monkeypatch.setattr(
        native.io, "restore_clipboard", lambda _snap: events.append("restore")
    )
    monkeypatch.setattr(
        native.io, "set_clipboard_text", lambda text: events.append(f"stage:{text}")
    )
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)

    emit_text("hi", EmitSettings(restore_clipboard=False, paste_delay_ms=0), _FakeIndicator())

    assert events == ["stage:hi"]


def _stateful_clipboard(
    monkeypatch: pytest.MonkeyPatch, initial: object
) -> dict[str, object]:
    """Install a fake clipboard that actually holds its content."""
    board: dict[str, object] = {
        "content": initial,
        "snapshots": 0,
        "restored": [],
        "order": [],
    }

    def _order(event: str) -> None:
        cast("list[str]", board["order"]).append(event)

    def _snapshot() -> object:
        board["snapshots"] = int(board["snapshots"]) + 1
        _order("snapshot")
        return board["content"]

    def _stage(text: str) -> None:
        _order(f"stage:{text}")
        board["content"] = text

    def _restore(snapshot: object) -> None:
        _order(f"restore:{snapshot}")
        cast("list[object]", board["restored"]).append(snapshot)
        board["content"] = snapshot

    monkeypatch.setattr(native.io, "snapshot_clipboard", _snapshot)
    monkeypatch.setattr(native.io, "set_clipboard_text", _stage)
    monkeypatch.setattr(native.io, "restore_clipboard", _restore)
    return board


def test_burst_keeps_the_original_snapshot_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sentences in a burst must still restore what the user had copied.

    Regression: the second sentence took a fresh snapshot — of the *first*
    sentence's text — and the stage cancelled the only good snapshot, so the
    user's clipboard was lost exactly when talking continuously.
    """
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)
    indicator = _FakeIndicator()
    settings = EmitSettings(restore_clipboard=True, paste_delay_ms=150)

    emit_text("第一句", settings, indicator)
    emit_text("第二句", settings, indicator)

    assert board["snapshots"] == 2  # read every time, but the value must not change
    assert board["content"] == "第二句"  # the last sentence is what waits to be pasted

    deadline = time.monotonic() + 2.0
    while not board["restored"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert board["restored"] == ["USER_ORIGINAL"]
    assert board["content"] == "USER_ORIGINAL"


def test_new_copy_between_bursts_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale snapshot must not clobber something the user copied later."""
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)
    indicator = _FakeIndicator()
    settings = EmitSettings(restore_clipboard=True, paste_delay_ms=0)

    emit_text("第一句", settings, indicator)
    assert board["content"] == "USER_ORIGINAL"  # restored immediately (delay 0)

    board["content"] = "USER_NEW_COPY"  # the user copies something else
    emit_text("第二句", settings, indicator)

    assert board["restored"][-1] == "USER_NEW_COPY"


def test_copy_during_the_restore_window_is_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user copies something while a restore is still pending.

    The pending snapshot is the burst's original; if the clipboard no longer holds
    our staged text, that snapshot is stale and must be replaced — otherwise the
    late restore overwrites the fresh copy.
    """
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)
    indicator = _FakeIndicator()
    slow = EmitSettings(restore_clipboard=True, paste_delay_ms=1_000)

    emit_text("第一句", slow, indicator)  # restore is now pending for 1 s
    board["content"] = "USER_NEW_COPY"  # the user copies something else
    emit_text("第二句", slow, indicator)

    deadline = time.monotonic() + 3.0
    while not board["restored"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert board["restored"][-1] == "USER_NEW_COPY"


def test_copy_inside_the_window_is_kept_when_no_new_sentence_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user copies during the restore window and then says nothing.

    The late restore must notice the clipboard is no longer our staged text and
    leave the fresh copy alone.
    """
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)

    emit_text("一句", EmitSettings(restore_clipboard=True, paste_delay_ms=150), _FakeIndicator())
    board["content"] = "USER_NEW_COPY"  # the user copies something else

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if any(str(item).startswith("restore") for item in cast("list[str]", board["order"])):
            break
        if int(board["snapshots"]) > 2:  # restore ran and re-read the clipboard
            break
        time.sleep(0.02)
    time.sleep(0.05)

    assert board["restored"] == []  # nothing written back
    assert board["content"] == "USER_NEW_COPY"


def test_undelivered_text_stays_on_the_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the fallback used to restore the original while telling the
    user to paste manually — they pasted their old clipboard instead."""
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: False)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    monkeypatch.setattr(native.io, "focused_control_class", lambda: "SomeUnknownControl")
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "probe_focused", lambda: SimpleNamespace(writable=False))
    monkeypatch.setattr(native.io, "post_wm_paste_to_focus", lambda: False)
    monkeypatch.setattr(native.uia, "probe_focused", lambda: SimpleNamespace(writable=False))
    indicator = _FakeIndicator()

    channel = emit_text(
        "退化句", EmitSettings(restore_clipboard=True, paste_delay_ms=0), indicator
    )

    assert channel is Channel.CLIPBOARD
    assert board["content"] == "退化句"  # must stay, or the paste hint lies
    assert board["restored"] == []
    assert any("剪贴板" in text for text, _ms in indicator.flashes)


def test_uia_delivery_restores_the_clipboard_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: only KEYS set ``delivered``, so UIA/WM_PASTE never restored."""
    _stub_target(monkeypatch)
    monkeypatch.setattr(
        native.io, "focused_control_class", lambda: "Chrome_RenderWidgetHostHWND"
    )
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: False)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    board = _stateful_clipboard(monkeypatch, "USER_ORIGINAL")
    monkeypatch.setattr(native.io, "probe_focused", lambda: SimpleNamespace(writable=True))
    monkeypatch.setattr(native.io, "append_text", lambda _text: None)
    monkeypatch.setattr(native.uia, "probe_focused", lambda: SimpleNamespace(writable=True))
    monkeypatch.setattr(native.uia, "append_text", lambda _text: None)

    channel = emit_text(
        "hello", EmitSettings(restore_clipboard=True, paste_delay_ms=200), _FakeIndicator()
    )

    assert channel is Channel.UIA
    # Immediately: nothing read the clipboard on this path, so waiting 200 ms
    # would only delay putting the user's content back.
    assert board["restored"] == ["USER_ORIGINAL"]
    assert board["content"] == "USER_ORIGINAL"


class _Clock:
    """Deterministic monotonic clock for the TTL tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_probe_cache_reuses_a_value_within_the_ttl() -> None:
    clock = _Clock()
    cache = ProbeCache(ttl_s=4.0, clock=clock)
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return True

    assert cache.get(probe) is True
    clock.now = 3.9
    assert cache.get(probe) is True
    assert len(calls) == 1


def test_probe_cache_refreshes_once_the_ttl_elapses() -> None:
    clock = _Clock()
    cache = ProbeCache(ttl_s=4.0, clock=clock)
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return False  # a genuinely new blocker appears

    assert cache.get(probe) is False
    clock.now = 4.0
    assert cache.get(probe) is False
    assert len(calls) == 2


def test_probe_cache_does_not_serve_one_probe_another_probes_answer() -> None:
    clock = _Clock()
    cache = ProbeCache(ttl_s=4.0, clock=clock)

    def alive() -> bool:
        return True

    def dead() -> bool:
        return False

    assert cache.get(alive) is True
    assert cache.get(dead) is False  # identity change forces a fresh probe


def test_emit_text_probes_injection_once_per_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst of sentences must not re-press F13 for every emit."""
    _stub_target(monkeypatch)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    monkeypatch.setattr(native.io, "set_clipboard_text", lambda _text: None)

    def _noop_paste(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(native.io, "paste_text", _noop_paste)
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(native.io, "keyboard_injection_alive", probe)
    indicator = _FakeIndicator()
    settings = EmitSettings(restore_clipboard=True, paste_delay_ms=0)

    emit_text("a", settings, indicator)
    emit_text("b", settings, indicator)

    assert len(calls) == 1
