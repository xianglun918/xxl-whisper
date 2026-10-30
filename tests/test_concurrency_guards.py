"""Concurrency and failure guards the unit tests could not express.

Three behaviours were only ever checked by hand-run experiments, so a regression
in them would have shipped silently:

* two threads starting a hold at once (hook thread prewarm + worker StartHold)
  must open exactly one capture stream;
* a failed clipboard write must not release the snapshot, or the user's original
  is lost on the next sentence;
* a stray "restore" left over from a previous sentence must not fight a newer one.
"""

import threading
import time
from types import SimpleNamespace
from typing import cast

import app.recorder as recorder_mod
import numpy as np
import pytest
from app.emit import Channel, EmitSettings, emit_text
from app.recorder import Recorder

from app import native


class _Indicator:
    """Minimal indicator stub."""

    def __init__(self) -> None:
        self.flashes: list[str] = []

    def hide(self) -> None:
        pass

    def flash(self, text: str, _ms: int = 1500) -> None:
        self.flashes.append(text)


def _recorder_with_slow_open(monkeypatch: pytest.MonkeyPatch) -> tuple[Recorder, list[int]]:
    """A Recorder whose ``_open_stream`` takes long enough to race."""
    monkeypatch.setattr(native, "RECORDER_KEEP_OPEN", False)
    recorder = object.__new__(Recorder)
    recorder._lock = threading.Lock()
    recorder._lifecycle = threading.Lock()
    recorder._recording = False
    recorder._buffer = []
    recorder._buffered_samples = 0
    recorder._stream = None
    opened: list[int] = []

    def _open() -> None:
        if recorder._stream is not None:
            return
        time.sleep(0.02)  # wide enough for the other thread to enter start()
        opened.append(1)
        recorder._stream = SimpleNamespace()

    monkeypatch.setattr(recorder, "_open_stream", _open)
    monkeypatch.setattr(recorder, "_close_stream", lambda: setattr(recorder, "_stream", None))
    return recorder, opened


def test_concurrent_starts_open_one_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook thread and the worker can start the same hold at once."""
    recorder, opened = _recorder_with_slow_open(monkeypatch)
    barrier = threading.Barrier(2)

    def _run() -> None:
        barrier.wait()
        recorder.start()

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(opened) == 1


def test_concurrent_start_stop_stays_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    """After start/stop race, capture state and stream presence must agree."""
    recorder, _opened = _recorder_with_slow_open(monkeypatch)
    barrier = threading.Barrier(2)

    def _start() -> None:
        barrier.wait()
        recorder.start()

    def _stop() -> None:
        barrier.wait()
        recorder.stop()

    threads = [threading.Thread(target=_start), threading.Thread(target=_stop)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Either a live hold with a stream, or no capture at all — never a "recording"
    # with no stream (which would silently capture nothing) or a leaked stream.
    assert recorder._recording is (recorder._stream is not None)


def _clipboard(monkeypatch: pytest.MonkeyPatch, *, fail_restore: bool) -> dict[str, object]:
    """Install a stateful fake clipboard whose restore can fail."""
    board: dict[str, object] = {"content": "USER_ORIGINAL", "restored": [], "fail": fail_restore}

    monkeypatch.setattr(native.io, "snapshot_clipboard", lambda: board["content"])
    monkeypatch.setattr(
        native.io, "set_clipboard_text", lambda text: board.__setitem__("content", text)
    )
    monkeypatch.setattr(native.io, "paste_text", lambda *_a, **_k: None)
    monkeypatch.setattr(native.io, "keyboard_injection_alive", lambda: True)
    monkeypatch.setattr(native.io, "clipboard_has_non_text", lambda: False)
    monkeypatch.setattr(native.io, "foreground_window_title", lambda: "w")
    monkeypatch.setattr(native.io, "focused_control_hwnd", lambda: 0)
    monkeypatch.setattr(native.io, "focused_control_class", lambda: "Edit")

    def _restore(snapshot: object) -> None:
        if board["fail"]:
            msg = "clipboard busy"
            raise OSError(msg)
        cast("list[object]", board["restored"]).append(snapshot)
        board["content"] = snapshot

    monkeypatch.setattr(native.io, "restore_clipboard", _restore)
    return board


def test_failed_restore_keeps_the_original_for_the_next_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Releasing the snapshot before the write would lose the user's clipboard."""
    board = _clipboard(monkeypatch, fail_restore=True)
    settings = EmitSettings(restore_clipboard=True, paste_delay_ms=0)

    assert emit_text("第一句", settings, _Indicator()) is Channel.KEYS
    board["fail"] = False  # the clipboard becomes writable again
    assert emit_text("第二句", settings, _Indicator()) is Channel.KEYS

    assert board["content"] == "USER_ORIGINAL"


def test_stale_restore_does_not_clobber_the_newer_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending restore from sentence N must not run after sentence N+1 staged."""
    board = _clipboard(monkeypatch, fail_restore=False)
    slow = EmitSettings(restore_clipboard=True, paste_delay_ms=200)

    emit_text("第一句", slow, _Indicator())
    time.sleep(0.05)
    emit_text("第二句", slow, _Indicator())  # cancels/invalidates the first timer

    time.sleep(0.3)
    # The second sentence's own restore is the one that runs: the user's original
    # comes back exactly once, and never while a sentence is still waiting.
    assert cast("list[object]", board["restored"]) == ["USER_ORIGINAL"]
    assert board["content"] == "USER_ORIGINAL"


def test_partial_hold_buffer_is_not_shipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture nobody stopped must be dropped, not sent to the model."""
    recorder, _opened = _recorder_with_slow_open(monkeypatch)
    recorder._stream = SimpleNamespace()
    recorder._recording = True
    recorder._buffer = [np.zeros(16_000 * 200, dtype=np.float32)]
    recorder._buffered_samples = 16_000 * 200

    assert recorder.stop() is None


def test_real_open_stream_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard inside ``_open_stream`` itself must hold.

    Exercised through the real method (only ``sounddevice`` is stubbed), because a
    fake ``_open_stream`` would hide the guard it is supposed to verify.
    """
    constructed: list[object] = []

    class _FakeStream:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    def _factory(**_kwargs: object) -> _FakeStream:
        constructed.append(1)
        return _FakeStream()

    monkeypatch.setattr(native, "RECORDER_KEEP_OPEN", False)
    monkeypatch.setattr(recorder_mod.sd, "InputStream", _factory)
    instance = object.__new__(Recorder)
    instance._lock = threading.Lock()
    instance._lifecycle = threading.Lock()
    instance._recording = False
    instance._buffer = []
    instance._buffered_samples = 0
    instance._stream = None
    instance._device_name = ""
    instance._source = "mic"

    instance._open_stream()
    instance._open_stream()

    assert len(constructed) == 1
