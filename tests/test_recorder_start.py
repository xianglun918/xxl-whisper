"""Push-to-talk capture start must be safe to call twice (hook + worker).

Regression guard for a leak that only shows on macOS: the hook thread prewarms
the capture on key-down and the worker's ``StartHold`` calls ``start()`` again.
The idempotency check used to sit *after* ``_open_stream()``, so the second call
opened a second stream while ``stop()`` only closed the last one — one leaked
microphone stream (and a lit indicator) per sentence.
"""

import threading

import numpy as np
import pytest
from app.recorder import Recorder

from app import native


def _bare_recorder(
    monkeypatch: pytest.MonkeyPatch, *, keep_open: bool
) -> tuple[Recorder, list[int]]:
    """A Recorder with the stream layer replaced by counters."""
    monkeypatch.setattr(native, "RECORDER_KEEP_OPEN", keep_open)
    recorder = object.__new__(Recorder)
    recorder._lock = threading.Lock()
    recorder._lifecycle = threading.Lock()
    recorder._recording = False
    recorder._buffer = []
    recorder._buffered_samples = 0
    recorder._stream = None
    opened: list[int] = []

    def _open() -> None:
        opened.append(1)
        recorder._stream = object()

    monkeypatch.setattr(recorder, "_open_stream", _open)
    monkeypatch.setattr(recorder, "_close_stream", lambda: setattr(recorder, "_stream", None))
    return recorder, opened


def test_second_start_does_not_open_a_second_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, opened = _bare_recorder(monkeypatch, keep_open=False)

    recorder.start()  # hook thread, on key-down
    recorder.start()  # worker, on StartHold

    assert len(opened) == 1
    assert recorder._recording is True


def test_second_start_keeps_the_prewarmed_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder, _opened = _bare_recorder(monkeypatch, keep_open=False)
    recorder.start()
    recorder._buffer.append(np.ones(8, dtype=np.float32))
    recorder._buffered_samples = 8

    recorder.start()

    assert recorder._buffered_samples == 8  # the prewarm is why this exists


def test_start_after_stop_sink_reopens_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuous mode clears ``_recording``; a later hold must reopen."""
    recorder, opened = _bare_recorder(monkeypatch, keep_open=False)
    recorder.start()
    recorder.start_sink(threading.Event())  # type: ignore[arg-type]

    recorder.start()

    assert len(opened) == 2


def test_oversized_capture_is_dropped_not_shipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture nobody stopped must not be sent to the model as "speech".

    The buffer cap kept memory bounded, but ``stop()`` still concatenated the
    whole thing and handed minutes of stale audio to the recogniser.
    """
    recorder, _opened = _bare_recorder(monkeypatch, keep_open=True)
    recorder.start()
    recorder._buffer.append(np.ones(16_000 * 200, dtype=np.float32))  # 200 s
    recorder._buffered_samples = 16_000 * 200

    assert recorder.stop() is None
