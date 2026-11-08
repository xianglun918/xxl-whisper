"""Model download/load runs off the worker thread (P2).

The worker must only do state-machine decisions; blocking I/O (``ensure_model``
+ ``Recognizer`` build, translator download) is pushed onto a loader thread
that posts a ``ModelLoaded`` event back to the queue. These tests pin that
contract without constructing the full GUI app.
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import app.app as app_module
import pytest
from app.app import DictationApp, ModelLoaded


def _bare_app() -> DictationApp:
    """Build a DictationApp without ``__init__``; set only what these tests touch."""
    app = object.__new__(DictationApp)
    app._config = SimpleNamespace(
        model="sensevoice",
        num_threads=2,
        language="zh",
        disfluency="smooth",
        proxy="",
        translate=False,
        continuous=False,
    )
    app._queue = queue.Queue()
    app._recognizer = None
    app._translator = None
    app._model_ready = False
    app._model_downloaded = False
    app._model_generation = 0
    app._model_load_started_at = 0.0
    app._model_load_purpose = ""
    app._started_at = time.monotonic()
    app._hook = None
    app._mouse_hook = None
    app._session = None
    app._indicator = SimpleNamespace(
        show=lambda *_a: None,
        hide=lambda: None,
        flash=lambda *_a, **_k: None,
        update=lambda *_a: None,
        progress=lambda *_a: None,
    )
    app._tray = SimpleNamespace(refresh_menu=lambda: None, notify=lambda *_a, **_k: None)
    app._subtitles = lambda: False
    return app


class _FakeRecognizer:
    def __init__(self, *, files, num_threads, language, disfluency):
        self.files = files
        self.num_threads = num_threads
        self.language = language
        self.disfluency = disfluency


def test_start_recognizer_load_snapshots_and_posts_model_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _bare_app()
    seen: dict[str, object] = {}

    def fake_ensure_model(kind, _root, progress, *, proxy=""):
        seen["kind"] = kind
        seen["proxy"] = proxy
        progress("model.onnx", 1, 1)
        return SimpleNamespace()

    monkeypatch.setattr(app_module, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(app_module, "Recognizer", _FakeRecognizer)

    app._start_recognizer_load("sensevoice", purpose="init")

    msg = app._queue.get(timeout=5)
    assert isinstance(msg, ModelLoaded)
    assert msg.generation == 1
    assert isinstance(msg.recognizer, _FakeRecognizer)
    assert msg.recognizer.num_threads == 2
    assert msg.recognizer.language == "zh"
    assert msg.recognizer.disfluency == "smooth"
    assert seen == {"kind": "sensevoice", "proxy": ""}


def test_on_model_loaded_applies_and_discards_stale() -> None:
    app = _bare_app()
    app._model_generation = 5
    app._model_load_purpose = "swap"
    recognizer = SimpleNamespace()

    # A stale generation (an older load finishing after a newer one was asked
    # for) must not clobber the current recognizer.
    app._on_model_loaded(recognizer, generation=4)
    assert app._recognizer is None

    app._on_model_loaded(recognizer, generation=5)
    assert app._recognizer is recognizer
    assert app._model_ready is True


def test_on_model_loaded_failure_keeps_disarmed() -> None:
    app = _bare_app()
    app._model_generation = 1
    app._model_load_purpose = "swap"

    app._on_model_loaded(None, generation=1)

    # A failed load must leave the app disarmed (CapsLock stays native) so a
    # retry can succeed; re-arming now would swallow the hotkey with no model.
    assert app._recognizer is None
    assert app._model_ready is False


def test_toggle_translate_loads_off_worker() -> None:
    app = _bare_app()
    built: list[object] = []
    app._controls = SimpleNamespace(
        toggle_translate=lambda: True,
        build_translator=lambda **kwargs: built.append(kwargs) or SimpleNamespace(),
    )
    started: list[object] = []
    app._start_translator_load = lambda route=None: started.append(route)

    app._toggle_translate()

    assert started == [None]  # handed to a background loader
    assert built == []  # build_translator was NOT called on the worker thread
