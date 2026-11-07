"""Delivery: one sentence, translated once, then subtitled and/or typed.

The two entry paths (push-to-talk, continuous dictation) used to carry their own
copy of this decision, so the rules drifted: a hand-rolled translation cache, and
"keep the pill quiet while the bar is on" repeated at five call sites. These
tests pin the unified behaviour at the module's own interface.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from app.delivery import COMMIT_TEXT, DeliveryPorts, UtteranceDelivery
from app.type_batcher import TypeBatcher

from app import delivery as delivery_module


class _Indicator:
    def __init__(self) -> None:
        self.flashes: list[str] = []
        self.listen_flashes: list[tuple[str, str]] = []

    def flash(self, text: str, _ms: int) -> None:
        self.flashes.append(text)

    def flash_listen(self, flash_text: str, listen_text: str, _ms: int) -> None:
        self.listen_flashes.append((flash_text, listen_text))


@dataclass(slots=True)
class _Host:
    """The app side of the ports, plus what the tests assert on."""

    subtitles: bool = True
    overlay_only: bool = False
    translate_mode: str = "replace"
    #: False models "translation is off": the call happens, it just does nothing.
    translate_runs: bool = True
    #: When False the translator "ran" but produced nothing (model error path).
    translate_succeeds: bool = True
    translations: list[str] = field(default_factory=list)
    presented: list[tuple[str, str]] = field(default_factory=list)
    no_translator_notes: int = 0
    indicator: _Indicator = field(default_factory=_Indicator)

    def translate(self, text: str) -> tuple[str, bool]:
        self.translations.append(text)
        if not self.translate_runs:
            return "", False
        if not self.translate_succeeds:
            return "", True
        return f"T({text})", True

    def present(self, text: str, translated: str) -> None:
        self.presented.append((text, translated))

    def note(self) -> None:
        self.no_translator_notes += 1

    def ports(self) -> DeliveryPorts:
        return DeliveryPorts(
            config=lambda: SimpleNamespace(
                translate=True,
                translate_mode=self.translate_mode,
                restore_clipboard=False,
                paste_delay_ms=0,
                output_file="",
            ),
            subtitles=lambda: self.subtitles,
            overlay_only=lambda: self.overlay_only,
            present=self.present,
            translate=self.translate,
            on_no_translator=self.note,
            indicator=self.indicator,
        )


class _PasteFailureError(RuntimeError):
    """Injected paste failure (keep the message out of the raise site)."""


class _Pastes:
    """Records what reached the cursor, and can fail on demand."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.texts: list[str] = []
        self._fail_on = fail_on

    def __call__(self, text: str, _settings: Any, _indicator: Any) -> None:
        if text == self._fail_on:
            raise _PasteFailureError
        self.texts.append(text)


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _running(
    host: _Host, monkeypatch, pastes: _Pastes, **kwargs: Any
) -> tuple[UtteranceDelivery, threading.Event]:
    monkeypatch.setattr(delivery_module, "emit_text", pastes)
    stop = threading.Event()
    item = UtteranceDelivery(ports=host.ports(), **kwargs)
    item.start(stop)
    return item, stop


def test_subtitle_line_and_typing_share_one_translation(monkeypatch) -> None:
    """The sentence is translated once and the cached result is typed."""
    host = _Host()
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.deliver_now("你好")
    finally:
        stop.set()
        item.stop()

    assert host.translations == ["你好"]  # one MT pass, not two
    assert host.presented == [("你好", "T(你好)")]
    assert pastes.texts == ["T(你好)"]


def test_merged_batch_is_translated_on_its_own(monkeypatch) -> None:
    """A force-cut piece waits; the merged sentence is not served the old cache."""
    host = _Host()
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.submit("a", forced=True)
        item.submit("b", forced=True)
        assert _wait_for(lambda: len(host.translations) == 2)
        assert pastes.texts == []  # still batched, nothing typed yet
        item.submit("c", forced=False)
        assert _wait_for(lambda: pastes.texts == ["T(a b c)"])
    finally:
        stop.set()
        item.stop()

    # three subtitle translations + one for the merged sentence
    assert host.translations == ["a", "b", "c", "a b c"]


def test_typing_only_does_not_touch_the_bar(monkeypatch) -> None:
    """With no subtitle bar there is nothing to present and only one MT pass."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.deliver_now("hello")
    finally:
        stop.set()
        item.stop()

    assert host.presented == []
    assert host.translations == ["hello"]  # for the typing path
    assert pastes.texts == ["T(hello)"]
    assert host.no_translator_notes == 0  # never asked: no bar to explain


def test_overlay_only_flashes_instead_of_pasting(monkeypatch) -> None:
    """Subtitle-only mode must not inject anything, and must say it landed."""
    host = _Host(overlay_only=True)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.deliver_now("你好")
    finally:
        stop.set()
        item.stop()

    assert pastes.texts == []  # nothing typed, by the user's choice
    assert host.indicator.flashes == [COMMIT_TEXT]
    assert host.presented == [("你好", "T(你好)")]


def test_paste_failure_does_not_stop_the_next_sentence(monkeypatch, caplog) -> None:
    """One exploding paste is logged; the next sentence still lands."""
    host = _Host(subtitles=False)
    pastes = _Pastes(fail_on="T(boom)")
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.submit("boom")
        item.submit("fine")
        assert _wait_for(lambda: pastes.texts == ["T(fine)"])
    finally:
        stop.set()
        item.stop()

    assert "typing failed" in caplog.text


def test_flush_item_types_the_held_batch(monkeypatch) -> None:
    """submit_flush is the pause/exit path: held pieces must not be lost."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.submit("half a sentence", forced=True)
        # Consumed by the emitter and sitting in the batcher: no translation is
        # owed yet (no subtitle bar), and nothing may be typed yet.
        assert _wait_for(lambda: item.pending == 0)
        assert pastes.texts == []
        assert host.translations == []
        item.submit_flush()
        assert _wait_for(lambda: pastes.texts == ["T(half a sentence)"])
    finally:
        stop.set()
        item.stop()


def test_batch_window_expiry_types_without_a_new_sentence(monkeypatch) -> None:
    """A held piece is typed when the merge window closes, not only on flush."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    batcher = TypeBatcher(window_s=0.0, max_chars=10_000)
    item, stop = _running(host, monkeypatch, pastes, batcher=batcher)
    try:
        item.submit("still talking", forced=True)
        assert _wait_for(lambda: pastes.texts == ["T(still talking)"])
    finally:
        stop.set()
        item.stop()


def test_endpoint_latency_log_line_is_unchanged(monkeypatch, caplog) -> None:
    """The endpoint→shown line is what a "still slow" report is triaged from."""
    caplog.set_level(logging.INFO, logger="app.delivery")
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.submit("一句话", endpoint_ts=time.monotonic() - 0.5, forced=True)
        assert _wait_for(lambda: "endpoint→shown" in caplog.text)
    finally:
        stop.set()
        item.stop()

    assert "forced=True" in caplog.text


def test_translation_failure_falls_back_to_the_source_text(monkeypatch) -> None:
    """Losing the translation is an annoyance; losing the words is not allowed."""
    host = _Host(subtitles=False, translate_succeeds=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.deliver_now("原文")
    finally:
        stop.set()
        item.stop()

    assert pastes.texts == ["原文"]
    assert host.indicator.flashes == ["翻译失败，已上屏原文"]


def test_push_to_talk_ignores_a_leftover_force_cut_piece(monkeypatch) -> None:
    """A held key is one utterance: it must not be glued onto earlier pieces."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.submit("前半句", forced=True)  # a force-cut piece from continuous mode
        assert _wait_for(lambda: item.pending == 0)  # consumed, now batched
        assert pastes.texts == []  # still inside the batcher
        item.deliver_now("按住说话")
        assert pastes.texts == ["T(按住说话)"]  # not "前半句按住说话"
    finally:
        stop.set()
        item.stop()


def test_push_to_talk_paste_failure_reaches_the_caller(monkeypatch) -> None:
    """Push-to-talk runs on the worker thread: the error must not be swallowed.

    The worker is what flashes "出错了，详见日志" and raises the tray notice; if
    delivery ate the exception the user would speak, see nothing in the document
    and get no feedback at all.
    """
    host = _Host(subtitles=False)
    pastes = _Pastes(fail_on="T(boom)")
    item, stop = _running(host, monkeypatch, pastes)
    try:
        with pytest.raises(_PasteFailureError):
            item.deliver_now("boom")
    finally:
        stop.set()
        item.stop()


def test_stop_after_shutdown_queues_no_phantom_sentence(monkeypatch) -> None:
    """The shutdown log counts sentences, so a control item must not inflate it."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    stop.set()
    # Wait for the emitter to actually exit first: otherwise it could still
    # drain the very item this test is about, and both behaviours would pass.
    assert _wait_for(lambda: item._thread is not None and not item._thread.is_alive())
    item.stop()

    assert item.pending == 0


def test_announce_delivered_returns_to_the_listening_pill(monkeypatch) -> None:
    """Continuous typing-only mode flashes ✓ and falls back to 「聆听中」."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.announce_delivered()
    finally:
        stop.set()
        item.stop()

    assert host.indicator.listen_flashes == [("✓ 已上屏", "● 聆听中")]


def test_push_to_talk_with_bar_on_and_translation_off_translates_once(monkeypatch) -> None:
    """One decision, one call: a no-op translation must not be asked twice."""
    host = _Host(subtitles=True, translate_runs=False)
    pastes = _Pastes()
    item, stop = _running(host, monkeypatch, pastes)
    try:
        item.deliver_now("你好")
    finally:
        stop.set()
        item.stop()

    assert host.translations == ["你好"]  # asked once, not twice
    assert pastes.texts == ["你好"]  # falls back to the source text
    assert host.indicator.flashes == []  # attempted=False: no failure flash


def test_start_is_idempotent(monkeypatch) -> None:
    """Two starts must not leave two emitters draining one queue."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    monkeypatch.setattr(delivery_module, "emit_text", pastes)
    stop = threading.Event()
    item = UtteranceDelivery(ports=host.ports())
    item.start(stop)
    first = item._thread
    item.start(stop)
    try:
        assert item._thread is first
    finally:
        stop.set()
        item.stop()


def test_dropped_counts_the_oldest_sacrificed_sentence(monkeypatch) -> None:
    """A stale sentence is dropped rather than shown late — and counted."""
    host = _Host(subtitles=False)
    pastes = _Pastes()
    monkeypatch.setattr(delivery_module, "emit_text", pastes)
    item = UtteranceDelivery(ports=host.ports(), queue_max=1)
    item.submit("oldest")
    item.submit("newer")
    item.submit("newest")

    assert item.pending == 1
    assert item.dropped >= 1  # the oldest went, so "newest" is what survives
