"""One finished sentence, one delivery: translate once, then subtitle and/or type.

The same four steps used to exist twice, once per entry path:

* push-to-talk ran ``_emit`` (translate → subtitle → paste), one call site, and
  the ``Channel`` it returned was thrown away;
* continuous dictation ran ``_serve`` + ``_type_batch`` (translate → subtitle →
  :class:`~app.type_batcher.TypeBatcher` → paste), carrying its own hand-rolled
  translation cache.

Two implementations of one decision meant the rules drifted: "no sticky pill
while the bar is on" was repeated at five call sites, and the translation cache
was a tuple assembled inline. This module owns the decision once. Callers only
choose *when* a sentence is ready — :meth:`UtteranceDelivery.deliver_now` for a
whole push-to-talk utterance, :meth:`UtteranceDelivery.submit` for pieces of a
live transcript — and everything after that (translate once, reuse it for
typing, merge force-cut pieces, compose the payload, keep the pill quiet) is
implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app import logsafe
from app.emit import EmitSettings, emit_text
from app.emit_queue import EmitQueue, PendingEmit
from app.translator import compose_text
from app.type_batcher import TypeBatcher

if TYPE_CHECKING:
    from app.config import Config

log = logging.getLogger(__name__)

#: Confirmation flashed when the sentence went to the subtitle bar only.
COMMIT_TEXT: str = "✓ 已上屏"
COMMIT_FLASH_MS: int = 900

#: Sticky pill shown while continuous dictation is listening (the flash above
#: returns to it). Both live here so the wording has exactly one home.
LISTEN_TEXT: str = "● 聆听中"

#: Bounded queue between ASR and this module: deep enough to absorb a burst,
#: shallow enough that a stale subtitle is dropped rather than shown late.
QUEUE_MAX: int = 64

#: How long the emitter blocks on the queue before checking the typing batch.
_POLL_S: float = 0.2

#: How long :meth:`UtteranceDelivery.stop` waits for the emitter to unwind.
_STOP_TIMEOUT_S: float = 2.0


@dataclass(frozen=True, slots=True)
class DeliveryPorts:
    """What delivery needs from its host.

    ``config`` is a provider, not a value: the app replaces its frozen
    :class:`~app.config.Config` on every switch, and delivery must see the new
    one without being rebuilt.
    """

    #: The current settings (routing mode, clipboard policy, paste delay).
    config: Callable[[], Config]
    #: True when the subtitle bar is one of the outputs.
    subtitles: Callable[[], bool]
    #: True when the bar is the *only* output (nothing is pasted).
    overlay_only: Callable[[], bool]
    #: Push one finished sentence onto the bar.
    present: Callable[[str, str], None]
    #: Translate one sentence; ``(translation, attempted)``.
    translate: Callable[[str], tuple[str, bool]]
    #: Say why a subtitle line is the source text (host owns the wording).
    on_no_translator: Callable[[], None]
    #: The platform pill facade (``flash``).
    indicator: Any


class UtteranceDelivery:
    """Translate once per sentence, then subtitle it and/or paste it."""

    def __init__(
        self,
        *,
        ports: DeliveryPorts,
        batcher: TypeBatcher | None = None,
        queue_max: int = QUEUE_MAX,
    ) -> None:
        self._ports = ports
        self._queue = EmitQueue(maxsize=queue_max)
        self._batcher = batcher if batcher is not None else TypeBatcher()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._file_lock = threading.Lock()

    # -- interface -----------------------------------------------------------

    def start(self, stop_event: threading.Event) -> None:
        """Start the emitter thread that drains :meth:`submit`."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = stop_event
        self._thread = threading.Thread(target=self._loop, daemon=True, name="emit-worker")
        self._thread.start()

    def stop(self) -> None:
        """Type whatever is still batched, then let the emitter thread exit.

        The host sets its shutdown flag first; this only asks for the tail to be
        flushed so a half-sentence never dies with the process.
        """
        # Once the shutdown flag is set, the loop's own tail flush types the
        # batch; queueing a control item into a queue nobody drains again would
        # only make the shutdown log claim a sentence is still pending.
        if self._stop_event is None or not self._stop_event.is_set():
            self.submit_flush()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=_STOP_TIMEOUT_S)

    def submit(
        self, text: str, *, endpoint_ts: float | None = None, forced: bool = False
    ) -> bool:
        """Queue one finished ASR piece; False when an older one had to go."""
        return self._queue.submit(text, endpoint_ts=endpoint_ts, forced=forced)

    def submit_flush(self) -> None:
        """Ask the emitter to type whatever the batcher is holding.

        Safe from any thread, and never dropped: a lost flush is lost words.
        """
        self._queue.submit_flush()

    def _write_file(self, text: str) -> None:
        """Append one transcribed sentence to the configured output file.

        Best-effort and thread-safe: an unwritable path must never take
        dictation down, and both the push-to-talk (worker) and continuous
        (emitter) threads call this.
        """
        path = self._ports.config().output_file
        if not path:
            return
        try:
            with self._file_lock, Path(path).open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            log.warning("output file write failed: %s", path)

    def deliver_now(self, text: str) -> None:
        """Deliver one whole sentence on the calling thread (push-to-talk).

        Deliberately bypasses the batcher: a held key is one utterance, so a
        force-cut piece left over from an earlier continuous session must not be
        glued onto it. Failures propagate on purpose — this runs on the worker
        thread, whose handler is what tells the user "出错了，详见日志".
        """
        self._write_file(text)
        translated, attempted = self._subtitle_or_translate(text)
        if self._ports.overlay_only():
            self._ports.indicator.flash(COMMIT_TEXT, COMMIT_FLASH_MS)
            return
        # When the bar is on, the decision above already ran the translator (or
        # decided there was nothing to translate); reuse that answer instead of
        # asking the model a second time.
        cached = (translated, attempted) if self._ports.subtitles() else None
        self._type_text(text, cached=cached, raise_on_error=True)

    def announce_delivered(self) -> None:
        """Continuous, typing only: flash ✓, then fall back to the listen pill."""
        self._ports.indicator.flash_listen(COMMIT_TEXT, LISTEN_TEXT, COMMIT_FLASH_MS)

    def reconfigure(self, *, window_s: float, max_chars: int) -> None:
        """Rebuild the typing batcher for a new session's merge policy."""
        self._batcher = TypeBatcher(window_s=window_s, max_chars=max_chars)

    @property
    def pending(self) -> int:
        """How many sentences are waiting for translation."""
        return len(self._queue)

    @property
    def dropped(self) -> int:
        """How many sentences were sacrificed to keep the queue shallow."""
        return self._queue.dropped

    # -- emitter thread ------------------------------------------------------

    def _loop(self) -> None:
        """Daemon thread: subtitles per piece, typing per whole sentence.

        Runs MT and both outputs here so the decode thread never waits on them.
        The two channels deliberately keep different cadences: the subtitle bar
        is consumed while the speaker is still talking, so each piece goes up
        immediately; the cursor is a document, so force-cut pieces are glued back
        together by the batcher and pasted once.
        """
        while self._stop_event is not None and not self._stop_event.is_set():
            item = self._queue.get(timeout=_POLL_S)
            if item is None:
                if self._batcher.should_flush():
                    self._type_batch()
                continue
            if item.flush:
                self._type_batch()
                self._queue.task_done()
                continue
            try:
                self._deliver(item)
            except Exception:  # one bad sentence must not stop the loop
                log.exception("emit failed for %s", logsafe.describe_text(item.text))
            finally:
                self._queue.task_done()
        self._type_batch()  # never leave a half-sentence behind on shutdown

    def _deliver(self, item: PendingEmit) -> None:
        """Translate and deliver one finished ASR piece."""
        translated, attempted = self._subtitle_or_translate(item.text)
        if not self._ports.overlay_only():
            ready = self._batcher.add(item.text, forced=item.forced)
            if ready is not None:
                # Reuse the subtitle translation when the batcher emitted exactly
                # this sentence; a merged batch covers other text and must be
                # translated on its own.
                # Only reuse when a translation really ran: with subtitles off
                # ``translated`` is "" and reusing it would silently skip the
                # translation this sentence needs for typing.
                cached = (
                    (translated, attempted) if ready == item.text and attempted else None
                )
                self._type_text(ready, cached=cached)
        self._write_file(item.text)
        if item.endpoint_ts is not None:
            log.info(
                "continuous: endpoint→shown %.3f s (queued %.2f s, forced=%s)",
                time.monotonic() - item.endpoint_ts,
                item.enqueued_at - item.endpoint_ts,
                item.forced,
            )

    def _subtitle_or_translate(self, text: str) -> tuple[str, bool]:
        """Translate once and put the line on the bar when the bar is on.

        Returns ``("", False)`` when typing is the only output, which is what
        keeps the typing path from paying for a translation it does not need.
        """
        if not self._ports.subtitles():
            return "", False
        self._ports.on_no_translator()  # the host says it, and only if missing
        translated, attempted = self._ports.translate(text)
        self._ports.present(text, translated)
        return translated, attempted

    def _type_text(
        self,
        sentence: str,
        *,
        cached: tuple[str, bool] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        """Translate (if needed) and paste one whole sentence into the cursor.

        *cached* carries a translation already computed for this very sentence
        (the subtitle path does one), so the emitter does not run the NMT model
        twice per sentence. *raise_on_error* is for the push-to-talk path, which
        runs on the worker thread: there the exception is the only way the user
        learns that nothing was typed.
        """
        if not sentence:
            return
        translated, attempted = (
            cached if cached is not None else self._ports.translate(sentence)
        )
        try:
            emit_text(
                self._payload(sentence, translated, attempted=attempted),
                self._emit_settings(),
                self._ports.indicator,
            )
        except Exception:  # one failed paste must not lose the rest
            if raise_on_error:
                raise
            log.exception("typing failed for %s", logsafe.describe_text(sentence))

    def _type_batch(self) -> None:
        """Flush the batcher and paste the joined sentence, if any."""
        sentence = self._batcher.flush()
        if sentence:
            self._type_text(sentence)

    def _payload(self, text: str, translated: str, *, attempted: bool) -> str:
        """Return what should be pasted: source, translation, or both.

        Translation is best-effort by design: any model error falls back to the
        original text with a short pill message. Losing the translation is an
        annoyance; losing the user's words is not acceptable.
        """
        if translated.strip():
            return compose_text(text, translated, self._ports.config().translate_mode)
        if attempted and not self._ports.subtitles():
            self._ports.indicator.flash("翻译失败，已上屏原文", 1800)
        return text

    def _emit_settings(self) -> EmitSettings:
        config = self._ports.config()
        return EmitSettings(
            restore_clipboard=config.restore_clipboard,
            paste_delay_ms=config.paste_delay_ms,
        )
