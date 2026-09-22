"""Emit channel selection + runtime delivery for transcribed text.

Channels in priority order:

1. KEYS      — clipboard + injected Ctrl+V (fast, universal, but swallowed on
               machines with hostile low-level keyboard hooks)
2. WM_PASTE  — posted to classic Win32 controls (hook-proof, Chromium-blind)
3. UIA       — accessibility ValuePattern append (modern apps, hook-proof)
4. CLIPBOARD — text stays on the clipboard; the user pastes manually

Each channel can still fail at runtime, so the caller attempts them in the
returned order and stops at the first success.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, assert_never

from app import native

log = logging.getLogger(__name__)

#: How long an injection probe's answer is trusted. ``keyboard_injection_alive``
#: presses F13 and sleeps 10 ms, so probing on every sentence of a burst is
#: wasted work; caching makes a burst probe once. The trade-off: a keyboard-hook
#: blocker that appears mid-burst is only noticed up to this TTL late — well
#: under the time a user would perceive, while a probe that already ran this
#: burst is not repeated.
_INJECTION_TTL_S: float = 4.0


class ProbeCache:
    """Short-TTL cache for an expensive boolean probe, keyed on the callable.

    Keying on the probe's identity means a monkeypatched probe (tests) is never
    served a stale answer, while production's stable module function is cached.
    The clock is injectable so the TTL is unit-tested without sleeping.
    """

    def __init__(self, ttl_s: float, clock: Callable[[], float]) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._probe: Callable[[], bool] | None = None
        self._value = False
        self._expires_at = 0.0

    def get(self, probe: Callable[[], bool]) -> bool:
        """Return the cached value, refreshing it once the TTL has elapsed."""
        now = self._clock()
        if probe is self._probe and now < self._expires_at:
            return self._value
        value = probe()
        self._probe = probe
        self._value = value
        self._expires_at = now + self._ttl_s
        return value


_INJECTION_PROBE = ProbeCache(_INJECTION_TTL_S, time.monotonic)


class Channel(StrEnum):
    KEYS = "keys"
    WM_PASTE = "wm_paste"
    UIA = "uia"
    CLIPBOARD = "clipboard"


class DeliveryPath(StrEnum):
    """How text is delivered when the KEYS channel is reachable."""

    CLIPBOARD_PASTE = "clipboard_paste"
    TYPE_TEXT = "type_text"


def choose_delivery_path(clipboard_has_non_text: bool) -> DeliveryPath:
    """Pure decision: a non-text clipboard must never be overwritten.

    The clipboard-paste path stages the text with ``set_clipboard_text`` and
    can only restore a previous *text* clipboard (``_clipboard_text`` reads
    CF_UNICODETEXT alone), so an image/file/rich clipboard would be destroyed.
    A non-text clipboard therefore takes the clipboard-free typing path.
    """
    return DeliveryPath.TYPE_TEXT if clipboard_has_non_text else DeliveryPath.CLIPBOARD_PASTE


@dataclass(frozen=True, slots=True)
class TargetProbe:
    """Facts about the current paste target, gathered at emit time."""

    injection_alive: bool
    is_classic_control: bool
    uia_writable: bool


def channels_in_order(probe: TargetProbe) -> tuple[Channel, ...]:
    """Ordered channel attempts; CLIPBOARD is always the terminal fallback."""
    channels: list[Channel] = []
    if probe.injection_alive:
        channels.append(Channel.KEYS)
    if probe.is_classic_control:
        channels.append(Channel.WM_PASTE)
    if probe.uia_writable:
        channels.append(Channel.UIA)
    channels.append(Channel.CLIPBOARD)
    return tuple(channels)


#: Classic Win32 control class prefixes that honor a posted WM_PASTE.
CLASSIC_CONTROL_PREFIXES: tuple[str, ...] = (
    "Edit",
    "RichEdit",
    "RICHEDIT",
    "Notepad",
    "ConsoleWindowClass",
    "CASCADIA_HOSTING_WINDOW_CLASS",  # Windows Terminal
)


def is_classic_control(class_name: str) -> bool:
    """Return True when the class name belongs to a classic Win32 control."""
    return any(class_name.startswith(prefix) for prefix in CLASSIC_CONTROL_PREFIXES)


class _IndicatorLike(Protocol):
    """The two indicator operations the emit path needs."""

    def hide(self) -> None: ...
    def flash(self, text: str, ms: int = 1500) -> None: ...


@dataclass(frozen=True, slots=True)
class EmitSettings:
    restore_clipboard: bool
    paste_delay_ms: int


def emit_text(text: str, settings: EmitSettings, indicator: _IndicatorLike) -> Channel:
    """Stage the text on the clipboard, then deliver via the best live channel.

    Returns the channel that delivered, or CLIPBOARD when nothing could. Every
    probe is recomputed per call (nothing is cached), so a focus change between
    sentences is picked up on the next emit; the INFO log records the exact
    foreground window, focused control and channel so a stuck delivery can be
    pinpointed from ``app.log``.
    """
    window = native.io.foreground_window_title()
    focus_hwnd = native.io.focused_control_hwnd()
    control_class = native.io.focused_control_class()
    # Cached for a short TTL: a burst of sentences probes once, while a genuinely
    # new blocker is still detected after the TTL (see ProbeCache).
    alive = _INJECTION_PROBE.get(native.io.keyboard_injection_alive)
    log.info(
        "emit: target window=%r focus_hwnd=0x%X class=%r injection_alive=%s",
        window,
        focus_hwnd,
        control_class,
        alive,
    )
    path = choose_delivery_path(native.io.clipboard_has_non_text())
    log.info("emit: delivery path=%s", path)
    if path is DeliveryPath.TYPE_TEXT:
        return _emit_clipboard_free(text, alive, indicator)
    _stage_clipboard(text)  # always staged: manual paste also works
    probe = TargetProbe(
        injection_alive=alive,
        is_classic_control=is_classic_control(control_class),
        uia_writable=False if alive else native.uia.probe_focused().writable,
    )
    for channel in channels_in_order(probe):
        match channel:
            case Channel.KEYS:
                if _try_keys(text, settings, indicator):
                    return Channel.KEYS
            case Channel.WM_PASTE:
                if native.io.post_wm_paste_to_focus():
                    log.info("emit: delivered via WM_PASTE (class=%r)", control_class)
                    indicator.flash("已粘贴（WM_PASTE）", 1200)
                    return Channel.WM_PASTE
                log.info("emit: WM_PASTE skipped (no focused control resolved)")
            case Channel.UIA:
                if _try_uia(text, control_class, indicator):
                    return Channel.UIA
            case Channel.CLIPBOARD:
                pass  # terminal fallback handled after the loop
            case unreachable:
                assert_never(unreachable)
    log.warning(
        "emit: no channel delivered (window=%r class=%r focus_hwnd=0x%X); text on clipboard",
        window,
        control_class,
        focus_hwnd,
    )
    indicator.flash(f"已复制到剪贴板，请手动 {native.io.PASTE_COMBO}", 2500)
    return Channel.CLIPBOARD


def _stage_clipboard(text: str) -> None:
    """Best-effort stage of the text for the paste channels."""
    try:
        native.io.set_clipboard_text(text)
    except (native.io.PasteError, OSError) as exc:
        log.warning("emit: clipboard staging failed (%s); channels may still deliver", exc)


def _emit_clipboard_free(text: str, injection_alive: bool, indicator: _IndicatorLike) -> Channel:
    """Type the text without ever touching the clipboard.

    Used when the clipboard holds non-text content: staging the text would
    destroy it and the text-only restore cannot put it back, so the only safe
    delivery is ``type_text``. When injection is dead the text cannot be typed
    either; the clipboard is left untouched and the terminal CLIPBOARD state is
    reported (the caller must not claim the text was copied).
    """
    if injection_alive and _try_type_text(text, indicator):
        return Channel.KEYS
    log.warning("emit: clipboard-free typing unavailable; clipboard left untouched")
    indicator.flash("无法输入：剪贴板含非文本内容，未改动", 2500)
    return Channel.CLIPBOARD


def _try_type_text(text: str, indicator: _IndicatorLike) -> bool:
    try:
        native.io.type_text(text)
    except (native.io.PasteError, OSError) as exc:
        log.warning("emit: clipboard-free typing failed: %s", exc)
        return False
    log.info("emit: delivered via clipboard-free unicode typing")
    indicator.hide()
    return True


def _try_keys(text: str, settings: EmitSettings, indicator: _IndicatorLike) -> bool:
    try:
        native.io.paste_text(
            text,
            restore_clipboard=settings.restore_clipboard,
            delay_ms=settings.paste_delay_ms,
        )
    except (native.io.PasteError, OSError) as exc:
        log.warning("emit: keys path failed: %s", exc)
        return False
    log.info("emit: delivered via injected %s", native.io.PASTE_COMBO)
    indicator.hide()
    return True


def _try_uia(text: str, control_class: str, indicator: _IndicatorLike) -> bool:
    try:
        native.uia.append_text(text)
    except native.uia.UiaUnavailableError as exc:
        log.warning("emit: UIA failed: %s", exc)
        return False
    log.info("emit: appended via UIA (class=%r)", control_class)
    indicator.hide()
    return True
