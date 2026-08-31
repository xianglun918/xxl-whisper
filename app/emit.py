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
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, assert_never

from app import uia, winio

log = logging.getLogger(__name__)


class Channel(StrEnum):
    KEYS = "keys"
    WM_PASTE = "wm_paste"
    UIA = "uia"
    CLIPBOARD = "clipboard"


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

    Returns the channel that delivered, or CLIPBOARD when nothing could.
    """
    log.info("emit: target window %r", winio.foreground_window_title())
    winio.set_clipboard_text(text)  # always staged: manual Ctrl+V also works
    alive = winio.keyboard_injection_alive()
    control_class = winio.focused_control_class()
    probe = TargetProbe(
        injection_alive=alive,
        is_classic_control=is_classic_control(control_class),
        uia_writable=False if alive else uia.probe_focused().writable,
    )
    for channel in channels_in_order(probe):
        match channel:
            case Channel.KEYS:
                if _try_keys(text, settings, indicator):
                    return Channel.KEYS
            case Channel.WM_PASTE:
                if winio.post_wm_paste_to_focus():
                    log.info("emit: posted WM_PASTE (class=%r)", control_class)
                    indicator.flash("已粘贴（WM_PASTE）", 1200)
                    return Channel.WM_PASTE
            case Channel.UIA:
                if _try_uia(text, control_class, indicator):
                    return Channel.UIA
            case Channel.CLIPBOARD:
                pass  # terminal fallback handled after the loop
            case unreachable:
                assert_never(unreachable)
    log.warning("emit: no channel delivered (class=%r); text on clipboard", control_class)
    indicator.flash("已复制到剪贴板，请手动 Ctrl+V", 2500)
    return Channel.CLIPBOARD


def _try_keys(text: str, settings: EmitSettings, indicator: _IndicatorLike) -> bool:
    try:
        winio.paste_text(
            text,
            restore_clipboard=settings.restore_clipboard,
            delay_ms=settings.paste_delay_ms,
        )
    except (winio.PasteError, OSError) as exc:
        log.warning("emit: keys path failed: %s", exc)
        return False
    log.info("emit: delivered via injected Ctrl+V")
    indicator.hide()
    return True


def _try_uia(text: str, control_class: str, indicator: _IndicatorLike) -> bool:
    try:
        uia.append_text(text)
    except uia.UiaUnavailableError as exc:
        log.warning("emit: UIA failed: %s", exc)
        return False
    log.info("emit: appended via UIA (class=%r)", control_class)
    indicator.hide()
    return True
