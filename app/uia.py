"""UI Automation insertion: the hook-proof channel for modern apps.

uiautomation wraps the UIAutomationCore COM API. Two constraints shape the
design:

- TextPattern is read-only in UIA; the only write surface is ValuePattern,
  which SETS the whole value. We therefore append the utterance to the
  current value (dictation into a chat box is almost always append-at-end).
- COM must be initialized on the calling thread — we lazily import and
  initialize inside the worker thread, never at app import time.
"""

import logging
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

_thread_local = threading.local()


class UiaUnavailableError(Exception):
    """Raised when the focused control cannot be written through UIA."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Control(Protocol):
    """The slice of uiautomation's Control we consume."""

    def Exists(self, search_interval: float, search_timeout: float) -> bool: ...
    def GetPattern(self, pattern_id: int) -> object: ...
    def DocumentControl(self) -> "_Control": ...


@runtime_checkable
class _ValuePattern(Protocol):
    """The slice of uiautomation's ValuePattern we consume."""

    IsReadOnly: bool
    Value: str

    def SetValue(self, value: str) -> None: ...


@dataclass(frozen=True, slots=True)
class UiaTarget:
    """A focused, writable-by-ValuePattern control."""

    writable: bool


def probe_focused() -> UiaTarget:
    """Answer whether the focused control accepts UIA value writes.

    Never raises: any COM/UIA failure maps to writable=False.
    """
    auto = _auto()
    if auto is None:
        return UiaTarget(writable=False)
    try:
        control = auto.GetFocusedControl()
        if control is None:
            return UiaTarget(writable=False)
        return UiaTarget(writable=_is_writable(control, auto))
    except OSError as exc:  # COM/element-gone races are routine
        log.info("uia probe failed: %s", exc)
        return UiaTarget(writable=False)


def append_text(text: str) -> None:
    """Append ``text`` to the focused control's value, or raise."""
    auto = _auto()
    if auto is None:
        raise UiaUnavailableError(reason="uiautomation import failed")
    try:
        control = auto.GetFocusedControl()
        if control is None:
            raise UiaUnavailableError(reason="no focused control")
        _write(control, auto, text)
    except OSError as exc:
        raise UiaUnavailableError(reason=str(exc)) from exc


def _is_writable(control: _Control, auto: ModuleType) -> bool:
    """Check ValuePattern writability on the control or its DocumentControl."""
    for target in (control, control.DocumentControl()):
        if not target.Exists(0, 0):
            continue
        vp = target.GetPattern(auto.PatternId.ValuePattern)
        if vp is not None and isinstance(vp, _ValuePattern) and not vp.IsReadOnly:
            return True
    return False


def _write(control: _Control, auto: ModuleType, text: str) -> None:
    """Append text via ValuePattern on the control or its DocumentControl."""
    for target in (control, control.DocumentControl()):
        if not target.Exists(0, 0):
            continue
        vp = target.GetPattern(auto.PatternId.ValuePattern)
        if vp is not None and isinstance(vp, _ValuePattern) and not vp.IsReadOnly:
            current = vp.Value or ""
            vp.SetValue(current + text)
            return
    raise UiaUnavailableError(reason="focused control is not value-writable")


def _auto() -> ModuleType | None:
    """Import uiautomation once per thread and init COM on this thread."""
    module: ModuleType | None = getattr(_thread_local, "auto", None)
    if module is None:
        try:
            import uiautomation as auto  # noqa: PLC0415 — COM init must happen per-thread

            auto.UIAutomationInitializerInThread()
        except ImportError as exc:
            log.warning("uiautomation unavailable: %s", exc)
            return None
        module = auto
        _thread_local.auto = module
    return module
