"""Deferred, exception-proof menu callbacks (pure standard library).

Two hard-won rules live here, both about pystray's behaviour on Windows:

1. **Never rebuild the menu inside a menu action.** pystray invokes the action
   while it is still unwinding the native item; replacing the menu at that moment
   frees the item being invoked and wedges the menu thread — after which *no*
   click has any effect. The rebuild is therefore deferred a few dozen
   milliseconds.
2. **Never let an action raise into the platform menu loop.** A raised exception
   is silent from the user's point of view (the click simply does nothing), so
   actions are wrapped and failures are logged.

Keeping this out of :mod:`app.tray` means it can be unit-tested without pystray,
PIL or a display.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

#: Delay before a menu rebuild. Long enough for the click handler to return and
#: the native popup to close.
DEFAULT_REFRESH_DELAY_S: float = 0.12


def guarded(
    action: Callable[[], None] | None,
    on_error: Callable[[str], None],
) -> Callable[..., None] | None:
    """Return *action* wrapped so extra arguments and exceptions are harmless.

    pystray may call an action with ``(icon, item)``, ``(icon)`` or ``()``
    depending on backend and version, so the wrapper accepts any positional
    arguments and calls the action with none.
    """
    if action is None:
        return None

    def wrapper(*_args: object, **_kwargs: object) -> None:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - a menu click must never crash
            on_error(f"{type(exc).__name__}: {exc}")

    return wrapper


def schedule_refresh(
    refresh: Callable[[], None],
    on_error: Callable[[str], None],
    *,
    delay: float = DEFAULT_REFRESH_DELAY_S,
    timer_factory: Callable[..., threading.Timer] = threading.Timer,
) -> threading.Timer:
    """Run *refresh* shortly after the current menu action returns."""

    def run() -> None:
        try:
            refresh()
        except Exception as exc:  # noqa: BLE001 - refresh is best-effort
            on_error(f"refresh failed: {type(exc).__name__}: {exc}")

    timer = timer_factory(delay, run)
    timer.daemon = True
    timer.start()
    return timer
