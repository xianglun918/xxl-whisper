# ─── How to run ───
# uv run python -X utf8 scripts/probe_mac_tap.py
# Verifies a listen-only CGEventTap observes right-Cmd flagsChanged
# (down+up), the watchdog disable/enable cycle, and optionally a real 15s hold.

import sys
import threading
import time

import Quartz

_KEYCODE_RIGHT_CMD = 0x36
_STATE_HID = Quartz.kCGEventSourceStateHIDSystemState

_seen: list[tuple[int, int, int, float]] = []
_lock = threading.Lock()


def _cb(_proxy: object, typ: int, event: object, _refcon: object) -> object:
    if typ == Quartz.kCGEventFlagsChanged:
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        flags = Quartz.CGEventGetFlags(event)
        state = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceStateID)
        with _lock:
            _seen.append((keycode, flags, state, time.monotonic()))
    return event


def _post_flags_changed(keycode: int, flags: int) -> None:
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    ev = Quartz.CGEventCreateKeyboardEvent(src, keycode, True)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, flags)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)


def _snapshot() -> list[tuple[int, int, int, float]]:
    with _lock:
        return list(_seen)


def _create_tap() -> object | None:
    return Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        1 << Quartz.kCGEventFlagsChanged,
        _cb,
        None,
    )


def _start_loop(tap: object) -> tuple[threading.Thread, threading.Event, threading.Event]:
    stop = threading.Event()
    ready = threading.Event()

    def _loop() -> None:
        rl = Quartz.CFRunLoopGetCurrent()
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(rl, src, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)
        ready.set()
        while not stop.is_set():
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.05, False)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread, stop, ready


def _print_events(events: list[tuple[int, int, int, float]]) -> None:
    for kc, flags, state, ts in events:
        print(f"event keycode=0x{kc:02x} flags=0x{flags:x} state={state} t={ts:.3f}")


def _watchdog_demo(tap: object, observed: int) -> tuple[bool, bool]:
    Quartz.CGEventTapEnable(tap, False)
    _post_flags_changed(_KEYCODE_RIGHT_CMD, Quartz.kCGEventFlagMaskCommand)
    time.sleep(0.4)
    silenced = len(_snapshot()) == observed
    Quartz.CGEventTapEnable(tap, True)
    time.sleep(0.2)
    _post_flags_changed(_KEYCODE_RIGHT_CMD, Quartz.kCGEventFlagMaskCommand)
    time.sleep(0.5)
    resumed = len(_snapshot()) > observed
    return silenced, resumed


def _wait_real_hold(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with _lock:
            if any(kc == _KEYCODE_RIGHT_CMD and st == _STATE_HID for kc, _fl, st, _ts in _seen):
                return True
        time.sleep(0.1)
    return False


def main() -> int:
    preflight = bool(Quartz.CGPreflightListenEventAccess())
    print(f"Input Monitoring preflight: {preflight}")

    tap = _create_tap()
    print(f"tap created: {bool(tap)}")
    if not tap:
        print("VERDICT: BLOCKED — CGEventTapCreate failed; grant Input Monitoring then re-run")
        return 0

    thread, stop, ready = _start_loop(tap)
    ready.wait(timeout=2.0)
    time.sleep(0.2)

    _post_flags_changed(_KEYCODE_RIGHT_CMD, Quartz.kCGEventFlagMaskCommand)
    time.sleep(0.3)
    _post_flags_changed(_KEYCODE_RIGHT_CMD, 0)
    time.sleep(0.5)
    after_auto = _snapshot()
    _print_events(after_auto)
    observed = len(after_auto)

    silenced, resumed = _watchdog_demo(tap, observed)
    print(f"watchdog disable_silences={silenced} reenable_resumes={resumed}")

    if observed == 0:
        print("VERDICT: BLOCKED — tap created but no flagsChanged delivered; "
              "grant Input Monitoring (输入监控) to the host terminal then re-run")
        stop.set()
        thread.join(timeout=2.0)
        return 0

    down_ok = any(kc == _KEYCODE_RIGHT_CMD and (fl & Quartz.kCGEventFlagMaskCommand)
                  for kc, fl, _st, _ts in after_auto)
    up_ok = any(kc == _KEYCODE_RIGHT_CMD and not (fl & Quartz.kCGEventFlagMaskCommand)
                for kc, fl, _st, _ts in after_auto)
    print(f"auto down_ok={down_ok} up_ok={up_ok}")

    print("listening 15s for a REAL right-Cmd hold (hold it if you like)...")
    real_hold = _wait_real_hold(15.0)
    print(f"real_hold_observed={real_hold}")

    stop.set()
    thread.join(timeout=2.0)
    if down_ok and up_ok and silenced and resumed:
        print("VERDICT: OK")
        return 0
    print("VERDICT: FAIL — tap observed events but auto assertion failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
