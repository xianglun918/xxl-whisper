# ─── How to run ───
# uv run python -X utf8 scripts/probe_mac_paste.py
# Verifies NSPasteboard round-trip + CGEventPost Cmd+V lands in TextEdit —
# the production path (dictation pastes into OTHER apps). Findings so far:
# a Tk Entry target swallows synthetic Cmd+V even with frontmost+key+focus
# all confirmed (Tk aqua quirk — do not retry); osascript 'make new document'
# hangs on the Automation TCC prompt — so this probe uses `open` (LaunchServices,
# no TCC) + AX readback (already-granted Accessibility, reads live AXValue).
# Tries three post configurations (event source state × post tap point) while
# a listen-only diagnostic tap records what entered the session. Clipboard is
# snapshot and restored no matter what.

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import AppKit
import ApplicationServices
import Quartz

_SENTINEL = "PASTE_SENTINEL_123"
_PB_TYPE = AppKit.NSPasteboardTypeString
_KEY_V = 0x09
_KEY_RCMD = 0x36
_AX_MAX_DEPTH = 6
_TE_BUNDLE = "com.apple.TextEdit"


def _snapshot_pasteboard() -> list[tuple[str, bytes]]:
    pb = AppKit.NSPasteboard.generalPasteboard()
    snap = []
    for ptype in pb.types() or []:
        data = pb.dataForType_(ptype)
        if data is not None:
            snap.append((ptype, bytes(data)))
    return snap


def _restore_pasteboard(snap: list[tuple[str, bytes]]) -> None:
    pb = AppKit.NSPasteboard.generalPasteboard()
    pb.clearContents()
    for ptype, data in snap:
        pb.setData_forType_(AppKit.NSData.dataWithBytes_length_(data, len(data)), ptype)


def _set_string(text: str) -> None:
    pb = AppKit.NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, _PB_TYPE)


def _read_string() -> str:
    pb = AppKit.NSPasteboard.generalPasteboard()
    value = pb.stringForType_(_PB_TYPE)
    return str(value) if value is not None else ""


def _post_cmd_v(source_state: int, tap_point: int) -> None:
    src = Quartz.CGEventSourceCreate(source_state)
    cmd = Quartz.kCGEventFlagMaskCommand

    ev = Quartz.CGEventCreateKeyboardEvent(src, _KEY_RCMD, True)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, cmd)
    Quartz.CGEventPost(tap_point, ev)
    time.sleep(0.02)

    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, _KEY_V, down)
        Quartz.CGEventSetFlags(ev, cmd)
        Quartz.CGEventPost(tap_point, ev)
        time.sleep(0.02)

    ev = Quartz.CGEventCreateKeyboardEvent(src, _KEY_RCMD, False)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, 0)
    Quartz.CGEventPost(tap_point, ev)


class _TapRecorder:
    """Listen-only tap logging keyDown/keyUp/flagsChanged for diagnosis."""

    def __init__(self) -> None:
        self.enabled = False
        self.events: list[tuple[int, int, int]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            (1 << Quartz.kCGEventKeyDown)
            | (1 << Quartz.kCGEventKeyUp)
            | (1 << Quartz.kCGEventFlagsChanged),
            self._cb,
            None,
        )
        if not self._tap:
            return
        self.enabled = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _cb(self, _proxy: object, typ: int, event: object, _refcon: object) -> object:
        with self._lock:
            self.events.append(
                (
                    typ,
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode),
                    Quartz.CGEventGetFlags(event),
                )
            )
        return event

    def _loop(self) -> None:
        run_loop = Quartz.CFRunLoopGetCurrent()
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(run_loop, source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(self._tap, True)
        while not self._stop.is_set():
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.05, False)

    def mark(self) -> int:
        with self._lock:
            return len(self.events)

    def drain(self, since: int) -> list[tuple[int, int, int]]:
        with self._lock:
            return self.events[since:]

    def stop(self) -> None:
        self._stop.set()


def _ax_get(element: object, attribute: str) -> object:
    result = ApplicationServices.AXUIElementCopyAttributeValue(element, attribute, None)
    if isinstance(result, tuple) and result and isinstance(result[0], int):
        return result[1] if result[0] == 0 else None
    return result


def _find_text_area(element: object, depth: int = 0) -> object | None:
    if depth > _AX_MAX_DEPTH or element is None:
        return None
    if _ax_get(element, "AXRole") == "AXTextArea":
        return element
    for child in _ax_get(element, "AXChildren") or []:
        area = _find_text_area(child, depth + 1)
        if area is not None:
            return area
    return None


def _wait_text_area(app_element: object, timeout: float) -> object | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        window = _ax_get(app_element, "AXFocusedWindow")
        if window is not None:
            area = _find_text_area(window)
            if area is not None:
                return area
        time.sleep(0.2)
    return None


def _wait_textedit_pid(timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
            if app.bundleIdentifier() == _TE_BUNDLE:
                return app.processIdentifier()
        time.sleep(0.2)
    return None


def _activate_textedit(pid: int) -> None:
    for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
        if app.processIdentifier() == pid:
            app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
            return


def _wait_frontmost(timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        front = _frontmost_bundle()
        if front == _TE_BUNDLE:
            return front
        time.sleep(0.2)
    return _frontmost_bundle()


def _frontmost_bundle() -> str:
    front = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(front.bundleIdentifier()) if front is not None else ""


def _setup_target(doc_path: Path) -> object | None:
    """Open TextEdit on an empty probe doc; returns its AXTextArea or None."""
    doc_path.write_text("", encoding="utf-8")
    opened = subprocess.run(  # noqa: S603 — fixed /usr/bin/open, repo probe
        ["/usr/bin/open", "-a", "TextEdit", str(doc_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    print(f"open TextEdit doc rc={opened.returncode} path={doc_path}")
    if opened.returncode != 0:
        print("VERDICT: BLOCKED — TextEdit failed to open; re-run")
        return None
    pid = _wait_textedit_pid(10.0)
    print(f"TextEdit pid: {pid}")
    if pid is None:
        print("VERDICT: BLOCKED — TextEdit never showed up; re-run")
        return None
    _activate_textedit(pid)
    front = _wait_frontmost(5.0)
    if front != _TE_BUNDLE:
        print(f"frontmost bundle: {front!r} (want {_TE_BUNDLE!r})")
        if front == "com.apple.loginwindow":
            print("VERDICT: BLOCKED — screen is locked (login window frontmost); "
                  "unlock and re-run")
        else:
            print("VERDICT: BLOCKED — TextEdit is not frontmost; click its window and re-run")
        return None
    app_element = ApplicationServices.AXUIElementCreateApplication(pid)
    text_area = _wait_text_area(app_element, 10.0)
    print(f"text area reachable: {text_area is not None}")
    if text_area is None:
        print("VERDICT: BLOCKED — no AXTextArea in TextEdit's focused window; re-run")
        return None
    print(f"baseline doc: {_ax_get(text_area, 'AXValue')!r}")
    return text_area


def main() -> int:
    snap = _snapshot_pasteboard()
    preflight = bool(Quartz.CGPreflightPostEventAccess())
    print(f"Accessibility (post) preflight: {preflight}")
    if not preflight:
        print("VERDICT: BLOCKED — grant Accessibility (辅助功能) to the host terminal then re-run")
        return 0

    recorder = _TapRecorder()
    print(f"diagnostic tap enabled: {recorder.enabled}")

    doc_path = Path(tempfile.gettempdir()) / "xxl_whisper_paste_probe.txt"
    text_area = _setup_target(doc_path)
    if text_area is None:
        return 0

    attempts = [
        ("combined-source -> session-tap",
         Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGSessionEventTap),
        ("hid-source -> hid-tap",
         Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGHIDEventTap),
        ("hid-source -> session-tap",
         Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGSessionEventTap),
    ]
    winning = None
    try:
        for idx, (label, state, point) in enumerate(attempts):
            marker = f"{_SENTINEL}_{idx}"
            _set_string(marker)
            mark = recorder.mark()
            _post_cmd_v(state, point)
            time.sleep(0.8)
            doc = _ax_get(text_area, "AXValue") or ""
            observed = recorder.drain(mark)
            landed = marker in doc
            print(f"[{label}] pasteboard={_read_string()!r} landed={landed} "
                  f"tap_saw={len(observed)}")
            for typ, keycode, flags in observed:
                print(f"    tap type={typ} keycode=0x{keycode:02x} flags=0x{flags:x}")
            if landed and winning is None:
                winning = (label, state, point)

        if winning is not None:
            label, state, point = winning
            print(f"winning post config: {label}")
            _set_string(_SENTINEL)
            _post_cmd_v(state, point)
            time.sleep(0.3)
            _restore_pasteboard(snap)
            print("restore-timing demo: restored after 300 ms")
            print("VERDICT: OK")
            return 0

        print("VERDICT: FAIL — no post configuration landed Cmd+V in TextEdit; "
              "see tap evidence above")
        return 1
    finally:
        _restore_pasteboard(snap)
        recorder.stop()


if __name__ == "__main__":
    sys.exit(main())
