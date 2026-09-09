# ─── How to run ───
# uv run python -X utf8 scripts/probe_mac_paste.py
# Verifies NSPasteboard round-trip + CGEventPost Cmd+V lands in our own Tk
# Entry. Clipboard is snapshot and restored no matter the outcome.

import sys
import time
import tkinter as tk

import AppKit
import Quartz

_SENTINEL = "PASTE_SENTINEL_123"
_PB_TYPE = AppKit.NSPasteboardTypeString


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


def _post_cmd_v() -> None:
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    cmd = Quartz.kCGEventFlagMaskCommand

    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x36, True)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, cmd)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)

    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 0x09, down)
        Quartz.CGEventSetFlags(ev, cmd)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)

    ev = Quartz.CGEventCreateKeyboardEvent(src, 0x36, False)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, 0)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)


def main() -> int:
    snap = _snapshot_pasteboard()
    preflight = bool(Quartz.CGPreflightPostEventAccess())
    print(f"Accessibility (post) preflight: {preflight}")

    root = tk.Tk()
    root.title("PASTEPROBE")
    entry = tk.Entry(root, width=40)
    entry.pack(padx=20, pady=20)
    root.geometry("400x80+300+300")
    root.attributes("-topmost", True)
    root.update()
    root.lift()
    root.focus_force()
    entry.focus_force()
    root.update()
    time.sleep(2.0)
    focused = root.focus_displayof() is not None
    print(f"focused: {focused} entry_has_focus={entry.focus_get() == entry}")

    try:
        _set_string(_SENTINEL)
        _post_cmd_v()
        time.sleep(1.0)
        got = entry.get()
        print(f"paste -> {got!r}")

        if got == _SENTINEL:
            # Restore-timing demo: set → paste → restore clipboard after 300 ms.
            _set_string(_SENTINEL)
            _post_cmd_v()
            time.sleep(0.3)
            _restore_pasteboard(snap)
            print("restore-timing demo: restored after 300 ms")
            print("VERDICT: OK")
            return 0

        print("VERDICT: BLOCKED — Cmd+V did not land in our own Entry; "
              "grant Accessibility (辅助功能) to the host terminal then re-run")
        return 0
    finally:
        _restore_pasteboard(snap)
        root.destroy()


if __name__ == "__main__":
    sys.exit(main())
