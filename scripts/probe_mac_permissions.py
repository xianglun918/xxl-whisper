# ─── How to run ───
# uv run python -X utf8 scripts/probe_mac_permissions.py
# Verifies the three macOS TCC grants xxl-whisper needs: Accessibility
# (CGEventPost injection), Input Monitoring (listen-only tap), Microphone
# (CoreAudio). Reports deep-link URLs and the host terminal the grant applies to.

import os
import subprocess
import sys
import threading
import time

import ApplicationServices
import Quartz
import sounddevice as sd

_ACCESS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
_LISTEN_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
_MIC_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"

_TERMINALS = {
    "iTerm2", "Terminal", "Warp", "kitty", "alacritty", "WezTerm",
    "ghostty", "Hyper", "iterm", "vscode", "Code", "Cursor", "Ghostty",
}


def _ps(pid: int, fmt: str) -> str:
    try:
        return subprocess.run(  # noqa: S603
            ["/bin/ps", "-o", f"{fmt}=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def _comm(pid: int) -> str:
    return _ps(pid, "comm").split("/")[-1]


def host_terminal() -> str:
    pid = os.getpid()
    chain: list[str] = []
    for _ in range(16):
        ppid = _ps(pid, "ppid")
        if not ppid or ppid == "0":
            break
        pid = int(ppid)
        name = _comm(pid)
        chain.append(name)
        if name in _TERMINALS:
            return name
    return chain[-1] if chain else "unknown"


def _check_accessibility() -> bool:
    return bool(ApplicationServices.AXIsProcessTrusted())


def _check_input_monitoring() -> tuple[bool, int]:
    """Preflight fast-path, then a listen-only tap that must deliver a synthetic event."""
    if not Quartz.CGPreflightListenEventAccess():
        return False, 0

    observed: list[int] = []
    lock = threading.Lock()

    def cb(_proxy: object, typ: int, event: object, _refcon: object) -> object:
        if typ == Quartz.kCGEventFlagsChanged:
            with lock:
                observed.append(
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                )
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        1 << Quartz.kCGEventFlagsChanged,
        cb,
        None,
    )
    if not tap:
        return False, 0

    stop = threading.Event()

    def _loop() -> None:
        rl = Quartz.CFRunLoopGetCurrent()
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(rl, src, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)
        while not stop.is_set():
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.05, False)

    threading.Thread(target=_loop, daemon=True).start()
    time.sleep(0.2)

    src_state = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    ev = Quartz.CGEventCreateKeyboardEvent(src_state, 0x36, True)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)
    time.sleep(2.0)

    stop.set()
    with lock:
        return True, len(observed)


def _check_microphone() -> str:
    """Open a 16 kHz mono InputStream; without the TCC grant the open blocks."""
    result: dict[str, str] = {}

    def _open() -> None:
        try:
            stream = sd.InputStream(samplerate=16000, channels=1, blocksize=800)
            stream.start()
            stream.stop()
            stream.close()
            result["ok"] = "open-ok"
        except Exception as exc:  # noqa: BLE001
            result["ok"] = f"open-error:{type(exc).__name__}"

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()
    thread.join(timeout=3.0)
    return result.get("ok", "open-timeout")


def main() -> int:
    print("== macOS permissions probe ==")

    access = _check_accessibility()
    print(f"Accessibility (AXIsProcessTrusted): {access}")

    granted, observed = _check_input_monitoring()
    print(f"Input Monitoring (listen tap): preflight={granted} observed_events={observed}")

    mic = _check_microphone()
    print(f"Microphone (16 kHz InputStream): {mic}")

    term = host_terminal()
    print(f"Host terminal (grants apply to): {term}")
    print(f"Accessibility deep-link: {_ACCESS_URL}")
    print(f"Input Monitoring deep-link: {_LISTEN_URL}")
    print(f"Microphone deep-link: {_MIC_URL}")

    missing = []
    if not access:
        missing.append("Accessibility (辅助功能)")
    if not granted or observed == 0:
        missing.append("Input Monitoring (输入监控)")
    if not mic.startswith("open-ok"):
        missing.append("Microphone (麦克风)")

    if not missing:
        print("VERDICT: OK")
        return 0
    print(
        "VERDICT: BLOCKED — grant missing permissions in System Settings "
        "for the host terminal, then re-run: "
        + "; ".join(missing)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
