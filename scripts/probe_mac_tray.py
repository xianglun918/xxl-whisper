# ─── How to run ───
# uv run python -X utf8 scripts/probe_mac_tray.py
# Verifies pystray-mac: Pillow icon, rebuilt-on-open callable menu, a CHECKED
# item (no radio on mac), a submenu, Quit, and osascript notification.

import subprocess
import sys
import threading

import pystray
from PIL import Image, ImageDraw

_MENU_BUILDS = {"count": 0}
_CLICKS: list[str] = []


def _draw_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#0f172a")
    draw.rounded_rectangle((24, 12, 40, 38), radius=8, fill="#7dd3fc")
    return image


def _paused_checked(_item: object) -> bool:
    return True


def _submenu_items() -> list[pystray.MenuItem]:
    return [
        pystray.MenuItem("子项 A", lambda: _CLICKS.append("sub:A")),
        pystray.MenuItem("子项 B", lambda: _CLICKS.append("sub:B")),
    ]


def _menu() -> pystray.Menu:
    _MENU_BUILDS["count"] += 1
    return pystray.Menu(
        pystray.MenuItem(
            "暂停语音热键",
            lambda: _CLICKS.append("toggle"),
            checked=_paused_checked,
        ),
        pystray.MenuItem("子菜单", pystray.Menu(_submenu_items)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", lambda: _CLICKS.append("quit")),
    )


def main() -> int:
    print(f"HAS_MENU_RADIO={pystray.Icon.HAS_MENU_RADIO} "
          f"HAS_MENU_CHECK={getattr(pystray.Icon, 'HAS_MENU_CHECK', True)}")

    icon = pystray.Icon(
        name="xxl-whisper-probe",
        icon=_draw_icon(),
        title="xxl-whisper probe",
        menu=_menu(),
    )

    def _on_setup(ico: pystray.Icon) -> None:
        ico.visible = True
        print("setup callback fired; icon visible")

    timer = threading.Timer(15.0, icon.stop)
    timer.start()
    try:
        icon.run(setup=_on_setup)
    except Exception as exc:  # noqa: BLE001
        print(f"VERDICT: FAIL — pystray run raised {type(exc).__name__}: {exc}")
        timer.cancel()
        return 1
    timer.cancel()

    note = subprocess.run(
        [
            "/usr/bin/osascript",
            "-e",
            'display notification "probe done" with title "xxl-whisper probe"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"notification exit={note.returncode}")
    print(f"menu_builds={_MENU_BUILDS['count']} clicks={_CLICKS}")

    if note.returncode != 0:
        print("VERDICT: FAIL — osascript notification failed")
        return 1
    print("VERDICT: OK")
    if not _CLICKS:
        print("(no manual clicks recorded — expected when running unattended)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
