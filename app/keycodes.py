"""Platform key tables — pure data, importable on every platform.

These tables used to live in the Windows/macOS hook modules, so importing
:mod:`app.config` (which maps hotkey names to key codes and needs the default
hotkey) dragged a whole native backend in at import time. On macOS that pulled
in ``AppKit``/``Quartz``; anywhere else the bare ``else`` branch tried the same
thing and failed, which is why the platform-independent core could only be
imported — and therefore tested — on the two supported systems.

Keeping the numbers here means the hook modules still own the *behaviour*
(installing the hook, capturing keys) while the *data* is shared and testable.
"""

import sys
from collections.abc import Mapping
from types import MappingProxyType

#: Windows virtual-key codes for the preset hotkeys.
WIN_PRESET_KEYCODES: Mapping[str, int] = MappingProxyType(
    {
        "caps_lock": 0x14,
        "f2": 0x71,
        "f4": 0x73,
        "f6": 0x75,
        "f8": 0x77,
        "scroll_lock": 0x91,
        "mouse_x1": 0x05,  # VK_XBUTTON1 — mouse side button
        "mouse_x2": 0x06,  # VK_XBUTTON2
    }
)

#: macOS virtual keycodes for the preset hotkeys.
MAC_PRESET_KEYCODES: Mapping[str, int] = MappingProxyType(
    {
        "right_cmd": 0x36,
        "f2": 0x78,
        "f4": 0x76,
        "f6": 0x61,
        "f8": 0x64,
    }
)

#: Preset selected on a fresh Windows install.
WIN_DEFAULT_HOTKEY: str = "caps_lock"

#: Preset selected on a fresh macOS install.
MAC_DEFAULT_HOTKEY: str = "right_cmd"

#: Mouse buttons watched by the LL mouse hook on Windows.
WIN_MOUSE_KEYCODES: frozenset[int] = frozenset({0x05, 0x06})

#: No mouse side-button hotkeys on macOS.
MAC_MOUSE_KEYCODES: frozenset[int] = frozenset()

IS_WINDOWS: bool = sys.platform == "win32"

#: Only macOS switches tables: every other platform falls back to the Windows
#: set, which keeps the config defaults (``caps_lock``) coherent on a Linux CI
#: box. Those platforms have no backend anyway — see :mod:`app.native`.
IS_MACOS: bool = sys.platform == "darwin"

#: Active preset table for the running platform.
PRESET_KEYCODES: Mapping[str, int] = (
    MAC_PRESET_KEYCODES if IS_MACOS else WIN_PRESET_KEYCODES
)

#: Active default hotkey name for the running platform.
DEFAULT_HOTKEY: str = MAC_DEFAULT_HOTKEY if IS_MACOS else WIN_DEFAULT_HOTKEY

#: Active mouse-button set for the running platform.
MOUSE_KEYCODES: frozenset[int] = (
    MAC_MOUSE_KEYCODES if IS_MACOS else WIN_MOUSE_KEYCODES
)
