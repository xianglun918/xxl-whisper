# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the macOS onedir .app bundle (M6).

Build:  bash build.sh
        -> dist/xxl-whisper.app  +  dist/xxl-whisper-arm64.zip (ad-hoc signed)
Sign:   CODESIGN_IDENTITY="Developer ID Application: ..." bash build.sh
        (empty = ad-hoc, which is what the public zip ships with)

Mirrors build.bat's collection (sherpa_onnx / onnxruntime / sounddevice) but
produces a onedir .app: onefile unpacks on every launch and its bootloader
process model does not suit a long-running menu-bar agent (M1 finding).
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all

sys.path.insert(0, os.path.abspath("."))
from app import __version__ as APP_VERSION

CODESIGN_IDENTITY = os.environ.get("CODESIGN_IDENTITY") or None

datas = []
binaries = []
hiddenimports = ["sounddevice"]
# The mac modules import pyobjc with importlib (dynamic) to keep the Windows
# runner's type checker happy (M1 finding), so PyInstaller cannot see them —
# list them explicitly or the bundled app dies with ModuleNotFoundError.
for package in (
    "sherpa_onnx",
    "onnxruntime",
    "Quartz",
    "AppKit",
    "Foundation",
    "ApplicationServices",
    "ServiceManagement",
    "objc",
    "PyObjCTools",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="xxl-whisper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=CODESIGN_IDENTITY,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="xxl-whisper",
)

app = BUNDLE(
    coll,
    name="xxl-whisper.app",
    icon=None,
    bundle_identifier="com.xianglun918.xxl-whisper",
    info_plist={
        # Menu-bar agent: no Dock icon, no window.
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        # Required or the packaged app is silently denied the microphone.
        "NSMicrophoneUsageDescription": "xxl-whisper 需要麦克风来录制语音并在本地识别。",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
    },
)
