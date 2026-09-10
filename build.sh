#!/usr/bin/env bash
# ─── How to run ───
#   bash build.sh                       -> dist/xxl-whisper.app + xxl-whisper-arm64.zip (ad-hoc)
#   CODESIGN_IDENTITY="Developer ID Application: ..." bash build.sh   (notarizable)
# Requires: repo venv at .venv (uv sync)
set -euo pipefail

PY=".venv/bin/python"

rm -rf build dist
"$PY" -m PyInstaller --noconfirm --clean xxl-whisper-mac.spec

# Ship the onedir .app as a zip; ditto preserves symlinks + metadata so the
# extracted bundle still launches (plain zip mangles framework symlinks).
ditto -c -k --keepParent dist/xxl-whisper.app dist/xxl-whisper-arm64.zip

echo
echo "Done: dist/xxl-whisper.app"
echo "      dist/xxl-whisper-arm64.zip"
