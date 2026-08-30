# ─── How to run ───
# Dev:   .venv\Scripts\python.exe run.py
# Build: build.bat  -> dist\xl-whisper.exe
"""xxl-whisper dev entry point."""

from app.main import entry

if __name__ == "__main__":
    entry()
