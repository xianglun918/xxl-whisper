@echo off
rem ─── How to run ───
rem   build.bat            -> dist\xxl-whisper.exe (one file, no console)
rem Requires: repo venv at .venv (uv sync)
setlocal
set PY=.venv\Scripts\python.exe

rem Tk (tkinter) is Windows-only UI here: the status pill (indicator.py) and the
rem subtitle overlay (overlay.py). Listed explicitly so the frozen exe always
rem ships Tcl/Tk even if those imports move. macOS uses native AppKit instead.
%PY% -m PyInstaller ^
  --noconfirm --clean --onefile --noconsole ^
  --name xxl-whisper ^
  --collect-all sherpa_onnx ^
  --collect-all onnxruntime ^
  --hidden-import=sounddevice ^
  --hidden-import=soundcard ^
  --hidden-import=pystray ^
  --hidden-import=uiautomation ^
  --hidden-import=tkinter ^
  run.py

echo.
echo Done: dist\xxl-whisper.exe
endlocal
