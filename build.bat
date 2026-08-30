@echo off
rem ─── How to run ───
rem   build.bat            -> dist\xxl-whisper.exe (one file, no console)
rem Requires: repo venv at .venv (uv sync)
setlocal
set PY=.venv\Scripts\python.exe

%PY% -m PyInstaller ^
  --noconfirm --clean --onefile --noconsole ^
  --name xxl-whisper ^
  --collect-all sherpa_onnx ^
  --collect-all onnxruntime ^
  --hidden-import=sounddevice ^
  run.py

echo.
echo Done: dist\xxl-whisper.exe
endlocal
