@echo off
setlocal
title Ship Fault - Local Prediction Server
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment was not found.
    echo Create .venv and install requirements.txt before launching.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m ship_fault.cli --project "%~dp0." serve --host 127.0.0.1 --port 8000 --open-browser
if errorlevel 1 (
    echo The local prediction server could not start. See the error above.
    pause
    exit /b 1
)
endlocal
