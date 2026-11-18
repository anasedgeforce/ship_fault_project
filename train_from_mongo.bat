@echo off
setlocal
title Simulator Fault Prediction - Train from External MongoDB
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment was not found.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts\train_from_mongo.py
set "TRAIN_EXIT=%errorlevel%"
if "%TRAIN_EXIT%"=="0" (
    echo Training finished. Restart the prediction server to load the new models.
) else (
    echo Import or training failed. Review the message above; the previous model remains active.
)
pause
exit /b %TRAIN_EXIT%
