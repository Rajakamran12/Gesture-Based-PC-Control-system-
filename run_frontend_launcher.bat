@echo off
cd /d "%~dp0"

if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe .\frontend_launcher.py
) else (
    echo ERROR: .venv\Scripts\python.exe not found.
    echo Create .venv and install dependencies first.
    exit /b 1
)
