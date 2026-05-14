@echo off
REM Run modular Gesture-Based PC Control dashboard with MediaPipe environment
cd /d "%~dp0"
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe .\gesture_pc_control\main.py
) else (
    echo .venv not found. Please create it or install dependencies in your active Python environment.
)
