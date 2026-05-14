@echo off
echo Training gesture landmark model...

if exist .\.venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe .\gesture_pc_control\train_gesture_model.py
) else (
    echo ERROR: .venv\Scripts\python.exe not found.
    echo Create the environment and install requirements first.
    exit /b 1
)

echo.
echo Training complete.