@echo off
REM Helper script to run MediaPipe-based scripts with project virtual environment
echo Starting MediaPipe application...
if exist .\.venv\Scripts\python.exe (
	.\.venv\Scripts\python.exe .\gesture_pc_control\main.py
) else (
	echo ERROR: .venv\Scripts\python.exe not found.
	echo Create the environment and install requirements first.
	exit /b 1
)

