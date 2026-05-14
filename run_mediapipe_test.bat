@echo off
REM Helper script to run test.py with MediaPipe (for Raspberry Pi robot control)
echo Starting MediaPipe test application...
if exist .\.venv\Scripts\python.exe (
	.\.venv\Scripts\python.exe test.py
) else (
	echo ERROR: .venv\Scripts\python.exe not found.
	echo Create the environment and install requirements first.
	exit /b 1
)

