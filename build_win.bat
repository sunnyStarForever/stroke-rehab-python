@echo off
setlocal
cd /d "%~dp0"
echo Python-main Windows environment setup
if not exist .venv\Scripts\python.exe py -3 -m venv --system-site-packages .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe verify_runtime.py --models --ui
if errorlevel 1 exit /b 1
echo Setup complete. Start with: .venv\Scripts\python.exe main.py
