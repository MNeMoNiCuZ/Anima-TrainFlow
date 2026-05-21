@echo off
setlocal
cd /d %~dp0

set "PY_EXE=%~dp0venv\Scripts\python.exe"
set PYTHONDONTWRITEBYTECODE=1

if not exist "%PY_EXE%" (
    echo [ERROR] Venv Python not found at:
    echo "%PY_EXE%"
    echo Create it first with venv_create.bat
    pause
    exit /b 1
)

"%PY_EXE%" launch.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script crashed. Check the error message above.
    pause
)
