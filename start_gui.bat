@echo off
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"
echo [Danbooru Tag Generator] Preparing environment...

set "BOOTSTRAP_PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "BOOTSTRAP_PYTHON=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "BOOTSTRAP_PYTHON=python"
    ) else if exist "%USERPROFILE%\miniconda3\python.exe" (
        set "BOOTSTRAP_PYTHON=%USERPROFILE%\miniconda3\python.exe"
    )
)

if "%BOOTSTRAP_PYTHON%"=="" (
    echo [ERROR] Python is not found.
    echo Please install Python 3.10+ or add python to PATH, then run this script again.
    pause
    exit /b 1
)

echo [1/4] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv...
    "%BOOTSTRAP_PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple"

echo [2/4] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip -i "%PIP_INDEX_URL%" >nul

echo [3/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt -i "%PIP_INDEX_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo [4/4] Verifying GUI runtime packages...
"%PYTHON_EXE%" -m pip install flet-web==0.26.0 -i "%PIP_INDEX_URL%" >nul
if errorlevel 1 (
    echo [WARN] Failed to install flet-web automatically. Trying without it...
)

echo Launching GUI...
"%PYTHON_EXE%" gui.py

if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with a non-zero code.
    pause
    exit /b 1
)

exit /b 0
