@echo off
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"
echo [Danbooru Tag Generator] Preparing environment...

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not found.
    echo Please install uv first: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple"

echo [1/3] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv...
    uv venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/3] Installing dependencies with uv...
uv pip install --python "%VENV_PYTHON%" -r requirements.txt --index-url "%UV_INDEX_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo [3/3] Launching GUI...
uv run --python "%VENV_PYTHON%" gui.py

if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with a non-zero code.
    pause
    exit /b 1
)

exit /b 0
