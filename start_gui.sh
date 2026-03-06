#!/bin/bash

cd "$(dirname "$0")"
echo "[Danbooru Tag Generator] Preparing environment..."

if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not found."
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    read -p "Press Enter to exit..."
    exit 1
fi

VENV_PYTHON=".venv/bin/python"
UV_INDEX_URL="https://mirrors.ustc.edu.cn/pypi/simple"

echo "[1/3] Checking virtual environment..."
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating .venv..."
    uv venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

echo "[2/3] Installing dependencies with uv..."
uv pip install --python "$VENV_PYTHON" -r requirements.txt --index-url "$UV_INDEX_URL"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[3/3] Launching GUI..."
uv run --python "$VENV_PYTHON" gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] GUI exited with a non-zero code."
    read -p "Press Enter to exit..."
    exit 1
fi

exit 0
