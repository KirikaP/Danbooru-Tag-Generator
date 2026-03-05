#!/bin/bash

cd "$(dirname "$0")"
echo "[Danbooru Tag Generator] Preparing environment..."

BOOTSTRAP_PYTHON=""
if [ -f ".venv/bin/python" ]; then
    BOOTSTRAP_PYTHON=".venv/bin/python"
else
    if command -v python3 &> /dev/null; then
        BOOTSTRAP_PYTHON="python3"
    elif command -v python &> /dev/null; then
        BOOTSTRAP_PYTHON="python"
    elif [ -f "$HOME/miniconda3/bin/python" ]; then
        BOOTSTRAP_PYTHON="$HOME/miniconda3/bin/python"
    elif [ -f "$HOME/anaconda3/bin/python" ]; then
        BOOTSTRAP_PYTHON="$HOME/anaconda3/bin/python"
    fi
fi

if [ -z "$BOOTSTRAP_PYTHON" ]; then
    echo "[ERROR] Python is not found."
    echo "Please install Python 3.10+ or add python to PATH, then run this script again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[1/4] Checking virtual environment..."
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating .venv..."
    "$BOOTSTRAP_PYTHON" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

PYTHON_EXE=".venv/bin/python"
PIP_INDEX_URL="https://mirrors.ustc.edu.cn/pypi/simple"

echo "[2/4] Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip -i "$PIP_INDEX_URL" > /dev/null

echo "[3/4] Installing dependencies..."
"$PYTHON_EXE" -m pip install -r requirements.txt -i "$PIP_INDEX_URL"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[4/4] Verifying GUI runtime packages..."
"$PYTHON_EXE" -m pip install flet-web==0.26.0 -i "$PIP_INDEX_URL" > /dev/null
if [ $? -ne 0 ]; then
    echo "[WARN] Failed to install flet-web automatically. Trying without it..."
fi

echo "Launching GUI..."
"$PYTHON_EXE" gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] GUI exited with a non-zero code."
    read -p "Press Enter to exit..."
    exit 1
fi

exit 0
