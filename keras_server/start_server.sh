#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PORT="${PORT:-8777}"

# Create virtualenv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv "$VENV_DIR"
fi

# Activate and install deps
source "$VENV_DIR/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo "Starting Keras server on port $PORT..."
cd "$SCRIPT_DIR"
python server.py
