#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_FILE="$SCRIPT_DIR/app.py"
PORT="${PORT:-8502}"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x /opt/anaconda3/bin/python ]; then
  PYTHON_BIN="/opt/anaconda3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No usable Python found."
  exit 1
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit is not installed in: $PYTHON_BIN"
  echo "Install streamlit in that environment, then rerun this script."
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" -m streamlit run "$APP_FILE" --server.headless true --server.port "$PORT" --server.fileWatcherType poll
