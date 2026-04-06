#!/bin/bash
set -e

APP_DIR="/Users/shiyu/.openclaw/workspace/volresearch/iv_vega_hf/v1.5"
APP_FILE="$APP_DIR/app.py"
PORT="${PORT:-8502}"

find_python() {
  if [ -x "$APP_DIR/.venv/bin/python" ]; then
    echo "$APP_DIR/.venv/bin/python"
    return 0
  fi
  if [ -x "/opt/anaconda3/bin/python" ]; then
    echo "/opt/anaconda3/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "$(command -v python3)"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "$(command -v python)"
    return 0
  fi
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "未找到可用的 Python。"
  read -n 1 -s -r -p "按任意键退出..."
  exit 1
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  echo "当前 Python 环境里没有安装 streamlit。"
  echo "请先在用于运行可视化的环境里安装 streamlit，然后重新双击这个脚本。"
  echo "Python: $PYTHON_BIN"
  read -n 1 -s -r -p "按任意键退出..."
  exit 1
fi

cd "$APP_DIR"
exec "$PYTHON_BIN" -m streamlit run "$APP_FILE" --server.headless true --server.port "$PORT" --server.fileWatcherType poll
