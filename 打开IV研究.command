#!/bin/bash
set -e
APP_DIR="/Users/shiyu/.openclaw/workspace/volresearch/timeseries/iv_inspector_refactor_toggle/iv_inspector_refactor"
cd "$APP_DIR"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
exec streamlit run app.py --server.headless true --server.port 8501
