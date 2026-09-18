#!/usr/bin/env bash
# Terminal 2: FastAPI UI plus WebSocket. Start after vLLM is up. Open http://localhost:8003
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/.."

set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate sentence_coach
set -u

if [ -d "${SCRIPT_DIR}/../frontend" ]; then
  echo "Building frontend..."
  (cd "${SCRIPT_DIR}/../frontend" && npm install --silent && npm run build)
fi

exec python main.py
