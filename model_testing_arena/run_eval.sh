#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/abhishek-mathur/miniconda3/envs/sentence_coach/bin/python}"
CONFIG="${CONFIG:-$ROOT/contenders.config}"

if [[ ! -x "$PYTHON" && ! -f "$PYTHON" ]]; then
  echo "python interpreter not found: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 1
fi

if ! "$PYTHON" -c "import json; json.load(open('$CONFIG'))" >/dev/null; then
  echo "contenders.config is not valid JSON" >&2
  exit 1
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export ARENA_MODEL_TIMEOUT="${ARENA_MODEL_TIMEOUT:-5400}"

echo "[arena] interpreter: $PYTHON"
echo "[arena] config: $CONFIG"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || true
fi

"$PYTHON" "$ROOT/eval.py" --self-test
"$PYTHON" "$ROOT/eval.py" --ensure-dataset --config "$CONFIG"
"$PYTHON" "$ROOT/eval.py" --validate --config "$CONFIG"
"$PYTHON" "$ROOT/eval.py" --run --config "$CONFIG" --python "$PYTHON" "$@"
