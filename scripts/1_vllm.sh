#!/usr/bin/env bash
# Terminal 1: start vLLM. Wait until it says it is listening on 127.0.0.1:8000.
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

# PyTorch CUDA works without a local CUDA toolkit. FlashInfer JIT does not:
# warmup tries to compile sampling kernels with nvcc and dies if
# /usr/local/cuda is missing. Use the PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

exec vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.82 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000
