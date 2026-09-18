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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM 0.29 upgrades AWQ to Marlin and unpacks weights on GPU (~680MiB extra).
# That OOMs a 12GB card after the 14B AWQ weights are already loaded.
export VLLM_BATCH_INVARIANT=1

exec vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.88 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000
