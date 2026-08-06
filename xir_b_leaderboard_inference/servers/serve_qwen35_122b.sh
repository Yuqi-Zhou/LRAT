#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-122B-A10B-FP8 snapshot}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8002}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name qwen3.5-122b-a10b-fp8 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
