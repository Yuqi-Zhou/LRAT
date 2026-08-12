#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash-0731 snapshot}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name deepseek-v4-flash-0731 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --disable-uvicorn-access-log
