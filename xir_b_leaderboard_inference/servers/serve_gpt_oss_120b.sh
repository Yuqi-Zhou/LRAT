#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the openai/gpt-oss-120b snapshot}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8001}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"

# GPT-OSS uses vLLM's Harmony-native Responses path.  Do not attach a Qwen or
# DeepSeek tool/reasoning parser to this service.
exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name gpt-oss-120b \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --max-model-len 100000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
