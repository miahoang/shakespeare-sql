#!/bin/bash
# Start vLLM for Qwen2.5-Coder 32B (OpenAI-compatible API on port 8001)
# source /workspace/ambrosia/bin/activate  # Activate your venv first
# cd to project root before running

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --trust-remote-code \
    --max-num-seqs 32 \
    --enforce-eager \
    --disable-log-requests
  
