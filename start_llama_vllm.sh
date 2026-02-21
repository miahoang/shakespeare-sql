#!/bin/bash
# Start vLLM for Llama 3.1 8B (OpenAI-compatible API on port 8000)
# source /workspace/ambrosia/bin/activate  # Activate your venv first
# cd to project root before running

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8002 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --trust-remote-code
