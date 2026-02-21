#!/bin/bash
# Adapter study: Test gpt-4.1, gpt-5.2, and Qwen curriculum adapters
# Configurations: No RAG + LoRA  and  RAG (k=5, k_other=1) + LoRA
# Starts vLLM LoRA server for each adapter, runs 2 evals, then stops it.

set -e

PYTHON=/workspace/ambrosia/bin/python3

# Configuration
MODEL_URL="http://localhost:8000/v1"
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
LORA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LORA_PORT=8002
LORA_MODEL_URL="http://localhost:$LORA_PORT/v1"
LORA_GPU=1
LORA_GPU_UTIL=0.9
MAX_LORA_RANK=64
DEBERTA_MODEL="models/deberta-v3-base_diverse_20251115_160058"

# Adapter paths (stage4_mixed final model for each curriculum run)
LORA_ADAPTER_GPT41="models/llama_grpo_curriculum/curriculum_gpt-4_1-2025-04-14_20260214_103006/stage4_mixed/model"
LORA_ADAPTER_GPT52="models/llama_grpo_curriculum/curriculum_gpt-5_2-2025-12-11_20260214_053301/stage4_mixed/model"
LORA_ADAPTER_QWEN="models/llama_grpo_curriculum/curriculum_Qwen-Qwen3-235B-A22B-Instruct-2507-FP8_20260214_153240/stage4_mixed/model"

# Adapter names (used as --lora-adapter-name on the vLLM server)
LORA_ADAPTER_NAME_GPT41="llama-grpo-gpt41"
LORA_ADAPTER_NAME_GPT52="llama-grpo-gpt52"
LORA_ADAPTER_NAME_QWEN="llama-grpo-qwen"

# Results directory
RESULTS_DIR="outputs"
mkdir -p "$RESULTS_DIR"

# Common parameters
SEED=42

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

start_vllm() {
    local adapter_name=$1
    local adapter_path=$2

    echo "Starting vLLM LoRA server: $adapter_name"
    echo "  Base model : $LORA_BASE_MODEL"
    echo "  Adapter    : $adapter_path"
    echo "  Port       : $LORA_PORT  (GPU $LORA_GPU)"

    CUDA_VISIBLE_DEVICES=$LORA_GPU vllm serve "$LORA_BASE_MODEL" \
        --port $LORA_PORT \
        --enable-lora \
        --lora-modules "${adapter_name}=${adapter_path}" \
        --max-lora-rank $MAX_LORA_RANK \
        --gpu-memory-utilization $LORA_GPU_UTIL \
        --max-model-len 12288 \
        > "vllm_${adapter_name}.log" 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    echo "Waiting for server to be ready..."
    sleep 30
    local retries=10
    local i=0
    while [ $i -lt $retries ]; do
        if curl -s "http://localhost:$LORA_PORT/health" > /dev/null 2>&1; then
            echo "✓ vLLM server ready"
            return 0
        fi
        echo "  Still waiting... ($((i+1))/$retries)"
        sleep 10
        i=$((i+1))
    done

    echo "ERROR: vLLM server failed to start. See vllm_${adapter_name}.log"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    sleep 10
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -9 "$VLLM_PID" 2>/dev/null || true
        sleep 5
    fi
    echo "✓ vLLM server stopped"
}

cleanup() {
    echo ""
    echo "Interrupted. Cleaning up..."
    [ -n "$VLLM_PID" ] && stop_vllm
    exit 1
}

trap cleanup INT TERM

run_eval() {
    local label=$1        # e.g. "gpt41"
    local adapter_name=$2
    local rag_k=$3
    local rag_k_other=$4
    local suffix=$5       # e.g. "no_rag_lora" or "rag_lora"

    $PYTHON test_framework.py \
        --dataset ambrosia \
        --agent-type integrated \
        --model-url "$MODEL_URL" \
        --model-name "$MODEL_NAME" \
        --lora-model-url "$LORA_MODEL_URL" \
        --lora-adapter-name "$adapter_name" \
        --deberta-model "$DEBERTA_MODEL" \
        --rag-k "$rag_k" \
        --rag-k-other "$rag_k_other" \
        --use-lora-validation True \
        --output-path "$RESULTS_DIR/framework_ambrosia_${label}_${suffix}.json" \
        --seed $SEED
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=========================================="
echo "New Adapter Study"
echo "=========================================="
echo "Adapters:"
echo "  gpt-4.1 : $LORA_ADAPTER_GPT41"
echo "  gpt-5.2 : $LORA_ADAPTER_GPT52"
echo "  Qwen    : $LORA_ADAPTER_QWEN"
echo ""
echo "Configurations: No RAG + LoRA  |  RAG (k=5, k_other=0) + LoRA"
echo "Results dir   : $RESULTS_DIR"
echo "=========================================="
echo ""

#############################################################################
# ADAPTER: gpt-4.1
#############################################################################

echo "=========================================="
echo "ADAPTER: gpt-4.1  (1/3)"
echo "=========================================="
start_vllm "$LORA_ADAPTER_NAME_GPT41" "$LORA_ADAPTER_GPT41"

echo "[1/6] gpt-4.1: No RAG, LoRA..."
run_eval "gpt41" "$LORA_ADAPTER_NAME_GPT41" 0 0 "no_rag_lora"
echo "✓ Complete"

echo "[2/6] gpt-4.1: RAG (k=5), LoRA..."
run_eval "gpt41" "$LORA_ADAPTER_NAME_GPT41" 5 0 "rag_lora"
echo "✓ Complete"

stop_vllm
sleep 10

#############################################################################
# ADAPTER: gpt-5.2
#############################################################################

echo "=========================================="
echo "ADAPTER: gpt-5.2  (2/3)"
echo "=========================================="
start_vllm "$LORA_ADAPTER_NAME_GPT52" "$LORA_ADAPTER_GPT52"

echo "[3/6] gpt-5.2: No RAG, LoRA..."
run_eval "gpt52" "$LORA_ADAPTER_NAME_GPT52" 0 0 "no_rag_lora"
echo "✓ Complete"

echo "[4/6] gpt-5.2: RAG (k=5), LoRA..."
run_eval "gpt52" "$LORA_ADAPTER_NAME_GPT52" 5 0 "rag_lora"
echo "✓ Complete"

stop_vllm
sleep 10

#############################################################################
# ADAPTER: Qwen
#############################################################################

echo "=========================================="
echo "ADAPTER: Qwen  (3/3)"
echo "=========================================="
start_vllm "$LORA_ADAPTER_NAME_QWEN" "$LORA_ADAPTER_QWEN"

echo "[5/6] Qwen: No RAG, LoRA..."
run_eval "qwen" "$LORA_ADAPTER_NAME_QWEN" 0 0 "no_rag_lora"
echo "✓ Complete"

echo "[6/6] Qwen: RAG (k=5), LoRA..."
run_eval "qwen" "$LORA_ADAPTER_NAME_QWEN" 5 0 "rag_lora"
echo "✓ Complete"

stop_vllm

#############################################################################
# SUMMARY
#############################################################################

echo ""
echo "=========================================="
echo "New Adapter Study Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files generated:"
echo "  gpt-4.1:"
echo "    - framework_ambrosia_gpt41_no_rag_lora.json"
echo "    - framework_ambrosia_gpt41_rag_lora.json"
echo ""
echo "  gpt-5.2:"
echo "    - framework_ambrosia_gpt52_no_rag_lora.json"
echo "    - framework_ambrosia_gpt52_rag_lora.json"
echo ""
echo "  Qwen:"
echo "    - framework_ambrosia_qwen_no_rag_lora.json"
echo "    - framework_ambrosia_qwen_rag_lora.json"
echo ""
echo "Run '$PYTHON analyze_ablation_results.py' to compare results"
