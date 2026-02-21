#!/bin/bash
# Ablation study: Test all combinations of RAG and LoRA for both AmbiQT and Ambrosia

set -e  # Exit on error

PYTHON=/workspace/ambrosia/bin/python3

# Configuration
MODEL_URL="http://localhost:8000/v1"
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
LORA_BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LORA_PORT=8002
LORA_MODEL_URL="http://localhost:$LORA_PORT/v1"
LORA_ADAPTER_NAME="llama-grpo-curriculum"
LORA_ADAPTER="models/llama_grpo_curriculum/curriculum_20251123_055234/stage4_mixed/model"
LORA_GPU=1
LORA_GPU_UTIL=0.9
MAX_LORA_RANK=64
DEBERTA_MODEL="models/deberta-v3-base_diverse_20251115_160058"

# Results directory
RESULTS_DIR="outputs"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

start_vllm() {
    echo "Starting vLLM LoRA server..."
    echo "  Base model : $LORA_BASE_MODEL"
    echo "  Adapter    : $LORA_ADAPTER"
    echo "  Port       : $LORA_PORT  (GPU $LORA_GPU)"

    CUDA_VISIBLE_DEVICES=$LORA_GPU vllm serve "$LORA_BASE_MODEL" \
        --port $LORA_PORT \
        --enable-lora \
        --lora-modules "${LORA_ADAPTER_NAME}=${LORA_ADAPTER}" \
        --max-lora-rank $MAX_LORA_RANK \
        --gpu-memory-utilization $LORA_GPU_UTIL \
        --max-model-len 12288 \
        > "vllm_ablation.log" 2>&1 &

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

    echo "ERROR: vLLM server failed to start. See vllm_ablation.log"
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

echo "=========================================="
echo "Starting Ablation Study"
echo "=========================================="
echo ""

#############################################################################
# AMBIQT DATASET
#############################################################################

echo "=========================================="
echo "AMBIQT DATASET"
echo "=========================================="
echo ""

# 1. AmbiQT: No RAG, No LoRA
echo "[1/8] Running AmbiQT: No RAG, No LoRA..."
$PYTHON test_framework.py \
    --dataset ambiqt \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 0 \
    --rag-k-other 0 \
    --use-lora-validation False \
    --output-path "$RESULTS_DIR/framework_ambiqt_no_rag_no_lora.json" \
    --seed 42
echo "✓ Complete"
echo ""

2. AmbiQT: RAG, No LoRA
echo "[2/8] Running AmbiQT: RAG (k=5, k_other=0), No LoRA..."
$PYTHON test_framework.py \
    --dataset ambiqt \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 5 \
    --rag-k-other 0 \
    --use-lora-validation False \
    --output-path "$RESULTS_DIR/framework_ambiqt_rag_no_lora.json" \
    --seed 42
echo "✓ Complete"
echo ""

# 3. AmbiQT: No RAG, LoRA
echo "[3/8] Running AmbiQT: No RAG, LoRA..."
$PYTHON test_framework.py \
    --dataset ambiqt \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --lora-model-url "$LORA_MODEL_URL" \
    --lora-adapter "$LORA_ADAPTER" \
    --lora-adapter-name "$LORA_ADAPTER_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 0 \
    --rag-k-other 0 \
    --use-lora-validation True \
    --output-path "$RESULTS_DIR/framework_ambiqt_no_rag_lora.json" \
    --seed 42
echo "✓ Complete"
echo ""

# 4. AmbiQT: RAG, LoRA
echo "[4/8] Running AmbiQT: RAG (k=5, k_other=0), LoRA..."
$PYTHON test_framework.py \
    --dataset ambiqt \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --lora-model-url "$LORA_MODEL_URL" \
    --lora-adapter "$LORA_ADAPTER" \
    --lora-adapter-name "$LORA_ADAPTER_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 5 \
    --rag-k-other 0 \
    --use-lora-validation True \
    --output-path "$RESULTS_DIR/framework_ambiqt_rag_lora.json" \
    --seed 42
echo "✓ Complete"
echo ""

#############################################################################
# AMBROSIA DATASET
#############################################################################

echo "=========================================="
echo "AMBROSIA DATASET"
echo "=========================================="
echo ""

# 5. Ambrosia: No RAG, No LoRA
echo "[5/8] Running Ambrosia: No RAG, No LoRA..."
$PYTHON test_framework.py \
    --dataset ambrosia \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 0 \
    --rag-k-other 0 \
    --use-lora-validation False \
    --output-path "$RESULTS_DIR/framework_ambrosia_no_rag_no_lora_new.json" \
    --seed 42 \
    --exclude-unanswerable
echo "✓ Complete"
echo ""

# 6. Ambrosia: RAG, No LoRA
echo "[6/8] Running Ambrosia: RAG (k=5, k_other=0), No LoRA..."
$PYTHON test_framework.py \
    --dataset ambrosia \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 5 \
    --rag-k-other 0 \
    --use-lora-validation False \
    --output-path "$RESULTS_DIR/framework_ambrosia_rag_no_lora_new.json" \
    --seed 42 \
    --exclude-unanswerable
echo "✓ Complete"
echo ""

# Start vLLM for LoRA runs
start_vllm

# 7. Ambrosia: No RAG, LoRA
echo "[7/8] Running Ambrosia: No RAG, LoRA..."
$PYTHON test_framework.py \
    --dataset ambrosia \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --lora-model-url "$LORA_MODEL_URL" \
    --lora-adapter "$LORA_ADAPTER" \
    --lora-adapter-name "$LORA_ADAPTER_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 0 \
    --rag-k-other 0 \
    --use-lora-validation True \
    --output-path "$RESULTS_DIR/framework_ambrosia_no_rag_lora_new.json" \
    --seed 42 \
    --exclude-unanswerable
echo "✓ Complete"
echo ""

# 8. Ambrosia: RAG, LoRA
echo "[8/8] Running Ambrosia: RAG (k=5, k_other=0), LoRA..."
$PYTHON test_framework.py \
    --dataset ambrosia \
    --agent-type integrated \
    --model-url "$MODEL_URL" \
    --model-name "$MODEL_NAME" \
    --lora-model-url "$LORA_MODEL_URL" \
    --lora-adapter "$LORA_ADAPTER" \
    --lora-adapter-name "$LORA_ADAPTER_NAME" \
    --deberta-model "$DEBERTA_MODEL" \
    --rag-k 5 \
    --rag-k-other 0 \
    --use-lora-validation True \
    --output-path "$RESULTS_DIR/framework_ambrosia_rag_lora_new.json" \
    --seed 42 \
    --exclude-unanswerable
echo "✓ Complete"
echo ""

stop_vllm

#############################################################################
# SUMMARY
#############################################################################

# NOTE: Additional configuration to test in future:
# - RAG + LoRA WITHOUT DeBERTa router (force all questions as AA)
# This would require a separate test script that bypasses classification

echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files generated:"
echo "  AmbiQT:"
echo "    - framework_ambiqt_no_rag_no_lora.json"
echo "    - framework_ambiqt_rag_no_lora.json"
echo "    - framework_ambiqt_no_rag_lora.json"
echo "    - framework_ambiqt_rag_lora.json"
echo ""
echo "  Ambrosia:"
echo "    - framework_ambrosia_no_rag_no_lora.json"
echo "    - framework_ambrosia_rag_no_lora.json"
echo "    - framework_ambrosia_no_rag_lora.json"
echo "    - framework_ambrosia_rag_lora.json"
echo ""
echo "Run '$PYTHON analyze_ablation_results.py' to compare results"
