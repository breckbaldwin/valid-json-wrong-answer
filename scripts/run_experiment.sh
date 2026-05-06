#!/bin/bash
# Run per-grammar-role decomposition experiments.
#
# Usage:
#   bash scripts/run_experiment.sh all          # run everything (32B first)
#   bash scripts/run_experiment.sh 32b          # just 32B
#   bash scripts/run_experiment.sh 7b           # just 7B
#   bash scripts/run_experiment.sh 05b          # just 0.5B
#   bash scripts/run_experiment.sh decompose    # decompose only (skip training)
#
# Each scale runs: baseline decomposition → LoRA training → fine-tuned decomposition
# Biggest models first so we fail fast on memory issues.
#
# Prerequisites:
#   - bash scripts/smoketest.sh passes
#   - Data files in data/
#   - Models will be downloaded on first use

set -e

cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf_cache}

EPOCHS=10
DEVICE=cuda
RESTAURANTS_TRAIN=data/Restaurants_1_train.jsonl
RESTAURANTS_TEST=data/Restaurants_1_test.jsonl
RESTAURANTS_SCHEMA=data/Restaurants_1_schema.json
FLIGHTS_TRAIN=data/Flights_1_train.jsonl
FLIGHTS_TEST=data/Flights_1_test.jsonl
FLIGHTS_SCHEMA=data/Flights_1_schema.json
CUAD_TRAIN=data/cuad_train.jsonl
CUAD_TEST=data/cuad_test.jsonl
CUAD_SCHEMA=data/cuad_schema.json

mkdir -p results checkpoints

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

run_baseline_decompose() {
    local model=$1 scale=$2 dataset=$3 schema=$4 label=$5
    local output="results/${scale}_baseline_${label}.json"
    if [ -f "$output" ]; then
        echo ""
        echo ">>> [$scale] Baseline decomposition: $label  SKIP — $output exists"
        return
    fi
    echo ""
    echo ">>> [$scale] Baseline decomposition: $label"
    python src/decompose.py \
        --model "$model" \
        --data "$dataset" \
        --schema "$schema" \
        --device $DEVICE \
        --output "$output"
}

run_train() {
    local model=$1 scale=$2 data=$3 label=$4
    local final_ckpt="checkpoints/${scale}_${label}_lora_epoch${EPOCHS}"
    if [ -d "$final_ckpt" ]; then
        echo ""
        echo ">>> [$scale] LoRA training: $label  SKIP — $final_ckpt exists"
        return
    fi
    echo ""
    echo ">>> [$scale] LoRA training: $label ($EPOCHS epochs)"

    # 32B needs gradient checkpointing to fit in 80GB. max_seq_len uses
    # the default from src/train.py (2048) so CUAD's long prompts have
    # room for prompt context plus the JSON target.
    local extra_args=""
    if [ "$scale" = "32b" ]; then
        extra_args="--gradient-checkpointing"
    fi

    python src/train.py \
        --model "$model" \
        --data "$data" \
        --epochs $EPOCHS \
        --device $DEVICE \
        --checkpoint-dir "checkpoints" \
        --checkpoint-prefix "${scale}_${label}_lora" \
        $extra_args
}

run_finetuned_decompose() {
    local model=$1 scale=$2 dataset=$3 schema=$4 label=$5 epoch=$6
    local ckpt="checkpoints/${scale}_${label}_lora_epoch${epoch}"
    local output="results/${scale}_finetuned_${label}.json"
    if [ -f "$output" ]; then
        echo ""
        echo ">>> [$scale] Fine-tuned decomposition: $label (epoch $epoch)  SKIP — $output exists"
        return
    fi
    echo ""
    echo ">>> [$scale] Fine-tuned decomposition: $label (epoch $epoch)"
    python src/decompose.py \
        --model "$model" \
        --data "$dataset" \
        --schema "$schema" \
        --device $DEVICE \
        --checkpoint "$ckpt" \
        --output "$output"
}

run_scale() {
    local model=$1 scale=$2

    echo ""
    echo "============================================================"
    echo "  SCALE: $scale — $model"
    echo "============================================================"

    # --- Baseline decomposition ---
    run_baseline_decompose "$model" "$scale" "$RESTAURANTS_TEST" "$RESTAURANTS_SCHEMA" "restaurants"
    run_baseline_decompose "$model" "$scale" "$FLIGHTS_TEST" "$FLIGHTS_SCHEMA" "flights"
    run_baseline_decompose "$model" "$scale" "$CUAD_TEST" "$CUAD_SCHEMA" "cuad"

    # --- Training ---
    run_train "$model" "$scale" "$RESTAURANTS_TRAIN" "restaurants"
    run_train "$model" "$scale" "$FLIGHTS_TRAIN" "flights"
    run_train "$model" "$scale" "$CUAD_TRAIN" "cuad"

    # --- Fine-tuned decomposition ---
    run_finetuned_decompose "$model" "$scale" "$RESTAURANTS_TEST" "$RESTAURANTS_SCHEMA" "restaurants" "$EPOCHS"
    run_finetuned_decompose "$model" "$scale" "$FLIGHTS_TEST" "$FLIGHTS_SCHEMA" "flights" "$EPOCHS"
    run_finetuned_decompose "$model" "$scale" "$CUAD_TEST" "$CUAD_SCHEMA" "cuad" "$EPOCHS"

    echo ""
    echo ">>> [$scale] Complete. Results in results/${scale}_*.json"
}

run_decompose_only() {
    # Run decomposition on already-trained checkpoints (skip training)
    local model=$1 scale=$2

    echo ""
    echo "============================================================"
    echo "  DECOMPOSE ONLY: $scale — $model"
    echo "============================================================"

    run_baseline_decompose "$model" "$scale" "$RESTAURANTS_TEST" "$RESTAURANTS_SCHEMA" "restaurants"
    run_baseline_decompose "$model" "$scale" "$FLIGHTS_TEST" "$FLIGHTS_SCHEMA" "flights"
    run_baseline_decompose "$model" "$scale" "$CUAD_TEST" "$CUAD_SCHEMA" "cuad"
    run_finetuned_decompose "$model" "$scale" "$RESTAURANTS_TEST" "$RESTAURANTS_SCHEMA" "restaurants" "$EPOCHS"
    run_finetuned_decompose "$model" "$scale" "$FLIGHTS_TEST" "$FLIGHTS_SCHEMA" "flights" "$EPOCHS"
    run_finetuned_decompose "$model" "$scale" "$CUAD_TEST" "$CUAD_SCHEMA" "cuad" "$EPOCHS"
}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_32B="Qwen/Qwen2.5-32B-Instruct"
MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_05B="Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CMD=${1:-all}

case $CMD in
    32b)
        run_scale "$MODEL_32B" "32b"
        ;;
    7b)
        run_scale "$MODEL_7B" "7b"
        ;;
    05b)
        run_scale "$MODEL_05B" "05b"
        ;;
    all)
        echo "Running all scales: 32B → 7B → 0.5B"
        echo "Biggest first to fail fast on memory."
        echo ""
        run_scale "$MODEL_32B" "32b"
        run_scale "$MODEL_7B" "7b"
        run_scale "$MODEL_05B" "05b"
        ;;
    decompose)
        echo "Decompose only (skip training) — all scales"
        run_decompose_only "$MODEL_32B" "32b"
        run_decompose_only "$MODEL_7B" "7b"
        run_decompose_only "$MODEL_05B" "05b"
        ;;
    *)
        echo "Usage: bash scripts/run_experiment.sh {all|32b|7b|05b|decompose}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
ls -la results/*.json 2>/dev/null || echo "  (no results found)"
echo ""
echo "Next: python scripts/build_tables.py"
