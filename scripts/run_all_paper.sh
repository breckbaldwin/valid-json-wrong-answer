#!/usr/bin/env bash
# Reproduce every experiment referenced in "Valid JSON, Wrong Answer".
# Designed for a fresh RunPod A100 80GB with /workspace/valid-json-wrong-answer present.
# Idempotent: re-running picks up where any earlier phase left off, because
# every underlying script skips outputs that already exist.
#
# Usage:
#   bash scripts/run_all_paper.sh             # all phases in order
#   bash scripts/run_all_paper.sh phase2      # one named phase
#   bash scripts/run_all_paper.sh phase3 phase4   # multiple named phases
#
# Phases (each idempotent; safe to interrupt and resume):
#   phase1  Per-grammar-role decomposition (Flights + Restaurants + CUAD ×
#           3 scales, baseline + 10-epoch standard LoRA).  Produces
#           tab:aggregate, tab:flights_decomp, tab:restaurants_decomp,
#           tab:enum_trend, plus per-role decomposition for CUAD (used as
#           supporting evidence in §5.6).  Wraps existing
#           scripts/run_experiment.sh.
#           Wall-clock estimate: ~3.5 hours on A100 80GB (CUAD has longer
#           prompts than the SGD schemas).
#
#   phase2  Margin-gating baseline confidence extraction (Flights, Restaurants,
#           CUAD × 3 scales).  Produces probability distributions used by
#           tab:cuad_per_field_scale, tab:gating_flights, tab:margin_scaling
#           and the baseline cells of tab:head_to_head.
#           Wraps scripts/runpod_baseline.py.
#           Wall-clock estimate: ~30 min.
#
#   phase3  PCL fine-tuning (3-way labels) + eval (Flights, Restaurants, CUAD
#           × 3 scales).  Produces the PCL-FT cells of tab:head_to_head and
#           the data behind tab:pcl.
#           Wraps scripts/runpod_pcl_ft.py.
#           Wall-clock estimate: ~90 min.
#
#   phase4  Full-context CUAD re-evaluation.  Tests the §6.2 truncation
#           hypothesis: re-runs CUAD baseline + PCL-FT eval against
#           data/cuad_test_full.jsonl (~10K-token median) using existing
#           PCL-FT checkpoints.
#           Wall-clock estimate: ~25 min.
#
# Outputs (cumulative across all phases):
#   results/<scale>_{baseline,finetuned}_{flights,restaurants}.json
#   results/margin_gating/<dataset>_qwen<scale>{,_pcl}{,_full}.json
#   checkpoints/{<scale>_<label>_lora_epoch10,<dataset>_pcl_qwen<scale>/lora_epoch5}/
#   results/run_all/*.log
#
# Total wall-clock budget: ~6 hours on a single A100 80GB PCIe.

set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/hf_cache}

LOG_DIR=results/run_all
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "=== run_all_paper.sh started $(date) ==="
echo "Logs:    $LOG_DIR/"
echo "HF_HOME: $HF_HOME"
echo ""

# ---- Phase definitions -------------------------------------------------------

phase1() {
    echo
    echo "=================================================================="
    echo "Phase 1 | Per-role decomposition | $(date)"
    echo "  Flights + Restaurants + CUAD × {0.5B, 7B, 32B} × {baseline, 10-epoch LoRA}"
    echo "=================================================================="
    bash scripts/run_experiment.sh all 2>&1 | tee "$LOG_DIR/phase1_per_role.log"
}

phase2() {
    echo
    echo "=================================================================="
    echo "Phase 2 | Margin-gating baseline confidence | $(date)"
    echo "  Flights + Restaurants + CUAD × {0.5B, 7B, 32B}"
    echo "=================================================================="
    python scripts/runpod_baseline.py --scale all 2>&1 \
        | tee "$LOG_DIR/phase2_margin_baseline.log"
}

phase3() {
    echo
    echo "=================================================================="
    echo "Phase 3 | PCL fine-tuning + eval | $(date)"
    echo "  Flights + Restaurants + CUAD × {0.5B, 7B, 32B}"
    echo "=================================================================="
    python scripts/runpod_pcl_ft.py \
        --datasets flights,restaurants,cuad 2>&1 \
        | tee "$LOG_DIR/phase3_pcl_ft.log"
}

phase4() {
    echo
    echo "=================================================================="
    echo "Phase 4 | Full-context CUAD re-eval | $(date)"
    echo "  Tests §6.2 truncation hypothesis"
    echo "=================================================================="

    # 4a — baseline at full context
    python scripts/runpod_baseline.py \
        --datasets cuad \
        --test-data-override data/cuad_test_full.jsonl \
        --tag-suffix _full \
        --scale all 2>&1 \
        | tee "$LOG_DIR/phase4a_full_baseline.log"

    # 4b — PCL-FT at full context (uses existing checkpoints from phase 3)
    python scripts/runpod_pcl_ft.py \
        --datasets cuad \
        --stage eval \
        --test-data-override data/cuad_test_full.jsonl \
        --tag-suffix _full 2>&1 \
        | tee "$LOG_DIR/phase4b_full_pcl.log"
}

# ---- Dispatch ----------------------------------------------------------------

if [ "$#" -eq 0 ]; then
    SELECTED=(phase1 phase2 phase3 phase4)
else
    SELECTED=("$@")
fi

for p in "${SELECTED[@]}"; do
    case $p in
        phase1|phase2|phase3|phase4) "$p" ;;
        *) echo "Unknown phase: $p"; exit 1 ;;
    esac
done

ELAPSED=$(($(date +%s) - START))
echo
echo "=================================================================="
echo "  All requested phases complete."
echo "  Total wall-clock: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "=================================================================="
echo "Outputs:"
echo "  Per-role decomposition (Phase 1):"
ls results/*.json 2>/dev/null | grep -E "(baseline|finetuned)" | head -6 || true
echo "  Margin-gating probs (Phases 2-4):"
ls results/margin_gating/*.json 2>/dev/null | head -12 || true
echo "  LoRA checkpoints:"
ls -d checkpoints/*/lora_* 2>/dev/null | head -12 || true
echo
echo "Next: rsync results/ checkpoints/ back to Mac for Mac-side analysis."
