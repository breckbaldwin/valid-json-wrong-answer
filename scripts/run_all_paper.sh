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
#           Wall-clock estimate: ~2h 50m on A100 80GB (CUAD has longer
#           prompts than the SGD schemas).
#
#   phase2  Margin-gating baseline confidence extraction (Flights, Restaurants,
#           CUAD × 3 scales).  Produces probability distributions used by
#           tab:cuad_per_field_scale, tab:gating_flights, tab:margin_scaling
#           and the baseline cells of tab:head_to_head.
#           Wraps scripts/runpod_baseline.py.
#           Wall-clock estimate: ~2 min (forward pass only on A100).
#
#   phase3  PCL fine-tuning (3-way labels) + eval (Flights, Restaurants, CUAD
#           × 3 scales).  Produces the PCL-FT cells of tab:head_to_head and
#           the data behind tab:pcl.
#           Wraps scripts/runpod_pcl_ft.py.
#           Wall-clock estimate: ~1h 30m.
#
#   phase4  Full-context CUAD re-evaluation.  Tests the §6.2 truncation
#           hypothesis: re-runs CUAD baseline + PCL-FT eval against
#           data/cuad_test_full.jsonl (~10K-token median) using existing
#           PCL-FT checkpoints.
#           Wall-clock estimate: ~10 min.
#
#   phase5  Per-role decomposition against the PCL checkpoints from
#           Phase 3 (Flights + Restaurants + CUAD × 3 scales).  Phase 3
#           produces only confidence/margin JSONs; Table 8 (tab:pcl)
#           needs per-role loss against the PCL adapters as well.
#           Wall-clock estimate: ~5 min.
#
# Outputs (cumulative across all phases):
#   results/<scale>_{baseline,finetuned}_{flights,restaurants,cuad}.json
#   results/<scale>_pcl_finetuned_{flights,restaurants,cuad}.json
#   results/margin_gating/<dataset>_qwen<scale>{,_pcl}{,_full}.json
#   checkpoints/{<scale>_<label>_lora_epoch10,<dataset>_pcl_qwen<scale>/lora_epoch5}/
#   results/run_all/*.log
#
# Total wall-clock budget: ~5 hours on a single A100 80GB PCIe (the
# canonical seeded run for the paper completed in 4h 39m).

set -euo pipefail

cd "$(dirname "$0")/.."

# Activate the project's virtual environment so the python in PATH is
# the one with valjson/torch/peft installed. The README directs users
# to create this with `python3 -m venv venv` before running.
if [ ! -f venv/bin/activate ]; then
    echo "ERROR: venv/bin/activate not found. Set up first with:"
    echo "    python3 -m venv venv"
    echo "    source venv/bin/activate"
    echo "    pip install -r requirements.txt"
    exit 1
fi
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

precheck() {
    # Idempotency means existing outputs are skipped silently. That is a
    # foot-gun if you are expecting a fresh run. Count what is already
    # there and shout if anything is.
    local p1 p2 p3 p4a p4b p5 ckpt total
    # `|| true` on each pipeline: under set -e + pipefail, grep returning
    # non-zero on empty input would otherwise abort the whole script before
    # the report ever prints.
    p1=$(  { ls results/*_baseline_*.json results/*_finetuned_*.json 2>/dev/null || true; } | { grep -v "_pcl_" || true; } | wc -l | tr -d ' ')
    p2=$(  { ls results/margin_gating/*_qwen[0-9]*[bB].json     2>/dev/null || true; } | { grep -vE "_pcl|_full" || true; } | wc -l | tr -d ' ')
    p3=$(  { ls results/margin_gating/*_qwen*_pcl.json          2>/dev/null || true; } | { grep -v "_full"     || true; } | wc -l | tr -d ' ')
    p4a=$( { ls results/margin_gating/cuad_qwen*_full.json      2>/dev/null || true; } | { grep -v "_pcl"      || true; } | wc -l | tr -d ' ')
    p4b=$( { ls results/margin_gating/cuad_qwen*_pcl_full.json  2>/dev/null || true; } | wc -l | tr -d ' ')
    p5=$(  { ls results/*_pcl_finetuned_*.json                  2>/dev/null || true; } | wc -l | tr -d ' ')
    ckpt=$( { ls -d checkpoints/*/lora_*                         2>/dev/null || true; } | wc -l | tr -d ' ')
    total=$((p1 + p2 + p3 + p4a + p4b + p5))

    echo "=================================================================="
    echo "Pre-check: pre-existing outputs (idempotent scripts will SKIP these)"
    echo "=================================================================="
    printf "  Phase 1 per-role JSONs:        %2d / 18\n"  "$p1"
    printf "  Phase 2 margin-gating:         %2d /  9\n"  "$p2"
    printf "  Phase 3 PCL eval JSONs:        %2d /  9\n"  "$p3"
    printf "  Phase 3 LoRA checkpoints:      %2d total (training will skip)\n" "$ckpt"
    printf "  Phase 4a CUAD full-context:    %2d /  3\n"  "$p4a"
    printf "  Phase 4b CUAD full PCL:        %2d /  3\n"  "$p4b"
    printf "  Phase 5 PCL per-role JSONs:    %2d /  9\n"  "$p5"
    echo

    if [ "$total" -eq 0 ] && [ "$ckpt" -eq 0 ]; then
        echo "  Clean slate — every requested phase will run fully."
    else
        echo "  ⚠  $total existing result files (+ $ckpt checkpoints) will be"
        echo "  ⚠  SKIPPED. If you wanted a fresh run, ctrl-C now and:"
        echo "  ⚠      rm -rf results/ checkpoints/"
        echo "  ⚠  Otherwise the run will only fill in missing pieces."
        sleep 5
    fi
    echo
}

precheck

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

phase5() {
    # Per-role decomposition against the PCL-trained checkpoints from
    # Phase 3. Phase 3 produced confidence/margin JSONs (in
    # results/margin_gating/) but not per-role loss JSONs — and Table 8
    # (tab:pcl) needs the per-role boolean+enum loss against the PCL
    # checkpoints. Output filename matches the convention
    # build_tables.py reads:  results/<scale>_pcl_finetuned_<dataset>.json
    echo
    echo "=================================================================="
    echo "Phase 5 | Per-role decomposition of PCL checkpoints | $(date)"
    echo "  Flights + Restaurants + CUAD × {0.5B, 7B, 32B}"
    echo "  Produces results/<scale>_pcl_finetuned_<dataset>.json (Table 8)"
    echo "=================================================================="

    # NOTE the two scale namespaces. The PCL checkpoint dirs use '0.5b'
    # (with the dot); the build_tables.py output convention uses '05b'
    # (no dot). Both are encoded in the SCALES rows below.
    local SCALES_PHASE5=(
        "05b:0.5b:Qwen/Qwen2.5-0.5B-Instruct"
        "7b:7b:Qwen/Qwen2.5-7B-Instruct"
        "32b:32b:Qwen/Qwen2.5-32B-Instruct"
    )
    local DSETS_PHASE5=(
        "flights:data/Flights_1_schema_pcl.json:data/Flights_1_test_pcl.jsonl"
        "restaurants:data/Restaurants_1_schema_pcl.json:data/Restaurants_1_test_pcl.jsonl"
        "cuad:data/cuad_schema.json:data/cuad_test.jsonl"
    )

    {
        for SROW in "${SCALES_PHASE5[@]}"; do
            IFS=: read OUT CKPT HF <<< "$SROW"
            for DROW in "${DSETS_PHASE5[@]}"; do
                IFS=: read DSET SCHEMA TEST <<< "$DROW"
                local OUTPATH="results/${OUT}_pcl_finetuned_${DSET}.json"
                local CKPTDIR="checkpoints/${DSET}_pcl_qwen${CKPT}/lora_epoch5"
                if [ -f "$OUTPATH" ]; then
                    echo ">>> [phase5][${OUT}][${DSET}] SKIP — $OUTPATH exists"
                    continue
                fi
                if [ ! -d "$CKPTDIR" ]; then
                    echo ">>> [phase5][${OUT}][${DSET}] SKIP — checkpoint missing: $CKPTDIR"
                    continue
                fi
                echo ">>> [phase5][${OUT}][${DSET}] decomposing against $CKPTDIR"
                python src/decompose.py \
                    --model "$HF" \
                    --checkpoint "$CKPTDIR" \
                    --schema "$SCHEMA" \
                    --data "$TEST" \
                    --device cuda \
                    --output "$OUTPATH"
            done
        done
    } 2>&1 | tee "$LOG_DIR/phase5_pcl_decompose.log"
}

# ---- Dispatch ----------------------------------------------------------------

if [ "$#" -eq 0 ]; then
    SELECTED=(phase1 phase2 phase3 phase4 phase5)
else
    SELECTED=("$@")
fi

for p in "${SELECTED[@]}"; do
    case $p in
        phase1|phase2|phase3|phase4|phase5) "$p" ;;
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
