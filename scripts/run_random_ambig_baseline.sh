#!/usr/bin/env bash
# Random-ambiguous baseline: train + per-role decompose at all scales.
#
# Trains LoRA on data/Flights_1_train_random_ambig.jsonl (193/250 = 77.2%
# of training examples have refundable randomly relabeled to "ambiguous"),
# then decomposes against the PCL test set. Tests whether PCL's gain comes
# from the cue-based relabeling specifically, vs any abstain-class
# introduction at the same overall rate.
#
# Mirrors PCL phase 3 (training, but on random-ambig data instead of
# cue-driven PCL) + phase 5 (per-role decomposition against the resulting
# checkpoints), so output JSONs slot directly into build_tables.py.
#
# Outputs per scale:
#   checkpoints/flights_random_ambig_qwen<scale>/lora_epoch5/
#   results/<scale-out>_random_ambig_finetuned_flights.json
#
# Idempotent: skips work whose outputs already exist.
#
# Usage:
#   bash scripts/run_random_ambig_baseline.sh                # all 3 scales
#   bash scripts/run_random_ambig_baseline.sh 05b            # one scale
#   bash scripts/run_random_ambig_baseline.sh 7b 32b         # two scales

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f venv/bin/activate ]; then
    echo "ERROR: venv/bin/activate not found. Set up first with:"
    echo "    python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source venv/bin/activate

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

EPOCHS=5
TRAIN_DATA=data/Flights_1_train_random_ambig.jsonl
TEST_DATA=data/Flights_1_test_pcl.jsonl
SCHEMA=data/Flights_1_schema_pcl.json

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found. Generate first with:"
    echo "    python src/random_relabel.py \\"
    echo "        --input  data/Flights_1_train.jsonl \\"
    echo "        --output $TRAIN_DATA \\"
    echo "        --field  refundable --rate 0.772 --seed 42"
    exit 1
fi
for f in "$TEST_DATA" "$SCHEMA"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

# Each row: <output-scale-tag>:<checkpoint-scale-tag>:<HF model name>
# Output tag matches build_tables.py's filename convention (05b/7b/32b);
# checkpoint tag mirrors the PCL convention (0.5b/7b/32b).
ALL_SCALES=(
    "05b:0.5b:Qwen/Qwen2.5-0.5B-Instruct"
    "7b:7b:Qwen/Qwen2.5-7B-Instruct"
    "32b:32b:Qwen/Qwen2.5-32B-Instruct"
)

if [ "$#" -eq 0 ]; then
    SELECTED=("${ALL_SCALES[@]}")
else
    SELECTED=()
    for arg in "$@"; do
        match=""
        for row in "${ALL_SCALES[@]}"; do
            IFS=: read -r out _ _ <<< "$row"
            if [ "$out" = "$arg" ]; then match="$row"; break; fi
        done
        [ -n "$match" ] || { echo "ERROR: unknown scale '$arg' (want one of 05b 7b 32b)"; exit 1; }
        SELECTED+=("$match")
    done
fi

LOG_DIR=results/run_all
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/random_ambig_$(date +%Y%m%d_%H%M).log"

echo "=== run_random_ambig_baseline.sh started $(date) ===" | tee "$LOG"
echo "  train: $TRAIN_DATA" | tee -a "$LOG"
echo "  test:  $TEST_DATA"  | tee -a "$LOG"
echo "  scales:" | tee -a "$LOG"
for row in "${SELECTED[@]}"; do echo "    $row" | tee -a "$LOG"; done

START=$(date +%s)

for ROW in "${SELECTED[@]}"; do
    IFS=: read -r OUT CKPT HF <<< "$ROW"
    CKPT_DIR="checkpoints/flights_random_ambig_qwen${CKPT}"
    FINAL_CKPT="$CKPT_DIR/lora_epoch${EPOCHS}"
    OUT_JSON="results/${OUT}_random_ambig_finetuned_flights.json"

    echo "" | tee -a "$LOG"
    echo "==================================================================" | tee -a "$LOG"
    echo "=== ${OUT} | $(date) ===" | tee -a "$LOG"
    echo "==================================================================" | tee -a "$LOG"

    # ---- Train -----------------------------------------------------------
    if [ -d "$FINAL_CKPT" ]; then
        echo ">>> [train][${OUT}] SKIP — $FINAL_CKPT exists" | tee -a "$LOG"
    else
        mkdir -p "$CKPT_DIR"
        echo ">>> [train][${OUT}] training on $TRAIN_DATA → $CKPT_DIR" | tee -a "$LOG"
        python src/train.py \
            --model "$HF" \
            --data "$TRAIN_DATA" \
            --device cuda \
            --checkpoint-dir "$CKPT_DIR" \
            --checkpoint-prefix lora \
            --epochs "$EPOCHS" \
            --gradient-checkpointing \
            2>&1 | tee -a "$LOG"
    fi

    # ---- Decompose -------------------------------------------------------
    if [ -f "$OUT_JSON" ]; then
        echo ">>> [decompose][${OUT}] SKIP — $OUT_JSON exists" | tee -a "$LOG"
    else
        echo ">>> [decompose][${OUT}] decomposing on $TEST_DATA → $OUT_JSON" | tee -a "$LOG"
        python src/decompose.py \
            --model "$HF" \
            --checkpoint "$FINAL_CKPT" \
            --schema "$SCHEMA" \
            --data "$TEST_DATA" \
            --device cuda \
            --output "$OUT_JSON" \
            2>&1 | tee -a "$LOG"
    fi
done

ELAPSED=$(($(date +%s) - START))
echo "" | tee -a "$LOG"
echo "==================================================================" | tee -a "$LOG"
echo "=== Complete | $((ELAPSED/60))m $((ELAPSED%60))s ===" | tee -a "$LOG"
echo "==================================================================" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
ls -1 results/*_random_ambig_finetuned_flights.json 2>/dev/null | tee -a "$LOG" || true
echo "" | tee -a "$LOG"
echo "Next: extend build_tables.build_pcl to read random-ambig JSONs alongside" | tee -a "$LOG"
echo "the PCL JSONs, producing a 4-column tab:pcl (2-way / 3-way PCL / random-ambig / Δ)." | tee -a "$LOG"
