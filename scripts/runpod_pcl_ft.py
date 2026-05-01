#!/usr/bin/env python3
"""PCL fine-tune + margin-gating eval across scales on Flights_1.

For each of {0.5B, 7B, 32B}:
  1. Train LoRA on `data/Flights_1_train_pcl.jsonl` (3-way schema with
     `ambiguous`). Saves checkpoints under `checkpoints/pcl_qwen<scale>/`.
  2. Load base model + LoRA, extract per-record probability distributions
     on `data/Flights_1_test_pcl.jsonl`.
  3. Write `results/margin_gating/flights_qwen<scale>_pcl.json` in the
     same schema as `runpod_baseline.py` outputs so that
     `scripts/margin_gating_eval.py --reuse-probs` can consume it
     alongside the baseline files for head-to-head (E7).

Idempotent:
  - Training is skipped if the checkpoint dir exists and contains a
    final-epoch LoRA adapter.
  - Eval is skipped if the output JSON exists.
  - Delete the corresponding artefact to force a redo.

Run order: 0.5B → 7B → 32B (fastest first so failures surface early).

Usage:
    # Full sweep
    python scripts/runpod_pcl_ft.py

    # One scale only
    python scripts/runpod_pcl_ft.py --scales 0.5b

    # Custom epochs
    python scripts/runpod_pcl_ft.py --epochs 3

    # Only train, skip eval
    python scripts/runpod_pcl_ft.py --stage train

    # Only eval (assumes checkpoints exist)
    python scripts/runpod_pcl_ft.py --stage eval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Confidence extraction is delegated to valjson — same primitive that
# backs `valjson --confidence` on the CLI.
from valjson.confidence import analyze_confidence

SCALES: Dict[str, str] = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "32b":  "Qwen/Qwen2.5-32B-Instruct",
}
SCALE_ORDER = ["0.5b", "7b", "32b"]

RESULTS_DIR = REPO_ROOT / "results" / "margin_gating"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# Datasets supported. Each entry: dataset_key -> (schema_path, train_path,
# test_path, output_tag_prefix). The tag prefix is what gets concatenated
# with "_qwen<scale>_pcl" to form the output JSON basename. CUAD uses the
# same convention as Flights.
DATASETS: Dict[str, Tuple[Path, Path, Path, str]] = {
    "flights": (
        REPO_ROOT / "data" / "Flights_1_schema_pcl.json",
        REPO_ROOT / "data" / "Flights_1_train_pcl.jsonl",
        REPO_ROOT / "data" / "Flights_1_test_pcl.jsonl",
        "flights",
    ),
    "restaurants": (
        REPO_ROOT / "data" / "Restaurants_1_schema_pcl.json",
        REPO_ROOT / "data" / "Restaurants_1_train_pcl.jsonl",
        REPO_ROOT / "data" / "Restaurants_1_test_pcl.jsonl",
        "restaurants",
    ),
    "cuad": (
        REPO_ROOT / "data" / "cuad_schema.json",
        REPO_ROOT / "data" / "cuad_train.jsonl",
        REPO_ROOT / "data" / "cuad_test.jsonl",
        "cuad",
    ),
}


def train_scale(dataset_key: str, scale: str, model_name: str, epochs: int,
                extra_train_args: List[str], train_path: Path) -> Path:
    """Train LoRA for one (dataset, scale). Returns path to the final-epoch checkpoint.

    Skips training if `checkpoints/<dataset>_pcl_qwen<scale>/lora_epoch<epochs>/` exists.
    """
    ckpt_dir = CHECKPOINTS_DIR / f"{dataset_key}_pcl_qwen{scale}"
    final_ckpt = ckpt_dir / f"lora_epoch{epochs}"
    if final_ckpt.exists():
        print(f"[train][{dataset_key}][{scale}] SKIP — {final_ckpt} exists")
        return final_ckpt

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train][{dataset_key}][{scale}] model={model_name}, epochs={epochs}, ckpt={ckpt_dir}")
    t0 = time.time()
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "train.py"),
        "--model", model_name,
        "--data", str(train_path),
        "--device", "cuda",
        "--checkpoint-dir", str(ckpt_dir),
        "--checkpoint-prefix", "lora",
        "--epochs", str(epochs),
        "--gradient-checkpointing",
    ] + extra_train_args
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[train][{dataset_key}][{scale}] done in {time.time()-t0:.0f}s")
    return final_ckpt


def eval_scale(dataset_key: str, scale: str, model_name: str, ckpt_path: Path,
               schema_path: Path, test_path: Path, tag_prefix: str,
               tag_suffix: str = "") -> None:
    """Load base model + LoRA checkpoint and extract per-record confidence."""
    output_path = RESULTS_DIR / f"{tag_prefix}_qwen{scale}_pcl{tag_suffix}.json"
    if output_path.exists():
        print(f"[eval][{dataset_key}][{scale}] SKIP — {output_path.name} exists")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[eval][{dataset_key}][{scale}] loading base + LoRA from {ckpt_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(ckpt_path))
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    with open(schema_path) as f:
        schema = json.load(f)
    with open(test_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    enum_field_names = [k for k, p in schema.get("properties", {}).items() if "enum" in p]
    print(f"[eval][{dataset_key}][{scale}] fields: {enum_field_names}; records: {len(examples)}")

    all_fields = []
    t1 = time.time()
    for i, ex in enumerate(examples):
        prompt = ex.get("prompt", "")
        target = ex.get("target_json", "")
        example_id = str(ex.get("dialogue_id") or ex.get("id") or f"ex_{i}")
        if not prompt or not target:
            continue
        try:
            fields = analyze_confidence(
                model, tokenizer, prompt, target, schema,
                device="cuda", example_id=example_id,
            )
            for fc in fields:
                all_fields.append({
                    "example_id": fc.example_id,
                    "field":      fc.field_name,
                    "target":     fc.target_value,
                    "top":        fc.top_value,
                    "top_prob":   fc.top_prob,
                    "correct":    fc.correct,
                    "probs":      fc.probs,
                })
        except Exception as e:
            print(f"[eval][{dataset_key}][{scale}]  example {example_id} failed: {e}")
            continue
        if (i + 1) % 10 == 0 or i + 1 == len(examples):
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"[eval][{dataset_key}][{scale}]  {i+1}/{len(examples)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    total = len(all_fields)
    correct = sum(1 for f in all_fields if f["correct"])
    report = {
        "scale": scale,
        "dataset": f"{dataset_key}_pcl_ft",
        "checkpoint": str(ckpt_path),
        "total": total,
        "correct": correct,
        "fields": all_fields,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval][{dataset_key}][{scale}] WROTE {output_path}  ({total} obs, {correct} correct, {time.time()-t1:.0f}s)")

    # Free GPU before next scale.
    del model, base, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="flights",
                    help=f"Comma-separated subset of {list(DATASETS)}. Default: flights.")
    ap.add_argument("--scales", default=",".join(SCALE_ORDER),
                    help=f"Comma-separated subset of {SCALE_ORDER}. Default: all.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    ap.add_argument("--extra-train-args", default="",
                    help="Extra args passed verbatim to src/train.py (space-separated).")
    ap.add_argument("--test-data-override", default=None, type=Path,
                    help="Override the default test JSONL for eval. "
                         "Combine with --tag-suffix to keep outputs distinct. "
                         "Used to re-eval existing checkpoints against "
                         "alternative truncations (e.g. cuad_test_full.jsonl).")
    ap.add_argument("--tag-suffix", default="",
                    help="Suffix appended to eval output filenames "
                         "(e.g. '_full' → cuad_qwen32b_pcl_full.json).")
    args = ap.parse_args()

    selected_scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    unknown_scales = [s for s in selected_scales if s not in SCALES]
    if unknown_scales:
        print(f"ERROR: unknown scales {unknown_scales}", file=sys.stderr)
        return 1

    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown_datasets = [d for d in selected_datasets if d not in DATASETS]
    if unknown_datasets:
        print(f"ERROR: unknown datasets {unknown_datasets}", file=sys.stderr)
        return 1

    # Sanity-check data files up front.
    for d in selected_datasets:
        schema, train, test, _ = DATASETS[d]
        for p in (schema, train, test):
            if not p.exists():
                print(f"ERROR: missing {p}", file=sys.stderr)
                return 1

    extra_train_args = args.extra_train_args.split() if args.extra_train_args else []

    print("=" * 60)
    print("PCL fine-tune + margin-gating eval")
    print(f"  datasets: {selected_datasets}  scales: {selected_scales}  "
          f"epochs: {args.epochs}  stage: {args.stage}")
    print("=" * 60)

    for dataset_key in selected_datasets:
        schema_path, train_path, test_path, tag_prefix = DATASETS[dataset_key]
        for scale in selected_scales:
            model_name = SCALES[scale]
            print(f"\n--- {dataset_key} / {scale} ---")

            if args.stage in ("all", "train"):
                ckpt_path = train_scale(
                    dataset_key, scale, model_name, args.epochs,
                    extra_train_args, train_path,
                )
            else:
                ckpt_path = (CHECKPOINTS_DIR / f"{dataset_key}_pcl_qwen{scale}"
                             / f"lora_epoch{args.epochs}")
                if not ckpt_path.exists():
                    print(f"[eval][{dataset_key}][{scale}] ERROR — checkpoint not "
                          f"found at {ckpt_path}; run with --stage train first",
                          file=sys.stderr)
                    continue

            if args.stage in ("all", "eval"):
                effective_test = args.test_data_override or test_path
                eval_scale(dataset_key, scale, model_name, ckpt_path,
                           schema_path, effective_test, tag_prefix,
                           tag_suffix=args.tag_suffix)

    print("\nDone.")
    print("Mac-side analysis after rsync:")
    for dataset_key in selected_datasets:
        schema_path, _, test_path, tag_prefix = DATASETS[dataset_key]
        for scale in selected_scales:
            print(f"  python scripts/margin_gating_eval.py \\")
            print(f"      --schema {schema_path.relative_to(REPO_ROOT)} \\")
            print(f"      --data {test_path.relative_to(REPO_ROOT)} \\")
            print(f"      --model _ --tag {tag_prefix}_qwen{scale}_pcl --reuse-probs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
