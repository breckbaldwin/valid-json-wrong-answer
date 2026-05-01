#!/usr/bin/env python3
"""Baseline confidence extraction on RunPod for margin-gating analysis.

Runs a base Qwen model (no fine-tuning) on each test set, extracts per-record
probability distributions at every constrained-field position, and writes
valjson-`--confidence`-compatible JSON files. The outputs are meant to be
rsync'd back to the evaluation machine, where `scripts/margin_gating_eval.py
--reuse-probs` consumes them for the threshold sweep + per-field analysis.

Confidence extraction is delegated to `valjson.confidence.analyze_confidence`
(the same primitive that backs `valjson --confidence` on the CLI). The pod
must have `valjson>=2.0` installed — `requirements.txt` pins it, and
`setup_runpod.sh` installs from PyPI.

Usage (run 32B first as a smoke, then full sweep):

    # 1) Smoke — one scale, one dataset, to confirm setup works end-to-end.
    python scripts/runpod_baseline.py --scale 32b --datasets cuad

    # 2) Full 32B across all three datasets.
    python scripts/runpod_baseline.py --scale 32b

    # 3) Then 7B.
    python scripts/runpod_baseline.py --scale 7b

    # or do both back-to-back in 32B→7B order:
    python scripts/runpod_baseline.py --scale all

Outputs:
    results/margin_gating/{flights,restaurants,cuad}_qwen{7b,32b}.json

Each JSON matches `valjson --confidence --output <path>.json` format so
that `margin_gating_eval.py --reuse-probs --tag <name>_qwen<scale>` can
consume it unchanged.

Idempotent — skips datasets whose output file already exists, so the
script can be restarted after an interruption without re-running finished
datasets. Delete the specific output file to force a redo.

Transport back to the Mac (example with rsync):
    rsync -avz user@pod:/path/valid-json-wrong-answer/results/margin_gating/*.json \
        /Users/bb/forgejo/valid-json-wrong-answer/results/margin_gating/
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from valjson.confidence import analyze_confidence


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "margin_gating"


# --- Dataset catalogue ----------------------------------------------------

# Each entry: short_name -> (schema_path, data_path). Paths relative to repo
# root. Edit if your layout differs. Add entries for new datasets here; the
# rest of the script picks them up automatically.
DATASETS: Dict[str, Tuple[Path, Path]] = {
    "flights":     (REPO_ROOT / "data" / "Flights_1_schema_pcl.json",
                    REPO_ROOT / "data" / "Flights_1_test_pcl.jsonl"),
    "restaurants": (REPO_ROOT / "data" / "Restaurants_1_schema_pcl.json",
                    REPO_ROOT / "data" / "Restaurants_1_test_pcl.jsonl"),
    "cuad":        (REPO_ROOT / "data" / "cuad_schema.json",
                    REPO_ROOT / "data" / "cuad_test.jsonl"),
}

SCALES: Dict[str, str] = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "32b":  "Qwen/Qwen2.5-32B-Instruct",
}

# When --scale all is used, process 32B first (catches VRAM / config
# problems before committing the 7B run).
SCALE_ORDER = ["32b", "7b", "0.5b"]


# --- Orchestration --------------------------------------------------------

def process_dataset(
    model, tokenizer, short_name: str, schema_path: Path, data_path: Path,
    scale_label: str, device: str, max_examples: Optional[int] = None,
    tag_suffix: str = "",
) -> None:
    output_path = RESULTS_DIR / f"{short_name}_qwen{scale_label}{tag_suffix}.json"
    if output_path.exists():
        print(f"[{scale_label}][{short_name}] SKIP — output exists at {output_path.name}")
        return

    if not schema_path.exists():
        print(f"[{scale_label}][{short_name}] ERROR — schema not found: {schema_path}")
        return
    if not data_path.exists():
        print(f"[{scale_label}][{short_name}] ERROR — data not found: {data_path}")
        return

    with open(schema_path) as f:
        schema = json.load(f)
    with open(data_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if max_examples:
        examples = examples[:max_examples]

    enum_field_names = [k for k, p in schema.get("properties", {}).items() if "enum" in p]
    print(f"[{scale_label}][{short_name}] schema fields with enum: {enum_field_names}")
    print(f"[{scale_label}][{short_name}] examples: {len(examples)}")

    all_fields = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = ex.get("prompt", "")
        target = ex.get("target_json", "")
        example_id = str(ex.get("dialogue_id") or ex.get("id") or f"ex_{i}")
        if not prompt or not target:
            print(f"[{scale_label}][{short_name}]  skip example {example_id} (missing prompt/target)")
            continue
        try:
            fields = analyze_confidence(
                model, tokenizer, prompt, target, schema,
                device=device, example_id=example_id,
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
            print(f"[{scale_label}][{short_name}]  example {example_id} failed: {e}")
            continue
        if (i + 1) % 10 == 0 or i + 1 == len(examples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"[{scale_label}][{short_name}]  {i+1}/{len(examples)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    total = len(all_fields)
    correct = sum(1 for f in all_fields if f["correct"])
    report = {
        "scale": scale_label,
        "dataset": short_name,
        "total": total,
        "correct": correct,
        "fields": all_fields,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{scale_label}][{short_name}] WROTE {output_path}  ({total} field obs, {correct} correct, {time.time()-t0:.0f}s)")


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s, dtype={dtype}")
    return model, tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["0.5b", "7b", "32b", "all"], default="32b",
                    help="Model scale. 'all' runs 32B → 7B → 0.5B. Default 32B.")
    ap.add_argument("--datasets", default=",".join(DATASETS.keys()),
                    help=f"Comma-separated subset of {list(DATASETS)}. Default: all.")
    ap.add_argument("--device", default="cuda",
                    choices=["cpu", "cuda", "mps"])
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Limit examples per dataset (useful for smoke runs).")
    ap.add_argument("--test-data-override", default=None, type=Path,
                    help="Override the default test JSONL for the SELECTED datasets. "
                         "Combine with --tag-suffix to write to a distinct output file. "
                         "Used for re-evaluating against alternative truncations "
                         "(e.g. data/cuad_test_full.jsonl).")
    ap.add_argument("--tag-suffix", default="",
                    help="Suffix appended to output JSON filenames "
                         "(e.g. '_full' → cuad_qwen32b_full.json).")
    args = ap.parse_args()

    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in selected_datasets if d not in DATASETS]
    if unknown:
        print(f"ERROR: unknown dataset names {unknown}. Known: {list(DATASETS)}", file=sys.stderr)
        return 1

    scales = [args.scale] if args.scale != "all" else SCALE_ORDER

    print("=" * 60)
    print("RunPod baseline confidence extraction")
    print(f"  scales:   {scales}")
    print(f"  datasets: {selected_datasets}")
    print(f"  device:   {args.device}")
    print(f"  output:   {RESULTS_DIR}")
    print("=" * 60)

    for scale in scales:
        model_name = SCALES[scale]
        # Skip model load if every output for this scale already exists.
        all_done = all(
            (RESULTS_DIR / f"{d}_qwen{scale}{args.tag_suffix}.json").exists()
            for d in selected_datasets
        )
        if all_done:
            print(f"[{scale}] all outputs exist; skipping model load")
            continue

        model, tokenizer = load_model(model_name, args.device)
        try:
            for d in selected_datasets:
                schema_path, default_data_path = DATASETS[d]
                data_path = args.test_data_override or default_data_path
                process_dataset(model, tokenizer, d, schema_path, data_path,
                                scale, args.device, args.max_examples,
                                tag_suffix=args.tag_suffix)
        finally:
            del model, tokenizer
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\nDone.")
    print("To analyze on the Mac side after rsync:")
    for d in selected_datasets:
        for scale in scales:
            tag = f"{d}_qwen{scale}"
            schema, data = DATASETS[d]
            print(f"  python scripts/margin_gating_eval.py \\")
            print(f"      --schema {schema.relative_to(REPO_ROOT)} \\")
            print(f"      --data {data.relative_to(REPO_ROOT)} \\")
            print(f"      --model _ --device _ --tag {tag} --reuse-probs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
