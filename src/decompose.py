#!/usr/bin/env python3
"""Per-grammar-role loss decomposition for structured JSON output.

Thin CLI wrapper around `valjson.decompose.decompose_loss_batch` (the same
primitive that backs the published `valjson` CLI). The role-assignment
logic and teacher-forced loss math live in valjson — this script exists
only to preserve the invocation surface used by `scripts/runpod_*.py` and
`scripts/run_all_paper.sh`, and to write the JSON shape that
`scripts/build_tables.py` consumes.


Usage:
    # Baseline
    python src/decompose.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --schema data/Restaurants_1_schema.json \\
        --data   data/Restaurants_1_test.jsonl \\
        --output results/05b_baseline_restaurants.json

    # Fine-tuned (LoRA)
    python src/decompose.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --checkpoint checkpoints/restaurants_05b/lora_epoch10 \\
        --schema data/Restaurants_1_schema.json \\
        --data   data/Restaurants_1_test.jsonl \\
        --output results/05b_finetuned_restaurants.json
"""

import argparse
import json
import sys

from valjson.decompose import decompose_loss_batch
from valjson.report import load_model

PER_ROLE_ORDER = [
    "STRUCTURAL", "QUOTE", "KEY", "ENUM_VALUE", "BOOLEAN",
    "NUMBER", "FREE_TEXT", "WHITESPACE", "UNKNOWN",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Per-grammar-role loss decomposition (valjson)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    p.add_argument("--data", required=True, help="JSONL data file")
    p.add_argument("--schema", required=True, help="JSON Schema file")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output", default=None, help="Output JSON file")
    args = p.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'none (baseline)'}")
    print(f"Examples:   {len(examples)}")
    print(f"Device:     {args.device}")

    model, tokenizer = load_model(args.model, args.checkpoint, args.device)

    pairs = [(ex["prompt"], ex["target_json"]) for ex in examples]
    result = decompose_loss_batch(
        model, tokenizer, pairs, schema, args.device, verbose=True,
    )

    print()
    print("=" * 60)
    print("Per-Grammar-Role Loss Decomposition")
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'baseline (no fine-tuning)'}")
    print(f"Examples:   {len(examples)}")
    print("=" * 60)
    print(f"{'Role':<15} {'Mean Loss':>10} {'Tokens':>8}")
    print("-" * 35)
    for role_name in PER_ROLE_ORDER:
        if role_name not in result.per_role:
            continue
        stats = result.per_role[role_name]
        print(f"{role_name:<15} {stats.mean_loss:>10.4f} {stats.token_count:>8}")
    print("-" * 35)
    print(f"{'TOTAL':<15} {result.mean_loss:>10.4f} {result.total_tokens:>8}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "checkpoint": args.checkpoint,
                    "num_examples": len(examples),
                    "per_role": {
                        role: {"mean_loss": stats.mean_loss, "count": stats.token_count}
                        for role, stats in result.per_role.items()
                    },
                    "total_mean_loss": result.mean_loss,
                    "total_tokens": result.total_tokens,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
