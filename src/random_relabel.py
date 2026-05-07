#!/usr/bin/env python3
"""Generate a randomly-relabeled training set as a control for PCL.

Picks N examples uniformly at random and relabels their target field to
'ambiguous', where N matches the rate at which the cue-driven PCL relabels
the same training set. The point of the experiment is to ask: does
PCL's improvement come from the cue-based relabeling specifically, or
would any abstain-class introduction at the same rate work as well?

Rate 0.772 = 193/250 matches the cue-driven PCL relabel rate for the
Flights schema with cue='refund'. Seed pinned for reproducibility.
"""
import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field", required=True,
                    help="Field name to relabel to 'ambiguous'.")
    ap.add_argument("--rate", type=float, required=True,
                    help="Fraction of training examples to relabel.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.is_file():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    records = [json.loads(line) for line in open(in_path)]
    n_total = len(records)
    n_relabel = round(args.rate * n_total)

    # Sample without replacement; deterministic given seed.
    indices = list(range(n_total))
    rng.shuffle(indices)
    relabel_set = set(indices[:n_relabel])

    n_actually_relabeled = 0
    with open(out_path, "w") as f:
        for i, rec in enumerate(records):
            if i in relabel_set:
                tj = rec["target_json"]
                target = json.loads(tj) if isinstance(tj, str) else tj
                if args.field in target:
                    target[args.field] = "ambiguous"
                    n_actually_relabeled += 1
                rec["target_json"] = (
                    json.dumps(target, indent=2) if isinstance(tj, str) else target
                )
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {n_total} records to {out_path}")
    print(f"Randomly relabeled {n_actually_relabeled}/{n_total} = "
          f"{n_actually_relabeled/n_total*100:.1f}% to 'ambiguous' "
          f"(target rate {args.rate*100:.1f}%, seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
