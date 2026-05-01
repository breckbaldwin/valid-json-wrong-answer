#!/usr/bin/env python3
"""Evaluate inference-time margin gating on a JSON-extraction test set.

Produces the per-class margin statistics and threshold sweep that back
the paper's "inference-time evidential gating" result. Reusable across
(schema, test set, model, scale) tuples — one invocation = one row in
the eventual scale × schema grid.

Pipeline:
    valjson --confidence   →   probs JSON
    probs JSON             →   threshold sweep + RESULTS.md section

Per-record margin and per-threshold commit/abstain decisions are delegated
to `valjson.gate.gate_record` (the same primitive that backs the
`valjson --gate` CLI), so this script stays in lockstep with the public
tool. The threshold sweep is the paper-specific layer on top.

Usage (single run, all enum fields in the schema — default):
    python scripts/margin_gating_eval.py \\
        --schema /path/to/schema.json \\
        --data   /path/to/test.jsonl \\
        --model  Qwen/Qwen2.5-0.5B-Instruct \\
        --device mps \\
        --tag    flights_qwen05b

Limit to specific fields:
    ... --fields refundable,seating_class

Override the abstain target (auto-detected from schema by default):
    ... --abstain-target not_specified

Requires the venv active, see README.md for setup:
    source .venv/bin/activate

Outputs (at RESULTS_DIR=results/margin_gating by default):
    <tag>.json         — raw per-record probability distributions (valjson --confidence output)
    <tag>.md           — multi-field results section for this run
    RESULTS.md         — aggregate across every <tag>.md present, regenerated each run

Abstain-target auto-detection: each field's enum is searched for any
sentinel in AUTO_ABSTAIN_CANDIDATES (default: "ambiguous", "not_specified").
Fields without a matching sentinel are reported as pure commit/coverage
sweeps (no abstention-rate column).
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from valjson.gate import gate_record, COMMIT, ABSTAIN

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "margin_gating"

# Threshold grid for the sweep. Edit here to change every downstream table.
THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# Enum values that, if present in a field's schema, are treated as the
# field's abstain-target (the analog of "ambiguous" from PCL_MOTIVATION).
# First match wins. Add new candidates here if a new schema uses a
# different sentinel name. An explicit --abstain-target CLI flag overrides.
AUTO_ABSTAIN_CANDIDATES = ("ambiguous", "not_specified")


def run_confidence(
    schema: Path, data: Path, model: str, device: str,
    max_examples: int, probs_path: Path, verbose: bool,
) -> None:
    """Call `valjson --confidence` and cache the probability distribution JSON."""
    probs_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "valjson", "--confidence", str(probs_path),
        "--schema", str(schema),
        "--data", str(data),
        "--model", model,
        "--device", device,
        "--max-examples", str(max_examples),
        "--quiet",
    ]
    if verbose:
        print(f"[confidence] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def load_field_records(probs_path: Path, field_name: str) -> List[Dict]:
    """Pull per-record probability distributions for one field from the
    valjson --confidence output.

    Returns: list of dicts with keys {id, target, probs}. `probs` is the
    full distribution over allowed values (zero-filled for missing values).
    """
    with open(probs_path) as f:
        conf = json.load(f)
    records = []
    for fc in conf["fields"]:
        if fc["field"] != field_name:
            continue
        records.append({
            "id": fc["example_id"],
            "target": fc["target"],
            "probs": fc["probs"],
        })
    return records


def compute_margins(records: List[Dict], field_name: str) -> List[Dict]:
    """Attach margin (top − second) and top value to each record.

    Uses `valjson.gate.gate_record` so the margin formula is shared
    with `valjson --gate` and stays in sync with the published tool.
    Threshold is set to 0 here because we only need the margin/top;
    the per-threshold commit/abstain decisions are made downstream in
    `threshold_sweep` (which also calls `gate_record`).
    """
    for r in records:
        gates = gate_record({field_name: r["probs"]}, threshold=0.0)
        fg = gates[0]
        r["top"] = fg.top_value
        r["top_prob"] = fg.top_prob
        r["margin"] = fg.margin
    return records


def per_class_margin_stats(records: List[Dict]) -> List[Dict]:
    """Margin mean/range per gold-target class."""
    by_target: Dict[str, List[float]] = {}
    for r in records:
        by_target.setdefault(r["target"], []).append(r["margin"])
    rows = []
    for target, margins in sorted(by_target.items()):
        rows.append({
            "target": target,
            "n": len(margins),
            "mean": sum(margins) / len(margins),
            "min": min(margins),
            "max": max(margins),
        })
    return rows


def threshold_sweep(records: List[Dict], field_name: str, abstain_target: Optional[str]) -> List[Dict]:
    """For each threshold: commit/abstain counts, committed accuracy, and
    — if `abstain_target` is set — abstention rate on records whose gold
    equals the abstain target.

    For fields without an abstain target (pure boolean, no natural
    underspecified class), the abstention-rate column is reported as
    `None`; coverage and committed accuracy are always reported.

    Per-record commit/abstain decisions come from `valjson.gate.gate_record`
    so the threshold semantics are identical to `valjson --gate`.
    """
    discussed = [r for r in records if r["target"] != abstain_target]
    undiscussed = [r for r in records if r["target"] == abstain_target] if abstain_target else []
    n_disc = len(discussed)
    n_undisc = len(undiscussed)

    rows = []
    for thr in THRESHOLDS:
        commits, abstains = [], []
        for r in records:
            decision = gate_record({field_name: r["probs"]}, threshold=thr)[0].decision
            (commits if decision == COMMIT else abstains).append(r)

        disc_commits = [r for r in commits if r["target"] != abstain_target]
        disc_correct = sum(1 for r in disc_commits if r["top"] == r["target"])
        commit_acc = (disc_correct / len(disc_commits)) if disc_commits else 0.0
        retained_on_disc = (disc_correct / n_disc) if n_disc else 0.0

        if abstain_target is not None and n_undisc > 0:
            undisc_abstained = sum(1 for r in abstains if r["target"] == abstain_target)
            abstain_rate = undisc_abstained / n_undisc
        else:
            abstain_rate = None

        rows.append({
            "threshold": thr,
            "commits": len(commits),
            "abstains": len(abstains),
            "disc_commits": len(disc_commits),
            "commit_acc": commit_acc,
            "retained_on_disc": retained_on_disc,
            "undisc_abstain_rate": abstain_rate,
        })
    return rows


def auto_abstain_target(enum_values: List[str]) -> Optional[str]:
    """Pick the field's abstain-target from its enum if any sentinel is present."""
    for candidate in AUTO_ABSTAIN_CANDIDATES:
        if candidate in enum_values:
            return candidate
    return None


def field_role(enum_values: List[str]) -> str:
    """Short label for the field's grammar role, used in summary tables."""
    if set(enum_values) == {"True", "False"}:
        return "BOOL"
    if enum_values:
        return f"ENUM{len(enum_values)}"
    return "STRING"


def iter_schema_enum_fields(schema: dict):
    """Yield (name, enum_values) for every property with an enum."""
    for name, prop in (schema.get("properties") or {}).items():
        vals = prop.get("enum")
        if vals:
            yield name, list(vals)


def _render_sweep_table(sweep: List[Dict], abstain_target: Optional[str]) -> List[str]:
    lines = []
    has_abstain = abstain_target is not None and any(r["undisc_abstain_rate"] is not None for r in sweep)
    if has_abstain:
        lines.append(f"`retained_on_disc = disc_correct / total_gold_specified`  ·  "
                     f"`undisc_abstain = correct_abstention_rate_on_gold_{abstain_target}`")
        lines.append("")
        lines.append("| threshold | commit | abstain | spec_commits | commit_acc | retained_on_spec | undisc_abstain |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in sweep:
            ua = f"{r['undisc_abstain_rate']*100:.0f}%" if r["undisc_abstain_rate"] is not None else "—"
            lines.append(
                f"| {r['threshold']:.2f} | {r['commits']} | {r['abstains']} | "
                f"{r['disc_commits']} | {r['commit_acc']*100:.0f}% | "
                f"{r['retained_on_disc']*100:.0f}% | {ua} |"
            )
    else:
        lines.append("No natural abstain-target in this field's enum; table shows coverage × committed accuracy only.")
        lines.append("")
        lines.append("| threshold | commit | abstain | commit_acc |")
        lines.append("|---:|---:|---:|---:|")
        for r in sweep:
            lines.append(
                f"| {r['threshold']:.2f} | {r['commits']} | {r['abstains']} | "
                f"{r['commit_acc']*100:.0f}% |"
            )
    lines.append("")
    return lines


def render_field_subsection(
    field_name: str, role: str, abstain_target: Optional[str],
    records: List[Dict], class_stats: List[Dict], sweep: List[Dict],
) -> List[str]:
    lines = []
    baseline_acc = sum(1 for r in records if r["top"] == r["target"]) / len(records) if records else 0.0
    mean_margin = sum(r["margin"] for r in records) / len(records) if records else 0.0
    lines.append(f"### {field_name}")
    lines.append("")
    abst = f"`{abstain_target}`" if abstain_target else "none"
    lines.append(f"- role: `{role}` · abstain target: {abst} · n: {len(records)} · "
                 f"baseline acc: {baseline_acc*100:.0f}% · mean margin: {mean_margin:.3f}")
    lines.append("")
    lines.append("Margin by gold class:")
    lines.append("")
    lines.append("| target | n | mean margin | range |")
    lines.append("|---|---:|---:|---|")
    for s in class_stats:
        lines.append(
            f"| `{s['target']}` | {s['n']} | {s['mean']:.3f} | "
            f"{s['min']:.3f} – {s['max']:.3f} |"
        )
    lines.append("")
    lines.append("Threshold sweep:")
    lines.append("")
    lines.extend(_render_sweep_table(sweep, abstain_target))
    return lines


def render_section(
    tag: str, schema_path: Path, data_path: Path, model: str,
    field_reports: List[Dict],
) -> str:
    """Render a multi-field section. `field_reports` is a list of dicts
    with keys {name, role, abstain_target, records, class_stats, sweep}.
    """
    lines = []
    lines.append(f"## {tag}")
    lines.append("")
    lines.append(f"- **Schema:** `{schema_path}`")
    lines.append(f"- **Data:** `{data_path}`")
    lines.append(f"- **Model:** `{model}`")
    fields_line = ", ".join(f"`{f['name']}`" for f in field_reports)
    lines.append(f"- **Fields gated ({len(field_reports)}):** {fields_line}")
    lines.append("")

    # Cross-field summary
    lines.append("### Per-field baseline summary")
    lines.append("")
    lines.append("| Field | Role | abstain target | n | baseline acc | mean margin |")
    lines.append("|---|---|---|---:|---:|---:|")
    for f in field_reports:
        baseline_acc = (
            sum(1 for r in f["records"] if r["top"] == r["target"]) / len(f["records"])
            if f["records"] else 0.0
        )
        mean_margin = (
            sum(r["margin"] for r in f["records"]) / len(f["records"])
            if f["records"] else 0.0
        )
        abst = f"`{f['abstain_target']}`" if f["abstain_target"] else "—"
        lines.append(
            f"| `{f['name']}` | {f['role']} | {abst} | {len(f['records'])} | "
            f"{baseline_acc*100:.0f}% | {mean_margin:.3f} |"
        )
    lines.append("")

    # Per-field detail
    for f in field_reports:
        lines.extend(render_field_subsection(
            field_name=f["name"], role=f["role"],
            abstain_target=f["abstain_target"], records=f["records"],
            class_stats=f["class_stats"], sweep=f["sweep"],
        ))
    return "\n".join(lines)


def regenerate_index(results_dir: Path) -> None:
    """Rebuild RESULTS.md by concatenating every <tag>.md present."""
    sections = sorted(results_dir.glob("*.md"))
    sections = [s for s in sections if s.name != "RESULTS.md"]
    header = [
        "# Margin Gating Results",
        "",
        "Generated by `scripts/margin_gating_eval.py`. Each section below "
        "is one `(schema, test set, model)` run; runs cover every enum "
        "field in the schema by default. Re-running the script refreshes "
        "that run's section and regenerates this file.",
        "",
        "Reproduce any run:",
        "",
        "```bash",
        "source valid-json-wrong-answer/.venv/bin/activate",
        "python scripts/margin_gating_eval.py \\",
        "    --schema <schema.json> --data <test.jsonl> \\",
        "    --model <hf-name> --device mps \\",
        "    --tag <tag>",
        "```",
        "",
        "Optional: `--fields F1,F2` to limit, `--abstain-target X` to override "
        "the auto-detected sentinel, `--reuse-probs` to skip the model forward pass.",
        "",
    ]
    body = []
    for s in sections:
        body.append(s.read_text())
        body.append("")
    (results_dir / "RESULTS.md").write_text("\n".join(header + body) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--fields", default="ALL",
                    help="Comma-separated field names to gate on, or 'ALL' "
                         "(default) to iterate every enum field in the schema.")
    ap.add_argument("--abstain-target", default=None,
                    help="Enum value that means 'abstain is correct' for ALL "
                         "selected fields. If unset, auto-detect per field "
                         f"from {list(AUTO_ABSTAIN_CANDIDATES)}.")
    ap.add_argument("--tag", required=True, help="Short label for this run, e.g. flights_qwen05b.")
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--reuse-probs", action="store_true",
                    help="Skip the --confidence call and reuse existing <tag>.json.")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    probs_path = results_dir / f"{args.tag}.json"
    section_path = results_dir / f"{args.tag}.md"

    if args.reuse_probs:
        if not probs_path.exists():
            print(f"ERROR: --reuse-probs set but cache not found at {probs_path}.\n"
                  f"       Either drop --reuse-probs to compute it, or stage the "
                  f"file first (e.g. rsync from the run machine).",
                  file=sys.stderr)
            sys.exit(2)
    else:
        run_confidence(
            schema=args.schema, data=args.data,
            model=args.model, device=args.device,
            max_examples=args.max_examples,
            probs_path=probs_path, verbose=True,
        )

    with open(args.schema) as f:
        schema = json.load(f)
    schema_fields = dict(iter_schema_enum_fields(schema))

    if args.fields.strip().upper() == "ALL":
        selected = list(schema_fields.keys())
    else:
        selected = [s.strip() for s in args.fields.split(",") if s.strip()]
        unknown = [s for s in selected if s not in schema_fields]
        if unknown:
            print(f"ERROR: unknown field(s) {unknown} not in schema", file=sys.stderr)
            sys.exit(1)

    field_reports = []
    for name in selected:
        enum_values = schema_fields[name]
        role = field_role(enum_values)
        abstain_t = args.abstain_target if args.abstain_target else auto_abstain_target(enum_values)
        records = compute_margins(load_field_records(probs_path, name), field_name=name)
        if not records:
            print(f"WARN: no records for field '{name}'; skipping", file=sys.stderr)
            continue
        class_stats = per_class_margin_stats(records)
        sweep = threshold_sweep(records, field_name=name, abstain_target=abstain_t)
        field_reports.append({
            "name": name,
            "role": role,
            "abstain_target": abstain_t,
            "records": records,
            "class_stats": class_stats,
            "sweep": sweep,
        })

    if not field_reports:
        print("ERROR: no fields produced any data", file=sys.stderr)
        sys.exit(1)

    section = render_section(
        tag=args.tag, schema_path=args.schema, data_path=args.data,
        model=args.model, field_reports=field_reports,
    )
    section_path.write_text(section + "\n")
    print(f"Wrote section: {section_path}")

    regenerate_index(results_dir)
    print(f"Rebuilt:       {results_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
