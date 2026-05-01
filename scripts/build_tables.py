#!/usr/bin/env python3
"""Generate every table in paper/main.tex from the result artefacts.

Each table maps to a function below; each function reads result JSONs (or
training data) and prints both a console-friendly preview and the LaTeX
body to paste between \\begin{tabular}...\\end{tabular} in main.tex.

Usage:
    python scripts/build_tables.py                     # build all tables
    python scripts/build_tables.py --table aggregate   # one table by name

Sources by table (label in main.tex → input):

  tab:aggregate            results/<scale>_<cond>_<tag>.json (total_mean_loss)
  tab:flights_decomp       results/<scale>_<cond>_flights.json (per_role)
  fig:scaling              results/<scale>_<cond>_flights.json (per_role.BOOLEAN)
  tab:enum_cuad            results/<scale>_<cond>_cuad.json (per_role.ENUM_VALUE)
  tab:restaurants_decomp   results/<scale>_<cond>_restaurants.json (per_role)
  tab:cuad_per_field_scale results/margin_gating/cuad_qwen<scale>.json (--confidence)
  tab:lexical              data/Flights_1_train.jsonl + has_lexical_cue("refund")
  tab:pcl                  results/<scale>_pcl_finetuned_flights.json (per_role.BOOLEAN)

If an input file is missing, the function reports MISSING and continues.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCALES = [("0.5B", "05b"), ("7B", "7b"), ("32B", "32b")]
PER_ROLE_ORDER = [
    ("STRUCT", "STRUCTURAL"),
    ("QUOTE",  "QUOTE"),
    ("KEY",    "KEY"),
    ("ENUM",   "ENUM_VALUE"),
    ("BOOL",   "BOOLEAN"),
    ("FREE",   "FREE_TEXT"),
    ("WS",     "WHITESPACE"),
]


def _load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"  WARN MISSING: {path.relative_to(REPO)}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def _section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def _signed(pct: float) -> str:
    return f"$-${abs(pct):.0f}\\%" if pct < 0 else f"+{pct:.0f}\\%"


# =============================================================================
# Table 1: aggregate loss across schemas × scales
# =============================================================================
def build_aggregate(results_dir: Path) -> None:
    _section("Table 1 (tab:aggregate) — aggregate mean loss")
    schemas = [("Rest.", "restaurants"), ("Flights", "flights"), ("CUAD", "cuad")]

    rows = []
    for disp, tag in schemas:
        for disp_scale, scale in SCALES:
            b = _load(results_dir / f"{scale}_baseline_{tag}.json")
            f = _load(results_dir / f"{scale}_finetuned_{tag}.json")
            if not (b and f):
                continue
            base = b["total_mean_loss"]
            ft = f["total_mean_loss"]
            rows.append((disp, disp_scale, base, ft, (ft - base) / base * 100))

    print(f"\n{'Schema':<10} {'Scale':<6} {'Base':>8} {'FT':>8} {'Change':>9}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<6} {r[2]:>8.3f} {r[3]:>8.3f} {r[4]:>+8.0f}%")

    print("\n% --- LaTeX body for tab:aggregate ---")
    print("\\toprule")
    print("\\textbf{Schema} & \\textbf{Scale} & \\textbf{Base} & \\textbf{FT} & \\textbf{Change} \\\\")
    print("\\midrule")
    last = None
    for sch, scl, b, f, ch in rows:
        if sch != last and last is not None:
            print("\\midrule")
        head = f"\\multirow{{3}}{{*}}{{{sch}}} & {scl}" if sch != last else f"                        & {scl}"
        print(f"{head} & {b:.3f} & {f:.3f} & {_signed(ch)} \\\\")
        last = sch
    print("\\bottomrule")


# =============================================================================
# Per-role decomposition tables (Flights, Restaurants)
# =============================================================================
def _per_role_table(results_dir: Path, tag: str, label: str) -> None:
    rows = {}
    for _, scale in SCALES:
        b = _load(results_dir / f"{scale}_baseline_{tag}.json")
        f = _load(results_dir / f"{scale}_finetuned_{tag}.json")
        rows[scale] = (b, f)

    print(f"\n{'Role':<6}", end="")
    for ds, _ in SCALES:
        print(f"  {ds+' Base':>10} {ds+' FT':>10}", end="")
    print()
    for short, key in PER_ROLE_ORDER:
        line = f"{short:<6}"
        for _, scale in SCALES:
            b, f = rows[scale]
            bv = (b or {}).get("per_role", {}).get(key, {}).get("mean_loss")
            fv = (f or {}).get("per_role", {}).get(key, {}).get("mean_loss")
            line += f"  {bv if bv is None else f'{bv:>10.3f}'} {fv if fv is None else f'{fv:>10.3f}'}"
        print(line)
    line = f"{'TOTAL':<6}"
    for _, scale in SCALES:
        b, f = rows[scale]
        line += f"  {(b or {}).get('total_mean_loss', float('nan')):>10.3f} {(f or {}).get('total_mean_loss', float('nan')):>10.3f}"
    print(line)

    print(f"\n% --- LaTeX body for {label} ---")
    print("\\toprule")
    print(" & \\multicolumn{2}{c}{\\textbf{0.5B}} & \\multicolumn{2}{c}{\\textbf{7B}} & \\multicolumn{2}{c}{\\textbf{32B}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    print("\\textbf{Role} & Base & FT & Base & FT & Base & FT \\\\")
    print("\\midrule")
    for short, key in PER_ROLE_ORDER:
        cells = []
        for _, scale in SCALES:
            b, f = rows[scale]
            bv = (b or {}).get("per_role", {}).get(key, {}).get("mean_loss", 0.0)
            fv = (f or {}).get("per_role", {}).get(key, {}).get("mean_loss", 0.0)
            cells.append(f"{bv:.2f}")
            cells.append(f"{fv:.2f}")
        # Highlight the boolean regression at 32B
        if short == "BOOL" and tag == "flights":
            ft32 = (rows["32b"][1] or {}).get("per_role", {}).get("BOOLEAN", {}).get("mean_loss", 0.0)
            base32 = (rows["32b"][0] or {}).get("per_role", {}).get("BOOLEAN", {}).get("mean_loss", 1.0)
            if ft32 > base32:
                cells[-1] = f"\\regress{{{ft32:.2f}}}"
            print(f"\\textbf{{{short}}} & " + " & ".join(f"\\textbf{{{c}}}" if i < 4 else c for i, c in enumerate(cells)) + " \\\\")
        else:
            print(f"{short:<7} & " + " & ".join(cells) + " \\\\")
    print("\\midrule")
    totals = []
    for _, scale in SCALES:
        b, f = rows[scale]
        totals.append(f"{(b or {}).get('total_mean_loss', 0.0):.2f}")
        totals.append(f"{(f or {}).get('total_mean_loss', 0.0):.2f}")
    print("TOTAL   & " + " & ".join(totals) + " \\\\")
    print("\\bottomrule")


def build_flights_decomp(results_dir: Path) -> None:
    _section("Table 2 (tab:flights_decomp) — Flights per-role loss")
    _per_role_table(results_dir, tag="flights", label="tab:flights_decomp")


def build_restaurants_decomp(results_dir: Path) -> None:
    _section("Table 5 (tab:restaurants_decomp) — Restaurants per-role loss")
    _per_role_table(results_dir, tag="restaurants", label="tab:restaurants_decomp")


# =============================================================================
# Single-role trend tables (Flights BOOL, CUAD ENUM)
# =============================================================================
def _single_role_trend(results_dir: Path, tag: str, role: str, label: str, regress_at: str | None = None) -> None:
    rows = []
    for disp_scale, scale in SCALES:
        b = _load(results_dir / f"{scale}_baseline_{tag}.json")
        f = _load(results_dir / f"{scale}_finetuned_{tag}.json")
        if not (b and f):
            continue
        bv = b["per_role"][role]["mean_loss"]
        fv = f["per_role"][role]["mean_loss"]
        rows.append((disp_scale, scale, bv, fv, (fv - bv) / bv * 100))

    print(f"\n{'Scale':<6} {'Base':>8} {'FT':>8} {'Change':>9}")
    for r in rows:
        print(f"{r[0]:<6} {r[2]:>8.3f} {r[3]:>8.3f} {r[4]:>+8.0f}%")

    print(f"\n% --- LaTeX body for {label} ---")
    print("\\toprule")
    print("\\textbf{Scale} & \\textbf{Base} & \\textbf{FT} & \\textbf{Change} \\\\")
    print("\\midrule")
    for disp_scale, scale, bv, fv, ch in rows:
        ft_cell = f"\\regress{{{fv:.3f}}}" if regress_at == scale else f"{fv:.3f}"
        ch_cell = f"\\regress{{{_signed(ch)}}}" if (regress_at == scale or (regress_at == "all" and ch > 0)) else _signed(ch)
        print(f"{disp_scale} & {bv:.3f} & {ft_cell} & {ch_cell} \\\\")
    print("\\bottomrule")


def build_boolean_trend(results_dir: Path) -> None:
    _section("Table 3 (fig:scaling) — Flights BOOLEAN trend across scales")
    _single_role_trend(results_dir, tag="flights", role="BOOLEAN",
                       label="fig:scaling", regress_at="32b")


def build_enum_cuad(results_dir: Path) -> None:
    _section("Table 4 (tab:enum_cuad) — CUAD ENUM_VALUE trend across scales")
    _single_role_trend(results_dir, tag="cuad", role="ENUM_VALUE",
                       label="tab:enum_cuad", regress_at="all")


# =============================================================================
# Table 6: CUAD per-field accuracy + mean margin (from --confidence outputs)
# =============================================================================
CUAD_FIELDS_DISPLAY = [
    "has_most_favored_nation",
    "has_liquidated_damages",
    "has_exclusivity",
    "has_anti_assignment",
    "governing_law",
    "renewal_term",
    "expiration_type",
]


def build_cuad_per_field(results_dir: Path) -> None:
    _section("Table 6 (tab:cuad_per_field_scale) — CUAD per-field baseline accuracy + margin")
    margin_dir = results_dir / "margin_gating"

    per_scale: dict[str, dict[str, dict[str, float]]] = {}
    for _, scale in SCALES:
        # try the canonical filename pattern
        candidates = [
            margin_dir / f"cuad_qwen{scale.replace('b', 'b')}.json",
            margin_dir / f"cuad_qwen{scale[:-1]}b.json",
            margin_dir / f"cuad_qwen0.5b.json" if scale == "05b" else None,
        ]
        d = None
        for c in candidates:
            if c and c.is_file():
                d = json.load(open(c))
                break
        if not d:
            print(f"  WARN MISSING confidence file for CUAD {scale}", file=sys.stderr)
            continue

        by_field: dict[str, dict[str, float | int]] = defaultdict(lambda: {"correct": 0, "total": 0, "margin_sum": 0.0})
        for fc in d["fields"]:
            field = fc["field"]
            target = fc["target"]
            probs = fc["probs"]
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            top_value, top_p = ranked[0]
            second_p = ranked[1][1] if len(ranked) > 1 else 0.0
            by_field[field]["total"] += 1
            by_field[field]["correct"] += int(top_value == target)
            by_field[field]["margin_sum"] += top_p - second_p
        per_scale[scale] = {
            f: {"acc": v["correct"] / v["total"] if v["total"] else 0.0,
                "margin": v["margin_sum"] / v["total"] if v["total"] else 0.0}
            for f, v in by_field.items()
        }

    if not per_scale:
        return

    fields_present = sorted({f for d in per_scale.values() for f in d})
    fields_to_show = [f for f in CUAD_FIELDS_DISPLAY if f in fields_present]

    print(f"\n{'Field':<28} {'0.5B':>14} {'7B':>14} {'32B':>14}")
    for fld in fields_to_show:
        line = f"{fld:<28}"
        for _, scale in SCALES:
            d = per_scale.get(scale, {}).get(fld)
            if d:
                line += f"  {d['acc']*100:>4.0f}% ({d['margin']:.2f})"
            else:
                line += "       -      "
        print(line)

    print("\n% --- LaTeX body for tab:cuad_per_field_scale ---")
    print("\\toprule")
    print("\\textbf{Field} & \\textbf{0.5B} & \\textbf{7B} & \\textbf{32B} \\\\")
    print("\\midrule")
    for fld in fields_to_show:
        cells = []
        for _, scale in SCALES:
            d = per_scale.get(scale, {}).get(fld)
            if d:
                cells.append(f"{d['acc']*100:.0f}\\% ({d['margin']:.2f})")
            else:
                cells.append("---")
        latex_field = fld.replace("_", "\\_")
        print(f"\\texttt{{{latex_field}}} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")


# =============================================================================
# Table 7: lexical analysis on Flights training data
# =============================================================================
def build_lexical(_results_dir: Path) -> None:
    _section("Table 7 (tab:lexical) — refundable conditional on lexical cue 'refund'")
    from src.presupposition_label import has_lexical_cue

    train_path = REPO / "data" / "Flights_1_train.jsonl"
    if not train_path.is_file():
        print(f"  WARN MISSING: {train_path.relative_to(REPO)}", file=sys.stderr)
        return

    cue = "refund"
    discussed = {"True": 0, "False": 0}
    not_discussed = {"True": 0, "False": 0}
    n_total = 0
    for line in open(train_path):
        rec = json.loads(line)
        target = json.loads(rec["target_json"]) if isinstance(rec.get("target_json"), str) else rec.get("target_json") or {}
        ref = target.get("refundable")
        if ref not in ("True", "False"):
            continue
        n_total += 1
        bucket = discussed if has_lexical_cue(rec["prompt"], cue) else not_discussed
        bucket[ref] += 1

    n_disc = discussed["True"] + discussed["False"]
    n_undisc = not_discussed["True"] + not_discussed["False"]

    def pct(part: int, whole: int) -> str:
        return f"{part / whole * 100:.0f}\\%" if whole else "---"

    print(f"\n{'Context':<50} {'True':>8} {'False':>8}")
    print(f"{'Discussed (' + str(n_disc) + ' examples)':<50} "
          f"{pct(discussed['True'], n_disc):>8} {pct(discussed['False'], n_disc):>8}")
    print(f"{'Not discussed (' + str(n_undisc) + ' examples)':<50} "
          f"{pct(not_discussed['True'], n_undisc):>8} {pct(not_discussed['False'], n_undisc):>8}")
    print(f"\nTotal examples: {n_total}")

    print("\n% --- LaTeX body for tab:lexical ---")
    print("\\toprule")
    print("\\textbf{Context} & \\textbf{True} & \\textbf{False} \\\\")
    print("\\midrule")
    print(f"Refundability discussed ({n_disc} examples) & "
          f"{pct(discussed['True'], n_disc)} & {pct(discussed['False'], n_disc)} \\\\")
    print(f"Not discussed ({n_undisc} examples) & "
          f"{pct(not_discussed['True'], n_undisc)} & {pct(not_discussed['False'], n_undisc)} \\\\")
    print("\\bottomrule")


# =============================================================================
# Table 8: PCL with vs without (ENUM/BOOLEAN loss)
# =============================================================================
def build_pcl(results_dir: Path) -> None:
    _section("Table 8 (tab:pcl) — Flights BOOLEAN loss with vs without PCL")
    print("Reads:")
    print("  baseline (2-way std LoRA): results/<scale>_finetuned_flights.json (BOOLEAN)")
    print("  PCL (3-way std LoRA):      results/<scale>_pcl_finetuned_flights.json (BOOLEAN)")
    print()

    rows = []
    for disp_scale, scale in SCALES:
        f2 = _load(results_dir / f"{scale}_finetuned_flights.json")
        f3 = _load(results_dir / f"{scale}_pcl_finetuned_flights.json")
        v2 = f2 and f2["per_role"]["BOOLEAN"]["mean_loss"]
        v3 = f3 and f3["per_role"]["BOOLEAN"]["mean_loss"]
        rows.append((disp_scale, v2, v3))

    print(f"\n{'Scale':<6} {'2-way':>10} {'3-way (PCL)':>14} {'Change':>9}")
    for ds, v2, v3 in rows:
        if v2 is None or v3 is None:
            print(f"{ds:<6} {v2 if v2 is None else f'{v2:>10.3f}'} "
                  f"{v3 if v3 is None else f'{v3:>14.3f}'}     n/a")
        else:
            ch = (v3 - v2) / v2 * 100
            print(f"{ds:<6} {v2:>10.3f} {v3:>14.3f} {ch:>+8.0f}%")

    print("\n% --- LaTeX body for tab:pcl ---")
    print("\\toprule")
    print("\\textbf{Scale} & \\textbf{2-way Std LoRA} & \\textbf{3-way Std LoRA} & \\textbf{Change} \\\\")
    print("\\midrule")
    for ds, v2, v3 in rows:
        if v2 is None or v3 is None:
            print(f"{ds} & --- & --- & --- \\\\")
            continue
        ch = (v3 - v2) / v2 * 100
        v2_cell = f"\\regress{{{v2:.2f} ({_signed(ch)} from base)}}" if ch < -50 else f"{v2:.2f}"
        # The published table uses a different framing; keep the structure simple.
        print(f"{ds} & {v2:.2f} & {v3:.2f} & {_signed(ch)} \\\\")
    print("\\bottomrule")


# =============================================================================
# Entry point
# =============================================================================
TABLES = {
    "aggregate":            build_aggregate,
    "flights_decomp":       build_flights_decomp,
    "boolean_trend":        build_boolean_trend,
    "enum_cuad":            build_enum_cuad,
    "restaurants_decomp":   build_restaurants_decomp,
    "cuad_per_field":       build_cuad_per_field,
    "lexical":              build_lexical,
    "pcl":                  build_pcl,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--table", choices=list(TABLES) + ["all"], default="all")
    args = ap.parse_args()

    targets = list(TABLES) if args.table == "all" else [args.table]
    for name in targets:
        TABLES[name](args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
