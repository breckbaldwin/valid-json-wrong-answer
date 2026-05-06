#!/usr/bin/env python3
"""Build anonymized supplementary zip for NeurIPS submission.

Stages the parts of the repo that support reproducibility, runs string
replacements to scrub author/host identifiers, and zips. Output:
  paper/supplementary.zip   (target: <100 MB)

Excludes raw third-party datasets (SGD, CUAD) and LoRA checkpoints by
default; reviewers regenerate via the included scripts. Pass
``--include-checkpoints`` to bundle adapter weights — only safe when the
checkpoints dir is small (e.g. local smoke tests). The full paper sweep
produces ~2 GB of 32B adapters and will not fit under the 100 MB cap.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "_supplementary_stage"
OUT = REPO / "paper" / "supplementary.zip"

INCLUDE_DIRS = ["src", "scripts", "results"]
INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    # Schemas — required by decompose.py / runpod_*.py / margin_gating_eval.py
    "data/Restaurants_1_schema.json",
    "data/Restaurants_1_schema_pcl.json",
    "data/Flights_1_schema.json",
    "data/Flights_1_schema_pcl.json",
    "data/cuad_schema.json",
    # Train splits (vanilla + PCL-relabeled)
    "data/Restaurants_1_train.jsonl",
    "data/Restaurants_1_train_pcl.jsonl",
    "data/Flights_1_train.jsonl",
    "data/Flights_1_train_pcl.jsonl",
    "data/cuad_train.jsonl",
    "data/cuad_train_full.jsonl",
    # Test splits (vanilla + PCL-relabeled; full-context CUAD)
    "data/Restaurants_1_test.jsonl",
    "data/Restaurants_1_test_pcl.jsonl",
    "data/Flights_1_test.jsonl",
    "data/Flights_1_test_pcl.jsonl",
    "data/cuad_test.jsonl",
    "data/cuad_test_full.jsonl",
]
# `Experiment Plan.md` deliberately excluded: it is a live internal
# planning doc with strategy notes that should not ship to reviewers.

# Checkpoints are excluded by default — full-sweep 32B adapters exceed
# the 100 MB cap. Pass --include-checkpoints to override (use only when
# the local checkpoints/ dir is small, e.g. smoke tests).
INCLUDE_DIRS_OPTIONAL = []

EXCLUDE_PATTERNS = re.compile(
    r"(__pycache__|\.pyc$|\.DS_Store|\.git/|\.venv/|\.tgz$|\.log$|results_Apr24/|"
    r"build_supplementary\.py$|setup_and_delete\.py$)"
)

TEXT_EXTS = {".py", ".sh", ".md", ".txt", ".tex", ".bib", ".json", ".jsonl",
             ".yaml", ".yml", ".cfg", ".ini", ".toml"}

# Plain string replacements applied to text files. Order matters — longer first.
SCRUB = [
    ("breckbaldwin@gmail.com", "anonymous@example.com"),
    ("git.breckbaldwin.com", "anonymized.example"),
    ("https://github.com/breckbaldwin/valid-json-wrong-answer", "https://anonymized.example/anon-paper"),
    ("github.com/breckbaldwin", "anonymized.example/anon"),
    ("github.com/validjson", "anonymized.example/anon-org"),
    ("validjson.com", "anonymized.example"),
    ("breckbaldwin", "anonymous"),
    ("Breck Baldwin", "Anonymous Author"),
    ("Breck", "Anonymous"),
    ("Independent Researcher", "Anonymous Affiliation"),
]

ANON_README = """# Supplementary Materials (Anonymized)

Code, generated train/test splits, schemas, and result artefacts
supporting the paper.

## Quick check: regenerate every paper table from the included results

Every table in the paper is built from the JSONs in `results/`. You
can verify that the paper's numbers match the data on any laptop, no
GPU required, no pip install required (the table builder is pure
Python and uses only the standard library):

```bash
unzip supplementary.zip && cd supplementary
python3 scripts/build_tables.py
```

This prints, for each of the 8 tables, a console-friendly preview
followed by the LaTeX body. Compare against the corresponding tables
in `paper/main.tex`. To see one table at a time:

```bash
python3 scripts/build_tables.py --table aggregate
# names: aggregate, flights_decomp, boolean_trend, enum_cuad,
#        restaurants_decomp, cuad_per_field, lexical, pcl
```

## Layout

- `scripts/build_tables.py` — generates every paper table from `results/`
- `src/`           — data preparation, decomposition, training, evaluation
- `scripts/`       — experiment drivers and table builders
- `data/`          — generated train/test JSONL splits and JSON Schemas
- `results/`       — per-condition result JSONs consumed by `build_tables.py`
- `requirements.txt` — pinned dependencies for the GPU pipeline

LoRA adapter checkpoints are **not** bundled (~2 GB at 32B; the 100 MB
cap forbids it). The included scripts retrain from scratch on an A100
if you want to re-run the full pipeline — see below.

## Full reproduction from scratch (optional, A100, ~5 hours)

A single A100 80GB GPU is sufficient for the entire sweep. The
canonical seeded run reported in the paper completed in 4 hours 39
minutes wall-clock on RunPod A100 80GB; ~5 hours is a safe budget.
The instructions below assume a fresh **RunPod** A100 pod (Ubuntu
image), but any host with CUDA + Python 3.10+ works.

Note that `results/` ships pre-populated so that `build_tables.py` can
run on a laptop. The pipeline scripts are **idempotent** — they skip
any work whose output already exists. To force a full re-run from
scratch, **delete the shipped results first**:

```bash
rm -rf results/
```

(Skip this step if you want to add to the existing results, e.g.
re-running a single missing condition; the scripts will only do the
work whose output is absent.)

### One-time pod setup

RunPod Ubuntu images are minimised and may be missing common tools
(`rsync`, `unzip`, `git`). Install whatever you need first:

```bash
apt-get update && apt-get install -y rsync unzip
```

Set two HuggingFace environment variables:

```bash
# HuggingFace token — required for any gated model downloads.
export HF_TOKEN=<your_token>

# Cache location — point at the persistent volume so re-launching
# the pod does not redownload the ~60 GB 32B weights.
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
```

Create a virtual environment called `venv` and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`run_all_paper.sh` activates `venv/` itself, so you don't need to keep
the venv sourced in the shell that launches the script — but having
it active is useful for ad-hoc invocations of the underlying tools.

### Run the full sweep

```bash
# Phase 1 — train baseline + standard LoRA at 3 scales × 3 schemas
# Phase 2 — extract per-record probabilities for margin-gating
# Phase 3 — train PCL-relabeled LoRA + extract probs
# Phase 4 — full-context CUAD (optional, generalisation check)
# Phase 5 — per-role decompose against PCL checkpoints (Table 8)
bash scripts/run_all_paper.sh

# Re-build the tables from the freshly produced results.
python scripts/build_tables.py
```

`run_all_paper.sh` is idempotent: training and evaluation skip work
whose output already exists, so an interrupted pod can be resumed by
simply re-running the same command.

## Third-party datasets (not redistributed)

The generated JSONL splits in `data/` are derived from these public
sources via `src/prepare_data.py` and `src/prepare_cuad.py`. The
generated splits are sufficient to reproduce the paper; the raw
sources are only needed if you want to regenerate them.

- Schema-Guided Dialogue (SGD): https://github.com/google-research-datasets/dstc8-schema-guided-dialogue (CC BY-SA 4.0)
- CUAD v1: https://www.atticusprojectai.org/cuad (CC BY 4.0)

## Anonymity

Author identifiers and repository URLs have been scrubbed. The original
public release URL will be provided upon acceptance.
"""


def is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS


def scrub(text: str) -> str:
    for old, new in SCRUB:
        text = text.replace(old, new)
    return text


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_text(src):
        try:
            content = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
            return
        dst.write_text(scrub(content), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def stage_tree(src_dir: Path, rel_root: Path) -> int:
    n = 0
    for root, dirs, files in os.walk(src_dir):
        rel = Path(root).relative_to(REPO)
        if EXCLUDE_PATTERNS.search(str(rel) + "/"):
            dirs[:] = []
            continue
        for fname in files:
            sp = Path(root) / fname
            rp = sp.relative_to(REPO)
            if EXCLUDE_PATTERNS.search(str(rp)):
                continue
            dp = rel_root / rp
            copy_file(sp, dp)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Bundle the checkpoints/ dir (default: excluded — too large for "
             "the 100 MB cap once the full sweep populates 32B adapters).",
    )
    args = ap.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    total = 0
    for d in INCLUDE_DIRS:
        sd = REPO / d
        if sd.is_dir():
            total += stage_tree(sd, STAGE)
    optional_dirs = list(INCLUDE_DIRS_OPTIONAL)
    if args.include_checkpoints:
        optional_dirs.append("checkpoints")
    for d in optional_dirs:
        sd = REPO / d
        if sd.is_dir():
            total += stage_tree(sd, STAGE)

    for f in INCLUDE_FILES:
        sp = REPO / f
        if sp.is_file():
            copy_file(sp, STAGE / f)
            total += 1

    (STAGE / "README.md").write_text(ANON_README, encoding="utf-8")

    # Verify no scrub-targets remain in any text file under stage
    leaks = []
    for root, _, files in os.walk(STAGE):
        for fn in files:
            p = Path(root) / fn
            if not is_text(p):
                continue
            try:
                t = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for old, _ in SCRUB:
                if old in t:
                    leaks.append((p.relative_to(STAGE), old))
    if leaks:
        print("ANONYMITY LEAKS — aborting:", file=sys.stderr)
        for path, term in leaks:
            print(f"  {path}: {term!r}", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    # All entries are written under a top-level `supplementary/` prefix so
    # that `unzip supplementary.zip` in any cwd produces one clean
    # directory rather than scattering src/ scripts/ data/ ... at top level.
    arc_root = Path("supplementary")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(STAGE):
            for fn in files:
                p = Path(root) / fn
                arc = arc_root / p.relative_to(STAGE)
                zf.write(p, arc)

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"staged files: {total}")
    print(f"output:       {OUT}")
    print(f"size:         {size_mb:.1f} MB (limit 100 MB)")
    if size_mb > 100:
        print("WARNING: exceeds 100 MB cap", file=sys.stderr)
        return 1
    shutil.rmtree(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
