#!/usr/bin/env python3
"""Build anonymized supplementary zip for NeurIPS submission.

Stages the parts of the repo that support reproducibility, runs string
replacements to scrub author/host identifiers, and zips. Output:
  paper/supplementary.zip   (target: <100 MB)

Excludes raw third-party datasets (SGD, CUAD) and refers reviewers to the
public sources via the staged README.
"""
from __future__ import annotations

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
    "Experiment Plan.md",
    "data/Restaurants_1_train.jsonl",
    "data/Restaurants_1_train_pcl.jsonl",
    "data/Restaurants_1_test.jsonl",
    "data/Flights_1_train.jsonl",
    "data/Flights_1_train_pcl.jsonl",
    "data/Flights_1_test.jsonl",
    "data/cuad_train.jsonl",
    "data/cuad_test.jsonl",
    "data/cuad_train_full.jsonl",
    "data/cuad_test_full.jsonl",
    "data/cuad_schema.json",
]

INCLUDE_DIRS_OPTIONAL = ["checkpoints"]

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
    ("breckbaldwin", "anonymous"),
    ("Breck Baldwin", "Anonymous Author"),
    ("Breck", "Anonymous"),
    ("Independent Researcher", "Anonymous Affiliation"),
]

ANON_README = """# Supplementary Materials (Anonymized)

This zip contains code, generated training/test splits, fine-tuned LoRA
adapter checkpoints, and result artifacts supporting the paper.

## Layout

- `src/`         — data preparation, decomposition, training, evaluation
- `scripts/`     — experiment drivers and analysis
- `data/`        — generated train/test JSONL splits and schemas
- `checkpoints/` — LoRA adapter weights at 0.5B / 7B / 32B (per condition)
- `results/`     — per-condition logs and per-grammar-role decompositions

## Third-party datasets (not redistributed)

- Schema-Guided Dialogue (SGD): https://github.com/google-research-datasets/dstc8-schema-guided-dialogue (CC BY-SA 4.0)
- CUAD v1: https://www.atticusprojectai.org/cuad (CC BY 4.0)

The generated JSONL splits in `data/` are derived from these sources via
`src/prepare_data.py` and `src/prepare_cuad.py`. To regenerate from the
raw sources, place the raw datasets at `data/sgd/` and `data/cuad/`,
then run those scripts.

## Reproducing the experiments

See `scripts/run_all_paper.sh` for the end-to-end command sequence.
Training was performed on a single A100 80GB GPU.

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
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    total = 0
    for d in INCLUDE_DIRS:
        sd = REPO / d
        if sd.is_dir():
            total += stage_tree(sd, STAGE)
    for d in INCLUDE_DIRS_OPTIONAL:
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
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(STAGE):
            for fn in files:
                p = Path(root) / fn
                arc = p.relative_to(STAGE)
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
