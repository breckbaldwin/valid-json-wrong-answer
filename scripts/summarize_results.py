#!/usr/bin/env python3
"""DEPRECATED — use scripts/build_tables.py.

This script previously printed a couple of comparison views over
results/*.json. It has been superseded by `scripts/build_tables.py`,
which produces the LaTeX body for every table in the paper from the
same JSON sources.

Kept as a thin shim so existing muscle memory and stale documentation
keep working: this script execs build_tables.py with the same arg list.
Will be removed in a future version.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TARGET = _REPO / "scripts" / "build_tables.py"

sys.stderr.write(
    "summarize_results.py is deprecated; forwarding to "
    "scripts/build_tables.py.\n"
)
os.execv(sys.executable, [sys.executable, str(_TARGET), *sys.argv[1:]])
