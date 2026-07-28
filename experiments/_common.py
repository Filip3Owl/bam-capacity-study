"""Shared plumbing for the experiments.

Import this module **first** in every experiment: it puts the project root on
`sys.path` so the scripts can be run directly
(`python experiments/exp1_recall.py`) without installation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from bam.config import TABLES_DIR  # noqa: E402


def header(number: int, title: str, question: str) -> float:
    """Print the experiment header and return the start timestamp."""
    print()
    print("=" * 78)
    print(f"EXPERIMENT {number} — {title}")
    print("=" * 78)
    print(f"Question: {question}")
    print("-" * 78)
    return time.time()


def footer(t0: float, files: list[Path]) -> None:
    """Print the experiment's closing summary."""
    print("-" * 78)
    for f in files:
        print(f"  written: {f.relative_to(ROOT)}")
    print(f"  total time: {time.time() - t0:.1f} s")


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write a results `DataFrame` to `results/tables/`."""
    target = TABLES_DIR / name
    df.to_csv(target, index=False, float_format="%.6g")
    return target
