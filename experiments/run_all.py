"""Run the five experiments in sequence and reproduce every result.

    python experiments/run_all.py

Everything is deterministic given `bam.config.SEED`: running twice produces
identical tables. Figures and GIFs are rewritten under `results/`.
"""

from __future__ import annotations

import time

import _common
import exp1_recall
import exp2_capacity
import exp3_noise
import exp4_crosstalk
import exp5_identification
from _common import ROOT

EXPERIMENTS = (
    exp1_recall,
    exp2_capacity,
    exp3_noise,
    exp4_crosstalk,
    exp5_identification,
)


def main() -> None:
    t0 = time.time()
    for module in EXPERIMENTS:
        module.main()

    figures = sorted((ROOT / "results" / "figures").glob("*.png"))
    gifs = sorted((ROOT / "results" / "gifs").glob("*.gif"))
    tables = sorted((ROOT / "results" / "tables").glob("*.csv"))

    print()
    print("=" * 78)
    print("STUDY COMPLETE")
    print("=" * 78)
    print(f"  {len(figures)} figures · {len(gifs)} GIFs · {len(tables)} tables")
    print(f"  total time: {(time.time() - t0) / 60:.1f} min")
    print(f"  results in {(ROOT / 'results')}")


if __name__ == "__main__":
    main()
