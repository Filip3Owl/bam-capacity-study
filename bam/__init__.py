"""
Bidirectional Associative Memory (BAM) — an experimental study.

Support package for the experiments in `experiments/`. It provides:

* `datasets`   — loading and binarisation of the Olivetti/ORL face database;
* `codebooks`  — Y-layer vector generation (Hadamard, random, correlated);
* `core`       — the BAM model and the four learning rules compared;
* `metrics`    — recall-quality metrics;
* `theory`     — analytical crosstalk predictions;
* `viz`        — dark theme and plotting primitives;
* `ptbr`       — Portuguese strings rendered inside the animations;
* `animation`  — assembly of the animated results.

Shape convention used throughout: patterns are **columns**. `X` has shape
(n, P) and `Y` has shape (p, P), both with values in {-1, +1}.
"""

from bam import animation, codebooks, core, datasets, metrics, ptbr, theory, viz
from bam.core import BAM, RULES, recall
from bam.metrics import normalized_hamming, overlap, perfect_recall_rate

__all__ = [
    "BAM",
    "RULES",
    "animation",
    "codebooks",
    "core",
    "datasets",
    "metrics",
    "normalized_hamming",
    "overlap",
    "perfect_recall_rate",
    "ptbr",
    "recall",
    "theory",
    "viz",
]

__version__ = "2.0.0"
