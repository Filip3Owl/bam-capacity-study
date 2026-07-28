"""Paths, global seed and configuration constants for the study."""

from __future__ import annotations

import os
import ssl
from pathlib import Path

# --- Directories ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
GIFS_DIR = RESULTS_DIR / "gifs"
TABLES_DIR = RESULTS_DIR / "tables"

for _d in (DATA_DIR, FIGURES_DIR, GIFS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility --------------------------------------------------------
# Master seed: every experiment derives its own seeds from it, so that a full
# run of `experiments/run_all.py` is deterministic.
SEED = 20250728

# Independent repetitions (seeds) used in the statistical sweeps.
N_SEEDS = 8

# --- Model parameters -------------------------------------------------------
MAX_ITER = 30  # maximum bidirectional resonance iterations
IMAGE_SIDE = 64  # Olivetti is 64x64, hence N_X = 4096
N_X = IMAGE_SIDE * IMAGE_SIDE

# --- Animation --------------------------------------------------------------
# Square canvas: the animations are meant to be read on a phone, where a wide
# figure shrinks until nothing is legible. Everything is laid out inside a
# 1080x1080 frame at 100 dpi.
GIF_SIDE_PX = 1080
GIF_DPI = 100
GIF_FIGSIZE = (GIF_SIDE_PX / GIF_DPI, GIF_SIDE_PX / GIF_DPI)
FPS = 8


def configure_ssl() -> None:
    """Point Python at the `certifi` certificate bundle.

    The Homebrew build of Python on macOS ships without a usable CA bundle,
    which makes `fetch_olivetti_faces` fail with CERTIFICATE_VERIFY_FAILED on
    the first download. Called by `datasets.load_olivetti`.
    """
    try:
        import certifi
    except ImportError:  # pragma: no cover - environment without certifi
        return
    path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=path)
