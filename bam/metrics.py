"""Recall-quality metrics.

Every function accepts (dim, B) matrices with one pattern per column and
returns one value per column, or accepts 1-D vectors and returns scalars.
"""

from __future__ import annotations

import numpy as np


def _as_2d(A: np.ndarray) -> tuple[np.ndarray, bool]:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        return A[:, None], True
    return A, False


def normalized_hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray | float:
    """Fraction of disagreeing bits between `A` and `B`, per column.

    0 for perfect recall, 0.5 for an uncorrelated answer, and 1 for the exact
    complement (which, in a BAM, is also a fixed point).
    """
    A2, scalar = _as_2d(A)
    B2, _ = _as_2d(B)
    d = (A2 != B2).mean(axis=0)
    return float(d[0]) if scalar else d


def overlap(A: np.ndarray, B: np.ndarray) -> np.ndarray | float:
    """Overlap `aᵀb/n` per column, in [-1, 1].

    The usual order parameter of associative memories: `m = 1` means exact
    recall, `m ≈ 0` a spurious state unrelated to the target.
    """
    A2, scalar = _as_2d(A)
    B2, _ = _as_2d(B)
    m = (A2 * B2).mean(axis=0)
    return float(m[0]) if scalar else m


def exact_match(A: np.ndarray, B: np.ndarray) -> np.ndarray | bool:
    """Boolean mask of identical columns."""
    A2, scalar = _as_2d(A)
    B2, _ = _as_2d(B)
    ok = (A2 == B2).all(axis=0)
    return bool(ok[0]) if scalar else ok


def perfect_recall_rate(A: np.ndarray, B: np.ndarray) -> float:
    """Fraction of columns recalled with **not a single wrong bit**.

    This is the strictest criterion and the study's headline metric: in a
    content-addressable memory, an answer with 3 % of its bits flipped is not
    partially correct — it is a spurious state.
    """
    A2, _ = _as_2d(A)
    B2, _ = _as_2d(B)
    return float((A2 == B2).all(axis=0).mean())


def bit_accuracy(A: np.ndarray, B: np.ndarray) -> float:
    """Overall fraction of correct bits (a milder metric than the above)."""
    A2, _ = _as_2d(A)
    B2, _ = _as_2d(B)
    return float((A2 == B2).mean())


def confidence_interval(samples: np.ndarray, *,
                        level: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the normal CI for the mean of `samples`.

    With few seeds the normal CI is approximate; it is reported anyway to give
    the scale of the between-repetition variability.
    """
    from scipy.stats import norm

    a = np.asarray(samples, dtype=np.float64).ravel()
    if a.size < 2:
        return float(a.mean()), 0.0
    z = float(norm.ppf(0.5 + level / 2.0))
    return float(a.mean()), float(z * a.std(ddof=1) / np.sqrt(a.size))


def bit_flip_noise(X: np.ndarray, prob: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Flip each bit of `X` independently with probability `prob`."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must lie in [0, 1]")
    mask = rng.random(X.shape) < prob
    return np.where(mask, -X, X)


def occlude(
    X: np.ndarray, fraction: float, side: int, *,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Zero out (occlude) the bottom `fraction` of rows of each column's image.

    Unlike bit noise, occlusion supplies no wrong information: hidden pixels
    enter with zero local field. This is the "recall from partial evidence"
    scenario, and the BAM does visibly better at it.
    """
    del rng  # occlusion is deterministic; signature matches bit_flip_noise
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    Xo = np.array(X, dtype=np.float64)
    n_rows = int(round(fraction * side))
    if n_rows:
        Xo.reshape(side, side, -1)[side - n_rows :, :, :] = 0.0
    return Xo
