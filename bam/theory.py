"""Analytical crosstalk predictions for the Hebbian BAM.

Probing the network with a stored pattern `x_j`, the field arriving at unit `i`
of the Y layer is

    h_i = n·y_{j,i} + Σ_{k≠j} y_{k,i} (x_kᵀ x_j).

Multiplying by `y_{j,i}` (which is ±1), the bit is correct whenever

    n + Σ_{k≠j} y_{j,i} y_{k,i} c_kj > 0,   c_kj := x_kᵀ x_j.

The products `y_{j,i} y_{k,i}` are ±1 and, for non-pathological codes, behave
like random signs along `i`. The sum is then a zero-mean variable of variance
`Σ_{k≠j} c_kj²`, which defines the signal-to-noise ratio

    SNR_j = n / sqrt(Σ_{k≠j} c_kj²)                         (X→Y direction)
    SNR_j = p / sqrt(Σ_{k≠j} d_kj²),  d_kj := y_kᵀ y_j      (Y→X direction)

and, under the Gaussian approximation, a per-bit error rate `Φ(-SNR_j)`.

Two consequences the study verifies empirically:

1. With a Hadamard codebook, `d_kj = 0` for every `k ≠ j`: crosstalk in the
   Y→X direction **vanishes exactly**, not merely on average. Recall of the
   face from its label is perfect for any number of stored patterns up to the
   rank limit.
2. In the X→Y direction crosstalk is imposed by the data. Faces are strongly
   correlated with one another (`c_kj/n` around 0.3–0.6 under median
   thresholding), which collapses the SNR well before Kosko's conjectured
   `min(n, p)` limit.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def crosstalk_snr(A: np.ndarray, *, per_pattern: bool = True) -> np.ndarray:
    """Hebbian SNR of the patterns in `A` (columns) against their own crosstalk.

    Args:
        A: (dim, P) bipolar matrix of the layer that **probes** the network.
            Pass `X` for the X→Y direction, `Y` for Y→X.
        per_pattern: if `False`, return the mean SNR as a scalar.

    Returns:
        (P,) vector with each pattern's SNR (or a scalar). Patterns perfectly
        orthogonal to all others receive `inf`.
    """
    A = np.asarray(A, dtype=np.float64)
    dim, P = A.shape
    G = A.T @ A  # (P, P); diagonal = dim
    off = G - np.diag(np.diag(G))
    noise = np.sqrt((off**2).sum(axis=0))
    with np.errstate(divide="ignore"):
        snr = np.where(noise > 0, dim / np.where(noise > 0, noise, 1.0), np.inf)
    return snr if per_pattern else float(np.mean(snr[np.isfinite(snr)]))


def predicted_bit_error(snr: np.ndarray) -> np.ndarray:
    """Predicted per-bit error rate, `Φ(-SNR)`."""
    return norm.cdf(-np.asarray(snr, dtype=np.float64))


def predicted_recall_rate(A: np.ndarray, target_dim: int) -> float:
    """Predicted fraction of patterns recalled with **no wrong bit at all**.

    Assumes independence across the `target_dim` bits of the destination layer:
    `P(perfect pattern) = (1 - Φ(-SNR))^target_dim`. The independence
    assumption is optimistic (errors are correlated through the same
    interfering memories), so this curve works as a practical upper bound.
    """
    snr = crosstalk_snr(A)
    pe = predicted_bit_error(snr)
    return float(np.mean((1.0 - pe) ** target_dim))


def predicted_capacity(
    A_full: np.ndarray, target_dim: int, *, threshold: float = 0.95
) -> int:
    """Largest `P` whose predicted recall rate still reaches `threshold`.

    Sweeps prefixes of `A_full` (its first `P` columns) and returns the largest
    `P` with `predicted_recall_rate >= threshold`. This is the theoretical
    "useful capacity", contrasted in the report with Kosko's `min(n, p)`.
    """
    best = 0
    for P in range(1, A_full.shape[1] + 1):
        if predicted_recall_rate(A_full[:, :P], target_dim) >= threshold:
            best = P
        else:
            break
    return best


def kosko_bound(n: int, p: int) -> int:
    """Kosko's original conjecture (1988): `min(n, p)` storable pairs.

    Included for contrast — the later literature (McEliece et al. 1987 for
    Hopfield; Haines & Hecht-Nielsen 1988 for BAM) showed the real capacity to
    be of order `min(n,p)/(2·log(min(n,p)))` even for random patterns, and this
    study shows it is far smaller still for correlated patterns such as faces.
    """
    return min(n, p)


def asymptotic_bound(n: int, p: int) -> float:
    """Asymptotic capacity `min(n,p) / (2·ln min(n,p))` for i.i.d. patterns."""
    m = min(n, p)
    return m / (2.0 * np.log(m)) if m > 1 else 1.0
