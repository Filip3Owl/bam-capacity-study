"""Experiment 1 — Bidirectional recall and the Lyapunov function.

Question
--------
Does the BAM converge, and where to? Kosko's theorem (1988) guarantees that the
energy `E = -xᵀWy` never rises under the bidirectional updates, whatever `W`
is. This experiment (i) checks that guarantee empirically over hundreds of
trajectories and (ii) shows that the guarantee is practically empty: the
network always converges, but with real faces the classical rule converges to
spurious states.

Outputs
-------
`01_bidirectional_recall.gif`, `fig01_energy.png`, `tab01_lyapunov.csv`.
"""

from __future__ import annotations

import _common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import footer, header, save_table

from bam import ptbr, viz
from bam.animation import bidirectional_recall_gif, noise_preroll
from bam.codebooks import build_codebook
from bam.config import FIGURES_DIR, MAX_ITER, SEED
from bam.core import BAM, RULE_LABELS, RULES
from bam.datasets import load_dataset, one_index_per_subject
from bam.metrics import bit_flip_noise, normalized_hamming

P = 20  # stored pairs
CODE_LEN = 64  # Y-layer code length
NOISE = 0.25  # fraction of probe bits flipped
N_LYAPUNOV_PROBES = 200


def setup(method: str = "mean-face"):
    """Dataset, codebook and the subset of stored pairs."""
    data = load_dataset(method=method)
    idx = one_index_per_subject(data.target)[:P]
    X = data.X[:, idx]
    Y = build_codebook("hadamard", P, CODE_LEN)
    return data, X, Y


def verify_lyapunov(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """Check energy monotonicity over many random trajectories.

    Probes are stored faces corrupted with increasing bit noise. For the linear
    rules Kosko's theorem predicts **exact** monotonicity; the dense rule uses
    the log-sum-exp energy of dense associative memories, for which we verify
    the same property empirically.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for rule in RULES:
        model = BAM(rule, max_iter=MAX_ITER).fit(X, Y)
        violations = converged = 0
        drops = []
        for _ in range(N_LYAPUNOV_PROBES):
            j = int(rng.integers(P))
            prob = float(rng.uniform(0.05, 0.45))
            probe = bit_flip_noise(X[:, j : j + 1], prob, rng)[:, 0]
            traj = model.recall(x0=probe)
            violations += int(not traj.energy_is_monotone())
            converged += int(traj.converged)
            e = np.asarray(traj.energies)
            drops.append(float(e[0] - e[-1]))
        rows.append({
            "rule": RULE_LABELS[rule],
            "trajectories": N_LYAPUNOV_PROBES,
            "monotonicity_violations": violations,
            "convergence_rate": converged / N_LYAPUNOV_PROBES,
            "mean_energy_drop": float(np.mean(drops)),
        })
    return pd.DataFrame(rows)


def figure_energy(X: np.ndarray, Y: np.ndarray, probe: np.ndarray, j: int):
    """Energy descent against effective error, for the four rules.

    The left panel shows the energy; since the raw scales are incomparable
    across rules (the dense rule does not even use `-xᵀWy`), each trajectory is
    normalised to [0, 1] between its initial value and its minimum — a common
    dimensionless quantity, "how much of the total descent has happened", which
    legitimises the single axis. The right panel shows the fraction of wrong
    bits in layer X on the same time scale.

    Read together, they are the result of the experiment: **all four energy
    curves descend, and three of them stop far from zero error**. Stability is
    not the same as correct recall.

    The curves overlap for much of the way, so each series also carries its own
    marker and a decreasing line width — coincident series then show as a halo
    instead of disappearing under one another.
    """
    viz.apply_theme()
    fig, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.135, wspace=0.22)

    longest = 0
    for order, rule in enumerate(RULES):
        model = BAM(rule, max_iter=MAX_ITER).fit(X, Y)
        traj = model.recall(x0=probe)
        e = np.asarray(traj.energies)
        span = e[0] - e.min()
        norm = (e - e.min()) / span if span > 0 else np.zeros_like(e)
        err = np.array([normalized_hamming(x, X[:, j]) for x in traj.xs])
        t = np.arange(len(e))
        longest = max(longest, len(e))
        style = dict(
            color=viz.RULE_COLOR[rule], lw=4.4 - 0.9 * order,
            marker=viz.RULE_MARKER[rule], ms=9.0 - 1.0 * order,
            markeredgecolor=viz.SURFACE, markeredgewidth=1.4,
            label=RULE_LABELS[rule], zorder=3 + order,
        )
        ax_e.plot(t, norm, **style)
        ax_h.plot(t, err, **style)

    for ax in (ax_e, ax_h):
        ax.set_xlim(-0.35, longest - 0.5)
        ax.set_xlabel("Half-iteration (one layer update)")
    ax_e.set_ylim(-0.06, 1.08)
    ax_e.set_ylabel("Normalised energy")
    ax_e.yaxis.set_major_formatter(viz.num_fmt(1))
    ax_e.set_title("Lyapunov energy", fontsize=10.5)

    ax_h.set_ylim(-0.02, 0.42)
    ax_h.set_ylabel("Fraction of wrong bits in layer X")
    ax_h.yaxis.set_major_formatter(viz.pct_fmt(0))
    ax_h.set_title("Effective recall error", fontsize=10.5)
    ax_h.axhline(0.0, color=viz.GOOD, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax_h.annotate("exact recall", xy=(0.0, 0.0), xytext=(8, 8),
                  textcoords="offset points", ha="left", fontsize=8.5,
                  color=viz.GOOD)

    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=4,
               bbox_to_anchor=(0.988, 0.845), columnspacing=1.6, handlelength=1.9)

    viz.figure_title(
        fig,
        "The energy never rises — but descending is not the same as being right",
        f"Same probe ({int(NOISE * 100)}% of bits flipped) · {P} stored pairs · "
        f"Hadamard codes",
    )
    viz.credit(fig)
    target = FIGURES_DIR / "fig01_energy.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> None:
    t0 = header(
        1, "Bidirectional recall and the Lyapunov function",
        "Does the network converge? And does it converge to the right pair?",
    )
    data, X, Y = setup()
    print(f"n = {X.shape[0]} bits per face · p = {Y.shape[0]} bits per code · "
          f"P = {X.shape[1]} stored pairs")

    rng = np.random.default_rng(SEED)
    j = 0
    probe = bit_flip_noise(X[:, j : j + 1], NOISE, rng)[:, 0]
    print(f"Probe: face {j} with {int(NOISE * 100)}% of its bits flipped "
          f"(initial Hamming = {float(normalized_hamming(probe, X[:, j])):.3f})")

    files = []
    for rule in RULES:
        model = BAM(rule, max_iter=MAX_ITER).fit(X, Y)
        traj = model.recall(x0=probe)
        err = float(normalized_hamming(traj.x_final, X[:, j]))
        print(f"  {RULE_LABELS[rule]:22s} converged in {traj.n_iter} iterations · "
              f"final error = {err:.3f}")

    # The animation demonstrates the capability, so it uses the rule that
    # delivers it; the failure of the classical rule is the subject of the
    # static figure below and of every sweep in the other experiments.
    files.append(bidirectional_recall_gif(
        BAM("dense", max_iter=MAX_ITER).fit(X, Y),
        probe=probe, x_target=X[:, j], y_target=Y[:, j], side=data.side,
        name="01_bidirectional_recall",
        preroll=noise_preroll(X[:, j], NOISE, SEED),
    ))
    files.append(figure_energy(X, Y, probe, j))

    print(f"\nChecking energy monotonicity over {N_LYAPUNOV_PROBES} "
          f"trajectories per rule...")
    table = verify_lyapunov(X, Y)
    print(table.to_string(index=False))
    files.append(save_table(table, "tab01_lyapunov.csv"))

    footer(t0, files)


if __name__ == "__main__":
    main()
