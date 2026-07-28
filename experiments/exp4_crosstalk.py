"""Experiment 4 — Crosstalk: predicting the error before measuring it.

Question
--------
Experiments 2 and 3 show *that* the classical BAM fails on faces. This one
shows *why*, and checks whether the proposed mechanism predicts the observed
error quantitatively.

The model is the crosstalk signal-to-noise ratio (`bam.theory`): probing with a
stored pattern, the local field is the wanted signal plus the summed
interference of all the other patterns, whose magnitude is fixed by the
correlations between them. If the model is right, the measured per-bit error
rate should follow `Φ(-SNR)` with no fitted parameter.

Three strands:

A. **Validation**: predicted vs. observed per-bit error, across two
   binarisations, two codebooks, two directions and nine memory sizes.
B. **Does decorrelating the data help?** A direct comparison between
   thresholding each face at its own median and at the dataset's mean face.
C. **The trade-off frontier**: the least-squares rule has a weight `λ` that
   splits the fitting capacity between the two directions. No `W` gets both
   exactly right; what exists is a Pareto frontier, and this experiment
   measures it.

This experiment is the study's analytical core and produces no animation: its
result is a scatter of predictions against measurements, which is read, not
watched.

Outputs
-------
`fig04_crosstalk.png`, `fig05_tradeoff.png`, `tab04_crosstalk.csv`,
`tab04_tradeoff.csv`.
"""

from __future__ import annotations

import _common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import footer, header, save_table
from scipy.stats import spearmanr

from bam import viz
from bam.codebooks import CODEBOOK_LABELS, build_codebook, orthogonality
from bam.config import FIGURES_DIR, SEED
from bam.core import BAM
from bam.datasets import BINARIZATION_LABELS, load_dataset, mean_correlation
from bam.metrics import perfect_recall_rate
from bam.theory import crosstalk_snr, predicted_bit_error

CODE_LEN = 256
P_GRID = (5, 10, 20, 40, 60, 90, 120, 160, 200)
BINARIZATIONS = ("median", "mean-face")
CODEBOOKS = ("hadamard", "random")

P_TRADEOFF = 20
CODE_LEN_TRADEOFF = 64
LAMBDA_GRID = np.round(np.linspace(0.0, 1.0, 21), 3)

BIN_COLOR = {"median": viz.SERIES[1], "mean-face": viz.SERIES[0]}
DIRECTION_MARKER = {"X→Y": "o", "Y→X": "^"}
BIN_SHORT = {"median": "median threshold", "mean-face": "mean-face threshold"}
BIN_TINY = {"median": "median", "mean-face": "mean face"}


def validate_theory() -> pd.DataFrame:
    """Predicted vs. observed per-bit error, for the Hebb rule.

    Only the Hebb rule enters here: it is the one the crosstalk theory was
    derived for. The other rules do not have the form `W = Σ x_k y_kᵀ` and
    therefore admit no such signal-plus-interference decomposition.
    """
    rows = []
    for method in BINARIZATIONS:
        data = load_dataset(method=method)
        rng = np.random.default_rng(SEED)
        order = rng.permutation(data.n_patterns)
        for P in P_GRID:
            X = data.X[:, order[:P]]
            for cb in CODEBOOKS:
                Y = build_codebook(cb, P, CODE_LEN, seed=SEED)
                model = BAM("hebb").fit(X, Y)
                Yhat = model.associate_forward(X)
                Xhat = model.associate_backward(Y)

                rows.append({
                    "binarization": method, "codebook": cb, "P": P,
                    "direction": "X→Y",
                    "snr": float(np.mean(crosstalk_snr(X))),
                    "predicted_error": float(np.mean(
                        predicted_bit_error(crosstalk_snr(X)))),
                    "observed_error": float((Yhat != Y).mean()),
                    "observed_perfect": perfect_recall_rate(Yhat, Y),
                    "mean_correlation": mean_correlation(X),
                    "code_orthogonality": orthogonality(Y),
                })
                snr_y = crosstalk_snr(Y)
                rows.append({
                    "binarization": method, "codebook": cb, "P": P,
                    "direction": "Y→X",
                    "snr": float(np.mean(snr_y[np.isfinite(snr_y)]))
                           if np.isfinite(snr_y).any() else float("inf"),
                    "predicted_error": float(np.mean(predicted_bit_error(snr_y))),
                    "observed_error": float((Xhat != X).mean()),
                    "observed_perfect": perfect_recall_rate(Xhat, X),
                    "mean_correlation": mean_correlation(X),
                    "code_orthogonality": orthogonality(Y),
                })
    return pd.DataFrame(rows)


def sweep_tradeoff():
    """Pareto frontier between the two directions, sweeping `λ`."""
    data = load_dataset(method="mean-face")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(data.n_patterns)
    X = data.X[:, order[:P_TRADEOFF]]
    Y = build_codebook("hadamard", P_TRADEOFF, CODE_LEN_TRADEOFF)
    rows = []
    for lam in LAMBDA_GRID:
        model = BAM("least_squares", lam=float(lam), max_iter=20).fit(X, Y)
        Xr, Yr, _, _ = model.recall_batch(X0=X)
        rows.append({
            "lambda": float(lam),
            "forward_pass": perfect_recall_rate(model.associate_forward(X), Y),
            "backward_pass": perfect_recall_rate(model.associate_backward(Y), X),
            "resonance_x": perfect_recall_rate(Xr, X),
            "resonance_y": perfect_recall_rate(Yr, Y),
        })
    return pd.DataFrame(rows), data, X, Y


def figure_crosstalk(df: pd.DataFrame):
    """Panel A: SNR against P. Panel B: predicted vs. observed error."""
    viz.apply_theme()
    fig, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(11.6, 5.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.665, bottom=0.135,
                        wspace=0.24)

    # --- A: X→Y SNR collapses; Y→X SNR depends on the codebook
    for method in BINARIZATIONS:
        g = df[(df.binarization == method) & (df.direction == "X→Y")
               & (df.codebook == "hadamard")].sort_values("P")
        ax_s.plot(g["P"], g["snr"], color=BIN_COLOR[method], lw=2.4, marker="o",
                  ms=7, markeredgecolor=viz.SURFACE, markeredgewidth=1.2,
                  label=f"X→Y · {BIN_SHORT[method]}")
    g = df[(df.direction == "Y→X") & (df.codebook == "random")
           & (df.binarization == "mean-face")].sort_values("P")
    ax_s.plot(g["P"], g["snr"], color=viz.SERIES[2], lw=2.4, marker="^", ms=7,
              markeredgecolor=viz.SURFACE, markeredgewidth=1.2,
              label="Y→X · random codebook")

    ax_s.axhline(2.0, color=viz.INK_MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
    ax_s.annotate("SNR = 2  (≈ 2 % error per bit)", xy=(max(P_GRID), 2.0),
                  xytext=(-4, 6), textcoords="offset points", ha="right",
                  fontsize=8.5, color=viz.INK_MUTED)
    ax_s.set_yscale("log")
    ax_s.set_xlabel("Stored pairs  P")
    ax_s.set_ylabel("Crosstalk signal-to-noise ratio")
    ax_s.set_title("Crosstalk grows with the memory", fontsize=10.5)
    ax_s.legend(loc="upper right", fontsize=8.5)

    # --- B: parameter-free validation
    g = df[np.isfinite(df.predicted_error)]
    for direction, marker in DIRECTION_MARKER.items():
        for method in BINARIZATIONS:
            sub = g[(g.direction == direction) & (g.binarization == method)]
            ax_v.scatter(sub["predicted_error"], sub["observed_error"], s=48,
                         marker=marker, color=BIN_COLOR[method],
                         edgecolors=viz.SURFACE, linewidths=1.1, zorder=3,
                         label=f"{direction} · {BIN_TINY[method]}")
    lim = [1e-4, 0.6]
    ax_v.plot(lim, lim, color=viz.INK_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax_v.annotate("exact prediction", xy=(0.02, 0.02), xytext=(9, -4),
                  textcoords="offset points", fontsize=8.5, color=viz.INK_MUTED,
                  rotation=45, rotation_mode="anchor")
    ax_v.set_xscale("log")
    ax_v.set_yscale("log")
    ax_v.set_xlim(*lim)
    ax_v.set_ylim(*lim)
    ax_v.set_xlabel("Predicted error per bit  Φ(−SNR)")
    ax_v.set_ylabel("Observed error per bit")
    ax_v.set_title("The theory predicts the error with no fitted parameter",
                   fontsize=10.5)
    ax_v.legend(loc="upper left", fontsize=8, ncols=2)

    viz.figure_title(
        fig,
        "Why the classical BAM fails on faces: crosstalk",
        "Hebb rule · n = 4096 · p = 256 · points span 9 memory sizes × "
        "2 binarisations × 2 codebooks",
    )
    # The Hadamard case does not fit on the left plot: its SNR is infinite, and
    # a log axis has nowhere to put infinity. It is stated in the header.
    fig.text(0.012, 0.875,
             "With a Hadamard codebook the Y→X SNR is infinite — crosstalk "
             "cancels exactly, not merely on average.",
             ha="left", va="top", fontsize=9, color=viz.SERIES[2])
    viz.credit(fig)
    target = FIGURES_DIR / "fig04_crosstalk.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def figure_tradeoff(comp: pd.DataFrame):
    """The Pareto frontier between the two BAM directions."""
    viz.apply_theme()
    fig, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.68, bottom=0.14,
                        wspace=0.26)

    ax_l.plot(comp["lambda"], comp["forward_pass"], color=viz.SERIES[0], lw=2.6,
              marker="o", ms=6, markeredgecolor=viz.SURFACE, markeredgewidth=1.1,
              label="X→Y  (face → code)")
    ax_l.plot(comp["lambda"], comp["backward_pass"], color=viz.SERIES[1], lw=2.6,
              marker="^", ms=6.5, markeredgecolor=viz.SURFACE, markeredgewidth=1.1,
              label="Y→X  (code → face)")
    ax_l.set_xlabel("Weight λ given to the X→Y direction")
    ax_l.set_ylabel("Single-pass perfect recall")
    ax_l.set_ylim(-0.04, 1.06)
    ax_l.yaxis.set_major_formatter(viz.pct_fmt(0))
    ax_l.xaxis.set_major_formatter(viz.num_fmt(1))
    ax_l.set_title("The two directions compete for one matrix", fontsize=10.5)
    ax_l.legend(loc="center right", fontsize=9)

    ax_p.plot(comp["backward_pass"], comp["forward_pass"], color=viz.SERIES[2],
              lw=2.4, marker="o", ms=6, markeredgecolor=viz.SURFACE,
              markeredgewidth=1.1, zorder=3)
    for _, row in comp.iterrows():
        if row["lambda"] in (0.0, 0.5, 1.0):
            ax_p.annotate(f"λ = {row['lambda']:.1f}",
                          xy=(row["backward_pass"], row["forward_pass"]),
                          xytext=(8, 6), textcoords="offset points", fontsize=9,
                          color=viz.INK_SECONDARY)
    ax_p.set_xlabel("Y→X recall  (code → face)")
    ax_p.set_ylabel("X→Y recall  (face → code)")
    ax_p.set_xlim(-0.04, 1.12)
    ax_p.set_ylim(-0.04, 1.12)
    ax_p.xaxis.set_major_formatter(viz.pct_fmt(0))
    ax_p.yaxis.set_major_formatter(viz.pct_fmt(0))
    ax_p.set_title("Pareto frontier", fontsize=10.5)
    ax_p.scatter([1.0], [1.0], s=110, marker="*", color=viz.WARNING, zorder=5)
    # No leader line: any stroke from the star would cross the frontier.
    ax_p.text(0.06, 0.42, "★  unreachable ideal\n     with a single matrix",
              ha="left", va="center", fontsize=9, color=viz.WARNING,
              linespacing=1.5)

    viz.figure_title(
        fig,
        "One weight matrix, two directions, one compromise",
        f"Least-squares rule · P = {P_TRADEOFF} pairs · p = {CODE_LEN_TRADEOFF} · "
        f"Hadamard codes",
    )
    viz.credit(fig)
    target = FIGURES_DIR / "fig05_tradeoff.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> None:
    t0 = header(
        4, "Crosstalk and the trade-off between directions",
        "Is the observed error predictable from the pattern correlations?",
    )

    df = validate_theory()
    files = [save_table(df, "tab04_crosstalk.csv")]

    print("Mean correlation between stored faces (P = 200):")
    for method in BINARIZATIONS:
        c = df[(df.binarization == method) & (df.P == 200)].iloc[0]["mean_correlation"]
        print(f"  {BINARIZATION_LABELS[method]:38s} {float(c):.3f}")

    print("\nMean codebook orthogonality (P = 200; 0 = perfectly orthogonal):")
    for cb in CODEBOOKS:
        o = df[(df.codebook == cb) & (df.P == 200)].iloc[0]["code_orthogonality"]
        print(f"  {CODEBOOK_LABELS[cb]:38s} {float(o):.4f}")

    # Agreement between theory and measurement. Zero-error points are reported
    # separately: on a log scale they have no position, yet they are exactly
    # the cases where the prediction is exact (infinite SNR, Hadamard codes).
    exact = df[(df.predicted_error == 0) & (df.observed_error == 0)]
    positive = df[(df.predicted_error > 0) & (df.observed_error > 0)]
    conservative = df[(df.predicted_error > 0) & (df.observed_error == 0)]

    r_log = float(np.corrcoef(np.log10(positive.predicted_error),
                              np.log10(positive.observed_error))[0, 1])
    rho = float(spearmanr(df.predicted_error, df.observed_error).statistic)
    rel_err = float(np.median(
        np.abs(positive.observed_error - positive.predicted_error)
        / positive.predicted_error))

    print(f"\nAgreement between predicted and observed error ({len(df)} points):")
    print(f"  exact prediction of zero error ................... {len(exact)} points")
    print(f"  positive prediction, positive measurement ........ {len(positive)} points")
    print(f"      Pearson correlation in log-log ............... r = {r_log:.3f}")
    print(f"      median relative error of the prediction ...... {rel_err:.3f}")
    print(f"  positive prediction, zero measurement ............ "
          f"{len(conservative)} points")
    print(f"  Spearman correlation over all points ............. rho = {rho:.3f}")

    print("\nMean X→Y SNR (Hadamard codebook):")
    for method in BINARIZATIONS:
        g = df[(df.binarization == method) & (df.direction == "X→Y")
               & (df.codebook == "hadamard")].sort_values("P")
        samples = " · ".join(f"P={int(r.P)}: {float(r.snr):.2f}"
                             for _, r in g.iterrows() if r.P in (5, 20, 90, 200))
        print(f"  {method:12s} {samples}")

    comp, _, _, _ = sweep_tradeoff()
    files.append(save_table(comp, "tab04_tradeoff.csv"))
    best = comp.assign(total=comp.forward_pass + comp.backward_pass).sort_values(
        "total", ascending=False).iloc[0]
    print(f"\nBest compromise: λ = {float(best['lambda']):.2f} → "
          f"X→Y {float(best.forward_pass):.2f} · "
          f"Y→X {float(best.backward_pass):.2f}")

    files.append(figure_crosstalk(df))
    files.append(figure_tradeoff(comp))
    footer(t0, files)


if __name__ == "__main__":
    main()
