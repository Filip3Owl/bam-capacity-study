"""Experiment 2 — Storage capacity.

Question
--------
How many `(face, code)` pairs does a BAM actually store? Kosko (1988)
conjectured `min(n, p)`; for random patterns the asymptotic analysis gives
`min(n,p) / (2·ln min(n,p))`. Both numbers assume uncorrelated patterns. Real
faces are not uncorrelated. This experiment measures the effective capacity
along three axes:

* **learning rule** (4 rules);
* **Y-layer codebook** (orthogonal Hadamard vs. random);
* **recall mode** (single pass vs. full bidirectional resonance).

The third axis is the one the classical literature does not separate, and it is
decisive: with Hadamard codes the Y→X direction is *exact* in a single pass for
any `P`, but resonance feeds back the error of the X→Y direction and destroys
that result.

Outputs
-------
`02_capacity.gif`, `fig02_capacity.png`, `tab02_capacity*.csv`.
"""

from __future__ import annotations

import _common
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import footer, header, save_table

from bam import ptbr, viz
from bam.animation import (
    BADGE_GAP,
    PANEL_BOTTOM,
    PANEL_SUBTITLE_SIZE,
    PANEL_TITLE_SIZE,
    badge,
    label_ends,
    new_square_figure,
    save_gif,
    series_line,
    takeaway,
)
from bam.codebooks import CODEBOOK_LABELS, build_codebook
from bam.config import FIGURES_DIR, SEED
from bam.core import BAM, RULE_LABELS, RULES
from bam.datasets import load_dataset
from bam.metrics import confidence_interval, perfect_recall_rate
from bam.theory import asymptotic_bound, kosko_bound

CODE_LEN = 256  # code length: allows up to 255 Hadamard codes
P_GRID = (5, 10, 20, 40, 60, 90, 120, 160, 200)
CODEBOOKS = ("hadamard", "random")
N_SEEDS = 6
MAX_ITER = 20
MAX_MT_EPOCHS = 30


def sweep(data) -> pd.DataFrame:
    """Measure perfect-recall rate across the whole configuration grid."""
    rows = []
    total = len(P_GRID) * len(CODEBOOKS) * len(RULES) * N_SEEDS
    done = 0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(SEED + seed)
        # One fixed random order per seed: the sets of P pairs are nested (the
        # P=10 set contains the P=5 one), which makes the capacity curve a
        # genuine degradation rather than a fresh resample at every point.
        order = rng.permutation(data.n_patterns)
        for P in P_GRID:
            X = data.X[:, order[:P]]
            for cb in CODEBOOKS:
                Y = build_codebook(cb, P, CODE_LEN, seed=SEED + seed)
                for rule in RULES:
                    model = BAM(rule, max_iter=MAX_ITER,
                                max_mt_epochs=MAX_MT_EPOCHS).fit(X, Y)
                    Xr, Yr, iters, converged = model.recall_batch(X0=X)
                    rows.append({
                        "seed": seed, "P": P, "codebook": cb, "rule": rule,
                        "forward_pass": perfect_recall_rate(
                            model.associate_forward(X), Y),
                        "backward_pass": perfect_recall_rate(
                            model.associate_backward(Y), X),
                        "resonance_x": perfect_recall_rate(Xr, X),
                        "resonance_y": perfect_recall_rate(Yr, Y),
                        "mean_iterations": float(iters.mean()),
                        "convergence": float(converged.mean()),
                    })
                    done += 1
        print(f"  seed {seed + 1}/{N_SEEDS} — {done}/{total} configurations",
              flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95 % CI half-width across seeds."""
    metrics = ["forward_pass", "backward_pass", "resonance_x", "resonance_y"]
    out = []
    for (P, cb, rule), g in df.groupby(["P", "codebook", "rule"], sort=False):
        row = {"P": P, "codebook": cb, "rule": rule}
        for m in metrics:
            mean, ci = confidence_interval(g[m].to_numpy())
            row[m], row[f"{m}_ci95"] = mean, ci
        out.append(row)
    return pd.DataFrame(out).sort_values(["codebook", "rule", "P"])


def useful_capacity(res: pd.DataFrame, metric: str,
                    threshold: float = 0.95) -> pd.DataFrame:
    """Largest `P` on the grid at which the metric still reaches `threshold`."""
    out = []
    for (cb, rule), g in res.groupby(["codebook", "rule"], sort=False):
        ok = g.sort_values("P")
        ok = ok[ok[metric] >= threshold]["P"]
        out.append({"codebook": cb, "rule": rule, "metric": metric,
                    "useful_capacity": int(ok.max()) if len(ok) else 0})
    return pd.DataFrame(out)


def figure_capacity(res: pd.DataFrame, n: int):
    """Small multiples: codebook down the rows, recall direction across."""
    viz.apply_theme()
    columns = [
        ("forward_pass", "X→Y  (single pass)"),
        ("backward_pass", "Y→X  (single pass)"),
        ("resonance_x", "X  (full resonance)"),
    ]
    fig, axs = plt.subplots(len(CODEBOOKS), len(columns), figsize=(12.4, 6.6),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.705, bottom=0.115,
                        hspace=0.30, wspace=0.10)

    for i, cb in enumerate(CODEBOOKS):
        for j, (metric, title) in enumerate(columns):
            ax = axs[i, j]
            for order, rule in enumerate(RULES):
                g = res[(res.codebook == cb) & (res.rule == rule)].sort_values("P")
                x, y = g["P"].to_numpy(), g[metric].to_numpy()
                ci = g[f"{metric}_ci95"].to_numpy()
                color = viz.RULE_COLOR[rule]
                ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.16, lw=0,
                                zorder=2)
                # Decreasing line widths: coincident series (frequent here,
                # where several rules saturate at 0 % or 100 %) stay visible
                # under one another as a halo instead of vanishing.
                ax.plot(x, y, color=color, lw=5.2 - 1.1 * order,
                        marker=viz.RULE_MARKER[rule], ms=8.5 - 1.0 * order,
                        markeredgecolor=viz.SURFACE, markeredgewidth=1.2,
                        label=RULE_LABELS[rule], zorder=3 + order)
            ax.set_ylim(-0.04, 1.06)
            ax.yaxis.set_major_formatter(viz.pct_fmt(0))
            if i == 0:
                ax.set_title(title, fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"{CODEBOOK_LABELS[cb]}\nperfect recall", fontsize=9.5)
            if i == len(CODEBOOKS) - 1:
                ax.set_xlabel("Stored pairs  P")

    limit = asymptotic_bound(n, CODE_LEN)
    for ax in axs.ravel():
        ax.axvline(limit, color=viz.INK_MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
    axs[-1, -1].annotate(
        f"asymptotic bound for\ni.i.d. patterns ≈ {limit:.0f} pairs",
        xy=(limit, 0.50), xytext=(10, 0), textcoords="offset points",
        fontsize=8.5, color=viz.INK_MUTED, va="center",
    )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=4,
               bbox_to_anchor=(0.988, 0.855), columnspacing=1.8, handlelength=1.9)

    viz.figure_title(
        fig,
        "Real BAM capacity on faces: far below the classical conjecture",
        f"n = {n} · p = {CODE_LEN} · Kosko's conjecture predicts "
        f"min(n, p) = {kosko_bound(n, CODE_LEN)} pairs · mean of {N_SEEDS} seeds, "
        f"band = 95 % CI",
    )
    viz.credit(fig)
    target = FIGURES_DIR / "fig02_capacity.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def capacity_gif(data, res: pd.DataFrame):
    """Square GIF: one variable changes — the codebook — and capacity moves 20×.

    Both rows use the same network and the same Hebb rule, and both are probed
    the same way: hand the network a stored code and ask for the face back. The
    only difference is whether the codes are mutually orthogonal.
    """
    rng = np.random.default_rng(SEED)
    order = rng.permutation(data.n_patterns)
    n_examples = 2
    shown = ("hadamard", "random")

    frames = []
    for P in P_GRID:
        X = data.X[:, order[:P]]
        recovered = {}
        for cb in shown:
            Y = build_codebook(cb, P, CODE_LEN, seed=SEED)
            model = BAM("hebb").fit(X, Y)
            recovered[cb] = model.associate_backward(Y)[:, :n_examples]
        frames.append((P, X[:, :n_examples], recovered))

    fig, top = new_square_figure("02_capacity")
    badge_y, top = top + 0.008, top - BADGE_GAP
    gs = fig.add_gridspec(
        3, n_examples, height_ratios=[1.0, 1.0, 1.15],
        left=0.135, right=0.965, top=top, bottom=PANEL_BOTTOM,
        hspace=0.38, wspace=0.08,
    )
    axes = {cb: [fig.add_subplot(gs[i, k]) for k in range(n_examples)]
            for i, cb in enumerate(shown)}
    ax_c = fig.add_subplot(gs[2, :])
    ax_c.set_xlim(0, max(P_GRID) * 1.32)
    ax_c.set_ylim(-0.05, 1.10)
    ax_c.set_xlabel(ptbr.AXIS_STORED_PAIRS)
    ax_c.set_ylabel(ptbr.AXIS_PERFECT_RECALL, fontsize=13)
    ax_c.yaxis.set_major_formatter(viz.pct_fmt(0, ptbr=True))

    curves = {
        cb: res[(res.codebook == cb) & (res.rule == "hebb")]
        .sort_values("P")[["P", "backward_pass"]].to_numpy()
        for cb in shown
    }

    movable: list = []

    def update(i: int) -> None:
        nonlocal movable
        for a in movable:
            a.remove()
        movable = []
        P, targets, recovered = frames[i]
        for cb in shown:
            for k, ax in enumerate(axes[cb]):
                ax.clear()
                ax.set_box_aspect(1.0)
                wrong = float((recovered[cb][:, k] != targets[:, k]).mean())
                caption = (ptbr.EXACT_SHORT if wrong == 0 else
                           ptbr.WRONG_SHORT.format(
                               value=viz.format_percent(wrong, 1, ptbr=True)))
                viz.face_panel(
                    ax, recovered[cb][:, k], data.side,
                    ptbr.CODEBOOKS[cb] if k == 0 else "", subtitle=caption,
                    border=viz.GOOD if wrong == 0 else viz.CRITICAL,
                    title_size=PANEL_TITLE_SIZE,
                    subtitle_size=PANEL_SUBTITLE_SIZE,
                )
        ends = []
        for order_c, cb in enumerate(shown):
            d = curves[cb]
            upto = int((d[:, 0] <= P).sum())
            movable.append(series_line(
                ax_c, d[:upto, 0], d[:upto, 1], color=viz.CODEBOOK_COLOR[cb],
                marker="o" if cb == "hadamard" else "s", order=order_c))
            ends.append((d[upto - 1, 0], d[upto - 1, 1], ptbr.CODEBOOKS[cb]))
        movable += label_ends(ax_c, ends)
        movable.append(badge(fig, badge_y, ptbr.STORED_PAIRS_COUNT.format(n=P)))
        if i == len(frames) - 1:
            movable += takeaway(fig, "02_capacity")

    return save_gif(fig, update, len(frames), "02_capacity",
                    fps=2, hold_first=2, hold_last=12)


def main() -> None:
    t0 = header(
        2, "Storage capacity",
        "How many pairs does the BAM really hold, and what decides that number?",
    )
    data = load_dataset(method="mean-face")
    print(f"n = {data.n} · p = {CODE_LEN} · P grid = {list(P_GRID)} · "
          f"{N_SEEDS} seeds")
    print(f"Kosko's conjecture: min(n, p) = {kosko_bound(data.n, CODE_LEN)} pairs")
    print(f"Asymptotic i.i.d. bound: "
          f"{asymptotic_bound(data.n, CODE_LEN):.0f} pairs\n")

    df = sweep(data)
    res = summarize(df)

    files = [save_table(df, "tab02_capacity.csv"),
             save_table(res, "tab02_capacity_summary.csv")]

    caps = pd.concat([useful_capacity(res, m)
                      for m in ("forward_pass", "backward_pass", "resonance_x")])
    files.append(save_table(caps, "tab02_useful_capacity.csv"))

    print("\nUseful capacity (largest P with >= 95 % perfect recall):")
    print(caps.pivot_table(index=["codebook", "rule"], columns="metric",
                           values="useful_capacity").to_string())

    print("\nHighlights:")
    for cb in CODEBOOKS:
        for rule in RULES:
            g = res[(res.codebook == cb) & (res.rule == rule)]
            top_row = g[max(P_GRID) == g.P].iloc[0]
            print(f"  {CODEBOOK_LABELS[cb]:22s} {RULE_LABELS[rule]:22s} "
                  f"P={max(P_GRID)}: X→Y {top_row.forward_pass:.2f} · "
                  f"Y→X {top_row.backward_pass:.2f} · "
                  f"resonance {top_row.resonance_x:.2f}")

    files.append(figure_capacity(res, data.n))
    files.append(capacity_gif(data, res))
    footer(t0, files)


if __name__ == "__main__":
    main()
