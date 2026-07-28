"""Experiment 3 — Robustness to noise and occlusion.

Question
--------
A content-addressable memory is only useful if the probe may be corrupted. How
large is each rule's basin of attraction, and is it the same in both
directions?

Three corruption modes are swept:

A. **Bit noise on the face** — each pixel is flipped with probability `q`.
   *Active* corruption: the probe asserts wrong things.
B. **Occlusion of the face** — the bottom rows of the image are zeroed. This is
   *passive* corruption: hidden pixels enter with zero local field, i.e. the
   probe merely stays silent. A BAM should tolerate far more occlusion than
   noise, and the experiment measures how much.
C. **Bit noise on the code** — corruption hits layer Y instead. This is the
   direction in which Hadamard codes make the Hebbian BAM exact, and where the
   classical rule finally defends itself well.

Outputs
-------
`03_robustness.gif`, `fig03_robustness.png`, `tab03_*.csv`.
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
from bam.codebooks import build_codebook
from bam.config import FIGURES_DIR, SEED
from bam.core import BAM, RULE_LABELS, RULES
from bam.datasets import load_dataset, one_index_per_subject
from bam.metrics import (
    bit_accuracy,
    bit_flip_noise,
    confidence_interval,
    occlude,
    perfect_recall_rate,
)

P = 20
CODE_LEN = 64
N_SEEDS = 6
MAX_ITER = 20
NOISE_LEVELS = np.round(np.arange(0.0, 0.51, 0.05), 3)
OCCLUSION_LEVELS = np.round(np.arange(0.0, 0.91, 0.10), 3)
CODE_NOISE_LEVELS = np.round(np.arange(0.0, 0.51, 0.05), 3)

# Panel title: the scenario name plus the recall mode being measured, so the
# reader never has to hunt for that information in a floating note.
PANEL_TITLES = {
    "noise_x": "Bit noise on the face\nrecall by resonance",
    "occlusion_x": "Occlusion of the face\nrecall by resonance",
    "noise_y": "Bit noise on the code\nsingle-pass Y→X recall",
}


def sweep(data) -> pd.DataFrame:
    """Measure recall under the three corruption scenarios."""
    idx = one_index_per_subject(data.target)[:P]
    X = data.X[:, idx]
    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(SEED + 100 * seed)
        Y = build_codebook("hadamard", P, CODE_LEN)
        models = {r: BAM(r, max_iter=MAX_ITER).fit(X, Y) for r in RULES}
        for rule, model in models.items():
            for level in NOISE_LEVELS:
                probe = bit_flip_noise(X, float(level), rng)
                Xr, Yr, _, _ = model.recall_batch(X0=probe)
                rows.append(_row(seed, "noise_x", level, rule, Xr, X,
                                 model.associate_forward(probe), Y))
            for level in OCCLUSION_LEVELS:
                probe = occlude(X, float(level), data.side)
                Xr, Yr, _, _ = model.recall_batch(X0=probe)
                rows.append(_row(seed, "occlusion_x", level, rule, Xr, X,
                                 model.associate_forward(probe), Y))
            for level in CODE_NOISE_LEVELS:
                probe_y = bit_flip_noise(Y, float(level), rng)
                Xp = model.associate_backward(probe_y)
                Xr, Yr, _, _ = model.recall_batch(Y0=probe_y)
                rows.append({
                    "seed": seed, "scenario": "noise_y", "level": float(level),
                    "rule": rule,
                    "x_perfect_pass": perfect_recall_rate(Xp, X),
                    "x_bits_pass": bit_accuracy(Xp, X),
                    "x_perfect_resonance": perfect_recall_rate(Xr, X),
                    "y_perfect_pass": perfect_recall_rate(Yr, Y),
                })
        print(f"  seed {seed + 1}/{N_SEEDS}", flush=True)
    return pd.DataFrame(rows)


def _row(seed, scenario, level, rule, Xr, X, Yp, Y) -> dict:
    """One result row for the scenarios that corrupt layer X."""
    return {
        "seed": seed,
        "scenario": scenario,
        "level": float(level),
        "rule": rule,
        "x_perfect_resonance": perfect_recall_rate(Xr, X),
        "x_bits_resonance": bit_accuracy(Xr, X),
        "y_perfect_pass": perfect_recall_rate(Yp, Y),
        "y_bits_pass": bit_accuracy(Yp, Y),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95 % CI across seeds, for whichever metrics are present."""
    metrics = [c for c in df.columns
               if c not in ("seed", "scenario", "level", "rule")]
    out = []
    for (sc, level, rule), g in df.groupby(["scenario", "level", "rule"],
                                           sort=False):
        row = {"scenario": sc, "level": level, "rule": rule}
        for m in metrics:
            values = g[m].dropna().to_numpy()
            if values.size:
                mean, ci = confidence_interval(values)
                row[m], row[f"{m}_ci95"] = mean, ci
        out.append(row)
    return pd.DataFrame(out)


def critical_level(res: pd.DataFrame, scenario: str, metric: str,
                   threshold: float = 0.90) -> pd.DataFrame:
    """Largest corruption level at which the metric still reaches `threshold`."""
    out = []
    for rule, g in res[res.scenario == scenario].groupby("rule", sort=False):
        ok = g.sort_values("level")
        ok = ok[ok[metric] >= threshold]["level"]
        out.append({"scenario": scenario, "rule": rule, "metric": metric,
                    "critical_level": float(ok.max()) if len(ok) else float("nan")})
    return pd.DataFrame(out)


def figure_robustness(res: pd.DataFrame):
    """Small multiples: one corruption scenario per column."""
    viz.apply_theme()
    panels = [
        ("noise_x", "x_perfect_resonance", "Fraction of face bits flipped"),
        ("occlusion_x", "x_perfect_resonance", "Fraction of the face occluded"),
        ("noise_y", "x_perfect_pass", "Fraction of code bits flipped"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.9), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.985, top=0.60, bottom=0.155,
                        wspace=0.09)

    for ax, (sc, metric, xlabel) in zip(axs, panels, strict=True):
        for order, rule in enumerate(RULES):
            g = res[(res.scenario == sc) & (res.rule == rule)].sort_values("level")
            x, y = g["level"].to_numpy(), g[metric].to_numpy()
            ci = g[f"{metric}_ci95"].to_numpy()
            color = viz.RULE_COLOR[rule]
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.16, lw=0,
                            zorder=2)
            ax.plot(x, y, color=color, lw=5.2 - 1.1 * order,
                    marker=viz.RULE_MARKER[rule], ms=8.5 - 1.0 * order,
                    markeredgecolor=viz.SURFACE, markeredgewidth=1.2,
                    label=RULE_LABELS[rule], zorder=3 + order)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.04, 1.06)
        ax.xaxis.set_major_formatter(viz.pct_fmt(0))
        ax.yaxis.set_major_formatter(viz.pct_fmt(0))
        ax.set_title(PANEL_TITLES[sc], fontsize=10.5, linespacing=1.5)
    axs[0].set_ylabel("Faces recalled with no wrong bit")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=4,
               bbox_to_anchor=(0.988, 0.845), columnspacing=1.8, handlelength=1.9)
    viz.figure_title(
        fig,
        "BAM robustness depends on which layer gets corrupted",
        f"P = {P} stored pairs · Hadamard codes · mean of {N_SEEDS} seeds, "
        f"band = 95 % CI",
    )
    viz.credit(fig)
    target = FIGURES_DIR / "fig03_robustness.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


GIF_LEVELS = np.round(np.arange(0.0, 0.81, 0.10), 3)


def robustness_gif(data):
    """Square GIF contrasting the two ways of destroying the probe.

    Both columns are damaged by the same fraction at the same time, so the
    comparison is a controlled one: what changes is only *how* the pixels are
    lost. Flipped pixels assert the opposite of the truth; erased pixels merely
    fall silent, entering the network with zero local field.

    The curves are computed here rather than read from the statistical sweep so
    that the animation and its plot describe exactly the same runs.
    """
    idx = one_index_per_subject(data.target)[:P]
    X = data.X[:, idx]
    Y = build_codebook("hadamard", P, CODE_LEN)
    model = BAM("dense", max_iter=MAX_ITER).fit(X, Y)
    n_examples = 2
    rng = np.random.default_rng(SEED)
    draw = rng.random(X.shape)  # cumulative mask: the noise only grows

    kinds = ("noise", "occlusion")
    labels = {"noise": ptbr.NOISE_LABEL, "occlusion": ptbr.OCCLUSION_LABEL}
    colors = {"noise": viz.SERIES[1], "occlusion": viz.SERIES[2]}
    markers = {"noise": "o", "occlusion": "^"}

    frames, curves = [], {k: [] for k in kinds}
    for level in GIF_LEVELS:
        probes, recovered = {}, {}
        for kind in kinds:
            probe = (np.where(draw < level, -X, X) if kind == "noise"
                     else occlude(X, float(level), data.side))
            Xr, _, _, _ = model.recall_batch(X0=probe)
            probes[kind] = probe[:, :n_examples]
            recovered[kind] = Xr[:, :n_examples]
            curves[kind].append(perfect_recall_rate(Xr, X))
        frames.append((float(level), probes, recovered))
    curves = {k: np.column_stack([GIF_LEVELS, v]) for k, v in curves.items()}

    fig, top = new_square_figure("03_robustness")
    badge_y, top = top + 0.008, top - BADGE_GAP
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.15],
        left=0.135, right=0.965, top=top, bottom=PANEL_BOTTOM,
        hspace=0.38, wspace=0.10,
    )
    ax_probe = [fig.add_subplot(gs[0, k]) for k in range(2)]
    ax_out = [fig.add_subplot(gs[1, k]) for k in range(2)]
    ax_c = fig.add_subplot(gs[2, :])
    ax_c.set_xlim(-0.02, float(max(GIF_LEVELS)) * 1.34)
    ax_c.set_ylim(-0.05, 1.10)
    ax_c.set_xlabel(ptbr.AXIS_CORRUPTION)
    ax_c.set_ylabel(ptbr.AXIS_PERFECT_RECALL, fontsize=13)
    ax_c.xaxis.set_major_formatter(viz.pct_fmt(0, ptbr=True))
    ax_c.yaxis.set_major_formatter(viz.pct_fmt(0, ptbr=True))

    movable: list = []

    def update(i: int) -> None:
        nonlocal movable
        for a in movable:
            a.remove()
        movable = []
        level, probes, recovered = frames[i]
        for col, kind in enumerate(kinds):
            for ax in (ax_probe[col], ax_out[col]):
                ax.clear()
                ax.set_box_aspect(1.0)
            viz.face_panel(ax_probe[col], probes[kind][:, 0], data.side,
                           labels[kind], border=viz.INK_MUTED,
                           title_size=PANEL_TITLE_SIZE)
            wrong = float((recovered[kind][:, 0] != X[:, 0]).mean())
            caption = (ptbr.EXACT_SHORT if wrong == 0 else
                       ptbr.WRONG_SHORT.format(
                           value=viz.format_percent(wrong, 1, ptbr=True)))
            viz.face_panel(ax_out[col], recovered[kind][:, 0], data.side,
                           ptbr.PANEL_RECOVERED, subtitle=caption,
                           border=viz.GOOD if wrong == 0 else viz.CRITICAL,
                           title_size=PANEL_TITLE_SIZE,
                           subtitle_size=PANEL_SUBTITLE_SIZE)
        ends = []
        for order, kind in enumerate(kinds):
            d = curves[kind]
            movable.append(series_line(ax_c, d[: i + 1, 0], d[: i + 1, 1],
                                       color=colors[kind], marker=markers[kind],
                                       order=order))
            ends.append((d[i, 0], d[i, 1], labels[kind].split(" —")[0]))
        movable += label_ends(ax_c, ends)
        movable.append(badge(fig, badge_y, ptbr.CORRUPTION_LEVEL.format(
            value=viz.format_percent(level, 0, ptbr=True))))
        if i == len(frames) - 1:
            movable += takeaway(fig, "03_robustness")

    return save_gif(fig, update, len(frames), "03_robustness",
                    fps=2, hold_first=2, hold_last=12)


def main() -> None:
    t0 = header(
        3, "Robustness to noise and occlusion",
        "How much corruption does the probe tolerate, and does the answer "
        "change with the corrupted layer?",
    )
    data = load_dataset(method="mean-face")
    print(f"P = {P} pairs · p = {CODE_LEN} · Hadamard codes · {N_SEEDS} seeds\n")

    df = sweep(data)
    res = summarize(df)
    files = [save_table(df, "tab03_robustness.csv"),
             save_table(res, "tab03_robustness_summary.csv")]

    critical = pd.concat([
        critical_level(res, "noise_x", "x_perfect_resonance"),
        critical_level(res, "occlusion_x", "x_perfect_resonance"),
        critical_level(res, "noise_y", "x_perfect_pass"),
    ])
    files.append(save_table(critical, "tab03_critical_levels.csv"))

    print("\nMaximum corruption tolerated with >= 90 % perfect recall:")
    print(critical.pivot_table(index="rule", columns="scenario",
                               values="critical_level").to_string(na_rep="—"))

    print("\nFragility of exactness (Y→X direction, single pass):")
    for rule in RULES:
        g = res[(res.scenario == "noise_y") & (res.rule == rule)]
        clean = float(g[g.level == 0.0].iloc[0]["x_perfect_pass"])
        ten = float(g[g.level == 0.10].iloc[0]["x_perfect_pass"])
        print(f"  {RULE_LABELS[rule]:22s} noise-free {clean:.2f} → "
              f"with 10 % code noise {ten:.2f}")

    files.append(figure_robustness(res))
    files.append(robustness_gif(data))
    footer(t0, files)


if __name__ == "__main__":
    main()
