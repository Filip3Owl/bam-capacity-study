"""Experiment 5 — Application: telling who it is, and recalling what they look like.

Question
--------
The previous experiments use an artificial one-to-one association. A BAM gets
more interesting when the association is **many-to-one**: several photographs
of the same subject share one identity code. That gives the two directions
distinct semantic roles:

* **X→Y** becomes *identification*: given an unseen photograph, who is this?
* **Y→X** becomes *evocation*: given the name, what does this person look like?

And there is an exact, rather pretty result hiding in there. With Hadamard
codes and the Hebb rule, `W = Σ_k x_k y_kᵀ` and therefore

    W y_c = Σ_k x_k (y_kᵀ y_c) = p · Σ_{k ∈ class c} x_k,

so `sgn(W y_c)` is **exactly the pixel-wise majority vote of that subject's
training photographs**. The Hebbian BAM does not retrieve a stored photograph:
it synthesises the class prototype. The experiment verifies this identity
numerically.

Protocol
--------
7 of the 10 poses of each subject go into training (280 stored pairs), 3 are
held out (120 never-seen photographs). Since the code is a label, the X→Y
output is decoded to the nearest codeword — that is how a real system would
read it. A nearest-neighbour classifier over the same binarised vectors serves
as an honest baseline.

Outputs
-------
`04_identification.gif`, `fig06_identification.png`, `tab05_identification.csv`.
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
    BODY_SIZE,
    PANEL_BOTTOM,
    PANEL_SUBTITLE_SIZE,
    PANEL_TITLE_SIZE,
    badge,
    new_square_figure,
    save_gif,
    takeaway,
)
from bam.codebooks import build_codebook
from bam.config import FIGURES_DIR, SEED
from bam.core import BAM, RULE_LABELS, RULES
from bam.datasets import load_dataset, train_test_split
from bam.metrics import confidence_interval, perfect_recall_rate

CODE_LEN = 64  # 64 >= 40 + 1: the 40 non-degenerate Hadamard codes fit
N_TRAIN = 7  # poses per subject used for training (the other 3 are held out)
N_SEEDS = 6
MAX_ITER = 20


def decode(Yhat: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Index of the nearest (highest-correlation) codeword for each column.

    The raw Y-layer output is rarely a valid code; in a real system it goes
    through a decoder. With a Hadamard codebook this step is the
    maximum-likelihood decoding of an orthogonal code.
    """
    return np.argmax(codes.T @ Yhat, axis=0)


def majority_vote(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Class prototypes: pixel-wise majority vote."""
    classes = np.unique(target)
    out = np.empty((X.shape[0], len(classes)))
    for i, c in enumerate(classes):
        out[:, i] = np.sign(X[:, target == c].sum(axis=1))
    return out


def evaluate(data, seed: int) -> tuple[pd.DataFrame, dict]:
    """Fit the four rules and measure identification and evocation."""
    train, test = train_test_split(data.target, n_train=N_TRAIN, seed=SEED + seed)
    X_tr, y_tr = data.X[:, train], data.target[train]
    X_te, y_te = data.X[:, test], data.target[test]

    classes = np.unique(data.target)
    codes = build_codebook("hadamard", len(classes), CODE_LEN)
    Y_tr = codes[:, y_tr]  # the same code repeated for every pose
    prototypes = majority_vote(X_tr, y_tr)

    rows, artifacts = [], {}
    for rule in RULES:
        model = BAM(rule, max_iter=MAX_ITER, max_mt_epochs=30).fit(X_tr, Y_tr)

        pred_te = decode(model.associate_forward(X_te), codes)
        pred_tr = decode(model.associate_forward(X_tr), codes)
        evoked = model.associate_backward(codes)

        rows.append({
            "seed": seed,
            "rule": rule,
            "test_accuracy": float((pred_te == y_te).mean()),
            "train_accuracy": float((pred_tr == y_tr).mean()),
            "exact_code_test": perfect_recall_rate(
                model.associate_forward(X_te), codes[:, y_te]),
            "evocation_equals_prototype": perfect_recall_rate(evoked, prototypes),
            "evocation_matching_bits": float((evoked == prototypes).mean()),
        })
        artifacts[rule] = {"model": model, "evoked": evoked,
                           "predicted_test": pred_te}

    # Baseline: nearest neighbour by correlation, over the same vectors.
    nn = y_tr[np.argmax(X_tr.T @ X_te, axis=0)]
    rows.append({
        "seed": seed,
        "rule": "nearest_neighbour",
        "test_accuracy": float((nn == y_te).mean()),
        "train_accuracy": 1.0,
        "exact_code_test": float("nan"),
        "evocation_equals_prototype": float("nan"),
        "evocation_matching_bits": float("nan"),
    })

    artifacts["_data"] = {"X_te": X_te, "y_te": y_te, "codes": codes,
                          "prototypes": prototypes, "classes": classes}
    return pd.DataFrame(rows), artifacts


ALL_LABELS = {**RULE_LABELS,
              "nearest_neighbour": "Nearest neighbour (baseline)"}
ALL_COLORS = {**viz.RULE_COLOR, "nearest_neighbour": viz.INK_MUTED}
BAR_LABELS = {
    "hebb": "Hebb",
    "multiple_training": "Multiple training",
    "least_squares": "Least squares",
    "dense": "Dense (exponential)",
    "nearest_neighbour": "Nearest neighbour\n(baseline, no BAM)",
}


def figure_identification(res: pd.DataFrame, artifacts: dict, side: int):
    """Accuracy bars on the left; evoked prototypes on the right."""
    viz.apply_theme()
    d = artifacts["_data"]
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.5, 0.30, 1, 1, 1],
                          left=0.145, right=0.965, top=0.695, bottom=0.135,
                          hspace=0.34, wspace=0.10)
    ax_b = fig.add_subplot(gs[:, 0])

    order = list(RULES) + ["nearest_neighbour"]
    means, errors, colors, labels = [], [], [], []
    for key in order:
        g = res[res.rule == key]["test_accuracy"].to_numpy()
        mean, ci = confidence_interval(g)
        means.append(mean)
        errors.append(ci)
        colors.append(ALL_COLORS[key])
        labels.append(BAR_LABELS[key])

    y = np.arange(len(order))
    ax_b.barh(y, means, height=0.62, color=colors, zorder=3)
    ax_b.errorbar(means, y, xerr=errors, fmt="none", ecolor=viz.INK_SECONDARY,
                  elinewidth=1.4, capsize=4, zorder=4)
    for yi, m in zip(y, means, strict=True):
        ax_b.text(m + 0.018, yi, f"{100 * m:.1f}%", va="center", ha="left",
                  fontsize=9.5, color=viz.INK_PRIMARY)
    ax_b.set_yticks(y, labels, fontsize=9.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.12)
    ax_b.xaxis.set_major_formatter(viz.pct_fmt(0))
    ax_b.set_xlabel("Identification accuracy on unseen photographs")
    ax_b.grid(axis="y", visible=False)
    chance = 1.0 / len(d["classes"])
    ax_b.axvline(chance, color=viz.INK_MUTED, lw=1.1, ls=(0, (5, 4)), zorder=4)
    ax_b.annotate("chance (1/40)", xy=(chance, -0.62), xytext=(4, 0),
                  textcoords="offset points", fontsize=8.5, color=viz.INK_MUTED,
                  va="center", annotation_clip=False)

    for col, c in enumerate((0, 1, 2)):
        ax_p = fig.add_subplot(gs[0, 2 + col])
        viz.face_panel(ax_p, d["prototypes"][:, c], side,
                       "Class prototype" if col == 0 else "",
                       subtitle=f"subject {c + 1}")
        ax_e = fig.add_subplot(gs[1, 2 + col])
        evoked = artifacts["hebb"]["evoked"][:, c]
        same = bool((evoked == d["prototypes"][:, c]).all())
        viz.face_panel(ax_e, evoked, side,
                       "Evoked by the Hebbian BAM" if col == 0 else "",
                       subtitle="identical" if same else "different",
                       border=viz.GOOD if same else viz.CRITICAL)

    viz.figure_title(
        fig,
        "Telling who it is, and recalling what they look like",
        f"{N_TRAIN} poses per subject stored (280 pairs) · 120 never-seen test "
        f"photographs · 40 identities · mean of {N_SEEDS} seeds",
    )
    fig.text(0.012, 0.855,
             "Right: the class prototype (majority vote of the training poses) "
             "and what the Hebbian BAM evokes from the code. They are the same "
             "vector, bit for bit.",
             ha="left", va="top", fontsize=9, color=viz.SERIES[2])
    viz.credit(fig)
    target = FIGURES_DIR / "fig06_identification.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def identification_gif(data, artifacts: dict, accuracy: float,
                       rule: str = "dense", n_examples: int = 10):
    """Square GIF: an unseen photograph goes in, a name and a face come out.

    The left panel is the input the network has never seen. The middle strip is
    the code it produces, marked against the correct identity's codeword. The
    right panel is what the *reverse* direction evokes from the answer — and
    that face is not any stored photograph, it is the class prototype.
    """
    d = artifacts["_data"]
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(d["X_te"].shape[1], n_examples, replace=False)
    model = artifacts[rule]["model"]
    predictions = artifacts[rule]["predicted_test"]

    fig, top = new_square_figure("04_identification", n_train=N_TRAIN)
    badge_y, top = top + 0.008, top - BADGE_GAP
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 0.17, 0.34],
        left=0.055, right=0.965, top=top, bottom=PANEL_BOTTOM,
        hspace=0.34, wspace=0.10,
    )
    ax_photo = fig.add_subplot(gs[0, 0])
    ax_evoked = fig.add_subplot(gs[0, 1])
    ax_code = fig.add_subplot(gs[1, :])
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    movable: list = []
    hits = 0

    def update(i: int) -> None:
        nonlocal movable, hits
        for a in movable:
            a.remove()
        movable = []
        for ax in (ax_photo, ax_evoked, ax_code):
            ax.clear()
        if i == 0:
            hits = 0

        col = chosen[i]
        truth = int(d["y_te"][col])
        pred = int(predictions[col])
        right = pred == truth
        hits = sum(1 for k in chosen[: i + 1]
                   if int(predictions[k]) == int(d["y_te"][k]))

        for ax in (ax_photo, ax_evoked):
            ax.set_box_aspect(1.0)
        viz.face_panel(ax_photo, d["X_te"][:, col], data.side,
                       ptbr.PANEL_TEST_PHOTO,
                       subtitle=ptbr.TRUE_SUBJECT.format(k=truth + 1),
                       border=viz.INK_MUTED, title_size=PANEL_TITLE_SIZE,
                       subtitle_size=PANEL_SUBTITLE_SIZE)
        viz.face_panel(ax_evoked, artifacts[rule]["evoked"][:, pred], data.side,
                       ptbr.PANEL_PROTOTYPE,
                       subtitle=ptbr.ANSWERED.format(k=pred + 1),
                       border=viz.GOOD if right else viz.CRITICAL,
                       title_size=PANEL_TITLE_SIZE,
                       subtitle_size=PANEL_SUBTITLE_SIZE)

        y_hat = model.associate_forward(d["X_te"][:, col : col + 1])[:, 0]
        viz.code_panel(ax_code, y_hat, ptbr.PANEL_NETWORK_CODE,
                       target=d["codes"][:, truth],
                       title_size=PANEL_TITLE_SIZE, marker_size=24)

        movable.append(ax_txt.text(
            0.0, 1.05, ptbr.CORRECT if right else ptbr.WRONG,
            transform=ax_txt.transAxes, va="top", ha="left", fontsize=BODY_SIZE + 3,
            color=viz.GOOD if right else viz.CRITICAL,
        ))
        movable.append(ax_txt.text(
            0.0, 0.55, ptbr.ACCURACY_NOTE.format(
                value=viz.format_percent(accuracy, 1, ptbr=True)),
            transform=ax_txt.transAxes, va="top", ha="left", fontsize=BODY_SIZE,
            color=viz.INK_SECONDARY,
        ))
        movable.append(ax_txt.text(
            0.0, 0.10, ptbr.RED_SQUARES_NOTE, transform=ax_txt.transAxes,
            va="top", ha="left", fontsize=BODY_SIZE - 3, color=viz.INK_MUTED,
        ))
        movable.append(badge(fig, badge_y, ptbr.PHOTO_COUNTER.format(
            k=i + 1, n=n_examples) + f"  ·  {hits}/{i + 1}"))
        if i == n_examples - 1:
            movable += takeaway(fig, "04_identification")

    return save_gif(fig, update, n_examples, "04_identification",
                    fps=2, hold_first=1, hold_last=12)


def main() -> None:
    t0 = header(
        5, "Application — face identification and prototype evocation",
        "Is the BAM good for anything once the association is many-to-one?",
    )
    data = load_dataset(method="mean-face")
    print(f"Protocol: {N_TRAIN} training poses and {10 - N_TRAIN} test poses per "
          f"subject · 40 identities · p = {CODE_LEN}\n")

    parts, artifacts = [], None
    for seed in range(N_SEEDS):
        df, art = evaluate(data, seed)
        parts.append(df)
        if artifacts is None:
            artifacts = art
        print(f"  seed {seed + 1}/{N_SEEDS}", flush=True)
    res = pd.concat(parts, ignore_index=True)
    files = [save_table(res, "tab05_identification.csv")]

    summary = (res.groupby("rule", sort=False).agg(["mean", "std"])["test_accuracy"]
               .reindex(list(RULES) + ["nearest_neighbour"]))
    print("\nIdentification accuracy on unseen photographs:")
    for key, row in summary.iterrows():
        print(f"  {ALL_LABELS[key]:34s} {float(row['mean']):.3f} "
              f"± {float(row['std']):.3f}")

    print("\nEvocation: does the Y→X output match the class prototype?")
    for rule in RULES:
        g = res[res.rule == rule]
        print(f"  {RULE_LABELS[rule]:22s} classes identical to prototype: "
              f"{float(g.evocation_equals_prototype.mean()):.2f} · "
              f"matching bits: {float(g.evocation_matching_bits.mean()):.4f}")

    files.append(figure_identification(res, artifacts, data.side))
    files.append(identification_gif(
        data, artifacts,
        accuracy=float(res[res.rule == "dense"].test_accuracy.mean()),
    ))
    footer(t0, files)


if __name__ == "__main__":
    main()
