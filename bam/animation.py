"""GIF assembly for the animated results.

Four animations are produced, one per question the study answers. Each is built
to the same rules:

* **One message per animation.** The title states a question, the panels answer
  it, and the last frames spell out the conclusion in words — a viewer should
  never have to infer the punchline from the axes.
* **A fixed frame of reference.** Axis limits are set before the first frame, so
  what moves on screen is the data and never the scale. Space for the closing
  caption is reserved from the start, so nothing shifts when it appears.
* **Few, large marks.** Two example faces rather than six small ones; thick
  lines; type no smaller than about 2 % of the frame width.
* **Identity never by colour alone.** Every series carries a marker shape and a
  direct end label as well as its colour.
* **Square 1080×1080 canvas**, because these are read on phone screens where a
  wide figure shrinks until its labels vanish.

All rendered text is Portuguese and comes from `bam.ptbr`.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from bam import ptbr, viz
from bam.config import FPS, GIF_FIGSIZE, GIFS_DIR
from bam.core import BAM

# Type scale used by every animation, in points at 100 dpi.
FONT_SCALE = 1.8
TITLE_SIZE = 30.0
SUBTITLE_SIZE = 15.5
PANEL_TITLE_SIZE = 15.0
PANEL_SUBTITLE_SIZE = 13.0
BODY_SIZE = 14.5
TAKEAWAY_SIZE = 15.0
CREDIT_SIZE = 11.0

# Vertical band reserved at the bottom for the closing caption. Reserved from
# the first frame so that its appearance never reflows the panels above; the
# margin also has to clear the bottom axis label.
TAKEAWAY_Y = 0.046
PANEL_BOTTOM = 0.170

# Vertical room reserved above the panels for the sweep badge ("P = 200 pairs",
# "80% corrupted"). It gets its own line so it can never land on a panel title.
BADGE_GAP = 0.042

_SUPERSCRIPTS = str.maketrans(
    "0123456789",
    "⁰¹²³⁴⁵⁶⁷⁸⁹",
)


def _superscript(n: int) -> str:
    """Render an integer with superscript digits (for scale labels)."""
    return str(n).translate(_SUPERSCRIPTS)


def new_square_figure(name: str, **fmt) -> tuple:
    """Create the square canvas with its Portuguese title block already set.

    Args:
        name: key into `ptbr.TITLES`.
        **fmt: substitutions for the title/subtitle format strings.

    Returns:
        `(fig, top)` — the figure and the y coordinate below which the panels
        may start.
    """
    viz.apply_theme(scale=FONT_SCALE)
    title, subtitle = ptbr.TITLES[name]
    fig = plt.figure(figsize=GIF_FIGSIZE)
    # A 30 pt line occupies ~0.047 of a 1080 px canvas, a 15.5 pt line ~0.025.
    # The title block is measured rather than guessed so that two-line titles
    # never run into the subtitle.
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    title_block = 0.047 * title_lines + 0.012
    viz.figure_title(
        fig, title, subtitle.format(**fmt),
        title_size=TITLE_SIZE, subtitle_size=SUBTITLE_SIZE,
        y=0.985, gap=title_block,
    )
    viz.credit(fig, ptbr.CREDIT, fontsize=CREDIT_SIZE, y=0.010)
    top = 0.985 - title_block - 0.026 * subtitle_lines - 0.030
    return fig, top


def takeaway(fig, name: str, *, color: str | None = None,
             width: int = 66) -> list:
    """Draw the animation's closing caption in the reserved bottom band.

    The text is wrapped here rather than in `bam.ptbr` so that the strings stay
    editable prose instead of pre-broken lines.

    Returns the artists created, so the caller can remove them on the next
    frame of a loop.
    """
    text = textwrap.fill(ptbr.TAKEAWAYS[name], width=width)
    color = viz.SERIES[2] if color is None else color
    bar = fig.text(0.012, TAKEAWAY_Y, "▍", ha="left", va="bottom",
                   fontsize=TAKEAWAY_SIZE + 8, color=color)
    body = fig.text(0.042, TAKEAWAY_Y, text, ha="left", va="bottom",
                    fontsize=TAKEAWAY_SIZE, color=viz.INK_PRIMARY,
                    linespacing=1.45)
    return [bar, body]


def badge(fig, y: float, text: str):
    """The sweep counter, on its own line above the panels."""
    return fig.text(0.012, y, text, fontsize=18, fontweight="bold",
                    color=viz.INK_PRIMARY, ha="left", va="bottom")


def save_gif(
    fig,
    update: Callable[[int], None],
    n_frames: int,
    name: str,
    *,
    fps: int = FPS,
    hold_last: int = 16,
    hold_first: int = 4,
    out_dir: Path | None = None,
) -> Path:
    """Render `n_frames` calls of `update` into a looping GIF.

    Args:
        update: function that draws frame `i` (0-indexed).
        hold_first: extra frames repeating the first state.
        hold_last: extra frames repeating the last state, so the closing
            caption stays on screen long enough to be read.

    Returns:
        Path of the written file. (Pillow merges consecutive identical frames
        into a single longer-duration frame, so the holds cost no file size.)
    """
    out_dir = GIFS_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.gif"

    order = [0] * hold_first + list(range(n_frames)) + [n_frames - 1] * hold_last

    anim = FuncAnimation(
        fig, lambda k: update(order[k]), frames=len(order),
        interval=1000 / fps, blit=False, repeat=True,
    )
    anim.save(target, writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": viz.PAGE})
    plt.close(fig)
    return target


def series_line(ax, x, y, *, color: str, marker: str, order: int = 0,
                zorder_base: int = 3):
    """One data series drawn at the animation's mark weights.

    Line widths decrease with `order` so that coincident series — frequent in
    this study, where several conditions saturate together at 0 % or 100 % —
    remain visible as rings around one another instead of vanishing.
    """
    (line,) = ax.plot(x, y, color=color, lw=7.0 - 1.6 * order, marker=marker,
                      ms=13.0 - 1.6 * order, markeredgecolor=viz.SURFACE,
                      markeredgewidth=1.6, zorder=zorder_base + order)
    return line


def noise_preroll(
    x_clean: np.ndarray, final_prob: float, seed: int, *, n_frames: int = 7
) -> list[tuple[np.ndarray, str]]:
    """Intro frames showing the face being progressively corrupted.

    Without this intro the viewer sees the noisy probe already made and has no
    way to judge how much information was destroyed before recall. The frames
    use **cumulative** noise (the same mask keeps growing), so the last frame is
    exactly the probe used for recall.
    """
    rng = np.random.default_rng(seed)
    draw = rng.random(x_clean.shape)
    frames = []
    for k in range(n_frames):
        prob = final_prob * (k / (n_frames - 1)) if n_frames > 1 else final_prob
        x = np.where(draw < prob, -x_clean, x_clean)
        caption = ptbr.STAGE_CLEAN if k == 0 else ptbr.corrupting(prob * 100)
        frames.append((x, caption))
    return frames


# ---------------------------------------------------------------------------
# Animation 1 — bidirectional recall
# ---------------------------------------------------------------------------
@dataclass
class _Frame:
    """One rendered instant of the bidirectional-recall animation."""

    x: np.ndarray
    y: np.ndarray
    stage: str
    phase: int  # 0 = building the probe, 1 = X→Y→X, 2 = Y→X only
    t: float | None = None  # position on the energy axis
    energy: float | None = None


def bidirectional_recall_gif(
    model: BAM,
    *,
    probe: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    side: int,
    name: str,
    preroll: list[tuple[np.ndarray, str]],
    fps: int = 4,
) -> Path:
    """The defining capability of a BAM, in two acts.

    Act 1 feeds a corrupted face and lets the network resonate until it
    returns both the face and its code. Act 2 then erases the face entirely and
    hands the network nothing but the code — from which the same face
    reappears. One weight matrix, two directions.

    The energy panel carries the theoretical argument: it descends in both
    acts, and jumps only where *we* reset the state, never where the network
    updates it.
    """
    traj_fwd = model.recall(x0=probe)
    traj_bwd = model.recall(y0=y_target)

    e_fwd = np.asarray(traj_fwd.energies)
    e_bwd = np.asarray(traj_bwd.energies)
    n_fwd = len(e_fwd)

    frames: list[_Frame] = [
        _Frame(x=x, y=np.zeros_like(y_target), stage=caption, phase=0)
        for x, caption in preroll
    ]
    for i in range(n_fwd):
        frames.append(_Frame(
            x=traj_fwd.xs[i], y=traj_fwd.ys[i], t=float(i), energy=float(e_fwd[i]),
            stage=_stage_name(traj_fwd.steps[i], i == n_fwd - 1), phase=1,
        ))
    for i in range(len(e_bwd)):
        frames.append(_Frame(
            x=traj_bwd.xs[i], y=traj_bwd.ys[i], t=float(n_fwd + i),
            energy=float(e_bwd[i]), phase=2,
            stage=(ptbr.STAGE_ERASE if i == 0
                   else _stage_name(traj_bwd.steps[i], i == len(e_bwd) - 1)),
        ))

    fig, top = new_square_figure(name)
    gs = fig.add_gridspec(
        4, 2, height_ratios=[1.0, 0.13, 0.34, 0.86],
        left=0.115, right=0.965, top=top, bottom=PANEL_BOTTOM,
        hspace=0.26, wspace=0.10,
    )
    ax_x = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_code = fig.add_subplot(gs[1, :])
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    ax_e = fig.add_subplot(gs[3, :])

    all_energy = np.concatenate([e_fwd, e_bwd])
    e_min, e_max = float(all_energy.min()), float(all_energy.max())
    span = e_max - e_min
    margin = 0.12 * (span + 1e-12)
    # Energies span ~1 (dense rule) to ~1e8 (multiple training); large
    # magnitudes are rescaled by a power of ten spelled out in the axis label.
    exponent = 0
    while span / 10**exponent >= 1e4:
        exponent += 3
    scale = 10.0**exponent
    span_scaled = span / scale
    decimals = 0 if span_scaled >= 10 else (2 if span_scaled >= 0.1 else 4)
    suffix = f"  (× 10{_superscript(exponent)})" if exponent else ""

    total = n_fwd + len(e_bwd)
    ax_e.set_xlim(-0.5, total - 0.5)
    ax_e.set_ylim((e_min - margin) / scale, (e_max + margin) / scale)
    ax_e.set_xlabel(ptbr.AXIS_HALF_ITER)
    ax_e.set_ylabel(f"{ptbr.AXIS_ENERGY}{suffix}")
    ax_e.set_title(ptbr.PANEL_ENERGY, fontsize=PANEL_TITLE_SIZE + 2,
                   color=viz.INK_PRIMARY, pad=8)
    ax_e.yaxis.set_major_formatter(viz.num_fmt(decimals, ptbr=True))
    # The divider marks where we reset the state; the network itself never
    # raises the energy, and the two descents make that visible.
    ax_e.axvline(n_fwd - 0.5, color=viz.INK_MUTED, lw=1.4, ls=(0, (4, 4)),
                 zorder=1)

    movable: list = []

    def update(k: int) -> None:
        nonlocal movable
        for a in movable:
            a.remove()
        movable = []
        ax_x.clear()
        ax_err.clear()
        ax_code.clear()

        f = frames[k]
        viz.face_panel(ax_x, f.x, side, ptbr.PANEL_X_STATE,
                       title_size=PANEL_TITLE_SIZE)
        wrong_x = viz.error_panel(ax_err, f.x, x_target, side, ptbr.PANEL_X_ERROR,
                                  title_size=PANEL_TITLE_SIZE)
        viz.code_panel(ax_code, f.y, ptbr.PANEL_Y_STATE, target=y_target,
                       title_size=PANEL_TITLE_SIZE, marker_size=24)

        if f.phase:
            seg = [g for g in frames if g.phase == f.phase and g.t <= f.t]
            movable.append(ax_e.plot(
                [g.t for g in seg], [g.energy / scale for g in seg],
                color=viz.SERIES[0], lw=3.6, zorder=3)[0])
            movable.append(ax_e.plot(
                [f.t], [f.energy / scale], marker="o", ms=13,
                color=viz.SERIES[0], markeredgecolor=viz.SURFACE,
                markeredgewidth=2.6, zorder=5)[0])
            if f.phase == 2:  # keep act 1 on screen while act 2 runs
                first = [g for g in frames if g.phase == 1]
                movable.append(ax_e.plot(
                    [g.t for g in first], [g.energy / scale for g in first],
                    color=viz.SERIES[0], lw=3.6, alpha=0.45, zorder=2)[0])

        phase_text = ptbr.PHASE_LABELS[max(f.phase - 1, 0)] if f.phase else ""
        lines = [phase_text, f.stage,
                 ptbr.WRONG_BITS.format(
                     value=viz.format_percent(wrong_x, 1, ptbr=True))]
        if f.energy is not None:
            lines.append(ptbr.ENERGY_VALUE.format(
                value=viz.format_number(f.energy, decimals, ptbr=True)))
        movable.append(ax_txt.text(
            0.0, 1.06, "\n".join(lines), transform=ax_txt.transAxes, va="top",
            ha="left", fontsize=BODY_SIZE, color=viz.INK_SECONDARY,
            linespacing=1.45,
        ))

        exact = wrong_x == 0.0 and np.array_equal(f.y, y_target)
        if f.phase and (exact or f.stage == ptbr.STAGE_STABLE):
            status = ptbr.STATUS_RECALLED if exact else ptbr.STATUS_SPURIOUS
            movable.append(ax_txt.text(
                1.0, 1.06, f"● {status}", transform=ax_txt.transAxes,
                ha="right", va="top", fontsize=BODY_SIZE,
                color=viz.GOOD if exact else viz.CRITICAL))

        if k == len(frames) - 1:
            movable += takeaway(fig, name)

    return save_gif(fig, update, len(frames), name, fps=fps,
                    hold_first=3, hold_last=20)


def _stage_name(step: str, is_last: bool) -> str:
    """Human-readable caption for one half-iteration."""
    if is_last:
        return ptbr.STAGE_STABLE
    return {
        "probe": ptbr.STAGE_PROBE_SHOWN,
        "X->Y": ptbr.STAGE_FORWARD,
        "Y->X": ptbr.STAGE_BACKWARD,
    }.get(step, step)


def label_ends(ax, ends: Sequence[tuple[float, float, str]], **kw) -> list:
    """End-of-line labels at a shared x, stacked so they never collide."""
    if not ends:
        return []
    x_label = max(e[0] for e in ends)
    return viz.label_line_ends(
        ax, [(x_label, y, text) for _, y, text in ends],
        fontsize=kw.pop("fontsize", BODY_SIZE - 2.0),
        separation=kw.pop("separation", 0.115), **kw,
    )
