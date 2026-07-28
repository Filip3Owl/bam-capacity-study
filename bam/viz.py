"""Dark theme and shared plotting primitives for every figure and animation.

Visual design decisions
-----------------------
* **Pure black surface** (`#000000`). The categorical palette below was
  validated against that surface: lightness band, chroma floor, colour-vision-
  deficiency separation (worst adjacent pair ΔE 8.4 protan), normal-vision
  floor (ΔE 19.8) and contrast ≥ 3:1 — all pass.
* **At most four series** per line chart, which is exactly the number of
  learning rules compared. Colours always follow the same fixed order and are
  never recycled.
* **Colour follows the entity**: each learning rule keeps one colour throughout
  the project, so filtering series never repaints the survivors.
* **Identity is never carried by colour alone**: every series also gets a
  legend entry and, where it fits, a distinct marker and a direct end label.
* Text uses neutral ink (white/grey), never the series colour; grid and axes
  stay recessive.

Language: static figures are labelled in English by their callers; the
animations pass Portuguese strings (see `bam.ptbr`). The primitives here are
language-agnostic — every piece of text arrives as an argument.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FuncFormatter

# --- Surfaces and ink -------------------------------------------------------
SURFACE = "#000000"
PAGE = "#000000"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"

# --- Categorical palette (fixed order, validated on a black surface) --------
SERIES = ("#3987e5", "#d95926", "#199e70", "#c98500")

# --- Status colours (reserved; never used as "series 5") --------------------
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# --- Ramps ------------------------------------------------------------------
BLUE_SEQUENTIAL = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "bam_blue", list(reversed(BLUE_SEQUENTIAL))
)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "bam_blue_red", ["#184f95", "#3987e5", "#383835", "#e66767", "#d03b3b"]
)
# Binary map for the error panels: a correct bit fades into the background,
# a wrong bit lights up in the critical-status red.
CMAP_ERROR = ListedColormap(["#101010", CRITICAL])

# --- One fixed colour and marker per entity ---------------------------------
RULE_COLOR = {
    "hebb": SERIES[0],
    "multiple_training": SERIES[1],
    "least_squares": SERIES[2],
    "dense": SERIES[3],
}
RULE_MARKER = {
    "hebb": "o",
    "multiple_training": "s",
    "least_squares": "^",
    "dense": "D",
}
CODEBOOK_COLOR = {
    "hadamard": SERIES[0],
    "random": SERIES[1],
    "correlated": SERIES[2],
}

FONT = ["DejaVu Sans"]


def apply_theme(*, scale: float = 1.0) -> None:
    """Install the dark theme into matplotlib's global `rcParams`.

    Args:
        scale: multiplier for every font size. The animations use a value above
            1 because they are read at phone size, where the default sizes are
            illegible.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": PAGE,
            "figure.edgecolor": PAGE,
            "savefig.facecolor": PAGE,
            "savefig.edgecolor": PAGE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.linewidth": 0.8 * scale,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 12 * scale,
            "axes.titleweight": "bold",
            "axes.labelsize": 10 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7 * scale,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "text.color": INK_PRIMARY,
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "legend.frameon": False,
            "legend.fontsize": 9 * scale,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0 * scale,
            "lines.markersize": 6.0 * scale,
            "lines.solid_capstyle": "round",
            "image.cmap": "gray",
            "figure.dpi": 100,
        }
    )


# --- Number formatting ------------------------------------------------------
# Thin space as the thousands separator: it is unambiguous in both English and
# Brazilian Portuguese, where the comma and the period swap roles.
THIN_SPACE = "\u2009"


def format_number(value: float, decimals: int = 0, *, ptbr: bool = False) -> str:
    """Format a number, optionally with the Brazilian decimal comma."""
    text = f"{value:,.{decimals}f}".replace(",", THIN_SPACE)
    return text.replace(".", ",") if ptbr else text


def format_percent(value: float, decimals: int = 0, *, ptbr: bool = False) -> str:
    """Format a fraction as a percentage string."""
    text = f"{100 * value:.{decimals}f}"
    return (text.replace(".", ",") if ptbr else text) + "%"


def num_fmt(decimals: int = 0, *, ptbr: bool = False) -> FuncFormatter:
    """Axis formatter for plain numbers."""
    return FuncFormatter(lambda v, _p: format_number(v, decimals, ptbr=ptbr))


def pct_fmt(decimals: int = 0, *, ptbr: bool = False) -> FuncFormatter:
    """Axis formatter for percentages."""
    return FuncFormatter(lambda v, _p: format_percent(v, decimals, ptbr=ptbr))


# --- Panel primitives -------------------------------------------------------
def face_panel(
    ax,
    vector: np.ndarray,
    side: int,
    title: str = "",
    *,
    subtitle: str = "",
    border: str | None = None,
    title_size: float = 10.0,
    subtitle_size: float = 8.5,
) -> None:
    """Draw one X-layer vector as an image, with no axes.

    Values in {-1, 0, +1}: a zero (occluded pixel) shows as mid grey, which
    makes occlusion visually distinct from bit noise.
    """
    ax.imshow(vector.reshape(side, side), cmap="gray", vmin=-1, vmax=1,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        if border is not None:
            spine.set_color(border)
            spine.set_linewidth(2.4)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK_PRIMARY, pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=subtitle_size, color=INK_MUTED, labelpad=4)


def error_panel(ax, recalled: np.ndarray, target: np.ndarray,
                side: int, title: str = "", *, title_size: float = 10.0) -> float:
    """Map of the wrong bits; returns the fraction of wrong bits."""
    err = (recalled != target).astype(float).reshape(side, side)
    ax.imshow(err, cmap=CMAP_ERROR, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK_PRIMARY, pad=6)
    return float(err.mean())


def code_panel(ax, y: np.ndarray, title: str = "", *,
               target: np.ndarray | None = None,
               title_size: float = 10.0, marker_size: float = 14.0) -> None:
    """Draw one Y-layer vector as a ribbon of bipolar bits."""
    ax.imshow(y[None, :], cmap="gray", vmin=-1, vmax=1, aspect="auto",
              interpolation="nearest")
    if target is not None:
        wrong = np.flatnonzero(y != target)
        if wrong.size:
            ax.scatter(wrong, np.zeros_like(wrong), s=marker_size, color=CRITICAL,
                       marker="s", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK_PRIMARY, pad=6)


def label_line_ends(
    ax,
    ends: Sequence[tuple[float, float, str]],
    *,
    dx_pt: float = 8.0,
    fontsize: float = 8.5,
    separation: float = 0.055,
) -> list:
    """Direct end-of-line labels, pushed apart so they never collide.

    Series that saturate at the same value — a constant situation in this
    study, where several rules collapse together at 0 % — would otherwise stack
    their labels on the same pixel. Here the labels are sorted by height and
    separated by at least `separation` (in y-axis units), preserving the real
    vertical order of the series.

    Args:
        ends: list of `(x, y, text)` at each series' end point.
        separation: minimum vertical gap between labels, in axis units.

    Returns:
        The artists created, so animations can remove them.
    """
    if not ends:
        return []
    ordered = sorted(ends, key=lambda t: t[1])
    heights: list[float] = []
    for _, y, _ in ordered:
        heights.append(y if not heights else max(y, heights[-1] + separation))

    artists = []
    for (x, _, text), y_adj in zip(ordered, heights, strict=True):
        artists.append(
            ax.annotate(
                text, xy=(x, y_adj), xytext=(dx_pt, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=fontsize, va="center", ha="left",
                annotation_clip=False,
            )
        )
    return artists


DEFAULT_CREDIT = "Olivetti (AT&T/ORL) face database · 400 images · 40 subjects"


def credit(fig, text: str = DEFAULT_CREDIT, *, fontsize: float = 7.5,
           y: float = 0.012) -> None:
    """Discreet data-source note in the figure's bottom-left corner.

    It sits on the left because the bottom-right corner is reserved, in the
    trajectory animations, for the network's final-status message.
    """
    fig.text(0.012, y, text, ha="left", va="bottom", fontsize=fontsize,
             color=INK_MUTED)


def figure_title(fig, title: str, subtitle: str = "", *,
                 title_size: float = 14.0, subtitle_size: float = 9.5,
                 y: float = 0.975, gap: float = 0.05) -> None:
    """Standard title and subtitle at the top of a figure."""
    fig.text(0.012, y, title, ha="left", va="top", fontsize=title_size,
             fontweight="bold", color=INK_PRIMARY)
    if subtitle:
        fig.text(0.012, y - gap, subtitle, ha="left", va="top",
                 fontsize=subtitle_size, color=INK_SECONDARY)
