"""One visual style for every figure of the project.

Colour is assigned by the job it does:
  * identity of a few series  -> the fixed categorical order ``SERIES`` (never more than four at
    once here, always with direct labels or a legend);
  * "ours vs. reference"      -> one accent hue + neutral grey;
  * magnitude (heat maps)     -> a single-hue ramp, light = low;
  * ten land-cover classes    -> never ten hues: small multiples, direct labels or a heat map.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#a9a7a0"       # de-emphasised series (reference values, "all other" points)

# categorical slots, fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACCENT = SERIES[0]

# sequential ramp (one hue, light -> dark); the surface colour stands for "zero"
_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", _BLUES)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "figure.titlesize": 12, "figure.titleweight": "semibold",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6.5, "lines.markeredgewidth": 1.6, "lines.markeredgecolor": SURFACE,
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def save(fig, path, **kw) -> None:
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"[fig] {path}")


def heatmap(ax, m: np.ndarray, row_labels, col_labels, fmt: str = "{:.0f}", vmin: float = 0.0,
            vmax: float | None = None, hide_below: float | None = None, fontsize: float = 7.5,
            cmap=SEQ) -> None:
    """Annotated single-hue heat map; cell text flips to white on dark cells."""
    vmax = float(np.nanmax(m)) if vmax is None else vmax
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # 2-px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, m.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isnan(v) or (hide_below is not None and v < hide_below):
                continue
            rel = (v - vmin) / max(1e-12, vmax - vmin)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color="#ffffff" if rel > 0.55 else INK)


def image_grid(images, n_cols: int, titles=None, row_labels=None, cell: float = 0.9,
               title_size: float = 6.5, label_width: float = 1.3, suptitle: str | None = None):
    """Grid of small RGB images (no interpolation), optional per-image titles / row labels."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    extra_h = 0.32 if titles is not None else 0.04
    left = label_width if row_labels is not None else 0.05
    fig_w = left + n_cols * cell + 0.05
    fig_h = n_rows * (cell + extra_h) + (0.45 if suptitle else 0.08)
    fig = plt.figure(figsize=(fig_w, fig_h))
    top = 1 - (0.45 / fig_h if suptitle else 0.02)
    gs = fig.add_gridspec(n_rows, n_cols, left=left / fig_w, right=1 - 0.05 / fig_w, top=top,
                          bottom=0.02, wspace=0.06, hspace=0.06 + extra_h / cell)
    for k, img in enumerate(images):
        ax = fig.add_subplot(gs[k // n_cols, k % n_cols])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if titles is not None:
            ax.set_title(titles[k], fontsize=title_size, pad=2, loc="center", fontweight="normal",
                         color=INK2)
        if row_labels is not None and k % n_cols == 0:
            ax.text(-0.12, 0.5, row_labels[k // n_cols], transform=ax.transAxes, ha="right",
                    va="center", fontsize=8.5, color=INK)
    if suptitle:
        fig.suptitle(suptitle, x=0.02 if row_labels is None else 0.01, ha="left", y=0.995,
                     fontsize=10.5)
    return fig
