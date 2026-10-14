"""
Plot clusters.json's 2D t-SNE of every benchmark instance, coloured by corpus, for the paper.

    python -m analysis.stats.plot_tsne

Each corpus is drawn as its points over a shaded region that encloses most of them, with a
direct label beside it that doubles as the key. Run analysis.stats.cluster first if the
features changed.

Writes analysis/outputs/plots/dataset_tsne.pdf (for the paper) and .png (to look at).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage
from scipy.stats import gaussian_kde

from analysis.paths import OUTPUTS, PLOTS
from analysis.stats.plot_feature_correlation import INK, INK_2

plt.rcParams.update({
    "pdf.fonttype": 42,    # TrueType, not Type 3, which publishers reject
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],   # ships with matplotlib
})

# The reference palette's slots 1-3, which pass every check across all pairs, as a
# scatter needs. Each corpus also gets its own marker, so the panel survives greyscale.
CORPORA = [  # (corpus, paper name, colour, marker), in the paper's order
    ("llm", "LLM-Synthesized", "#2a78d6", "o"),
    ("grammar", "Grammar-Sampled", "#eb6834", "D"),
    ("legal", "Legally Grounded", "#1baf7a", "^"),
]
TSNE_SIZE = (2.69, 3.1)    # inches; the paper's subfigure, 0.49 of acmsmall's text width
BANDWIDTH = 3.5            # t-SNE units; smooths each corpus into a region
REGION_SHARE = 0.95        # each region encloses this share of its corpus's points
GAP = 1.5                  # t-SNE units kept clear around a label
CROWDED = 12               # t-SNE units; a label nearer than this to another corpus is penalised
# A side of its main region that a corpus's label must go on, where the spot chosen
# automatically reads worse. A corpus left out goes wherever place_label finds best.
LABEL_SIDE = {"llm": "right"}
# Marker areas (pt^2), set so the three shapes look about equally heavy.
MARK_SIZE = {"o": 7, "D": 4.5, "^": 7.5}


def region(xy, grid):
    """The density of `xy` over `grid`, and the level that encloses REGION_SHARE of `xy`."""
    # gaussian_kde scales its bandwidth by the data's spread; undo that, so every corpus is
    # smoothed by the same BANDWIDTH whatever its spread.
    kde = gaussian_kde(xy.T)
    kde.set_bandwidth(BANDWIDTH / np.sqrt(np.diag(kde.covariance / kde.factor ** 2).mean()))
    z = kde(grid.reshape(2, -1)).reshape(grid.shape[1:])
    return z, np.quantile(kde(xy.T), 1 - REGION_SHARE)


def darker(colour, by=0.3):
    """`colour` mixed with black, for the outline of a mark."""
    return tuple(c * (1 - by) for c in to_rgb(colour))


def place_label(size, own, others, blocked, cell, allowed, centre, along=(1, 1)):
    """Where to centre a label of `size` (data units): right beside its own corpus's main
    region, preferring a side away from the other corpora.

    `own` and `others` are the distances, from every grid cell, to that region and to the
    other corpora's regions; `blocked` marks the cells a label must not cover (regions,
    points, earlier labels, and everything outside the axes). A spot only counts if the
    label is clearly nearer its own corpus than any other, so a reader never wonders which
    group it names, and another corpus only costs a spot when it is within CROWDED. Only
    centres in `allowed` are considered. Among spots equally near the region, the one
    nearest `centre`, the region's middle (grid indices), wins, so a label sits level
    with its group rather than at a corner; `along` weighs the x and y offsets, so a label
    held to the left or right is only pulled level with the middle, not towards it.
    Returns the centre's grid indices.
    """
    half = [int(np.ceil((size[k] / 2 + GAP) / cell[k])) for k in (0, 1)]
    footprint = (2 * half[0] + 1, 2 * half[1] + 1)
    hits = ndimage.maximum_filter(blocked, size=footprint, mode="constant", cval=True)
    d_own = ndimage.minimum_filter(own, size=footprint)
    d_other = ndimage.minimum_filter(others, size=footprint)
    ok = allowed & ~hits & (d_own + 3 < d_other)
    ii, jj = np.indices(own.shape)
    off_centre = np.hypot(along[0] * (ii - centre[0]) * cell[0],
                          along[1] * (jj - centre[1]) * cell[1])
    score = np.where(ok, d_own + 0.3 * np.maximum(CROWDED - d_other, 0) + 0.15 * off_centre,
                     np.inf)
    return np.unravel_index(np.argmin(score), score.shape), half


def plot_tsne(clusters, out_dir):
    points = clusters["points"]
    xy = {c: np.array([p["tsne_2d"] for p in points if p["corpus"] == c]) for c, *_ in CORPORA}
    every = np.vstack(list(xy.values()))
    fig = plt.figure(figsize=TSNE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    # t-SNE distances are isotropic, so one unit is as long across as up: widen whichever
    # range is short for the panel's shape, keeping the points centred.
    lo, hi = every.min(axis=0) - 6, every.max(axis=0) + 6
    shape = np.array([1, TSNE_SIZE[1] / TSNE_SIZE[0]])
    unit = max((hi - lo) / shape)
    mid = (lo + hi) / 2
    lo, hi = mid - unit * shape / 2, mid + unit * shape / 2
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    # One grid over the whole panel, for the regions and for placing the labels.
    n = 400
    grid = np.mgrid[lo[0]:hi[0]:n * 1j, lo[1]:hi[1]:n * 1j]
    cell = (hi - lo) / (n - 1)

    def cells(pts):
        """Grid indices of `pts`."""
        return np.clip(np.rint((pts - lo) / cell).astype(int), 0, n - 1)

    # Regions first, then points, biggest corpus first, so the smaller ones sit on top.
    order = sorted(CORPORA, key=lambda c: -len(xy[c[0]]))
    masks = {}
    for corpus, _, colour, _ in order:
        z, level = region(xy[corpus], grid)
        ax.contourf(grid[0], grid[1], z, levels=[level, z.max()], colors=[colour], alpha=0.12)
        ax.contour(grid[0], grid[1], z, levels=[level], colors=[colour], linewidths=0.6, alpha=0.7)
        masks[corpus] = z >= level
    for corpus, _, colour, marker in order:
        ax.scatter(xy[corpus][:, 0], xy[corpus][:, 1], s=MARK_SIZE[marker], marker=marker,
                   facecolors=colour, edgecolors=darker(colour), linewidths=0.3, alpha=0.85,
                   zorder=3)

    # A direct label beside each corpus, which is also the key: its marker, name and size.
    # It goes beside the corpus's main region, the one holding most of its points.
    taken = np.zeros((n, n), bool)
    for pts in cells(every):
        taken[pts[0], pts[1]] = True
    taken = ndimage.binary_dilation(taken, iterations=2) | np.any(list(masks.values()), axis=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    for corpus, name, colour, marker in CORPORA:
        parts, _ = ndimage.label(masks[corpus])
        idx = cells(xy[corpus])
        hit = parts[idx[:, 0], idx[:, 1]]          # the region part each point falls in, 0 for none
        main = parts == np.bincount(hit[hit > 0]).argmax()
        others = np.any([m for c, m in masks.items() if c != corpus], axis=0)
        own = ndimage.distance_transform_edt(~main, sampling=cell)
        other = ndimage.distance_transform_edt(~others, sampling=cell)

        title = ax.text(0, 0, name, fontsize=7.4, fontweight="bold", color=INK, va="top")
        sub = ax.text(0, 0, f"{len(xy[corpus])} constraints", fontsize=6.4, color=INK_2, va="top")
        bb = [t.get_window_extent(renderer) for t in (title, sub)]
        swatch = 11   # display pixels: room for the marker before the text
        width_px = swatch + max(b.width for b in bb)
        (x0, y0), (x1, y1) = to_data.transform([[0, 0], [width_px, bb[0].height + bb[1].height + 1]])
        w, h = x1 - x0, y1 - y0
        ii, jj = np.indices((n, n))
        xs, ys = np.nonzero(main)
        side = LABEL_SIDE.get(corpus)
        allowed = {   # centre past the region's middle on that side, level with or beside it
            "right": (ii > xs.mean()) & (jj >= ys.min()) & (jj <= ys.max()),
            "left": (ii < xs.mean()) & (jj >= ys.min()) & (jj <= ys.max()),
            "above": (jj > ys.mean()) & (ii >= xs.min()) & (ii <= xs.max()),
            "below": (jj < ys.mean()) & (ii >= xs.min()) & (ii <= xs.max()),
            None: np.ones((n, n), bool),
        }[side]
        along = {"right": (0, 1), "left": (0, 1), "above": (1, 0), "below": (1, 0)}.get(side, (1, 1))
        (i, j), half = place_label((w, h), own, other, taken, cell, allowed,
                                   (xs.mean(), ys.mean()), along)
        cx, cy = grid[0][i, j], grid[1][i, j]
        left, top = cx - w / 2, cy + h / 2
        ax.scatter([left + w * swatch * 0.4 / width_px], [top - h * 0.26], s=MARK_SIZE[marker] * 2.2,
                   marker=marker, facecolors=colour, edgecolors=darker(colour), linewidths=0.4,
                   zorder=4)
        title.set_position((left + w * swatch / width_px, top))
        sub.set_position((left + w * swatch / width_px, top - h * 0.53))
        taken[max(i - half[0], 0):i + half[0] + 1, max(j - half[1], 0):j + half[1] + 1] = True

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"dataset_tsne.{ext}", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_dir / 'dataset_tsne'}.pdf and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", default=str(OUTPUTS / "clusters.json"))
    ap.add_argument("--output-dir", default=str(PLOTS))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.clusters) as f:
        plot_tsne(json.load(f), out_dir)


if __name__ == "__main__":
    main()
