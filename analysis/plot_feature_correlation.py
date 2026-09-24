"""
Plot the Spearman correlation between features over every benchmark instance.

    python analysis/plot_feature_correlation.py

Features only, no runtimes, so all instances in features.csv are used. Features are
ordered by complete-linkage clustering on 1 - |rho|; a family is a group in which
every pair has |rho| >= --family, and is outlined on the plot.

Writes analysis/plots/feature_correlation.png.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).parent
# Diverging scale: red (negative) -> grey (zero) -> green (positive). Green is the
# desired direction everywhere it is used: for speedups it means the encoding gains.
CMAP = LinearSegmentedColormap.from_list(
    "red_gray_green", ["#a12a28", "#e34948", "#f0efec", "#3db07a", "#1b7a4e"])
INK, INK_2, SURFACE = "#1f1f1e", "#52514e", "#fcfcfb"


def ink_on(color):
    """Text colour that contrasts with `color` (any matplotlib colour spec)."""
    def lin(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in to_rgb(color))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.36 else INK


def constant_features(df):
    """Feature columns with a single value over all instances."""
    cols = [c for c in df.columns if c not in ("id", "corpus") and not c.startswith("label_")]
    return [c for c in cols if df[c].nunique(dropna=True) < 2]


def feature_correlation(df, family=0.8):
    """Feature-by-feature Spearman rho, ordered by complete-linkage clustering on 1 - |rho|.

    Returns (rho, order, family_of): rho reindexed to `order`, and a dict feature ->
    family id, where every pair inside a family has |rho| >= `family`.

    Constant columns are left out: their correlation with anything is undefined, and a
    NaN in the distance matrix breaks the clustering. `constant_features` lists them.
    """
    cols = [c for c in df.columns if c not in ("id", "corpus") and not c.startswith("label_")]
    cols = [c for c in cols if c not in constant_features(df)]
    rho = df[cols].astype(float).corr(method="spearman")
    dist = 1 - rho.abs().values
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), "complete")
    order = [cols[i] for i in leaves_list(Z)]
    family_of = dict(zip(cols, (int(x) for x in fcluster(Z, 1 - family, "distance"))))
    return rho.loc[order, order], order, family_of


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(HERE / "features.csv"))
    ap.add_argument("--family", type=float, default=0.8, help="min |rho| between every pair in a family")
    ap.add_argument("--output", default=str(HERE / "plots/feature_correlation.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    const = constant_features(df)
    if const:
        print(f"left out {len(const)} constant column(s): {', '.join(const)}")
    rho, order, family = feature_correlation(df, args.family)

    fig, ax = plt.subplots(figsize=(17, 15), facecolor=SURFACE)
    im = ax.imshow(rho.values, cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
    n = len(order)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=9, color=INK_2)
    ax.set_yticklabels(order, fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue            # diagonal is 1 by definition
            v = rho.values[i, j]
            ax.text(j, i, f"{v:+.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=7, color=ink_on(CMAP((v + 1) / 2)))

    start = 0
    for k in range(1, n + 1):
        if k == n or family[order[k]] != family[order[start]]:
            if k - start > 1:
                ax.add_patch(Rectangle((start - 0.5, start - 0.5), k - start, k - start,
                                       fill=False, edgecolor=INK, linewidth=2))
            start = k

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Spearman rho", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title(f"Feature correlation, all {len(df)} instances", loc="left", fontsize=15,
                 color=INK, fontweight="semibold", pad=28)
    ax.text(0, 1.012, f"Green: positive, red: negative. Boxes = families (every pair |rho| >= {args.family}).",
            transform=ax.transAxes, fontsize=10, color=INK_2)

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
