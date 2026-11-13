"""
Plot how each feature relates to each encoding's speedup over the baseline.

    python -m analysis.stats.plot_speedup_heatmap --timeout 20000     # or set DATESAT_TIMEOUT_MS
    python -m analysis.stats.plot_speedup_heatmap --corpus legal

Reads joined.csv. A cell is the Spearman rho, over the complete instances, between a
feature and one encoding's speedup (the baseline's cost / the encoding's cost). rho > 0
(light, yellow): the encoding gains on the baseline as the feature grows; rho < 0 (dark,
blue): it loses ground. The scale gets lighter all the way from -1 to +1, so it reads the
same in greyscale, and every cell has its signed value written on it.

Costs and instances are instance_costs.py's, as in every other stage: one cost per
instance, the median over its runs, with a timed-out run counting at the timeout, and no
instance left out for its timeouts. So an instance where only one side timed out is ranked
at that bound (the true speedup is at least as extreme, so this is conservative), and one
where both timed out has speedup 1.

Stars mark cells that survive Benjamini-Hochberg correction across the whole grid
(* q < 0.05, ** q < 0.01). Rows keep the column order of features.csv, so every
heatmap (pooled or per corpus) lists features in the same order.

Writes results/analysis/plots/speedup_heatmap[_<corpus>].png.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.paths import OUTPUTS, PLOTS
from analysis.stats.instance_costs import complete_costs, encodings_only
from analysis.stats.join_results import RESULT_COLUMNS as NON_FEATURES, add_timeout_arg, timeout_seconds
from analysis.stats.plot_feature_correlation import CMAP, INK, INK_2, SURFACE, ink_on


def bh_qvalues(p):
    """Benjamini-Hochberg q-values; NaN p-values stay NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def speedup_correlations(df, timeout_cost, baseline="simple", corpus=None):
    """Spearman rho, BH q-value and instance count per (feature, encoding), over the
    complete instances, timeouts at the timeout.

    Returns (rho, q, n): two DataFrames indexed by feature with one column per
    encoding, and a dict encoding -> number of instances used. The router's own results
    are left out: it is not an encoding.
    """
    df = encodings_only(df)
    if corpus:
        df = df[df["corpus"] == corpus]
    costs, _ = complete_costs(df, timeout_cost)
    features = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("label_")]
    feats = df.groupby("id")[features].first().loc[costs.index]
    encodings = sorted(e for e in costs.columns if e != baseline)

    rho = pd.DataFrame(np.nan, index=features, columns=encodings)
    pval = rho.copy()
    n = {}
    for e in encodings:
        speedup = costs[baseline] / costs[e]
        n[e] = len(costs)
        for f in features:
            x = feats[f].astype(float)
            if x.nunique() < 2:
                continue            # constant within this scope: no correlation to report
            r, p = spearmanr(x, speedup)
            rho.loc[f, e], pval.loc[f, e] = r, p

    q = pd.DataFrame(bh_qvalues(pval.values.ravel()).reshape(pval.shape),
                     index=features, columns=encodings)
    return rho, q, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--baseline", default="simple")
    ap.add_argument("--corpus", help="restrict to one corpus (llm, grammar, legal)")
    add_timeout_arg(ap)
    ap.add_argument("--output", help="default: results/analysis/plots/speedup_heatmap[_<corpus>].png")
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)

    rho, q, n = speedup_correlations(pd.read_csv(args.joined), timeout, args.baseline,
                                     args.corpus)
    encodings = list(rho.columns)
    order = list(rho.index)

    fig, ax = plt.subplots(figsize=(6.5, 13), facecolor=SURFACE)
    im = ax.imshow(np.ma.masked_invalid(rho.values), cmap=CMAP, vmin=-1, vmax=1,
                   aspect="auto", interpolation="nearest")
    ax.set_facecolor(SURFACE)       # constant features show as empty cells, marked "const"
    ax.set_xticks(range(len(encodings)))
    ax.set_xticklabels([f"{e}\nn={n[e]}" for e in encodings], fontsize=10, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(len(order)):
        for j in range(len(encodings)):
            v, qv = rho.values[i, j], q.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "const", ha="center", va="center", fontsize=7, color=INK_2)
                continue
            stars = "**" if qv < 0.01 else "*" if qv < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{stars}", ha="center", va="center", fontsize=8,
                    color=ink_on(CMAP((v + 1) / 2)))

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.03)
    cbar.set_label(f"Spearman rho(feature, speedup vs {args.baseline})", color=INK_2)
    cbar.outline.set_visible(False)
    scope = f"{args.corpus} corpus" if args.corpus else "all corpora"
    ax.set_title(f"Feature vs speedup over {args.baseline}, {scope}", loc="left",
                 fontsize=13, color=INK, fontweight="semibold", pad=64)
    note = "One cost per instance; timeouts count at the timeout."
    ax.text(0, 1.045, "Light (yellow): encoding gains on baseline as feature grows; dark (blue): it loses ground. "
            f"{note}\n* BH q < 0.05, ** q < 0.01",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)

    suffix = f"_{args.corpus}" if args.corpus else ""
    out = Path(args.output) if args.output else PLOTS / f"speedup_heatmap{suffix}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
