"""
Plot how each feature relates to each encoding's speedup over the baseline.

    python analysis/plot_speedup_heatmap.py
    python analysis/plot_speedup_heatmap.py --corpus legal

Reads joined.csv. A cell is the Spearman rho between a feature and `speedup`
(baseline_time / time) for one encoding. rho > 0 (green): the encoding gains on the
baseline as the feature grows; rho < 0 (red): it loses ground.

By default rows where only one side timed out are kept at their bound (see
join_results.py): the true speedup is at least as extreme, so ranking it at the bound
is conservative. Rows where both timed out have no speedup and are left out.
`--timeouts drop` uses exact speedups only (both sides finished).

Stars mark cells that survive Benjamini-Hochberg correction across the whole grid
(* q < 0.05, ** q < 0.01). Rows keep the column order of features.csv, so every
heatmap (pooled or per corpus) lists features in the same order.

Writes analysis/plots/speedup_heatmap[_<corpus>][_exact].png.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from plot_feature_correlation import CMAP, INK, INK_2, SURFACE, ink_on

HERE = Path(__file__).parent
NON_FEATURES = {"id", "corpus", "encoding", "run", "time", "status", "solved",
                "baseline_time", "baseline_status", "speedup", "speedup_bound"}


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


def speedup_correlations(df, baseline="simple", corpus=None, timeouts="bound"):
    """Spearman rho, BH q-value and row count per (feature, encoding).

    Returns (rho, q, n): two DataFrames indexed by feature with one column per
    encoding, and a dict encoding -> number of rows used.
    """
    if corpus:
        df = df[df["corpus"] == corpus]
    df = df[df["speedup"].notna() & (df["encoding"] != baseline)]
    if timeouts == "drop":
        df = df[df["speedup_bound"] == "exact"]
    features = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("label_")]
    encodings = sorted(df["encoding"].unique())

    rho = pd.DataFrame(np.nan, index=features, columns=encodings)
    pval = rho.copy()
    n = {}
    for e in encodings:
        sub = df[df["encoding"] == e]
        n[e] = len(sub)
        for f in features:
            x = sub[f].astype(float)
            if x.nunique() < 2:
                continue            # constant within this subset: no correlation to report
            r, p = spearmanr(x, sub["speedup"])
            rho.loc[f, e], pval.loc[f, e] = r, p

    q = pd.DataFrame(bh_qvalues(pval.values.ravel()).reshape(pval.shape),
                     index=features, columns=encodings)
    return rho, q, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(HERE / "joined.csv"))
    ap.add_argument("--baseline", default="simple")
    ap.add_argument("--corpus", help="restrict to one corpus (llm, grammar, legal)")
    ap.add_argument("--timeouts", choices=("bound", "drop"), default="bound",
                    help="keep one-sided timeouts at their bound, or use exact speedups only")
    ap.add_argument("--output", help="default: plots/speedup_heatmap[_<corpus>].png")
    args = ap.parse_args()

    rho, q, n = speedup_correlations(pd.read_csv(args.joined), args.baseline,
                                     args.corpus, args.timeouts)
    encodings = list(rho.columns)
    order = list(rho.index)

    fig, ax = plt.subplots(figsize=(6.5, 13), facecolor=SURFACE)
    im = ax.imshow(np.ma.masked_invalid(rho.values), cmap=CMAP, vmin=-1, vmax=1,
                   aspect="auto", interpolation="nearest")
    ax.set_facecolor("#dcdad5")     # constant features show as flat grey
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
    note = ("One-sided timeouts kept at their bound." if args.timeouts == "bound"
            else "Both sides solved only.")
    ax.text(0, 1.045, "Green: encoding gains on baseline as feature grows; red: it loses ground. "
            f"{note}\n* BH q < 0.05, ** q < 0.01",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)

    suffix = (f"_{args.corpus}" if args.corpus else "") + ("_exact" if args.timeouts == "drop" else "")
    out = Path(args.output) if args.output else HERE / "plots" / f"speedup_heatmap{suffix}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
