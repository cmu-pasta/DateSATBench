"""
Plot the cactus plot of every corpus for the paper: instances solved against time.

    python -m analysis.stats.plot_cactus --timeout 20000     # or set DATESAT_TIMEOUT_MS

One panel per corpus, one line per encoding. A line counts the instances the encoding solves
within the time on the x axis in every run, averaged over the runs, so it climbs to its
`solved` at the timeout; higher and further left is better. The instances and the runs are
solver_outcomes()'s, so a line ends at the table's Solved count, the mean over the runs. The
report draws the same plot.

Writes analysis/outputs/plots/cactus.pdf (for the paper) and .png (to look at).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from analysis.paths import OUTPUTS, PLOTS
from analysis.stats import paper_style as ps
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.solver_outcomes import solved_times

SIZE = (ps.TEXT_WIDTH, 2.15)   # inches


def time_label(t):
    return f"{t * 1000:g} ms" if t < 1 else f"{t:g} s"


def plot(joined, timeout_s, out_dir):
    ps.apply()
    times = {c: solved_times(joined, timeout_s, c) for c, *_ in ps.CORPORA}
    runs = joined["run"].nunique()
    lo = min(ts[0] for per in times.values() for ts in per.values() if ts)
    ticks = [10.0 ** p for p in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(timeout_s))))
             if 10.0 ** p < timeout_s]
    encodings = [e for e, *_ in ps.ENCODINGS if any(e in per for per in times.values())]

    fig, axes = plt.subplots(1, len(ps.CORPORA), figsize=SIZE)
    for ax, (c, name, *_) in zip(axes, ps.CORPORA):
        n = len(joined.loc[joined["corpus"] == c, "id"].unique())
        for e in encodings:
            ts = times[c].get(e, [])
            # a step up of 1 / runs at every solved run, then flat to the timeout at the solved count
            x = [ts[0]] + ts + [timeout_s] if ts else [lo, timeout_s]
            y = [k / runs for k in [0, *range(1, len(ts) + 1), len(ts)]] if ts else [0, 0]
            ax.step(x, y, where="post", color=ps.COLOURS[e], linewidth=1.2, solid_capstyle="butt")
        ax.axvline(timeout_s, color=ps.INK_3, linewidth=0.6, linestyle=(0, (1, 1.5)))   # the timeout
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.4, timeout_s * 1.25)
        ax.xaxis.set_major_locator(FixedLocator(ticks + [timeout_s]))
        ax.xaxis.set_minor_locator(NullLocator())
        # the timeout is labelled; a power of ten too near it to fit a label is not
        ax.set_xticklabels([time_label(t) if t == timeout_s or t < timeout_s / 3 else ""
                            for t in ticks + [timeout_s]])
        ax.set_ylim(0, n * 1.03)
        ax.set_title(f"{name} ({n})", pad=5)
        ax.grid(axis="y", color=ps.RULE, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("time per instance", labelpad=2)
    axes[0].set_ylabel("instances solved", labelpad=3)

    handles = [Line2D([], [], color=ps.COLOURS[e], linewidth=1.6) for e in encodings]
    fig.legend(handles, [ps.NAMES[e] for e in encodings], loc="upper center", ncol=len(encodings),
               frameon=False, fontsize=ps.LABEL_PT, handlelength=1.8, columnspacing=1.6,
               handletextpad=0.5, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cactus.{ext}", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_dir / 'cactus'}.pdf and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--out-dir", default=str(PLOTS))
    add_timeout_arg(ap)
    args = ap.parse_args()
    plot(pd.read_csv(args.joined), timeout_seconds(ap, args), Path(args.out_dir))


if __name__ == "__main__":
    main()
