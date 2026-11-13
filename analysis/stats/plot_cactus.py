"""
Plot the cactus plot of every corpus for the paper: instances solved against time.

    python -m analysis.stats.plot_cactus --timeout 20000     # or set DATESAT_TIMEOUT_MS

One panel per corpus, one line per encoding, each in its own colour and dash pattern
(paper_style.py), so the lines stay apart printed in greyscale. A line counts the instances the encoding solves
within the time on the x axis in every run, averaged over the runs, so it climbs to its
`solved` at the timeout; higher and further left is better. The instances and the runs are
solver_outcomes()'s, so a line ends at the table's Solved count, the mean over the runs. The
report draws the same plot. --with-router adds the router's live runs (run_router_cv.py) as a
heavier black line, when joined.csv holds them; the router is not an encoding, so it is left out
otherwise.

Writes results/analysis/plots/cactus.pdf (for the paper) and .png (to look at).
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
from analysis.stats.instance_costs import ROUTER
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.solver_outcomes import run_counts, solved_times

SIZE = (ps.TEXT_WIDTH, 2.15)   # inches
LINE_PT = 1.2                  # an encoding's line


def time_label(t):
    return f"{t * 1000:g} ms" if t < 1 else f"{t:g} s"


def time_axis(times, encodings, timeout_s):
    """The fastest solved run over every corpus, and the powers of ten from it up to the timeout."""
    lo = min(times[c][e][0] for c in times for e in encodings if times[c].get(e))
    ticks = [10.0 ** p for p in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(timeout_s))))
             if 10.0 ** p < timeout_s]
    return lo, ticks


def step_curve(ts, runs, lo, timeout_s):
    """An encoding's line: a step up of 1 / runs at every solved run, then flat to the timeout
    at the solved count."""
    x = [ts[0]] + ts + [timeout_s] if ts else [lo, timeout_s]
    y = [k / runs for k in [0, *range(1, len(ts) + 1), len(ts)]] if ts else [0, 0]
    return x, y


def plot(joined, timeout_s, out_dir, with_router=False):
    usetex = ps.apply()
    times = {c: solved_times(joined, timeout_s, c) for c, *_ in ps.CORPORA}
    runs = run_counts(joined)
    shown = [e for e, *_ in ps.ENCODINGS] + ([ROUTER] if with_router else [])
    encodings = [e for e in shown if any(e in per for per in times.values())]
    colours = {**ps.COLOURS, ROUTER: ps.ROUTER}
    names = {**ps.NAMES, ROUTER: r"\textbf{Router}" if usetex else "Router"}
    widths = {e: LINE_PT * (ps.ROUTER_WIDTH if e == ROUTER else 1) for e in encodings}
    lo, ticks = time_axis(times, encodings, timeout_s)

    fig, axes = plt.subplots(1, len(ps.CORPORA), figsize=SIZE)
    for ax, (c, name, *_) in zip(axes, ps.CORPORA):
        n = len(joined.loc[joined["corpus"] == c, "id"].unique())
        for e in encodings:
            x, y = step_curve(times[c].get(e, []), runs[e], lo, timeout_s)
            ax.step(x, y, where="post", color=colours[e], linewidth=widths[e],
                    linestyle=ps.linestyle(e, widths[e]), solid_capstyle="butt", dash_capstyle="butt")
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

    # the key shows each line as it is drawn, long enough for its whole dash pattern
    handles = [Line2D([], [], color=colours[e], linewidth=widths[e], linestyle=ps.linestyle(e, widths[e]),
                      dash_capstyle="butt") for e in encodings]
    wide = len(encodings) <= len(ps.ENCODINGS)          # the router's line makes the row tighter
    fig.legend(handles, [names[e] for e in encodings], loc="upper center", ncol=len(encodings),
               frameon=False, fontsize=ps.LABEL_PT, handlelength=2.6 if wide else 2.2,
               columnspacing=1.3 if wide else 0.7, handletextpad=0.5 if wide else 0.4,
               borderaxespad=0.1)
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
    ap.add_argument("--with-router", action="store_true",
                    help="add the router's live runs as a line, when joined.csv holds them")
    add_timeout_arg(ap)
    args = ap.parse_args()
    plot(pd.read_csv(args.joined), timeout_seconds(ap, args), Path(args.out_dir), args.with_router)


if __name__ == "__main__":
    main()
