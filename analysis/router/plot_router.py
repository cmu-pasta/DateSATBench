"""
Plot the router's cross-validated results for the paper: three panels over all corpora.

    python -m analysis.router.plot_router

(a) speedup over always-baseline in total time and (b) median speedup per instance, for
always one encoding and for the router, with the oracle as a dashed line; (c) how often the
router picks each encoding, by its vote or by falling back, with a tick for how many
instances each encoding is the fastest on. The rows of (a) and (b) keep one order, so the
two panels can be read across: no single encoding leads in both.

Reads crossval_router.py's router_eval.json and router_eval.csv, as the report's
Specialized router section does, so the numbers are the report's: the router measured in its
live runs (run_router_cv.py) when joined.csv held them, else predicted.

Writes analysis/outputs/plots/router.pdf (for the paper) and .png (to look at).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.paths import CROSS_VALIDATION, PLOTS
from analysis.report.build_report import fastest_counts
from analysis.stats import paper_style as ps

SIZE = (ps.TEXT_WIDTH, 2.15)   # inches
BAR = 0.62                     # bar height, in rows


def style(ax, title):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color=ps.RULE, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", pad=12)


def speedup_panel(ax, table, key, rows, y, title):
    o = table["oracle"][key]
    top = o * 1.2
    for n in rows:
        v = table["router" if n == "router" else f"always {n}"][key]
        router = n == "router"
        ax.barh(y[n], v, height=BAR, color=ps.ROUTER if router else ps.COLOURS[n], linewidth=0, zorder=2)
        if router:   # inside its bar, clear of the oracle's line
            bold = plt.rcParams["text.usetex"]
            ax.text(v - top * 0.025, y[n], rf"\textbf{{{v:.2f}}}$\boldsymbol{{\times}}$" if bold else ps.times(v),
                    va="center", ha="right", fontsize=ps.TICK_PT, color="white")
        else:
            ax.text(v + top * 0.02, y[n], ps.times(v), va="center", ha="left",
                    fontsize=ps.TICK_PT, color=ps.INK_2)
    # the oracle, a ceiling: its line stops under its label
    ax.plot([o, o], [-0.6, y["router"] + 0.5], color=ps.INK_3, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(o, y["router"] + 0.62, "oracle " + ps.times(o), ha="center", va="bottom",
            fontsize=ps.TICK_PT, color=ps.INK_2)
    ax.axvline(1, color=ps.INK_3, linewidth=0.6, zorder=1.5)      # always the baseline, behind the bars
    ax.set_xlim(0, top)
    ax.xaxis.set_major_formatter(lambda x, _: ps.times(x, "g"))
    style(ax, title)


def picks_panel(ax, summary, fallbacks, fastest, rows, y, title):
    fb = fallbacks[0] if len(fallbacks) == 1 else None
    top = max(max(summary["router_picks"].values()), max(fastest.values()) if fastest else 0) * 1.22
    for e in rows[1:]:
        picks, colour = summary["router_picks"][e], ps.COLOURS[e]
        fell = summary["router_fell_back"] if e == fb else 0
        ax.barh(y[e], picks - fell, height=BAR, color=colour, linewidth=0)
        if fell:
            ax.barh(y[e], fell, left=picks - fell, height=BAR, color="white", edgecolor=colour,
                    hatch="//////", linewidth=0.6)
        end = picks
        if fastest:
            ax.plot([fastest[e]] * 2, [y[e] - 0.45, y[e] + 0.45], color=ps.INK, linewidth=1.1,
                    solid_capstyle="butt")
            if picks < fastest[e] < picks + top * 0.17:     # the count would sit on the tick
                end = fastest[e]
        ax.text(end + top * 0.025, y[e], f"{picks:g}", va="center", ha="left",
                fontsize=ps.TICK_PT, color=ps.INK_2)
    ax.set_xlim(0, top)
    # the router's own row is empty here, so the key goes there
    handles = [Patch(facecolor=ps.INK_3, linewidth=0, label="vote")]
    if fb:
        handles.append(Patch(facecolor="white", edgecolor=ps.INK_3, hatch="//////", linewidth=0.6,
                             label="fallback"))
    if fastest:
        handles.append(Line2D([], [], color=ps.INK, linewidth=1.1, label="fastest"))
    ax.legend(handles=handles, loc="center left", ncol=len(handles), frameon=False,
              bbox_to_anchor=(0, y["router"]), bbox_transform=ax.get_yaxis_transform(),
              handlelength=1.2, handleheight=0.9, columnspacing=1.0, handletextpad=0.4,
              borderaxespad=0, fontsize=ps.TICK_PT)
    style(ax, title)


def plot(cv_path, out_dir):
    usetex = ps.apply()
    cv = json.loads(Path(cv_path).read_text())
    s = cv["overall"]
    encodings = sorted(n[len("always "):] for n in s["table"] if n.startswith("always "))
    fastest = (fastest_counts(Path(cv_path).with_name("router_eval.csv"), encodings) or {}).get("all")
    rows = ["router"] + [e for e, *_ in ps.ENCODINGS if e in encodings]   # top to bottom
    y = {r: len(rows) - 1 - i for i, r in enumerate(rows)}

    fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True)
    speedup_panel(axes[0], s["table"], "speedup_total", rows, y, "(a) Speedup in total time")
    speedup_panel(axes[1], s["table"], "speedup_median", rows, y, "(b) Median speedup")
    picks_panel(axes[2], s, cv["fallbacks"], fastest, rows, y, "(c) Encodings picked")
    axes[0].set_ylim(-0.6, len(rows) - 0.1)
    axes[0].set_yticks([y[r] for r in rows])
    axes[0].set_yticklabels([(r"\textbf{Router}" if usetex else "Router") if r == "router" else ps.NAMES[r]
                             for r in rows], fontsize=ps.LABEL_PT, color=ps.INK)
    fig.tight_layout(w_pad=1.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"router.{ext}", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_dir / 'router'}.pdf and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router-cv", default=str(CROSS_VALIDATION / "router_eval.json"))
    ap.add_argument("--out-dir", default=str(PLOTS))
    args = ap.parse_args()
    plot(args.router_cv, Path(args.out_dir))


if __name__ == "__main__":
    main()
