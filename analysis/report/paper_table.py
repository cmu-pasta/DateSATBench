r"""
Write the report's solver-outcomes table as a LaTeX tabular for the paper.

    python -m analysis.report.paper_table --timeout 20000     # outputs/outcomes_table.tex, and its plots
    python -m analysis.report.paper_table --with-router       # the router's live runs as a last row

The numbers are solver_outcomes()'s, the ones the report shows, one block per corpus. The
file holds the colour definitions and the tabular only; the paper wraps it in its own float:
\begin{table} \caption{...} \input{outcomes_table} \end{table}. It needs xcolor, booktabs,
makecell and graphicx, and the paper's encoding macros (\naive, \epoch, ...). Each corpus's
cactus plot is written beside it, as outcomes_table_cactus_<corpus>.pdf.

Columns: the encoding; the instances it solves, as a bar of sat, unsat and timed-out
instances with the three counts beside it, each in a column of its own and counted in every
run and averaged over the runs (so it can be fractional); mean and median time per
instance, each the median over the runs, with the standard deviation over the instances
after the mean, as mean +- SD (solver_outcomes' sd_time); and, spanning the corpus's rows, its cactus plot as plot_cactus
draws a panel, one line per encoding. Each line has its own colour and dash pattern, so it
can be told apart in black and white; a sample before each encoding's name shows it. The
best in each corpus is bold and underlined: the most instances solved, on the sat and unsat
counts that add up to it, and the lowest mean and median time.

The router is not an encoding, so its live runs (run_router_cv.py), when joined.csv holds
them, are left out unless --with-router asks for them as a row of their own, measured like
every encoding.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from analysis.paths import OUTPUTS
from analysis.stats import paper_style as ps
from analysis.stats.instance_costs import ROUTER, encodings_only
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.plot_cactus import step_curve, time_axis, time_label
from analysis.stats.solver_outcomes import fmt_count, run_counts, solved_times, solver_outcomes

# The paper's names and order: corpora as in the dataset section, encodings as in the
# solver section.
CORPORA = {"llm": "LLM-Synthesized", "grammar": "Grammar-Sampled", "legal": "Legally Grounded"}
ENCODINGS = {"simple": r"\naive", "epoch_days": r"\epoch", "hybrid_init_ymd": r"\hybridymd",
             "hybrid_init_epoch": r"\hybridepoch", "hybrid_both": r"\hybridboth",
             "alpha_beta": r"\ab"}
# The report's colours: sat, unsat and timed out; grey text.
COLOURS = {"dsSat": "2F9E55", "dsUnsat": "F0A030", "dsTimeout": "D43D3D", "dsDim": "85837B"}
COUNT_GAP = r"@{\hspace{4pt}}"   # between the sat, unsat and timed-out columns
PLUS_MINUS = r"@{{\scriptsize\color{dsDim}\ $\pm$\ }}"   # between the mean and its SD
LINES = {**ps.COLOURS, ROUTER: ps.ROUTER}   # each encoding's line, as in the paper's figures
BAR_CM = 1.25         # the solved bar, all instances
GAP_PT = 0.6          # between the solved bar's segments
MIN_CM = 0.04         # a segment that is not empty is at least this wide
ROW_PT = 10.0         # a row before stretching: acmsmall's \footnotesize line
STRETCH = 1.1         # \arraystretch, a little air between the rows
CACTUS_W_IN = 1.665   # a corpus's plot: the width the table leaves; its height is its rows'
CACTUS_PT = 7         # the plots' axis labels, as large as the table's grey text
LINE_PT = 1.0         # an encoding's line, in the plots and in its samples
SAMPLE_PT = 13        # a sample of an encoding's line, before its name
# Each encoding's dash pattern (on, off, ... in pt; None is solid), so the lines stay apart
# printed in black and white, where Simple and Hybrid init_ymd come out the same grey and the
# two epoch encodings nearly so: solid, dashed, dotted, dash-dot and dash-dot-dot.
DASHES = {"simple": (6, 2), "epoch_days": (1, 1.3), "hybrid_init_ymd": None,
          "hybrid_init_epoch": (4, 1.3, 1, 1.3), "alpha_beta": (4, 1.3, 1, 1.3, 1, 1.3)}


def dim(text):
    return rf"{{\scriptsize\color{{dsDim}}{text}}}"


def solved_bar(r):
    segs = [(c, r[k]) for c, k in (("dsSat", "sat"), ("dsUnsat", "unsat"), ("dsTimeout", "timeout"))
            if r[k] > 0]
    room = BAR_CM - (len(segs) - 1) * GAP_PT / 28.45
    return rf"\hspace{{{GAP_PT}pt}}".join(
        rf"\textcolor{{{c}}}{{\rule{{{max(MIN_CM, room * n / r['instances']):.3f}cm}}{{1.3ex}}}}"
        for c, n in segs)


def line_sample(e):
    """A short stretch of the encoding's line, in its colour and dash pattern, drawn with rules."""
    colour = LINES.get(e, ps.INK_2).lstrip("#").upper()
    pattern, parts, x, i = DASHES.get(e) or (SAMPLE_PT, 0), [], 0.0, 0
    while x < SAMPLE_PT - 1e-6:
        step = min(pattern[i % len(pattern)], SAMPLE_PT - x)
        parts.append(rf"\rule[0.5ex]{{{step:g}pt}}{{{LINE_PT:g}pt}}" if i % 2 == 0
                     else rf"\hspace{{{step:g}pt}}")
        x, i = x + step, i + 1
    return rf"\textcolor[HTML]{{{colour}}}{{{''.join(parts)}}}"


def cactus(times, runs, encodings, n, lo, ticks, timeout_s, height_in, path):
    """One corpus's cactus plot, drawn as plot_cactus draws a panel but sized to hang beside
    the corpus's rows: no title or legend, since the table has them, and only 0 and every
    instance labelled on the instance axis."""
    w, h = CACTUS_W_IN * 72, height_in * 72              # in pt, as the margins are
    left, bottom, right, top = 15, 11, 5, 2              # room for the labels
    fig = plt.figure(figsize=(CACTUS_W_IN, height_in))
    ax = fig.add_axes((left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h))
    for e in encodings:
        x, y = step_curve(times.get(e, []), runs[e], lo, timeout_s)
        dashes = DASHES.get(e)                           # matplotlib scales a pattern by the width
        ax.step(x, y, where="post", color=LINES.get(e, ps.INK_2), linewidth=LINE_PT,
                linestyle=(0, [d / LINE_PT for d in dashes]) if dashes else "-",
                solid_capstyle="butt")
    ax.axhline(n, color=ps.RULE, linewidth=0.5, zorder=0)                          # every instance
    ax.axvline(timeout_s, color=ps.INK_3, linewidth=0.5, linestyle=(0, (1, 1.5)))   # the timeout
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.4, timeout_s * 1.25)
    ax.set_ylim(0, n * 1.04)
    ax.xaxis.set_major_locator(FixedLocator(ticks + [timeout_s]))
    ax.xaxis.set_minor_locator(NullLocator())
    # 1 ms, 1 s and the timeout are labelled; the powers of ten between them are not
    ax.set_xticklabels([time_label(t) if t == timeout_s or round(np.log10(t)) % 3 == 0 else ""
                        for t in ticks + [timeout_s]])
    ax.yaxis.set_major_locator(FixedLocator([0, n / 2, n]))
    ax.set_yticklabels(["0", "", f"{n}"])
    ax.tick_params(labelsize=CACTUS_PT, length=2, pad=1.5, color=ps.INK_3, labelcolor=ps.INK_3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(ps.INK_3)
    fig.savefig(path)
    plt.close(fig)


def best(values, low):
    """The encodings whose shown value is the best in their corpus: the lowest if `low`,
    else the highest. Ties all count."""
    if not values:
        return set()
    target = (min if low else max)(values.values())
    return {e for e, v in values.items() if v == target}


def ordered(t):
    return [e for e in ENCODINGS if e in t.index] + [e for e in t.index if e not in ENCODINGS]


def encoding_name(e):
    return "Router" if e == ROUTER else ENCODINGS.get(e, r"\texttt{" + e.replace("_", r"\_") + "}")


def time_formats(ms):
    """How the median time, and the mean time and its SD, are printed: in ms, or in s."""
    if ms:
        return (lambda x: f"{1000 * x:.1f}"), (lambda x: f"{1000 * x:.0f}")
    return (lambda x: f"{x:.2f}"), (lambda x: f"{x:.2f}")


def rows(t, ms, widest):
    tfmt, mfmt = time_formats(ms)
    encs = ordered(t)
    shown = lambda fmt, col, es: {e: float(fmt(t.loc[e, col])) for e in es}   # compared as printed
    top = {"solved": best(shown(fmt_count, "solved", encs), low=False),
           "mean": best(shown(mfmt, "mean_time", encs), low=True),
           "median": best(shown(tfmt, "median_time", encs), low=True)}
    # the best value in its corpus, bold and underlined
    mark = lambda text, col, e: rf"\underline{{\textbf{{{text}}}}}" if e in top[col] else text

    out = []
    for e in encs:
        r = t.loc[e]
        k = {c: fmt_count(r[c]) for c in ("sat", "unsat", "timeout")}
        name = rf"{line_sample(e)}\ {encoding_name(e)}"
        # The bar and the sat, unsat and timed-out counts each in a column of their own, so the
        # counts line up down the table; sat and unsat add up to the solved count.
        solved = " & ".join([solved_bar(r), mark(k["sat"], "solved", e),
                             mark(k["unsat"], "solved", e), k["timeout"]])
        mean = mark(mfmt(r.mean_time), "mean", e)
        # The time right-aligned in a slot as wide as the widest, so the times line up and the
        # column can centre them under its heading, which the time heading above makes wide.
        median = rf"\hphantom{{\textbf{{{widest}}}}}\llap{{{mark(tfmt(r.median_time), 'median', e)}}}"
        out.append(f"{name} & {solved} & {mean} & {dim(mfmt(r.sd_time))} & {median} & \\\\")
    return out


def latex_table(joined, timeout_s, stem, with_router=False):
    """The tabular, after writing each corpus's cactus plot to <stem>_cactus_<corpus>.pdf."""
    if not with_router:
        joined = encodings_only(joined)
    elif len(set(run_counts(joined).values())) > 1:          # the headers name one count of runs
        raise ValueError(f"the router and the encodings have different numbers of runs: "
                         f"{run_counts(joined)}")
    tables = {c: solver_outcomes(joined, timeout_s, corpus=c) for c in CORPORA}
    ms = all(t["median_time_max"].max() < 1 for t in tables.values())
    widest = max((time_formats(ms)[0](x) for t in tables.values() for x in t["median_time"]), key=len)
    runs = joined["run"].nunique()
    sw = {k: rf"\textcolor{{ds{k}}}{{\rule{{0.8ex}}{{0.8ex}}}}" for k in ("Sat", "Unsat", "Timeout")}
    legend = dim(rf"{sw['Sat']} sat\enspace {sw['Unsat']} unsat\enspace {sw['Timeout']} timed out")
    sd = dim(r"$\pm$ SD")
    unit = "ms" if ms else "s"
    ps.apply()
    times = {c: solved_times(joined, timeout_s, c) for c in CORPORA}
    counts = run_counts(joined)
    encs = ordered(next(iter(tables.values())))
    lo, ticks = time_axis(times, encs, timeout_s)
    plots = {c: f"{stem.name}_cactus_{c}" for c in CORPORA}
    # A plot spans its corpus's name row, the 2pt below it and its encodings' rows. Its top
    # is raised to the top of the name row and it takes up no height, so the rows keep theirs.
    # A row is \arraystretch times \normalbaselineskip tall; inside a tabular \baselineskip is 0.
    height_in = ((len(encs) + 1) * STRETCH * ROW_PT + 2) / 72
    for c, t in tables.items():
        cactus(times[c], counts, encs, int(t.iloc[0].instances), lo, ticks, timeout_s,
               height_in, stem.parent / f"{plots[c]}.pdf")
    span = rf"\dimexpr {(len(encs) + 1) * STRETCH:g}\normalbaselineskip+2pt\relax"
    hang = rf"-\dimexpr {(len(encs) + 0.3) * STRETCH:g}\normalbaselineskip+2pt\relax"
    lines = [f"% Written by python -m analysis.report.paper_table from {runs} runs, "
             f"timeout {timeout_s:g} s. Do not edit by hand.",
             *(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLOURS.items()),
             rf"\renewcommand{{\arraystretch}}{{{STRETCH:g}}}",
             # encoding; bar, sat, unsat, timed out; mean +- SD, median; plot
             rf"\begin{{tabular}}{{@{{}}l l@{{\ \ }}r{COUNT_GAP}r{COUNT_GAP}r r{PLUS_MINUS}l c l@{{}}}}",
             r"\toprule",
             rf"\multicolumn{{5}}{{l}}{{}} & \multicolumn{{3}}{{c}}"
             rf"{{Time ({unit})\ {dim(f'median of {runs} runs')}}} & \\",
             r"\cmidrule(lr){6-8}",
             # every heading centred over its column, or over the columns it names
             rf"\multicolumn{{1}}{{@{{}}c}}{{Encoding}} & \multicolumn{{4}}{{c}}{{\makecell{{Solved\\{legend}}}}} "
             rf"& \multicolumn{{2}}{{c}}{{Mean {sd}}} & \multicolumn{{1}}{{c}}{{\makecell{{Median\\instance}}}} "
             rf"& \multicolumn{{1}}{{c@{{}}}}{{\makecell{{Instances solved\\{dim(f'by time, mean of {runs} runs')}}}}} \\"]
    for i, (c, t) in enumerate(tables.items()):
        lines.append(r"\midrule" if i == 0 else r"\midrule[\heavyrulewidth]")
        n = dim(f"{int(t.iloc[0].instances)} instances")
        plot = rf"\raisebox{{{hang}}}[0pt][0pt]{{\includegraphics[height={span}]{{{plots[c]}}}}}"
        # the corpus's name in the encodings' column, its size above the bars it measures
        lines.append(rf"\textbf{{{CORPORA[c]}}} & \multicolumn{{4}}{{l}}{{{n}}} & \multicolumn{{2}}{{c}}{{}} & & {plot} \\")
        lines.append(r"\addlinespace[2pt]")
        lines.extend(rows(t, ms, widest))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--with-router", action="store_true",
                    help="add the router's live runs as a row, when joined.csv holds them")
    ap.add_argument("--output", default=str(OUTPUTS / "outcomes_table.tex"),
                    help="the tabular; the cactus plots are written beside it")
    add_timeout_arg(ap)
    args = ap.parse_args()
    try:
        tex = latex_table(pd.read_csv(args.joined), timeout_seconds(ap, args),
                          Path(args.output).with_suffix(""), args.with_router)
    except ValueError as e:
        ap.error(str(e))
    Path(args.output).write_text(tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
