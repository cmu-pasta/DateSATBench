r"""
Write the report's solver-outcomes table as a LaTeX tabular for the paper.

    python -m analysis.report.paper_table --timeout 20000     # outputs/outcomes_table.tex
    python -m analysis.report.paper_table --no-ranges         # without the lowest-highest run
    python -m analysis.report.paper_table --with-router       # the router's live runs as a last row

The numbers are solver_outcomes()'s, the ones the report shows, one block per corpus. The
file holds the colour definitions and the tabular only; the paper wraps it in its own float:
\begin{table} \caption{...} \input{outcomes_table} \end{table}. It needs xcolor, booktabs
and makecell, and the paper's encoding macros (\naive, \epoch, ...).

Columns: the encoding; the instances it solves, as a bar of sat, unsat and timed-out
instances with the solved count and share beside it and the three counts under it, each
counted in every run and averaged over the runs (so it can be fractional); total
time and median instance time, each the median over the runs; and, against the baseline,
the instances it is faster on and the median speedup, also the median over the runs. Under
each median over the runs, in grey, the lowest and highest run.

The router is not an encoding, so its live runs (run_router_cv.py), when joined.csv holds
them, are left out unless --with-router asks for them as a row of their own, measured like
every encoding.
"""

import argparse
from pathlib import Path

import pandas as pd

from analysis.paths import OUTPUTS
from analysis.stats.instance_costs import ROUTER, encodings_only
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.solver_outcomes import fmt_count, fmt_pct, run_counts, solver_outcomes

# The paper's names and order: corpora as in the dataset section, encodings as in the
# solver section.
CORPORA = {"llm": "LLM-Synthesized", "grammar": "Grammar-Sampled", "legal": "Legally Grounded"}
ENCODINGS = {"simple": r"\naive", "epoch_days": r"\epoch", "hybrid_init_ymd": r"\hybridymd",
             "hybrid_init_epoch": r"\hybridepoch", "hybrid_both": r"\hybridboth",
             "alpha_beta": r"\ab"}
# The report's colours: sat, unsat and timed out; the "faster on" bar and its track; grey text.
COLOURS = {"dsSat": "2F9E55", "dsUnsat": "F0A030", "dsTimeout": "D43D3D",
           "dsBar": "55544E", "dsTrack": "E2E0D9", "dsDim": "85837B"}
BAR_CM = 1.25         # the solved bar, all instances
MINIBAR_CM = 0.7      # the "faster on" bar, all instances
GAP_PT = 0.6          # between the solved bar's segments
MIN_CM = 0.04         # a segment that is not empty is at least this wide


def dim(text):
    return rf"{{\scriptsize\color{{dsDim}}{text}}}"


def two_lines(top, bottom, align="r"):
    return rf"\makecell[{align}]{{{top}\\{dim(bottom)}}}" if bottom else top


def solved_bar(r):
    segs = [(c, r[k]) for c, k in (("dsSat", "sat"), ("dsUnsat", "unsat"), ("dsTimeout", "timeout"))
            if r[k] > 0]
    room = BAR_CM - (len(segs) - 1) * GAP_PT / 28.45
    return rf"\hspace{{{GAP_PT}pt}}".join(
        rf"\textcolor{{{c}}}{{\rule{{{max(MIN_CM, room * n / r['instances']):.3f}cm}}{{1.3ex}}}}"
        for c, n in segs)


def minibar(frac):
    return (rf"\rlap{{\textcolor{{dsTrack}}{{\rule{{{MINIBAR_CM}cm}}{{0.9ex}}}}}}"
            rf"\rlap{{\textcolor{{dsBar}}{{\rule{{{MINIBAR_CM * frac:.3f}cm}}{{0.9ex}}}}}}"
            rf"\hspace{{{MINIBAR_CM}cm}}")


def run_range(r, col, fmt, ranges):
    return f"{fmt(r[col + '_min'])}--{fmt(r[col + '_max'])}" if ranges else None


def best(values, low):
    """The encodings whose shown value is the best in their corpus: the lowest if `low`,
    else the highest. Ties all count."""
    if not values:
        return set()
    target = (min if low else max)(values.values())
    return {e for e, v in values.items() if v == target}


def rows(t, baseline, ranges, ms):
    tfmt = (lambda x: f"{1000 * x:.1f}") if ms else (lambda x: f"{x:.2f}")
    totfmt, sfmt = (lambda x: f"{x:.1f}"), (lambda x: f"{x:.2f}")
    encs = [e for e in ENCODINGS if e in t.index] + [e for e in t.index if e not in ENCODINGS]
    others = [e for e in encs if e != baseline]
    shown = lambda fmt, col, es: {e: float(fmt(t.loc[e, col])) for e in es}   # compared as printed
    top = {"solved": best(shown(fmt_count, "solved", encs), low=False),
           "total": best(shown(totfmt, "total_time", encs), low=True),
           "median": best(shown(tfmt, "median_time", encs), low=True),
           "wins": best({e: int(t.loc[e, "wins"]) for e in others}, low=False),
           "speedup": best(shown(sfmt, "median_speedup", others), low=False)}
    bold = lambda text, col, e: rf"\textbf{{{text}}}" if e in top[col] else text

    out = []
    for e in encs:
        r = t.loc[e]
        k = {c: fmt_count(r[c]) for c in ("sat", "unsat", "timeout", "solved")}
        name = "Router" if e == ROUTER else ENCODINGS.get(e, r"\texttt{" + e.replace("_", r"\_") + "}")
        pct = dim(f"({fmt_pct(r.solve_rate)}" + r"\%)")
        solved = two_lines(rf"{solved_bar(r)}\ \ {bold(k['solved'], 'solved', e)}\ {pct}",
                           f"{k['sat']} / {k['unsat']} / {k['timeout']}", "l")
        total = two_lines(bold(totfmt(r.total_time), "total", e),
                          run_range(r, "total_time", totfmt, ranges))
        median = two_lines(bold(tfmt(r.median_time), "median", e),
                           run_range(r, "median_time", tfmt, ranges))
        if e == baseline:
            faster, speedup = dim("baseline"), dim(r"$1.00\times$")
        else:
            share = dim(f"({fmt_pct(r.wins / r.instances, 0)}" + r"\%)")
            faster = rf"{minibar(r.wins / r.instances)}\ \ {bold(int(r.wins), 'wins', e)}\ {share}"
            x = rf"${sfmt(r.median_speedup)}\times$"
            speedup = two_lines(rf"{{\boldmath{x}}}" if e in top["speedup"] else x,
                                run_range(r, "median_speedup", sfmt, ranges))
        out.append(f"{name} & {solved} & {total} & {median} & {faster} & {speedup} \\\\")
    return out


def latex_table(joined, timeout_s, baseline="simple", ranges=True, with_router=False):
    if not with_router:
        joined = encodings_only(joined)
    elif len(set(run_counts(joined).values())) > 1:          # the headers name one count of runs
        raise ValueError(f"the router and the encodings have different numbers of runs: "
                         f"{run_counts(joined)}")
    tables = {c: solver_outcomes(joined, timeout_s, baseline, c) for c in CORPORA}
    ms = all(t["median_time_max"].max() < 1 for t in tables.values())
    runs = joined["run"].nunique()
    sw = {k: rf"\textcolor{{ds{k}}}{{\rule{{0.8ex}}{{0.8ex}}}}" for k in ("Sat", "Unsat", "Timeout")}
    legend = dim(f"{sw['Sat']} sat / {sw['Unsat']} unsat / {sw['Timeout']} timed out")
    unit = "ms" if ms else "s"
    lines = [f"% Written by python -m analysis.report.paper_table from {runs} runs, "
             f"timeout {timeout_s:g} s. Do not edit by hand.",
             *(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLOURS.items()),
             r"\begin{tabular}{@{}l l r r l r@{}}",
             r"\toprule",
             rf" & & \multicolumn{{2}}{{c}}{{Time, median of {runs} runs}} & "
             rf"\multicolumn{{2}}{{c}}{{Versus {ENCODINGS.get(baseline, baseline)}}} \\",
             r"\cmidrule(lr){3-4} \cmidrule(l){5-6}",
             rf"Encoding & \makecell[l]{{Solved\\{legend}}} "
             rf"& Total (s) & Median instance ({unit}) & Faster on & Median speedup \\"]
    for i, (c, t) in enumerate(tables.items()):
        lines.append(r"\midrule" if i == 0 else r"\midrule[\heavyrulewidth]")
        n = dim(f"{int(t.iloc[0].instances)} instances")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{CORPORA[c]}}}\ {n}}} \\")
        lines.append(r"\addlinespace[2pt]")
        lines.extend(rows(t, baseline, ranges, ms))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--baseline", default="simple")
    ap.add_argument("--no-ranges", dest="ranges", action="store_false",
                    help="leave out the lowest and highest run under each median over the runs")
    ap.add_argument("--with-router", action="store_true",
                    help="add the router's live runs as a row, when joined.csv holds them")
    ap.add_argument("--output", default=str(OUTPUTS / "outcomes_table.tex"))
    add_timeout_arg(ap)
    args = ap.parse_args()
    try:
        tex = latex_table(pd.read_csv(args.joined), timeout_seconds(ap, args), args.baseline,
                          args.ranges, args.with_router)
    except ValueError as e:
        ap.error(str(e))
    Path(args.output).write_text(tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
