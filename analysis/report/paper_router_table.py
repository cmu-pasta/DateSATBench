r"""
Write the router's solve rate as a one-row LaTeX tabular for the paper.

    python -m analysis.report.paper_router_table     # results/analysis/router_solve_table.tex

Reads crossval_router.py's router_eval.json, as the report's Specialized router section
does, and gives every strategy's share of solved instances over all corpora: always one
encoding, the router (out of fold: measured in its live runs by run_router_cv.py when
joined.csv held them, else predicted and averaged over the repeats) and the oracle (the
fastest encoding in each run). Every count is taken in every run and averaged over the runs, as in
the outcomes table. The share is out of every instance, the ones no encoding solves
included, so the always-one-encoding columns match the outcomes table (paper_table.py).
The best strategy short of the oracle is in bold.

Like paper_table.py, the file holds the colour definitions and the tabular only, for the
paper to wrap in its own float; it needs xcolor and booktabs, and the paper's
encoding macros.
"""

import argparse
import json
from pathlib import Path

from analysis.paths import CROSS_VALIDATION, OUTPUTS
from analysis.report.paper_table import COLOURS, ENCODINGS
from analysis.stats.solver_outcomes import fmt_pct


def router_table(cv):
    s = cv["overall"]
    n = s["instances"]          # every instance, as in Table 1
    solved = {name: s["instances"] - t["timeouts"] for name, t in s["table"].items()}
    always = [e for e in ENCODINGS if f"always {e}" in solved]
    names = [f"always {e}" for e in always] + ["router", "oracle"]
    shown = {k: fmt_pct(solved[k] / n) for k in names}          # compared as printed
    best = max(float(shown[k]) for k in names if k != "oracle")

    cells = []
    for k in names:
        pct = shown[k] + r"\%"
        if k != "oracle" and float(shown[k]) == best:
            pct = rf"\textbf{{{pct}}}"
        cells.append(pct)
    heads = [ENCODINGS[e] for e in always] + ["Router", "Oracle"]
    how = (f"the router measured on {cv['live']['runs']} live runs" if cv.get("router_measured")
           else "the router predicted")
    lines = [f"% Written by python -m analysis.report.paper_router_table from {cv['folds']}-fold "
             f"cross-validation repeated {cv['repeats']} times, {how}, timeout "
             f"{cv['timeout_cost_s']:g} s. Do not edit by hand.",
             *(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLOURS.items()),
             rf"\begin{{tabular}}{{@{{}}l *{{{len(always)}}}{{r}} @{{\hspace{{14pt}}}} r r@{{}}}}",
             r"\toprule",
             rf" & \multicolumn{{{len(always)}}}{{c}}{{Always one encoding}} & & \\",
             rf"\cmidrule(lr){{2-{len(always) + 1}}}",
             " & " + " & ".join(heads) + r" \\",
             r"\midrule",
             f"Solved & {' & '.join(cells)} " + r"\\",
             r"\bottomrule",
             r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router-cv", default=str(CROSS_VALIDATION / "router_eval.json"))
    ap.add_argument("--output", default=str(OUTPUTS / "router_solve_table.tex"))
    args = ap.parse_args()
    Path(args.output).write_text(router_table(json.loads(Path(args.router_cv).read_text())))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
