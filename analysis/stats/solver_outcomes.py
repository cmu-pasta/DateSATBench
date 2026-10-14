"""
Summarise what each encoding did: how many instances it solves, how long it takes in
total and on the median instance, and how it compares with the baseline.

    python -m analysis.stats.solver_outcomes --timeout 20000          # all corpora pooled
    python -m analysis.stats.solver_outcomes --corpus legal           # one corpus
    python -m analysis.stats.solver_outcomes --compare                # pooled, then every corpus
    python -m analysis.stats.solver_outcomes --compare --output analysis/outputs/outcomes.csv

Reads joined.csv and prints one row per encoding for each scope. --output writes the
same rows as CSV with a `scope` column (all, llm, legal, grammar). build_report.py
calls solver_outcomes() for every scope, so the report shows these exact numbers.

Every number comes from instance_costs.py, the measurement the router is trained and
cross-validated on: an instance's cost is the median over its runs, a timed-out run counts
at the timeout (--timeout, or $DATESAT_TIMEOUT_MS), and every instance that every encoding
ran without an error counts. An instance on which every encoding timed out counts as a
timeout for every encoding. The router's cross-validation leaves those out, since no
encoding is faster than another on them, so always-<encoding> there has `unsolvable` fewer
timeouts and `unsolvable` x the timeout less total time than the encoding's row here.

Columns
  instances            instances in the scope with no errored or missing run; the same
                       for every row
  sat, unsat, timeout  instances this encoding answered sat, answered unsat, or timed out on
  solved               sat + unsat
  solve_rate           solved / instances
  total_time           sum of the costs, timeouts at the timeout (seconds)
  median_time          median cost over the instances, timeouts at the timeout
  wins                 instances on which it costs less than the baseline
  median_speedup       median over the instances of the baseline's cost / its cost
  unsolvable           of those instances, the ones on which every encoding timed out;
                       they count as a timeout in every row
  incomplete           instances with an errored or missing run, left out of every other
                       column

The baseline's win and speedup columns are empty: it is not compared with itself.
"""

import argparse

import pandas as pd

from analysis.paths import OUTPUTS
from analysis.stats.instance_costs import all_timeout_ids, answers, complete_costs
from analysis.stats.join_results import add_timeout_arg, timeout_seconds

CORPORA = ["llm", "legal", "grammar"]
COUNTS = ["instances", "sat", "unsat", "timeout", "solved", "wins", "unsolvable", "incomplete"]
FLOATS = ["solve_rate", "total_time", "median_time", "median_speedup"]


def solver_outcomes(df, timeout_cost, baseline="simple", corpus=None):
    """One row per encoding (baseline first) for one scope; corpus None or 'all' pools."""
    if corpus and corpus != "all":
        df = df[df["corpus"] == corpus]
    costs, incomplete = complete_costs(df, timeout_cost)
    unsolvable = int(costs.index.isin(all_timeout_ids(df)).sum())
    answer = answers(df).reindex(index=costs.index, columns=costs.columns)
    encodings = [baseline] + sorted(e for e in costs.columns if e != baseline)
    n = len(costs)

    rows = []
    for e in encodings:
        c = costs[e]
        solved = c < timeout_cost
        speedup = costs[baseline] / c
        compared = e != baseline
        rows.append({
            "encoding": e,
            "instances": n,
            "sat": int((solved & (answer[e] == "sat")).sum()),
            "unsat": int((solved & (answer[e] == "unsat")).sum()),
            "timeout": int((~solved).sum()),
            "solved": int(solved.sum()),
            "solve_rate": solved.mean() if n else None,
            "total_time": c.sum(),
            "median_time": c.median() if n else None,
            "wins": int((speedup > 1).sum()) if compared else None,
            "median_speedup": speedup.median() if compared and n else None,
            "unsolvable": unsolvable,
            "incomplete": incomplete,
        })
    out = pd.DataFrame(rows).set_index("encoding")
    return out.astype({c: "Int64" for c in COUNTS}).astype({c: "float64" for c in FLOATS})


def fmt_time(t):
    if pd.isna(t):
        return "-"
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def display(t, scope):
    """Human-readable version of one scope's table."""
    rows = []
    for e, r in t.iterrows():
        base = pd.isna(r["wins"])
        rows.append({
            "encoding": e,
            "solved": f"{r['solved']}/{r['instances']} ({100 * r['solve_rate']:.1f}%)",
            "sat": r["sat"], "unsat": r["unsat"], "timeout": r["timeout"],
            "total time": f"{r['total_time']:.1f} s",
            "median time": fmt_time(r["median_time"]),
            "beats baseline": "baseline" if base
            else f"{r['wins']}/{r['instances']} ({100 * r['wins'] / r['instances']:.0f}%)",
            "median speedup": "-" if pd.isna(r["median_speedup"]) else f"{r['median_speedup']:.2f}x",
        })
    first = t.iloc[0]
    head = (f"{scope}: {int(first['instances'])} instances, {int(first['unsolvable'])} of which no "
            f"encoding solves; left out: {int(first['incomplete'])} with an errored or missing run")
    return head + "\n" + pd.DataFrame(rows).set_index("encoding").to_string()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--baseline", default="simple")
    add_timeout_arg(ap)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--corpus", choices=CORPORA, help="one corpus instead of all pooled")
    scope.add_argument("--compare", action="store_true",
                       help="pooled, then every corpus, one table each")
    ap.add_argument("--output", help="also write the rows as CSV, with a `scope` column")
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)

    joined = pd.read_csv(args.joined)
    scopes = ["all"] + CORPORA if args.compare else [args.corpus or "all"]
    tables = {s: solver_outcomes(joined, timeout, args.baseline, s) for s in scopes}
    print("\n\n".join(display(t, s) for s, t in tables.items()))

    if args.output:
        out = pd.concat([t.assign(scope=s) for s, t in tables.items()]).reset_index()
        out = out[["scope"] + [c for c in out.columns if c != "scope"]]
        out.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
