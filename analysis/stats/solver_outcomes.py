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
cross-validated on: a timed-out run counts at the timeout (--timeout, or
$DATESAT_TIMEOUT_MS), and every instance that every encoding ran without an error counts.
The solved counts are taken in every run and averaged over the runs, so they can be
fractional. Total time, mean time, median time and median speedup are computed on every run
separately and reported as the median over the runs, with the fastest and slowest run as its
_min and _max; the standard deviation beside the mean time is that of the run the median
picks. The wins compare instance costs, the median over an instance's runs. An instance on
which every encoding timed out counts as a timeout for every encoding. The router's
cross-validation counts the same way, so its always-<encoding> rows have this table's
timeouts and total time. The router's live runs (run_router_cv.py), when joined, are a row
of their own, on the same instances: which instances count is decided by the encodings
alone, and every run of the router must have a result without an error on every one of
them. Every per-run figure of a row is taken over that row's own runs, as the router can
have fewer runs than the encodings.

Columns
  instances            instances in the scope with no errored or missing run; the same
                       for every row
  sat, unsat, timeout  mean over the runs of the instances this encoding answered sat,
                       answered unsat, or timed out on in that run
  solved               sat + unsat
  solve_rate           solved / instances: the mean over the runs of the run's share solved
  total_time           median over the runs of the run's summed time on the instances,
                       timeouts at the timeout (seconds)
  mean_time            median over the runs of the run's mean time on the instances,
                       timeouts at the timeout: total_time / instances
  sd_time              standard deviation over the instances of the time in the run whose
                       mean_time is the median, timeouts at the timeout; with an even number
                       of runs, the mean of the two middle runs', as mean_time is the mean of
                       their means
  median_time         median over the runs of the run's median time on the instances,
                       timeouts at the timeout
  wins                 instances on which it costs less than the baseline
  median_speedup       median over the runs of the run's median over the instances of the
                       baseline's time / its time, run k against the baseline's run k
  <column>_min/max     the lowest and highest run's value of each of those four
  unsolvable           of those instances, the ones on which every encoding timed out;
                       they count as a timeout in every row
  incomplete           instances with an errored or missing run, left out of every other
                       column

The baseline's win and speedup columns are empty: it is not compared with itself.
"""

import argparse
import math

import pandas as pd

from analysis.paths import OUTPUTS
from analysis.stats.instance_costs import (ROUTER, all_timeout_ids, complete_costs, cost_table,
                                           encodings_only, run_costs, run_outcomes)
from analysis.stats.join_results import add_timeout_arg, timeout_seconds

CORPORA = ["llm", "legal", "grammar"]
COUNTS = ["instances", "wins", "unsolvable", "incomplete"]
OUTCOMES = ["sat", "unsat", "timeout"]        # each the mean over the runs of a count
PER_RUN = ["total_time", "mean_time", "median_time", "median_speedup"]
FLOATS = (OUTCOMES + ["solved", "solve_rate"] + [f"{c}{s}" for c in PER_RUN for s in ("", "_min", "_max")]
          + ["sd_time"])


def scope_costs(df, timeout_cost):
    """complete_costs of the encodings, with the router's costs as one more column when df
    holds its results. Raises ValueError unless every run of the router has a result without
    an error on every instance every encoding completed (a live evaluation joined half-way
    has runs that cover only some corpora)."""
    costs, incomplete = complete_costs(encodings_only(df), timeout_cost)
    router = df[df["encoding"] == ROUTER]
    if len(router):
        times = run_costs(router, timeout_cost, costs.index)[ROUTER].dropna()
        expected = router["run"].nunique() * len(costs)
        if len(times) != expected:
            raise ValueError(f"the router's runs miss, or have an error on, {expected - len(times)} "
                             f"of {expected} (run, instance) pairs over the instances every "
                             f"encoding completed")
        costs = costs.join(cost_table(router, timeout_cost)[ROUTER])
    return costs, incomplete


def solver_outcomes(df, timeout_cost, baseline="simple", corpus=None):
    """One row per encoding (baseline first) for one scope; corpus None or 'all' pools."""
    if corpus and corpus != "all":
        df = df[df["corpus"] == corpus]
    costs, incomplete = scope_costs(df, timeout_cost)
    unsolvable = int(costs.index.isin(all_timeout_ids(df)).sum())
    outcome = run_outcomes(df, timeout_cost, costs.index)
    runs = run_costs(df, timeout_cost, costs.index)
    encodings = [baseline] + sorted(e for e in costs.columns if e != baseline)
    n = len(costs)

    rows = []
    for e in encodings:
        c = costs[e]
        # each outcome's count in every run, then the mean over the runs
        counts = {k: outcome[e].dropna().eq(k).groupby(level="run").sum().mean() if n else None
                  for k in OUTCOMES}
        speedup = costs[baseline] / c
        compared = e != baseline
        own = runs[e].dropna().groupby(level="run")                    # the row's own runs
        per_run = {"total_time": own.sum(), "mean_time": own.mean(), "median_time": own.median()} if n else {}
        if compared and n:
            per_run["median_speedup"] = (runs[baseline] / runs[e]).dropna().groupby(level="run").median()
        sd = None
        if n:
            # the SD of the run whose mean is the median; with an even number of runs the median
            # averages the two middle runs' means, so their SDs are averaged too
            middle = per_run["mean_time"].sort_values(kind="stable").index
            sd = own.std()[middle[[(len(middle) - 1) // 2, len(middle) // 2]]].mean()
        rows.append({
            "encoding": e,
            "instances": n,
            **counts,
            "solved": counts["sat"] + counts["unsat"] if n else None,
            "solve_rate": (counts["sat"] + counts["unsat"]) / n if n else None,
            "wins": int((speedup > 1).sum()) if compared else None,
            "unsolvable": unsolvable,
            "incomplete": incomplete,
            **{f"{k}{s}": getattr(per_run[k], f)() if k in per_run else None
               for k in PER_RUN for s, f in (("", "median"), ("_min", "min"), ("_max", "max"))},
            "sd_time": sd,
        })
    out = pd.DataFrame(rows).set_index("encoding")
    return out.astype({c: "Int64" for c in COUNTS}).astype({c: "float64" for c in FLOATS})


def run_counts(df):
    """How many runs each encoding (and the router, if joined) has."""
    return {e: int(n) for e, n in df.groupby("encoding")["run"].nunique().items()}


def solved_times(df, timeout_cost, corpus=None):
    """For a cactus plot: each encoding's times on the runs it solved, over every run,
    fastest first, on the same instances as solver_outcomes(). Each run counts 1 / runs, so
    a list's length divided by the encoding's number of runs (run_counts) is its `solved`."""
    if corpus and corpus != "all":
        df = df[df["corpus"] == corpus]
    costs, _ = scope_costs(df, timeout_cost)
    times = run_costs(df, timeout_cost, costs.index)
    return {e: sorted(t[t < timeout_cost].tolist()) for e, t in times.items()}


def fmt_count(x):
    """A count, or a mean of counts over the runs to one decimal."""
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def fmt_pct(share, digits=1):
    """A share as a percentage to `digits` decimals, a half rounded up, as the report's pct()
    does: 175.1 of 200 is 87.6%, where the float 0.8755, just under, rounds to 87.5%."""
    scaled = float(f"{100 * share * 10 ** digits:.12g}")      # drop the float error first
    return f"{math.floor(scaled + 0.5) / 10 ** digits:.{digits}f}"


def fmt_range(r, col, fmt):
    """A per-run column as 'median (min-max)'."""
    if pd.isna(r[col]):
        return "-"
    return f"{fmt(r[col])} ({fmt(r[col + '_min'])}-{fmt(r[col + '_max'])})"


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
            "solved": f"{fmt_count(r['solved'])}/{r['instances']} ({fmt_pct(r['solve_rate'])}%)",
            **{k: fmt_count(r[k]) for k in OUTCOMES},
            "total time": fmt_range(r, "total_time", lambda t: f"{t:.1f} s"),
            "median time": fmt_range(r, "median_time", fmt_time),
            "beats baseline": "baseline" if base
            else f"{r['wins']}/{r['instances']} ({fmt_pct(r['wins'] / r['instances'], 0)}%)",
            "median speedup": fmt_range(r, "median_speedup", lambda x: f"{x:.2f}x"),
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
