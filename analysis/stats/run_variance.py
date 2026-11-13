"""
How much the results vary from run to run: for each encoding (and the router, when joined)
and each scope, its mean time and its number of solved instances in every run, and how far
apart the runs are.

    python -m analysis.stats.run_variance --timeout 20000
    python -m analysis.stats.run_variance --joined results/analysis/joined.csv
    python -m analysis.stats.run_variance --timeout 20000 \
        --results results/runs/encodings results/runs/router

--results reads the result files (<dir>/<corpus>/run_N/<encoding>_int.json) directly instead
of joined.csv, so no features are needed; give the router's runs as one more directory.

Measured as solver_outcomes measures it (instance_costs.py), on the same instances: a
timed-out run counts at the timeout, and only instances every encoding ran without an
error count. A run's mean time is its mean over those instances; its solved count is the
instances it answered sat or unsat within the timeout.

Columns, per (scope, row)
  runs                       how many runs it has
  mean_time_median/min/max   the run's mean time (s): median, lowest and highest run
  mean_time_spread_pct       (max - min) / median of the run's mean time, in percent
  mean_time_dev_pct          the largest |run - median| / median, in percent
  solved_min/max, solved_spread   solved instances: lowest, highest run, and their difference
  solve_rate_min/max         solved / instances in the lowest and highest run, in percent

The summary at the end gives, over the encodings (not the router), the largest spread of
the mean time and of the solved count, per corpus and pooled, and the router's lowest and
highest solve rate. With a single run every spread is 0.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.paths import OUTPUTS
from analysis.stats.instance_costs import ROUTER, run_costs, run_outcomes
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.solver_outcomes import CORPORA, scope_costs


def load_results(dirs):
    """The columns of joined.csv this script uses, read from results directories laid out as
    run_benchmarks writes them (multi-run <corpus>/run_N/ or single-run <corpus>/)."""
    rows, seen = [], set()
    for root in map(Path, dirs):
        multi = sorted(root.glob("*/run_*/*_int.json"))
        files = [(f, f.parts[-3], f.parts[-2]) for f in multi] or \
                [(f, f.parts[-2], "run_1") for f in sorted(root.glob("*/*_int.json"))]
        if not files:
            raise SystemExit(f"no result files in {root}")
        for f, corpus, run in files:
            encoding = f.stem.removesuffix("_int")
            for rec in json.loads(f.read_text()):
                iid = rec.get("constraint_id", rec.get("id"))
                if (iid, encoding, run) in seen:
                    raise SystemExit(f"{iid} on {encoding} in {run} is in more than one directory ({f})")
                seen.add((iid, encoding, run))
                rows.append({"id": iid, "corpus": corpus, "encoding": encoding, "run": run,
                             "time": rec["execution_time"], "status": rec["status"]})
    return pd.DataFrame(rows)


def scope_variance(df, timeout_cost, corpus=None):
    """One row per encoding (and the router) for one scope; corpus None or 'all' pools."""
    if corpus and corpus != "all":
        df = df[df["corpus"] == corpus]
    costs, _ = scope_costs(df, timeout_cost)
    n = len(costs)
    times = run_costs(df, timeout_cost, costs.index)
    outcomes = run_outcomes(df, timeout_cost, costs.index)
    rows = []
    for e in sorted(times.columns, key=lambda e: (e == ROUTER, e)):
        mean = times[e].dropna().groupby(level="run").mean()
        solved = outcomes[e].dropna().isin(["sat", "unsat"]).groupby(level="run").sum()
        med = mean.median()
        rows.append({
            "encoding": e,
            "instances": n,
            "runs": len(mean),
            "mean_time_median": med,
            "mean_time_min": mean.min(),
            "mean_time_max": mean.max(),
            "mean_time_spread_pct": 100 * (mean.max() - mean.min()) / med,
            "mean_time_dev_pct": 100 * (mean - med).abs().max() / med,
            "solved_min": int(solved.min()),
            "solved_max": int(solved.max()),
            "solved_spread": int(solved.max() - solved.min()),
            "solve_rate_min": 100 * solved.min() / n,
            "solve_rate_max": 100 * solved.max() / n,
        })
    return pd.DataFrame(rows).set_index("encoding")


def summary(tables):
    lines = []
    for scope, t in tables.items():
        enc = t[t.index != ROUTER]
        i, j = enc["mean_time_spread_pct"].idxmax(), enc["solved_spread"].idxmax()
        lines.append(f"{scope:8s} encodings: mean time spread at most "
                     f"{enc['mean_time_spread_pct'].max():.1f}% ({i}; largest deviation from "
                     f"the median {enc['mean_time_dev_pct'].max():.1f}%), solved spread at most "
                     f"{enc['solved_spread'].max()} ({j})")
        if ROUTER in t.index:
            r = t.loc[ROUTER]
            lines.append(f"{'':8s} router: solve rate {r['solve_rate_min']:.1f}% to "
                         f"{r['solve_rate_max']:.1f}% ({int(r['solved_min'])}-{int(r['solved_max'])} of "
                         f"{int(r['instances'])}), over {int(r['runs'])} runs")
    per_corpus = pd.concat([t for s, t in tables.items() if s != "all"])
    enc = per_corpus[per_corpus.index != ROUTER]
    lines.append(f"over the corpora: mean time spread at most {enc['mean_time_spread_pct'].max():.1f}%, "
                 f"solved spread at most {enc['solved_spread'].max()}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    src.add_argument("--results", nargs="+", metavar="DIR",
                     help="results directories to read instead of joined.csv")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="ENCODING",
                    help="encodings to leave out, e.g. simple")
    add_timeout_arg(ap)
    ap.add_argument("--output", help="also write the rows as CSV, with a `scope` column")
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)

    joined = load_results(args.results) if args.results else pd.read_csv(args.joined)
    joined = joined[~joined["encoding"].isin(args.exclude)]
    tables = {s: scope_variance(joined, timeout, s) for s in ["all"] + CORPORA}
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.4g}".format):
        for s, t in tables.items():
            print(f"== {s}\n{t.drop(columns='instances').to_string()}\n")
    print(summary(tables))
    if args.output:
        out = pd.concat([t.assign(scope=s) for s, t in tables.items()]).reset_index()
        out[["scope"] + [c for c in out.columns if c != "scope"]].to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
