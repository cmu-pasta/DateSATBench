"""
Summarise what each encoding did: status counts, solve times and wins over the baseline.

    python analysis/solver_outcomes.py                    # all corpora pooled
    python analysis/solver_outcomes.py --corpus legal     # one corpus
    python analysis/solver_outcomes.py --compare          # pooled, then every corpus
    python analysis/solver_outcomes.py --compare --output analysis/outcomes.csv

Reads joined.csv and prints one row per encoding for each scope. --output writes the
same rows as CSV with a `scope` column (all, llm, legal, grammar). build_report.py
calls solver_outcomes() for every scope, so the report shows these exact numbers.

Columns
  rows                 measurements for this encoding (instances x runs)
  sat, unsat, timeout  status counts; `error` counts any other status
  solved               sat + unsat
  median_time          median time over the rows THIS encoding solved. Each encoding
                       solves a different subset, so this does not compare encodings:
                       one that solves only the easy instances looks fast.
  n_common             (instance, run) pairs that every encoding solved; the same
                       for every row of a scope
  median_time_common   median time over those shared pairs: the like-for-like time
  wins / compared      rows with speedup > 1 over the baseline, out of rows that have
                       a speedup. One-sided timeouts count at their bound, rows where
                       both sides timed out have none (see join_results.py).
  median_speedup       median speedup over exact pairs only (both sides solved)
  exact, lower, upper  how many of this encoding's speedups are exact or bounds
  no_speedup           rows with no speedup (both timed out, or either errored)

The baseline's win and speedup columns are empty: it is not compared with itself.
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CORPORA = ["llm", "legal", "grammar"]
KEY = ["id", "run"]
COUNTS = ["rows", "sat", "unsat", "timeout", "error", "solved", "n_common",
          "wins", "compared", "exact", "lower", "upper", "no_speedup"]


def solver_outcomes(df, baseline="simple", corpus=None):
    """One row per encoding (baseline first) for one scope; corpus None or 'all' pools."""
    if corpus and corpus != "all":
        df = df[df["corpus"] == corpus]
    encodings = [baseline] + sorted(e for e in df["encoding"].unique() if e != baseline)

    # A pair is shared only if every encoding has a row for it and solved it.
    solved_by = df.pivot_table(index=KEY, columns="encoding", values="solved", aggfunc="max")
    solved_by = solved_by.reindex(columns=encodings)
    common = solved_by.index[(solved_by == 1).all(axis=1)]

    rows = []
    for e in encodings:
        s = df[df["encoding"] == e]
        status = s["status"]
        solved = s[s["solved"] == 1]
        shared = s[s.set_index(KEY).index.isin(common)]
        sp = s[s["speedup"].notna()]
        exact = sp[sp["speedup_bound"] == "exact"]
        compared = e != baseline
        rows.append({
            "encoding": e,
            "rows": len(s),
            "sat": int((status == "sat").sum()),
            "unsat": int((status == "unsat").sum()),
            "timeout": int((status == "timeout").sum()),
            "error": int((~status.isin(["sat", "unsat", "timeout"])).sum()),
            "solved": len(solved),
            "median_time": solved["time"].median() if len(solved) else None,
            "n_common": len(common),
            "median_time_common": shared["time"].median() if len(shared) else None,
            "wins": int((sp["speedup"] > 1).sum()) if compared else None,
            "compared": len(sp) if compared else None,
            "median_speedup": exact["speedup"].median() if compared and len(exact) else None,
            "exact": int((s["speedup_bound"] == "exact").sum()),
            "lower": int((s["speedup_bound"] == "lower").sum()),
            "upper": int((s["speedup_bound"] == "upper").sum()),
            "no_speedup": int(s["speedup"].isna().sum()),
        })
    out = pd.DataFrame(rows).set_index("encoding")
    return out.astype({c: "Int64" for c in COUNTS}).astype(
        {c: "float64" for c in ("median_time", "median_time_common", "median_speedup")})


def fmt_time(t):
    if pd.isna(t):
        return "-"
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


def display(t, scope):
    """Human-readable version of one scope's table."""
    rows = []
    for e, r in t.iterrows():
        base = pd.isna(r["compared"])
        rows.append({
            "encoding": e,
            "solved": f"{r['solved']}/{r['rows']}",
            "sat": r["sat"], "unsat": r["unsat"], "timeout": r["timeout"],
            "median time (own solves)": fmt_time(r["median_time"]),
            "median time (solved by all)": fmt_time(r["median_time_common"]),
            "beats baseline": "baseline" if base else "-" if r["compared"] == 0
            else f"{r['wins']}/{r['compared']} ({100 * r['wins'] / r['compared']:.0f}%)",
            "median exact speedup": "-" if pd.isna(r["median_speedup"])
            else f"{r['median_speedup']:.2f}x",
        })
    n_common = int(t["n_common"].iloc[0])
    head = f"{scope}: {int(t['rows'].iloc[0])} rows per encoding, {n_common} solved by every encoding"
    return head + "\n" + pd.DataFrame(rows).set_index("encoding").to_string()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", default=str(HERE / "joined.csv"))
    ap.add_argument("--baseline", default="simple")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--corpus", choices=CORPORA, help="one corpus instead of all pooled")
    scope.add_argument("--compare", action="store_true",
                       help="pooled, then every corpus, one table each")
    ap.add_argument("--output", help="also write the rows as CSV, with a `scope` column")
    args = ap.parse_args()

    joined = pd.read_csv(args.joined)
    scopes = ["all"] + CORPORA if args.compare else [args.corpus or "all"]
    tables = {s: solver_outcomes(joined, args.baseline, s) for s in scopes}
    print("\n\n".join(display(t, s) for s, t in tables.items()))

    if args.output:
        out = pd.concat([t.assign(scope=s) for s, t in tables.items()]).reset_index()
        out = out[["scope"] + [c for c in out.columns if c != "scope"]]
        out.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
