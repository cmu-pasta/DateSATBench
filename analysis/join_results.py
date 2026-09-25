"""
Join solver timings to the feature matrix.

    python analysis/join_results.py --results results/bench-datetime-bound --timeout 60000

Reads every <corpus>/run_N/<approach>_<impl>.json (or, for a single-run eval,
<corpus>/<approach>_<impl>.json) under the results root and joins each record to
features.csv on id. Nothing is aggregated: every timing measurement becomes its own
row, so an instance solved by 4 encodings over 3 runs contributes 12 rows.

Speedup is relative to the baseline encoding (--baseline, default `simple`) on the
same instance in the same run:

    speedup = baseline_time / time          (> 1 means faster than the baseline)

Timeouts. The solver timeout the results were run with must be given, in ms, with
--timeout or the DATESAT_TIMEOUT_MS environment variable (--timeout wins). A timed-out
side counts at that timeout, so a one-sided timeout still bounds the speedup, and the
row keeps it:

    baseline timed out, encoding finished  ->  speedup = timeout / time,
                                               a LOWER bound  (speedup_bound = "lower")
    encoding timed out, baseline finished  ->  speedup = baseline_time / timeout,
                                               an UPPER bound (speedup_bound = "upper")
    both finished                          ->  exact          (speedup_bound = "exact")
    both timed out, or either errored      ->  speedup empty  (speedup_bound empty)

Rows are never dropped from joined.csv; analyses skip rows with an empty speedup.
`time` and `baseline_time` stay as measured; filter on `solved` before using `time` as
a measurement.

Writes analysis/joined.csv, one row per (instance, encoding, run).
"""

import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
FINISHED = {"sat", "unsat"}
TIMEOUT_ENV = "DATESAT_TIMEOUT_MS"


def add_timeout_arg(ap):
    """--timeout in ms, falling back to $DATESAT_TIMEOUT_MS; there is no default."""
    ap.add_argument("--timeout", type=float, default=os.environ.get(TIMEOUT_ENV),
                    help=f"solver timeout in ms that the results were run with "
                         f"(default: ${TIMEOUT_ENV}; one of the two is required)")


def timeout_seconds(ap, args):
    """The timeout from add_timeout_arg in seconds, or exit if it was not given."""
    if args.timeout is None:
        ap.error(f"give the solver timeout with --timeout <ms> or {TIMEOUT_ENV}")
    if args.timeout <= 0:
        ap.error(f"the timeout must be positive, got {args.timeout:g} ms")
    return args.timeout / 1000


def check_run_config(results, timeout_s):
    """Warn when the results' run_config.json records a different timeout."""
    cfg = Path(results) / "run_config.json"
    if not cfg.exists():
        return
    run_ms = json.loads(cfg.read_text()).get("timeout_ms")
    if run_ms is not None and run_ms / 1000 != timeout_s:
        print(f"WARNING: timeout is {timeout_s:g} s but {cfg} records "
              f"timeout_ms={run_ms} ({run_ms / 1000:g} s)")


def speedup_and_bound(time, status, base_time, base_status, timeout):
    """Speedup over the baseline, and whether it is exact or a lower/upper bound.

    A timed-out side counts at `timeout` (seconds) instead of its measured time.
    """
    if status in FINISHED and base_status in FINISHED:
        bound = "exact"
    elif status in FINISHED and base_status == "timeout":
        bound, base_time = "lower", timeout
    elif status == "timeout" and base_status in FINISHED:
        bound, time = "upper", timeout
    else:
        return None, None
    if not time or base_time is None:
        return None, None
    return base_time / time, bound


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(HERE.parent / "results/bench-datetime-bound"),
                    help="results directory (multi-run <corpus>/run_N/ or single-run <corpus>/)")
    ap.add_argument("--features", default=str(HERE / "features.csv"))
    ap.add_argument("--baseline", default="simple", help="encoding that speedups are measured against")
    add_timeout_arg(ap)
    ap.add_argument("--output", default=str(HERE / "joined.csv"))
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)
    check_run_config(args.results, timeout)

    root = Path(args.results)
    obs = {}                          # (id, encoding, run) -> (time, status)
    corpora, encodings, run_ids = set(), set(), set()

    multi = sorted(root.glob("*/run_*/*_int.json"))
    files = [(f, f.parts[-3], f.parts[-2]) for f in multi] or \
            [(f, f.parts[-2], "run_1") for f in sorted(root.glob("*/*_int.json"))]
    for f, corpus_dir, run_id in files:
        encoding = f.stem.replace("_int", "")
        corpora.add(corpus_dir)
        encodings.add(encoding)
        run_ids.add(run_id)
        for rec in json.loads(f.read_text()):
            iid = rec["constraint_id"] if "constraint_id" in rec else rec["id"]
            obs[(iid, encoding, run_id)] = (rec["execution_time"], rec["status"])

    print(f"corpora: {sorted(corpora)}")
    print(f"encodings: {sorted(encodings)}")
    print(f"runs: {sorted(run_ids)}")
    print(f"timeout: {timeout:g} s")
    if args.baseline not in encodings:
        raise SystemExit(f"baseline encoding {args.baseline!r} not found in results")

    feats = {r["id"]: r for r in csv.DictReader(open(args.features))}
    feature_cols = [
        c for c in next(iter(feats.values()))
        if c not in ("id", "corpus", "label_status", "label_execution_time")
    ]

    rows = []
    missing, no_baseline = set(), 0
    for (iid, encoding, run_id), (time, status) in sorted(obs.items()):
        if iid not in feats:
            missing.add(iid)
            continue
        base = obs.get((iid, args.baseline, run_id))
        if base is None:
            no_baseline += 1
        base_time, base_status = base if base else (None, None)

        speedup, bound = speedup_and_bound(time, status, base_time, base_status, timeout)

        row = {
            "id": iid,
            "corpus": feats[iid]["corpus"],
            "encoding": encoding,
            "run": run_id,
            "time": time,
            "status": status,
            "solved": int(status in FINISHED),
            "baseline_time": base_time,
            "baseline_status": base_status,
            "speedup": speedup,
            "speedup_bound": bound,
        }
        for c in feature_cols:
            row[c] = feats[iid][c]
        rows.append(row)

    if missing:
        print(f"WARNING: {len(missing)} result ids had no matching feature row")
    if no_baseline:
        print(f"WARNING: {no_baseline} rows had no {args.baseline} measurement in the same run")

    cols = ["id", "corpus", "encoding", "run", "time", "status", "solved",
            "baseline_time", "baseline_status", "speedup", "speedup_bound"] + feature_cols
    out = Path(args.output)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_inst = len({r["id"] for r in rows})
    n_bound = {b: sum(r["speedup_bound"] == b for r in rows) for b in ("exact", "lower", "upper")}
    n_none = sum(r["speedup"] is None for r in rows)
    print(f"\nWrote {len(rows)} rows ({n_inst} instances, {len(encodings)} encodings, "
          f"{len(run_ids)} runs) to {out}")
    print(f"  speedup: {n_bound['exact']} exact, {n_bound['lower']} lower bound, "
          f"{n_bound['upper']} upper bound, {n_none} empty")


if __name__ == "__main__":
    main()
