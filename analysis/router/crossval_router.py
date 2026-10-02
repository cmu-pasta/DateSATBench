"""
Stage 6: measure the router on constraints it has not seen, by cross-validation over every
instance.

    python -m analysis.router.crossval_router --timeout 20000     # or set DATESAT_TIMEOUT_MS

It measures one fixed setting: by default the one train_router.py trains (its
DEFAULT_SETTINGS and DEFAULT_TREES: leaf size, depth, margin and trees per forest), so
the figures describe the model DateSat uses. It does not search for a better setting;
--min-samples-leaf, --max-depth, --margin and --trees measure another one.

Features. By default the router sees every feature column. With --selected it sees only
the features select_features.py kept (outputs/selected_features.json), and the results go
to model_cross_validation_selected/ instead, so the two can be compared.

How. The instances of joined.csv, dropped as in train_router.py (each one a single
example per pair, its cost the median over its runs), are cut into --folds folds (default
5), stratified by corpus: each corpus is shuffled and dealt evenly into the folds, so
every fold holds the same share of every corpus. For every fold, the router is trained on
the other folds and routes the fold, so every instance is routed once by a router that
has not seen it; the fallback is the best performing encoding on the other folds. The
whole is repeated --repeats times (default 5), each time with a different cut into folds,
and the results are averaged over the repeats. The cuts come from a fixed seed and the
forests use train_router.py's, so a run of this script always gives the same results.

Writes to analysis/outputs/model_cross_validation/ (with --selected,
model_cross_validation_selected/):
    router_eval.json   the setting, the feature columns, and averaged over the repeats,
                       overall and for each
                       corpus: a table of always each encoding, the router and the
                       oracle (total time, timeouts, and speedups over always
                       --baseline: of the totals, and the median and geometric mean of
                       the per-instance speedup), the router's picks and fallbacks, how often it picks
                       the fastest encoding, how much of the gap to the oracle it closes,
                       and the spread of the router's total over the repeats
    router_eval.csv    one row per repeat and instance: its fold and corpus, its cost on
                       every encoding, the best encoding and its cost, the router's pick,
                       whether it fell back, the fold's fallback and the router's cost
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from analysis.paths import CROSS_VALIDATION, CROSS_VALIDATION_SELECTED, OUTPUTS, REPO
from analysis.router.train_router import (DEFAULT_SETTINGS, DEFAULT_TREES, ROUTER,
                                          feature_columns, fit_forests, instance_features,
                                          route, training_set)
from analysis.stats.join_results import add_timeout_arg, timeout_seconds
from analysis.stats.select_features import selected_columns

SEED = 0


def strategy_costs(costs, picks):
    """Per-instance cost of every strategy: always-<encoding>, the router (the cost of the
    encoding it picks), and the oracle."""
    out = {f"always {e}": costs[e] for e in costs.columns}
    out["router"] = pd.Series([costs.at[iid, picks[iid]] for iid in costs.index], index=costs.index)
    out["oracle"] = costs.min(axis=1)
    return pd.DataFrame(out)


def summary_table(strategies, baseline, timeout_cost):
    """Total time, timeouts and speedup over always-`baseline`, one row per strategy of
    strategy_costs: of the totals, and the median and geometric mean over instances of the
    per-instance speedup (as in solver_outcomes.py, timeouts at the timeout)."""
    base = strategies[f"always {baseline}"]
    per_instance = strategies.rdiv(base, axis=0)
    return pd.DataFrame({
        "total_s": strategies.sum(),
        "timeouts": (strategies >= timeout_cost).sum(),
        "speedup_total": base.sum() / strategies.sum(),
        "speedup_median": per_instance.median(),
        "speedup_geomean": np.exp(np.log(per_instance).mean()),
    })


def gap_closed(table, start, strategy="router"):
    """How much of the distance from strategy `start` to the oracle `strategy` covers."""
    gap = table.at[start, "total_s"] - table.at["oracle", "total_s"]
    if gap == 0:
        return None
    return float((table.at[start, "total_s"] - table.at[strategy, "total_s"]) / gap)


def summarize(strategies, picks, fell_back, baseline, best, timeout_cost):
    """The summary of one set of instances: summary_table, the router's picks and
    fallbacks, how often it picks the fastest encoding, and how much of the gap to the
    oracle it closes, from always `baseline` and from always `best`."""
    table = summary_table(strategies, baseline, timeout_cost)
    starts = dict.fromkeys([f"always {baseline}", f"always {best}"])   # one if they match
    return {
        "instances": len(strategies),
        "table": table,
        "picks": picks.value_counts(),
        "fell_back": int(fell_back.sum()),
        "picks_fastest": float((strategies["router"] == strategies["oracle"]).mean()),
        "gap_closed": {"router": {s: gap_closed(table, s) for s in starts}},
    }


def number(x):
    """A count as an int, or an average over repeats rounded."""
    x = float(x)
    return int(x) if x.is_integer() else round(x, 2)


def print_summary(title, s, baseline, encodings, fallback, margin):
    print(f"\n{title} ({s['instances']} instances)")
    print(f"{'strategy':22} {'total s':>9} {'timeouts':>9}   speedup over always {baseline}")
    print(f"{'':22} {'':>9} {'':>9}   {'total':>7} {'median':>8}")
    for name, r in s["table"].iterrows():
        print(f"{name:22} {r.total_s:9.1f} {number(r.timeouts):>9}   "
              f"{r.speedup_total:6.2f}x {r.speedup_median:7.2f}x")
    print("router picks: " + ", ".join(f"{e} {number(s['picks'].get(e, 0))}" for e in encodings))
    print(f"fell back to {fallback} on {number(s['fell_back'])} instances (margin {margin:g})")
    print(f"router picks the fastest encoding on {s['picks_fastest']:.1%} of instances")
    for router, closed in s["gap_closed"].items():
        for start, frac in closed.items():
            if frac is not None:
                print(f"{router} closes {frac:.1%} of the gap from {start} to the oracle")


def summary_json(s, encodings):
    return {
        "instances": s["instances"],
        "table": {name: {"total_s": round(float(r.total_s), 3), "timeouts": number(r.timeouts),
                         "speedup_total": round(float(r.speedup_total), 4),
                         "speedup_median": round(float(r.speedup_median), 4),
                         "speedup_geomean": round(float(r.speedup_geomean), 4)}
                  for name, r in s["table"].iterrows()},
        "router_picks": {e: number(s["picks"].get(e, 0)) for e in encodings},
        "router_fell_back": number(s["fell_back"]),
        "router_picks_fastest": round(s["picks_fastest"], 4),
        "gap_closed_to_oracle": {r: {k: None if v is None else round(v, 4) for k, v in c.items()}
                                 for r, c in s["gap_closed"].items()},
        "router_total_sd_s": round(s.get("router_total_sd", 0.0), 3),
    }


def out_of_fold_picks(costs, feats, strata, min_samples_leaf, max_depth, margin, n_splits,
                      repeats, trees):
    """Route every instance of `costs` with routers trained on the other folds, with one
    setting.

    Returns one row per repeat and instance: the fold, the pick, whether it fell back, the
    fold's fallback, and the ids the router was trained on.
    """
    encodings = sorted(costs.columns)
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    rows = []
    for k, (train, test) in enumerate(folds.split(costs, strata.loc[costs.index])):
        repeat, fold = divmod(k, n_splits)
        c_train, c_test = costs.iloc[train], costs.iloc[test]
        fallback = c_train.sum().idxmin()
        forests = fit_forests(c_train, feats.loc[c_train.index], trees, max_depth,
                              min_samples_leaf, n_jobs=-1)
        picks, fell_back, _ = route(forests, encodings, feats.loc[c_test.index], fallback, margin)
        rows += [{"repeat": repeat, "fold": fold, "id": iid, "pick": picks[iid],
                  "fell_back": bool(fell_back[iid]), "fallback": fallback,
                  "trained_on": tuple(c_train.index)}
                 for iid in c_test.index]
    return pd.DataFrame(rows)


def average_summaries(summaries, baseline, best):
    """The mean of summarize results, one per repeat, over the same instances. The gap
    closed is recomputed from the averaged totals, and router_total_sd is the standard
    deviation of the router's total over the repeats."""
    table = pd.concat([s["table"] for s in summaries]).groupby(level=0, sort=False).mean()
    starts = dict.fromkeys([f"always {baseline}", f"always {best}"])
    return {
        "instances": summaries[0]["instances"],
        "table": table,
        "picks": pd.concat([s["picks"] for s in summaries], axis=1).fillna(0).mean(axis=1),
        "fell_back": sum(s["fell_back"] for s in summaries) / len(summaries),
        "picks_fastest": sum(s["picks_fastest"] for s in summaries) / len(summaries),
        "gap_closed": {"router": {s: gap_closed(table, s) for s in starts}},
        "router_total_sd": float(np.std([s["table"].at["router", "total_s"] for s in summaries],
                                        ddof=1)) if len(summaries) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    add_timeout_arg(ap)
    ap.add_argument("--timeout-cost", type=float,
                    help="ms that a timed-out run counts as (default: the timeout)")
    ap.add_argument("--trees", type=int, default=DEFAULT_TREES, help="trees per forest")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_SETTINGS["max_depth"],
                    help="maximum depth of each tree")
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_SETTINGS["min_samples_leaf"],
                    help="fewest training instances a leaf may rest on")
    ap.add_argument("--margin", type=float, default=DEFAULT_SETTINGS["margin"],
                    help="how far above 0.5 a pick must beat the fallback to be kept")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--baseline", default="simple", help="encoding that speedups are measured against")
    ap.add_argument("--selected", action="store_true",
                    help="use only the features in selected_features.json (select_features.py)")
    ap.add_argument("--output-dir", help="default: outputs/model_cross_validation/, or "
                                         "model_cross_validation_selected/ with --selected")
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = str(CROSS_VALIDATION_SELECTED if args.selected else CROSS_VALIDATION)
    timeout = timeout_seconds(ap, args)
    timeout_cost = timeout if args.timeout_cost is None else args.timeout_cost / 1000
    if timeout_cost < timeout:
        ap.error(f"--timeout-cost must be at least the timeout ({timeout * 1000:g} ms)")
    if not 0 <= args.margin < 0.5:
        ap.error(f"--margin must be in [0, 0.5), got {args.margin:g}")
    leaf, depth, margin = args.min_samples_leaf, args.max_depth, args.margin

    joined = pd.read_csv(args.joined)
    joined = joined[joined["encoding"] != ROUTER]
    costs, dropped = training_set(joined, timeout_cost)
    cols = feature_columns(joined)
    if args.selected:
        selected = selected_columns()
        missing = [c for c in selected if c not in cols]
        if missing:
            ap.error(f"selected features not in {args.joined}: {', '.join(missing)}")
        cols = [c for c in cols if c in selected]
    feats = instance_features(joined, cols, costs.index)
    corpus = joined.groupby("id")["corpus"].first().loc[costs.index]
    encodings = sorted(costs.columns)
    if args.baseline not in encodings:
        ap.error(f"--baseline {args.baseline!r} is not one of {encodings}")
    best = costs.sum().idxmin()

    print(f"{len(costs)} instances (dropped {dropped['errored_or_missing']} errored or missing "
          f"and {dropped['all_timeout']} all-timeout): "
          + ", ".join(f"{c} {n}" for c, n in corpus.value_counts().sort_index().items())
          + f"; timeouts count as {timeout_cost:g} s")
    print(f"features: {len(cols)} columns" + (" (selected_features.json)" if args.selected else ""))
    print(f"setting: {args.trees} trees, depth <= {depth}, at least {leaf} instances per leaf, "
          f"margin {margin:g}; {args.folds}-fold cross-validation stratified by corpus, "
          f"repeated {args.repeats} times ...")
    oof = out_of_fold_picks(costs, feats, corpus, leaf, depth, margin, args.folds, args.repeats,
                            args.trees).drop(columns="trained_on")

    overall, by_corpus, logs = [], {c: [] for c in sorted(corpus.unique())}, []
    for repeat, g in oof.groupby("repeat"):
        g = g.set_index("id").loc[costs.index]
        strategies = strategy_costs(costs, g["pick"])
        overall.append(summarize(strategies, g["pick"], g["fell_back"], args.baseline, best,
                                 timeout_cost))
        for c in by_corpus:
            m = corpus == c
            by_corpus[c].append(summarize(strategies[m], g["pick"][m], g["fell_back"][m],
                                          args.baseline, best, timeout_cost))
        log = pd.DataFrame({"repeat": repeat, "fold": g["fold"], "corpus": corpus})
        for e in encodings:
            log[f"cost_{e}"] = costs[e]
        log["best"] = costs.idxmin(axis=1)
        log["best_cost"] = strategies["oracle"]
        log["router"] = g["pick"]
        log["fell_back"] = g["fell_back"]
        log["fallback"] = g["fallback"]
        log["router_cost"] = strategies["router"]
        logs.append(log)
    overall = average_summaries(overall, args.baseline, best)
    by_corpus = {c: average_summaries(s, args.baseline, best) for c, s in by_corpus.items()}

    fallbacks = sorted(oof["fallback"].unique())
    print(f"\n===== out-of-fold, averaged over {args.repeats} repeats =====")
    print(f"router total over the repeats: {overall['table'].at['router', 'total_s']:.1f} s "
          f"+/- {overall['router_total_sd']:.1f} s (standard deviation)")
    print_summary("all corpora", overall, args.baseline, encodings, "/".join(fallbacks), margin)
    for c, s in by_corpus.items():
        print_summary(c, s, args.baseline, encodings, "/".join(fallbacks), margin)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(logs).to_csv(out / "router_eval.csv", index_label="id", float_format="%.6g")
    summary = {
        "joined": os.path.relpath(args.joined, REPO),
        "evaluated_on": f"every instance, out of fold: {args.folds}-fold cross-validation "
                        f"stratified by corpus, repeated {args.repeats} times",
        "instances": len(costs),
        "dropped": dropped,
        "timeout_cost_s": timeout_cost,
        "folds": args.folds,
        "repeats": args.repeats,
        "trees": args.trees,
        "setting": {"min_samples_leaf": leaf, "max_depth": depth, "margin": margin},
        "feature_columns": cols,
        "fallbacks": fallbacks,
        "baseline": args.baseline,
        "best_performing_encoding": best,
        "overall": summary_json(overall, encodings),
        "by_corpus": {c: summary_json(s, encodings) for c, s in by_corpus.items()},
    }
    (out / "router_eval.json").write_text(json.dumps(summary, indent=1))

    if summary["setting"] != DEFAULT_SETTINGS or args.trees != DEFAULT_TREES:
        print(f"\nNOTE: this is not the setting train_router.py trains by default "
              f"({DEFAULT_SETTINGS}, {DEFAULT_TREES} trees)")
    print(f"\nWrote {out / 'router_eval.json'} and {out / 'router_eval.csv'}")


if __name__ == "__main__":
    main()
