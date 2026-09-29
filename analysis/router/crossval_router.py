"""
Stage 6: cross-validate the router over every instance, to choose its settings and to
measure how it does on constraints it has not seen.

    python -m analysis.router.crossval_router --timeout 20000     # or set DATESAT_TIMEOUT_MS

Why. Each pairwise forest weighs an example by the time gap between its two encodings, so
a leaf says "A is faster" exactly when A's cost, averaged over the leaf's training
instances, is lower than B's. With leaves allowed to rest on a single instance, that
average is one instance's cost: the trees memorize the training constraints, above all
the few whose timeouts carry most of the weight, and route new constraints with
confidence they have not earned. A minimum number of instances per leaf makes every
leaf's average rest on several constraints; the margin makes the router leave the
fallback only on a clear vote. Cross-validation chooses both.

How. The instances of joined.csv, dropped as in train_router.py (each one a single
example per pair, its cost the median over its runs), are cut into --folds folds (default
5), stratified by corpus: each corpus is shuffled and dealt evenly into the folds, so
every fold holds the same share of every corpus. For every fold, the router is trained on
the other folds and routes the fold, so every instance is routed once by a router that
has not seen it; the fallback is the best performing encoding on the other folds. The
whole is repeated --repeats times (default 3), each time with a different cut into folds,
and the results are averaged over the repeats. The cuts come from a fixed seed, so a run
of this script always gives the same results.

A setting is a leaf size (min_samples_leaf), a maximum depth and a margin; the number of
trees stays at --trees. Its score is the out-of-fold total time, averaged over the
repeats, with a timeout counting at the timeout cost. The best setting is the one with
the lowest score; train_router.py's defaults should be it. The same cross-validation both
chooses the best setting and reports how it does, so the report is slightly optimistic:
part of why a setting wins is luck on these instances.

Writes to analysis/outputs/model_cross_validation/:
    router_tuning.csv  one row per setting, best first: out-of-fold total time, its spread
                       (standard deviation) over the repeats, timeouts, the share of
                       instances it fell back on, always using the fold's fallback, the
                       oracle, and the share of the gap between the two that it closes
    router_eval.json   the best setting, averaged over the repeats, overall and for each
                       corpus: a table of always each encoding, the router and the
                       oracle (total time, timeouts, speedups over always
                       --baseline), the router's picks and fallbacks, how often it picks
                       the fastest encoding, and how much of the gap to the oracle it closes
    router_eval.csv    the best setting, one row per repeat and instance: its fold and
                       corpus, its cost on every encoding, the best encoding and its cost,
                       the router's pick, whether it fell back, the fold's fallback and
                       the router's cost
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from analysis.paths import CROSS_VALIDATION, OUTPUTS, REPO
from analysis.router.train_router import (DEFAULT_SETTINGS, ROUTER, feature_columns,
                                          fit_forests, instance_features, pair_probabilities,
                                          route, training_set)
from analysis.stats.join_results import add_timeout_arg, timeout_seconds

SEED = 0
LEAF_SIZES = (1, 5, 10, 20, 40)
DEPTHS = (4, 8, 16)
MARGINS = (0.0, 0.1, 0.2, 0.3, 0.4)
KEY = ["min_samples_leaf", "max_depth", "margin"]
BEFORE_TUNING = (1, 8, 0.0)        # the settings the router had before cross-validation


def strategy_costs(costs, picks):
    """Per-instance cost of every strategy: always-<encoding>, the router (the cost of the
    encoding it picks), and the oracle."""
    out = {f"always {e}": costs[e] for e in costs.columns}
    out["router"] = pd.Series([costs.at[iid, picks[iid]] for iid in costs.index], index=costs.index)
    out["oracle"] = costs.min(axis=1)
    return pd.DataFrame(out)


def summary_table(strategies, baseline, timeout_cost):
    """Total time, timeouts and speedup over always-`baseline`, one row per strategy of
    strategy_costs."""
    base = strategies[f"always {baseline}"]
    return pd.DataFrame({
        "total_s": strategies.sum(),
        "timeouts": (strategies >= timeout_cost).sum(),
        "speedup_total": base.sum() / strategies.sum(),
        "speedup_geomean": np.exp(np.log(strategies.rdiv(base, axis=0)).mean()),
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
    print(f"{'':22} {'':>9} {'':>9}   {'total':>7} {'geomean':>8}")
    for name, r in s["table"].iterrows():
        print(f"{name:22} {r.total_s:9.1f} {number(r.timeouts):>9}   "
              f"{r.speedup_total:6.2f}x {r.speedup_geomean:7.2f}x")
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
                         "speedup_geomean": round(float(r.speedup_geomean), 4)}
                  for name, r in s["table"].iterrows()},
        "router_picks": {e: number(s["picks"].get(e, 0)) for e in encodings},
        "router_fell_back": number(s["fell_back"]),
        "router_picks_fastest": round(s["picks_fastest"], 4),
        "gap_closed_to_oracle": {r: {k: None if v is None else round(v, 4) for k, v in c.items()}
                                 for r, c in s["gap_closed"].items()},
    }


def out_of_fold_picks(costs, feats, strata, settings, margins, n_splits, repeats, trees):
    """Route every instance of `costs` with routers trained on the other folds.

    `settings` are (min_samples_leaf, max_depth) pairs; every one is combined with every
    margin. Returns one row per repeat, setting, margin and instance: the fold, the pick,
    whether it fell back, the fold's fallback, and the ids the router was trained on.
    """
    encodings = sorted(costs.columns)
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    rows = []
    for k, (train, test) in enumerate(folds.split(costs, strata.loc[costs.index])):
        repeat, fold = divmod(k, n_splits)
        c_train, c_test = costs.iloc[train], costs.iloc[test]
        fallback = c_train.sum().idxmin()
        trained_on = tuple(c_train.index)
        for leaf, depth in settings:
            forests = fit_forests(c_train, feats.loc[c_train.index], trees, depth, leaf, n_jobs=-1)
            probs = pair_probabilities(forests, feats.loc[c_test.index])
            for margin in margins:
                picks, fell_back, _ = route(forests, encodings, feats.loc[c_test.index],
                                            fallback, margin, probs=probs)
                rows += [{"repeat": repeat, "fold": fold, "id": iid, "min_samples_leaf": leaf,
                          "max_depth": depth, "margin": margin, "pick": picks[iid],
                          "fell_back": bool(fell_back[iid]), "fallback": fallback,
                          "trained_on": trained_on}
                         for iid in c_test.index]
    return pd.DataFrame(rows)


def score(oof, costs, timeout_cost):
    """One row per setting (indexed by KEY), best first: out-of-fold total time and
    timeouts, the share of instances it fell back on, always using the fold's fallback,
    the oracle, and the share of the gap from the fallback to the oracle it closes. Totals
    are summed over the instances of a repeat and averaged over the repeats."""
    lookup = costs.stack()
    oof = oof.assign(cost=lookup.loc[list(zip(oof["id"], oof["pick"]))].to_numpy(),
                     fallback_cost=lookup.loc[list(zip(oof["id"], oof["fallback"]))].to_numpy())
    oof = oof.assign(timeout=oof["cost"] >= timeout_cost)
    per_repeat = oof.groupby(KEY + ["repeat"]).agg(
        total_s=("cost", "sum"), timeouts=("timeout", "sum"), fell_back=("fell_back", "mean"),
        always_fallback_s=("fallback_cost", "sum"))
    table = per_repeat.groupby(level=KEY).mean()
    table["total_sd"] = per_repeat["total_s"].groupby(level=KEY).std()
    table["oracle_s"] = costs.loc[oof["id"].unique()].min(axis=1).sum()
    gap = table["always_fallback_s"] - table["oracle_s"]
    table["gap_closed"] = (table["always_fallback_s"] - table["total_s"]) / gap
    return table.sort_values(["total_s", "timeouts"])


def average_summaries(summaries, baseline, best):
    """The mean of summarize results, one per repeat, over the same
    instances. The gap closed is recomputed from the averaged totals."""
    table = pd.concat([s["table"] for s in summaries]).groupby(level=0, sort=False).mean()
    starts = dict.fromkeys([f"always {baseline}", f"always {best}"])
    return {
        "instances": summaries[0]["instances"],
        "table": table,
        "picks": pd.concat([s["picks"] for s in summaries], axis=1).fillna(0).mean(axis=1),
        "fell_back": sum(s["fell_back"] for s in summaries) / len(summaries),
        "picks_fastest": sum(s["picks_fastest"] for s in summaries) / len(summaries),
        "gap_closed": {"router": {s: gap_closed(table, s) for s in starts}},
    }


def print_tuning(table):
    default = tuple(DEFAULT_SETTINGS[k] for k in KEY)
    show = table.head(10)
    for extra in (BEFORE_TUNING, default):
        if extra not in show.index:
            show = pd.concat([show, table.loc[[extra]]])
    print(f"\nout-of-fold, averaged over repeats; always the fallback: "
          f"{table['always_fallback_s'].iloc[0]:.1f} s, oracle: {table['oracle_s'].iloc[0]:.1f} s")
    print(f"{'leaf':>5} {'depth':>5} {'margin':>6} {'total s':>9} {'+/- sd':>7} {'timeouts':>9} "
          f"{'fell back':>9} {'gap closed':>10}")
    for setting, r in show.iterrows():
        leaf, depth, margin = setting
        mark = ("   (before tuning)" if setting == BEFORE_TUNING else "") + \
               ("   (train_router's default)" if setting == default else "")
        print(f"{leaf:5d} {depth:5d} {margin:6.1f} {r.total_s:9.1f} {r.total_sd:7.1f} "
              f"{r.timeouts:9.1f} {r.fell_back:9.1%} {r.gap_closed:10.1%}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    add_timeout_arg(ap)
    ap.add_argument("--timeout-cost", type=float,
                    help="ms that a timed-out run counts as (default: the timeout)")
    ap.add_argument("--trees", type=int, default=50, help="trees per forest")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--baseline", default="simple", help="encoding that speedups are measured against")
    ap.add_argument("--output-dir", default=str(CROSS_VALIDATION))
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)
    timeout_cost = timeout if args.timeout_cost is None else args.timeout_cost / 1000
    if timeout_cost < timeout:
        ap.error(f"--timeout-cost must be at least the timeout ({timeout * 1000:g} ms)")

    joined = pd.read_csv(args.joined)
    joined = joined[joined["encoding"] != ROUTER]
    costs, dropped = training_set(joined, timeout_cost)
    feats = instance_features(joined, feature_columns(joined), costs.index)
    corpus = joined.groupby("id")["corpus"].first().loc[costs.index]
    encodings = sorted(costs.columns)
    if args.baseline not in encodings:
        ap.error(f"--baseline {args.baseline!r} is not one of {encodings}")
    best = costs.sum().idxmin()

    settings = [(leaf, depth) for leaf in LEAF_SIZES for depth in DEPTHS]
    print(f"{len(costs)} instances (dropped {dropped['errored_or_missing']} errored or missing "
          f"and {dropped['all_timeout']} all-timeout): "
          + ", ".join(f"{c} {n}" for c, n in corpus.value_counts().sort_index().items())
          + f"; timeouts count as {timeout_cost:g} s")
    print(f"{len(settings)} settings x {len(MARGINS)} margins, {args.folds}-fold cross-validation "
          f"stratified by corpus, repeated {args.repeats} times ...")
    oof = out_of_fold_picks(costs, feats, corpus, settings, MARGINS, args.folds, args.repeats,
                            args.trees)
    table = score(oof, costs, timeout_cost)
    print_tuning(table)

    # ---- the best setting: its tables, overall and per corpus ------------------
    leaf, depth, margin = table.index[0]
    chosen = oof[(oof["min_samples_leaf"] == leaf) & (oof["max_depth"] == depth)
                 & (oof["margin"] == margin)].drop(columns="trained_on")
    overall, by_corpus, logs = [], {c: [] for c in sorted(corpus.unique())}, []
    for repeat, g in chosen.groupby("repeat"):
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

    fallbacks = sorted(chosen["fallback"].unique())
    print(f"\n===== best setting: min_samples_leaf {leaf}, max_depth {depth}, margin {margin:g}; "
          f"out-of-fold, averaged over {args.repeats} repeats =====")
    print(f"(chosen by the same cross-validation, so slightly optimistic)")
    print_summary("all corpora", overall, args.baseline, encodings, "/".join(fallbacks), margin)
    for c, s in by_corpus.items():
        print_summary(c, s, args.baseline, encodings, "/".join(fallbacks), margin)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "router_tuning.csv", float_format="%.6g")
    pd.concat(logs).to_csv(out / "router_eval.csv", index_label="id", float_format="%.6g")
    summary = {
        "joined": os.path.relpath(args.joined, REPO),
        "evaluated_on": f"every instance, out of fold: {args.folds}-fold cross-validation "
                        f"stratified by corpus, repeated {args.repeats} times; the setting "
                        f"is chosen by the same cross-validation",
        "instances": len(costs),
        "dropped": dropped,
        "timeout_cost_s": timeout_cost,
        "folds": args.folds,
        "repeats": args.repeats,
        "trees": args.trees,
        "setting": {"min_samples_leaf": int(leaf), "max_depth": int(depth), "margin": float(margin)},
        "fallbacks": fallbacks,
        "baseline": args.baseline,
        "best_performing_encoding": best,
        "overall": summary_json(overall, encodings),
        "by_corpus": {c: summary_json(s, encodings) for c, s in by_corpus.items()},
    }
    (out / "router_eval.json").write_text(json.dumps(summary, indent=1))

    setting = {"min_samples_leaf": leaf, "max_depth": depth, "margin": margin}
    if setting != DEFAULT_SETTINGS:
        print(f"\nNOTE: train_router.py's defaults are {DEFAULT_SETTINGS}; the best setting is "
              f"{setting}. Update DEFAULT_SETTINGS there.")
    print(f"\nWrote {out / 'router_tuning.csv'}, {out / 'router_eval.json'} and "
          f"{out / 'router_eval.csv'}")


if __name__ == "__main__":
    main()
