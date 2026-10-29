"""
Stage 6: measure the router on constraints it has not seen, by cross-validation over every
instance.

    python -m analysis.router.crossval_router --timeout 20000     # or set DATESAT_TIMEOUT_MS

It measures one fixed setting: by default the one train_router.py trains (its
DEFAULT_SETTINGS and DEFAULT_TREES: leaf size, depth, features per split, margin, what a
timeout counts as in training, and trees per forest), so the figures describe the model
DateSat uses. It does not search for a better setting; --min-samples-leaf, --max-depth,
--max-features, --margin, --train-timeout-cost and --trees measure another one. Scores always count a timeout at
--timeout-cost (default: the timeout), whatever it counts as in training.

Features. By default the router sees every feature column. With --selected it sees only
the features select_features.py kept (outputs/selected_features.json), and the results go
to model_cross_validation_selected/ instead, so the two can be compared.

How. The instances of joined.csv, kept as in train_router.py (each one a single
example per pair, its cost the median over its runs), are cut into --folds folds (default
5), stratified by corpus: each corpus is shuffled and dealt evenly into the folds, so
every fold holds the same share of every corpus. For every fold, the router is trained on
the other folds and routes the fold, so every instance is routed once by a router that
has not seen it; the fallback is the best performing encoding on the other folds. The
whole is repeated --repeats times (default 10, one per run of the live evaluation below),
each time with a different cut into folds. The cuts come from a fixed seed and the forests
use train_router.py's, so a run of this script always gives the same results.

Predicted and measured. Every fold's model is written for DateSat's router approach
(fold_models/), and analysis/run/run_router_cv.py runs DateSat's router with them: its run
N solves every instance with the model of repeat N - 1 that was not trained on it. Until
joined.csv holds those results (encoding `router`), the router is *predicted*: on every
instance it is charged the recorded cost of the encoding it picks (the median over that
encoding's runs), which leaves out the time routing takes, and the results are averaged
over the repeats. Once joined.csv holds them, the router's row is *measured*: every live
run is summarized as its repeat's prediction is, with the router costing on each instance
its time in that run, routing included, and the summaries are averaged over the live runs,
as the predicted ones are over the repeats. (Each run routes with other models, so a median
over the runs of an instance would mix the models' picks.) Its picks and fallbacks are the
predicted ones of the repeats the live runs used, and the `live` block says how far the live
runs agree with them: the share of live picks that are the predicted pick (crossval stops
if any is not: the live runs used other models), the time routing took, and the live time
less the routing against the recorded cost of the encoding picked, which shows whether the
machine ran as fast as when the encodings were recorded. Every encoding's figures and the oracle's always come from the
encodings' recorded runs. With --selected the router is always predicted: the live runs
used every feature.

Writes to analysis/outputs/model_cross_validation/ (with --selected,
model_cross_validation_selected/):
    router_eval.json   the setting, the feature columns, whether the router is measured,
                       and, overall and for each corpus, averaged over the repeats (or the
                       repeats the live runs used): a table of always each encoding, the
                       router and the
                       oracle (total time, timeouts, and speedups over always
                       --baseline: of the totals, and the median and geometric mean of
                       the per-instance speedup; the timeouts are counted in every run
                       and averaged over the runs, as in solver_outcomes.py, the
                       oracle taking the fastest encoding in each run), the router's
                       picks and fallbacks, how often it picks
                       the fastest encoding, how much of the gap to the oracle it closes,
                       and the spread of the router's total over the repeats (measured:
                       over the live runs); the same with the router predicted, averaged
                       over every repeat, under `predicted`; and, measured, the `live` block,
                       with the predicted totals of the repeats the live runs used
    router_eval.csv    one row per repeat and instance: its fold and corpus, its cost on
                       every encoding, the best encoding ("none" where every encoding
                       costs the same) and its cost, the router's pick,
                       whether it fell back, the fold's fallback and the router's predicted
                       cost, and, measured, the time and pick of the live run that used
                       the repeat
    fold_models.json   the setting and, for every repeat, every fold's model file (with its
                       SHA-256 and fallback), the fold of every instance and the encoding
                       its fold's model picks for it; run_router_cv.py and
                       check_datesat_router.py read it
    fold_models/       repeat_<r>_fold_<k>.json, the model trained on every fold of repeat
                       r but k, in train_router.py's router.json form; ~0.6 MB each,
                       committed with the rest
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from analysis.paths import CROSS_VALIDATION, CROSS_VALIDATION_SELECTED, OUTPUTS, REPO
from analysis.router.train_router import (DEFAULT_SETTINGS, DEFAULT_TREES, feature_columns,
                                          fit_forests, instance_features, max_features_arg,
                                          route, router_json, training_set)
from analysis.stats.instance_costs import ROUTER, encodings_only, run_costs
from analysis.stats.join_results import FINISHED, add_timeout_arg, timeout_seconds
from analysis.stats.select_features import selected_columns

SEED = 0
DEFAULT_REPEATS = 10           # one cut into folds per run of run_router_cv.py
DRIFT_TOLERANCE = 0.05         # how far from 1 the live runs' speed may drift before a warning
FOLD_MODELS = "fold_models"    # the fold models' directory; with .json, their manifest


def strategy_costs(costs, picks):
    """Per-instance cost of every strategy: always-<encoding>, the router (the cost of the
    encoding it picks), and the oracle."""
    out = {f"always {e}": costs[e] for e in costs.columns}
    out["router"] = pd.Series([costs.at[iid, picks[iid]] for iid in costs.index], index=costs.index)
    out["oracle"] = costs.min(axis=1)
    return pd.DataFrame(out)


def strategy_run_costs(runs, picks):
    """strategy_costs in every run: one row per (run, instance) of `runs` (run_costs), each
    strategy's time in that run. The router runs the encoding it picks for the instance, and
    the oracle takes the fastest encoding in that run."""
    pick = runs.columns.get_indexer(picks.loc[runs.index.get_level_values("id")])
    out = {f"always {e}": runs[e] for e in runs.columns}
    out["router"] = pd.Series(runs.to_numpy()[np.arange(len(runs)), pick], index=runs.index)
    out["oracle"] = runs.min(axis=1)
    return pd.DataFrame(out)


def measured_strategy_costs(costs, router):
    """strategy_costs with the router measured: its cost on every instance is `router`'s,
    the median over its live runs."""
    out = {f"always {e}": costs[e] for e in costs.columns}
    out["router"] = router.loc[costs.index]
    out["oracle"] = costs.min(axis=1)
    return pd.DataFrame(out)


def measured_run_costs(runs, router_runs):
    """strategy_run_costs with the router measured: its time in each of its live runs,
    `router_runs`, one value per (run, id). A strategy has no value in a run it was not run
    in."""
    out = {f"always {e}": runs[e] for e in runs.columns}
    out["router"] = router_runs
    out["oracle"] = runs.min(axis=1)
    return pd.DataFrame(out)


def summary_table(strategies, run_strategies, baseline, timeout_cost):
    """Total time, timeouts and speedup over always-`baseline`, one row per strategy of
    strategy_costs: of the totals, and the median and geometric mean over instances of the
    per-instance speedup (as in solver_outcomes.py, timeouts at the timeout). The timeouts
    are counted in every run of a strategy, from strategy_run_costs, and averaged over its
    runs."""
    base = strategies[f"always {baseline}"]
    per_instance = strategies.rdiv(base, axis=0)
    return pd.DataFrame({
        "total_s": strategies.sum(),
        "timeouts": pd.Series({name: (r.dropna() >= timeout_cost).groupby(level="run").sum().mean()
                               for name, r in run_strategies.items()}),
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


def summarize(strategies, run_strategies, picks, fell_back, baseline, best, timeout_cost):
    """The summary of one set of instances: summary_table, the router's picks and
    fallbacks, how often it picks the fastest encoding, and how much of the gap to the
    oracle it closes, from always `baseline` and from always `best`."""
    table = summary_table(strategies, run_strategies, baseline, timeout_cost)
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
                      repeats, trees, train_costs=None,
                      max_features=DEFAULT_SETTINGS["max_features"], on_fold=None):
    """Route every instance of `costs` with routers trained on the other folds, with one
    setting. The forests learn from `train_costs` (default: `costs`), the same instances
    with timeouts counted at the training cost; the fallback comes from `costs`. For every
    fold, on_fold(repeat, fold, forests, fallback, train_ids, test_ids) is called, if given,
    with the forests that route it.

    Returns one row per repeat and instance: the fold, the pick, whether it fell back, the
    fold's fallback, and the ids the router was trained on.
    """
    if train_costs is None:
        train_costs = costs
    encodings = sorted(costs.columns)
    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    rows = []
    for k, (train, test) in enumerate(folds.split(costs, strata.loc[costs.index])):
        repeat, fold = divmod(k, n_splits)
        c_train, c_test = costs.iloc[train], costs.iloc[test]
        fallback = c_train.sum().idxmin()
        forests = fit_forests(train_costs.loc[c_train.index], feats.loc[c_train.index], trees,
                              max_depth, min_samples_leaf, max_features, n_jobs=-1)
        if on_fold is not None:
            on_fold(repeat, fold, forests, fallback, c_train.index, c_test.index)
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


def run_number(label):
    """N, from a run's label run_N."""
    m = re.fullmatch(r"run_(\d+)", str(label))
    if not m:
        raise ValueError(f"the router's results have a run {label!r}, not run_N")
    return int(m.group(1))


def live_router(live, ids, timeout_cost, repeats):
    """The router's live results on the instances `ids`: its time in every run (one value
    per (run, id), a timeout at `timeout_cost`) and the repeat each run used (run_N used
    repeat N - 1's models). Raises ValueError unless every instance has a result without an
    error in every run, and every run has a repeat."""
    repeat_of = {r: run_number(r) - 1 for r in sorted(live["run"].unique(), key=run_number)}
    if max(repeat_of.values()) >= repeats:
        raise ValueError(f"the router's results have {max(repeat_of, key=run_number)}, but the "
                         f"cross-validation has only {repeats} repeats")
    per_run = run_costs(live, timeout_cost, ids)[ROUTER].dropna()
    expected = len(ids) * len(repeat_of)
    if len(per_run) != expected:
        raise ValueError(f"the router's results miss, or have an error on, "
                         f"{expected - len(per_run)} of the {expected} runs of the "
                         f"{len(ids)} instances")
    return per_run, repeat_of


def measured_summaries(per_repeat, costs, runs, router_runs, repeat_of, baseline, timeout_cost):
    """One summary per live run, over the instances of `costs`: the summary of the repeat
    the run used (`per_repeat`, one per repeat), with the table's router row taken from the
    run's times (`router_runs`, one value per (run, id)). average_summaries of them is the
    measured router, as average_summaries of the repeats' is the predicted one."""
    out = []
    for run, repeat in repeat_of.items():
        times = router_runs.xs(run, level="run", drop_level=False)
        table = summary_table(measured_strategy_costs(costs, times.droplevel("run")),
                              measured_run_costs(runs, times), baseline, timeout_cost)
        out.append(dict(per_repeat[repeat], table=table))
    return out


def live_agreement(live, ids, oof, repeat_of, costs, timeout_cost):
    """How the router's live runs on the instances `ids` compare with the prediction
    (`oof`, out_of_fold_picks' rows): how many live picks are known (a run killed at the hard
    timeout does not report one) and how many are the predicted pick of their run's repeat;
    the time routing took per run of an instance, and its share of the router's time; and
    the median, over the runs in which the router finished and the encoding it picked has a
    recorded cost under the timeout, of the live time less the routing over that recorded
    cost: about 1 when the machine runs as fast as when the encodings were recorded."""
    rows = live[live["id"].isin(ids)]
    predicted = oof.set_index(["repeat", "id"])["pick"]
    picked = rows[rows["routed_to"].notna()]
    expected = predicted.loc[list(zip(picked["run"].map(repeat_of), picked["id"]))].to_numpy()
    routing = picked["routing_time"].astype(float)
    done = picked[picked["status"].isin(FINISHED) & (picked["time"] < timeout_cost)]
    recorded = np.array([costs.at[i, e] for i, e in zip(done["id"], done["routed_to"])])
    under = recorded < timeout_cost
    ratio = (done["time"] - done["routing_time"].astype(float)).to_numpy()[under] / recorded[under]
    return {
        "runs": len(repeat_of),
        "repeats_used": sorted(repeat_of.values()),
        "results": len(rows),
        "picks_known": len(picked),
        "picks_agree": int((picked["routed_to"].to_numpy() == expected).sum()),
        "routing_time_s": {"median": round(float(routing.median()), 6),
                           "mean": round(float(routing.mean()), 6),
                           "max": round(float(routing.max()), 6)},
        "routing_share": round(float(routing.sum() / picked["time"].sum()), 6),
        "time_vs_recorded_median": round(float(np.median(ratio)), 4) if len(ratio) else None,
        "time_vs_recorded_n": len(ratio),
    }


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    add_timeout_arg(ap)
    ap.add_argument("--timeout-cost", type=float,
                    help="ms that a timed-out run counts as (default: the timeout)")
    ap.add_argument("--train-timeout-cost", type=float,
                    default=DEFAULT_SETTINGS["train_timeout_cost_s"] * 1000,
                    help="ms that a timed-out run counts as in the training examples")
    ap.add_argument("--trees", type=int, default=DEFAULT_TREES, help="trees per forest")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_SETTINGS["max_depth"],
                    help="maximum depth of each tree")
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_SETTINGS["min_samples_leaf"],
                    help="fewest training instances a leaf may rest on")
    ap.add_argument("--max-features", type=max_features_arg,
                    default=DEFAULT_SETTINGS["max_features"],
                    help="share of the features each split chooses among, or sqrt or log2")
    ap.add_argument("--margin", type=float, default=DEFAULT_SETTINGS["margin"],
                    help="how far above 0.5 a pick must beat the fallback to be kept")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help="cuts into folds, one per run of run_router_cv.py")
    ap.add_argument("--features-meta", default=str(OUTPUTS / "features_meta.json"),
                    help="records whether the features kept the injected bounds")
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
    train_timeout_cost = args.train_timeout_cost / 1000
    if timeout_cost < timeout:
        ap.error(f"--timeout-cost must be at least the timeout ({timeout * 1000:g} ms)")
    if train_timeout_cost < timeout:
        ap.error(f"--train-timeout-cost must be at least the timeout ({timeout * 1000:g} ms)")
    if not 0 <= args.margin < 0.5:
        ap.error(f"--margin must be in [0, 0.5), got {args.margin:g}")
    leaf, depth, margin = args.min_samples_leaf, args.max_depth, args.margin
    max_features = args.max_features

    joined = pd.read_csv(args.joined)
    live = joined[joined["encoding"] == ROUTER]
    joined = encodings_only(joined)
    if args.selected and len(live):
        print("the router's live results are left out: they used every feature")
        live = live.iloc[:0]
    bounds = json.loads(Path(args.features_meta).read_text())["bounds"]
    costs, dropped, all_timeout = training_set(joined, timeout_cost)
    train_costs, _, _ = training_set(joined, train_timeout_cost)    # the same instances
    runs = run_costs(joined, timeout_cost, costs.index)[costs.columns]   # for the timeouts
    run_ids = runs.index.get_level_values("id")
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

    print(f"{len(costs)} instances (dropped {dropped['errored_or_missing']} errored or missing; "
          f"kept {all_timeout} on which every encoding timed out): "
          + ", ".join(f"{c} {n}" for c, n in corpus.value_counts().sort_index().items())
          + f"; timeouts count as {timeout_cost:g} s")
    print(f"features: {len(cols)} columns" + (" (selected_features.json)" if args.selected else ""))
    print(f"setting: {args.trees} trees, depth <= {depth}, at least {leaf} instances per leaf, "
          f"{max_features} of the features per split, margin {margin:g}, timeouts count as {train_timeout_cost:g} s in training; "
          f"{args.folds}-fold cross-validation stratified by corpus, "
          f"repeated {args.repeats} times ...")
    measured = len(live) > 0
    if measured:
        try:
            live_runs, repeat_of = live_router(live, costs.index, timeout_cost, args.repeats)
        except ValueError as e:
            raise SystemExit(f"{e}; rerun run_router_cv.py, or join without its results")
        run_of = {r: run for run, r in repeat_of.items()}
        live_picks = live.set_index(["run", "id"])["routed_to"]
    out = Path(args.output_dir)
    models_dir = out / FOLD_MODELS
    joined_rel = os.path.relpath(args.joined, REPO)
    fold_models, fold_texts = {}, {}         # written only once every check has passed

    def save_fold_model(repeat, fold, forests, fallback, train_ids, test_ids):
        """Write the fold's forests for DateSat, checked on the instances they route."""
        try:
            model = router_json(forests, encodings, cols, feats.loc[test_ids].to_numpy(dtype=float),
                                fallback, margin, bounds,
                                {"joined": joined_rel, "instances": len(train_ids),
                                 "timeout_s": timeout, "repeat": repeat, "fold": fold})
        except ValueError as e:
            raise SystemExit(f"repeat {repeat}, fold {fold}: {e}; fold models not written")
        text = json.dumps(model, separators=(",", ":"))
        name = f"repeat_{repeat}_fold_{fold}.json"
        fold_texts[name] = text
        fold_models[(repeat, fold)] = {"model": f"{FOLD_MODELS}/{name}", "sha256": sha256(text),
                                       "fallback": fallback}

    oof = out_of_fold_picks(costs, feats, corpus, leaf, depth, margin, args.folds, args.repeats,
                            args.trees, train_costs, max_features,
                            on_fold=save_fold_model).drop(columns="trained_on")

    # the predicted router's summary of every repeat, overall and for each corpus
    per_repeat, per_repeat_by_corpus, logs = [], {c: [] for c in sorted(corpus.unique())}, []
    for repeat, g in oof.groupby("repeat"):
        g = g.set_index("id").loc[costs.index]
        strategies = strategy_costs(costs, g["pick"])
        run_strategies = strategy_run_costs(runs, g["pick"])
        per_repeat.append(summarize(strategies, run_strategies, g["pick"], g["fell_back"],
                                    args.baseline, best, timeout_cost))
        for c, summaries in per_repeat_by_corpus.items():
            m = corpus == c
            in_c = run_ids.isin(corpus.index[m])
            summaries.append(summarize(strategies[m], run_strategies[in_c], g["pick"][m],
                                       g["fell_back"][m], args.baseline, best, timeout_cost))
        log = pd.DataFrame({"repeat": repeat, "fold": g["fold"], "corpus": corpus})
        for e in encodings:
            log[f"cost_{e}"] = costs[e]
        log["best"] = costs.idxmin(axis=1).where(costs.nunique(axis=1) > 1, "none")   # no fastest where all tie
        log["best_cost"] = strategies["oracle"]
        log["router"] = g["pick"]
        log["fell_back"] = g["fell_back"]
        log["fallback"] = g["fallback"]
        log["router_cost"] = strategies["router"]
        if measured:
            run = run_of.get(repeat)
            in_run = live_runs.xs(run, level="run") if run else pd.Series(dtype=float)
            log["live_time"] = in_run.reindex(costs.index)
            log["live_routed_to"] = (live_picks.xs(run, level="run").reindex(costs.index)
                                     if run else None)
        logs.append(log)
    predicted = average_summaries(per_repeat, args.baseline, best)
    predicted_by_corpus = {c: average_summaries(s, args.baseline, best)
                           for c, s in per_repeat_by_corpus.items()}
    if measured:
        used = sorted(repeat_of.values())

        def predicted_total(summaries):
            """The predicted router's total, averaged over the repeats the live runs used."""
            return round(float(average_summaries([summaries[r] for r in used], args.baseline, best)
                               ["table"].at["router", "total_s"]), 3)

        live_ids = live_runs.index.get_level_values("id")

        def measured_scope(summaries, m):
            in_scope = corpus.index[m]
            return average_summaries(
                measured_summaries(summaries, costs[m], runs[run_ids.isin(in_scope)],
                                   live_runs[live_ids.isin(in_scope)], repeat_of,
                                   args.baseline, timeout_cost), args.baseline, best)

        overall = measured_scope(per_repeat, pd.Series(True, index=corpus.index))
        by_corpus = {c: measured_scope(s, corpus == c) for c, s in per_repeat_by_corpus.items()}
        agreement = live_agreement(live, costs.index, oof, repeat_of, costs, timeout_cost)
        agreement["predicted_total_s"] = predicted_total(per_repeat)
        agreement["predicted_total_by_corpus_s"] = {c: predicted_total(s)
                                                    for c, s in per_repeat_by_corpus.items()}
        if agreement["picks_agree"] != agreement["picks_known"]:
            raise SystemExit(f"{agreement['picks_known'] - agreement['picks_agree']} of the "
                             f"router's live picks differ from this cross-validation's: the live "
                             f"runs used other fold models than these; nothing written")
        drift = agreement["time_vs_recorded_median"]
        if drift is not None and abs(drift - 1) > DRIFT_TOLERANCE:
            print(f"WARNING: the router's live time less routing is {drift}x the recorded time of "
                  f"the encoding it picked (median): the live runs ran on a faster or slower "
                  f"machine than the encodings, so the measured router is not comparable with them")
    else:
        overall, by_corpus = predicted, predicted_by_corpus

    fallbacks = sorted(oof["fallback"].unique())
    if measured:
        print(f"\n===== the router measured on {len(used)} live runs (repeats "
              f"{', '.join(map(str, used))}); every other strategy as recorded =====")
        print(f"router total, averaged over the live runs: {overall['table'].at['router', 'total_s']:.1f} s "
              f"+/- {overall['router_total_sd']:.1f} s (standard deviation of a run's total); "
              f"predicted {agreement['predicted_total_s']:.1f} s")
        print(f"live picks known on {agreement['picks_known']} of {agreement['results']} runs, "
              f"{agreement['picks_agree']} of them the predicted pick; routing took "
              f"{1000 * agreement['routing_time_s']['median']:.2f} ms per instance (median), "
              f"{agreement['routing_share']:.2%} of the router's time; live time less routing over "
              f"the picked encoding's recorded time: {agreement['time_vs_recorded_median']} (median)")
    else:
        print(f"\n===== out-of-fold, averaged over {args.repeats} repeats (predicted) =====")
        print(f"router total over the repeats: {overall['table'].at['router', 'total_s']:.1f} s "
              f"+/- {overall['router_total_sd']:.1f} s (standard deviation)")
    print_summary("all corpora", overall, args.baseline, encodings, "/".join(fallbacks), margin)
    for c, s in by_corpus.items():
        print_summary(c, s, args.baseline, encodings, "/".join(fallbacks), margin)

    # Every check has passed: write the fold models, each replaced in one step, so a
    # run_router_cv.py running meanwhile never finds a model missing; then the rest.
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, text in fold_texts.items():
        tmp = models_dir / f"{name}.tmp"
        tmp.write_text(text)
        os.replace(tmp, models_dir / name)
    for stale in models_dir.glob("repeat_*_fold_*.json"):
        if stale.name not in fold_texts:
            stale.unlink()
    pd.concat(logs).to_csv(out / "router_eval.csv", index_label="id", float_format="%.6g")
    summary = {
        "joined": os.path.relpath(args.joined, REPO),
        "evaluated_on": f"every instance, out of fold: {args.folds}-fold cross-validation "
                        f"stratified by corpus, repeated {args.repeats} times"
                        + (f"; the router measured on {len(used)} live runs" if measured else ""),
        "router_measured": measured,
        "instances": len(costs),
        "dropped": dropped,
        "all_timeout": all_timeout,
        "timeout_cost_s": timeout_cost,
        "folds": args.folds,
        "repeats": args.repeats,
        "trees": args.trees,
        "setting": {"min_samples_leaf": leaf, "max_depth": depth, "max_features": max_features,
                    "margin": margin, "train_timeout_cost_s": train_timeout_cost},
        "feature_columns": cols,
        "fallbacks": fallbacks,
        "baseline": args.baseline,
        "best_performing_encoding": best,
        "overall": summary_json(overall, encodings),
        "by_corpus": {c: summary_json(s, encodings) for c, s in by_corpus.items()},
        "predicted": {"overall": summary_json(predicted, encodings),
                      "by_corpus": {c: summary_json(s, encodings)
                                    for c, s in predicted_by_corpus.items()}},
        "live": agreement if measured else None,
        "fold_models": f"{FOLD_MODELS}.json",
    }
    (out / "router_eval.json").write_text(json.dumps(summary, indent=1))
    by_repeat = {r: g.set_index("id") for r, g in oof.groupby("repeat")}
    manifest = {
        "joined": joined_rel,
        "seed": SEED,
        "folds": args.folds,
        "repeats": args.repeats,
        "trees": args.trees,
        "setting": summary["setting"],
        "features": cols,
        "bounds": bounds,
        "timeout_s": timeout,
        "models": [[fold_models[(r, k)] for k in range(args.folds)] for r in range(args.repeats)],
        "fold_of": [{i: int(k) for i, k in by_repeat[r]["fold"].items()} for r in range(args.repeats)],
        "pick": [by_repeat[r]["pick"].to_dict() for r in range(args.repeats)],
    }
    tmp = out / f"{FOLD_MODELS}.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, out / f"{FOLD_MODELS}.json")

    if summary["setting"] != DEFAULT_SETTINGS or args.trees != DEFAULT_TREES:
        print(f"\nNOTE: this is not the setting train_router.py trains by default "
              f"({DEFAULT_SETTINGS}, {DEFAULT_TREES} trees)")
    print(f"\nWrote {out / 'router_eval.json'}, {out / 'router_eval.csv'}, "
          f"{out / FOLD_MODELS}.json and {len(fold_models)} models in {models_dir}")


if __name__ == "__main__":
    main()
