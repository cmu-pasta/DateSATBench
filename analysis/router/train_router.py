"""
Stage 7: train the encoding router, one random forest per pair of encodings.

    python -m analysis.router.train_router --timeout 20000     # or set DATESAT_TIMEOUT_MS

The router looks at a constraint's features and picks the one encoding that solves it.
For every pair of encodings (A, B) it has a forest that answers a single question: on
this constraint, which of A and B is faster? This script trains those forests from
joined.csv and saves them with the rule that routes by them (route() below applies it,
and DateSat's router does the same): the encoding that wins the most pairs is picked,
and the router uses --fallback (default: the best performing encoding, the one with the
lowest total time on the training data) instead on a tie, or when the pick does not beat
the fallback in their own forest with probability above 0.5 + --margin.

Rows of joined.csv whose encoding is `router` are results of the router itself, not
training data, and are ignored.

Costs. An instance's cost on an encoding is the median time over its runs, with a
timed-out run counting at --timeout-cost (default: the timeout). Instances are dropped
when
    - any encoding has an errored or missing run on them, or
    - every encoding timed out on them, so no encoding is faster than another.
Instances on which only some encodings time out are kept: they are the examples that
teach the router which choices are dangerous.

Overfitting. With leaves allowed to rest on one instance, the forests memorize the
training constraints. --min-samples-leaf keeps every leaf on several. Its default, with
the depth, margin and tree defaults (DEFAULT_SETTINGS, DEFAULT_TREES), is the router's
setting; crossval_router.py measures it by cross-validation, and does not change it.

Examples. For the A-vs-B forest, every instance on which A and B cost different amounts
is one example. The input is its features (every feature column, but not corpus, so the
router cannot learn which generator made a constraint), the label is the faster of the
two, and the weight is the gap |cost_A - cost_B| in seconds. A near-tie barely counts; a
timeout avoided counts a lot.

Writes to analysis/outputs/model/:
    router.joblib      the forests, keyed by (A, B), plus everything in router_meta.json
    router_meta.json   encodings, feature column order, whether the features were
                       extracted with the injected bounds kept, timeout cost, instance
                       counts, total cost per encoding and the best performing one (it
                       depends on the results), the fallback and margin, and per pair
                       the example counts and the top feature importances
    router.json        the model for DateSat's router approach, in plain JSON: the
                       settings above that DateSat needs, and every tree as parallel node
                       arrays (feature, threshold, left, right, and p_a, the probability
                       at a leaf that the pair's first encoding is faster). At a node, go
                       left if x[feature] <= threshold, with the features rounded to 32-bit
                       floats first, as scikit-learn does. Leaves have left == -1. A
                       forest's probability is the mean of its trees' leaf p_a. Before
                       writing it,
                       the trees are walked on every training instance and checked
                       against scikit-learn's predict_proba.
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from analysis.paths import MODEL, OUTPUTS, REPO
from analysis.stats.join_results import FINISHED, add_timeout_arg, timeout_seconds

RANDOM_STATE = 0
TOP_FEATURES = 5
ROUTER = "router"                  # the encoding name the router's own results carry
FORMAT_VERSION = 1                 # of router.json
EXPORT_TOLERANCE = 1e-9
# The router's setting. crossval_router.py measures it; ROUTER.md says why it was chosen.
DEFAULT_SETTINGS = {"min_samples_leaf": 10, "max_depth": 16, "margin": 0.0}
DEFAULT_TREES = 50                 # trees per forest

# The columns join_results.py writes ahead of the feature columns.
RESULT_COLS = {"id", "corpus", "encoding", "run", "time", "status", "solved",
               "baseline_time", "baseline_status", "speedup", "speedup_bound"}


def feature_columns(joined):
    """The router's inputs: the feature columns of joined.csv, in order."""
    return [c for c in joined.columns if c not in RESULT_COLS and not c.startswith("label_")]


def cost_table(joined, timeout_cost):
    """Cost in seconds of each encoding on each instance: one row per id, one column per
    encoding. A timed-out run counts at `timeout_cost`, and runs are combined by their
    median. A cell is NaN when any of its runs errored or it has no runs."""
    finished = joined["status"].isin(FINISHED)
    timed_out = joined["status"] == "timeout"
    cost = joined["time"].where(finished, timeout_cost).where(finished | timed_out)
    runs = joined.assign(cost=cost).groupby(["id", "encoding"])["cost"]
    return runs.median().where(runs.count() == runs.size()).unstack("encoding")


def all_timeout_ids(joined):
    """Instances on which every encoding timed out in every run."""
    timed_out = joined["status"].eq("timeout").groupby(joined["id"]).all()
    return set(timed_out.index[timed_out])


def training_set(joined, timeout_cost):
    """The cost table the router trains on, and how many instances were dropped.

    Drops instances with an errored or missing run on any encoding, then those on which
    every encoding timed out. Returns (costs, dropped), where dropped counts each kind:
    {"errored_or_missing": n, "all_timeout": n}.
    """
    costs = cost_table(joined, timeout_cost)
    incomplete = costs.isna().any(axis=1)
    costs = costs[~incomplete]
    all_timeout = costs.index.isin(all_timeout_ids(joined))
    return costs[~all_timeout], {"errored_or_missing": int(incomplete.sum()),
                                 "all_timeout": int(all_timeout.sum())}


def instance_features(joined, cols, ids):
    """One row of feature columns `cols` per instance in `ids`, in that order."""
    return joined.groupby("id")[cols].first().loc[ids]


def pair_examples(costs, a, b):
    """Training examples for the a-vs-b forest from a cost table with no NaNs.

    Returns (ids, label, weight): label is the name of the faster encoding and weight
    the gap between the two in seconds. Instances where both cost the same are left out.
    """
    gap = costs[b] - costs[a]                  # > 0 where a is faster
    gap = gap[gap != 0]
    label = pd.Series(np.where(gap > 0, a, b), index=gap.index)
    return gap.index, label, gap.abs()


def fit_forests(costs, feats, trees, max_depth, min_samples_leaf, n_jobs=None):
    """One forest per pair of encodings (A, B), keyed by (A, B), trained on the pair's
    examples from a cost table with no NaNs and the instances' features. No leaf rests on
    fewer than `min_samples_leaf` instances."""
    forests = {}
    for a, b in combinations(sorted(costs.columns), 2):
        ids, label, weight = pair_examples(costs, a, b)
        forest = RandomForestClassifier(n_estimators=trees, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        random_state=RANDOM_STATE, n_jobs=n_jobs)
        forest.fit(feats.loc[ids].to_numpy(), label.to_numpy(), sample_weight=weight.to_numpy())
        forests[(a, b)] = forest
    return forests


def tree_arrays(tree, k):
    """A fitted tree as the parallel node arrays router.json stores. `k` is the column of
    tree_.value that holds the pair's first encoding, or None if the forest never saw it
    win (p_a is then 0 everywhere)."""
    t = tree.tree_
    value = t.value[:, 0, :]
    p_a = np.zeros(t.node_count) if k is None else value[:, k] / value.sum(axis=1)
    leaf = t.children_left == -1
    return {
        "feature": np.where(leaf, -1, t.feature).tolist(),
        "threshold": np.where(leaf, 0.0, t.threshold).tolist(),
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "p_a": np.where(leaf, p_a, 0.0).tolist(),
    }


def forest_arrays(forest, a):
    """Every tree of a pairwise forest as tree_arrays, for its first encoding `a`."""
    classes = list(forest.classes_)
    k = classes.index(a) if a in classes else None
    return [tree_arrays(tree, k) for tree in forest.estimators_]


def walk_forest(trees, x):
    """P(first encoding is faster) from a forest in router.json form, for one feature
    vector `x` already rounded to 32-bit floats: scikit-learn compares float32 features
    with the thresholds, and a feature such as 2/15 lands on the other side of a threshold
    otherwise. DateSat's router walks the trees the same way."""
    total = 0.0
    for t in trees:
        node = 0
        while t["left"][node] != -1:
            go_left = x[t["feature"][node]] <= t["threshold"][node]
            node = t["left"][node] if go_left else t["right"][node]
        total += t["p_a"][node]
    return total / len(trees)


def export_error(forest, trees, a, X):
    """The largest difference, over the rows of X, between walking `trees` and the
    forest's own predict_proba for encoding `a`."""
    classes = list(forest.classes_)
    expected = (forest.predict_proba(X)[:, classes.index(a)] if a in classes
                else np.zeros(len(X)))
    X32 = X.astype(np.float32).astype(np.float64)
    walked = np.array([walk_forest(trees, x) for x in X32.tolist()])
    return float(np.abs(walked - expected).max())


def pair_probability(forest, X, a):
    """P(a is the faster encoding) from a pairwise forest, for every row of X."""
    classes = list(forest.classes_)
    if a not in classes:                       # the forest only ever saw the other side win
        return np.zeros(len(X))
    return forest.predict_proba(X)[:, classes.index(a)]


def vote(beats, encodings, fallback, margin=0.0):
    """Pick an encoding for one instance from its pairwise probabilities, `beats[(a, b)]` =
    P(a is faster than b) for every pair of forests.

    Returns (pick, fell_back, wins): the picked encoding, whether the fallback overrode the
    vote, and each encoding's number of pairwise wins.
    """
    wins = dict.fromkeys(encodings, 0)
    beats = dict(beats)
    for (a, b), p in list(beats.items()):
        beats[(b, a)] = 1 - p
        if p > 0.5:
            wins[a] += 1
        elif p < 0.5:
            wins[b] += 1

    top = max(wins.values())
    leaders = [e for e in encodings if wins[e] == top]
    pick = leaders[0]
    fell_back = len(leaders) > 1 or (pick != fallback and beats[(pick, fallback)] <= 0.5 + margin)
    return (fallback if fell_back else pick), fell_back, wins


def pair_probabilities(forests, feats):
    """P(a is faster than b) from every pairwise forest (a, b), for every row of `feats`."""
    X = feats.to_numpy()
    return {(a, b): pair_probability(forest, X, a) for (a, b), forest in forests.items()}


def route(forests, encodings, feats, fallback, margin=0.0, probs=None):
    """vote for every instance (row) of `feats`. `probs`, from pair_probabilities, saves
    running the forests again when only the fallback or margin change.

    Returns (picks, fell_back, wins), all indexed like `feats`.
    """
    if probs is None:
        probs = pair_probabilities(forests, feats)
    results = [vote({pair: float(p[i]) for pair, p in probs.items()}, encodings, fallback, margin)
               for i in range(len(feats))]
    picks = pd.Series([r[0] for r in results], index=feats.index)
    fell_back = pd.Series([r[1] for r in results], index=feats.index)
    wins = pd.DataFrame([r[2] for r in results], index=feats.index, columns=encodings)
    return picks, fell_back, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--features-meta", default=str(OUTPUTS / "features_meta.json"),
                    help="records whether the features kept the injected bounds")
    add_timeout_arg(ap)
    ap.add_argument("--timeout-cost", type=float,
                    help="ms that a timed-out run counts as (default: the timeout)")
    ap.add_argument("--trees", type=int, default=DEFAULT_TREES, help="trees per forest")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_SETTINGS["max_depth"],
                    help="maximum depth of each tree")
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_SETTINGS["min_samples_leaf"],
                    help="fewest training instances a leaf may rest on")
    ap.add_argument("--fallback", help="encoding the router uses when the vote is close "
                                       "(default: the best performing encoding)")
    ap.add_argument("--margin", type=float, default=DEFAULT_SETTINGS["margin"],
                    help="how far above 0.5 a pick must beat the fallback to be kept")
    ap.add_argument("--output-dir", default=str(MODEL))
    args = ap.parse_args()
    timeout = timeout_seconds(ap, args)
    timeout_cost = timeout if args.timeout_cost is None else args.timeout_cost / 1000
    if timeout_cost < timeout:
        ap.error(f"--timeout-cost must be at least the timeout ({timeout * 1000:g} ms)")
    if not 0 <= args.margin < 0.5:
        ap.error(f"--margin must be in [0, 0.5), got {args.margin:g}")
    bounds = json.loads(Path(args.features_meta).read_text())["bounds"]

    joined = pd.read_csv(args.joined)
    n_router = int((joined["encoding"] == ROUTER).sum())
    joined = joined[joined["encoding"] != ROUTER]
    cols = feature_columns(joined)
    n_total = joined["id"].nunique()
    costs, dropped = training_set(joined, timeout_cost)
    feats = instance_features(joined, cols, costs.index)
    encodings = sorted(costs.columns)

    if n_router:
        print(f"ignored {n_router} rows of the router's own results")
    print(f"encodings: {encodings}")
    print(f"features: {len(cols)} columns, extracted with the injected bounds {bounds}")
    print(f"timeout: {timeout:g} s, counted as {timeout_cost:g} s")
    print(f"instances: {n_total}; dropped {dropped['errored_or_missing']} with an errored or "
          f"missing run and {dropped['all_timeout']} on which every encoding timed out; "
          f"training on {len(costs)}")

    pairs = {}
    print(f"\n{len(encodings) * (len(encodings) - 1) // 2} forests of {args.trees} trees, "
          f"depth <= {args.max_depth}, at least {args.min_samples_leaf} instances per leaf:")
    forests = fit_forests(costs, feats, args.trees, args.max_depth, args.min_samples_leaf)
    for (a, b), forest in forests.items():
        _, label, weight = pair_examples(costs, a, b)
        wins = {e: int((label == e).sum()) for e in (a, b)}
        gaps = {e: round(float(weight[label == e].sum()), 3) for e in (a, b)}
        order = np.argsort(-forest.feature_importances_)[:TOP_FEATURES]
        top = [{"feature": cols[i], "importance": round(float(forest.feature_importances_[i]), 4)}
               for i in order]
        pairs[f"{a} vs {b}"] = {"examples": len(label), "faster": wins,
                                "weight_s": gaps, "top_features": top}
        print(f"  {a} vs {b}: {len(label)} examples; {a} faster on {wins[a]} "
              f"({gaps[a]:.1f} s), {b} on {wins[b]} ({gaps[b]:.1f} s)")
        print("    top features: " + ", ".join(f"{t['feature']} {t['importance']:.2f}" for t in top))

    total = costs.sum()
    fallback = args.fallback or total.idxmin()
    if fallback not in encodings:
        ap.error(f"--fallback {fallback!r} is not one of {encodings}")

    print(f"\nchecking the exported trees against scikit-learn on {len(costs)} instances ...")
    X = feats.to_numpy(dtype=float)
    exported = []
    for (a, b), forest in forests.items():
        trees = forest_arrays(forest, a)
        err = export_error(forest, trees, a, X)
        if err > EXPORT_TOLERANCE:
            raise SystemExit(f"{a} vs {b}: the exported trees differ from scikit-learn by "
                             f"{err:g}; router.json not written")
        exported.append({"a": a, "b": b, "trees": trees})
    if bounds != "keep":
        print("WARNING: the features were extracted with the injected bounds stripped; "
              "DateSat's router refuses such a model (extract them with --bounds keep)")

    meta = {
        "joined": os.path.relpath(args.joined, REPO),
        "sklearn_version": sklearn.__version__,
        "timeout_s": timeout,
        "timeout_cost_s": timeout_cost,
        "trees": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": RANDOM_STATE,
        "instances": {"total": n_total,
                      "dropped_errored_or_missing": dropped["errored_or_missing"],
                      "dropped_all_timeout": dropped["all_timeout"],
                      "trained_on": len(costs)},
        "encodings": encodings,
        "feature_columns": cols,
        "bounds": bounds,
        "total_cost_s": {e: round(float(total[e]), 3) for e in encodings},
        "best_single_encoding": total.idxmin(),
        "fallback": fallback,
        "margin": args.margin,
        "pairs": pairs,
    }
    router_json = {
        "format_version": FORMAT_VERSION,
        "bounds": bounds,
        "encodings": encodings,
        "features": cols,
        "fallback": fallback,
        "margin": args.margin,
        "trained_on": {"joined": meta["joined"], "instances": len(costs), "timeout_s": timeout},
        "pairs": exported,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(meta, forests=forests), out / "router.joblib")
    (out / "router_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "router.json").write_text(json.dumps(router_json, separators=(",", ":")))
    print(f"best single encoding on the training data: {meta['best_single_encoding']}; "
          f"fallback {fallback}, margin {args.margin:g}")
    print(f"Wrote {out / 'router.joblib'}, {out / 'router_meta.json'} and {out / 'router.json'}")


if __name__ == "__main__":
    main()
