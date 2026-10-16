import math

import numpy as np
import pandas as pd

from analysis.router.train_router import (all_timeout_ids, cost_table, feature_columns,
                                          fit_forests, pair_examples, route)

TIMEOUT = 20.0


def rows(iid, encoding, *runs):
    """joined.csv rows for one instance on one encoding; each run is (time, status)."""
    return [
        {"id": iid, "corpus": "grammar", "encoding": encoding, "run": f"run_{i}",
         "time": time, "status": status, "solved": int(status in ("sat", "unsat")),
         "baseline_time": None, "baseline_status": None, "speedup": None,
         "speedup_bound": None, "n_atoms": 3, "label_status": "sat"}
        for i, (time, status) in enumerate(runs, 1)
    ]


JOINED = pd.DataFrame(
    rows("fast", "a", (1.0, "sat"), (3.0, "sat"))
    + rows("fast", "b", (0.5, "sat"), (0.5, "sat"))
    + rows("one-sided", "a", (20.007, "timeout"), (20.007, "timeout"))
    + rows("one-sided", "b", (4.0, "unsat"), (4.0, "unsat"))
    + rows("all-timeout", "a", (20.007, "timeout"), (20.007, "timeout"))
    + rows("all-timeout", "b", (20.007, "timeout"), (20.007, "timeout"))
    + rows("tie", "a", (1.0, "sat"), (1.0, "sat"))
    + rows("tie", "b", (1.0, "sat"), (1.0, "sat"))
    + rows("error", "a", (0.1, "error"), (0.2, "sat"))
    + rows("error", "b", (0.3, "sat"), (0.3, "sat"))
)


def test_runs_combine_by_median_and_timeouts_cost_the_timeout():
    costs = cost_table(JOINED, TIMEOUT)
    assert costs.loc["fast", "a"] == 2.0
    assert costs.loc["one-sided", "a"] == TIMEOUT


def test_an_errored_run_leaves_no_cost():
    costs = cost_table(JOINED, TIMEOUT)
    assert math.isnan(costs.loc["error", "a"])
    assert costs.loc["error", "b"] == 0.3


def test_all_timeout_instances():
    assert all_timeout_ids(JOINED) == {"all-timeout"}


def test_pair_examples_keep_one_sided_timeouts_and_skip_ties():
    costs = cost_table(JOINED, TIMEOUT).drop(index=["all-timeout", "error"])
    ids, label, weight = pair_examples(costs, "a", "b")
    assert sorted(ids) == ["fast", "one-sided"]
    assert label["fast"] == "b" and label["one-sided"] == "b"
    assert weight["fast"] == 1.5
    assert weight["one-sided"] == TIMEOUT - 4.0


def test_features_leave_out_corpus_results_and_labels():
    assert feature_columns(JOINED) == ["n_atoms"]


def random_instances(n, seed=0):
    """A cost table over encodings a and b, and features, for n made-up instances."""
    rng = np.random.default_rng(seed)
    ids = [f"i{k}" for k in range(n)]
    costs = pd.DataFrame({"a": rng.uniform(0, 2, n), "b": rng.uniform(0, 2, n)}, index=ids)
    feats = pd.DataFrame({"n_atoms": rng.integers(0, 50, n), "depth": rng.uniform(size=n)},
                         index=ids)
    return costs, feats


def test_every_leaf_rests_on_at_least_min_samples_leaf_instances():
    costs, feats = random_instances(80)
    forests = fit_forests(costs, feats, trees=5, max_depth=16, min_samples_leaf=8)
    for tree in forests[("a", "b")].estimators_:
        t = tree.tree_
        assert t.n_node_samples[t.children_left == -1].min() >= 8


FEATS = pd.DataFrame({"n_atoms": [3]}, index=["i1"])


class FixedForest:
    """A stand-in pairwise forest that gives every row the same probabilities."""

    def __init__(self, a, b, p_a):
        self.classes_ = np.array([a, b])
        self.p_a = p_a

    def predict_proba(self, X):
        return np.tile([self.p_a, 1 - self.p_a], (len(X), 1))


def forests(p_xy, p_xz, p_yz):
    """Pairwise forests over x, y, z, given P(first of the pair is faster)."""
    return {("x", "y"): FixedForest("x", "y", p_xy),
            ("x", "z"): FixedForest("x", "z", p_xz),
            ("y", "z"): FixedForest("y", "z", p_yz)}


def test_the_encoding_with_most_wins_is_picked():
    picks, fell_back, wins = route(forests(0.9, 0.8, 0.3), ["x", "y", "z"], FEATS, "z")
    assert picks["i1"] == "x" and not fell_back["i1"]
    assert wins.loc["i1"].to_dict() == {"x": 2, "y": 0, "z": 1}


def test_a_tie_at_the_top_falls_back():
    # x beats y, y beats z, z beats x: one win each.
    picks, fell_back, _ = route(forests(0.9, 0.2, 0.9), ["x", "y", "z"], FEATS, "z")
    assert picks["i1"] == "z" and fell_back["i1"]


def test_a_pick_must_beat_the_fallback_by_the_margin():
    fs = forests(0.9, 0.6, 0.3)                  # x wins both of its pairs, beats z at 0.6
    picks, _, _ = route(fs, ["x", "y", "z"], FEATS, "z", margin=0.0)
    assert picks["i1"] == "x"
    picks, fell_back, _ = route(fs, ["x", "y", "z"], FEATS, "z", margin=0.2)
    assert picks["i1"] == "z" and fell_back["i1"]


if __name__ == "__main__":
    test_runs_combine_by_median_and_timeouts_cost_the_timeout()
    test_an_errored_run_leaves_no_cost()
    test_all_timeout_instances()
    test_pair_examples_keep_one_sided_timeouts_and_skip_ties()
    test_features_leave_out_corpus_results_and_labels()
    test_every_leaf_rests_on_at_least_min_samples_leaf_instances()
    test_the_encoding_with_most_wins_is_picked()
    test_a_tie_at_the_top_falls_back()
    test_a_pick_must_beat_the_fallback_by_the_margin()
    print("all router tests passed")
