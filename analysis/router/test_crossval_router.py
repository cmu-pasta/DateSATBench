import numpy as np
import pandas as pd

from analysis.router.crossval_router import (average_summaries, out_of_fold_picks,
                                             strategy_costs, summarize, summary_table)
from analysis.router.test_train_router import random_instances


def test_every_instance_is_routed_once_per_repeat_by_a_router_that_did_not_see_it():
    costs, feats = random_instances(30)
    strata = pd.Series(["llm"] * 15 + ["legal"] * 15, index=costs.index)
    oof = out_of_fold_picks(costs, feats, strata, min_samples_leaf=1, max_depth=8, margin=0.0,
                            n_splits=3, repeats=2, trees=3)
    for _, g in oof.groupby("repeat"):
        assert sorted(g["id"]) == sorted(costs.index)
    for _, g in oof.groupby(["repeat", "fold"]):
        assert not set(g["id"]) & set(g["trained_on"].iloc[0])


def test_every_fold_holds_the_same_share_of_every_corpus():
    costs, feats = random_instances(60)
    strata = pd.Series(["llm"] * 20 + ["legal"] * 40, index=costs.index)
    oof = out_of_fold_picks(costs, feats, strata, min_samples_leaf=1, max_depth=8, margin=0.0,
                            n_splits=5, repeats=2, trees=3)
    for _, g in oof.groupby(["repeat", "fold"]):
        assert strata.loc[g["id"]].value_counts().to_dict() == {"llm": 4, "legal": 8}


def test_average_summaries_averages_the_repeats():
    costs = pd.DataFrame({"x": [1.0, 20.0], "y": [3.0, 2.0]}, index=["i1", "i2"])
    runs = [pd.Series({"i1": "x", "i2": "y"}), pd.Series({"i1": "x", "i2": "x"})]
    fell_back = pd.Series({"i1": False, "i2": True})
    summaries = [summarize(strategy_costs(costs, p), p, fell_back, "y", "y", 20.0) for p in runs]
    avg = average_summaries(summaries, "y", "y")
    assert avg["table"].loc["router", "total_s"] == 12.0        # (3 + 21) / 2
    assert avg["table"].loc["router", "timeouts"] == 0.5
    assert avg["picks"].to_dict() == {"x": 1.5, "y": 0.5}
    assert avg["picks_fastest"] == 0.75                         # 2 of 2, then 1 of 2
    assert avg["fell_back"] == 1
    assert np.isclose(avg["router_total_sd"], np.std([3.0, 21.0], ddof=1))


def test_summary_table():
    timeout_cost = 20.0
    costs = pd.DataFrame({"x": [1.0, timeout_cost], "y": [4.0, 2.0]}, index=["i1", "i2"])
    picks = pd.Series({"i1": "x", "i2": "y"})
    table = summary_table(strategy_costs(costs, picks), "y", timeout_cost)

    assert table.loc["always x", "total_s"] == 21.0
    assert table.loc["always x", "timeouts"] == 1
    assert table.loc["router", "total_s"] == 3.0
    assert table.loc["oracle", "total_s"] == 3.0
    assert table.loc["router", "speedup_total"] == 2.0            # 6 s / 3 s
    assert np.isclose(table.loc["router", "speedup_geomean"], 2.0)  # sqrt(4/1 * 2/2)
    assert table.loc["always y", "speedup_total"] == 1.0


def test_the_router_costs_its_picks_recorded_solve_time():
    costs = pd.DataFrame({"x": [1.0, 19.8], "y": [4.0, 2.0]}, index=["i1", "i2"])
    picks = pd.Series({"i1": "x", "i2": "y"})
    strategies = strategy_costs(costs, picks)
    assert list(strategies.columns) == ["always x", "always y", "router", "oracle"]
    assert strategies["router"].to_dict() == {"i1": 1.0, "i2": 2.0}
    assert "router_time_s" not in summary_table(strategies, "y", 20.0).columns


if __name__ == "__main__":
    test_every_instance_is_routed_once_per_repeat_by_a_router_that_did_not_see_it()
    test_every_fold_holds_the_same_share_of_every_corpus()
    test_average_summaries_averages_the_repeats()
    test_summary_table()
    test_the_router_costs_its_picks_recorded_solve_time()
    print("all crossval_router tests passed")
