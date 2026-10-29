import numpy as np
import pandas as pd

from analysis.router.crossval_router import (average_summaries, live_agreement, live_router,
                                             measured_summaries, out_of_fold_picks,
                                             strategy_costs, strategy_run_costs, summarize,
                                             summary_table)
from analysis.router.test_train_router import random_instances


def runs_of(*per_run):
    """A run_costs table, one row per (run, id), from one cost table per run."""
    return pd.concat({k + 1: c for k, c in enumerate(per_run)}, names=["run", "id"])


def test_every_instance_is_routed_once_per_repeat_by_a_router_that_did_not_see_it():
    costs, feats = random_instances(30)
    strata = pd.Series(["llm"] * 15 + ["legal"] * 15, index=costs.index)
    oof = out_of_fold_picks(costs, feats, strata, min_samples_leaf=1, max_depth=8, margin=0.0,
                            n_splits=3, repeats=2, trees=3)
    for _, g in oof.groupby("repeat"):
        assert sorted(g["id"]) == sorted(costs.index)
    for _, g in oof.groupby(["repeat", "fold"]):
        assert not set(g["id"]) & set(g["trained_on"].iloc[0])


def test_on_fold_gets_every_folds_forests_and_its_instances():
    costs, feats = random_instances(30)
    strata = pd.Series(["llm"] * 15 + ["legal"] * 15, index=costs.index)
    seen = {}
    oof = out_of_fold_picks(costs, feats, strata, min_samples_leaf=1, max_depth=8, margin=0.0,
                            n_splits=3, repeats=2, trees=3,
                            on_fold=lambda r, k, forests, fallback, train, test:
                                seen.update({(r, k): (sorted(forests), fallback, set(train), set(test))}))
    assert sorted(seen) == [(r, k) for r in range(2) for k in range(3)]
    for (r, k), (pairs, fallback, train, test) in seen.items():
        g = oof[(oof["repeat"] == r) & (oof["fold"] == k)]
        assert pairs == [("a", "b")] and set(g["fallback"]) == {fallback}
        assert test == set(g["id"]) and not train & test and train | test == set(costs.index)


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
    summaries = [summarize(strategy_costs(costs, p), strategy_run_costs(runs_of(costs), p), p,
                           fell_back, "y", "y", 20.0) for p in runs]
    avg = average_summaries(summaries, "y", "y")
    assert avg["table"].loc["router", "total_s"] == 12.0        # (3 + 21) / 2
    assert avg["table"].loc["router", "timeouts"] == 0.5
    assert avg["picks"].to_dict() == {"x": 1.5, "y": 0.5}
    assert avg["picks_fastest"] == 0.75                         # 2 of 2, then 1 of 2
    assert avg["fell_back"] == 1
    assert np.isclose(avg["router_total_sd"], np.std([3.0, 21.0], ddof=1))


def test_summary_table():
    timeout_cost = 20.0
    run1 = pd.DataFrame({"x": [1.0, timeout_cost], "y": [4.0, 2.0]}, index=["i1", "i2"])
    run2 = pd.DataFrame({"x": [1.0, 19.0], "y": [4.0, 2.0]}, index=["i1", "i2"])
    costs = pd.DataFrame({"x": [1.0, 19.5], "y": [4.0, 2.0]}, index=["i1", "i2"])   # the medians
    picks = pd.Series({"i1": "x", "i2": "y"})
    table = summary_table(strategy_costs(costs, picks), strategy_run_costs(runs_of(run1, run2), picks),
                          "y", timeout_cost)

    assert table.loc["always x", "total_s"] == 20.5
    assert table.loc["always x", "timeouts"] == 0.5               # timed out in 1 of 2 runs
    assert table.loc["router", "timeouts"] == 0
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
    run_strategies = strategy_run_costs(runs_of(costs), picks)
    assert "router_time_s" not in summary_table(strategies, run_strategies, "y", 20.0).columns


def test_timeouts_are_counted_in_every_run_and_the_oracle_takes_each_runs_fastest():
    run1 = pd.DataFrame({"x": [20.0], "y": [5.0]}, index=["i1"])
    run2 = pd.DataFrame({"x": [3.0], "y": [20.0]}, index=["i1"])
    picks = pd.Series({"i1": "x"})
    run_strategies = strategy_run_costs(runs_of(run1, run2), picks)
    assert run_strategies["router"].tolist() == [20.0, 3.0]
    assert run_strategies["oracle"].tolist() == [5.0, 3.0]
    costs = pd.DataFrame({"x": [11.5], "y": [12.5]}, index=["i1"])
    table = summary_table(strategy_costs(costs, picks), run_strategies, "y", 20.0)
    assert table["timeouts"].to_dict() == {"always x": 0.5, "always y": 0.5, "router": 0.5,
                                           "oracle": 0.0}


def live_rows(*rows):
    """The router's rows of joined.csv from (id, run, time, status, routed_to, routing_time)."""
    return pd.DataFrame(rows, columns=["id", "run", "time", "status", "routed_to", "routing_time"]
                        ).assign(encoding="router")


# i1: x is fast and y times out; i2: y is the faster. Recorded over two runs.
COSTS = pd.DataFrame({"x": [1.0, 4.0], "y": [20.0, 2.0]}, index=["i1", "i2"])
RUNS = runs_of(COSTS, COSTS).rename(index={1: "run_1", 2: "run_2"}, level="run")
LIVE = live_rows(("i1", "run_1", 1.1, "sat", "x", 0.001), ("i2", "run_1", 2.2, "sat", "y", 0.001),
                 ("i1", "run_2", 20.3, "timeout", None, None), ("i2", "run_2", 2.0, "sat", "y", 0.002))


def test_live_router_gives_every_runs_time_and_repeat():
    runs, repeat_of = live_router(LIVE, COSTS.index, 20.0, repeats=2)
    assert repeat_of == {"run_1": 0, "run_2": 1}
    assert runs.loc[("run_1", "i1")] == 1.1 and runs.loc[("run_2", "i1")] == 20.0   # a timeout at 20


def refusal(f, *args, **kwargs):
    """The message of the ValueError f raises."""
    try:
        f(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{f.__name__} raised no ValueError")


def test_live_results_that_miss_a_run_or_outrun_the_repeats_are_refused():
    assert "miss" in refusal(live_router, LIVE.iloc[:3], COSTS.index, 20.0, repeats=2)
    assert "repeats" in refusal(live_router, LIVE, COSTS.index, 20.0, repeats=1)


def predicted_repeats():
    """The predicted summary of two repeats: x then y on i1, y on i2 in both."""
    out = []
    for p1 in ("x", "y"):
        picks = pd.Series({"i1": p1, "i2": "y"})
        out.append(summarize(strategy_costs(COSTS, picks), strategy_run_costs(RUNS, picks), picks,
                             pd.Series({"i1": False, "i2": False}), "x", "x", 20.0))
    return out


def test_every_live_run_is_summarized_like_its_repeat_then_averaged():
    runs, repeat_of = live_router(LIVE, COSTS.index, 20.0, repeats=2)
    per_run = measured_summaries(predicted_repeats(), COSTS, RUNS, runs, repeat_of, "x", 20.0)
    assert np.allclose([s["table"].loc["router", "total_s"] for s in per_run], [3.3, 22.0])
    assert [s["table"].loc["router", "timeouts"] for s in per_run] == [0, 1]
    s = average_summaries(per_run, "x", "x")
    assert np.isclose(s["table"].loc["router", "total_s"], 12.65)    # the mean of the runs' totals
    assert s["table"].loc["router", "timeouts"] == 0.5
    assert s["table"].loc["always y", "timeouts"] == 1.0              # every other row as recorded
    assert s["picks"].to_dict() == {"y": 1.5, "x": 0.5}              # the repeats' predicted picks
    assert np.isclose(s["router_total_sd"], np.std([3.3, 22.0], ddof=1))


def test_a_run_uses_its_own_repeats_picks():
    runs, repeat_of = live_router(LIVE[LIVE["run"] == "run_2"], COSTS.index, 20.0, repeats=2)
    [s] = measured_summaries(predicted_repeats(), COSTS, RUNS, runs, repeat_of, "x", 20.0)
    assert s["picks"].to_dict() == {"y": 2}                           # repeat 1's picks
    assert s["table"].loc["router", "timeouts"] == 1


def test_live_agreement_compares_the_live_picks_with_the_prediction():
    oof = pd.DataFrame({"repeat": [0, 0, 1, 1], "id": ["i1", "i2", "i1", "i2"],
                        "pick": ["x", "y", "x", "x"]})
    a = live_agreement(LIVE, COSTS.index, oof, {"run_1": 0, "run_2": 1}, COSTS, 20.0)
    assert (a["results"], a["picks_known"], a["picks_agree"]) == (4, 3, 2)   # i2 in run_2 differs
    assert a["routing_time_s"]["max"] == 0.002
    assert np.isclose(a["routing_share"], 0.004 / 5.3, rtol=0, atol=1e-6)    # rounded to 6 places
    assert a["time_vs_recorded_median"] == 1.099             # less routing: 1.099, 1.0995, 0.999


if __name__ == "__main__":
    test_every_instance_is_routed_once_per_repeat_by_a_router_that_did_not_see_it()
    test_on_fold_gets_every_folds_forests_and_its_instances()
    test_every_fold_holds_the_same_share_of_every_corpus()
    test_average_summaries_averages_the_repeats()
    test_summary_table()
    test_the_router_costs_its_picks_recorded_solve_time()
    test_timeouts_are_counted_in_every_run_and_the_oracle_takes_each_runs_fastest()
    test_live_router_gives_every_runs_time_and_repeat()
    test_live_results_that_miss_a_run_or_outrun_the_repeats_are_refused()
    test_every_live_run_is_summarized_like_its_repeat_then_averaged()
    test_a_run_uses_its_own_repeats_picks()
    test_live_agreement_compares_the_live_picks_with_the_prediction()
    print("all crossval_router tests passed")
