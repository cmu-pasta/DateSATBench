import pandas as pd

from analysis.stats.solver_outcomes import fmt_pct, run_counts, solved_times, solver_outcomes

TIMEOUT = 20.0


def joined(rows):
    """joined.csv rows from (id, encoding, run, status, time) tuples, all in one corpus."""
    return pd.DataFrame(rows, columns=["id", "encoding", "run", "status", "time"]).assign(corpus="llm")


# i1: x solves it in 1 of 4 runs, so its median time is the timeout; y solves it every run.
# i2: both solve it every run, x as unsat. i3: neither solves it in any run.
ROWS = ([("i1", "x", r, "sat" if r == 1 else "timeout", 1.0 if r == 1 else 20.01) for r in range(1, 5)]
        + [("i1", "y", r, "sat", 2.0) for r in range(1, 5)]
        + [("i2", "x", r, "unsat", 0.5) for r in range(1, 5)]
        + [("i2", "y", r, "unsat", 3.0) for r in range(1, 5)]
        + [(i, e, r, "timeout", 20.02) for i in ["i3"] for e in "xy" for r in range(1, 5)])


def test_the_solved_counts_are_the_mean_over_the_runs_of_each_runs_count():
    t = solver_outcomes(joined(ROWS), TIMEOUT, baseline="y")
    x = t.loc["x"]
    assert (x["sat"], x["unsat"], x["timeout"]) == (0.25, 1.0, 1.75)
    assert x["solved"] == 1.25
    assert x["solve_rate"] == 1.25 / 3
    assert t.loc["y", "solved"] == 2.0
    assert t.loc["y", "unsolvable"] == 1          # i3: every encoding timed out in every run


def test_the_router_is_a_row_but_does_not_decide_what_no_encoding_solves():
    router = [("i1", "router", r, "sat", 2.5) for r in range(1, 5)] \
        + [("i2", "router", r, "unsat", 0.6) for r in range(1, 5)] \
        + [("i3", "router", r, "sat" if r == 1 else "timeout", 19.0 if r == 1 else 20.02) for r in range(1, 5)]
    t = solver_outcomes(joined(ROWS + router), TIMEOUT, baseline="y")
    assert t.loc["router", "solved"] == 2.25 and t.loc["router", "total_time"] == 23.1
    assert t.loc["y", "unsolvable"] == 1          # i3: no encoding solves it, whatever the router does
    assert t.drop(index="router").equals(solver_outcomes(joined(ROWS), TIMEOUT, baseline="y"))


def test_a_row_with_fewer_runs_is_measured_over_its_own_runs():
    router = [("i1", "router", 1, "sat", 2.5), ("i2", "router", 1, "unsat", 0.5),
              ("i3", "router", 1, "timeout", 20.02)]
    df = joined(ROWS + router)
    r = solver_outcomes(df, TIMEOUT, baseline="y").loc["router"]
    assert (r["solved"], r["timeout"]) == (2.0, 1.0)         # its one run, not a mean over four
    assert r["total_time"] == r["total_time_min"] == r["total_time_max"] == 23.0
    assert run_counts(df) == {"router": 1, "x": 4, "y": 4}
    assert len(solved_times(df, TIMEOUT)["router"]) / run_counts(df)["router"] == r["solved"]


def test_a_router_error_is_refused_rather_than_dropping_the_instance_for_everyone():
    router = [(i, "router", r, "sat", 1.0) for i in ("i1", "i2", "i3") for r in range(1, 5)]
    router[0] = ("i1", "router", 1, "error", 0.1)
    try:
        solver_outcomes(joined(ROWS + router), TIMEOUT, baseline="y")
    except ValueError as e:
        assert "1 of 12 (run, instance) pairs" in str(e)
    else:
        raise AssertionError("an errored router run was not refused")


def test_a_router_run_that_covers_only_some_instances_is_refused():
    router = [(i, "router", 1, "sat", 1.0) for i in ("i1", "i2", "i3")] + [("i1", "router", 2, "sat", 1.0)]
    try:
        solver_outcomes(joined(ROWS + router), TIMEOUT, baseline="y")
    except ValueError as e:
        assert "2 of 6 (run, instance) pairs" in str(e)
    else:
        raise AssertionError("a half-finished router run was not refused")


def test_a_finished_run_over_the_timeout_is_a_timeout():
    rows = [("i1", e, 1, "sat", 20.5) for e in "xy"]
    t = solver_outcomes(joined(rows), TIMEOUT, baseline="y")
    assert t.loc["x", "solved"] == 0 and t.loc["x", "timeout"] == 1


def test_the_cactus_ends_at_the_solved_count():
    times = solved_times(joined(ROWS), TIMEOUT)
    assert times["x"] == [0.5, 0.5, 0.5, 0.5, 1.0]  # every solved run, fastest first
    assert len(times["x"]) / 4 == solver_outcomes(joined(ROWS), TIMEOUT, baseline="y").loc["x", "solved"]


def test_a_half_rounds_up_in_percentages():
    assert fmt_pct(175.1 / 200) == "87.6"           # the float is just under 0.8755
    assert fmt_pct(65 / 200, 0) == "33"             # 32.5
    assert fmt_pct(685 / 800) == "85.6"             # 85.625 rounds to 85.6 at one decimal
    assert fmt_pct(1.0) == "100.0" and fmt_pct(0.0, 0) == "0"


if __name__ == "__main__":
    test_the_solved_counts_are_the_mean_over_the_runs_of_each_runs_count()
    test_the_router_is_a_row_but_does_not_decide_what_no_encoding_solves()
    test_a_row_with_fewer_runs_is_measured_over_its_own_runs()
    test_a_router_error_is_refused_rather_than_dropping_the_instance_for_everyone()
    test_a_router_run_that_covers_only_some_instances_is_refused()
    test_a_finished_run_over_the_timeout_is_a_timeout()
    test_the_cactus_ends_at_the_solved_count()
    test_a_half_rounds_up_in_percentages()
    print("all solver_outcomes tests passed")
