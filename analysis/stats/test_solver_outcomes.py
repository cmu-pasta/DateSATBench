import pandas as pd

from analysis.stats.solver_outcomes import fmt_pct, solved_times, solver_outcomes

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
    test_a_finished_run_over_the_timeout_is_a_timeout()
    test_the_cactus_ends_at_the_solved_count()
    test_a_half_rounds_up_in_percentages()
    print("all solver_outcomes tests passed")
