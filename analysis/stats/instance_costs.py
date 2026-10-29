"""
The one timing measurement the analysis uses: what an instance costs on an encoding.

Every stage that reports solve times or learns from them goes through this module, so the
solver outcomes, the feature x speedup heatmap and the router all measure the same thing:

  * An instance's **cost** on an encoding is the median time over its runs, with a
    timed-out run counting at the timeout. An instance is one measurement, however many
    runs it has.
  * The **complete** instances are those every encoding ran without an error. Every stage
    counts all of them; none leaves out timeouts. An instance on which every encoding timed
    out counts as a timeout for every encoding, and as a speedup of 1 (`all_timeout_ids`
    names them, for stages that report how many there are).
  * A **speedup** over the baseline is the baseline's cost divided by the encoding's cost
    on the same instance, timeouts included at the timeout.
  * What an encoding **solves** is counted per run, not from the costs: in every run it
    solves the instances it finished within the timeout, and the solved count (and the sat,
    unsat and timed-out counts) is the mean of those per-run counts. The solve rate is that
    mean over the instances, i.e. the mean over the runs of each run's share solved.
  * The solver outcomes' **total time**, **median instance** and **median speedup** are
    taken per run, not from the costs: each is computed on every run separately (run k of
    one encoding against run k of the baseline, timeouts at the timeout) and reported as
    the median of those per-run values, with their range.
"""

from analysis.stats.join_results import FINISHED


def cost_table(joined, timeout_cost):
    """Cost in seconds of each encoding on each instance: one row per id, one column per
    encoding. A timed-out run counts at `timeout_cost`, and runs are combined by their
    median. A cell is NaN when any of its runs errored or it has no runs."""
    runs = joined.assign(cost=run_cost(joined, timeout_cost)).groupby(["id", "encoding"])["cost"]
    return runs.median().where(runs.count() == runs.size()).unstack("encoding")


def run_cost(joined, timeout_cost):
    """Each row's time in seconds, a timeout at `timeout_cost`, NaN for an errored run."""
    finished = joined["status"].isin(FINISHED)
    timed_out = joined["status"] == "timeout"
    return joined["time"].where(finished, timeout_cost).where(finished | timed_out)


def run_costs(joined, timeout_cost, ids):
    """Each run's time in seconds on the instances `ids`, a timeout at `timeout_cost`: one
    row per (run, id), one column per encoding."""
    rows = joined[joined["id"].isin(ids)]
    return (rows.assign(cost=run_cost(rows, timeout_cost))
            .set_index(["run", "id", "encoding"])["cost"].unstack("encoding"))


def run_outcomes(joined, timeout_cost, ids):
    """Each run's outcome on the instances `ids`: its answer (sat or unsat) where it finished
    within the timeout, else "timeout". One row per (run, id), one column per encoding."""
    rows = joined[joined["id"].isin(ids)]
    solved = run_cost(rows, timeout_cost) < timeout_cost
    return (rows.assign(outcome=rows["status"].where(solved, "timeout"))
            .set_index(["run", "id", "encoding"])["outcome"].unstack("encoding"))


def all_timeout_ids(joined):
    """Instances on which every encoding timed out in every run."""
    timed_out = joined["status"].eq("timeout").groupby(joined["id"]).all()
    return set(timed_out.index[timed_out])


def complete_costs(joined, timeout_cost):
    """The cost table of the complete instances, and how many have an errored or missing
    run on some encoding and were dropped."""
    costs = cost_table(joined, timeout_cost)
    incomplete = costs.isna().any(axis=1)
    return costs[~incomplete], int(incomplete.sum())
