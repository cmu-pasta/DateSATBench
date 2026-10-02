"""
The one timing measurement the analysis uses: what an instance costs on an encoding.

Every stage that reports solve times or learns from them goes through this module, so the
solver outcomes, the feature x speedup heatmap and the router all measure the same thing:

  * An instance's **cost** on an encoding is the median time over its runs, with a
    timed-out run counting at the timeout. An instance is one measurement, however many
    runs it has.
  * The **usable** instances are those every encoding ran without an error, less those on
    which every encoding timed out. No encoding is faster than another on those, so they
    are counted on their own (`dropped["all_timeout"]`) rather than as ties.
  * An encoding **solves** an instance when its cost is below the timeout.
  * A **speedup** over the baseline is the baseline's cost divided by the encoding's cost
    on the same instance, timeouts included at the timeout.
"""

from analysis.stats.join_results import FINISHED


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


def usable_costs(joined, timeout_cost):
    """The cost table of the usable instances, and how many instances were dropped.

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


def answers(joined):
    """The answer (sat or unsat) of each encoding on each instance, from its finished runs:
    one row per id, one column per encoding, NaN where no run finished."""
    done = joined[joined["status"].isin(FINISHED)]
    return done.groupby(["id", "encoding"])["status"].first().unstack("encoding")
