#!/usr/bin/env python3
"""
Generate the grammar-based dataset by swarm sampling from grammar.fan.

The sampler draws one value for every flag in flags.toml, samples a batch of
instances with that setting, and repeats until it has enough instances. flag.md
explains what each flag does.

Each instance is built in four steps:
1. Place the literal window: a stretch of `literal_spread` days, placed uniformly
   within 0001-01-01 .. 9999-12-31, that every date literal of the instance is drawn
   from.
2. Generate `num_constraints` constraints one at a time with fandango: a
   `<constraint>`, or a single `<unit_constraint>` when `disjunctions` is off.
3. Redraw any constraint that compares a variable with itself and no offset, such as
   `D1 < D1`.
4. Append the anchors: `D == Date(y, m, d)` for floor(anchor_frac * num_date_vars)
   of the variables the constraints use.
No bounds are added here. To bound every date variable, run
`python -m datesatbench.utils.bounds inject` separately.

Before each batch, the sampler also copies the whole flag setting into the grammar's
FLAGS dict. The flags that shape literals and variables (the variable pool, the offset
chains, the Period fields and caps, `!=`, and how often date literals fall near a month
end) are read by the grammar from there. The literal window goes into the grammar's
INSTANCE dict before each instance.

Output format (JSON): an array of objects with fields
- "id": "grammar-<n>", numbered in generation order
- "declarations": the date variables the constraints use, e.g. ["D0: date", "D3: date"]
- "constraints": the sampled constraints, then the anchors
- "size": number of constraints, anchors included
- "flags": the flag setting the instance was sampled with

Next to it, decisions.csv has one row per flag setting: the batch number, the ids of
the instances sampled with it, how many instances and constraints that is, and the
value of every flag.
"""

import argparse
import csv
import datetime
import json
import math
import random
import re
import sys
import tomllib
from pathlib import Path

from fandango import Fandango

# At least one of these must be above 0, or no Period can be generated.
PERIOD_CAPS = ("max_period_years", "max_period_months", "max_period_days")

# The ordinal of 9999-12-31, the last date the solver represents. 0001-01-01 is 1.
LAST_ORDINAL = datetime.date.max.toordinal()

# fandango's node budget for one constraint. It only limits how long a disjunction may
# get, since the grammar's generators build everything below a comparison, and it is far
# above any length the grammar reaches, so it never cuts a constraint short.
MAX_NODES = 100_000

# A comparison as the grammar writes it: a date variable, an operator, and a right-hand
# side that is either a date literal or a date variable with a chain of offsets.
UNIT_RE = re.compile(r"(D\d+) (==|!=|<=|>=|<|>) (.+)")
DATE_VAR_RE = re.compile(r"\bD(\d+)\b")

# Redraws allowed for one constraint before giving up. Every flag setting accepts a
# large share of constraints, so this is only reached if something is broken.
MAX_ATTEMPTS = 10_000


def load_config(config_file):
    """Read flags.toml and check that every flag can be drawn from."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    flags = config["flags"]

    for name, spec in flags.items():
        is_list = isinstance(spec, list) and len(spec) > 0
        is_range = isinstance(spec, dict) and set(spec) == {"min", "max"} and spec["min"] <= spec["max"]
        if not (is_list or is_range):
            raise ValueError(f"{config_file}: {name} must be a non-empty list or a range {{ min = a, max = b }}")
    if not any(value > 0 for cap in PERIOD_CAPS for value in flags[cap]):
        raise ValueError(f"{config_file}: at least one max_period_* flag must allow a value above 0")
    if min(flags["literal_spread"]) < 1:
        raise ValueError(f"{config_file}: every literal_spread value must be at least 1 day")

    return config["sampling"], flags


def draw(spec):
    """Draw one value: uniformly from a list, or from a { min, max } range in steps of 0.01."""
    if isinstance(spec, dict):
        return random.randint(round(spec["min"] * 100), round(spec["max"] * 100)) / 100
    return random.choice(spec)


def show(spec, value):
    """Format a flag value for the log and decisions.csv; a value drawn from a range as a percentage."""
    return f"{round(value * 100)}%" if isinstance(spec, dict) else value


def draw_setting(flags):
    """Draw a value for every flag. The Period caps are redrawn while all three are 0."""
    setting = {name: draw(spec) for name, spec in flags.items()}
    while not any(setting[cap] for cap in PERIOD_CAPS):
        for cap in PERIOD_CAPS:
            setting[cap] = draw(flags[cap])
    return setting


def has_bare_self_comparison(constraint):
    """Return True if a comparison in the constraint compares a variable with itself
    and no offset, such as D1 < D1. A self-comparison with an offset, such as
    D1 < (D1 + Period(0, 0, 3)), is kept."""
    for unit in constraint.split(" || "):
        lhs, _, rhs = UNIT_RE.fullmatch(unit).groups()
        if rhs == lhs:
            return True
    return False


def generate_constraint(grammar, setting):
    """Generate one constraint that respects the sampler's flags."""
    start = "<constraint>" if setting["disjunctions"] else "<unit_constraint>"
    for _ in range(MAX_ATTEMPTS):
        constraint = str(grammar.fuzz(start, max_nodes=MAX_NODES))
        if not has_bare_self_comparison(constraint):
            return constraint
    raise RuntimeError(f"no valid constraint after {MAX_ATTEMPTS} attempts with {setting}")


def literal_window(literal_spread):
    """Place a window of literal_spread days uniformly within 0001-01-01 .. 9999-12-31.

    Returns its first and last day as ordinals. A spread of the whole range or more
    gives the whole range.
    """
    size = min(literal_spread, LAST_ORDINAL)
    first = random.randint(1, LAST_ORDINAL - size + 1)
    return first, first + size - 1


def anchor_constraints(constraints, setting, random_date_ctor):
    """Pin floor(anchor_frac * num_date_vars) of the used variables to a literal date."""
    used = sorted(set(DATE_VAR_RE.findall(" ".join(constraints))), key=int)
    # In whole percent, so that e.g. 15% of 20 is exactly 3 and not 2.9999...
    percent = round(setting["anchor_frac"] * 100)
    num_anchors = min(percent * setting["num_date_vars"] // 100, len(used))
    pinned = sorted(random.sample(used, num_anchors), key=int)
    return [f"D{v} == {random_date_ctor()}" for v in pinned]


def declarations(constraints):
    """Declare every date variable the constraints use, in numeric order."""
    used = sorted(set(DATE_VAR_RE.findall(" ".join(constraints))), key=int)
    return [f"D{v}: date" for v in used]


def generate_dataset(grammar_file, flags, num_samples, batch_size):
    """Sample num_samples instances, drawing a new flag setting every batch_size.

    Returns the instances and one decision row per flag setting.
    """
    grammar = Fandango(open(grammar_file), use_cache=False).grammar
    spec_globals = grammar.get_spec_env()[0]
    grammar_flags = spec_globals["FLAGS"]
    grammar_instance = spec_globals["INSTANCE"]
    random_date_ctor = spec_globals["random_date_ctor"]

    instances = []
    decisions = []
    num_batches = math.ceil(num_samples / batch_size)
    for batch in range(num_batches):
        setting = draw_setting(flags)
        grammar_flags.update(setting)
        print(f"batch {batch + 1}/{num_batches}: " + ", ".join(f"{k}={show(flags[k], v)}" for k, v in setting.items()))

        batch_instances = []
        for _ in range(min(batch_size, num_samples - len(instances))):
            grammar_instance["literal_window"] = literal_window(setting["literal_spread"])
            constraints = [generate_constraint(grammar, setting) for _ in range(setting["num_constraints"])]
            constraints += anchor_constraints(constraints, setting, random_date_ctor)
            batch_instances.append(
                {
                    "id": f"grammar-{len(instances) + len(batch_instances) + 1}",
                    "declarations": declarations(constraints),
                    "constraints": constraints,
                    "size": len(constraints),
                    "flags": setting,
                }
            )
        instances += batch_instances
        decisions.append(
            {
                "batch": batch + 1,
                "first_id": batch_instances[0]["id"],
                "last_id": batch_instances[-1]["id"],
                "num_instances": len(batch_instances),
                "num_constraints_total": sum(inst["size"] for inst in batch_instances),
                **{name: show(flags[name], value) for name, value in setting.items()},
            }
        )
    return instances, decisions


def main():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Swarm-sample constraint sets from grammar.fan and write them as JSON"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=script_dir / "flags.toml",
        help="Flag configuration (default: flags.toml next to this script)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        help="Number of constraint sets to generate (default: num_samples in the config)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=script_dir.parent / "constraints" / "constraints.json",
        help="Output file (default: ../constraints/constraints.json)",
    )
    args = parser.parse_args()

    try:
        sampling, flags = load_config(args.config)
    except (OSError, ValueError, KeyError) as e:
        print(f"Error reading config: {e}")
        sys.exit(1)
    num_samples = args.num_samples if args.num_samples is not None else sampling["num_samples"]
    random.seed(sampling["seed"])

    instances, decisions = generate_dataset(
        script_dir / "grammar.fan", flags, num_samples, sampling["batch_size"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(instances, f, indent=2)
    decisions_file = args.output.with_name("decisions.csv")
    with open(decisions_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Wrote {len(instances)} constraint sets to {args.output}")
    print(f"Wrote {len(decisions)} flag settings to {decisions_file}")


if __name__ == "__main__":
    main()
