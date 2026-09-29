# DateSATBench's Grammar-Based Dataset

This directory holds the grammar definition and generator code for producing DateSAT constraints from a formal grammar. The shipped dataset lives under `constraints/`; the grammar and generator source are under `generator/`.

## How the dataset is built

We swarm-sample 300 constraint sets from the formal grammar in `generator/grammar.fan` with the [fandango](https://github.com/fandango-fuzzer/fandango) fuzzer. Before each batch of 5, the generator draws a new setting for every flag in `generator/flags.toml`: how many constraints and date variables an instance has, how large and what shape its Periods are, how long its offset chains and disjunctions get, how many of its variables are pinned to a literal date, how close together its date literals lie, and how often they sit at the end of a month. `generator/flag.md` explains every flag. There is no further selection or filtering step: the sampled constraint sets are the dataset.

Each instance draws its date literals from a window of `literal_spread` days (a month up to the whole range) placed within the full positive-years range 0001-01-01 to 9999-12-31, the same window as the `positive_years` bounds of `datesatbench/utils/bounds.py`. Unlike the LLM and legal datasets, the grammar dataset is not restricted to 1900–2100. The generator does not add bounds of its own: run `bounds inject` separately (see below).

## Generating the dataset (`generate_constraints.py`)

Uses fandango's Python API, so run it with a Python environment that has `fandango` installed (for example the DateSat repo's `.venv`).

```bash
# From repository root: generate the dataset with the settings in flags.toml
python -m datesatbench.grammar_constraints.generator.generate_constraints

# Generate a different number of constraint sets
python -m datesatbench.grammar_constraints.generator.generate_constraints -n 100
```

**Command line arguments:**

- `-c`/`--config`: Flag configuration (default: `generator/flags.toml`)
- `-n`/`--num-samples`: Number of constraint sets to generate (default: `num_samples` in the config, `300`)
- `-o`/`--output`: Output file (default: `constraints/constraints.json`)

The number of constraint sets per flag setting (`batch_size`) and the random seed are set in `flags.toml`; the same config and seed give the same dataset.

Output is written to `datesatbench/grammar_constraints/constraints/constraints.json`, with `decisions.csv` next to it. This file serves as the grammar-based benchmark in DateSATBench.

## Adding and removing bounds (`bounds`)

Bounds are a separate step. `datesatbench/utils/bounds.py inject` appends `D >= Date(...)` and `D <= Date(...)` for every date variable and records the window in an `injected_bound` field; `remove` takes them out again. Both rewrite the three datasets under `datesatbench/` in place, and both are no-ops for entries that are already in the requested state:

```bash
# From repository root: the positive_years bounds, 0001-01-01 .. 9999-12-31
python -m datesatbench.utils.bounds inject

# A custom window (replaces any other window an entry carries)
python -m datesatbench.utils.bounds inject --min 1950/1/1 --max 2050/12/31 --name years_1950_2050

# Take the bounds out again
python -m datesatbench.utils.bounds remove
```

Regenerating the dataset writes it without bounds, so run `bounds inject` again afterwards if you want them.

## Output format

Each entry in `constraints.json` is a JSON object with:

- `id`: `grammar-<n>`, numbered in generation order
- `declarations`: the date variables the constraints use, e.g. `D0: date`
- `constraints`: list of DateSAT DSL constraint strings: the sampled constraints, then the anchors (`D == Date(y, m, d)`)
- `size`: number of constraints in the entry, anchors included
- `flags`: the flag setting the entry was sampled with

`decisions.csv` has one row per flag setting: `batch`, the `first_id` and `last_id` of the entries sampled with it, `num_instances` (entries) and `num_constraints_total` (constraints across those entries), followed by one column per flag.
