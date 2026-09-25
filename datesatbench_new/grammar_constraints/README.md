# DateSATBench's Grammar-Based Dataset

This directory holds the grammar definition and generator code for producing DateSAT constraints from a formal grammar. The shipped dataset lives under `constraints/`; the grammar and generator source are under `generator/`.

## How the dataset is built

We take the formal grammar in `generator/grammar.fan` and randomly produce 300 constraint sets from it using the [fandango](https://github.com/fandango-fuzzer/fandango) fuzzer. There is no further selection or filtering step: the 300 randomly sampled constraint sets are the dataset.

Date literals in the generated constraints range over the full positive-years range 0001-01-01 to 9999-12-31, the same bounds as the `positive_years` variant under `datesatbench_bounded/`. Unlike the LLM and legal datasets, the grammar dataset is not restricted to 1900–2100.

## Generating the dataset (`generate_constraints.py`)

Samples constraint sets from the grammar with fandango and converts the output to JSON. The `fandango` CLI must be installed and on your `PATH`.

```bash
# From repository root — generate the 300-constraint dataset (default)
python -m datesatbench_new.grammar_constraints.generator.generate_constraints

# Generate a different number of constraint sets
python -m datesatbench_new.grammar_constraints.generator.generate_constraints -n 100
```

**Command line arguments:**

- `-n`/`--num-samples`: Number of constraint sets to generate (default: `300`)
- `-m`/`--max-nodes`: Maximum AST nodes per generated sample (default: `1000`)

Output is written to `datesatbench_new/grammar_constraints/constraints/constraints.json`. This file serves as the grammar-based benchmark in DateSATBench.

## Output format

Each entry in `constraints.json` is a JSON object with:

- `id`: `grammar-<n>`, numbered in generation order
- `declarations`: variable declarations inferred from the constraints (`D0`–`D9` dates, `B0`–`B9` bools, `I0`–`I9` ints)
- `constraints`: list of DateSAT DSL constraint strings
- `size`: number of constraints in the entry
