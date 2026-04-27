# DateSATBench's Grammar-Based Dataset Generator

This directory holds grammar definitions and generator code for producing DateSAT constraints from a formal grammar. Outputs and bundled examples live under `constraints/`; source for generation is under `generator/`.

## Processing Pipeline

### Step 1: Generate Constraints (`generate_constraints.py`)

Uses the [fandango](https://github.com/fandango-fuzzer/fandango) fuzzer to sample constraint sets from the formal grammar defined in `generator/grammar.fan`, then converts the output to JSON.

```bash
# From repository root — generate 10 constraint sets (default)
python -m datesatbench.grammar_constraints.generator.generate_constraints

# Generate a custom number of constraint sets
python -m datesatbench.grammar_constraints.generator.generate_constraints -n 100
```

**Command line arguments:**

- `-n`/`--num-samples`: Number of constraint sets to generate (default: `10`)
- `-m`/`--max-nodes`: Maximum AST nodes per generated sample (default: `1000`)

Output is written to `datesatbench/grammar_constraints/constraints/constraints.json`.

### Step 2: Pick Benchmarks (`pick_benchmarks.py`)

Reads solver results from `results/naive_int.json` and splits constraints into three files based on their solver status. Within each category, constraints are sorted by execution time (slowest first) so the hardest instances are ranked at the top. Duplicate constraints (matched by exact constraint list) are skipped when appending to existing files.

```bash
python -m datesatbench.grammar_constraints.generator.pick_benchmarks
```

Outputs (written to `constraints/`):
- `sat_constraints.json` — satisfiable constraints
- `unsat_constraints.json` — unsatisfiable constraints
- `timeout_constraints.json` — constraints that exceeded the solver timeout

### Step 3: Merge Benchmarks (`merge_benchmarks.py`)

Merges all JSON files in the `constraints/` directory into a single `constraints.json` for distribution.

```bash
python -m datesatbench.grammar_constraints.generator.merge_benchmarks
```

Output is written to `datesatbench/grammar_constraints/constraints/constraints.json`. This file serves as the grammar-based benchmark in DateSATBench.
