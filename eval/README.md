# DateSAT evaluation

**Note:** Commands below assume you run them from the **DateSATBench repo root**.

`eval/` contains the **evaluation + plotting utilities** for running the DateSAT solver on the DateSATBench datasets.

## Prereqs

The solver lives in the separate [DateSAT repo](https://github.com/cmu-pasta/DateSAT). Install it into the Python environment you run `eval/` with:

```bash
pip install -e /path/to/DateSat            # the solver (z3, lark, dateutil come with it)
pip install -e ".[eval]"                   # plotting libraries and pytest for eval/
```

## Run the evaluation

Run all datasets in `datesatbench/` (and run analysis at the end):

```bash
python3 eval/run_benchmarks.py
```

Run only one dataset (short names: `llm`, `grammar`, `legal`):

```bash
python3 eval/run_benchmarks.py --datesatbenchs llm
```

Add approach filtering / timeout, etc.:

```bash
python3 eval/run_benchmarks.py \
  --timeout 20000 \
  --approaches alpha_beta_table \
  --datesatbenchs legal
```

To run a different dataset directory, pass it with `--datesatbench-repo` (or set `DATESATBENCH_REPO`). Bounds are part of the dataset text: add or remove them in place with `python -m datesatbench.utils.bounds inject|remove` before running.

`--timeout` only bounds the Z3 check; building the constraints is not covered by it and can hang (e.g. `simple` on a `Period` with tens of thousands of days). Each instance therefore runs in a child process that is killed after `--hard-timeout` ms of wall-clock time (default: 20000). A killed instance is recorded as `"status": "timeout"` with `"hard_timeout": true`. Each result also splits `execution_time` into `build_time` (constructing the constraints) and `solve_time` (the Z3 check and model extraction).

### Output location

Results are written to `<results-dir>/<tag>/`:

- `<results-dir>` defaults to `results/` in this repo (or, with `--datesatbench-repo`, in the DateSATBench checkout that contains the dataset). Override with `--results-dir`.
- `<tag>` defaults to a `YYYYmmdd_HHMMSS` timestamp. Override with `--tag`; invocations sharing a tag write into the same directory.
- Every run goes in its own `run_N/`. Each invocation takes, per dataset, the lowest `N` whose `run_N/` has no result file for the approaches being run. So re-running with the same `--tag` adds `run_2`, `run_3`, …, while invocations covering *different* approaches (e.g. one per approach, launched in parallel) share a `run_N`. `--runs K` produces `K` consecutive run directories.

```
<results-dir>/<tag>/
├── run_config.json                # timeout, approaches, datasets, dataset root
└── <llm|grammar|legal>/
    └── run_N/
        ├── <approach>_int.json    # per-constraint status, time, solution
        └── checked_summary_with_baseline.json   # unless --no-analysis
```

A result file is written only once its approach finishes every constraint, so a killed run leaves its `run_N/` reusable. Starting the *same* approach again while an earlier invocation is still running will pick the same `run_N`.


## Utils

The `eval/utils/` directory contains utility scripts for analysis and plotting:

- **`plot_benchmark_stats.py`**: plots dataset stats (vars/constraints).
- **`plot_normalized_speedup.py`**: plots results from `run_benchmarks.py` outputs (run evaluation first).
- **`compute_time.py`**: execution time statistics from result JSON files.
- **`validation.py`**: validates solver solutions against constraints using concrete execution. Its tests are in `test_validation.py`; run them with `python -m pytest`.

`plot_normalized_speedup.py` and `compute_time.py` need the timeout the results were run with, in ms, and refuse to run without it: pass `--timeout <ms>` or set `DATESAT_TIMEOUT_MS`. Timed-out runs count at that timeout.
