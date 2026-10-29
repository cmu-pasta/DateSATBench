# Analysis

Runs the [DateSAT](https://github.com/cmu-pasta/DateSAT) solver on DateSATBench and relates
its results to features of the benchmark instances.

## Prereqs

The solver lives in the separate DateSAT repo. Install it into the Python environment you run
`analysis/` with:

```bash
pip install -e /path/to/DateSat            # the solver (z3, lark, dateutil come with it)
pip install -e ".[analysis]"               # pandas, scikit-learn, plotting libraries, pytest
```

## Layout

| Folder | Stage | What it holds |
|---|---|---|
| `run/` | 0 | `run_benchmarks.py` (runs DateSat on the datasets and writes `results/<tag>/`), `run_router_cv.py` (the router's live evaluation, with the cross-validation's fold models), `validation.py` (checks every solution by concrete execution) and their tests |
| `features/` | 1 | The constraint parser, `extract_features.py`, `emulate_encodings.py` (counts, without solving, what each DateSat encoding would emit; no feature uses it at present), `check_emulation.py` (compares those counts with the formulas DateSat builds), `features.md` (documents every feature column) and its tests |
| `stats/` | 2–4 | `join_results.py` (joins solver timings to the features), `cluster.py` (PCA and t-SNE projections of the features to 2D and 3D), `solver_outcomes.py`, the `plot_*.py` scripts (`plot_tsne.py` draws the paper's t-SNE figure from `clusters.json`) and `select_features.py` (one feature per family of correlated features) |
| `report/` | 5 | `build_report.py` and the HTML template it fills in |
| `router/` | 6–7 | `crossval_router.py` (measures the router's setting on unseen constraints, by cross-validation, and writes the fold models), `train_router.py` (trains the encoding router on every instance: one random forest per pair of encodings), `check_datesat_router.py` (checks that DateSat routes as the cross-validation predicts) and their tests |
| `outputs/` | | Everything the stages write: `features.csv`, `features_meta.json`, `selected_features.json`, `joined.csv`, `clusters.json`, `plots/`, `report.html`, `model/` (the router trained on every instance), `model_cross_validation/` (the cross-validation's results and fold models) and `model_cross_validation_selected/` (the same on the selected features only) |

`paths.py` holds the locations the stages share.

`outputs/model/router.json` is the model for DateSat's `router` approach: copy it to `datesat/router/model.json` in the DateSat repo, or point `DATESAT_ROUTER_MODEL` at it. DateSat only accepts a model whose features kept the injected bounds (`extract_features --bounds keep`).

## Running the solver (stage 0)

Run every dataset in `datesatbench/`, then check every solution:

```bash
python -m analysis.run.run_benchmarks
```

Run only one dataset (short names: `llm`, `grammar`, `legal`):

```bash
python -m analysis.run.run_benchmarks --datesatbenchs llm
```

Choose the approaches, the timeout, etc.:

```bash
python -m analysis.run.run_benchmarks \
  --timeout 20000 \
  --approaches alpha_beta_table \
  --datesatbenchs legal
```

To run a different dataset directory, pass it with `--datesatbench-repo` (or set `DATESATBENCH_REPO`). Bounds are part of the dataset text: add or remove them in place with `python -m datesatbench.utils.bounds inject|remove` before running.

`--timeout` only bounds the Z3 check; building the constraints is not covered by it and can hang (e.g. `simple` on a `Period` with tens of thousands of days). Each instance therefore runs in a child process that is killed after `--hard-timeout` ms of wall-clock time (default: 20000). A killed instance is recorded as `"status": "timeout"` with `"hard_timeout": true`. Each result also splits `execution_time` into `build_time` (constructing the constraints) and `solve_time` (the Z3 check and model extraction).

After the runs, `validation.py` checks every solution by executing the constraints on it, and writes `checked_summary_with_baseline.json`; `--no-analysis` skips that check. To check results again later: `python -m analysis.run.validation results/<tag>/llm/run_1`.

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

## Running the pipeline

`analysis` is a package, so run each stage as a module from the repository root:

```
export DATESAT_TIMEOUT_MS=20000                          # the --timeout the results were run with
python -m analysis.features.extract_features             # 1: outputs/features.csv
python -m analysis.stats.join_results                    # 2: outputs/joined.csv
python -m analysis.stats.cluster                         # 3: outputs/clusters.json
python -m analysis.stats.plot_speedup_heatmap            # 4: outputs/plots/ (and --corpus llm|legal|grammar)
python -m analysis.stats.plot_feature_correlation        # 4: outputs/plots/feature_correlation.png
python -m analysis.stats.select_features                 # 4: outputs/selected_features.json
python -m analysis.stats.plot_feature_correlation --selected   # 4: outputs/plots/feature_correlation_selected.png
python -m analysis.stats.plot_tsne                       # 4: outputs/plots/dataset_tsne.pdf, the paper's t-SNE figure
python -m analysis.report.build_report                   # 5: outputs/report.html
python -m analysis.router.crossval_router                # 6: outputs/model_cross_validation/router_eval.json, router_eval.csv, fold_models.json, fold_models/
python -m analysis.router.crossval_router --selected     # 6: the same on the selected features, in outputs/model_cross_validation_selected/
python -m analysis.router.train_router                   # 7: outputs/model/router.joblib, router_meta.json, router.json
```

`python -m analysis.stats.solver_outcomes --compare` prints a summary of what each encoding
did, per corpus. Every stage takes `--help`. `join_results --exclude <encoding> ...` leaves
encodings in the results out of `joined.csv` and so out of every later stage; the report's
header names them.

To analyse other results without overwriting `outputs/`, point `--results` (stages 2 and 5)
at them and set `DATESAT_ANALYSIS_OUTPUTS` to another directory, which every stage then
reads and writes instead. Create the directory first.

The report's *Specialized router* section reads the outputs of stages 6 and 7
(`model_cross_validation/router_eval.json` and `model/router_meta.json`), so rebuild the
report after running them; without them the section says so and stays empty. Its
*Feature reduction* section reads `selected_features.json` and both cross-validations,
with and without `--selected`.

## Fewer features

`select_features` cuts every family of strongly correlated features (every pair has
Spearman |rho| >= `--threshold`, default 0.8, the families outlined in
`feature_correlation.png`) to its medoid, the member with the highest mean |rho| to the
rest, and drops constant features. It looks only at the features, not at solver results,
so `crossval_router --selected` measures the router on the selected features without a
bias from the choice. `train_router` and DateSat's model still use every feature.

## How well the router does on unseen constraints

`crossval_router` measures the router on constraints it has not seen: 5-fold
cross-validation over every instance, stratified by corpus and repeated 10 times, in which
every instance is routed by a router trained on the other folds. It measures one fixed
setting, `train_router`'s (`DEFAULT_SETTINGS` and `DEFAULT_TREES`), and does not search
for a better one; pass `--min-samples-leaf`, `--max-depth`, `--max-features`, `--margin`,
`--train-timeout-cost` or `--trees` to measure another. Its results go to `model_cross_validation/`; the model DateSat uses,
trained on every instance with the same setting, goes to `model/`.

Until the live evaluation below has run, the router is *predicted*: on every instance it is
charged the recorded time of the encoding it picks, without the time routing takes.

## The router's live evaluation

`crossval_router` also writes every fold's model for DateSat (`model_cross_validation/
fold_models/`, not committed, listed in `fold_models.json` with every instance's fold and
predicted pick). `run_router_cv` runs DateSat's `router` approach with them, one run per
repeat: in run N every instance is solved by the model of repeat N - 1 that was not trained
on it. Everything else is `run_benchmarks`': the same timeouts, the runs one after another,
the corpora in the same order, the instances in dataset order, one worker per run and
corpus. Its results go to `results/<tag>/<corpus>/run_N/router_int.json`. Before solving,
it runs `check_datesat_router`, which checks that DateSat's copy of the feature extractor
gives the same features and DateSat's router the predicted pick on every (repeat, instance),
and stops on any difference.

```
python -m analysis.router.crossval_router                     # writes the fold models
python -m analysis.router.check_datesat_router                # optional: run_router_cv runs it too
python -m analysis.run.run_router_cv --tag router-cv-10runs   # 10 runs; resumes if stopped
python -m analysis.stats.join_results --exclude hybrid_both \
    --results results/all-10runs-bounded-datesat28d8186 results/router-cv-10runs
python -m analysis.router.crossval_router                     # the router now measured
python -m analysis.report.build_report \
    --results results/all-10runs-bounded-datesat28d8186 results/router-cv-10runs
```

The fold models must be the ones `run_router_cv` used: run it from the same checkout, or copy
`model_cross_validation/` over with the folder `fold_models/`. Once `joined.csv` holds the
router's results, `crossval_router` reports it as *measured*: every live run is summarized
as its repeat's prediction is, with the router charged its time in that run (routing
included), and the summaries are averaged over the runs. `router_eval.json`'s `live` block
says how the live runs compare with the prediction: pick agreement (a mismatch stops
`crossval_router`), routing time, and the live time less routing against the picked
encoding's recorded time, about 1 on the machine that recorded the encodings
(`run_router_cv` records the machine in `run_config.json`). The report's Solver outcomes and cactus plot
show the router beside the encodings, measured the same way; `paper_table --with-router`
and `plot_cactus --with-router` add it to the paper's table and figure. No stage that
compares encodings counts it: not the oracle, the router's training, the instances no
encoding solves, or the heatmaps.

## Tests

```
python -m pytest                                    # analysis/run/: validation and run_router_cv; needs DateSat
python -m analysis.features.test_features_md        # runs every example in features.md
python -m analysis.features.test_extract_features
python -m analysis.features.check_emulation         # needs DateSat; takes about 10 minutes
python -m analysis.router.test_train_router
python -m analysis.router.test_crossval_router
python -m analysis.stats.test_select_features
python -m analysis.stats.test_solver_outcomes
python -m analysis.stats.test_join_results
python -m analysis.report.test_build_report
```
