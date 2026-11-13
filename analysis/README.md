# Analysis

Runs the [DateSAT](https://github.com/cmu-pasta/DateSAT) solver on DateSATBench and relates its results to features of the instances. 
The solver's runs are in `results/runs/`, everything derived from them in `results/analysis/` (see `results/README.md`).

## Setup

```bash
pip install -e ".[analysis]"        # pandas, scikit-learn, plotting libraries, pytest
pip install -e <path to DateSAT>    # only to run the solver (stage 0) and the tests that say so
```

`analysis` is a package: run everything as a module from the repository root. 
Every script takes `--help`, and its docstring describes what it computes.

## Layout

| Folder | Stage | What it holds |
|---|---|---|
| `run/` | 0 | `run_benchmarks.py` runs DateSAT on the datasets, `run_router_cv.py` runs the router's live evaluation, `validation.py` checks every solution by concrete execution |
| `features/` | 1 | `extract_features.py` and the parser it uses; `features.md` documents every feature |
| `stats/` | 2–4 | `join_results.py` joins timings to features; `cluster.py`, `select_features.py`, `solver_outcomes.py`, `run_variance.py` and the `plot_*.py` scripts |
| `report/` | 5 | `build_report.py` and its HTML template; `paper_table.py` and `paper_router_table.py` write the paper's tables |
| `router/` | 6–7 | `crossval_router.py` cross-validates the encoding router, `train_router.py` trains it on every instance, `plot_router.py` draws its figure |

`paths.py` holds the locations the stages share.

## Rebuilding the analysis from the shipped runs

```bash
export DATESAT_TIMEOUT_MS=20000                           # the timeout the runs used
python -m analysis.features.extract_features              # 1: features.csv
python -m analysis.stats.join_results                     # 2: joined.csv
python -m analysis.stats.cluster                          # 3: clusters.json
python -m analysis.stats.plot_speedup_heatmap             # 4: plots/ (also --corpus llm|legal|grammar)
python -m analysis.stats.plot_feature_correlation         # 4: plots/feature_correlation.png
python -m analysis.stats.select_features                  # 4: selected_features.json
python -m analysis.stats.plot_feature_correlation --selected
python -m analysis.stats.plot_tsne                        # 4: plots/dataset_tsne.pdf
python -m analysis.report.build_report                    # 5: report.html
python -m analysis.router.crossval_router                 # 6: model_cross_validation/
python -m analysis.router.crossval_router --selected      # 6: model_cross_validation_selected/
python -m analysis.router.train_router                    # 7: model/
```

Stages 2 and 5 read `results/runs/encodings` and `results/runs/router` unless `--results` names other directories. 
Every stage writes to `results/analysis/`, overwriting what is there; set `DATESAT_ANALYSIS_OUTPUTS` to an existing directory to write elsewhere.
Rebuild the report after stages 6 and 7, which its router sections read.

`python -m analysis.stats.solver_outcomes --compare` prints what each encoding did per dataset.

## Running the solver (stage 0)

```bash
python -m analysis.run.run_benchmarks --runs 10 --tag my-run          # every dataset and encoding
python -m analysis.run.run_benchmarks --datesatbenchs llm --approaches alpha_beta
```

- Results go to `results/runs/<tag>/<dataset>/run_N/`; `<tag>` defaults to a timestamp, and running again with the same tag adds `run_2`, `run_3`, …
- `--timeout` (ms) bounds the Z3 check. Each instance also runs in a child process that is killed after `--hard-timeout` ms (default 20000) and recorded as a timeout.
- After the runs, every solution is checked by executing the constraints on it (`checked_summary_with_baseline.json`); `--no-analysis` skips the check. 
  The date operations are evaluated by the Verus oracle, which must be built first from the mechanized proofs that accompany DateSAT (`verus src/main.rs --compile -o build/datesat-oracle` in `mechanized_proofs/`). DateSAT looks for it in `mechanized_proofs/build/datesat-oracle` beside its checkout; set `DATESAT_VERUS_ORACLE` to a binary built elsewhere;
  the check fails if it is missing. `DATESAT_VALIDATION_SEMANTICS=python` uses DateSAT's Python `Date` class instead. 
  The summary records which one was used as `validation_semantics`.

## The router

The router picks an encoding per instance from its features: one random forest per pair of encodings.

- **Cross-validation** (`crossval_router`): 5 folds, stratified by dataset, repeated 10 times. 
  Every instance is routed by a model trained on the other folds. 
  It measures one fixed setting, `train_router`'s, and does not search for a better one. 
  It also writes every fold's model to `model_cross_validation/fold_models/`.
- **Live evaluation** (`run_router_cv`): runs DateSAT's `router` approach with those fold models, one run per repeat, so every instance is solved by a model that was not trained on it, and the time routing takes is included. These are the runs in `results/runs/router`.
- **Final model** (`train_router`): trained on every instance, written to `model/router.json` in the form DateSAT loads.

To repeat the live evaluation and fold it into the analysis:

```bash
python -m analysis.run.run_router_cv --tag my-router            # 10 runs; resumes if stopped
python -m analysis.stats.join_results --results results/runs/encodings results/runs/my-router
python -m analysis.router.crossval_router --from-fold-models    # reports the router as measured; trains nothing
python -m analysis.report.build_report --results results/runs/encodings results/runs/my-router
```

`run_router_cv` uses the shipped fold models and records their checksum; the analysis refuses router runs made with other models.

## Tests

```bash
python -m pytest                                    # run/: validation and run_router_cv; needs DateSAT and the Verus oracle
python -m analysis.features.check_emulation         # needs DateSAT; about 10 minutes
python -m analysis.features.test_features_md        # runs every example in features.md
python -m analysis.features.test_extract_features
python -m analysis.router.test_train_router
python -m analysis.router.test_crossval_router
python -m analysis.stats.test_select_features
python -m analysis.stats.test_solver_outcomes
python -m analysis.stats.test_join_results
python -m analysis.report.test_build_report
```
