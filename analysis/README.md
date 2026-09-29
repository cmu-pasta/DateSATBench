# Analysis

Relates DateSAT's solver results on DateSATBench to features of the benchmark instances.

## Layout

| Folder | Stage | What it holds |
|---|---|---|
| `features/` | 1 | The constraint parser, `extract_features.py`, `features.md` (documents every feature column) and its tests |
| `stats/` | 2–4 | `join_results.py` (joins solver timings to the features), `cluster.py`, `solver_outcomes.py` and the two `plot_*.py` scripts |
| `report/` | 5 | `build_report.py` and the HTML template it fills in |
| `router/` | 6–7 | `crossval_router.py` (chooses the router's leaf size, depth and margin, and measures it on unseen constraints, by cross-validation), `train_router.py` (trains the encoding router on every instance: one random forest per pair of encodings) and their tests |
| `outputs/` | | Everything the stages write: `features.csv`, `features_meta.json`, `joined.csv`, `clusters.json`, `plots/`, `report.html`, `model/` (the router trained on every instance), `model_cross_validation/` (the cross-validation's results) |

`paths.py` holds the locations the stages share.

`outputs/model/router.json` is the model for DateSat's `router` approach: copy it to `datesat/router/model.json` in the DateSat repo, or point `DATESAT_ROUTER_MODEL` at it. DateSat only accepts a model whose features kept the injected bounds (`extract_features --bounds keep`).

## Running the pipeline

`analysis` is a package, so run each stage as a module from the repository root:

```
export DATESAT_TIMEOUT_MS=60000                          # the --timeout the results were run with
python -m analysis.features.extract_features             # 1: outputs/features.csv
python -m analysis.stats.join_results                    # 2: outputs/joined.csv
python -m analysis.stats.cluster                         # 3: outputs/clusters.json
python -m analysis.stats.plot_speedup_heatmap            # 4: outputs/plots/ (and --corpus llm|legal|grammar)
python -m analysis.stats.plot_feature_correlation        # 4: outputs/plots/feature_correlation.png
python -m analysis.report.build_report                   # 5: outputs/report.html
python -m analysis.router.crossval_router                # 6: outputs/model_cross_validation/router_tuning.csv, router_eval.json, router_eval.csv
python -m analysis.router.train_router                   # 7: outputs/model/router.joblib, router_meta.json, router.json
```

`python -m analysis.stats.solver_outcomes --compare` prints a summary of what each encoding
did, per corpus. Every stage takes `--help`.

The report's *Specialized router* section reads the outputs of stages 6 and 7
(`model_cross_validation/router_eval.json` and `model/router_meta.json`), so rebuild the
report after running them; without them the section says so and stays empty.

## How well the router does on unseen constraints

`crossval_router` measures the router on constraints it has not seen: 5-fold
cross-validation over every instance, stratified by corpus and repeated 3 times, in which
every instance is routed by a router trained on the other folds. The same
cross-validation chooses the leaf size, depth and margin, and `train_router`'s defaults
(`DEFAULT_SETTINGS`) should be the setting it chose; it says so when they are not. Its
results go to `model_cross_validation/`; the model DateSat uses, trained on every
instance with those settings, goes to `model/`.

## Tests

```
python -m analysis.features.test_features_md        # runs every example in features.md
python -m analysis.features.test_extract_features
python -m analysis.router.test_train_router
python -m analysis.router.test_crossval_router
python -m analysis.report.test_build_report
```
