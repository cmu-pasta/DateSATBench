# Analysis

Relates DateSAT's solver results on DateSATBench to features of the benchmark instances.

## Layout

| Folder | Stage | What it holds |
|---|---|---|
| `features/` | 1 | The constraint parser, `extract_features.py`, `emulate_encodings.py` (counts, without solving, what each DateSat encoding would emit), `check_emulation.py` (compares those counts with the formulas DateSat builds), `features.md` (documents every feature column) and its tests |
| `stats/` | 2–4 | `join_results.py` (joins solver timings to the features), `cluster.py` (PCA and t-SNE projections of the features to 2D and 3D), `solver_outcomes.py`, the two `plot_*.py` scripts and `select_features.py` (one feature per family of correlated features) |
| `report/` | 5 | `build_report.py` and the HTML template it fills in |
| `router/` | 6–7 | `crossval_router.py` (measures the router's setting on unseen constraints, by cross-validation), `train_router.py` (trains the encoding router on every instance: one random forest per pair of encodings) and their tests |
| `outputs/` | | Everything the stages write: `features.csv`, `features_meta.json`, `selected_features.json`, `joined.csv`, `clusters.json`, `plots/`, `report.html`, `model/` (the router trained on every instance), `model_cross_validation/` (the cross-validation's results) and `model_cross_validation_selected/` (the same on the selected features only) |

`paths.py` holds the locations the stages share.

`outputs/model/router.json` is the model for DateSat's `router` approach: copy it to `datesat/router/model.json` in the DateSat repo, or point `DATESAT_ROUTER_MODEL` at it. DateSat only accepts a model whose features kept the injected bounds (`extract_features --bounds keep`).

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
python -m analysis.report.build_report                   # 5: outputs/report.html
python -m analysis.router.crossval_router                # 6: outputs/model_cross_validation/router_eval.json, router_eval.csv
python -m analysis.router.crossval_router --selected     # 6: the same on the selected features, in outputs/model_cross_validation_selected/
python -m analysis.router.train_router                   # 7: outputs/model/router.joblib, router_meta.json, router.json
```

`python -m analysis.stats.solver_outcomes --compare` prints a summary of what each encoding
did, per corpus. Every stage takes `--help`.

To analyse other results without overwriting `outputs/`, point `--results` (stages 2 and 5)
at them and set `DATESAT_ANALYSIS_OUTPUTS` to another directory, which every stage then
reads and writes instead. Create the directory first. For example,
`outputs-combined-6496e/` is the pipeline run on
`results/combined-2runs-bounded-datesat6496e` with
`DATESAT_ANALYSIS_OUTPUTS=analysis/outputs-combined-6496e` and
`extract_features --dataset-root datesatbench --bounds keep`.

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
cross-validation over every instance, stratified by corpus and repeated 5 times, in which
every instance is routed by a router trained on the other folds. It measures one fixed
setting, `train_router`'s (`DEFAULT_SETTINGS` and `DEFAULT_TREES`), and does not search
for a better one; pass `--min-samples-leaf`, `--max-depth`, `--margin` or `--trees` to
measure another. Its results go to `model_cross_validation/`; the model DateSat uses,
trained on every instance with the same setting, goes to `model/`.

## Tests

```
python -m analysis.features.test_features_md        # runs every example in features.md
python -m analysis.features.test_extract_features
python -m analysis.features.check_emulation         # needs DateSat; takes about 10 minutes
python -m analysis.router.test_train_router
python -m analysis.router.test_crossval_router
python -m analysis.stats.test_select_features
python -m analysis.report.test_build_report
```
