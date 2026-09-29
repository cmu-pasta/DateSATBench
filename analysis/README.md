# Analysis

Relates DateSAT's solver results on DateSATBench to features of the benchmark instances.

## Layout

| Folder | Stage | What it holds |
|---|---|---|
| `features/` | 1 | The constraint parser, `extract_features.py`, `features.md` (documents every feature column) and its tests |
| `stats/` | 2–4 | `join_results.py` (joins solver timings to the features), `cluster.py`, `solver_outcomes.py` and the two `plot_*.py` scripts |
| `report/` | 5 | `build_report.py` and the HTML template it fills in |
| `outputs/` | | Everything the stages write: `features.csv`, `features_meta.json`, `joined.csv`, `clusters.json`, `plots/`, `report.html` |

`paths.py` holds the locations the stages share.

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
```

`python -m analysis.stats.solver_outcomes --compare` prints a summary of what each encoding
did, per corpus. Every stage takes `--help`.

## Tests

```
python -m analysis.features.test_features_md        # runs every example in features.md
python -m analysis.features.test_extract_features
```
