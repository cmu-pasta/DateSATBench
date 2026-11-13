# Results

Every run uses a 20 s timeout and is repeated 10 times on all 800 instances. The paper's
numbers come from `runs/encodings` and `runs/router`; the two `*-bounded` directories are an
earlier round kept for reference (see [The bounded runs](#the-bounded-runs)).

```
results/
├── runs/                      what the solver did
│   ├── encodings/             each of the 5 encodings on every instance
│   ├── router/                the router's live evaluation
│   ├── encodings-bounded/     earlier round: the solver in bounded mode, 6 encodings
│   └── router-bounded/        earlier round: the router's live evaluation on it
└── analysis/                  what analysis/ derived from the runs
```

## `runs/`

```
runs/<encodings|router>/
├── run_config.json                          timeout, approaches, DateSAT version, machine
└── <llm|legal|grammar>/run_<1..10>/
    ├── <approach>_int.json                  per instance: status, time, solution
    └── checked_summary_with_baseline.json   every solution checked by concrete execution
```

The approaches in `encodings/` are `simple`, `epoch_days`, `hybrid_init_ymd`, `hybrid_init_epoch` and `alpha_beta`; `router/` has `router` alone. 
In run N of `router/`, each instance is routed by a model that was not trained on it (see `analysis/README.md`).

Each `checked_summary_with_baseline.json` records, under `validation_semantics`, how the date operations of the solutions were evaluated. 
It is `verus` in every run: date validity, ordering and Date + Period are answered by the Verus oracle (`mechanized_proofs/src/oracle.rs`, with DateSAT), the executable reference semantics; 
Boolean and integer subterms are evaluated in Python. See `analysis/README.md`.

### The bounded runs

`encodings-bounded/` and `router-bounded/` have the same layout and the same 800 instances,
but are not what the paper reports and are not read by any analysis default:

| | `encodings/`, `router/` (the paper) | `encodings-bounded/`, `router-bounded/` |
|---|---|---|
| Solver mode | unbounded: only the bounds written in each instance apply | bounded: DateSAT also restricts every date, intermediate results included, to years 1..9999 (what `--python-bounds` selects today) |
| Encodings | the 5 of the paper | those 5 and `hybrid_both`, which has since been removed from DateSAT and from `analysis/` |
| DateSAT commit | `677e3bc` (encodings), `320d4b9` (router) | `28d8186` (encodings), `23e1c25` (router) |
| Original tag | `all-10runs-unbounded-datesat677e3bc`, `router-cv-10runs-unbounded` | `all-10runs-bounded-datesat28d8186`, `router-cv-10runs` |
| Validation | by the Verus oracle; `validation_semantics` recorded | by DateSAT's Python `Date` class; `validation_semantics` not recorded |
| Fold models (router) | the shipped ones in `analysis/model_cross_validation/` | an earlier set that is no longer shipped (checksum in `run_config.json`) |

The bounded runs predate the `solver_mode` and `python_bounds` fields, so their files do not
record the mode, and the `results_dir` in their `checked_summary_with_baseline.json` still
names the original tag. To analyse them, pass `--results results/runs/encodings-bounded`
(with `--exclude hybrid_both`) and set `DATESAT_ANALYSIS_OUTPUTS` to another directory so
`analysis/` is left alone. `router-bounded` cannot be joined with the shipped
cross-validation: the analysis refuses router runs made with other fold models.

## `analysis/`

| File | What it is |
|---|---|
| `features.csv`, `features_meta.json` | The features of every instance |
| `selected_features.json` | One feature per family of correlated features |
| `joined.csv` | One row per instance, encoding and run: time, status, speedup, features |
| `clusters.json` | PCA and t-SNE projections of the features |
| `plots/` | The figures (`dataset_tsne` is the paper's Fig. 6b, `router` its Fig. 7) |
| `outcomes_table.tex`, `outcomes_table_cactus_*.pdf` | The paper's Table 1 and the plots in its last column |
| `router_solve_table.tex` | The solve rates of Fig. 7b as a table |
| `report.html` | A standalone interactive report of all of the above; open it in a browser |
| `model/` | The router trained on every instance |
| `model_cross_validation/` | The router's cross-validation: results (`router_eval.*`) and the per-fold models the live evaluation ran with |
| `model_cross_validation_selected/` | The same on the selected features only |
