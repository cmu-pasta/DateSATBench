# DateSATBench

The DateSATBench benchmark, the code that evaluates the
[DateSAT](https://github.com/cmu-pasta/DateSAT) solver on it, and the results reported in
the paper.

| Folder | What it holds | Details |
|---|---|---|
| `datesatbench/` | The three benchmark datasets (800 instances) and the generators that produced them | `datesatbench/README.md` |
| `analysis/` | Code that runs DateSAT on the datasets, extracts features, trains and evaluates the encoding router, and builds the plots and report | `analysis/README.md` |
| `results/` | The solver runs behind the paper and everything the analysis derived from them | `results/README.md` |

## Setup

All commands run from this directory.

```bash
pip install -e .               # the datasets (no dependencies)
pip install -e ".[analysis]"   # plus what analysis/ needs
pip install -e ".[llm]"        # plus what the LLM-based generators need
```

Running the solver also needs DateSAT itself, which lives in a separate repository: clone
[DateSAT](https://github.com/cmu-pasta/DateSAT) and run
`pip install -e <path to DateSAT checkout>`. Reading the results and rebuilding the tables, plots and
report from them does not.
