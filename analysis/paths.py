"""Locations shared by the analysis pipeline."""

import os
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
RESULTS = REPO / "results"
# The solver's runs, one directory per tag: encodings/ (every encoding, 10 runs) and
# router/ (the router's live evaluation, 10 runs).
RUNS = RESULTS / "runs"
DEFAULT_RUNS = [RUNS / "encodings", RUNS / "router"]
# Everything the pipeline writes: features.csv, selected_features.json, joined.csv,
# clusters.json, plots/, report.html, model/, model_cross_validation/,
# model_cross_validation_selected/. $DATESAT_ANALYSIS_OUTPUTS puts them elsewhere, so an
# analysis of other results leaves the shipped outputs alone.
OUTPUTS = Path(os.environ.get("DATESAT_ANALYSIS_OUTPUTS") or RESULTS / "analysis").resolve()
PLOTS = OUTPUTS / "plots"
MODEL = OUTPUTS / "model"                              # the router, trained on every instance
CROSS_VALIDATION = OUTPUTS / "model_cross_validation"  # crossval_router.py's results
SELECTED_FEATURES = OUTPUTS / "selected_features.json" # select_features.py's pick
CROSS_VALIDATION_SELECTED = OUTPUTS / "model_cross_validation_selected"  # ... with --selected
