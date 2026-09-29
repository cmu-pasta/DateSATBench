"""Locations shared by the analysis pipeline."""

from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
# Everything the pipeline writes: features.csv, joined.csv, clusters.json, plots/, report.html,
# model/, model_cross_validation/.
OUTPUTS = ANALYSIS / "outputs"
PLOTS = OUTPUTS / "plots"
MODEL = OUTPUTS / "model"                              # the router, trained on every instance
CROSS_VALIDATION = OUTPUTS / "model_cross_validation"  # crossval_router.py's results
