"""Locations shared by the analysis pipeline."""

from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
# Everything the pipeline writes: features.csv, joined.csv, clusters.json, plots/, report.html.
OUTPUTS = ANALYSIS / "outputs"
PLOTS = OUTPUTS / "plots"
