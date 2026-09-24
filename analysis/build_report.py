"""
Stage 5: build the standalone analysis report from the outputs of stages 2-4.

    python analysis/build_report.py

Reads features.csv (+ features_meta.json), joined.csv and clusters.json, computes the
solver outcomes and speedup heatmaps for every corpus (and timeout mode) and the
feature-feature correlation (same functions as solver_outcomes.py and the PNG scripts,
so the numbers match), and inlines everything into
report_template.html. Writes a single self-contained analysis/report.html. Only Plotly
and the IBM Plex webfonts are fetched from a CDN when the page opens.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import build_matrix, profile_clusters
from extract_features import FEATURE_GROUPS
from plot_feature_correlation import constant_features, feature_correlation
from plot_speedup_heatmap import speedup_correlations
from solver_outcomes import COUNTS as OUTCOME_COUNTS, solver_outcomes

HERE = Path(__file__).parent
REPO = HERE.parent
CORPORA = ["llm", "legal", "grammar"]


def clean(x, nd=4):
    """JSON-safe number: NaN -> None, floats rounded."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, float):
        return round(x, nd)
    return x


def frame(df, nd=4):
    return {r: {c: clean(float(df.loc[r, c]), nd) for c in df.columns} for r in df.index}


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(HERE / "features.csv"))
    ap.add_argument("--joined", default=str(HERE / "joined.csv"))
    ap.add_argument("--clusters", default=str(HERE / "clusters.json"))
    ap.add_argument("--results", default=str(REPO / "results/bench-datetime-bound"))
    ap.add_argument("--baseline", default="simple")
    ap.add_argument("--template", default=str(HERE / "report_template.html"))
    ap.add_argument("--output", default=str(HERE / "report.html"))
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    joined = pd.read_csv(args.joined)
    clusters = json.loads(Path(args.clusters).read_text())
    meta_path = Path(args.features).with_name(Path(args.features).stem + "_meta.json")
    fmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    run_cfg_path = Path(args.results) / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}

    encodings = sorted(joined["encoding"].unique())
    others = [e for e in encodings if e != args.baseline]

    # ---- outcomes per scope x encoding (same function as the CLI) --------
    outcomes = {}
    for c in ["all"] + CORPORA:
        t = solver_outcomes(joined, args.baseline, c)
        outcomes[c] = {
            e: {k: None if pd.isna(v) else int(v) if k in OUTCOME_COUNTS else clean(float(v), 6)
                for k, v in row.items()}
            for e, row in t.to_dict(orient="index").items()
        }

    # ---- speedup heatmaps: every scope x timeout mode -------------------
    heat = {}
    for mode in ("bound", "drop"):
        heat[mode] = {}
        for c in [None] + CORPORA:
            rho, q, n = speedup_correlations(joined, args.baseline, c, mode)
            heat[mode][c or "all"] = {"rho": frame(rho), "q": frame(q, 6), "n": n}

    # ---- feature-feature correlation ------------------------------------
    rho, order, _ = feature_correlation(feats)      # order: related features sit together
    corr = {
        "order": order,
        "z": [[clean(float(v), 3) for v in row] for row in rho.values],
        "constant": constant_features(feats),     # left out of the matrix
    }

    # ---- clusters --------------------------------------------------------
    cl = {
        k: clusters[k] for k in (
            "kmeans_k", "silhouette_by_k", "hdbscan_clusters", "hdbscan_noise",
            "ari_kmeans_vs_corpus", "ari_hdbscan_vs_corpus", "pca_explained_variance",
            "pca_loadings", "cluster_profiles",
        )
    }
    # Same drivers cluster.py computes for K-means, for the HDBSCAN labels too.
    Z, names, *_ = build_matrix(feats)
    by_id = {p["id"]: p["hdbscan"] for p in clusters["points"]}
    cl["hdbscan_profiles"] = profile_clusters(Z, np.array([by_id[i] for i in feats["id"]]), names)
    cl["points"] = [
        {"i": p["id"], "c": p["corpus"], "k": p["kmeans"], "h": p["hdbscan"],
         "p": p["pca"], "t": p["tsne"]}
        for p in clusters["points"]
    ]

    data = {
        "meta": {
            "generated": date.today().isoformat(),
            "bounds": fmeta.get("bounds", "unknown"),
            "dataset_root": rel(fmeta["dataset_root"]) if "dataset_root" in fmeta else None,
            "results": rel(args.results),
            "runs": int(joined["run"].nunique()),
            "timeout_s": (run_cfg.get("timeout_ms") or 0) / 1000 or None,
            "baseline": args.baseline,
            "encodings": others,
            "corpora": {c: int((feats["corpus"] == c).sum()) for c in CORPORA},
            "n_instances": len(feats),
            "n_rows": len(joined),
        },
        "groups": FEATURE_GROUPS,
        "outcomes": outcomes,
        "heat": heat,
        "corr": corr,
        "clusters": cl,
    }

    template = Path(args.template).read_text()
    if "__DATA__" not in template:
        raise SystemExit(f"{args.template} has no __DATA__ placeholder")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    out = Path(args.output)
    out.write_text(template.replace("__DATA__", payload))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
