"""
Stage 5: build the standalone analysis report from the outputs of stages 2-4.

    python -m analysis.report.build_report --timeout 60000     # or set DATESAT_TIMEOUT_MS

Reads features.csv (+ features_meta.json), joined.csv and clusters.json, computes the
solver outcomes and speedup heatmaps for every corpus (and timeout mode) and the
feature-feature correlation (same functions as solver_outcomes.py and the PNG scripts,
so the numbers match), and inlines everything into
report_template.html. The specialized-router section comes from the outputs of stages 6
and 7: crossval_router.py's router_eval.json and train_router.py's router_meta.json; it
is left out when crossval_router.py has not been run. Writes a single self-contained analysis/outputs/report.html. Only Plotly
and the IBM Plex webfonts are fetched from a CDN when the page opens.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.features.extract_features import FEATURE_GROUPS
from analysis.paths import CROSS_VALIDATION, MODEL, OUTPUTS, REPO
from analysis.stats.cluster import build_matrix, profile_clusters
from analysis.stats.join_results import add_timeout_arg, check_run_config, timeout_seconds
from analysis.stats.plot_feature_correlation import constant_features, feature_correlation
from analysis.stats.plot_speedup_heatmap import speedup_correlations
from analysis.stats.solver_outcomes import COUNTS as OUTCOME_COUNTS, solver_outcomes

HERE = Path(__file__).parent
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


def router_section(cv_path, meta_path):
    """The router's cross-validated results (crossval_router.py) and the settings of the
    model trained on every instance (train_router.py), or None when there are no
    cross-validated results. `matches_cv` says whether the model was trained with the
    setting the cross-validation chose."""
    cv_path, meta_path = Path(cv_path), Path(meta_path)
    if not cv_path.exists():
        return None
    cv = json.loads(cv_path.read_text())
    scopes = {"all": cv["overall"], **cv["by_corpus"]}
    out = {
        "cv": {k: cv[k] for k in ("instances", "dropped", "folds", "repeats", "trees", "setting",
                                  "fallbacks", "baseline", "best_performing_encoding",
                                  "timeout_cost_s")},
        "scopes": {c: {"instances": s["instances"], "table": s["table"],
                       "picks": s["router_picks"], "fell_back": s["router_fell_back"],
                       "picks_fastest": s["router_picks_fastest"],
                       "gap_closed": s["gap_closed_to_oracle"]["router"].get(
                           f"always {cv['best_performing_encoding']}")}
                   for c, s in scopes.items()},
        "model": None,
    }
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        out["model"] = {
            "encodings": m["encodings"], "trees": m["trees"], "max_depth": m["max_depth"],
            "min_samples_leaf": m["min_samples_leaf"], "margin": m["margin"],
            "fallback": m["fallback"], "trained_on": m["instances"]["trained_on"],
            "n_features": len(m["feature_columns"]),
            "pairs": [{"pair": pair, "top": [t["feature"] for t in p["top_features"][:2]]}
                      for pair, p in m["pairs"].items()],
            "matches_cv": all(m[k] == v for k, v in cv["setting"].items()) and m["trees"] == cv["trees"],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(OUTPUTS / "features.csv"))
    ap.add_argument("--joined", default=str(OUTPUTS / "joined.csv"))
    ap.add_argument("--clusters", default=str(OUTPUTS / "clusters.json"))
    ap.add_argument("--results", default=str(REPO / "results/bench-datetime-bound"))
    ap.add_argument("--baseline", default="simple")
    add_timeout_arg(ap)
    ap.add_argument("--router-cv", default=str(CROSS_VALIDATION / "router_eval.json"),
                    help="crossval_router.py's results")
    ap.add_argument("--router-model", default=str(MODEL / "router_meta.json"),
                    help="train_router.py's model settings")
    ap.add_argument("--template", default=str(HERE / "report_template.html"))
    ap.add_argument("--output", default=str(OUTPUTS / "report.html"))
    args = ap.parse_args()
    timeout_s = timeout_seconds(ap, args)
    check_run_config(args.results, timeout_s)

    feats = pd.read_csv(args.features)
    joined = pd.read_csv(args.joined)
    clusters = json.loads(Path(args.clusters).read_text())
    meta_path = Path(args.features).with_name(Path(args.features).stem + "_meta.json")
    fmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

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
            "timeout_s": timeout_s,
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
        "router": router_section(args.router_cv, args.router_model),
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
