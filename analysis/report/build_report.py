"""
Stage 5: build the standalone analysis report from the outputs of stages 2-4.

    python -m analysis.report.build_report --timeout 60000     # or set DATESAT_TIMEOUT_MS

Reads features.csv (+ features_meta.json), joined.csv and clusters.json (the PCA and
t-SNE projections), computes the
solver outcomes and speedup heatmaps for every corpus (and timeout mode), all from the one
measurement of instance_costs.py that the router uses too, and the
feature-feature correlation (same functions as solver_outcomes.py and the PNG scripts,
so the numbers match), and inlines everything into
report_template.html. The specialized-router section comes from the outputs of stages 6
and 7: crossval_router.py's router_eval.json (and, for how often each encoding is the
fastest, the router_eval.csv beside it) and train_router.py's router_meta.json; it is
left out when crossval_router.py has not been run. The feature-reduction section comes
from select_features.py's selected_features.json and, for the router with those features,
from crossval_router.py --selected; it is left out when select_features.py has not been
run. Writes a single self-contained analysis/outputs/report.html. Only Plotly
and the IBM Plex webfonts are fetched from a CDN when the page opens.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from analysis.features.extract_features import FEATURE_GROUPS
from analysis.paths import (CROSS_VALIDATION, CROSS_VALIDATION_SELECTED, MODEL, OUTPUTS, REPO,
                            SELECTED_FEATURES)
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


def fastest_counts(csv_path, encodings):
    """How many instances each encoding is the fastest on, for "all" and for every corpus,
    from crossval_router.py's router_eval.csv (its `best` column is the same in every
    repeat), or None when there is no such file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    log = pd.read_csv(csv_path)
    log = log[log["repeat"] == log["repeat"].min()]
    scopes = {"all": log, **{c: g for c, g in log.groupby("corpus")}}
    return {c: {e: int((g["best"] == e).sum()) for e in encodings} for c, g in scopes.items()}


def solve_rates(summary):
    """Instances each strategy of a crossval_router.py summary solves within the timeout,
    and their share: the instances less its timeouts, which for the router are averaged
    over the repeats."""
    n = summary["instances"]
    return {name: {"solved": clean(float(n - t["timeouts"])),
                   "rate": clean(float((n - t["timeouts"]) / n))}
            for name, t in summary["table"].items()}


def router_section(cv_path, meta_path):
    """The router's cross-validated results (crossval_router.py) and the settings of the
    model trained on every instance (train_router.py), or None when there are no
    cross-validated results. `matches_cv` says whether the model was trained with the
    setting the cross-validation measured. Each scope also gets every strategy's solve
    rate and, from router_eval.csv beside the results, how many instances each encoding
    is the fastest on (None without that file)."""
    cv_path, meta_path = Path(cv_path), Path(meta_path)
    if not cv_path.exists():
        return None
    cv = json.loads(cv_path.read_text())
    scopes = {"all": cv["overall"], **cv["by_corpus"]}
    encodings = sorted(n[len("always "):] for n in cv["overall"]["table"] if n.startswith("always "))
    fastest = fastest_counts(cv_path.with_name("router_eval.csv"), encodings)
    out = {
        "cv": {k: cv[k] for k in ("instances", "dropped", "folds", "repeats", "trees", "setting",
                                  "fallbacks", "baseline", "best_performing_encoding",
                                  "timeout_cost_s")},
        "scopes": {c: {"instances": s["instances"], "table": s["table"],
                       "solve": solve_rates(s),
                       "picks": s["router_picks"], "fell_back": s["router_fell_back"],
                       "fastest": fastest.get(c) if fastest else None,
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


def reduction_section(selection_path, feats, cv_all_path, cv_selected_path):
    """The features select_features.py kept, the correlation left among them (ordered as
    in the full matrix), and the router cross-validated with every feature and with the
    selected ones, or None when there is no selection.

    `router` is None unless both cross-validations exist. `same_setting` says whether
    they measured the same setting on the same instances, and `matches_selection`
    whether the one with the selected features used exactly this selection.
    """
    selection_path = Path(selection_path)
    if not selection_path.exists():
        return None
    sel = json.loads(selection_path.read_text())
    rho, order, _ = feature_correlation(feats[sel["selected"]])
    out = {k: sel[k] for k in ("threshold", "n_features", "constant", "families", "selected",
                               "most_correlated_left")}
    out["corr"] = {"order": order, "z": [[clean(float(v), 3) for v in row] for row in rho.values]}
    out["router"] = None
    cv_all_path, cv_selected_path = Path(cv_all_path), Path(cv_selected_path)
    if not (cv_all_path.exists() and cv_selected_path.exists()):
        return out
    runs = {"all": json.loads(cv_all_path.read_text()),
            "selected": json.loads(cv_selected_path.read_text())}
    a, s = runs["all"], runs["selected"]
    best = a["best_performing_encoding"]

    def scope(summary):
        t = summary["table"]
        return {"instances": summary["instances"], "router": t["router"],
                "best": t[f"always {best}"], "oracle": t["oracle"],
                "router_total_sd": summary["router_total_sd_s"],
                "picks_fastest": summary["router_picks_fastest"],
                "gap_closed": summary["gap_closed_to_oracle"]["router"].get(f"always {best}")}

    out["router"] = {
        "best": best,
        "baseline": a["baseline"],
        "folds": a["folds"],
        "repeats": a["repeats"],
        "n_features": {k: len(r.get("feature_columns", [])) or None for k, r in runs.items()},
        "same_setting": all(a[k] == s[k] for k in ("instances", "folds", "repeats", "trees",
                                                   "setting", "timeout_cost_s")),
        "matches_selection": sorted(s.get("feature_columns", [])) == sorted(sel["selected"]),
        "scopes": {c: {k: scope(r["overall"] if c == "all" else r["by_corpus"][c])
                       for k, r in runs.items()}
                   for c in ["all", *a["by_corpus"]]},
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
    ap.add_argument("--selected-features", default=str(SELECTED_FEATURES),
                    help="select_features.py's selection")
    ap.add_argument("--router-cv-selected", default=str(CROSS_VALIDATION_SELECTED / "router_eval.json"),
                    help="crossval_router.py --selected's results")
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
        t = solver_outcomes(joined, timeout_s, args.baseline, c)
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
            rho, q, n = speedup_correlations(joined, timeout_s, args.baseline, c, mode)
            heat[mode][c or "all"] = {"rho": frame(rho), "q": frame(q, 6), "n": n}

    # ---- feature-feature correlation ------------------------------------
    rho, order, _ = feature_correlation(feats)      # order: related features sit together
    corr = {
        "order": order,
        "z": [[clean(float(v), 3) for v in row] for row in rho.values],
        "constant": constant_features(feats),     # left out of the matrix
    }

    # ---- 2D and 3D projections -------------------------------------------
    cl = {
        "pca_explained_variance": clusters["pca_explained_variance"],
        "points": [
            {"i": p["id"], "c": p["corpus"], "p": p["pca"], "t2": p["tsne_2d"], "t3": p["tsne_3d"]}
            for p in clusters["points"]
        ],
    }

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
        "reduction": reduction_section(args.selected_features, feats, args.router_cv,
                                       args.router_cv_selected),
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
