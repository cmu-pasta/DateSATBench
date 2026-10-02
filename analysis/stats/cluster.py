"""
Project DateSATBench instances from the extracted feature set to 2D and 3D.

    python -m analysis.stats.cluster --input analysis/outputs/features.csv --output analysis/outputs/clusters.json

The projections are BLIND to the corpus label; the report colours the points
by corpus afterwards, to show whether the features tell the generators apart.

Pipeline: median-impute any missing values -> log1p on skewed counts ->
standardise -> PCA to 3 components (the 2D view uses the first two) and t-SNE
to 2D and to 3D. The two t-SNEs are fitted separately, because a 2D t-SNE is
not the first two axes of a 3D one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from analysis.paths import OUTPUTS

SKEW_THRESHOLD = 2.0
RANDOM_STATE = 0


def build_matrix(df):
    drop = {"id", "corpus", "label_status", "label_execution_time"}
    cols = [c for c in df.columns if c not in drop]
    X = df[cols].apply(pd.to_numeric, errors="coerce")

    missing = X.isna().sum()
    X = X.fillna(X.median())

    logged = []
    for c in cols:
        v = X[c]
        if v.min() >= 0 and v.skew() > SKEW_THRESHOLD:
            X[c] = np.log1p(v)
            logged.append(c)

    keep = [c for c in cols if X[c].std() > 1e-9]
    dropped_constant = [c for c in cols if c not in keep]
    X = X[keep]

    Z = StandardScaler().fit_transform(X.values)
    return Z, keep, logged, dropped_constant, missing[missing > 0].to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(OUTPUTS / "features.csv"))
    ap.add_argument("--output", default=str(OUTPUTS / "clusters.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    Z, names, logged, dropped, missing = build_matrix(df)
    print(f"matrix: {Z.shape[0]} instances x {Z.shape[1]} columns")
    print(f"  log1p applied to {len(logged)} skewed columns")
    if dropped:
        print(f"  dropped {len(dropped)} zero-variance columns: {dropped}")

    # ---- projections, blind to corpus ------------------------------------
    # PCA's components are nested, so the 2D view is the first two of these three.
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    P = pca.fit_transform(Z)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA variance explained: "
          f"PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%} "
          f"(total {evr[:3].sum():.1%})")

    loadings = {}
    for i in range(3):
        order = np.argsort(-np.abs(pca.components_[i]))[:6]
        loadings[f"PC{i+1}"] = [
            {"feature": names[j], "loading": round(float(pca.components_[i][j]), 3)}
            for j in order
        ]
        top = ", ".join(f"{names[j]}({pca.components_[i][j]:+.2f})" for j in order[:4])
        print(f"  PC{i+1}: {top}")

    T2, T3 = (TSNE(
        n_components=n, perplexity=30, init="pca",
        learning_rate="auto", random_state=RANDOM_STATE,
    ).fit_transform(Z) for n in (2, 3))

    # ---- emit ------------------------------------------------------------
    points = []
    for i in range(len(df)):
        points.append({
            "id": df["id"].iloc[i],
            "corpus": df["corpus"].iloc[i],
            "pca": [round(float(v), 4) for v in P[i]],
            "tsne_2d": [round(float(v), 4) for v in T2[i]],
            "tsne_3d": [round(float(v), 4) for v in T3[i]],
            "status": (None if pd.isna(df["label_status"].iloc[i])
                       else df["label_status"].iloc[i]),
            "time": (None if pd.isna(df["label_execution_time"].iloc[i])
                     else round(float(df["label_execution_time"].iloc[i]), 3)),
            "n_free_date_vars": int(df["n_free_date_vars"].iloc[i]),
            "n_atoms": int(df["n_atoms"].iloc[i]),
            "max_abs_days": int(df["max_abs_days"].iloc[i]),
        })

    result = {
        "n_instances": len(df),
        "n_columns": Z.shape[1],
        "logged_columns": logged,
        "dropped_constant_columns": dropped,
        "missing_counts": missing,
        "pca_explained_variance": [round(float(v), 4) for v in evr],
        "pca_loadings": loadings,
        "points": points,
    }
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
