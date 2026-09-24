"""
Cluster DateSATBench instances on the extracted feature set and project to 3D.

    python analysis/cluster.py --input analysis/features.csv --output analysis/clusters.json

Clustering is BLIND to the corpus label; agreement with corpus is measured
afterwards (adjusted Rand index) to test whether the features find structure
beyond "which generator produced this instance".

Pipeline: median-impute any missing values -> log1p on skewed counts ->
standardise -> KMeans with k chosen by silhouette, plus HDBSCAN -> PCA and
t-SNE to 3D.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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


def choose_k(Z, kmin=2, kmax=10):
    scores = {}
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        lab = km.fit_predict(Z)
        scores[k] = float(silhouette_score(Z, lab))
    best = max(scores, key=scores.get)
    return best, scores


def profile_clusters(Z, labels, feature_names, top=6):
    """For each cluster, the features whose mean deviates most from the global mean."""
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        if mask.sum() == 0:
            continue
        dev = Z[mask].mean(axis=0)          # Z is standardised, so this IS the z-score
        order = np.argsort(-np.abs(dev))[:top]
        out[str(c)] = {
            "size": int(mask.sum()),
            "drivers": [
                {"feature": feature_names[i], "z": round(float(dev[i]), 2)}
                for i in order
            ],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    ap.add_argument("--input", default=str(here / "features.csv"))
    ap.add_argument("--output", default=str(here / "clusters.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    Z, names, logged, dropped, missing = build_matrix(df)
    print(f"matrix: {Z.shape[0]} instances x {Z.shape[1]} columns")
    print(f"  log1p applied to {len(logged)} skewed columns")
    if dropped:
        print(f"  dropped {len(dropped)} zero-variance columns: {dropped}")

    # ---- clustering, blind to corpus ------------------------------------
    k, sil_scores = choose_k(Z)
    km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
    km_labels = km.fit_predict(Z)
    print(f"\nKMeans: k={k} by silhouette ({sil_scores[k]:.3f})")
    print("  silhouette by k: " + ", ".join(f"{kk}:{vv:.3f}" for kk, vv in sil_scores.items()))

    hdb = HDBSCAN(min_cluster_size=10, min_samples=5)
    hdb_labels = hdb.fit_predict(Z)
    n_hdb = len(set(hdb_labels) - {-1})
    n_noise = int((hdb_labels == -1).sum())
    print(f"HDBSCAN: {n_hdb} clusters, {n_noise} points left as noise")

    corpus = df["corpus"].values
    ari_km = adjusted_rand_score(corpus, km_labels)
    ari_hdb = adjusted_rand_score(corpus, hdb_labels)
    print(f"\nAgreement with corpus (adjusted Rand index):")
    print(f"  KMeans  {ari_km:.3f}")
    print(f"  HDBSCAN {ari_hdb:.3f}")

    # ---- 3D projections --------------------------------------------------
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

    T = TSNE(
        n_components=3, perplexity=30, init="pca",
        learning_rate="auto", random_state=RANDOM_STATE,
    ).fit_transform(Z)

    # ---- cross-tabulation ------------------------------------------------
    xtab = pd.crosstab(df["corpus"], km_labels)
    print(f"\ncorpus x kmeans cluster:\n{xtab.to_string()}")

    profiles = profile_clusters(Z, km_labels, names)
    print("\ncluster drivers (z-score vs global mean):")
    for c, p in profiles.items():
        d = ", ".join(f"{x['feature']}{x['z']:+.1f}" for x in p["drivers"][:4])
        print(f"  cluster {c} (n={p['size']:3d}): {d}")

    # ---- emit ------------------------------------------------------------
    points = []
    for i in range(len(df)):
        points.append({
            "id": df["id"].iloc[i],
            "corpus": df["corpus"].iloc[i],
            "kmeans": int(km_labels[i]),
            "hdbscan": int(hdb_labels[i]),
            "pca": [round(float(v), 4) for v in P[i]],
            "tsne": [round(float(v), 4) for v in T[i]],
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
        "kmeans_k": k,
        "silhouette_by_k": {str(kk): round(vv, 4) for kk, vv in sil_scores.items()},
        "hdbscan_clusters": n_hdb,
        "hdbscan_noise": n_noise,
        "ari_kmeans_vs_corpus": round(ari_km, 4),
        "ari_hdbscan_vs_corpus": round(ari_hdb, 4),
        "pca_explained_variance": [round(float(v), 4) for v in evr],
        "pca_loadings": loadings,
        "cluster_profiles": profiles,
        "crosstab": xtab.to_dict(),
        "points": points,
    }
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
