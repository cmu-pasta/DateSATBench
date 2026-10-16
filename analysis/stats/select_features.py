"""
Pick a smaller set of features, one per family of strongly correlated features.

    python -m analysis.stats.select_features

The same families that plot_feature_correlation.py outlines: complete-linkage clustering
on 1 - |Spearman rho| over every instance, cut so that every pair inside a family has
|rho| >= --threshold. Each family keeps one feature, its medoid: the one with the highest
mean |rho| to the rest of the family, so the kept feature is the one that best stands in
for the others (a tie goes to the feature that comes first in features.csv). Constant
features are dropped outright.

Two kept features from different families can still reach the threshold, because
complete linkage bounds the pairs inside a family, not between them. So the clustering
is repeated on the kept features until no pair of them has |rho| >= --threshold; a
family then also holds the features its members had absorbed.

Only the features are used, never the solver results, so a cross-validation run on the
selected features (crossval_router.py --selected) is not biased by the choice.

Writes analysis/outputs/selected_features.json:
    threshold, the number of features in features.csv, the constant ones, every family
    of more than one feature (the kept feature, and each dropped one with its signed rho
    to the kept one), the selected features in features.csv order, and the most
    correlated pair left among them.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import OUTPUTS, REPO, SELECTED_FEATURES
from analysis.stats.plot_feature_correlation import constant_features, feature_correlation

DEFAULT_THRESHOLD = 0.8        # the family boxes of plot_feature_correlation.py


def feature_names(df):
    """The feature columns of features.csv, in order."""
    return [c for c in df.columns if c not in ("id", "corpus") and not c.startswith("label_")]


def medoid(family, rho):
    """The member of `family` with the highest mean |rho| to the other members; the first
    one on a tie. `rho` is a matrix of |rho| over at least the family's features."""
    score = [rho.loc[f, [g for g in family if g != f]].mean() for f in family]
    return family[int(np.argmax(score))]


def select_features(df, threshold=DEFAULT_THRESHOLD):
    """One feature per family of features whose every pair has |rho| >= `threshold`.

    Returns (selected, families, constant): the kept features in column order, a dict
    kept feature -> the features it stands in for (in column order), and the constant
    features that were dropped.
    """
    names = feature_names(df)
    constant = constant_features(df)
    kept = [c for c in names if c not in constant]
    absorbed = {f: [] for f in kept}
    rho = df[kept].astype(float).corr(method="spearman").abs()
    while True:
        _, _, family_of = feature_correlation(df[kept], threshold)
        groups = {}
        for f in kept:
            groups.setdefault(family_of[f], []).append(f)
        reps = []
        for family in groups.values():
            rep = medoid(family, rho)
            for f in family:
                if f != rep:
                    absorbed[rep] += [f] + absorbed.pop(f)
            reps.append(rep)
        if len(reps) == len(kept):
            break
        kept = [c for c in kept if c in reps]
    families = {f: sorted(absorbed[f], key=names.index) for f in kept if absorbed[f]}
    return kept, families, constant


def most_correlated_pair(df, cols):
    """(a, b, rho) for the pair of `cols` with the largest |Spearman rho|."""
    rho = df[cols].astype(float).corr(method="spearman")
    r = rho.abs().values.copy()
    np.fill_diagonal(r, -1)
    i, j = np.unravel_index(np.argmax(r), r.shape)
    return cols[i], cols[j], float(rho.iat[i, j])


def selected_columns(path=SELECTED_FEATURES):
    """The selected features that select_features.py wrote to `path`."""
    return json.loads(Path(path).read_text())["selected"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(OUTPUTS / "features.csv"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="min |rho| between every pair in a family")
    ap.add_argument("--output", default=str(SELECTED_FEATURES))
    args = ap.parse_args()
    if not 0 < args.threshold <= 1:
        ap.error(f"--threshold must be in (0, 1], got {args.threshold:g}")

    df = pd.read_csv(args.features)
    names = feature_names(df)
    selected, families, constant = select_features(df, args.threshold)
    rho = df[names].astype(float).corr(method="spearman")
    a, b, r = most_correlated_pair(df, selected)

    print(f"{len(names)} features; dropped {len(constant)} constant: {', '.join(constant)}")
    print(f"{len(families)} families of |rho| >= {args.threshold:g}, "
          f"{sum(len(d) + 1 for d in families.values())} features, each kept as one:")
    for kept, dropped in families.items():
        print(f"  keep {kept}; drop " + ", ".join(f"{f} ({rho.at[kept, f]:+.2f})" for f in dropped))
    print(f"kept {len(selected)} features; the most correlated pair left is {a} and {b}, "
          f"rho {r:+.3f}")

    out = Path(args.output)
    out.write_text(json.dumps({
        "features": os.path.relpath(args.features, REPO),
        "threshold": args.threshold,
        "n_features": len(names),
        "constant": constant,
        "families": [{"kept": kept, "dropped": [{"feature": f, "rho": round(float(rho.at[kept, f]), 4)}
                                                for f in dropped]}
                     for kept, dropped in families.items()],
        "selected": selected,
        "most_correlated_left": {"a": a, "b": b, "rho": round(r, 4)},
    }, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
