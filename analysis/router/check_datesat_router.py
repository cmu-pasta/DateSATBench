"""
Check that DateSat's router routes as the cross-validation predicts.

    python -m analysis.router.check_datesat_router     # needs DateSat installed, not pandas

Two checks, over every instance of the datasets as DateSat sees them (the injected bounds
kept, as the features were extracted):

  * features: DateSat's copy of the feature extractor (datesat/router/features.py and
    parser.py) gives every feature the models read the value DateSATBench's
    extract_features.py gives it;
  * picks: with every fold model crossval_router.py wrote (fold_models.json lists them),
    DateSat's router picks for every instance of the fold the encoding fold_models.json
    records, the pick of scikit-learn's forests.

run_router_cv.py runs both before it solves anything, and stops on any difference. A
difference means DateSat's copy of the extractor or its model reader has drifted from
DateSATBench's: copy extract_features.py and datesat_parser.py over DateSat's again (see the
note at the top of each), or rerun crossval_router.py if the fold models changed since.

It also says whether DateSat's bundled model, datesat/router/model.json, is the router
train_router.py trained on every instance (outputs/model/router.json). The live evaluation
does not use it.
"""

import argparse
import hashlib
import json
from pathlib import Path

from analysis.features.extract_features import DATASET_ROOT, DATASETS, features_for, load
from analysis.paths import CROSS_VALIDATION, MODEL

SHOWN = 20          # problems printed at most


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_entries(dataset_root=DATASET_ROOT):
    """Every instance of the datasets as DateSat sees it, with its corpus: (corpus, entry)."""
    for corpus, rel in DATASETS.items():
        for entry in load(Path(dataset_root) / rel):
            yield corpus, entry


def check(manifest_path, dataset_root=DATASET_ROOT):
    """The differences found between DateSat's router and the cross-validation, one string
    each; none when DateSat routes every instance as fold_models.json predicts."""
    from datesat import router
    from datesat.router.features import features_for as datesat_features_for

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    problems, entries = [], {}
    for corpus, entry in dataset_entries(dataset_root):
        entries[entry["id"]] = entry
        ours, datesats = features_for(entry, corpus), datesat_features_for(entry, "")
        differ = [f for f in manifest["features"] if ours[f] != datesats.get(f)]
        if differ:
            problems.append(f"{entry['id']}: DateSat's features differ on {', '.join(differ)}")

    for repeat, folds in enumerate(manifest["models"]):
        for fold, m in enumerate(folds):
            path = manifest_path.parent / m["model"]
            if not path.exists() or sha256_of(path) != m["sha256"]:
                problems.append(f"{path} is missing or not the model fold_models.json lists; "
                                "rerun crossval_router.py")
                continue
            for iid, f in manifest["fold_of"][repeat].items():
                if f != fold:
                    continue
                if iid not in entries:
                    problems.append(f"{iid} is in fold_models.json but not in the datasets")
                    continue
                pick, _ = router.route(entries[iid], str(path))
                expected = manifest["pick"][repeat][iid]
                if pick != expected:
                    problems.append(f"repeat {repeat}, fold {fold}, {iid}: DateSat picks {pick}, "
                                    f"the cross-validation {expected}")
    return problems


def bundled_model_note():
    """Whether DateSat's bundled model is train_router.py's router.json."""
    from datesat.router.model import BUNDLED_MODEL
    trained = MODEL / "router.json"
    if not trained.exists():
        return f"no {trained} to compare DateSat's bundled model with"
    same = BUNDLED_MODEL.exists() and sha256_of(BUNDLED_MODEL) == sha256_of(trained)
    return (f"DateSat's bundled model {'is' if same else 'is NOT'} {trained}"
            + ("" if same else f"; copy it to {BUNDLED_MODEL}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(CROSS_VALIDATION / "fold_models.json"),
                    help="crossval_router.py's list of fold models")
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT))
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    n_models = sum(len(f) for f in manifest["models"])
    print(f"checking DateSat's router against {args.manifest}: {len(manifest['features'])} "
          f"features, {n_models} fold models over {manifest['repeats']} repeats ...")
    problems = check(args.manifest, args.dataset_root)
    print(bundled_model_note())
    if problems:
        print(f"\n{len(problems)} differences:")
        for p in problems[:SHOWN]:
            print(f"  {p}")
        if len(problems) > SHOWN:
            print(f"  ... and {len(problems) - SHOWN} more")
        raise SystemExit(1)
    routed = sum(len(r) for r in manifest["fold_of"])
    print(f"DateSat computes every feature as DateSATBench does, and routes all {routed} "
          f"(repeat, instance) pairs as predicted")


if __name__ == "__main__":
    main()
