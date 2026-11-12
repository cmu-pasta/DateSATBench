"""Tests for run_router_cv.py and check_datesat_router.py; they need DateSat installed."""

import hashlib
import importlib.util
import json
import sys

import pytest

from analysis.features.extract_features import DATASETS
from analysis.router.check_datesat_router import check
from analysis.run import run_router_cv

ONE = ["x >= Date(2000, 1, 1)"]
TWO = ["x >= Date(2000, 1, 1)", "x <= Date(2000, 12, 31)"]
ENTRIES = {"llm": [("llm-a", ONE), ("llm-b", TWO)], "grammar": [("grammar-sat-1", ONE)],
           "legal": [("legal-1", TWO)]}
# epoch_days is faster on one atom (n_atoms <= 1.5), simple on more
TREE = {"feature": [0, -1, -1], "threshold": [1.5, 0.0, 0.0], "left": [1, -1, -1],
        "right": [2, -1, -1], "p_a": [0.0, 1.0, 0.0]}
FOLD_OF = {"llm-a": 0, "llm-b": 1, "grammar-sat-1": 0, "legal-1": 1}
PICK = {"llm-a": "epoch_days", "llm-b": "simple", "grammar-sat-1": "epoch_days", "legal-1": "simple"}


def make(tmp_path, pick=PICK):
    """A dataset root with ENTRIES, and a manifest of one repeat of two folds, both with the
    model of TREE. Returns (dataset root, manifest path)."""
    root = tmp_path / "datesatbench"
    for corpus, rel in DATASETS.items():
        entries = [{"id": i, "declarations": ["x: date"], "constraints": c} for i, c in ENTRIES[corpus]]
        path = root / rel
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(map(json.dumps, entries)) if path.suffix == ".jsonl"
                        else json.dumps(entries))
    text = json.dumps({"format_version": 1, "bounds": "keep", "encodings": ["epoch_days", "simple"],
                       "features": ["n_atoms"], "fallback": "simple", "margin": 0.0, "trained_on": {},
                       "pairs": [{"a": "epoch_days", "b": "simple", "trees": [TREE]}]})
    models = tmp_path / "cv" / "fold_models"
    models.mkdir(parents=True)
    folds = []
    for k in range(2):
        (models / f"repeat_0_fold_{k}.json").write_text(text)
        folds.append({"model": f"fold_models/repeat_0_fold_{k}.json",
                      "sha256": hashlib.sha256(text.encode()).hexdigest(), "fallback": "simple"})
    manifest = tmp_path / "cv" / "fold_models.json"
    manifest.write_text(json.dumps({"seed": 0, "folds": 2, "repeats": 1, "trees": 1, "setting": {},
                                    "features": ["n_atoms"], "models": [folds],
                                    "fold_of": [FOLD_OF], "pick": [pick]}))
    return root, manifest


def run(monkeypatch, root, manifest, results):
    monkeypatch.setattr(sys, "argv", ["run_router_cv", "--manifest", str(manifest),
                                      "--dataset-root", str(root), "--tag", "t",
                                      "--results-dir", str(results), "--no-analysis"])
    run_router_cv.main()


def test_every_instance_is_solved_with_its_folds_model(tmp_path, monkeypatch):
    root, manifest = make(tmp_path)
    run(monkeypatch, root, manifest, tmp_path / "results")
    for corpus, entries in ENTRIES.items():
        out = tmp_path / "results" / "t" / corpus / "run_1" / "router_int.json"
        records = json.loads(out.read_text())
        assert [r["id"] for r in records] == [i for i, _ in entries]
        for r in records:
            assert r["status"] == "sat" and r["routed_to"] == PICK[r["id"]]
            assert (r["repeat"], r["fold"]) == (0, FOLD_OF[r["id"]])
            assert r["router_model"] == f"fold_models/repeat_0_fold_{FOLD_OF[r['id']]}.json"
    config = json.loads((tmp_path / "results" / "t" / "run_config.json").read_text())
    assert config["approaches"] == ["router"] and config["timeout_ms"] == 20000


def test_a_finished_run_is_skipped_on_resume(tmp_path, monkeypatch):
    root, manifest = make(tmp_path)
    run(monkeypatch, root, manifest, tmp_path / "results")
    out = tmp_path / "results" / "t" / "llm" / "run_1" / "router_int.json"
    out.write_text("[]")
    run(monkeypatch, root, manifest, tmp_path / "results")
    assert out.read_text() == "[]"


def test_nothing_runs_when_datesat_routes_otherwise(tmp_path, monkeypatch):
    root, manifest = make(tmp_path, pick=dict(PICK, **{"llm-a": "simple"}))
    assert check(manifest, root) == ["repeat 0, fold 0, llm-a: DateSat picks epoch_days, "
                                     "the cross-validation simple"]
    with pytest.raises(SystemExit, match="does not route"):
        run(monkeypatch, root, manifest, tmp_path / "results")
    assert not (tmp_path / "results" / "t").exists()


@pytest.mark.parametrize("gone", ["fold_models.json", "fold_models/repeat_0_fold_1.json"])
def test_missing_fold_models_are_trained_first(tmp_path, monkeypatch, gone):
    root, manifest = make(tmp_path)
    path = manifest.parent / gone
    text = path.read_text()
    path.unlink()
    trained = []

    def train(out_dir):            # stands in for crossval_router.py, which writes it back
        trained.append(out_dir)
        path.write_text(text)

    monkeypatch.setattr(run_router_cv, "train_fold_models", train)
    run(monkeypatch, root, manifest, tmp_path / "results")
    assert trained == [manifest.parent]
    assert (tmp_path / "results" / "t" / "llm" / "run_1" / "router_int.json").exists()


def test_fold_models_that_are_there_are_not_retrained(tmp_path, monkeypatch):
    root, manifest = make(tmp_path)
    monkeypatch.setattr(run_router_cv, "train_fold_models", lambda out_dir: pytest.fail("retrained"))
    run(monkeypatch, root, manifest, tmp_path / "results")


@pytest.mark.skipif(all(importlib.util.find_spec(m) for m in ("pandas", "sklearn")),
                    reason="this Python can train the fold models")
def test_training_without_pandas_or_scikit_learn_says_so(tmp_path):
    with pytest.raises(SystemExit, match="needs pandas and scikit-learn"):
        run_router_cv.train_fold_models(tmp_path)


def test_results_from_other_fold_models_are_refused(tmp_path, monkeypatch):
    root, manifest = make(tmp_path)
    run(monkeypatch, root, manifest, tmp_path / "results")
    data = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(dict(data, seed=1)))
    with pytest.raises(SystemExit, match="fold_models_sha256"):
        run(monkeypatch, root, manifest, tmp_path / "results")
