import json
import tempfile
from pathlib import Path

from analysis.report.build_report import router_section


def summary(total, oracle):
    return {"instances": 2,
            "table": {"always x": {"total_s": 4.0, "timeouts": 0, "speedup_total": 1.0, "speedup_geomean": 1.0},
                      "router": {"total_s": total, "timeouts": 0.5, "speedup_total": 4.0 / total, "speedup_geomean": 1.5},
                      "oracle": {"total_s": oracle, "timeouts": 0, "speedup_total": 4.0 / oracle, "speedup_geomean": 2.0}},
            "router_picks": {"x": 2}, "router_fell_back": 0.5, "router_picks_fastest": 0.75,
            "gap_closed_to_oracle": {"router": {"always x": 0.5}}}


CV = {"instances": 2, "dropped": {"errored_or_missing": 0, "all_timeout": 1}, "folds": 5,
      "repeats": 3, "trees": 50, "setting": {"min_samples_leaf": 10, "max_depth": 16, "margin": 0.0},
      "fallbacks": ["x"], "baseline": "x", "best_performing_encoding": "x", "timeout_cost_s": 20.0,
      "overall": summary(3.0, 2.0), "by_corpus": {"llm": summary(2.5, 2.0)}}
META = {"encodings": ["x", "y"], "trees": 50, "max_depth": 16, "min_samples_leaf": 10,
        "margin": 0.0, "fallback": "x", "instances": {"trained_on": 2}, "feature_columns": ["f", "g"],
        "pairs": {"x vs y": {"top_features": [{"feature": "f", "importance": 0.6},
                                              {"feature": "g", "importance": 0.3},
                                              {"feature": "h", "importance": 0.1}]}}}


def write(d, cv, meta):
    (Path(d) / "cv.json").write_text(json.dumps(cv))
    (Path(d) / "meta.json").write_text(json.dumps(meta))
    return Path(d) / "cv.json", Path(d) / "meta.json"


def test_router_section_has_every_scope_and_the_models_settings():
    with tempfile.TemporaryDirectory() as d:
        r = router_section(*write(d, CV, META))
    assert sorted(r["scopes"]) == ["all", "llm"]
    assert r["scopes"]["all"]["table"]["oracle"]["total_s"] == 2.0
    assert r["scopes"]["llm"]["gap_closed"] == 0.5
    assert r["cv"]["setting"] == CV["setting"]
    assert r["model"]["trained_on"] == 2 and r["model"]["n_features"] == 2
    assert r["model"]["pairs"] == [{"pair": "x vs y", "top": ["f", "g"]}]
    assert r["model"]["matches_cv"]


def test_a_model_trained_with_other_settings_is_flagged():
    with tempfile.TemporaryDirectory() as d:
        r = router_section(*write(d, CV, dict(META, min_samples_leaf=5)))
    assert not r["model"]["matches_cv"]


def test_no_cross_validation_means_no_section():
    with tempfile.TemporaryDirectory() as d:
        assert router_section(Path(d) / "missing.json", Path(d) / "missing_meta.json") is None


if __name__ == "__main__":
    test_router_section_has_every_scope_and_the_models_settings()
    test_a_model_trained_with_other_settings_is_flagged()
    test_no_cross_validation_means_no_section()
    print("all build_report tests passed")
