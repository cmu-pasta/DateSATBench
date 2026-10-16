import json
import tempfile
from pathlib import Path

import pandas as pd

from analysis.report.build_report import reduction_section, router_section


def summary(total, oracle):
    return {"instances": 2,
            "table": {"always x": {"total_s": 4.0, "timeouts": 0, "speedup_total": 1.0, "speedup_geomean": 1.0},
                      "router": {"total_s": total, "timeouts": 0.5, "speedup_total": 4.0 / total, "speedup_geomean": 1.5},
                      "oracle": {"total_s": oracle, "timeouts": 0, "speedup_total": 4.0 / oracle, "speedup_geomean": 2.0}},
            "router_picks": {"x": 2}, "router_fell_back": 0.5, "router_picks_fastest": 0.75,
            "gap_closed_to_oracle": {"router": {"always x": 0.5}}, "router_total_sd_s": 0.5}


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


def test_router_section_has_solve_rates_and_how_often_each_encoding_is_fastest():
    log = pd.DataFrame({"id": ["i1", "i2", "i1", "i2"], "repeat": [0, 0, 1, 1],
                        "corpus": ["llm", "legal", "llm", "legal"], "best": ["x", "y", "x", "y"]})
    with tempfile.TemporaryDirectory() as d:
        cv, meta = write(d, CV, META)
        r = router_section(cv, meta)
        assert r["scopes"]["all"]["fastest"] is None      # no router_eval.csv yet
        log.to_csv(Path(d) / "router_eval.csv", index=False)
        r = router_section(cv, meta)
    assert r["scopes"]["all"]["solve"]["router"] == {"solved": 1.5, "rate": 0.75}
    assert r["scopes"]["all"]["solve"]["oracle"] == {"solved": 2.0, "rate": 1.0}
    assert r["scopes"]["all"]["fastest"] == {"x": 1}      # one repeat counted, encodings of the table
    assert r["scopes"]["llm"]["fastest"] == {"x": 1}


def test_a_model_trained_with_other_settings_is_flagged():
    with tempfile.TemporaryDirectory() as d:
        r = router_section(*write(d, CV, dict(META, min_samples_leaf=5)))
    assert not r["model"]["matches_cv"]


def test_no_cross_validation_means_no_section():
    with tempfile.TemporaryDirectory() as d:
        assert router_section(Path(d) / "missing.json", Path(d) / "missing_meta.json") is None


FEATS = pd.DataFrame({"id": ["i1", "i2", "i3"], "corpus": "llm",
                      "f": [1, 2, 3], "g": [2, 4, 6], "h": [3, 1, 2]})
SELECTION = {"threshold": 0.8, "n_features": 3, "constant": [],
             "families": [{"kept": "f", "dropped": [{"feature": "g", "rho": 1.0}]}],
             "selected": ["f", "h"], "most_correlated_left": {"a": "f", "b": "h", "rho": -0.5}}


def write_reduction(d, selection, cv_all, cv_selected):
    paths = [Path(d) / n for n in ("selection.json", "cv_all.json", "cv_selected.json")]
    for path, obj in zip(paths, (selection, cv_all, cv_selected)):
        if obj is not None:
            path.write_text(json.dumps(obj))
    return paths


def test_reduction_section_compares_the_two_cross_validations():
    cv_all = dict(CV, feature_columns=["f", "g", "h"])
    cv_sel = dict(CV, feature_columns=["f", "h"], overall=summary(3.5, 2.0))
    with tempfile.TemporaryDirectory() as d:
        sel, a, s = write_reduction(d, SELECTION, cv_all, cv_sel)
        r = reduction_section(sel, FEATS, a, s)
    assert r["selected"] == ["f", "h"] and sorted(r["corr"]["order"]) == ["f", "h"]
    assert r["router"]["n_features"] == {"all": 3, "selected": 2}
    assert r["router"]["scopes"]["all"]["selected"]["router"]["total_s"] == 3.5
    assert r["router"]["scopes"]["llm"]["all"]["gap_closed"] == 0.5
    assert r["router"]["same_setting"] and r["router"]["matches_selection"]


def test_a_stale_selection_or_other_setting_is_flagged():
    cv_all = dict(CV, feature_columns=["f", "g", "h"])
    cv_sel = dict(CV, feature_columns=["f"], repeats=5)
    with tempfile.TemporaryDirectory() as d:
        sel, a, s = write_reduction(d, SELECTION, cv_all, cv_sel)
        r = reduction_section(sel, FEATS, a, s)
    assert not r["router"]["same_setting"]
    assert not r["router"]["matches_selection"]


def test_no_selection_means_no_section_and_no_cross_validation_means_no_comparison():
    with tempfile.TemporaryDirectory() as d:
        sel, a, s = write_reduction(d, None, CV, CV)
        assert reduction_section(sel, FEATS, a, s) is None
    with tempfile.TemporaryDirectory() as d:
        sel, a, s = write_reduction(d, SELECTION, CV, None)
        r = reduction_section(sel, FEATS, a, s)
    assert r["router"] is None and r["families"] == SELECTION["families"]


if __name__ == "__main__":
    test_router_section_has_every_scope_and_the_models_settings()
    test_router_section_has_solve_rates_and_how_often_each_encoding_is_fastest()
    test_a_model_trained_with_other_settings_is_flagged()
    test_no_cross_validation_means_no_section()
    test_reduction_section_compares_the_two_cross_validations()
    test_a_stale_selection_or_other_setting_is_flagged()
    test_no_selection_means_no_section_and_no_cross_validation_means_no_comparison()
    print("all build_report tests passed")
