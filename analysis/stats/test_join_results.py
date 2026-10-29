import csv
import json
import sys
import tempfile
from pathlib import Path

from analysis.stats import join_results


def write_results(root, encoding, records, run="run_1", corpus="llm"):
    path = Path(root) / corpus / run / f"{encoding}_int.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


def join(*args):
    """Run join_results with these arguments; returns its rows, or the SystemExit."""
    argv = sys.argv
    sys.argv = ["join_results", "--timeout", "20000", *args]
    try:
        join_results.main()
    except SystemExit as e:
        return e
    finally:
        sys.argv = argv


def test_the_router_joins_from_its_own_results_directory():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "features.csv").write_text("id,corpus,n_atoms,label_status\ni1,llm,3,\ni2,llm,1,\n")
        for e in ("simple", "epoch_days"):
            write_results(d / "encodings", e, [{"id": i, "execution_time": 1.0, "status": "sat"}
                                               for i in ("i1", "i2")])
        write_results(d / "router", "router", [
            {"id": "i1", "execution_time": 0.5, "status": "sat", "routed_to": "epoch_days",
             "routing_time": 0.001},
            {"id": "i2", "execution_time": 20.1, "status": "timeout", "hard_timeout": True}])
        out = d / "joined.csv"
        join("--results", str(d / "encodings"), str(d / "router"),
             "--features", str(d / "features.csv"), "--output", str(out))
        rows = {(r["id"], r["encoding"]): r for r in csv.DictReader(out.open())}
        assert list(next(iter(rows.values())))[:len(join_results.RESULT_COLUMNS)] == join_results.RESULT_COLUMNS
        assert rows[("i1", "router")]["routed_to"] == "epoch_days"
        assert rows[("i1", "router")]["speedup"] == "2.0"
        assert rows[("i2", "router")]["routed_to"] == ""        # killed before it reported a pick
        assert rows[("i1", "simple")]["routing_time"] == ""


def test_two_results_directories_may_not_hold_the_same_run():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "features.csv").write_text("id,corpus,n_atoms\ni1,llm,3\n")
        for root in ("a", "b"):
            write_results(d / root, "simple", [{"id": "i1", "execution_time": 1.0, "status": "sat"}])
        e = join("--results", str(d / "a"), str(d / "b"), "--features", str(d / "features.csv"),
                 "--output", str(d / "joined.csv"))
        assert isinstance(e, SystemExit) and "more than one" in str(e)


if __name__ == "__main__":
    test_the_router_joins_from_its_own_results_directory()
    test_two_results_directories_may_not_hold_the_same_run()
    print("all join_results tests passed")
