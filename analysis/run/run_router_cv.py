"""
Run DateSat's router on the datasets with the fold models of crossval_router.py: the
router's live evaluation.

    python -m analysis.run.run_router_cv --tag router     # every repeat, 20 s timeout

crossval_router.py cuts the instances into 5 folds --repeats times (default 10), each time
at random and stratified by corpus, and writes, for every fold, the model trained on the
other folds (fold_models.json lists them). The fold models are committed; when they are
missing, this script runs crossval_router.py first to train them, which needs pandas and
scikit-learn in the Python that runs it. Run N of this script uses repeat N - 1: DateSat's
router approach solves every instance with the model of the fold that holds it, so no
instance is solved by a model trained on it, and every run uses another cut.

Everything else is run_benchmarks.py's, so the router is timed as the encodings were: the
same solver timeout and hard wall-clock limit; the runs one after another; in each run the
corpora in run_benchmarks.py's order (llm, grammar, legal) and each corpus's instances in
dataset order, one at a time, in one worker process per run and corpus. DateSat's time for
the router covers what it covers for an encoding (building the constraints in the picked
encoding and solving them) plus the routing: computing the features from the constraint
text and walking the forests. Loading a model is not counted: a worker loads each of its
models once, before the timed region of the first instance routed by it, as importing Z3
is not counted either.

Before solving anything it runs check_datesat_router.py, and stops if DateSat would route
any instance differently from the cross-validation's prediction.

Results go where run_benchmarks.py writes them, <results-dir>/<tag>/<corpus>/run_N/
router_int.json, in the same format with each instance's repeat, fold and model added, so
join_results.py reads them as it reads an encoding's (give it both results directories). A
run of a corpus whose file exists is skipped, so a stopped evaluation resumes where it
stopped; results from other fold models, another DateSat commit or another machine are
refused. run_config.json records the timeouts, the fold models' manifest and its SHA-256,
DateSat's commit, and the machine (host, CPU, Python and Z3 versions): the router is only
comparable with encodings recorded on the same machine, and crossval_router.py's `live` block
shows whether it ran as fast. After the runs, every solution is checked by concrete
execution, as run_benchmarks.py does (--no-analysis skips it).
"""

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import z3

from analysis.features.extract_features import DATASET_ROOT, DATASETS
from analysis.paths import CROSS_VALIDATION, REPO
from analysis.router.check_datesat_router import check, sha256_of
from analysis.run.run_benchmarks import (HARD_TIMEOUT_MS, TIMEOUT_MS, _load_constraints,
                                         _SolverWorker, check_run_dir, datesat,
                                         run_constraint_with_approach)

APPROACH = "router"
MANIFEST = "fold_models.json"      # crossval_router.py's list of the fold models it writes


def missing_models(manifest_path):
    """The fold models that manifest_path lists but are not there, or [manifest_path] when it
    is not there itself."""
    if not manifest_path.exists():
        return [manifest_path]
    models = json.loads(manifest_path.read_text())["models"]
    return [manifest_path.parent / m["model"] for folds in models for m in folds
            if not (manifest_path.parent / m["model"]).exists()]


def train_fold_models(out_dir):
    """Run crossval_router.py with this Python, which trains the fold models and writes them
    with their manifest into out_dir (with its router_eval.json and .csv)."""
    absent = [m for m in ("pandas", "sklearn") if importlib.util.find_spec(m) is None]
    if absent:
        raise SystemExit(f"training the fold models needs pandas and scikit-learn, and "
                         f"{sys.executable} lacks {' and '.join(absent)}: install them with "
                         f"pip install -e \".[analysis]\" from {REPO}")
    cmd = [sys.executable, "-m", "analysis.router.crossval_router", "--output-dir", str(out_dir)]
    print(f"training the fold models: {' '.join(cmd)}\n")
    if subprocess.run(cmd, cwd=REPO).returncode:
        raise SystemExit("crossval_router.py failed; no fold models to run with")


def datesat_commit():
    """The commit of the DateSat checkout in use, "-dirty" when its datesat/ has uncommitted
    changes, or None when it is not a git checkout."""
    root = Path(datesat.__file__).resolve().parent.parent
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", "datesat") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def machine():
    """What the runs ran on: the host, its CPU, and the Python and Z3 versions."""
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((l.split(":", 1)[1].strip() for l in cpuinfo.read_text().splitlines()
                    if l.startswith("model name")), cpu)
    else:
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, check=True).stdout.strip() or cpu
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"host": platform.node(), "cpu": cpu, "platform": platform.platform(),
            "python": platform.python_version(), "z3": z3.get_version_string()}


def route_corpus(entries, manifest, manifest_dir, repeat, timeout_ms, hard_timeout_ms):
    """Solve every entry with the router, each routed by the model of its fold in `repeat`,
    in one worker. Returns the result records, in the entries' order, and the ids whose
    pick differs from the cross-validation's prediction."""
    fold_of, models, predicted = (manifest["fold_of"][repeat], manifest["models"][repeat],
                                  manifest["pick"][repeat])
    records, differ = [], []
    with _SolverWorker() as worker:
        for entry in entries:
            iid = entry.get("id")
            if iid not in fold_of:
                print(f"WARNING: {iid} is in no fold of the cross-validation; skipped")
                continue
            fold = fold_of[iid]
            model = models[fold]["model"]
            result = run_constraint_with_approach(
                entry, APPROACH, "int", worker, timeout_ms, False, hard_timeout_ms,
                router_model=str(manifest_dir / model))
            result.update(repeat=repeat, fold=fold, router_model=model)
            if "routed_to" in result and result["routed_to"] != predicted[iid]:
                differ.append(iid)
            records.append(result)
    return records, differ


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(CROSS_VALIDATION / "fold_models.json"),
                    help="crossval_router.py's list of fold models")
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT),
                    help="dataset root containing the three *_constraints directories")
    ap.add_argument("--runs", type=int,
                    help="runs to make, run N with repeat N - 1 (default: one per repeat)")
    ap.add_argument("--corpora", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                    help="corpora to run (default: all three)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_MS,
                    help=f"solver timeout in ms (default: {TIMEOUT_MS})")
    ap.add_argument("--hard-timeout", type=int, default=HARD_TIMEOUT_MS,
                    help=f"wall-clock limit per instance in ms (default: {HARD_TIMEOUT_MS})")
    ap.add_argument("--tag", help="results subdirectory (default: a YYYYmmdd_HHMMSS timestamp)")
    ap.add_argument("--results-dir", default=str(REPO / "results" / "runs"),
                    help="directory under which <tag>/ is created")
    ap.add_argument("--no-analysis", action="store_true",
                    help="skip checking the solutions after the runs")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    missing = missing_models(manifest_path)
    if missing:
        if manifest_path.name != MANIFEST:
            ap.error(f"{missing[0]} is missing, and crossval_router.py only writes {MANIFEST}")
        print(f"{missing[0]} is missing" + (f" ({len(missing)} fold models)" if len(missing) > 1 else ""))
        train_fold_models(manifest_path.parent)
        if missing_models(manifest_path):
            raise SystemExit(f"crossval_router.py did not write every fold model {manifest_path} lists")
    manifest = json.loads(manifest_path.read_text())
    runs = args.runs or manifest["repeats"]
    if not 1 <= runs <= manifest["repeats"]:
        ap.error(f"--runs must be between 1 and the {manifest['repeats']} repeats of {manifest_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.results_dir).expanduser().resolve() / (args.tag or timestamp)

    config_path = root / "run_config.json"
    config = {
        "timeout_ms": args.timeout,
        "hard_timeout_ms": args.hard_timeout,
        "approaches": [APPROACH],
        "datasets": sorted(args.corpora),
        "maxsat": False,
        "timestamp": timestamp,
        "tag": root.name,
        "datesatbench_root": str(Path(args.dataset_root).resolve()),
        "fold_models": str(manifest_path),
        "fold_models_sha256": sha256_of(manifest_path),
        "cross_validation": {k: manifest[k] for k in ("seed", "folds", "repeats", "trees", "setting")},
        "datesat_commit": datesat_commit(),
        "machine": machine(),
    }
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        for key in ("timeout_ms", "hard_timeout_ms", "fold_models_sha256", "datesat_commit", "machine"):
            if prior.get(key) != config[key]:
                raise SystemExit(f"{root} holds results made with another {key} "
                                 f"({prior.get(key)!r}, now {config[key]!r}); use another --tag")
        config.update(timestamp=prior["timestamp"],
                      datasets=sorted(set(prior["datasets"]) | set(args.corpora)))

    print(f"fold models: {manifest_path} ({manifest['folds']} folds x {manifest['repeats']} repeats)")
    print(f"runs: {runs}, corpora: {', '.join(args.corpora)}, timeout {args.timeout} ms, "
          f"hard timeout {args.hard_timeout} ms")
    print(f"DateSat: {config['datesat_commit']}; machine: {config['machine']}")
    if config["datesat_commit"] is None or config["datesat_commit"].endswith("-dirty"):
        print("WARNING: DateSat's datesat/ is not a clean commit, so these results cannot be "
              "traced to its code; commit it first")
    print(f"results: {root}\n")
    problems = check(manifest_path, args.dataset_root)
    if problems:
        print("\n".join(problems[:20]))
        raise SystemExit(f"DateSat's router does not route as the cross-validation predicts "
                         f"({len(problems)} differences); run check_datesat_router.py")
    print("DateSat's router routes every instance as the cross-validation predicts\n")

    root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    done = []
    for run in range(1, runs + 1):
        for corpus in args.corpora:
            out = root / corpus / f"run_{run}" / f"{APPROACH}_int.json"
            if out.exists():
                print(f"run {run}, {corpus}: {out} exists, skipped")
                continue
            entries = _load_constraints(str(Path(args.dataset_root) / DATASETS[corpus]))
            print(f"\n{'=' * 70}\nrun {run} of {runs} (repeat {run - 1}), {corpus}: "
                  f"{len(entries)} instances\n{'=' * 70}")
            records, differ = route_corpus(entries, manifest, manifest_path.parent, run - 1,
                                           args.timeout, args.hard_timeout)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(records, indent=2, default=str))
            solved = sum(r["status"] in ("sat", "unsat") for r in records)
            print(f"\nrun {run}, {corpus}: solved {solved} of {len(records)} in "
                  f"{sum(r['execution_time'] for r in records):.1f} s; wrote {out}")
            if differ:
                print(f"WARNING: on {len(differ)} instances DateSat picked another encoding than "
                      f"predicted: {', '.join(differ[:10])}")
            done.append(out.parent)

    if not args.no_analysis:
        for results_dir in done:
            print(f"\n{'=' * 60}\nchecking the solutions in {results_dir}\n{'=' * 60}")
            check_run_dir(results_dir)


if __name__ == "__main__":
    main()
