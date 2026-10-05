"""
Check emulate_encodings.py against the formulas DateSat actually builds.

    python -m analysis.features.check_emulation [--encodings hybrid_ymd ...] [--limit N]

For every DateSatBench instance and encoding, builds the formula with DateSat (without
solving it), counts the distinct div, mod and if-then-else nodes in its assertions, and
compares them with emulate()'s divmod and ite counts. Distinct means by Z3 term identity,
the same sharing the emulation assumes. Needs DateSat installed. Prints every mismatch and
a per-encoding summary; exits 1 if any instance mismatches.

`simple` formulas whose emulated size exceeds --max-simple-divmod are skipped, as building
them takes minutes.
"""

import argparse
import contextlib
import io
import sys
from collections import Counter

import z3

from analysis.features.datesat_parser import parse_constraint, parse_declarations
from analysis.features.emulate_encodings import ENCODINGS, emulate
from analysis.features.extract_features import DATASETS, load
from analysis.paths import REPO


def build(entry, approach):
    """The assertions DateSat builds for `entry` with `approach`, without solving."""
    from datesat.api import DateSATBuilder
    from datesat.constraint_parser import ConstraintParser
    from datesat.core import Date, Period

    code = ConstraintParser().parse_constraint_data(
        {"constraints": entry["constraints"], "declarations": entry["declarations"]})
    env = {"DateSATBuilder": lambda: DateSATBuilder(approach=approach, implementation="int"),
           "Date": Date, "Period": Period}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, env)
    return env["builder"].solver.solver.assertions()


def count_nodes(assertions):
    """(div/mod, if-then-else) nodes in the assertions, each distinct term once."""
    seen, stack, n = set(), list(assertions), Counter()
    while stack:
        t = stack.pop()
        tid = t.get_id()
        if tid in seen:
            continue
        seen.add(tid)
        if z3.is_app(t):
            k = t.decl().kind()
            if k in (z3.Z3_OP_IDIV, z3.Z3_OP_MOD, z3.Z3_OP_REM):
                n["divmod"] += 1
            elif k == z3.Z3_OP_ITE:
                n["ite"] += 1
            stack.extend(t.children())
    return n["divmod"], n["ite"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=str(REPO / "datesatbench"))
    ap.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=ENCODINGS)
    ap.add_argument("--limit", type=int, help="first N instances of each corpus only")
    ap.add_argument("--max-simple-divmod", type=int, default=20000)
    ap.add_argument("--show", type=int, default=10, help="mismatches to print per encoding")
    args = ap.parse_args()

    from pathlib import Path
    root = Path(args.dataset_root)
    tally = {e: Counter() for e in args.encodings}
    shown = Counter()
    for corpus, rel in DATASETS.items():
        entries = load(root / rel)[: args.limit]
        for entry in entries:
            sorts = parse_declarations(entry["declarations"])
            trees = [parse_constraint(s, sorts) for s in entry["constraints"]]
            emu = emulate(trees, sorts)
            for enc in args.encodings:
                want = (emu[enc]["divmod"], emu[enc]["ite"])
                if enc == "simple" and want[0] > args.max_simple_divmod:
                    tally[enc]["skipped"] += 1
                    continue
                try:
                    got = count_nodes(build(entry, enc))
                except Exception as exc:  # noqa: BLE001 - report and go on
                    tally[enc]["error"] += 1
                    print(f"{enc:12} {entry['id']}: build failed: {exc}")
                    continue
                if got == want:
                    tally[enc]["match"] += 1
                else:
                    tally[enc]["mismatch"] += 1
                    if shown[enc] < args.show:
                        shown[enc] += 1
                        print(f"{enc:12} {entry['id']}: built divmod={got[0]} ite={got[1]}, "
                              f"emulated divmod={want[0]} ite={want[1]}")
    print()
    for enc, t in tally.items():
        print(f"{enc:12} match {t['match']}, mismatch {t['mismatch']}, "
              f"skipped {t['skipped']}, build errors {t['error']}")
    sys.exit(1 if any(t["mismatch"] for t in tally.values()) else 0)


if __name__ == "__main__":
    main()
