#!/usr/bin/env python3
"""
Inject or remove per-variable date bounds in the DateSATBench datasets, in place.

Injecting appends two constraints to each entry for every declaration of type
`date`, and records the window in the entry's `injected_bound` field:

    <var> >= Date(y0, m0, d0)
    <var> <= Date(y1, m1, d1)

Removing takes those constraints and the field out again.

    python -m datesatbench.utils.bounds inject      # positive_years, 0001-01-01 .. 9999-12-31
    python -m datesatbench.utils.bounds inject --min 1900/3/1 --max 2100/2/28 --name years_1900_2100
    python -m datesatbench.utils.bounds remove

Both are safe to run again. Injecting leaves an entry that already carries the
window as it is, and replaces the bounds of an entry that carries a different
window, so an entry never has more than one pair per variable. Removing leaves
an entry without bounds as it is. A file in which no entry changes is not
rewritten.

The files are the three datasets under datesatbench/ (or --root):

    llm_constraints/constraints/constraints.json
    grammar_constraints/constraints/constraints.json
    legal_doc_constraints/constraints/constraints.jsonl

This puts the bound in the BENCHMARK (constraint text) rather than in the
solver, so an unbounded DateSAT (bound='none') can be evaluated against
bounded problems. Note the semantics differ from solver-level bounds:
injected constraints only restrict the declared variables - intermediate
results of date arithmetic remain unbounded, and out-of-window excursions
that return in range are satisfiable.
"""

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (name, (y, m, d) min, (y, m, d) max) injected when no --min/--max is given
POSITIVE_YEARS = ("positive_years", (1, 1, 1), (9999, 12, 31))

# Dataset name -> constraints file relative to the datesatbench root
DATASETS = {
    "llm_constraints": Path("llm_constraints") / "constraints" / "constraints.json",
    "grammar_constraints": Path("grammar_constraints") / "constraints" / "constraints.json",
    "legal_doc_constraints": Path("legal_doc_constraints") / "constraints" / "constraints.jsonl",
}


def parse_ymd(text: str) -> tuple[int, int, int]:
    """Parse a Y/M/D (or Y-M-D) date string into an (y, m, d) tuple."""
    parts = re.split(r"[/-]", text.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected a date like 1900/3/1 or 1900-3-1, got: {text!r}"
        )
    y, m, d = (int(p) for p in parts)
    if not 1 <= m <= 12:
        raise argparse.ArgumentTypeError(f"Month out of range in {text!r}")
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    dim = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    if not 1 <= d <= dim:
        raise argparse.ArgumentTypeError(f"Day out of range in {text!r}")
    return (y, m, d)


def load_entries(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def dump_entries(entries: list[dict], path: Path) -> None:
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    else:
        path.write_text(json.dumps(entries, indent=2))


def date_var_names(entry: dict) -> list[str]:
    """Names of all date-typed variables declared in this entry."""
    names = []
    for decl in entry.get("declarations", []):
        name, _, typ = decl.partition(":")
        if typ.strip() == "date":
            names.append(name.strip())
    return names


def bound_record(name: str, lo: tuple, hi: tuple) -> dict:
    """The `injected_bound` field for a window."""
    return {"name": name, "min": "Date(%d, %d, %d)" % lo, "max": "Date(%d, %d, %d)" % hi}


def bound_constraints(entry: dict, bound: dict) -> list[str]:
    """The constraints a window adds to the entry: a lower and an upper bound per date variable."""
    return [
        c
        for var in date_var_names(entry)
        for c in (f"{var} >= {bound['min']}", f"{var} <= {bound['max']}")
    ]


def own_constraints(entry: dict) -> list[str]:
    """The entry's constraints without its injected bounds.

    Checks that the bounds its `injected_bound` names really are the last
    constraints, so an entry is never silently truncated.
    """
    constraints = entry.get("constraints", [])
    bound = entry.get("injected_bound")
    if bound is None:
        return list(constraints)
    expected = bound_constraints(entry, bound)
    cut = len(constraints) - len(expected)
    if cut < 0 or constraints[cut:] != expected:
        raise ValueError(
            f"{entry.get('id', '<no id>')}: the {bound['name']} bounds are not at the end of its constraints"
        )
    return constraints[:cut]


def inject_bounds(entry: dict, bound: dict) -> dict:
    """Return the entry with `bound` injected, or the entry itself if it already carries it.

    Bounds for a different window are taken out first.
    """
    constraints = own_constraints(entry)
    if entry.get("injected_bound") == bound:
        return entry
    return {
        **entry,
        "constraints": constraints + bound_constraints(entry, bound),
        "injected_bound": bound,
    }


def remove_bounds(entry: dict) -> dict:
    """Return the entry without its injected bounds, or the entry itself if it has none."""
    if "injected_bound" not in entry:
        return entry
    stripped = {key: value for key, value in entry.items() if key != "injected_bound"}
    stripped["constraints"] = own_constraints(entry)
    return stripped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject or remove per-variable date bounds in the DateSATBench datasets, in place.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inject the positive_years bounds, 0001-01-01 .. 9999-12-31
  python -m datesatbench.utils.bounds inject

  # Inject a custom window (replaces any other window an entry carries)
  python -m datesatbench.utils.bounds inject --min 1950/1/1 --max 2050/12/31 --name years_1950_2050

  # Take the bounds out again
  python -m datesatbench.utils.bounds remove
        """,
    )
    parser.add_argument(
        "action",
        choices=["inject", "remove"],
        help="inject: append the bounds (positive_years unless --min, --max and --name are given); "
        "remove: take out whatever bounds an entry carries",
    )
    parser.add_argument(
        "--min",
        type=parse_ymd,
        default=None,
        metavar="Y/M/D",
        help="Lower bound to inject, e.g. 1900/3/1 (requires --max and --name)",
    )
    parser.add_argument(
        "--max",
        type=parse_ymd,
        default=None,
        metavar="Y/M/D",
        help="Upper bound to inject, e.g. 2100/2/28 (requires --min and --name)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Window name recorded in each entry's injected_bound (requires --min and --max)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(REPO_ROOT / "datesatbench"),
        help="Directory containing llm_constraints/ etc., whose files are rewritten "
        "(default: <repo>/datesatbench)",
    )
    args = parser.parse_args()

    custom = [args.min, args.max, args.name]
    if any(v is not None for v in custom):
        if args.action == "remove":
            parser.error("--min, --max, and --name only apply to inject")
        if not all(v is not None for v in custom):
            parser.error("--min, --max, and --name must be given together")
        if args.min > args.max:
            parser.error("--min must not be after --max")
        name, lo, hi = args.name, args.min, args.max
    else:
        name, lo, hi = POSITIVE_YEARS
    bound = bound_record(name, lo, hi)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: datasets directory not found: {root}")
        return 1

    if args.action == "inject":
        print(f"Injecting '{name}' [{bound['min']} .. {bound['max']}] under {root}")
    else:
        print(f"Removing bounds under {root}")

    # Update every dataset before writing any, so a malformed entry leaves all files untouched.
    updates = []
    for dataset_name, rel_path in DATASETS.items():
        path = root / rel_path
        if not path.exists():
            print(f"  ⚠️  skipping {dataset_name}: {path} not found")
            continue
        entries = load_entries(path)
        try:
            if args.action == "inject":
                updated = [inject_bounds(e, bound) for e in entries]
            else:
                updated = [remove_bounds(e) for e in entries]
        except ValueError as e:
            print(f"Error in {path}: {e}")
            return 1
        changed = sum(u is not e for u, e in zip(updated, entries))
        updates.append((dataset_name, path, updated, len(entries), changed))

    for dataset_name, path, updated, total, changed in updates:
        if changed:
            dump_entries(updated, path)
        if args.action == "inject":
            print(f"  {dataset_name}: injected into {changed} of {total} entries ({total - changed} already had them)")
        else:
            print(f"  {dataset_name}: removed from {changed} of {total} entries ({total - changed} had none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
