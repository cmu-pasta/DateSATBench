#!/usr/bin/env python3
"""
Generate bounded variants of the DateSATBench datasets by injecting explicit
range constraints on every date variable.

For each dataset entry, every declaration of type `date` gets two extra
constraints appended to its constraint list:

    <var> >= Date(y0, m0, d0)
    <var> <= Date(y1, m1, d1)

The bound window and the variant name are given on the command line:

    python tools/inject_bounds.py --min 1900/3/1 --max 2100/2/28 --name years_1900_2100

Running with no window arguments generates the two default variants:

    positive_years    0001-01-01 .. 9999-12-31
    years_1900_2100   1900-03-01 .. 2100-02-28

This puts the bound in the BENCHMARK (constraint text) rather than in the
solver, so an unbounded DateSAT (bound='none') can be evaluated against
bounded problems. Note the semantics differ from solver-level bounds:
injected constraints only restrict the declared variables - intermediate
results of date arithmetic remain unbounded, and out-of-window excursions
that return in range are satisfiable.

Output layout mirrors the source datasets so each variant root is a drop-in
replacement for --datesatbench-repo in DateSAT's eval/run_benchmarks.py:

    <output>/<name>/llm_constraints/constraints/constraints.json
    <output>/<name>/grammar_constraints/constraints/constraints.json
    <output>/<name>/legal_doc_constraints/constraints/constraints.jsonl
    <output>/<name>/bound_manifest.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (name, (y, m, d) min, (y, m, d) max) generated when no --min/--max is given
DEFAULT_VARIANTS = [
    ("positive_years", (1, 1, 1), (9999, 12, 31)),
    ("years_1900_2100", (1900, 3, 1), (2100, 2, 28)),
]

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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def inject_bounds(entry: dict, name: str, lo: tuple, hi: tuple) -> dict:
    """Return a copy of the entry with bound constraints appended for every date var."""
    (y0, m0, d0), (y1, m1, d1) = lo, hi
    bounded = dict(entry)
    constraints = list(entry.get("constraints", []))
    for var in date_var_names(entry):
        constraints.append(f"{var} >= Date({y0}, {m0}, {d0})")
        constraints.append(f"{var} <= Date({y1}, {m1}, {d1})")
    bounded["constraints"] = constraints
    bounded["injected_bound"] = {
        "name": name,
        "min": f"Date({y0}, {m0}, {d0})",
        "max": f"Date({y1}, {m1}, {d1})",
    }
    return bounded


def generate_variant(name: str, lo: tuple, hi: tuple, source: Path, output: Path) -> None:
    """Generate one bounded variant of all datasets under <output>/<name>/."""
    (y0, m0, d0), (y1, m1, d1) = lo, hi
    variant_root = output / name
    print(f"\n=== Variant '{name}': "
          f"[{y0:04d}-{m0:02d}-{d0:02d} .. {y1:04d}-{m1:02d}-{d1:02d}] ===")

    manifest = {
        "name": name,
        "min": list(lo),
        "max": list(hi),
        "source": str(source),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/inject_bounds.py",
        "datasets": {},
    }

    for dataset_name, rel_path in DATASETS.items():
        src_file = source / rel_path
        if not src_file.exists():
            print(f"  ⚠️  skipping {dataset_name}: {src_file} not found")
            continue

        entries = load_entries(src_file)
        bounded_entries = [inject_bounds(e, name, lo, hi) for e in entries]
        injected = sum(2 * len(date_var_names(e)) for e in entries)

        dst_file = variant_root / rel_path
        dump_entries(bounded_entries, dst_file)

        manifest["datasets"][dataset_name] = {
            "entries": len(bounded_entries),
            "injected_constraints": injected,
            "file": str(rel_path),
        }
        print(f"  {dataset_name}: {len(bounded_entries)} entries, "
              f"{injected} bound constraints -> {dst_file}")

    (variant_root / "bound_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest: {variant_root / 'bound_manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject per-variable date bounds into DateSATBench datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # The two default variants (positive_years, years_1900_2100)
  python tools/inject_bounds.py

  # A custom window with a custom variant name
  python tools/inject_bounds.py --min 1950/1/1 --max 2050/12/31 --name years_1950_2050
        """,
    )
    parser.add_argument(
        "--min",
        type=parse_ymd,
        default=None,
        metavar="Y/M/D",
        help="Lower bound date, e.g. 1900/3/1 (requires --max and --name)",
    )
    parser.add_argument(
        "--max",
        type=parse_ymd,
        default=None,
        metavar="Y/M/D",
        help="Upper bound date, e.g. 2100/2/28 (requires --min and --name)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Variant name; becomes the output subdirectory "
        "(requires --min and --max)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(REPO_ROOT / "datesatbench"),
        help="Source datesatbench directory containing llm_constraints/ etc. "
        "(default: <repo>/datesatbench)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "datesatbench_bounded"),
        help="Output root; one subdirectory per variant is created "
        "(default: <repo>/datesatbench_bounded)",
    )
    args = parser.parse_args()

    custom = [args.min, args.max, args.name]
    if any(v is not None for v in custom) and not all(v is not None for v in custom):
        parser.error("--min, --max, and --name must be given together")

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.is_dir():
        print(f"Error: source directory not found: {source}")
        return 1

    if args.name is not None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name):
            parser.error(f"--name must be filesystem-friendly, got: {args.name!r}")
        if args.min > args.max:
            parser.error("--min must not be after --max")
        variants = [(args.name, args.min, args.max)]
    else:
        variants = DEFAULT_VARIANTS

    for name, lo, hi in variants:
        generate_variant(name, lo, hi, source, output)

    example = variants[0][0]
    print(
        "\nDone. Use a variant as a drop-in dataset root, e.g.:\n"
        f"  python eval/run_benchmarks.py --mode eval --bound none \\\n"
        f"      --datesatbench-repo {output}/{example} --tag bench-{example}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
