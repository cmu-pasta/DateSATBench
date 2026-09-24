"""
Run every example in features.md through extract_features.py.

    python3 analysis/test_features_md.py

An example is a line (or wrapped line) inside a code block under a `### <column>` heading:

    a == Date(2020,1,1);  b == a + Period(0,0,5);  c > b      →  1   (comment)

Left of the arrow: `;`-separated constraints, plus optional `name: sort` items that declare
a variable used in no constraint. Variables are declared by name: `a`..`e` are dates,
`x`, `y`, `n` are ints, `f`, `g` are bools; only the variables that occur are declared.
Right of the arrow: the expected value of that column (compared to 3 decimals); anything
after it is a comment. `(same example)` reuses the previous example's constraints.

Also checks that every `###` heading is a column of features.csv and that every feature
column has at least one example. Exits 1 on any failure.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import COLUMNS, features_for  # noqa: E402

HERE = Path(__file__).parent
FEATURES = [c for c in COLUMNS if c not in ("id", "corpus") and not c.startswith("label_")]
SORT_OF = {**{v: "date" for v in "abcde"}, **{v: "int" for v in "xyn"}, **{v: "bool" for v in "fg"}}
SKIP = {"Date", "Period", "True", "False", "year", "month", "day"}
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
DECL = re.compile(r"(\w+)\s*:\s*(date|int|bool)")


def examples(md):
    """Yield (feature, lhs, expected, line_no) for every arrow line in a feature's code block."""
    feature, in_block, buf, start = None, False, "", 0
    for i, line in enumerate(md.splitlines(), 1):
        if line.startswith("### "):
            feature = line[4:].strip()
            continue
        if line.startswith("## "):
            feature = None
            continue
        if line.startswith("```"):
            in_block, buf = not in_block, ""
            continue
        if not in_block or feature is None:
            continue
        if not buf:
            start = i
        buf += " " + line.strip()
        if "→" in buf:
            lhs, rhs = buf.split("→", 1)
            buf = ""
            yield feature, lhs.strip(), rhs.strip().split()[0], start


def build_entry(lhs, prev):
    if lhs.startswith("(same example)"):
        if prev is None:
            raise ValueError("'(same example)' with no previous example")
        return prev
    decls, cons = {}, []
    for item in (x.strip() for x in lhs.split(";")):
        if not item:
            continue
        m = DECL.fullmatch(item)
        if m:
            decls[m[1]] = m[2]
        else:
            cons.append(item)
    for c in cons:
        for v in IDENT.findall(c):
            if v in SKIP or v in decls:
                continue
            if v not in SORT_OF:
                raise ValueError(f"no sort convention for variable {v!r} in {c!r}")
            decls[v] = SORT_OF[v]
    return {"id": "doc", "declarations": [f"{v}: {s}" for v, s in decls.items()],
            "constraints": cons}


def main():
    md = (HERE / "features.md").read_text()
    headings = [l[4:].strip() for l in md.splitlines() if l.startswith("### ")]
    failures = []
    for h in headings:
        if h not in FEATURES:
            failures.append(f"heading '### {h}' is not a column of features.csv")
    covered = set()
    prev = None
    n = 0
    for feature, lhs, expected, line in examples(md):
        n += 1
        try:
            entry = build_entry(lhs, prev)
            prev = entry
            got = features_for(entry, "doc")[feature]
            ok = round(float(got), 3) == round(float(expected), 3)
        except Exception as e:  # noqa: BLE001
            got, ok = f"error: {e}", False
        covered.add(feature)
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} L{line:<4} {feature:24} expected {expected:<7} got {got}")
        if not ok:
            failures.append(f"features.md:{line} {feature}: expected {expected}, got {got}")
    for c in FEATURES:
        if c not in covered:
            failures.append(f"column {c} has no example in features.md")

    print(f"\n{n} examples, {len(covered)}/{len(FEATURES)} feature columns covered")
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all examples match extract_features.py")


if __name__ == "__main__":
    main()
