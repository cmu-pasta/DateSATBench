"""
Extract a descriptive feature set over DateSATBench.

Writes one CSV row per benchmark instance. Usage:

    python -m analysis.features.extract_features --output analysis/outputs/features.csv
    python -m analysis.features.extract_features --bounds keep     # count the injected bound atoms

Conventions:
  * Identifier columns are `id` and `corpus`.
  * Solver-derived columns are prefixed `label_` so they cannot be mistaken for
    features. They exist only for the grammar corpus.
  * A ratio with no denominator (e.g. `mixed_frac` with no date arithmetic) is 0;
    the matching count (`n_date_period_ops` = 0) says why.
"""

import argparse
import csv
import json
import math
from collections import deque
from itertools import combinations
from pathlib import Path

from analysis.features.datesat_parser import (
    BinOp, BoolLit, DateAdd, DateCtor, Field, IntLit, PeriodConst, UnOp, Var,
    COMPARISONS, ORDERING,
    canonical, date_chain_base, date_chain_len, flatten, is_comparison,
    is_connective, is_ground_date, parse_constraint, parse_declarations,
    to_nnf, walk,
)
from analysis.features.emulate_encodings import ENCODINGS, emulate
from analysis.paths import OUTPUTS, REPO

# The results under results/bench-datetime-bound were run on this bounded variant, so
# features are extracted from the same text. The injected bound atoms are stripped
# before extraction unless --bounds keep is passed.
DATASET_ROOT = REPO / "datesatbench_bounded/positive_years"
DATASETS = {
    "llm": "llm_constraints/constraints/constraints.json",
    "grammar": "grammar_constraints/constraints/constraints.json",
    "legal": "legal_doc_constraints/constraints/constraints.jsonl",
}


def strip_injected_bounds(entry):
    """Drop the atoms datesatbench/utils/bounds.py injected: two per date variable, at the end.

    Checks the tail really is those atoms, so an entry is never silently truncated.
    """
    b = entry.get("injected_bound")
    if not b:
        return entry
    date_vars = [d.split(":")[0].strip() for d in entry["declarations"]
                 if d.split(":")[1].strip() == "date"]
    expected = [c for v in date_vars for c in (f"{v} >= {b['min']}", f"{v} <= {b['max']}")]
    k = len(expected)
    if k and entry["constraints"][-k:] != expected:
        raise ValueError(f"{entry.get('id')}: injected bounds not found at the end of constraints")
    return {**entry, "constraints": entry["constraints"][:len(entry["constraints"]) - k]}


def load(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# structural helpers
# --------------------------------------------------------------------------

def top_level_conjuncts(trees):
    """The constraint list is one conjunction; split it, and split on `&&` too."""
    out = []
    for t in trees:
        out.extend(flatten(t, "&&"))
    return out


def pinned_variables(conjuncts, sorts):
    """Variables fixed by an unconditional top-level equality, closed transitively."""
    pinned = set()
    changed = True
    while changed:
        changed = False
        for c in conjuncts:
            if not (isinstance(c, BinOp) and c.op == "=="):
                continue
            for lhs, rhs in ((c.lhs, c.rhs), (c.rhs, c.lhs)):
                if not (isinstance(lhs, Var) and lhs.name in sorts):
                    continue
                if lhs.name in pinned:
                    continue
                if rhs_is_determined(rhs, pinned):
                    pinned.add(lhs.name)
                    changed = True
    return pinned


def rhs_is_determined(n, pinned):
    """True if this expression evaluates to a fixed value given `pinned`."""
    if isinstance(n, (IntLit, BoolLit, PeriodConst)):
        return True
    if isinstance(n, Var):
        return n.name in pinned
    if isinstance(n, DateCtor):
        return all(rhs_is_determined(c, pinned) for c in (n.y, n.m, n.d))
    if isinstance(n, DateAdd):
        return rhs_is_determined(n.base, pinned)
    if isinstance(n, Field):
        return rhs_is_determined(n.base, pinned)
    if isinstance(n, BinOp):
        return rhs_is_determined(n.lhs, pinned) and rhs_is_determined(n.rhs, pinned)
    if isinstance(n, UnOp):
        return rhs_is_determined(n.operand, pinned)
    return False


def atoms_with_implication_flag(tree):
    """Yield (atom, under_impl) where under_impl means an `->` sits above the atom.

    Both sides of the implication count: the antecedent as well as the consequent.
    """
    def rec(n, under_impl):
        if is_comparison(n) or isinstance(n, (Var, BoolLit)) and n.sort == "bool":
            yield n, under_impl
            return
        if isinstance(n, BinOp) and n.op == "->":
            yield from rec(n.lhs, True)
            yield from rec(n.rhs, True)
            return
        if isinstance(n, BinOp) and n.op in ("&&", "||"):
            yield from rec(n.lhs, under_impl)
            yield from rec(n.rhs, under_impl)
            return
        if isinstance(n, UnOp) and n.op == "!":
            yield from rec(n.operand, under_impl)
            return
        if is_comparison(n):
            yield n, under_impl
    yield from rec(tree, False)


def is_leap_year(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def nnf_depth_and_polarity(tree):
    """Max connective depth to an atom, and the count of odd-polarity orderings.

    A flat chain of one connective is one level: `a || b || c` has depth 1, and a
    level is added only where the connective alternates (`&&` under `||` or vice versa).
    """
    nnf = to_nnf(tree)
    max_depth = 0
    neg_ordering = 0
    total_ordering = 0

    def rec(n, depth, negated, parent_op=None):
        nonlocal max_depth, neg_ordering, total_ordering
        if isinstance(n, UnOp) and n.op == "!":
            rec(n.operand, depth, not negated, parent_op)
            return
        if isinstance(n, BinOp) and n.op in ("&&", "||"):
            d = depth if n.op == parent_op else depth + 1
            rec(n.lhs, d, negated, n.op)
            rec(n.rhs, d, negated, n.op)
            return
        max_depth = max(max_depth, depth)
        if isinstance(n, BinOp) and n.op in ORDERING:
            total_ordering += 1
            if negated:
                neg_ordering += 1

    rec(nnf, 0, False)
    return max_depth, neg_ordering, total_ordering


def atom_variables(atom, sorts):
    return {n.name for n in walk(atom) if isinstance(n, Var) and n.name in sorts}


def period_op_kind(lhs_vars, rhs_vars):
    """How a date comparison relates its sides, given the variables of each: "self" when
    both mention the same single variable (a < a + p), "const" when one side mentions
    none (a + p < Date(...)), and "cross" otherwise (a < b + p)."""
    if len(lhs_vars) == 1 and lhs_vars == rhs_vars:
        return "self"
    if not lhs_vars or not rhs_vars:
        return "const"
    return "cross"


def graph_diameter(edges):
    """The most edges on a shortest path between two nodes of an undirected graph, over
    every connected component; 0 for a graph with no edges."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    diameter = 0
    for start in adj:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        diameter = max(diameter, max(dist.values()))
    return diameter


def arithmetic_structure(trees, sorts):
    """The Arithmetic structure columns: what the `date ± period` operations on variables
    connect, and how large their steps are, counted over the date comparisons that hold
    them. Self-referential operations (a < a + p) count only in n_self_ops, self_op_frac
    and sum_abs_days_all."""
    f = dict.fromkeys(ARITHMETIC_STRUCTURE, 0)
    edges = set()
    for t in trees:
        for clause in flatten(to_nnf(t), "&&"):
            in_disjunction = any(isinstance(n, BinOp) and n.op == "||" for n in walk(clause))
            for atom in walk(clause):
                if not (is_comparison(atom) and atom.lhs.sort == "date"):
                    continue
                f["n_date_eq"] += atom.op == "=="
                f["n_date_neq"] += atom.op == "!="
                ops = [n for n in walk(atom) if isinstance(n, DateAdd) and not is_ground_date(n)]
                if not ops:
                    continue
                lhs, rhs = atom_variables(atom.lhs, sorts), atom_variables(atom.rhs, sorts)
                kind = period_op_kind(lhs, rhs)
                f[f"n_{kind}_ops"] += len(ops)
                if kind == "self":
                    continue
                edges |= {(u, v) for u in lhs for v in rhs if u != v}
                for n in ops:
                    p = n.period
                    f["n_arith_in_disj" if in_disjunction else "n_arith_unit"] += 1
                    f["n_arith_neq"] += atom.op == "!="
                    f["n_year_ops"] += p.ny != 0
                    f["n_month_ops"] += p.nm != 0
                    f["n_day_ops"] += p.nd != 0
                    f["n_mixed_ops"] += p.months != 0 and p.nd != 0
                    f["n_day_steps_ge28"] += abs(p.nd) >= 28
                    f["sum_abs_days"] += abs(p.nd)
                    f["sum_abs_months"] += abs(p.months)
                    f["max_abs_days_nonself"] = max(f["max_abs_days_nonself"], abs(p.nd))
                    f["max_abs_months_nonself"] = max(f["max_abs_months_nonself"], abs(p.months))
    n_ops = f["n_self_ops"] + f["n_cross_ops"] + f["n_const_ops"]
    f["self_op_frac"] = f["n_self_ops"] / n_ops if n_ops else 0.0
    f["sum_abs_days_all"] = sum(abs(n.period.nd) for t in trees for n in walk(t)
                                if isinstance(n, DateAdd) and not is_ground_date(n))
    f["arith_graph_diameter"] = graph_diameter(edges)
    f["n_arith_vars"] = len({v for e in edges for v in e})
    years = literal_years(trees)
    f["lit_year_span"] = max(years) - min(years) if years else 0
    return f


def literal_years(trees):
    """Years of the literal dates, without the calendar's first and last day (the injected
    bounds)."""
    return [n.y.value for t in trees for n in walk(t)
            if isinstance(n, DateCtor) and all(isinstance(c, IntLit) for c in (n.y, n.m, n.d))
            and (n.y.value, n.m.value, n.d.value) not in ((1, 1, 1), (9999, 12, 31))]


def chain_abs_days(n):
    """Sum of |days| over a chain of `± period` steps, down to the date it starts from."""
    total = 0
    while isinstance(n, DateAdd):
        total += abs(n.period.nd)
        n = n.base
    return total


def representation_demands(trees):
    """The Representation demands columns: which form each date value is needed in, how the
    date comparisons are written, and how far the literals are from the encodings' epoch."""
    f = dict.fromkeys(REPRESENTATION_DEMANDS, 0)
    ym_bases, day_bases, field_bases = set(), set(), set()
    kinds = {}                                    # date variable -> kinds of step applied to it
    for t in trees:
        for n in walk(t):
            if isinstance(n, DateAdd) and not is_ground_date(n):
                kind = "ym" if n.period.months else "day"
                (ym_bases if kind == "ym" else day_bases).add(canonical(n.base))
                root = date_chain_base(n)
                if isinstance(root, Var):
                    kinds.setdefault(root.name, set()).add(kind)
                if isinstance(n.base, DateAdd):
                    f["n_chained_ops"] += 1
                    f["n_arith_kind_switches"] += (n.base.period.months != 0) != (kind == "ym")
                f["max_chain_abs_days"] = max(f["max_chain_abs_days"], chain_abs_days(n))
            elif isinstance(n, Field) and not is_ground_date(n.base):
                field_bases.add(canonical(n.base))
            elif is_comparison(n) and n.lhs.sort == "date":
                ground = [is_ground_date(s) for s in (n.lhs, n.rhs)]
                f["n_lit_date_cmps"] += any(ground) and not all(ground)
                f["n_var_var_cmps"] += isinstance(n.lhs, Var) and isinstance(n.rhs, Var)
                f["n_date_ordering_cmps"] += n.op in ORDERING
    f["n_ym_bases"] = len(ym_bases)
    f["n_day_bases"] = len(day_bases)
    f["n_field_bases"] = len(field_bases)
    f["n_vars_both_arith"] = sum(1 for k in kinds.values() if len(k) == 2)
    f["max_lit_year_dist"] = max((abs(y - EPOCH_YEAR) for y in literal_years(trees)), default=0)
    return f


def encoding_emulation(trees, sorts):
    """The Encoding emulation and Encoding contrasts columns, from emulate_encodings.py."""
    counts = emulate(trees, sorts)
    f = {f"emu_{e}_{m}": counts[e][m] for e, ms in EMULATED.items() for m in ms}
    for measure in ("divmod", "ite"):
        for a, b in combinations(sorted(ENCODINGS), 2):
            f[f"log2_{measure}_{a}_vs_{b}"] = math.log2(
                (counts[a][measure] + 1) / (counts[b][measure] + 1))
    return f


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def features_for(entry, corpus):
    sorts = parse_declarations(entry["declarations"])
    trees = [parse_constraint(s, sorts) for s in entry["constraints"]]
    conjuncts = top_level_conjuncts(trees)
    f = {"id": entry.get("id", ""), "corpus": corpus}

    # ---- search space -------------------------------------------------
    declared = {"date": [], "int": [], "bool": []}
    for name, s in sorts.items():
        if s in declared:
            declared[s].append(name)
    pinned = pinned_variables(conjuncts, sorts)
    n_vars = len(sorts)

    f["n_free_date_vars"] = len([v for v in declared["date"] if v not in pinned])
    f["n_free_int_vars"] = len([v for v in declared["int"] if v not in pinned])
    f["n_free_bool_vars"] = len([v for v in declared["bool"] if v not in pinned])
    f["pinned_var_frac"] = (len(pinned) / n_vars) if n_vars else 0.0
    for s in ("date", "int", "bool"):
        n_s = len(declared[s])
        n_pinned_s = sum(1 for v in declared[s] if v in pinned)
        f[f"pinned_{s}_frac"] = (n_pinned_s / n_s) if n_s else 0.0

    date_exprs = {
        canonical(n) for t in trees for n in walk(t)
        if n.sort == "date" and not isinstance(n, Var)
    }
    f["n_date_subexprs"] = len(date_exprs | set(declared["date"]))

    # ---- boolean skeleton ---------------------------------------------
    all_atoms = []
    under_impl = 0
    for t in trees:
        for atom, g in atoms_with_implication_flag(t):
            all_atoms.append(atom)
            under_impl += int(g)
    n_atoms = len(all_atoms)
    f["n_atoms"] = n_atoms

    date_cmps = [a for a in all_atoms if is_comparison(a) and a.lhs.sort == "date"]
    int_cmps = [a for a in all_atoms if is_comparison(a) and a.lhs.sort == "int"]
    all_cmps = [a for a in all_atoms if is_comparison(a)]
    f["n_date_comparisons"] = len(date_cmps)
    f["n_int_comparisons"] = len(int_cmps)
    f["ordering_cmp_frac"] = (
        sum(1 for a in all_cmps if a.op in ORDERING) / len(all_cmps)
    ) if all_cmps else 0.0

    n_or = sum(1 for t in trees for n in walk(t) if isinstance(n, BinOp) and n.op == "||")
    n_impl = sum(1 for t in trees for n in walk(t) if isinstance(n, BinOp) and n.op == "->")
    f["n_case_splits"] = n_or + n_impl

    widths = [len(flatten(n, "||")) for t in trees for n in walk(t)
              if isinstance(n, BinOp) and n.op == "||"]
    f["max_disjunction_width"] = max(widths) if widths else 1

    depths, neg_ord, tot_ord = [], 0, 0
    for t in trees:
        d, no, to = nnf_depth_and_polarity(t)
        depths.append(d)
        neg_ord += no
        tot_ord += to
    f["bool_alternation_depth"] = max(depths) if depths else 0
    f["neg_ordering_frac"] = (neg_ord / tot_ord) if tot_ord else 0.0

    f["implication_atom_frac"] = (under_impl / n_atoms) if n_atoms else 0.0
    # In NNF `a -> b` is `!a || b` and `!(a && b)` is `!a || !b`, so both count as splits.
    f["n_unit_constraints"] = sum(
        1 for t in trees for c in flatten(to_nnf(t), "&&")
        if not any(isinstance(n, BinOp) and n.op == "||" for n in walk(c))
    )

    # ---- date arithmetic ----------------------------------------------
    all_adds = [n for t in trees for n in walk(t) if isinstance(n, DateAdd)]
    ground_adds = [n for n in all_adds if is_ground_date(n)]
    live = [n for n in all_adds if not is_ground_date(n)]

    f["n_date_period_ops"] = len(live)
    f["ground_arith_frac"] = (len(ground_adds) / len(all_adds)) if all_adds else 0.0

    f["max_abs_years"] = max((abs(n.period.ny) for n in live), default=0)
    f["max_abs_months"] = max((abs(n.period.nm) for n in live), default=0)
    f["max_abs_days"] = max((abs(n.period.nd) for n in live), default=0)

    mixed = sum(1 for n in live if n.period.months != 0 and n.period.nd != 0)
    f["mixed_frac"] = (mixed / len(live)) if live else 0.0

    f["max_date_chain_len"] = max(
        (date_chain_len(n) for t in trees for n in walk(t) if n.sort == "date"),
        default=0,
    )

    # ---- component extraction ------------------------------------------
    fields = [n for t in trees for n in walk(t) if isinstance(n, Field)]
    f["n_dot_year"] = sum(1 for n in fields if n.fname == "year")
    f["n_dot_month"] = sum(1 for n in fields if n.fname == "month")
    f["n_dot_day"] = sum(1 for n in fields if n.fname == "day")
    denom = len(fields) + len(date_cmps)
    f["property_access_frac"] = (len(fields) / denom) if denom else 0.0

    f["n_symbolic_date_ctors"] = sum(
        1 for t in trees for n in walk(t)
        if isinstance(n, DateCtor) and not all(isinstance(c, IntLit) for c in (n.y, n.m, n.d))
    )

    # ---- calendar corners ----------------------------------------------
    lit_ctors = [
        n for t in trees for n in walk(t)
        if isinstance(n, DateCtor) and all(isinstance(c, IntLit) for c in (n.y, n.m, n.d))
    ]
    f["uses_feb29"] = int(any(n.m.value == 2 and n.d.value == 29 for n in lit_ctors))
    f["uses_leap_year"] = int(any(is_leap_year(n.y.value) for n in lit_ctors))
    # Day 28 or later: where month steps clamp and day steps roll over into the next month.
    f["near_month_end_frac"] = (
        sum(1 for n in lit_ctors if n.d.value >= 28) / len(lit_ctors)
    ) if lit_ctors else 0.0

    # ---- variable coupling ----------------------------------------------
    edges = set()
    for a in all_atoms:
        vs = sorted(atom_variables(a, sorts))
        for u, v in combinations(vs, 2):
            edges.add((u, v))

    parent = {v: v for v in sorts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    if n_vars:
        comps = {}
        for v in sorts:
            comps.setdefault(find(v), []).append(v)
        f["n_components"] = len(comps)
        f["largest_component_frac"] = max(len(c) for c in comps.values()) / n_vars
        possible = n_vars * (n_vars - 1) / 2
        f["graph_density"] = (len(edges) / possible) if possible else 0.0
    else:
        f["n_components"] = 0
        f["largest_component_frac"] = 0.0
        f["graph_density"] = 0.0

    f["mixed_sort_coupling"] = sum(
        1 for a in all_atoms
        if any(isinstance(n, Field) for n in walk(a))
        and any(isinstance(n, Var) and sorts.get(n.name) == "int" for n in walk(a))
    )

    # ---- arithmetic structure -------------------------------------------
    f.update(arithmetic_structure(trees, sorts))

    # ---- representation demands, encoding emulation and contrasts ------
    f.update(representation_demands(trees))
    f.update(encoding_emulation(trees, sorts))

    # ---- labels (solver output, NOT features) ---------------------------
    f["label_execution_time"] = entry.get("execution_time")
    entry_id = str(entry.get("id", ""))
    f["label_status"] = entry_id.split("-")[1] if entry_id.startswith("grammar-") else None

    return f


ARITHMETIC_STRUCTURE = [
    "n_self_ops", "n_cross_ops", "n_const_ops", "self_op_frac",
    "n_year_ops", "n_month_ops", "n_day_ops", "n_mixed_ops", "n_day_steps_ge28",
    "sum_abs_days", "sum_abs_months", "max_abs_days_nonself", "max_abs_months_nonself",
    "sum_abs_days_all", "n_arith_unit", "n_arith_in_disj", "n_arith_neq",
    "n_date_eq", "n_date_neq", "arith_graph_diameter", "n_arith_vars", "lit_year_span"]

# Year of DateSat's epoch, 2000-03-01, which the day-count forms count from.
EPOCH_YEAR = 2000
REPRESENTATION_DEMANDS = [
    "n_ym_bases", "n_day_bases", "n_field_bases", "n_vars_both_arith",
    "n_arith_kind_switches", "n_chained_ops", "max_chain_abs_days",
    "n_lit_date_cmps", "n_var_var_cmps", "n_date_ordering_cmps", "max_lit_year_dist"]
# The emulated counts kept per encoding. `simple` converts nothing, and `simple` and
# `alpha_beta` write every date comparison component by component, so those columns would
# repeat other ones.
EMULATED = {"simple": ("divmod", "ite"),
            "epoch_days": ("divmod", "ite", "conversions"),
            "hybrid_ymd": ("divmod", "ite", "conversions", "lex_cmps"),
            "hybrid_epoch": ("divmod", "ite", "conversions", "lex_cmps"),
            "hybrid_both": ("divmod", "ite", "conversions", "lex_cmps"),
            "alpha_beta": ("divmod", "ite", "conversions")}
ENCODING_EMULATION = [f"emu_{e}_{m}" for e, ms in EMULATED.items() for m in ms]
ENCODING_CONTRASTS = [f"log2_{measure}_{a}_vs_{b}" for measure in ("divmod", "ite")
                      for a, b in combinations(sorted(ENCODINGS), 2)]

FEATURE_GROUPS = [
    ("Search space", [
        "n_free_date_vars", "n_free_int_vars", "n_free_bool_vars", "pinned_var_frac",
        "pinned_date_frac", "pinned_int_frac", "pinned_bool_frac", "n_date_subexprs"]),
    ("Boolean skeleton", [
        "n_atoms", "n_date_comparisons", "n_int_comparisons", "ordering_cmp_frac",
        "n_case_splits", "max_disjunction_width", "bool_alternation_depth",
        "implication_atom_frac", "neg_ordering_frac", "n_unit_constraints"]),
    ("Date arithmetic", [
        "n_date_period_ops", "ground_arith_frac", "max_abs_years", "max_abs_months",
        "max_abs_days", "mixed_frac", "max_date_chain_len"]),
    ("Component extraction", [
        "n_dot_year", "n_dot_month", "n_dot_day", "property_access_frac",
        "n_symbolic_date_ctors"]),
    ("Calendar corners", ["uses_feb29", "uses_leap_year", "near_month_end_frac"]),
    ("Variable coupling", [
        "n_components", "largest_component_frac", "graph_density", "mixed_sort_coupling"]),
    ("Arithmetic structure", ARITHMETIC_STRUCTURE),
    ("Representation demands", REPRESENTATION_DEMANDS),
    ("Encoding emulation", ENCODING_EMULATION),
    ("Encoding contrasts", ENCODING_CONTRASTS),
]

COLUMNS = (["id", "corpus"]
           + [f for _, feats in FEATURE_GROUPS for f in feats]
           + ["label_status", "label_execution_time"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(OUTPUTS / "features.csv"))
    ap.add_argument("--dataset-root", default=str(DATASET_ROOT),
                    help="dataset root containing the three *_constraints directories")
    ap.add_argument("--bounds", choices=("keep", "strip"), default="strip",
                    help="keep or strip the atoms utils/bounds.py injected (default: strip)")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    print(f"dataset root: {root}")
    print(f"injected bounds: {args.bounds}")
    rows = []
    for corpus, rel in DATASETS.items():
        entries = load(root / rel)
        for e in entries:
            if args.bounds == "strip":
                e = strip_injected_bounds(e)
            rows.append(features_for(e, corpus))
        print(f"  {corpus}: {len(entries)} instances")

    out = Path(args.output)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})

    # Sidecar so later stages (build_report.py) can say how these features were made.
    meta = out.with_name(out.stem + "_meta.json")
    meta.write_text(json.dumps({"bounds": args.bounds, "dataset_root": str(root)}, indent=1))

    n_feat = len([c for c in COLUMNS if not c.startswith(("id", "corpus", "label_"))])
    print(f"\nWrote {len(rows)} rows x {len(COLUMNS)} columns to {out}")
    print(f"  {n_feat} features, 2 labels, 2 identifiers")


if __name__ == "__main__":
    main()
