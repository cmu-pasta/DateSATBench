# features(5): the 36 columns of features.csv

Each entry gives the feature's meaning in one or two lines, then a small example with the
value it produces. `test_features_md.py` runs every example below through
`extract_features.py`, so the numbers are checked, not hand-computed.

Conventions that apply to all features:

- An instance's constraint list is treated as **one big AND**.
- Period arithmetic is **folded first**: `Period(1,0,0) * 2 + Period(0,0,3)` becomes the single
constant `(2y, 0m, 3d)` and never counts as an operation.
- Injected bounds are **stripped by default**; see INJECTED BOUNDS below. The examples in this
file have no bounds.
- A ratio whose denominator is 0 is written as 0.
- In the examples, `a`–`e` are `date` variables, `x`, `y`, `n` are `int`, and `f`, `g` are
`bool`. Only the variables that appear are declared, unless an example adds one explicitly
as `d: date`. `(same example)` reuses the constraints of the example just above.

Sections follow the column order of `features.csv` and of the heatmaps.

---



## INJECTED BOUNDS (`--bounds keep|strip`)

Features are extracted from `datesatbench_bounded/positive_years`, the variant the solver
results were run on. `tools/inject_bounds.py` appended two constraints to the end of every
entry for each date variable `d`:

```
d >= Date(1, 1, 1);  d <= Date(9999, 12, 31)
```

`extract_features.py --bounds` decides whether these count:

- `strip` (default): the bounds are removed before extraction. The script checks that the
last constraints of each entry are exactly these, and fails otherwise.
- `keep`: the bounds are part of the instance, exactly as the solver saw it.

Every instance gets the same bounds, so they add nothing that tells instances apart. They only
inflate counts. Affected features (mean over all 450 instances):

```
feature                keep     strip    instances that change
n_date_subexprs        14.61    12.61    450
n_atoms                22.78    10.62    450
n_date_comparisons     20.20     8.04    450
n_unit_constraints     17.10     4.94    450
ordering_cmp_frac       0.77     0.53    417
implication_atom_frac   0.12     0.21    151
property_access_frac    0.07     0.12    109
neg_ordering_frac       0.02     0.04     50
```

All other features are identical in both modes. Each bound mentions a single variable, so the
variable-coupling graph does not change; the bounds are `>=`/`<=`, not `==`, so nothing gets
pinned; and the two bound literals are not leap years. `n_date_subexprs` changes because the
two bound literals count as date values.

The chosen mode is recorded in `features_meta.json` and shown in `report.html`. The flag only
affects `features.csv`. To carry it through the analysis, rerun the downstream steps:

```
python analysis/extract_features.py --bounds keep
python analysis/join_results.py
python analysis/cluster.py
python analysis/plot_speedup_heatmap.py            # and --corpus llm|legal|grammar
python analysis/plot_feature_correlation.py
python analysis/build_report.py
```

---



## SEARCH SPACE

Several columns here depend on which variables are **pinned**. A variable `v` is pinned when
some top-level conjunct is `v == e` (either side) and `e` is already known: a literal, or an
expression whose variables are all pinned themselves. Pinning is closed transitively, so
`a == Date(2020,1,1)` pins `a`, and then `b == a + Period(0,0,5)` pins `b`. Only a top-level
`==` pins: `b > a`, `b <= Date(...)`, and an equality inside `||` or `->` leave the variable
free. For every sort, pinned + free = declared.



### n_free_date_vars

Declared date variables that are not pinned.

```
a == Date(2020,1,1);  b == a + Period(0,0,5);  c > b      →  1   (only c; a and b are pinned)
```



### n_free_int_vars

Same as `n_free_date_vars`, for `int` variables.

```
x == 3;  y > x                                            →  1   (y)
```



### n_free_bool_vars

Same as `n_free_date_vars`, for `bool` variables.

```
f == True;  g -> a < b                                    →  1   (g; f is pinned)
```



### pinned_var_frac

Pinned variables / all declared variables, over every sort together. The three columns
after it split the same count by sort, each over its own sort's declarations, so a corpus
that declares no `int` or `bool` variables gets 0 there.

```
a == Date(2020,1,1);  b > a;  x == 3;  f == True;  g -> b < a
                                                          →  0.6   (a, x, f pinned; of 5)
```



### pinned_date_frac

Pinned date variables / declared date variables.

```
(same example)                                            →  0.5   (a pinned, b free)
```



### pinned_int_frac

Pinned int variables / declared int variables.

```
(same example)                                            →  1.0   (x pinned)
```



### pinned_bool_frac

Pinned bool variables / declared bool variables. A bool is pinned by a top-level `f == True`
or `f == False`; the same equality as the left side of an `->` does not pin it. On the
shipped datasets this column is 0 for every instance: only legal declares bools, and its
bool equalities are almost all antecedents of an `->`. `cluster.py` and the correlation
plot leave constant columns out automatically.

```
(same example)                                            →  0.5   (f pinned, g free)
```



### n_date_subexprs

Number of **distinct** date-valued expressions, counting declared date variables and literal
dates. It measures how many date values the solver has to represent.

```
a + Period(0,1,0) < b;  a + Period(0,1,0) != Date(2020,5,1)
                                                          →  4   (a, b, a+1mo, Date(2020,5,1))
```

---



## BOOLEAN SKELETON

Four columns here were renamed so the name says what is measured; see RENAMED COLUMNS at
the end for the old names.



### n_atoms

Leaves of the Boolean formula: comparisons, bare `bool` variables, and `True`/`False`. A
bool equality such as `f == True` is one atom. Atoms that are neither date nor int
comparisons are bool atoms, so `n_atoms - n_date_comparisons - n_int_comparisons` counts
them.

```
a < Date(2020,1,1) || f;  x == 3                          →  3
```



### n_date_comparisons

Comparisons whose two operands are dates: variables, literal dates, or date arithmetic. A
comparison on a field such as `a.year == 2020` mentions a date variable but compares two
ints, so it is counted by `n_int_comparisons` instead; `property_access_frac` is built on
that split.

```
a < b;  a == Date(2020,1,1) + Period(0,0,1)               →  2
a < b;  a.year == 2020                                    →  1   (a.year == 2020 compares ints)
```



### n_int_comparisons

Comparisons between two ints. A field access such as `a.year` is an int, so it counts here.
Comparisons between bools (`f == True`, `f == g`) count in neither this nor
`n_date_comparisons`.

```
a.year == y;  a.month == 2;  a.day > 10                   →  3
a < b;  a.year == 2020                                    →  1
```



### ordering_cmp_frac

Ordering comparisons (`< <= > >=`) / all comparisons, bool comparisons included. Equality
fixes a value, ordering leaves a range.

```
a < b || b < c || a == c;  a != b                         →  0.5   (2 of 4)
```



### n_case_splits

Number of binary `||` and `->` operators as written, so a three-arm OR counts 2. Negation is
not pushed first: `!(a < b && b < c)` is a split in NNF but is not counted here (compare
`n_unit_constraints`, which does push it).

```
a < b || b < c || a == c                                  →  2
f -> a < b || b < c                                       →  2
```



### max_disjunction_width

Most arms in a single `||` chain. It is 1 when there is no `||`. An `->` is not counted as
an arm, so `f -> a < b` has width 1.

```
a < b || b < c || a == c                                  →  3
f -> a < b                                                →  1
```



### bool_alternation_depth

How many times AND and OR alternate on the way from a constraint root down to an atom, after
`->` is rewritten as `!a || b` and negations are pushed to the leaves. A lone atom is 0 and a
flat chain is 1. Parentheses that keep the same connective add nothing, which is why this is
an alternation depth rather than a nesting depth.

```
a < b                                                     →  0
a < b || b < c || a == c                                  →  1
a < b || (b < c && a != c)                                →  2
```



### implication_atom_frac

Atoms with an `->` somewhere above them / all atoms. Both sides of the arrow count, the
antecedent as well as the consequent.

```
f -> a < b;  a > Date(2000,1,1)                           →  0.667   (f and a<b, of 3)
```



### neg_ordering_frac

Ordering comparisons that end up negated after pushing `!` to the leaves / all ordering
comparisons. The left side of an `->` counts as negated. Equalities are on neither side of
the ratio.

```
!(a < b);  a <= Date(2030,1,1)                            →  0.5
f -> a < b;  a != b                                       →  0
```



### n_unit_constraints

Top-level conjuncts that are a single literal once negations are pushed to the leaves: one
comparison or bool variable, possibly negated, with no `||` around it. Checked after pushing
negations, not on the surface syntax: `a -> b` (= `!a || b`) and `!(a && b)` (= `!a || !b`)
are splits even though no `||` is written, while `!(a || b)` (= `!a && !b`) gives two unit
facts. This matters mostly for legal, which uses `->` heavily.

```
a < b || b < c;  a != b;  x == 3                          →  2
f -> a < b;  a != b                                       →  1
!(a < b || b < a)                                         →  2
```

---



## DATE ARITHMETIC



### n_date_period_ops

`date ± period` operations that involve a variable. Operations on a literal date are
excluded because they fold to a constant (see `ground_arith_frac`).

```
b == a + Period(1,2,40);  b < Date(2020,1,1) + Period(0,0,10);  a > a - Period(0,0,3)
                                                          →  2   (the Date(2020,1,1) one folds)
```



### ground_arith_frac

`date ± period` operations on a literal date / all `date ± period` operations.

```
(same example)                                            →  0.333   (1 of 3)
```



### max_abs_years

Largest **years** component of any period added to a variable date. Like `n_date_period_ops`,
periods added to a literal date are ignored.

```
b == a + Period(1,2,40);  a > a - Period(3,0,0)           →  3
```



### max_abs_months

Largest **months** component, same rules.

```
b == a + Period(1,2,40);  a > a - Period(3,0,0)           →  2
```



### max_abs_days

Largest **days** component, same rules.

```
b == a + Period(1,2,40);  a > a - Period(3,0,0)           →  40
```



### mixed_frac

Fraction of `n_date_period_ops` whose period has **both** days and months (years count as
months).

```
b == a + Period(1,2,40);  a > a - Period(0,0,3)           →  0.5   (the first has both)
```



### max_date_chain_len

Longest run of `± period` applied one after another to the same date. Calendar arithmetic
is not associative, so `(a + 1mo) + 1d` is not folded into one step.

```
b == a                                                    →  0
b == a + Period(0,1,1)                                    →  1
b == (a + Period(0,1,0)) + Period(0,0,1)                  →  2
```

---



## COMPONENT EXTRACTION



### n_dot_year

Number of `.year` accesses.

```
a.year == y;  a.month == 2;  a.day > 10                   →  1
```



### n_dot_month

Number of `.month` accesses.

```
(same example)                                            →  1
```



### n_dot_day

Number of `.day` accesses.

```
(same example)                                            →  1
```



### property_access_frac

Field accesses / (field accesses + date comparisons). 1.0 means the instance works entirely
on year/month/day numbers, 0.0 means it works entirely on whole dates.

```
a.year == y;  a.month == 2;  a.day > 10;  a < Date(2020,2,29)
                                                          →  0.75   (3 / (3 + 1))
```



### n_symbolic_date_ctors

`Date(...)` calls with a non-literal argument, which build a date from numbers (the reverse of
a field access).

```
b == Date(a.year + 1, a.month, 1)                         →  1
```

---



## CALENDAR CORNERS

Both flags look only at **literal** dates, `Date(y, m, d)` with three integer arguments. A
symbolic constructor such as `Date(a.year, 2, 29)` is counted by `n_symbolic_date_ctors`,
not here.



### uses_feb29

1 if any literal `Date(y, 2, 29)` appears, otherwise 0.

```
a < Date(2020,2,29)                                       →  1
```



### uses_leap_year

1 if any literal date falls in a leap year, otherwise 0. A year is a leap year when it is
divisible by 4, except century years, which must be divisible by 400. Arithmetic on such a
date can land on or step over Feb 29, so the encoding's leap-year rule is exercised even
when no `Date(y, 2, 29)` is written.

```
a < Date(2024,3,1)                                        →  1
a < Date(2000,3,1)                                        →  1   (400-year rule)
a < Date(1900,3,1)                                        →  0   (100-year rule)
a < Date(2024,2,29) + Period(0,0,1);  b == Date(a.year, 2, 29)
                                                          →  1   (the literal; the symbolic one is ignored)
```

---



## VARIABLE COUPLING

The graph has one node per declared variable and an edge between any two variables that
appear in the same atom.

### n_components

Number of connected groups of variables, i.e. independent subproblems.

```
a < b;  b < c;  d: date                                   →  2   ({a,b,c}, {d})
```



### largest_component_frac

Size of the largest group / number of variables.

```
(same example)                                            →  0.75
```



### graph_density

Edges / possible edges, where possible edges = n(n-1)/2.

```
(same example)                                            →  0.333   (2 of 6)
```



### mixed_sort_coupling

Atoms that tie a date field to an `int` **variable** (not a literal).

```
a.year == y;  a.month == 2                                →  1   (only the first)
```

---

