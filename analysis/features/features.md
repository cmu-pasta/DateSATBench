# features(5): the 106 columns of features.csv

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
results were run on. `datesatbench/utils/bounds.py inject` appended two constraints to the end of every
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
near_month_end_frac     0.48     0.23    426
ordering_cmp_frac       0.77     0.53    417
implication_atom_frac   0.12     0.21    151
property_access_frac    0.07     0.12    109
neg_ordering_frac       0.02     0.04     50
```

Four columns of REPRESENTATION DEMANDS and ENCODING EMULATION, added after this table was
made, count the bounds too, because they count comparisons with a literal date:
`n_lit_date_cmps` and `n_date_ordering_cmps`, two more per date variable, and
`emu_hybrid_ymd_lex_cmps` and `emu_hybrid_epoch_lex_cmps`, which count a bound whenever its
variable is in (year, month, day) form by the end of the instance. The bounds add no `div`,
`mod` or if-then-else terms under any encoding.

All other features are identical in both modes. Each bound mentions a single variable, so the
variable-coupling graph does not change; the bounds are `>=`/`<=`, not `==`, so nothing gets
pinned; and the two bound literals are not leap years. `n_date_subexprs` changes because the
two bound literals count as date values. `near_month_end_frac` changes because
`Date(9999, 12, 31)` has day 31 and `Date(1, 1, 1)` has day 1: each date variable adds one
literal that counts and one that does not, which pulls the share towards 0.5. Only instances
whose share is already exactly 0.5 keep their value.

The chosen mode is recorded in `features_meta.json` and shown in `report.html`. The flag only
affects `features.csv`. To carry it through the analysis, rerun the downstream steps:

```
export DATESAT_TIMEOUT_MS=20000                    # the --timeout the results were run with
python -m analysis.features.extract_features --bounds keep
python -m analysis.stats.join_results
python -m analysis.stats.cluster
python -m analysis.stats.plot_speedup_heatmap            # and --corpus llm|legal|grammar
python -m analysis.stats.plot_feature_correlation
python -m analysis.report.build_report
```

`join_results.py` and `build_report.py` refuse to run without a timeout: pass `--timeout <ms>`
or set `DATESAT_TIMEOUT_MS`. Timed-out runs count at that timeout when computing speedups.

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

All three columns look only at **literal** dates, `Date(y, m, d)` with three integer
arguments. A symbolic constructor such as `Date(a.year, 2, 29)` is counted by
`n_symbolic_date_ctors`, not here.



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



### near_month_end_frac

Literal dates whose day is 28 or later / all literal dates. Only the day number is checked,
not the length of the month: the last day of every month counts, and so does day 28 of a
31-day month. Month ends are where calendar arithmetic stops being uniform: a month or year
step from day 29, 30 or 31 can run past the end of a shorter month and is clamped to its
last day (Jan 31 + 1 month is Feb 28), and a day step from the last day of a month rolls
over into the next month. The grammar generator's `near_month_end_prob` puts each literal on
day 28 or later with that probability; with literals drawn uniformly over all dates, about
1 in 9 lands there.

```
a >= Date(2020,1,5);  a <= Date(2020,1,31);  b == Date(2021,6,15)
                                                          →  0.333   (day 31, of 3)
a < Date(2023,2,28);  b > Date(2024,2,29)                 →  1   (both are the last day of February)
a < Date(2023,3,28)                                       →  1   (day 28 counts, though March has 31)
a < b + Period(0,1,0)                                     →  0   (no literal dates)
a < Date(2024,1,5);  b == Date(a.year, 2, 29)             →  0   (the symbolic one is ignored)
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




## ARITHMETIC STRUCTURE

What the `date ± period` operations on variables connect, and how large their steps are.
`n_date_period_ops` and the `max_abs_*` columns above treat every such operation alike, but
two operations with the same period can cost a solver very different amounts: `a != a +
Period(0,0,1000)` compares a date with itself and is decided almost for free, while `a != b
+ Period(0,0,1000)` ties two dates together.

Each operation is classified by the date comparison it sits in, from the variables (of any
sort) that each side of the comparison mentions:

- **self-referential**: both sides mention the same single variable, `a < a + Period(0,1,0)`;
- **constant**: one side mentions no variable, `a + Period(1,0,0) < Date(2020,1,1)`;
- **cross**: anything else, `b < a + Period(0,1,0)`.

Only `date ± period` operations on a variable count, as in `n_date_period_ops`; operations
on a literal date fold to a constant. Unless a column says otherwise, it counts the cross
and constant operations and leaves the self-referential ones out. A comparison on a field,
such as `(a + Period(0,1,0)).year == 2020`, compares ints and holds no counted operation.



### n_self_ops

Self-referential operations.

```
b < a + Period(0,1,0);  a != a + Period(0,0,3);  a + Period(1,0,0) < Date(2020,1,1)
                                                          →  1   (a != a + 3d)
```



### n_cross_ops

Cross operations.

```
(same example)                                            →  1   (b < a + 1mo)
```



### n_const_ops

Constant operations.

```
(same example)                                            →  1   (a + 1y < Date(2020,1,1))
```



### self_op_frac

`n_self_ops` / all three kinds.

```
(same example)                                            →  0.333   (1 of 3)
```



### n_year_ops

Operations whose period has a years part.

```
b < a + Period(1,2,0);  c > b - Period(0,0,40);  a != a + Period(3,0,0)
                                                          →  1   (the self-referential 3y is left out)
```



### n_month_ops

Operations whose period has a months part. Years are not converted to months here, so
`Period(1,0,0)` does not count.

```
(same example)                                            →  1
```



### n_day_ops

Operations whose period has a days part.

```
(same example)                                            →  1
```



### n_mixed_ops

Operations whose period has both days and months, with years counting as months, as in
`mixed_frac`.

```
b < a + Period(1,0,40);  c > b - Period(0,2,0)            →  1   (the first)
```



### n_day_steps_ge28

Operations whose days part is 28 or more (either sign), a step that can cross a month end
whatever day it starts from.

```
b < a + Period(0,0,28);  c > b - Period(0,1,27);  a != a + Period(0,0,90)
                                                          →  1   (the 28)
```



### sum_abs_days

Sum of the days parts, ignoring their sign. `max_abs_days` records only the largest.

```
b < a + Period(0,1,40);  c > b - Period(0,0,5);  a != a + Period(0,0,90)
                                                          →  45   (40 + 5)
```



### sum_abs_months

Sum of the months parts, ignoring their sign, with years counting as 12 months each.

```
b < a + Period(1,2,0);  c > b - Period(0,5,0);  a != a + Period(3,0,0)
                                                          →  19   (14 + 5)
```



### max_abs_days_nonself

Largest days part, ignoring its sign: `max_abs_days` without the self-referential operations.

```
b < a + Period(0,0,40);  a != a + Period(0,0,1000)        →  40   (max_abs_days is 1000)
```



### max_abs_months_nonself

Largest months part, ignoring its sign, with years counting as 12 months each, and without
the self-referential operations. (`max_abs_months` counts only the months part.)

```
b < a + Period(1,2,0);  a != a + Period(0,50,0)           →  14   (max_abs_months is 50)
```



### sum_abs_days_all

Sum of the days parts, ignoring their sign, over **every** operation on a variable:
self-referential ones included, and wherever the operation sits, a field access included.
An encoding that steps through days one at a time pays for each of them.

```
b < a + Period(0,1,40);  a != a + Period(0,0,90);  b > Date(2020,1,1) + Period(0,0,7)
                                                          →  130   (the literal's 7 folds)
```



### n_arith_unit

Operations in a top-level conjunct that has no `||` once negations are pushed to the leaves
(see `n_unit_constraints`), so the solver must satisfy them.

```
b > a + Period(0,0,7);  c > b + Period(0,0,7) || c == a;  f -> c < a + Period(0,1,0)
                                                          →  1   (the first)
```



### n_arith_in_disj

Operations in a top-level conjunct with an `||`, including one written as `->`.

```
(same example)                                            →  2
```



### n_arith_neq

Operations in a `!=` comparison. The comparison counts as written: a negated `==` is not a
`!=`.

```
b != a + Period(0,1,0);  !(c == b + Period(0,0,1));  c < a + Period(1,0,0)
                                                          →  1   (the first)
```



### n_date_eq

Date comparisons with `==`, counted as written, as for `n_arith_neq`. A field comparison
such as `a.year == 2020` compares ints and does not count.

```
a == Date(2020,1,1);  b != a;  !(c == b);  a.year == 2020 →  2   (a == ..., c == b)
```



### n_date_neq

Date comparisons with `!=`, counted the same way.

```
(same example)                                            →  1
```



### arith_graph_diameter

The graph has one node per variable and an edge between two variables that sit on opposite
sides of a comparison holding a cross operation. This is the most edges on a shortest path
between two of its variables: how long a chain of date arithmetic links one variable to
another.

```
b >= a + Period(0,0,7);  c >= b + Period(0,0,7);  d >= c + Period(0,0,7);  e < d
                                                          →  3   (a-b-c-d; e < d has no arithmetic)
```



### n_arith_vars

Variables with at least one edge in that graph.

```
(same example)                                            →  4
```



### lit_year_span

Largest minus smallest year over the literal dates, 0 when there are none. `Date(1,1,1)` and
`Date(9999,12,31)`, the calendar's first and last day, are left out, because they are the
injected bounds. As in CALENDAR CORNERS, only literal dates count.

```
a > Date(2010,5,1);  a < Date(2024,1,1);  b == Date(a.year,2,29);  a >= Date(1,1,1)
                                                          →  14
```

---



## REPRESENTATION DEMANDS

Which form each date value is needed in, and how the date comparisons are written. Section 4
of the DateSAT paper explains the encodings' speed by the form they keep a date in. An
encoding that keeps a **day count** (`epoch_days`) adds days with one addition and compares
two dates with one integer comparison, but must recover the year, month and day with
integer divisions for every year or month step and every field read. An encoding that keeps
the **year, month and day** (`simple`) adds years and months cheaply, but adds days one day
at a time and compares dates lexicographically, which is a disjunction. The hybrid encodings
keep both and convert when an operation needs the form that is out of date. These columns
count those demands from the constraint text alone, without following any one encoding;
ENCODING EMULATION below follows each encoding exactly.

A **year/month step** is a `date ± period` operation on a variable whose period has a years
or months part; a **day step** is one whose period has only a days part. The **date a step
starts from** is the date expression it is applied to, and two starting dates are the same
when they are written the same way. As in DATE ARITHMETIC, steps on a literal date fold to a
constant and do not count.



### n_ym_bases

Distinct dates that a year/month step starts from. An encoding that keeps a day count
decodes each of them to (year, month, day).

```
b == a + Period(0,1,0);  c == a + Period(1,0,5);  d == (a + Period(0,0,3)) + Period(0,2,0)
                                                          →  2   (a, and a + 3d)
```



### n_day_bases

Distinct dates that a day step starts from.

```
(same example)                                            →  1   (a)
```



### n_field_bases

Distinct dates whose `.year`, `.month` or `.day` is read. An encoding that keeps a day count
decodes each of them.

```
a.year == 2020;  a.month == 3;  (b + Period(0,0,1)).day == 1
                                                          →  2   (a, and b + 1d)
```



### n_vars_both_arith

Date variables that both a day step and a year/month step start from, directly or through a
chain of steps. The paper's hybrid encoding is built on the observation that real
constraints rarely mix the two on one date.

```
b == a + Period(0,1,0);  c == a + Period(0,0,5);  d == b + Period(0,0,1)
                                                          →  1   (a; b has day steps only)
```



### n_arith_kind_switches

Steps applied to the result of a step of the other kind: a year/month step on the result of
a day step, or the reverse. In the hybrid encodings each one converts the date to the other
form.

```
b == (a + Period(0,0,8)) + Period(0,1,0);  c == (a + Period(0,1,0)) + Period(0,2,0)
                                                          →  1   (+1m on a + 8d)
```



### n_chained_ops

Steps applied to the result of another step. `max_date_chain_len` records only the longest
chain.

```
(same example)                                            →  2
```



### max_chain_abs_days

Largest sum of the days parts, ignoring their sign, along one chain of steps. `simple`
unrolls day arithmetic one day at a time, so this is how deep its deepest unrolling goes,
the paper's *logical depth*. Unlike the ARITHMETIC STRUCTURE columns, it counts every step
on a variable, self-referential ones and field accesses included.

```
b == (a + Period(0,0,20)) - Period(0,1,15);  c == a + Period(0,0,30)
                                                          →  35   (20 + 15)
```



### n_lit_date_cmps

Date comparisons with a literal date on one side and not on the other. `simple` writes each
one lexicographically, `epoch_days` as one integer comparison. With `--bounds keep` the
injected bounds count here, two per date variable.

```
a < Date(2020,1,1);  b + Period(0,1,0) != Date(2020,5,1);  a < b;  a == b
                                                          →  2
```



### n_var_var_cmps

Date comparisons between two date variables, with no arithmetic on either side.

```
(same example)                                            →  2   (a < b, a == b)
```



### n_date_ordering_cmps

Date comparisons with `<`, `<=`, `>` or `>=`, each a disjunction when written
lexicographically. `ordering_cmp_frac` takes int and bool comparisons too.

```
(same example)                                            →  2   (a < Date(...), a < b)
```



### max_lit_year_dist

Largest distance in years from 2000, the year of DateSat's epoch 2000-03-01, over the
literal dates. `Date(1,1,1)` and `Date(9999,12,31)` are left out, as for `lit_year_span`,
because they are the injected bounds.

```
a > Date(1990,5,1);  a < Date(2024,1,1);  b >= Date(1,1,1)
                                                          →  24   (2024)
```

---



## ENCODING EMULATION

What each of DateSat's five encodings would emit for the instance, counted without solving
it. `emulate_encodings.py` walks the parsed constraints the way DateSat's builder does and
follows each encoding's rules from `datesat/symbolic_int/`: `simple` keeps (year, month,
day) and unrolls day steps one day at a time; `epoch_days` keeps a day count and decodes it
for year/month steps and field reads; `alpha_beta` keeps (months since the epoch, day of the
month) and goes through the day count when a day step leaves the month; `hybrid_ymd` and
`hybrid_epoch` keep both forms and convert when an operation needs the one that is out of
date. The hybrids start a variable in (year, month, day) form and as a day count
respectively, and which form is up to date changes as the constraints are built, so the
walk visits the constraints in DateSat's order and keeps the same state.

Four things are counted:

- **divmod**: integer `div` and `mod` terms, the paper's *arithmetic complexity*. The
  conversions between day counts and (year, month, day) and the leap-year test are made of
  them.
- **ite**: if-then-else terms, the paper's *logical depth* in count form.
- **conversions**: dates converted between the two forms, one per date value and direction.
- **lex_cmps**: date comparisons the hybrids write component by component, which are
  disjunctions for an ordering or `!=`. `simple` and `alpha_beta` write every date
  comparison this way and `epoch_days` none, so only the hybrids get this column.

A term DateSat builds twice from the same inputs is counted once, because Z3 shares
identical terms, and a term no assertion uses is not counted: reading only `.month` of a
day count uses 5 of the decoding's 8 divisions. The injected bounds are literal comparisons,
so with `--bounds keep` they change only the `lex_cmps` columns. The walk follows DateSat's
code and must be updated when DateSat's encodings change. Every divmod and ite example below
was checked against the formula DateSat builds.



### emu_simple_divmod

`div` and `mod` terms `simple` emits. Each day it unrolls tests for a leap year, three
`mod`s, so the count grows with the days added.

```
b == a + Period(0,0,5)                                    →  21   (5 days; a's and b's bounds)
```



### emu_simple_ite

If-then-else terms `simple` emits.

```
(same example)                                            →  46
```



### emu_epoch_days_divmod

`div` and `mod` terms `epoch_days` emits. A day step is one addition; a year/month step
decodes the date, adds the months, and encodes the result.

```
b == a + Period(0,0,5)                                    →  0
b == a + Period(0,1,0)                                    →  18   (decode a, add 1 month, encode)
a.month == 2                                              →  5   (the part of the decoding .month needs)
```



### emu_epoch_days_ite

If-then-else terms `epoch_days` emits.

```
b == a + Period(0,1,0)                                    →  11
```



### emu_epoch_days_conversions

Dates `epoch_days` converts.

```
(same example)                                            →  2   (a decoded, a + 1m encoded)
```



### emu_hybrid_ymd_divmod

`div` and `mod` terms `hybrid_ymd` emits. Every date variable starts in (year, month, day)
form, tied to a day count by an encoding, so it costs divisions before any arithmetic.

```
a < Date(2020,1,1);  a <= b                               →  16   (a and b tied to day counts)
```



### emu_hybrid_ymd_ite

If-then-else terms `hybrid_ymd` emits.

```
(same example)                                            →  10
```



### emu_hybrid_ymd_conversions

Dates `hybrid_ymd` converts.

```
b == a + Period(0,0,5)                                    →  4
```

`a` and `b` are encoded when they are declared. `a + 5d` is a day count, and comparing it
with `==` to `b`, which is in (year, month, day) form, decodes it into fresh year, month and
day variables, which are encoded again to tie them to it.



### emu_hybrid_ymd_lex_cmps

Date comparisons `hybrid_ymd` writes component by component: those whose sides are both in
(year, month, day) form.

```
a < Date(2020,1,1);  a <= b                               →  2
a != b + Period(0,0,5);  a < Date(2020,1,1)               →  1   (a < Date(...); b + 5d is a day count)
```



### emu_hybrid_epoch_divmod

`div` and `mod` terms `hybrid_epoch` emits. Its variables start as day counts, so they cost
nothing until an operation needs (year, month, day); the first one then decodes the date
and encodes the fresh variables back.

```
a < Date(2020,1,1);  a <= b                               →  0
b == a + Period(0,1,0)                                    →  36
```



### emu_hybrid_epoch_ite

If-then-else terms `hybrid_epoch` emits.

```
(same example)                                            →  19
```



### emu_hybrid_epoch_conversions

Dates `hybrid_epoch` converts.

```
(same example)                                            →  5
```



### emu_hybrid_epoch_lex_cmps

Date comparisons `hybrid_epoch` writes component by component. Once a field read has put a
variable in (year, month, day) form, it compares that variable with literals component by
component.

```
b > a + Period(0,0,7);  b.month == 2;  b < Date(2020,1,1) →  1   (the last one, after b.month)
```



### emu_alpha_beta_divmod

`div` and `mod` terms `alpha_beta` emits. A day step may leave the month, so it also emits a
conversion through the day count and back.

```
b == a + Period(0,0,5)                                    →  25
b == a + Period(0,1,0)                                    →  12   (a months step stays in alpha)
```



### emu_alpha_beta_ite

If-then-else terms `alpha_beta` emits.

```
b == a + Period(0,0,5)                                    →  17
```



### emu_alpha_beta_conversions

Dates `alpha_beta` converts.

```
(same example)                                            →  2   (a + 5d encoded and decoded)
```

---



## ENCODING CONTRASTS

How two encodings compare on the same instance. The router has one forest per pair of
encodings, each deciding which of the two is faster, and a decision tree can only split on
one column at a time: it cannot tell from `emu_epoch_days_divmod` and
`emu_hybrid_ymd_divmod` side by side which of the two is larger. These columns give it the
comparison directly, for every pair of encodings in the order the router's forests use:

    log2_<measure>_<a>_vs_<b> = log2((count of a + 1) / (count of b + 1))

for the two measures of the paper, `divmod` and `ite`, from ENCODING EMULATION. A value
below 0 means encoding `a` emits fewer such terms than `b`. The examples all use
`b == a + Period(0,1,5)`, for which the counts are:

```
             simple  epoch_days  hybrid_ymd  hybrid_epoch  alpha_beta
divmod           26          18          39            23          29
ite              51          11          23            13          22
```



### log2_divmod_alpha_beta_vs_epoch_days

```
b == a + Period(0,1,5)                                    →  0.659   (log2(30/19))
```



### log2_divmod_alpha_beta_vs_hybrid_epoch

```
(same example)                                            →  0.322
```



### log2_divmod_alpha_beta_vs_hybrid_ymd

```
(same example)                                            →  -0.415
```



### log2_divmod_alpha_beta_vs_simple

```
(same example)                                            →  0.152
```



### log2_divmod_epoch_days_vs_hybrid_epoch

```
(same example)                                            →  -0.337
```



### log2_divmod_epoch_days_vs_hybrid_ymd

```
(same example)                                            →  -1.074
```



### log2_divmod_epoch_days_vs_simple

```
(same example)                                            →  -0.507
```



### log2_divmod_hybrid_epoch_vs_hybrid_ymd

```
(same example)                                            →  -0.737
```



### log2_divmod_hybrid_epoch_vs_simple

```
(same example)                                            →  -0.170
```



### log2_divmod_hybrid_ymd_vs_simple

```
(same example)                                            →  0.567
```



### log2_ite_alpha_beta_vs_epoch_days

```
(same example)                                            →  0.939   (log2(23/12))
```



### log2_ite_alpha_beta_vs_hybrid_epoch

```
(same example)                                            →  0.716
```



### log2_ite_alpha_beta_vs_hybrid_ymd

```
(same example)                                            →  -0.061
```



### log2_ite_alpha_beta_vs_simple

```
(same example)                                            →  -1.177
```



### log2_ite_epoch_days_vs_hybrid_epoch

```
(same example)                                            →  -0.222
```



### log2_ite_epoch_days_vs_hybrid_ymd

```
(same example)                                            →  -1.000
```



### log2_ite_epoch_days_vs_simple

```
(same example)                                            →  -2.115
```



### log2_ite_hybrid_epoch_vs_hybrid_ymd

```
(same example)                                            →  -0.778
```



### log2_ite_hybrid_epoch_vs_simple

```
(same example)                                            →  -1.893
```



### log2_ite_hybrid_ymd_vs_simple

```
(same example)                                            →  -1.115
```

---
