# features(5): the 63 columns of features.csv

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

Features are extracted from `datesatbench/`, whose entries carry the bounds the solver
results were run with. `datesatbench/utils/bounds.py inject` appended two constraints to the end of every
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



## ENCODING COSTS

One column for each operation that a DateSat encoding pays heavily for. DateSat holds a date
in one of two **forms**: a **day count**, the number of days since its epoch 2000-03-01, or
**(year, month, day)**. The encodings keep different forms, so each one is slow on a
different operation:

- `simple` keeps (year, month, day). It adds days one day at a time, checking each day
against the length of the month, so its cost grows with the number of days added:
`sum_abs_days_all`.
- `alpha_beta` keeps the month and the day within the month. Adding days is one addition
when the result stays in the same month, and a conversion to a day count and back when it
does not. The solver cannot know in advance which case applies, so every step with a days
part carries both: `n_day_steps`.
- `epoch_days` keeps a day count. Adding days and comparing two dates are single integer
operations, but every date it must read as (year, month, day) costs a conversion by
integer division: `n_ymd_needs`.
- `hybrid_ymd` and `hybrid_epoch` hold a date in either form or in both, and build the
missing form only when an operation needs it, which is a **conversion**. They differ only
in the form a date variable starts in: (year, month, day) for `hybrid_ymd`, a day count
for `hybrid_epoch`. `form_balance` says which start saves conversions, and
`n_unavoidable_conversions` counts the conversions that neither start saves.

The columns count operations in the constraint text and do not follow DateSat's code.

A **step** is a `date ± period` operation on a variable. As in DATE ARITHMETIC, a step on a
literal date folds to a constant and does not count. The **date a step starts from** is the
date expression it is applied to, and two dates are the same when they are written the same
way.

The two hybrid columns rest on the form each operation needs:

- Reading `.year`, `.month` or `.day` needs (year, month, day).
- A step whose period has only a days part needs a day count. A step whose period has a
years or months part needs (year, month, day).
- A step gives (year, month, day) when its period has no days part, and a day count when it
has one.
- Two dates are compared in a common form. A literal date fits either form.

A **variable** here is a declared date variable, or a `Date(...)` with a part that is not a
literal. DateSat replaces such a `Date(...)` by an auxiliary date variable and sets its year,
month and day, so it is needed as (year, month, day). A variable is needed in the form that
the field reads and steps on it need. A variable compared with a step's result, by `==`,
`!=` or an ordering, is needed in the form that result is in. Comparisons with a literal
date, and comparisons between two variables, need neither form. A variable needed only as a
day count costs `hybrid_ymd` one conversion and `hybrid_epoch` none, a variable needed only
as (year, month, day) costs the reverse, and a variable needed in both forms costs one
conversion under either.

The order of the constraints is ignored. DateSat's exact count can depend on it, because a
variable that has already been converted has both forms when a later constraint uses it.



### sum_abs_days_all

Sum of the days parts, ignoring their sign, over every step: self-referential ones included,
and wherever the step sits, a field access included. This is how many days `simple` adds one
at a time.

```
b < a + Period(0,1,40);  a != a + Period(0,0,90);  b > Date(2020,1,1) + Period(0,0,7)
                                                          →  130   (the literal's 7 folds)
```



### n_day_steps

Steps whose period has a days part, with or without years or months, counted wherever they
sit, as for `sum_abs_days_all`. Each one carries a conversion to a day count and back under
`alpha_beta`, however small the days part is.

```
b == a + Period(0,0,5);  c == a + Period(0,1,3);  d == a + Period(1,0,0);
(b + Period(0,0,1)).day == 2
                                                          →  3   (all but the 1y)
b == a + Period(0,0,400)                                  →  1   (sum_abs_days_all is 400)
```



### n_ymd_needs

Distinct dates needed as (year, month, day): those a step with a years or months part starts
from, those whose `.year`, `.month` or `.day` is read, and every `Date(...)` with a part that
is not a literal. `epoch_days` converts each of them between a day count and
(year, month, day).

```
b == a + Period(0,1,0);  c == a + Period(1,0,5);  a.year == 2020;  d == b + Period(0,0,3);
c.month == 4
                                                          →  2   (a and c; b has only a day step)
b == Date(a.year, 2, 28)                                  →  2   (a, and the Date(...))
```



### form_balance

Variables needed only as a day count minus variables needed only as (year, month, day).
Below 0, starting the variables as (year, month, day) saves conversions, as `hybrid_ymd`
does; above 0, starting them as day counts does, as `hybrid_epoch` does.

```
a.month == 4;  b == a + Period(2,3,0);  b.year == 2028    →  -2   (a and b as (year, month, day))
b == a + Period(0,0,100);  a < b                          →  2    (a starts the step; b is compared with its result)
b == a + Period(0,0,10) + Period(0,1,0) + Period(0,0,20) + Period(2,0,0)
     + Period(0,0,30) + Period(0,3,0);  a < b             →  0    (a a day count, b (year, month, day))
```



### n_unavoidable_conversions

Conversions that every hybrid encoding pays, whatever form its variables start in. There are
two kinds:

- a variable needed in both forms;
- a **form switch**, a place where a step's result is needed in the form the step did not
make it in: a step that starts from it, a field read from it, or a comparison with another
step's result that is in the other form.

```
(same example)                                            →  5   (every step after the first)
a.year == 2020;  b == a + Period(0,0,5)                   →  1   (a, needed in both forms)
b == (a + Period(0,1,5)) + Period(0,1,0);  c == (a + Period(0,1,5)) + Period(0,0,3)
                                                          →  1   (+1m on a + 1m5d, a day count)
(a + Period(0,0,3)).month == 2;  a + Period(0,1,0) < b + Period(0,0,1)
                                                          →  3   (a in both forms; two form switches)
```
