# flags: swarm-testing switches for grammar.fan

Each entry gives the flag's meaning in one or two lines, then a small example of what the
generator produces under different settings, then why the flag is worth having.

The idea comes from swarm testing: before sampling each instance, the generator picks a
setting for every flag at random, and that choice holds for the whole instance. Some
instances then leave a feature out entirely, or come out much smaller or larger than usual,
which a single fixed grammar almost never does.

There are two kinds of flag:

- A **switch** names a feature. On means the grammar may produce that feature; off means it
never appears in the instance.
- A **count** has a set of allowed values, either a list or a range, and each instance gets
one value drawn uniformly from it. The list or range itself is a setting of the generator.

Conventions that apply to all flags:

- **All switches on, and every count at its current setting, reproduces the current
`grammar.fan`**, so the existing dataset stays reproducible.
- A flag is decided **once per instance**, not per constraint or per literal.
- Period arithmetic is **folded first**, as in `analysis/features/features.md`: the solver computes
`Period + Period` and `Period * k` before encoding (`datesat/core.py`), so a flag about a
Period, its fields or its size, is judged on the folded constant.

---



## SIZE



### num_constraints (count)

How many constraints the instance has. The current grammar hard-codes the list
`{5, 6, 7, 8, 9, 10, 15, 20}` as one alternative per count in `<constraint_list>`; this flag
turns that list into a setting. The list is `{5, 6, 7, 8, 9, 10}`.

```
num_constraints = 2:   D1 >= D0;  D1 <= (D0 + Period(0, 0, 30))
num_constraints = 5:   D1 >= D0;  D1 <= (D0 + Period(0, 0, 30));  D2 == (D1 + Period(0, 1, 0));
                       D2 < Date(2016, 3, 1);  D0 != D2
```

Counts below 5 are the obvious addition: the current grammar never produces an instance
with fewer than 5 constraints, yet a real question about dates is often just two or three of
them ("the response is due within 30 days of the bill").



### num_date_vars (count)

How many date variables the instance may use: with a value of `k`, every date variable is
drawn from `D0` … `D(k-1)`. The list is `{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`; the
current grammar always draws from all ten, `D0` … `D9`, which is the list `{10}`.

```
num_date_vars = 2:    D1 >= D0;  D1 <= (D0 + Period(0, 0, 30));  D0 > Date(2016, 3, 1)
num_date_vars = 10:   D7 >= D2;  D4 <= (D9 + Period(0, 0, 30));  D0 > Date(2016, 3, 1)
```

The value bounds the number of declared date variables rather than fixing it: declarations
are inferred from the constraints, so a name that is never drawn is never declared.

With ten names drawn uniformly, an instance spreads its constraints over nearly all of them
(9 on average in the current dataset), so few constraints share a variable and the instance
tends to fall apart into loosely coupled pieces. A small pool forces every constraint to talk
about the same few dates, which is where offsets start to interact (`b <= a + 30 days`,
`c >= b + 1 month`). Together with `num_constraints`, this flag sets how many constraints
each variable carries on average, which in turn shapes how likely an instance is to be
satisfiable.

A very small pool also makes a comparison of a variable with itself common. The sampler
redraws any constraint that contains a bare one, such as `D0 < D0`, whose truth depends
only on the operator. It keeps one with an offset, such as `D0 < (D0 - Period(0, 0, 3))`.
At 1, every comparison between variables compares the single date with itself through an
offset.

---



## SEARCH SPACE



### anchor_frac (count)

The share of the `num_date_vars` date variables that the instance pins to a literal date
with a top-level `D == Date(y, m, d)`. The share is drawn uniformly from the whole percents
0%, 1%, …, 25%, and the number of anchors is that share of `num_date_vars`, rounded down, so the
anchors never pin more than the share. Each anchor is a constraint of its own, added on top
of `num_constraints`, and pins a different variable that the rest of the instance uses; when
the instance uses fewer variables than that, every one it uses is pinned. The current
grammar adds none, which is the setting `{0%}`, though it still pins a variable now and then
by chance.

The range stops at 25% so that most variables stay free: free variables are what the solver
has to search over, and anchors are there to add the occasional concrete fact, not to
settle the instance.

Rounding down spreads the anchors evenly over what each `num_date_vars` allows. With 10
variables, 0 and 1 anchor each come up 10 times in 26, and 2 anchors 6 times in 26. With 8 variables it is 0 anchors 13 times in 26, 1 anchor 12 times and 2 anchors
once. With 3 or fewer variables it is always 0, and with 4 it is 1 only at exactly 25%.

```
num_date_vars = 5, anchor_frac = 12%:   0.6 anchors, rounded down to 0
    D1 >= D0;  D2 <= (D1 + Period(0, 0, 30));  D3 > D2;  D4 == (D3 + Period(0, 1, 0))
num_date_vars = 5, anchor_frac = 22%:   1.1 anchors, rounded down to 1
    D1 >= D0;  D2 <= (D1 + Period(0, 0, 30));  D3 > D2;  D4 == (D3 + Period(0, 1, 0));
    D0 == Date(2016, 3, 1)
```

A share rather than a number keeps the flag meaning the same at every `num_date_vars`: two
anchors pin everything in a 2-variable instance but barely anything in a 10-variable one. It
is also the quantity `analysis/features/features.md` measures, as `pinned_date_frac`.

Pinning follows `analysis/features/features.md`: an anchor pins its variable directly, and pinning
spreads from there, so `D2 == (D0 + Period(0, 1, 0))` pins `D2` as soon as `D0` is pinned.
The share of variables pinned can therefore end up above `anchor_frac`.

A pinned variable has only one possible value, so every constraint on it narrows its
neighbours directly: once `D0` is fixed, `D1 <= D0 + 30 days` is a plain range for `D1`.
In the current results, the number of free date variables (`n_free_date_vars`) tracks
runtime more closely than any other feature, for every encoding. The grammar rarely pins
anything by itself, because that takes a `==`, a literal on the right and no `||` around
it, all in the same constraint. Yet a real date question is often about a concrete case
("the bill was sent on 1 March 2016") rather than only a rule relating unknown dates.

---



## PERIODS



### multi_field_periods (switch)

On: a `Period(years, months, days)` may have several nonzero fields, as now. Off: every
Period has exactly one nonzero field, and which field is picked separately for each literal.

```
on:   D3 <= (D1 + Period(7, 12, 14));  D5 > (D2 - Period(2, 1, 97))
off:  D3 <= (D1 + Period(0, 0, 30));   D5 > (D2 - Period(1, 0, 0));  D0 == (D4 + Period(0, 6, 0))
```

Because of folding, off also rules out sums that mix fields: `Period(1, 0, 0) + Period(0, 0, 5)`
folds to `(1y, 0m, 5d)`, which is a multi-field Period.

Real-world periods are almost always a single unit ("30 days", "3 months", "1 year"), while
the grammar draws each field independently and almost never produces one. This matters to
the solver, not just to the look of the benchmark: a days-only offset and a months- or
years-only offset exercise different parts of each encoding (`simple` unrolls days one at a
time; months and years need end-of-month clamping), so which encoding wins can change.



### mixed_sign_periods (switch)

On: the fields of a Period may have opposite signs, such as `Period(1, -2, 0)`, which is ten
months forward. Off: every field is 0 or positive, and the `+` or `-` in front of the Period
gives the direction. The flag only matters when `multi_field_periods` is on, since a Period
with a single nonzero field has only one sign.

```
on:   D3 <= (D1 + Period(1, -2, 0));  D5 > (D2 - Period(-3, 0, 14))
off:  D3 <= (D1 + Period(1, 2, 0));   D5 > (D2 - Period(3, 0, 14))
```

With it on, each field gets its own sign, `+` or `-` with equal odds, so a Period with all
three fields nonzero has mixed signs three times in four. A written period almost never mixes directions, which
is why the switch keeps them out of the other instances.

Mixed signs are what a Period difference such as `Period(1, 0, 0) - Period(0, 2, 0)` folds
to, and the only way a single step can move forward in one unit and back in another:
`Date(2023, 1, 31) + Period(0, 1, -1)` clamps to Feb 28 and then steps back to Feb 27.



### max_period_years (count)

The largest absolute value the years field of a Period may have, after folding. A value of
0 leaves years out of the instance altogether. The list is `{0, 1, 10, 100, 1000}`. The current
grammar draws each literal's years from 0 to 10, but sums and products fold well past that
(up to 94 years in the current dataset), so what the solver sees has no cap: the list `{∞}`.

```
max_period_years = 10:   D1 <= (D0 + Period(7, 0, 0))
                         not D1 <= (D0 + (Period(7, 0, 0) * 2))       folds to 14 years
max_period_years = 0:    D1 <= (D0 + Period(0, 6, 30))
```



### max_period_months (count)

The same for the months field. The list is `{0, 11, 24, 12000}`; the current grammar draws each
literal's months from 0 to 20 and folds up to 199 months, the list `{∞}`.

```
max_period_months = 11:   D1 <= (D0 + Period(0, 6, 0))
max_period_months = 0:    D1 <= (D0 + Period(1, 0, 30))
```



### max_period_days (count)

The same for the days field. The list is `{0, 30, 365, 1000, 365000}`; the current grammar draws each
literal's days from 0 to 99 and folds up to 891 days, the list `{∞}`.

```
max_period_days = 30:   D1 <= (D0 + Period(0, 0, 30))
max_period_days = 0:    D1 <= (D0 + Period(0, 1, 0))
```

Notes on the three caps together:

- **At least one cap must be nonzero**, or every Period would be `Period(0, 0, 0)`.
- **Sizes are drawn on a log scale.** A field first draws its number of digits uniformly,
and then a value among the numbers up to the cap with that many digits. With a cap of 365,
a field is 1–9, 10–99 or 100–365 a third of the time each; with `multi_field_periods` on,
0 is one more choice, so each of the four comes up a quarter of the time. Drawn uniformly,
a field would almost always be as long as its cap: 97% of the numbers up to 365,000 have
five or six digits, so nearly every days offset in such an instance would be 27 years or
more.
- **With `multi_field_periods` off**, the one nonzero field is picked among the fields whose
cap is above 0. A cap of 0 is therefore how an instance uses a single unit throughout,
such as days only, rather than a different unit in each literal.
- **The cap on days matters most to the solver.** In the current results, `max_abs_days`
from `analysis/features/features.md` is the feature that best predicts how much faster `epoch_days`
is than `simple` (Spearman ρ = 0.57). `simple` unrolls a Period one day at a time, while
`epoch_days` adds a days-only Period as one integer. The grammar comment keeps day literals under 100
for this reason, but products and sums undo it: the median instance in the current dataset
still reaches a 380-day offset. A per-instance cap puts that cost in some instances only.
- **The caps on years and months matter mostly at 0.** In the current results neither size
is tied to which encoding wins the way days are. Their value is at 0: months and years are
where end-of-month clamping happens (Jan 31 + 1 month is Feb 28), so leaving both out gives
instances with pure day arithmetic and no clamping at all. The solver notices: `epoch_days`
has a fast path for a days-only Period, and only a Period with months or years makes it
decode the date to year, month and day and clamp (`datesat/symbolic_int/epoch_days_int.py`).



### max_chain_len (count)

The most `± Period` steps applied one after another to the same date variable, the same
measure as `max_date_chain_len` in `analysis/features/features.md`. With a value of `k`, every date
expression rooted at a variable draws its number of steps uniformly from 0 … `k`, so 0 means
a variable never gets an offset at all. The list is `{0, 1, 2, 3, 4, 5}`; the current grammar sets no cap, which is the
list `{∞}`.

```
max_chain_len = 0:   D1 >= D0;  D2 < (Date(2016, 3, 1) + Period(0, 0, 30))
max_chain_len = 1:   D1 <= (D0 + Period(0, 0, 30));  D2 > (D1 + (Period(0, 1, 0) * 2))
max_chain_len = 3:   D1 <= (((D0 + Period(0, 1, 0)) + Period(0, 1, 0)) - Period(0, 0, 3))
```

Only steps on a variable count. A Period added to a literal date folds to a constant, so it
is allowed at every value, 0 included. Because of folding, a Period sum or product such as
`D1 + (Period(0, 1, 0) * 2)` is a single step.

A chain cannot be folded the way a Period sum can, because calendar arithmetic depends on
the order of the steps: `(Date(2023, 1, 31) + Period(0, 1, 0)) + Period(0, 1, 0)` is
2023-03-28, while `Date(2023, 1, 31) + Period(0, 2, 0)` is 2023-03-31. Each link is therefore
its own calendar step for the solver to encode, with its own end-of-month clamp, and the
work grows with every link. The current grammar continues a chain about half the time, so
85% of the instances in the current dataset have a chain of 2 or more and the longest has 9,
while a real date question seldom applies more than one offset in a row ("30 days after the
filing date"). A cap of 2 or 3 still keeps order-dependent cases like the one above.

The grammar draws the length itself rather than leaving it to fandango, which decides
between ending a chain and adding a step with a coin flip, so that every extra step is half
as likely as the one before. At a cap of 5, fandango would give a variable 4 or 5 steps
about 1 time in 20; drawn uniformly, it is 1 time in 3.

At 0 the instance has no date arithmetic left for the solver: every remaining Period is
added to a literal and folds away, so all the work is in the comparisons and case splits.
That is the mirror image of `disjunctions` off, where the work is all in the arithmetic.

---



## BOOLEAN SKELETON



### disjunctions (switch)

On: a constraint may be an `||` of several comparisons, and `!=` is one of the comparison
operators, as now. Off: every constraint is a single comparison, and the operators are only
`== < <= > >=`.

```
on:   D8 == D0 || D5 == Date(5686, 5, 30) || D0 <= D2;  D0 != D5
off:  D8 == D0;  D5 <= Date(5686, 5, 30);  D0 < D2
```

`!=` sits under this flag because on dates it is a disjunction in disguise: `a != b` means
`a < b || a > b`, so the solver has to split on it just like on a written `||`. Implication
(`->`) is also a split, but it only comes with bool variables, so it will get its own flag
together with them.

Most real instances are a plain AND of comparisons. With `||` and `!=` always available, the
grammar essentially never produces one: with 5 to 10 constraints per instance, some
constraint almost always draws one or the other. Every grammar instance therefore makes the
solver search over case splits, and none tests the common case where the work is in the date
arithmetic rather than in the search.

---



## DATE LITERALS



### literal_spread (count)

How far apart, in days, the date literals of an instance may lie. Before each instance, the
sampler places a window of that many days uniformly within 0001-01-01 … 9999-12-31, and
every date literal of the instance, the anchors included, is drawn from it. The list is
`{30, 365, 3650, 36500, 3652059}`: about a month, a year, ten years and a hundred years, and
then the whole range, where a literal can be any date.

```
30:        D1 >= Date(2016, 3, 1);  D2 < Date(2016, 3, 24);  D0 == Date(2016, 2, 27)
3652059:   D1 >= Date(2016, 3, 1);  D2 < Date(7340, 11, 5);  D0 == Date(412, 6, 17)
```

Over the whole range, two literals are about 3,300 years apart on average, so the literals
of an instance hardly ever speak about the same stretch of time. A variable caught between
two of them has either centuries of room or none at all (98% of such variables in a
300-instance sample), and an offset of a month or a year between two variables changes
nothing about which literals they can reach. A narrow window keeps the literals of an
instance close together, as in a real question about one contract or one tax year, so that
offsets, bounds and month ends decide whether the instance holds.



### near_month_end_prob (count)

The probability that a date literal is a **month-end draw**: one that falls on day 28 or
later of its month, which takes in the last day of every month. The probability is drawn
uniformly from the whole percents 0%, 1%, …, 100%, and then every date literal in the
instance makes its own draw with it. A literal that is not a month-end draw is drawn
uniformly from the instance's window (see `literal_spread`), and about 1 in 9 of those lands
that late in its month anyway.
The anchors are date literals too, so they follow the flag.

```
50%:  D1 >= Date(2016, 1, 31);  D2 < (D1 + Period(0, 1, 0));  D3 == Date(4751, 9, 3)
0%:   D1 >= Date(2016, 1, 12);  D2 < (D1 + Period(0, 1, 0));  D3 == Date(4751, 9, 3)
```

The range covers every probability, so the share of literals on day 28 or later runs
from about 1 in 9 at 0%, through about 5 in 9 at 50%, to all of them at 100%.

The end of a month is where calendar arithmetic stops being uniform. A month or year step
from day 29, 30 or 31 can run past the end of a shorter month and has to be clamped
(Jan 31 + 1 month is Feb 28), and a day step from the last day rolls over into the next
month, or the next year. With uniform literals, few instances hold a variable close enough
to a month end for these cases to come up, however many month offsets they have.

A month-end draw is uniform over the days in the window that are the 28th or later, so a
31-day month holds four of them, a 30-day month three, and February one, or two in a leap
year. Over the whole range, Feb 29 therefore comes up in about 1 month-end draw in 170,
against 1 in 1,500 uniform draws. A window shorter than 28 days can hold no such day, and
then the draw is uniform instead; none of the listed spreads is that short.

The flag pays off most together with month or year offsets: with `max_period_years` and
`max_period_months` both 0 there is nothing to clamp, and only the day rollovers remain.

---
