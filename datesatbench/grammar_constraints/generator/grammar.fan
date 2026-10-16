##################################################
# Library imports
##################################################
import random
import datetime


##################################################
# Flags
##################################################

# The swarm-testing flags, which flags.toml defines and flag.md explains. Before every
# batch, generate_constraints.py draws a value for every flag and copies them all into
# this dict; the generator functions below read the ones they need. The dict is empty
# until then, so the grammar can only be sampled through generate_constraints.py.
FLAGS = {}

# What generate_constraints.py draws anew before every instance. "literal_window" is the
# stretch of literal_spread days, as the ordinals (first, last), that every date literal
# of the instance is drawn from.
INSTANCE = {}


##################################################
# Structure
##################################################

# An entry is one or more constraints separated by " ; ". The sampler generates each
# constraint on its own and decides how many there are, so this rule only fixes the
# syntax.
<start> ::= <constraint> | <constraint> " ; " <start>

# A constraint is a disjunction of comparisons, each with a date variable on the left.
<constraint> ::= <unit_constraint> | <unit_constraint> " || " <constraint>
<unit_constraint> ::= <date_var> <cmp_op> <expr>


##################################################
# Expressions
##################################################

# The right-hand side is a date literal, or a date variable followed by a chain of
# offsets. The generator draws the chain's length uniformly from 0 .. max_chain_len;
# left to fandango, each extra offset would be half as likely as the one before.
<expr> ::= <date_var_expr> | <date_ctor>
<date_var_expr> ::= <date_var> | "(" <date_var_expr> <add_sub_op> <period_ctor> ")" := random_date_var_expr()

# Every offset is a single Period literal. Period sums and multiples, and offsets on
# a literal date, all fold to one constant before the solver encodes anything, so a
# single literal already covers each of them.

<cmp_op> ::= " == " | " != " | " < " | " <= " | " > " | " >= " := random_cmp_op()
<add_sub_op> ::= " + " | " - "


##################################################
# Literals and variables
##################################################

# A rule ending in `:= <python expr>` takes its value from that expression instead of
# from fandango's own choice among the alternatives, which is not uniform. The value
# must still parse against the rule, so each rule spells out everything its
# generator can return.

# D0 .. D(k-1), where k is num_date_vars.
<date_var> ::= "D" ("0" | <nonzero_digit> <digit>*) := random_date_var()

# Drawn from the instance's literal window, which lies within 0001-01-01 .. 9999-12-31:
# the range the solver represents, matching the positive_years bounded variant. With
# probability near_month_end_prob, the day is 28 or later. The sub-rules only describe
# the syntax (no leading zeros); they do not rule out dates such as Feb 30.
<date_ctor> ::= "Date(" <date_year> ", " <date_month> ", " <date_day> ")" := random_date_ctor()
<date_year> ::= <nonzero_digit><digit>{0,3}
<date_month> ::= <nonzero_digit> | "1" ("0" | "1" | "2")
<date_day> ::= <nonzero_digit> | ("1" | "2") <digit> | "3" ("0" | "1")

# Period(years, months, days), with each field bounded by its max_period_* flag and
# drawn on a log scale.
<period_ctor> ::= "Period(" <period_field> ", " <period_field> ", " <period_field> ")" := random_period_ctor()
<period_field> ::= "0" | "-"? <nonzero_digit> <digit>*

# <digit> is fandango's built-in "0" .. "9".
<nonzero_digit> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"


##################################################
# Generator functions
##################################################

def random_date_var() -> str:
    """Return one of D0 .. D(k-1), where k is num_date_vars."""
    return "D%d" % random.randrange(FLAGS["num_date_vars"])


def random_date_var_expr() -> str:
    """Return a date variable with a chain of offsets, such as ((D1 + Period(0, 0, 3)) - Period(1, 0, 0)).

    The number of offsets is drawn uniformly from 0 .. max_chain_len, and each one is
    added or subtracted with equal probability.
    """
    expr = random_date_var()
    for _ in range(random.randint(0, FLAGS["max_chain_len"])):
        expr = "(%s%s%s)" % (expr, random.choice([" + ", " - "]), random_period_ctor())
    return expr


def random_cmp_op() -> str:
    """Return a comparison operator; `!=` only when disjunctions is on."""
    ops = [" == ", " < ", " <= ", " > ", " >= "]
    if FLAGS["disjunctions"]:
        ops.append(" != ")
    return random.choice(ops)


def random_date_ctor() -> str:
    """Return a Date(Y, M, D) from the instance's literal window.

    With probability near_month_end_prob, the date is drawn uniformly from the days in
    the window that are the 28th or later of their month. Otherwise it is drawn
    uniformly from the whole window, as it also is when the window holds no such day.
    """
    first, last = INSTANCE["literal_window"]
    near_month_end = random.random() < FLAGS["near_month_end_prob"]
    # Any 28 days in a row include a 28th or later, so only a shorter window can lack one.
    if near_month_end and last - first < 27:
        near_month_end = any(datetime.date.fromordinal(o).day >= 28 for o in range(first, last + 1))
    while True:
        d = datetime.date.fromordinal(random.randint(first, last))
        if d.day >= 28 or not near_month_end:
            return "Date(%d, %d, %d)" % (d.year, d.month, d.day)


def random_period_field(cap: int, allow_zero: bool) -> int:
    """Return a number from 0 .. cap, or from 1 .. cap without allow_zero, on a log scale.

    The number of digits is drawn uniformly first, with 0 as the one number of no
    digits, and then the number uniformly among those up to cap with that many digits.
    With a cap of 365, 0, 1 .. 9, 10 .. 99 and 100 .. 365 each come up a quarter of the
    time.
    """
    if cap == 0:
        return 0
    digits = random.randint(0 if allow_zero else 1, len(str(cap)))
    if digits == 0:
        return 0
    return random.randint(10 ** (digits - 1), min(10 ** digits - 1, cap))


def random_period_ctor() -> str:
    """Return a Period(Y, M, D) that respects the max_period_* caps.

    Every field is drawn with random_period_field. With multi_field_periods on, each
    field is drawn from [0, cap], and when mixed_sign_periods is on it also gets its
    own sign, so that fields can have opposite signs, such as Period(1, -2, 0). With
    it off, one field among those whose cap is above 0 is drawn from [1, cap] and the
    others are 0. The + or - in front of the Period gives the direction. An all-zero
    Period is a no-op, so it is never returned.
    """
    caps = (FLAGS["max_period_years"], FLAGS["max_period_months"], FLAGS["max_period_days"])
    if not any(caps):
        raise ValueError("at least one of the max_period_* caps must be above 0")
    if FLAGS["multi_field_periods"]:
        while True:
            fields = [random_period_field(cap, allow_zero=True) for cap in caps]
            if FLAGS["mixed_sign_periods"]:
                fields = [random.choice([-1, 1]) * field for field in fields]
            if any(fields):
                return "Period(%d, %d, %d)" % tuple(fields)
    fields = [0, 0, 0]
    i = random.choice([i for i, cap in enumerate(caps) if cap > 0])
    fields[i] = random_period_field(caps[i], allow_zero=False)
    return "Period(%d, %d, %d)" % tuple(fields)
