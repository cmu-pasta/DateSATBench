##################################################
# Library imports
##################################################
import random
import datetime


##################################################
# Structure
##################################################

# An entry is 5-10 constraints separated by " ; ": one alternative per count, and
# fandango picks among them uniformly.
<start> ::= <constraint_list>
<constraint_list> ::= <constraint> <next_constraint>{4} | <constraint> <next_constraint>{5} | <constraint> <next_constraint>{6} | <constraint> <next_constraint>{7} | <constraint> <next_constraint>{8} | <constraint> <next_constraint>{9}
<next_constraint> ::= " ; " <constraint>

# A constraint is a disjunction of date comparisons, bool-guarded implications and
# int comparisons.
<constraint> ::= <unit_constraint> | <unit_constraint> " || " <constraint>
<unit_constraint> ::= <date_var> <cmp_op> <expr> | "(" <bool_var_expr> ") -> (" <expr> <cmp_op> <expr> ")" | <int_var> <cmp_op> <int_var_expr>


##################################################
# Expressions
##################################################

<expr> ::= <date_var_expr> | <date_expr>
<date_var_expr> ::= <date_var> | "(" <date_var_expr> <add_sub_op> <period_expr> ")"

# Cannot have recursive date_expr since we do not have add_sub operations on date.
<date_expr> ::= <date_ctor> | "(" <date_ctor> <add_sub_op> <period_expr> ")"

# Only a Period literal can be multiplied: allowing `<period_expr> * <multiplier>`
# made the multipliers compound (x9 per level) into Periods of billions of days.
<period_expr> ::= <period_ctor> | "(" <period_ctor> <add_sub_op> <period_expr> ")" | "(" <period_ctor> " * " <multiplier> ")"

<bool_var_expr> ::= <bool_var> <bool_cmp_op> <bool_val>

# Date components (<date_year> etc.) double as int literals here.
<int_var_expr> ::= <int_var> | <date_year> | <date_month> | <date_day> | "(" <int_var> <arith_op> <int_var_expr> ")"

<cmp_op> ::= " == " | " != " | " < " | " <= " | " > " | " >= "
<add_sub_op> ::= " + " | " - "
<bool_cmp_op> ::= " == " | " != "
<arith_op> ::= " + " | " - " | " * " | " / " | " % "


##################################################
# Literals and variables
##################################################

# A rule ending in `:= <python expr>` takes its value from that expression instead of
# from fandango's own choice among the alternatives, which is not uniform (it picked
# multipliers 0 and 1 about twice as often as 9). The value must still parse against
# the rule, so each rule spells out exactly what its generator can return.

<date_var> ::= "D"<digit> := "D" + str(random.randint(0, 9))
<bool_var> ::= "B"<digit> := "B" + str(random.randint(0, 9))
<int_var> ::= <int_name> | <date_var> "." <date_field>
<int_name> ::= "I"<digit> := "I" + str(random.randint(0, 9))
<date_field> ::= "year" | "month" | "day"
<bool_val> ::= "True" | "False"

# x0 and x1 are left out: they only produce a zero Period or a no-op.
<multiplier> ::= "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" := str(random.randint(2, 9))

# Drawn uniformly over all valid dates 0001-01-01 .. 9999-12-31: the range the solver
# represents, matching the positive_years bounded variant. The sub-rules only describe
# the syntax (no leading zeros); they do not rule out dates such as Feb 30.
<date_ctor> ::= "Date(" <date_year> ", " <date_month> ", " <date_day> ")" := random_date_ctor()
# The <date_year> generator applies where it is used as an int literal.
<date_year> ::= <nonzero_digit><digit>{0,3} := str(random.randint(1, 9999))
<date_month> ::= <nonzero_digit> | "1" ("0" | "1" | "2")
<date_day> ::= <nonzero_digit> | ("1" | "2") <digit> | "3" ("0" | "1")

# Kept small on purpose: the simple approach unrolls a Period one day at a time, so
# large day counts make building the constraints very slow.
<period_ctor> ::= "Period(" <period_year> ", " <period_month> ", " <period_day> ")"
<period_year> ::= "0" | <nonzero_digit> | "10" := str(random.randint(0, 10))
<period_month> ::= "0" | <nonzero_digit> | "1"<digit> | "20" := str(random.randint(0, 20))
<period_day> ::= "0" | <nonzero_digit><digit>? := str(random.randint(0, 99))

# <digit> is fandango's built-in "0" .. "9".
<nonzero_digit> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"


##################################################
# Generator functions
##################################################

def random_date_ctor() -> str:
    """Return a Date(Y, M, D) drawn uniformly from 0001-01-01 .. 9999-12-31."""
    d = datetime.date.fromordinal(random.randint(1, datetime.date.max.toordinal()))
    return "Date(%d, %d, %d)" % (d.year, d.month, d.day)
