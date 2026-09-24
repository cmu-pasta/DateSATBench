##################################################
# Library imports
##################################################
import re
import random
import datetime


##################################################
# Generator grammar
##################################################

<start> ::= <constraint_list>

# NOTE: Manually unrolled on purpose to generate a even distribution of constraints!
<constraint_list> ::= <twenty_constraints> | <fifteen_constraints> | <ten_constraints> | <nine_constraints> | <eight_constraints> | <seven_constraints> | <six_constraints> | <five_constraints>

<twenty_constraints> ::= <ten_constraints> " ; " <ten_constraints>
<fifteen_constraints> ::= <ten_constraints> " ; " <five_constraints>
<ten_constraints> ::= <nine_constraints> " ; " <constraint>
<nine_constraints> ::= <eight_constraints> " ; " <constraint>
<eight_constraints> ::= <seven_constraints> " ; " <constraint>
<seven_constraints> ::= <six_constraints> " ; " <constraint>
<six_constraints> ::= <five_constraints> " ; " <constraint>
<five_constraints> ::= <four_constraints> " ; " <constraint>
<four_constraints> ::= <three_constraints> " ; " <constraint>
<three_constraints> ::= <two_constraints> " ; " <constraint>
<two_constraints> ::= <constraint> " ; " <constraint>


<constraint> ::= <unit_constraint> | <unit_constraint> " || " <constraint>

<unit_constraint> ::= <date_var> <cmp_op> <expr> 

<expr> ::= <date_var_expr> | <date_expr>

<date_var_expr> ::= <date_var> | "(" <date_var_expr> <add_sub_op> <period_expr> ")"

# Cannot have recursive date_expr since we do not have add_sub operations on date.
<date_expr> ::= <date_ctor> | "(" <date_ctor> <add_sub_op> <period_expr> ")"

<period_expr> ::= <period_ctor> | "(" <period_ctor> <add_sub_op> <period_expr> ")" | "(" <period_expr> " * " <digit> ")"

<date_ctor> ::= "Date(" <date_year> ", " <date_month> ", " <date_day> ")"
<period_ctor> ::= "Period(" <period_year> ", " <period_month> ", " <period_day> ")"

<cmp_op> ::= " == " | " != " | " < " | " <= " | " > " | " >= "
<add_sub_op> ::= " + " | " - "

<date_var> ::= "D"<digit>

# Years 1..9999: the full range representable by the solver, matching the positive_years bounded variant (0001-01-01 .. 9999-12-31).
# Numbers are drawn uniformly by the generator; the expansion forbids leading zeros (0007 is not a valid literal).
<date_year> ::= <nonzero_digit><digit>{0,3} := str(random.randint(1, 9999))
<date_month> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12"
<date_day> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13" | "14" | "15" | "16" | "17" | "18" | "19" | "20" | "21" | "22" | "23" | "24" | "25" | "26" | "27" | "28" | "29" | "30" | "31"

# Note: Set to small numbers on purpose
<period_year> ::= "0" | <nonzero_digit><digit>? := str(random.randint(0, 99))
<period_month> ::= "0" | <nonzero_digit><digit>? := str(random.randint(0, 99))
<period_day> ::= "0" | <nonzero_digit><digit>{0,2} := str(random.randint(0, 999))

<nonzero_digit> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

##################################################
# Post-processing
##################################################

where is_valid_date_ctor(str(<date_ctor>)) 

def is_valid_date_ctor(date_ctor: str) -> bool:
    """Return True iff `date_ctor` represents a valid calendar date in the
    positive-years range 0001-01-01 .. 9999-12-31 inclusive. This is the range
    representable by Python's datetime.date and by the DateSAT solver, and it
    matches the positive_years bounded variant of the benchmark.

    Examples of valid:
      Date(1, 1, 1)
      Date(2000, 2, 29)    # leap year
      Date(9999, 12, 31)

    Examples of invalid:
      Date(0, 5, 1)        # year 0 does not exist
      Date(2023, 2, 29)    # invalid day
    """

    # Match Date(Y, M, D) with a 1- to 4-digit year
    m = re.fullmatch(r"Date\((\d{1,4}),\s*(\d{1,2}),\s*(\d{1,2})\)", date_ctor.strip())
    if not m:
        return False

    year, month, day = map(int, m.groups())

    # datetime.date accepts exactly years 1..9999 and rejects invalid
    # month/day combinations (including Feb 29 in non-leap years).
    try:
        datetime.date(year, month, day)
    except ValueError:
        return False

    return True
