"""
Replay, without solving, how each of DateSat's five int encodings translates a constraint,
and count what the translation emits.

Section 4 of the DateSAT paper explains the encodings' speed by two measures of the formula
they emit: its **arithmetic complexity**, the integer divisions and modulos (the epoch
conversions and the leap-year test are full of them), and its **logical depth**, the nested
if-then-else terms (the Naive encoding, DateSat's `simple`, unrolls day arithmetic one day
at a time). `emulate` walks a parsed constraint the way DateSat's builder does and counts,
for every encoding:

    divmod        integer div and mod terms
    ite           if-then-else terms
    conversions   date values converted between the year/month/day form and the day count
    lex_cmps      date comparisons emitted component by component, which is a disjunction
                  for an ordering or a `!=`

A term that DateSat builds twice from the same inputs is counted once, because Z3 shares
identical terms, and a term that no assertion uses is not counted, because it never reaches
the solver. The hybrid encodings keep, for each date value, flags saying which of its two
forms is up to date, and the flags change as the constraints are built; the replay keeps the
same flags and visits the constraints in DateSat's order.

The counts follow DateSat's datesat/symbolic_int/*_int.py and datesat/constraint_parser.py,
and have to be updated when those change. They were checked on 2026-09-29 by building every
DateSatBench formula with DateSat (without solving) and counting its distinct div, mod and
if-then-else nodes: the replay matched on every instance and encoding but one. `simple`'s
largest formulas were too big to build and were skipped. The one miss is a Date(...) whose
argument is in parentheses, `Date((y - 1), 1, 1)`: DateSat's parser does not replace it by
an auxiliary variable and compares its fields instead, and the parsed tree here no longer
shows the parentheses.
"""

from itertools import count

from analysis.features.datesat_parser import (
    BinOp, DateAdd, DateCtor, Field, UnOp, Var, canonical, is_comparison, is_ground_date,
    walk,
)

ENCODINGS = ("simple", "epoch_days", "hybrid_ymd", "hybrid_epoch", "alpha_beta")

# (div/mod, if-then-else) terms of each building block, from DateSat's symbolic_int code.
DIM = (3, 3)        # days_in_month(y, m): y%4, y%100, y%400; If(m==2, If(leap, 29, 28), If(.., 30, 31))
CLAMP = (0, 2)      # eom_clamp on top of its days_in_month: If(d<1, 1, If(d>dim, dim, d))
NORM = (2, 0)       # normalize_month: (m-1) div 12, (m-1) mod 12
ENCODE = (5, 2)     # days_since_epoch_from_ymd: y/400, (153mp+2)/5, yoe/4, yoe/100, yoe/400
STEP = (3, 8)       # one day of simple's add_days_componentwise: a days_in_month and 5 Ifs
AB_YM = (1, 0)      # alpha_beta's ym_from_months_since_epoch: (alpha + c - 1) div 12
SELECT = (0, 2)     # alpha_beta's If(stays_in_month, ...) for alpha and for beta
# ymd_from_days_since_epoch, piece by piece: only the pieces a used field needs reach the
# solver, so reading .month alone costs 5 div/mod and 3 Ifs, and all three fields 8 and 4.
DECODE_DIVMOD = {"q400": ("y",), "r400": ("y", "m", "d"), "q100": ("y", "m", "d"),
                 "q4": ("y",), "r4": ("y", "m", "d"), "q1": ("y", "m", "d"),
                 "mp": ("y", "m", "d"), "dd": ("d",)}
DECODE_ITE = {"c100": ("y", "m", "d"), "c1": ("y", "m", "d"), "mwrap": ("y", "m"),
              "yadj": ("y",)}
YMD = ("y", "m", "d")

# `a < b` is Not(a >= b) and `a != b` is Not(a == b): each pair shares one term.
CMP_KIND = {">=": "ge", "<": "ge", "<=": "le", ">": "le", "==": "eq", "!=": "eq"}
MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}


class Value:
    """A date value during the replay, named by `key`; a literal date has `literal` set.
    `ymd` and `epoch` name the terms of its two forms (the hybrid encodings track whether
    each form exists and is up to date), and `run` is simple's current run of day steps."""

    def __init__(self, key, literal=False):
        self.key = self.ymd = self.epoch = key
        self.literal = literal
        self.ymd_exists = self.ymd_ok = self.epoch_ok = False
        self.run = 0


class Encoding:
    """Counts what one encoding emits. Subclasses implement declare, add, field and compare
    the way the matching DateSat encoding builds them."""

    def __init__(self):
        self.divmod = self.ite = self.conversions = self.lex_cmps = 0
        self.seen = set()
        self.fresh = count()

    def emit(self, key, cost):
        """Count a term once per distinct input, as Z3 shares identical terms."""
        if key not in self.seen:
            self.seen.add(key)
            self.divmod += cost[0]
            self.ite += cost[1]

    def once(self, key):
        """True the first time `key` is seen."""
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def encode(self, ym, value):
        """days_since_epoch_from_ymd of a (y, m, d) whose year and month are named `ym`:
        its divisions depend on the year and month only."""
        self.emit(("enc", ym), ENCODE)
        self.conversions += self.once(("conv-enc", value))

    def decode(self, epoch, fields=YMD):
        """ymd_from_days_since_epoch of the day count `epoch`, for the fields used."""
        for piece, needed_by in DECODE_DIVMOD.items():
            if any(f in needed_by for f in fields):
                self.emit(("dec", epoch, piece), (1, 0))
        for piece, needed_by in DECODE_ITE.items():
            if any(f in needed_by for f in fields):
                self.emit(("dec", epoch, piece), (0, 1))
        self.conversions += self.once(("conv-dec", epoch))

    def lex(self, op, a, b):
        """A comparison emitted component by component, counted once per distinct term."""
        self.lex_cmps += self.once(("lex", CMP_KIND[op], a, b))

    def result(self):
        return {"divmod": self.divmod, "ite": self.ite, "conversions": self.conversions,
                "lex_cmps": self.lex_cmps}


class Simple(Encoding):
    """(y, m, d) per date; day arithmetic unrolled one day at a time; comparisons
    lexicographic."""

    def __init__(self):
        super().__init__()
        self.steps = {}          # (date a run of day steps starts from, sign) -> longest run

    def declare(self, name):
        self.emit(("dim", name), DIM)                      # 1 <= d <= days_in_month(y, m)
        return Value(name)

    def add(self, v, p):
        start, run = v.ymd, v.run
        if p.months:
            # years * 12 + months is built as a term, so Period(1,0,0) and Period(0,12,0) differ
            self.emit(("norm", v.key, p.ny, p.nm), NORM)
            start, run = f"{v.key}{p.ny:+}y{p.nm:+}m", 0
            self.emit(("dim", start), DIM)
            self.emit(("clamp", start), CLAMP)
        if p.nd:
            if run and (run > 0) != (p.nd > 0):           # a new run from the current date
                start, run = f"{start}{run:+}d", 0
            run += p.nd
            # A run shares its first steps with every longer run from the same date, and a
            # forward step reuses the days_in_month of the date it leaves.
            k = (start, run > 0)
            self.steps[k] = max(self.steps.get(k, 0), abs(run))
            if run < 0:                                    # a backward run ends on a new month
                self.emit(("dim", f"{start}{run:+}d"), DIM)
        r = Value(f"{start}{run:+}d" if run else start)
        r.ymd, r.run = start, run
        return r

    def field(self, v, fname):
        pass

    def compare(self, op, a, b):
        self.lex(op, a.key, b.key)

    def result(self):
        n = sum(self.steps.values())
        self.divmod += n * STEP[0]
        self.ite += n * STEP[1]
        self.steps = {}
        return super().result()


class EpochDays(Encoding):
    """One day count per date. Year/month arithmetic decodes it to (y, m, d) and encodes the
    result back; reading a field decodes it."""

    def declare(self, name):
        return Value(name)

    def add(self, v, p):
        if not p.months:
            return Value(f"{v.key}{p.nd:+}d")
        self.decode(v.key)
        self.emit(("norm", v.key, p.ny, p.nm), NORM)
        ym = f"{v.key}{p.ny:+}y{p.nm:+}m"
        self.emit(("dim", ym), DIM)
        self.emit(("clamp", ym), CLAMP)
        self.encode(ym, ym)
        return Value(f"{ym}{p.nd:+}d" if p.nd else ym)

    def field(self, v, fname):
        self.decode(v.key, (fname[0],))

    def compare(self, op, a, b):
        pass


class AlphaBeta(Encoding):
    """(months since the epoch, day within the month) per date. A day step keeps the month
    when it can, and otherwise goes through the day count and back."""

    def declare(self, name):
        self.emit(("ym", name), AB_YM)                     # 0 <= beta < days_in_month(alpha)
        self.emit(("dim", name), DIM)
        return Value(name)

    def add(self, v, p):
        ym = v.key
        if p.months:                                       # alpha + (12 years + months)
            ym = f"{ym}{p.months:+}m"
            self.emit(("clamp", ym), CLAMP)
        self.emit(("ym", ym), AB_YM)
        self.emit(("dim", ym), DIM)
        if not p.nd:
            return Value(ym)
        r = f"{ym}{p.nd:+}d"
        self.encode(ym, r)
        self.decode(r)
        self.emit(("sel", r), SELECT)
        self.emit(("ym", r), AB_YM)                        # the result's own bounds
        self.emit(("dim", r), DIM)
        return Value(r)

    def field(self, v, fname):
        if fname != "day":
            self.emit(("ym", v.key), AB_YM)

    def compare(self, op, a, b):
        self.lex(op, a.key, b.key)


class Hybrid(Encoding):
    """Both forms per date, kept up to date lazily: year/month arithmetic on (y, m, d), day
    arithmetic on the day count, and a conversion whenever an operation needs a form that is
    out of date (DateSat's hybrid_*_int.py). hybrid_ymd starts a variable in (y, m, d) form,
    linked to its day count; hybrid_epoch starts it as a day count only."""

    ymd_first = True

    def declare(self, name):
        v = Value(name)
        if self.ymd_first:
            v.ymd_exists = v.ymd_ok = True
            self.encode(name, name)                        # epoch_var == encode(y, m, d)
            self.emit(("dim", name), DIM)                  # 1 <= d <= days_in_month(y, m)
        else:
            v.epoch_ok = True
        return v

    def to_epoch(self, v):
        if not v.epoch_ok:
            self.encode(v.ymd, v.ymd)
            v.epoch_ok = True

    def to_ymd(self, v):
        if v.ymd_ok and v.ymd_exists:
            return
        self.to_epoch(v)
        self.decode(v.epoch)
        if not v.ymd_exists:                               # fresh (y, m, d) variables, linked
            v.ymd = f"#ymd{next(self.fresh)}"
            self.encode(v.ymd, v.ymd)
            v.ymd_exists = True
        v.ymd_ok = True

    def add(self, v, p):
        r = Value(f"{v.key}{p.ny:+}y{p.nm:+}m{p.nd:+}d")
        if not p.months:
            self.to_epoch(v)
            r.epoch_ok, r.epoch = True, f"{v.epoch}{p.nd:+}d"
            return r
        self.to_ymd(v)
        self.emit(("norm", v.ymd, p.ny, p.nm), NORM)
        ym = f"{v.ymd}{p.ny:+}y{p.nm:+}m"
        self.emit(("dim", ym), DIM)
        self.emit(("clamp", ym), CLAMP)
        self.encode(ym, ym)
        if p.nd:
            r.epoch_ok, r.epoch = True, f"enc({ym}){p.nd:+}d"
        else:                                              # a fresh epoch_var, linked to (y, m, d)
            r.ymd_exists = r.ymd_ok = True
            r.ymd, r.epoch = ym, f"#epoch{next(self.fresh)}"
        return r

    def field(self, v, fname):
        self.to_ymd(v)

    def compare(self, op, a, b):
        if b.literal:
            if a.ymd_ok and a.ymd_exists:
                self.lex(op, a.ymd, b.key)
            else:
                self.to_epoch(a)
        elif a.epoch_ok and b.epoch_ok:
            pass
        elif a.ymd_ok and a.ymd_exists and b.ymd_ok and b.ymd_exists:
            self.lex(op, a.ymd, b.ymd)
        elif op in ("==", "!=") and a.epoch_ok:
            self.to_epoch(b)
            self.to_ymd(a)
        elif op in ("==", "!=") and b.epoch_ok:
            self.to_epoch(a)
            self.to_ymd(b)
        else:
            self.to_epoch(a)
            self.to_epoch(b)


class HybridYmd(Hybrid):
    ymd_first = True


class HybridEpoch(Hybrid):
    ymd_first = False


CLASSES = {"simple": Simple, "epoch_days": EpochDays, "hybrid_ymd": HybridYmd,
           "hybrid_epoch": HybridEpoch, "alpha_beta": AlphaBeta}


def is_symbolic_ctor(n):
    return isinstance(n, DateCtor) and not is_ground_date(n)


class Replay:
    """Visits a parsed instance in the order DateSat's builder does, for one encoding: the
    declared date variables in name order; the auxiliary variables that stand for Date(...)
    calls with a variable argument, with their year/month/day equalities; then the
    constraints in order, each operand of a comparison before the comparison."""

    def __init__(self, enc, trees, sorts):
        self.enc = enc
        self.vars = {}
        # DateSat replaces every Date(...) with a variable argument, in order of appearance,
        # by an auxiliary date variable _aux_date_<k> before it builds anything.
        self.aux = {}
        for t in trees:
            for n in walk(t):
                if is_symbolic_ctor(n):
                    self.aux[id(n)] = (f"_aux_date_{len(self.aux)}", n)
        aux = sorted(self.aux.values(), key=lambda a: a[0])
        for name in sorted(sorts):
            if sorts[name] == "date":
                self.vars[name] = enc.declare(name)
        for _, n in aux:                                   # bounds on the arguments
            for c in (n.y, n.m, n.d):
                self.int_expr(c)
        for name, _ in aux:
            self.vars[name] = enc.declare(name)
        for name, n in aux:                                # aux.year == y and so on
            for fname, c in zip(("year", "month", "day"), (n.y, n.m, n.d)):
                enc.field(self.vars[name], fname)
                self.int_expr(c)
        for t in trees:
            self.bool_expr(t)

    def date_expr(self, n):
        if is_ground_date(n):
            return Value(canonical(n), literal=True)
        if isinstance(n, Var):
            return self.vars[n.name]
        if isinstance(n, DateCtor):
            return self.vars[self.aux[id(n)][0]]
        if isinstance(n, DateAdd):
            return self.enc.add(self.date_expr(n.base), n.period)
        raise ValueError(f"not a date expression: {canonical(n)}")

    def int_expr(self, n):
        if isinstance(n, Field):
            v = self.date_expr(n.base)
            if not v.literal:
                self.enc.field(v, n.fname)
        elif isinstance(n, BinOp):
            self.int_expr(n.lhs)
            self.int_expr(n.rhs)
        elif isinstance(n, UnOp):
            self.int_expr(n.operand)

    def bool_expr(self, n):
        if is_comparison(n) and n.lhs.sort == "date":
            a, b = self.date_expr(n.lhs), self.date_expr(n.rhs)
            if a.literal and b.literal:
                return
            op = n.op
            if a.literal:                  # Date.__lt__ returns NotImplemented: b.__gt__(a)
                a, b, op = b, a, MIRROR[op]
            self.enc.compare(op, a, b)
        elif is_comparison(n) and n.lhs.sort == "int":
            self.int_expr(n.lhs)
            self.int_expr(n.rhs)
        elif isinstance(n, BinOp):
            self.bool_expr(n.lhs)
            self.bool_expr(n.rhs)
        elif isinstance(n, UnOp):
            self.bool_expr(n.operand)


def emulate(trees, sorts):
    """{encoding: {"divmod", "ite", "conversions", "lex_cmps"}} for a parsed instance."""
    out = {}
    for name, cls in CLASSES.items():
        enc = cls()
        Replay(enc, trees, sorts)
        out[name] = enc.result()
    return out
