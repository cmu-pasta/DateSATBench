r"""
The look the paper's matplotlib figures share: the paper's fonts, sizes and colours, chosen
so that every figure and table reads the same printed in greyscale.

    from analysis.stats import paper_style
    paper_style.apply()                  # before making the figure
    label = paper_style.NAMES["epoch_days"]

Text is set by LaTeX in the fonts acmart uses (Linux Libertine, with newtxmath), so a
figure's labels match the paper's body text, encoding names come out exactly as the
paper's macros print them (\naive, \epoch, ...), and the PDF embeds Type 1 fonts, which
publishers accept. Without a LaTeX installation, apply() falls back to matplotlib's STIX
fonts and NAMES to plain text.

Sizes are for a figure drawn at its printed size (acmsmall's text width is 5.49 in), next
to the paper's 10 pt body text.
"""

import shutil

import matplotlib.pyplot as plt

TEXT_WIDTH = 5.49      # inches, acmsmall's \textwidth
LABEL_PT = 8           # axis labels, row labels, panel titles, legends
TICK_PT = 7            # tick labels and the values written on marks

INK, INK_2, INK_3 = "#1f1f1e", "#52514e", "#85837b"   # text: primary, secondary, muted
RULE = "#e2e0d9"       # grid lines

# The paper's macros for the encoding names (macros.tex), for LaTeX's preamble.
MACROS = (r"\newcommand{\naive}{\textsc{Simple}}"
          r"\newcommand{\epoch}{\textsc{Epoch}-based}"
          r"\newcommand{\hybridymd}{\textsc{Hybrid}\textsubscript{\textit{init\_ymd}}}"
          r"\newcommand{\hybridepoch}{\textsc{Hybrid}\textsubscript{\textit{init\_epoch}}}"
          r"\newcommand{\ab}{\textsc{Alpha-Beta}}")

# Every figure and table must read the same printed in greyscale as in colour. So nothing is
# told apart by hue alone: the colours of a set are spread over lightness (CIE L*, which is
# what a greyscale print keeps), at least 11 apart, and every set has a second cue that
# needs no colour at all: a dash pattern for lines, a marker and outline for the corpora, a
# row label for bars. A colour added here must keep its set's lightness steps;
# check_greyscale() below fails if it does not.
MIN_L_STEP = 11

# Every encoding in the paper's order (its solver section), with its macro, its plain name
# and its colour. The lightness runs Simple 24, Hybrid init_ymd 36, Epoch-based 50,
# Hybrid init_epoch 64, Alpha-Beta 75; the two hybrids share the greens. The colours also
# pass the dataviz validator's colour-blindness checks with every pair compared, as crossing
# lines need (worst pair Delta E 10.5 under simulation, 22.8 for normal vision).
ENCODINGS = [  # (encoding, macro, plain name, colour)
    ("simple", r"\naive", "Simple", "#35248f"),
    ("epoch_days", r"\epoch", "Epoch-based", "#2a78d6"),
    ("hybrid_init_ymd", r"\hybridymd", "Hybrid init_ymd", "#006400"),
    ("hybrid_init_epoch", r"\hybridepoch", "Hybrid init_epoch", "#1baf7a"),
    ("alpha_beta", r"\ab", "Alpha-Beta", "#f0ac0c"),
]
COLOURS = {e: colour for e, _, _, colour in ENCODINGS}
ROUTER = "#111111"      # the router: black, darker than every encoding's colour
# Each encoding's dash pattern (on, off, ... in pt; None is solid), the cue that tells its
# line apart without colour: dashed, dotted, solid, dash-dot and dash-dot-dot. The router's
# line is solid too, and heavier (ROUTER_WIDTH times an encoding's).
DASHES = {"simple": (6, 2), "epoch_days": (1, 1.3), "hybrid_init_ymd": None,
          "hybrid_init_epoch": (4, 1.3, 1, 1.3), "alpha_beta": (4, 1.3, 1, 1.3, 1, 1.3)}
ROUTER_WIDTH = 1.6

# The corpora in the paper's order (its dataset section), lightness 34, 60 and 75. Each
# corpus also has its own marker and its own outline around its region of the t-SNE panel.
CORPORA = [  # (corpus, paper name, colour, marker)
    ("llm", "LLM-Synthesized", "#1c4fa0", "o"),
    ("grammar", "Grammar-Sampled", "#eb6834", "D"),
    ("legal", "Legally Grounded", "#5ccba0", "^"),
]
OUTLINES = {"llm": "-", "grammar": (0, (4, 2)), "legal": (0, (1, 1.5))}   # matplotlib linestyles

# How an instance ended, in the tables' solved bars: lightness 58, 76 and 31, so timeouts are
# the darkest segment.
OUTCOMES = {"sat": "#2f9e55", "unsat": "#efb118", "timeout": "#8f1d1d"}

# The scale of every heatmap: cividis, which runs from dark blue (-1) through grey (0) to
# light yellow (+1) and gets steadily lighter all the way, so a cell's grey says its sign
# and size. A red-to-green scale would print the two ends as the same dark grey.
HEATMAP = "cividis"
NAMES = {e: plain for e, _, plain, _ in ENCODINGS}   # replaced by the macros by apply()
TIMES = r"$\times$"     # the sign after a speedup, as the paper sets it


def apply():
    """Set matplotlib up for a paper figure. Returns True when LaTeX sets the text."""
    global TIMES
    check_greyscale()
    usetex = shutil.which("latex") is not None
    plt.rcParams.update({
        "font.size": LABEL_PT, "axes.titlesize": LABEL_PT, "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": TICK_PT,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42,   # TrueType for the STIX fallback, not Type 3, which publishers reject
    })
    if usetex:
        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif",
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[tt=false]{libertine}"
                                   r"\usepackage[libertine]{newtxmath}" + MACROS,
        })
        NAMES.update({e: macro for e, macro, _, _ in ENCODINGS})
        TIMES = r"$\times$"
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif",
                             "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})
        TIMES = "×"
    return usetex


def linestyle(encoding, linewidth):
    """The matplotlib linestyle of an encoding's line drawn `linewidth` pt wide: its dash
    pattern in pt, whatever the width (matplotlib scales a pattern by the width)."""
    dashes = DASHES.get(encoding)
    return (0, [d / linewidth for d in dashes]) if dashes else "-"


def lightness(colour):
    """CIE L* of a colour, 0 (black) to 100 (white): the grey it prints as."""
    from matplotlib.colors import to_rgb
    lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in to_rgb(colour))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def check_greyscale():
    """Raise if two colours of one set would print as nearly the same grey, or two encodings
    share a dash pattern. Returns the sets' lightnesses, for printing."""
    sets = {"encodings": {**COLOURS, "router": ROUTER},
            "corpora": {c: colour for c, _, colour, _ in CORPORA},
            "outcomes": OUTCOMES}
    out = {}
    for name, colours in sets.items():
        ls = sorted((lightness(c), k) for k, c in colours.items())
        for (a, ka), (b, kb) in zip(ls, ls[1:]):
            if b - a < MIN_L_STEP:
                raise ValueError(f"{name}: {ka} (L* {a:.0f}) and {kb} (L* {b:.0f}) are less than "
                                 f"{MIN_L_STEP} apart, and would print as the same grey")
        out[name] = {k: round(l) for l, k in ls}
    if len(set(DASHES.values())) < len(DASHES) or set(DASHES) != set(COLOURS):
        raise ValueError("every encoding needs a dash pattern of its own")
    return out


def times(x, fmt=".2f"):
    """A speedup as the paper writes it, e.g. 3.02x."""
    return f"{x:{fmt}}{TIMES}"


if __name__ == "__main__":     # python -m analysis.stats.paper_style: the greys the colours print as
    for name, ls in check_greyscale().items():
        print(f"{name}: " + ", ".join(f"{k} L*={l}" for k, l in ls.items()))
