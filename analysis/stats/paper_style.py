r"""
The look the paper's matplotlib figures share: the paper's fonts, sizes and colours.

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
          r"\newcommand{\hybridboth}{\textsc{Hybrid}\textsubscript{\textit{both}}}"
          r"\newcommand{\ab}{\textsc{Alpha-Beta}}")

# Every encoding in the paper's order (its solver section), with its macro, its plain name
# and its colour. The colours are slots of the dataviz reference palette and pass its
# validator with every pair compared, as crossing lines need (worst pair Delta E 15.6 for
# normal vision, 9.1 under colour-blindness simulation); the two hybrids share the greens.
ENCODINGS = [  # (encoding, macro, plain name, colour)
    ("simple", r"\naive", "Simple", "#4a3aa7"),
    ("epoch_days", r"\epoch", "Epoch-based", "#2a78d6"),
    ("hybrid_init_ymd", r"\hybridymd", "Hybrid init_ymd", "#008300"),
    ("hybrid_init_epoch", r"\hybridepoch", "Hybrid init_epoch", "#1baf7a"),
    ("alpha_beta", r"\ab", "Alpha-Beta", "#eda100"),
]
COLOURS = {e: colour for e, _, _, colour in ENCODINGS}
ROUTER = "#3d3c37"      # the router: dark, apart from every encoding's colour

# The corpora in the paper's order (its dataset section). The colours are the reference
# palette's slots 1-3, which pass every check across all pairs, as a scatter needs. Each
# corpus also gets its own marker, so the t-SNE panel survives greyscale.
CORPORA = [  # (corpus, paper name, colour, marker)
    ("llm", "LLM-Synthesized", "#2a78d6", "o"),
    ("grammar", "Grammar-Sampled", "#eb6834", "D"),
    ("legal", "Legally Grounded", "#1baf7a", "^"),
]
NAMES = {e: plain for e, _, plain, _ in ENCODINGS}   # replaced by the macros by apply()
TIMES = r"$\times$"     # the sign after a speedup, as the paper sets it


def apply():
    """Set matplotlib up for a paper figure. Returns True when LaTeX sets the text."""
    global TIMES
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


def times(x, fmt=".2f"):
    """A speedup as the paper writes it, e.g. 3.02x."""
    return f"{x:{fmt}}{TIMES}"
