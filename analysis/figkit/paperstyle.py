"""Portable figure style for LaTeX manuscripts (figkit).

Every figure is drawn at its final printed size and included in LaTeX at
exactly that width, so font sizes set here are the printed sizes.

Usage (once per figure script, before creating any figure):

    import sys; sys.path.insert(0, "<repo>/analysis/figkit")
    import paperstyle as ps
    ps.setup(textwidth_pt=468.33, font="STIXGeneral", fig_dir="<dir>")
    fig, ax = plt.subplots(figsize=(ps.HALF, 2.15))   # subfigure{0.49\\linewidth}

textwidth_pt: \\the\\linewidth of the target document (print it once with
\\showthe\\linewidth or \\typeout{\\the\\linewidth}).
font: the body font family as matplotlib knows it (check with
matplotlib.font_manager.fontManager.ttflist); the text font of the PDF body
can be read with pymupdf: page.get_fonts().
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import AnchoredText

# ---- printed font sizes (pt) ------------------------------------------
# Elsevier artwork sizing: 7 pt text, sub/superscripts >= 6 pt in print.
FS_LABEL = 8.0      # axis labels
FS_TICK = 7.0       # tick labels
FS_ANNOT = 8.0      # legends and in-figure annotations (often carry math)
FS_PANEL = 9.0      # panel letters
SCRIPT_RATIO = 0.75 # LaTeX's 8 pt -> 6 pt script ratio; mathtext default is 0.7

# ---- geometry, filled by setup() ---------------------------------------
TEXTWIDTH = None    # inches
FULL = None         # \includegraphics[width=\linewidth]
HALF = None         # subfigure{0.49\linewidth}
FIG_DIR = None


def width(frac):
    """Figure width in inches for \\includegraphics[width=frac\\linewidth]."""
    assert TEXTWIDTH is not None, "call paperstyle.setup() first"
    return frac * TEXTWIDTH


def style(font):
    return {
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.bbox": "standard",     # never 'tight': it changes the printed size
        "savefig.pad_inches": 0.0,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,             # embed TrueType, text stays text
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [font],
        "mathtext.fontset": "stix" if "STIX" in font else "dejavuserif",
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
        "legend.title_fontsize": FS_ANNOT,
        "legend.frameon": False,
        "legend.loc": "upper right",    # never rely on 'best'
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.25,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 2.0,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }


def setup(textwidth_pt, font, fig_dir):
    """Install the manuscript style for this process and fix the page geometry."""
    global TEXTWIDTH, FULL, HALF, FIG_DIR
    from matplotlib import font_manager
    assert font in {f.name for f in font_manager.fontManager.ttflist}, font
    assert os.path.isdir(fig_dir), fig_dir
    TEXTWIDTH = textwidth_pt / 72.27
    FULL = TEXTWIDTH
    HALF = 0.49 * TEXTWIDTH
    FIG_DIR = fig_dir
    plt.rcParams.update(style(font))
    # Private matplotlib constant (present in 3.8-3.10); scripts shrink by it.
    assert hasattr(_mathtext, "SHRINK_FACTOR")
    _mathtext.SHRINK_FACTOR = SCRIPT_RATIO


# ---- placement helpers: anchor each label by what it belongs to --------
def vline_label(ax, x, text, y=1.01, **kw):
    """Label a vertical reference line just above the axes frame (x data, y axes)."""
    kw.setdefault("fontsize", FS_ANNOT)
    return ax.text(x, y, text, transform=ax.get_xaxis_transform(),
                   ha=kw.pop("ha", "center"), va=kw.pop("va", "bottom"),
                   clip_on=False, **kw)


def hline_label(ax, y, text, x=0.99, dy=1.5, ha="right", va="bottom", **kw):
    """Label a horizontal reference line at the axes edge (y data, x axes), offset in pt."""
    kw.setdefault("fontsize", FS_ANNOT)
    sign = 1 if va == "bottom" else -1
    trans = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=ax.figure,
                                    y=sign * dy, units="points")
    return ax.text(x, y, text, transform=trans, ha=ha, va=va, **kw)


def end_label(ax, x, y, text, dx=3, dy=0, **kw):
    """Direct label anchored at a data point, offset in points (size-invariant)."""
    kw.setdefault("fontsize", FS_ANNOT)
    return ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                       textcoords="offset points",
                       ha=kw.pop("ha", "left"), va=kw.pop("va", "center"),
                       annotation_clip=False, **kw)


def corner_text(ax, text, loc="upper left", frameon=False, **kw):
    """Data-independent text anchored to an axes corner (statistics, notes)."""
    at = AnchoredText(text, loc=loc, frameon=frameon,
                      prop=dict(size=kw.pop("size", FS_ANNOT)),
                      pad=kw.pop("pad", 0.2), borderpad=kw.pop("borderpad", 0.3),
                      **kw)
    if frameon:
        at.patch.set_linewidth(0.5)
        at.patch.set_edgecolor("0.6")
    ax.add_artist(at)
    return at


def legend_above(ax, ncols, handles=None, labels=None, **kw):
    """Single-row legend outside the axes, centred above the frame."""
    args = [] if handles is None else [handles, labels]
    return ax.legend(*args, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     ncols=ncols, borderaxespad=0.2, **kw)


def plain_log_ticks(axis, ticks):
    """Integer/decimal labels on a log axis: 10^n exponents print below 6 pt."""
    axis.set_ticks(ticks)
    axis.set_ticklabels([f"{t:g}" for t in ticks])
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, name):
    """Write <name>.pdf and a .png preview into FIG_DIR; return the pdf path."""
    assert FIG_DIR is not None, "call paperstyle.setup() first"
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    return base + ".pdf"
