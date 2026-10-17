#!/usr/bin/env python3
"""
@file    figure_style.py
@brief   Shared publication-figure styling module for the GFM manuscript.
         Centralises a single colourblind-safe palette, one matplotlib rcParams
         block (typography, DPI, live-text vector output), and a unified
         save_fig() helper that writes every figure as both 300-dpi PNG and
         vector PDF with editable text. Imported by scripts/85_make_figures.py
         and any other plotting script so all panels share identical style.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalised 2026-09)
@ingroup core_visualization

Input:  No runtime inputs. On import the module exposes palette constants and
        the apply_style() / save_fig() helpers; callers invoke apply_style()
        once before creating figures.
Output: Side-effects on the global matplotlib rcParams (apply_style) and
        PNG + PDF files written to disk (save_fig).
Run:    Typical caller pattern:
            import figure_style as fs
            fs.apply_style()
            fig, ax = plt.subplots(...)
            # ... plotting code ...
            fs.save_fig(fig, "figures/fig1_panel_a")
"""

import os

import matplotlib.pyplot as plt

# =============================================================================
# Section 1: Colour palette definition (centralised, single source of truth)
# =============================================================================
# Okabe-Ito colourblind-safe qualitative palette. Eight colours is the maximum
# set that is universally distinguishable under both protanopia and deuteranopia
# simulations; never append ad-hoc extra colours — re-order the semantic
# assignments below instead.
OKABE_ITO_PALETTE = [
    "#0072B2",  # Index 0 — blue          (primary series)
    "#D55E00",  # Index 1 — vermillion    (comparison series)
    "#009E73",  # Index 2 — bluish green  (secondary series)
    "#CC79A7",  # Index 3 — reddish purple (auxiliary series)
    "#E69F00",  # Index 4 — orange        (additional baseline)
    "#56B4E9",  # Index 5 — sky blue      (additional baseline)
    "#F0E442",  # Index 6 — yellow        (rarely used; low contrast on white)
    "#000000",  # Index 7 — black         (reference / emphasis)
]

# Named indices into the Okabe-Ito palette. Using symbolic names rather than
# raw hex keeps the semantic meaning of each colour explicit and lets us
# remap the palette by editing a single block.
_PAL_IDX_BLUE = 0
_PAL_IDX_VERMILLION = 1
_PAL_IDX_GREEN = 2
_PAL_IDX_PURPLE = 3
_PAL_IDX_ORANGE = 4
_PAL_IDX_SKYBLUE = 5
_PAL_IDX_YELLOW = 6
_PAL_IDX_BLACK = 7

# =============================================================================
# Section 2: Semantic colour assignments (one meaning per colour per figure)
# =============================================================================
# Every colour constant below carries exactly one semantic role across the
# entire manuscript. Callers MUST reference these named constants rather than
# literals so that a global recolouring is a one-line edit.

# --- Main comparison-series colours -----------------------------------------
COL_MAIN = OKABE_ITO_PALETTE[_PAL_IDX_BLUE]        # Primary method / series under study (GFM)
COL_ALT = OKABE_ITO_PALETTE[_PAL_IDX_VERMILLION]   # Alternative / comparison / diagnostic series
COL_OK = OKABE_ITO_PALETTE[_PAL_IDX_GREEN]         # Secondary series (e.g. NAMR-only ablations)
COL_EXTRA = OKABE_ITO_PALETTE[_PAL_IDX_PURPLE]     # Auxiliary series (e.g. C3L-only ablations)

# --- Additional baseline-series colours -------------------------------------
COL_ORANGE = OKABE_ITO_PALETTE[_PAL_IDX_ORANGE]    # Fourth baseline series (orange)
COL_SKY = OKABE_ITO_PALETTE[_PAL_IDX_SKYBLUE]      # Fifth baseline series (sky blue)

# --- Structural / reference colours -----------------------------------------
COL_REF = "#999999"                               # Reference lines and reference conditions
COL_PAIRED = "#BBBBBB"                            # Seed-pairing connector lines between arms
COL_BLACK = OKABE_ITO_PALETTE[_PAL_IDX_BLACK]      # Fallback black for text / strong emphasis

# --- Fig. 1 heterogeneous graph node-type colours ----------------------------
COL_GENE = OKABE_ITO_PALETTE[_PAL_IDX_BLUE]        # Fig. 1 node type: gene      (circle marker)
COL_CANCER = OKABE_ITO_PALETTE[_PAL_IDX_VERMILLION]  # Fig. 1 node type: cancer type (square marker)
COL_PATIENT = OKABE_ITO_PALETTE[_PAL_IDX_GREEN]    # Fig. 1 node type: patient   (triangle marker)

# =============================================================================
# Section 3: Non-colour visual constants
# =============================================================================
CHANCE_LEVEL_AUC = 0.5     # Horizontal reference line value for AUC panels (random classifier)

# Typography parameters — centralised so all panels share the same point sizes.
# These are intentionally small (7–8 pt) because manuscript figures are
# typically typeset at ~80 mm width in LaTeX; designing at the printed size
# (rather than scaling down a larger canvas) guarantees readable labels.
TYPO_FONT_SIZE_BASE = 8        # Default font size (axes body text)
TYPO_TITLE_SIZE = 8            # Axes title size (matches base to avoid visual bloat)
TYPO_LABEL_SIZE = 8            # X/Y axis label size
TYPO_TICK_LABEL_SIZE = 7       # Tick-mark label size (one step smaller)
TYPO_LEGEND_SIZE = 7           # Legend text size
TYPO_LINE_WIDTH = 1.4          # Main data-line stroke width
TYPO_AXIS_LINE_WIDTH = 0.8     # Axes spine stroke width (thinner than data lines)

# Raster output configuration.
RASTER_DPI = 300               # Target pixels-per-inch for all PNG output
PDF_FONT_TYPE = 42             # Keep text as TrueType outlines (editable/searchable in vector PDF)


def apply_style():
    """
    @brief   Apply the shared matplotlib rcParams globally to every figure
             created after this call. Sets typography sizes, line weights,
             300-dpi raster resolution, and live-text (fonttype 42) vector
             output. Must be called exactly once, before plt.subplots().
    @return  None. All configuration is applied via side-effect on matplotlib's
             global rcParams dictionary.
    @note    If a caller needs to override a specific parameter for a single
             panel (e.g. a larger title on a splash figure), do so with a
             per-figure context manager rather than editing this function so
             the bulk of the manuscript stays consistent.
    """
    plt.rcParams.update({
        # --- Raster / figure resolution ---------------------------------
        "figure.dpi": RASTER_DPI,
        "savefig.dpi": RASTER_DPI,

        # --- Typography sizes -------------------------------------------
        "font.size": TYPO_FONT_SIZE_BASE,
        "axes.titlesize": TYPO_TITLE_SIZE,
        "axes.labelsize": TYPO_LABEL_SIZE,
        "xtick.labelsize": TYPO_TICK_LABEL_SIZE,
        "ytick.labelsize": TYPO_TICK_LABEL_SIZE,
        "legend.fontsize": TYPO_LEGEND_SIZE,

        # --- Line weights -----------------------------------------------
        "lines.linewidth": TYPO_LINE_WIDTH,
        "axes.linewidth": TYPO_AXIS_LINE_WIDTH,

        # --- Vector-output live text ------------------------------------
        # fonttype 42 == TrueType embedded outlines; keeps text selectable
        # and searchable in the assembled PDF, and avoids the Type-3 bitmap
        # fallback that journal production teams sometimes reject.
        "pdf.fonttype": PDF_FONT_TYPE,
        "ps.fonttype": PDF_FONT_TYPE,
    })


def save_fig(fig, stem):
    """
    @brief   Save a matplotlib Figure as both 300-dpi PNG (for quick preview)
             and vector PDF (for manuscript submission), then close the figure
             to free memory. The output directory is created on demand.

             This function deliberately does NOT pass bbox_inches="tight" to
             savefig; the caller must run fig.tight_layout() beforehand so
             the saved canvas has EXACTLY figsize × 300 dpi dimensions. This
             guarantees that when the manuscript LaTeX source includes the PDF
             with width=\\columnwidth, every figure is printed at scale 1.0 —
             so the type sizes set in apply_style() appear at their designed
             point size and raster content retains exactly 300 dpi. Using
             bbox_inches="tight" used to shrink the saved width by 1–3 %,
             silently rescaling text and pushing raster resolution below the
             300-dpi threshold (review R3, 2026-09-17).
    @param   fig    Matplotlib Figure instance to save. The caller is expected
                    to have already run tight_layout(), added titles/labels, etc.
    @param   stem   Output path STEM (WITHOUT file extension). The helper writes
                    <stem>.png and <stem>.pdf. If stem contains a parent
                    directory that does not yet exist it will be created with
                    os.makedirs(exist_ok=True).
    @return  None. Side-effects: two files written to disk; fig closed via plt.close().
    @note    After save_fig returns the figure handle is no longer valid; any
             further plotting must create a new figure with plt.subplots().
    """
    # Ensure the parent directory exists (or stem has no directory component).
    # "os.path.dirname(stem) or ''" handles bare filenames where dirname == ''.
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    # Raster preview at the configured publication resolution.
    fig.savefig(f"{stem}.png", dpi=RASTER_DPI)
    # Vector master — PDF fonttype is already set to 42 in apply_style().
    fig.savefig(f"{stem}.pdf")
    # Release the figure's memory; long notebook pipelines otherwise leak GIL-held
    # matplotlib Artists until the process exits.
    plt.close(fig)
    # Minimal progress line so the manuscript-build log records each output.
    print(f"saved {stem}.png/.pdf ({RASTER_DPI} dpi)")
