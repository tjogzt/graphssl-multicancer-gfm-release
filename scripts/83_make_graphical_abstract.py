#!/usr/bin/env python3
"""
@file    83_make_graphical_abstract.py
@brief   Build the Cell Reports Methods submission Graphical Abstract as a single
         16:9 vector PDF plus a 300-dpi PNG preview.

         The frozen September-2026 draft (figures/graphical_abstract_v1.pdf)
         embedded every glyph as Type-3 DejaVuSans / LastResortHE fallback, which
         the Cell Press production office rejects, and carried nine measured
         layout defects. This script re-typesets the SAME three-band layout, the
         SAME wording, and the SAME palette using Arial with pdf.fonttype = 42
         (TrueType outlines, zero Type-3), repairing only the measured
         collisions plus the task-box height.

         Measured defects repaired (all coordinates in PDF points, y-down):
           D1  Level-1/2/3 captions superimposed on the grey note column
               (22.03 / 71.83 / 48.51 pt overlap)      -> stacked two-line rows
           D2  middle column "Seeds ..." superimposed on "n = 5 same-code runs"
               (19.60 pt overlap)                      -> stacked two-line rows
           D3  "Denoising reconstruction" exceeded the task-box right edge by
               11.15 pt (x1 = 448.38 > 437.23)         -> stacked two-line rows
           D4  row-3 baseline (382.34) fell below the task-box bottom (381.57)
                                                       -> box bottom raised to 394.0
           D5  guideline / scope header shared the first body baseline
               (489.12 vs 487.65)                      -> header on its own line
           D6  the three level discs used two different greens (2 x #1A5A3A and
               1 x #1F8A4C)                            -> unified #1F8A4C
           D7  "PROBLEM" title overlapped the ESM-2 node box (x 187.52-250.72)
                                                       -> band titles lifted to
                                                          band_top + 15.5 pt
           D8  the node-flow diamond occluded the PROBLEM/METHOD band frames
                                                       -> band frames stroked on
                                                          top of the node boxes
           D9  "FINDINGS" title sat inside guideline card 1 (and on its pill)
                                                       -> titles lifted; cards
                                                          shifted down 14.0 pt

         Also replaced the two glyphs Arial does not carry (U+27C2 "PERPENDICULAR"
         and the letter "o" used as a bullet) with U+2260 and U+25E6.

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-10-26
@ingroup result_assembly_and_figures

Input:  No data files. Every literal is either (a) a layout constant
        reverse-engineered from the frozen v1 PDF, or (b) a manuscript number
        reached through the GA_NUM_* provenance constants below.
Output: <fig-dir>/graphical_abstract_v1.pdf   (vector, Arial, fonttype 42)
        <fig-dir>/graphical_abstract_v1.png   (4800 x 2700 px @ 300 dpi)
Run:    python scripts/83_make_graphical_abstract.py
        python scripts/83_make_graphical_abstract.py --fig-dir figures
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Cell Press figure compliance (CRM Methods track 2026):
#   - sans-serif family ordered Arial first, DejaVu Sans last
#   - pdf/ps fonttype = 42 (TrueType embedded); Type 3 is FORBIDDEN
#   - Arial carries every glyph used here, so no fallback face is ever reached
# NOTE: figure_style.apply_style() does NOT set the font family; the block
#       below must be repeated here for the same reason 85_make_figures.py
#       repeats it.
# ---------------------------------------------------------------------------
import matplotlib as _mpl_rc
_mpl_rc.rcParams["font.family"]     = "sans-serif"
_mpl_rc.rcParams["font.sans-serif"] = [
    "Arial",          # macOS/Office native, metric-compatible with Helvetica
    "Helvetica",      # PostScript native
    "Nimbus Sans",    # TeX Live/TinyTeX Helvetica clone
    "DejaVu Sans",    # LAST fallback (NOT Cell's preference)
]
_mpl_rc.rcParams["pdf.fonttype"]  = 42
_mpl_rc.rcParams["ps.fonttype"]   = 42
_mpl_rc.rcParams["mathtext.fontset"] = "custom"
_mpl_rc.rcParams["mathtext.rm"] = "Arial"
_mpl_rc.rcParams["mathtext.it"] = "Arial:italic"

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import figure_style as fs

# ===========================================================================
# Section 1: Non-ASCII glyph escapes (kept as escapes so this file stays ASCII)
# ===========================================================================
MIDDLE_DOT = "\u00b7"   # U+00B7  Arial index 257
EM_DASH    = "\u2014"   # U+2014  Arial index 178
NOT_EQUAL  = "\u2260"   # U+2260  Arial index 143  (replaces U+27C2, absent in Arial)
WHITE_BUL  = "\u25e6"   # U+25E6  Arial index 405  (replaces the letter "o" used as a bullet)

# ===========================================================================
# Section 2: Manuscript-number provenance (single source of truth for figures)
# ===========================================================================
GA_NUM_HGT_PARAMS = "3.89M"                      # manuscript/__main.tex L383,L411; results/freezer/p0_numbers_freeze_20261007.csv
GA_NUM_SEEDS      = "42 / 7 / 123 / 21 / 99"     # 5 equal-budget seeds (42 7 123 21 99)
GA_NUM_EPOCHS     = "500"                        # epochs per arm, equal-budget protocol
GA_NUM_EQV_EDGES  = "4,998"                      # results/freezer/master_numbers_20261009.csv L238 (equal-volume subgraph)
GA_NUM_CKPT_ROWS  = "96"                         # scripts/132_audit_checkpoints.py L76 (checkpoint provenance registry)
GA_NUM_CMD_STEPS  = "10"                         # .github/PULL_REQUEST_TEMPLATE.md L13 (10-command recipe)
GA_NUM_R_VERBS    = "5"                          # R/RunMultiCancerSSL.R L6,16,24,107 (5 exported verbs)

# ===========================================================================
# Section 3: Colour palette (recovered from the frozen v1 PDF draw records;
#            NOT the Okabe-Ito series palette used by the data figures)
# ===========================================================================
C_INK        = "#1A1A1A"   # two-line headline and guideline body text
C_BAND_TITLE = "#234F80"   # PROBLEM / METHOD / FINDINGS titles, subtitles, band frame
C_ESM        = "#5A9BD5"
C_TCGA       = "#C89B4A"
C_STRING     = "#7BA55A"
C_HGT        = "#A05E9A"
C_COL1       = "#2A6DB0"   # left  task column  (3 TASKS)
C_COL2       = "#B57B23"   # middle task column (5 x EQUAL-BUDGET)
C_COL3       = "#7A4EA6"   # right  task column (LEVEL-1/2/3)
C_DESC       = "#2B2B2B"   # grey task descriptions
C_MID_MAIN   = "#7A4A12"   # middle column main labels and ">" bullets
C_MID_GREY   = "#333333"   # middle column grey notes
C_LEVEL      = "#1F8A4C"   # unified level disc fill/edge and Level labels (D6)
C_LEVEL_GREY = "#2A2A2A"   # level grey notes
C_SCOPE_TXT  = "#6D1F1F"
C_SCOPE_EDGE = "#B22222"
C_REPRO_TXT  = "#115132"
C_REPRO_EDGE = "#1F8A4C"
C_PROBLEM_BODY = "#333333"
C_NODE_EDGE  = "#555555"
C_CONN       = "#444444"
C_WHITE      = "#FFFFFF"

BAND_FILL = {
    "problem":  "#EEF4FB",
    "method":   "#F6F1FA",
    "findings": "#EEF8EF",
}
SCOPE_FILL = "#FFF5F2"
REPRO_FILL = "#F4FBF4"

# ===========================================================================
# Section 4: Page and layout geometry (PDF points, y-down, reverse-engineered)
# ===========================================================================
PAGE_W, PAGE_H = 1152.0, 648.0        # 16 x 9 inch at 72 pt/in

BAND_L, BAND_R = 152.93, 1027.87
BAND_PROBLEM  = (BAND_L,  78.31, BAND_R, 224.12)
BAND_METHOD   = (BAND_L, 222.46, BAND_R, 407.07)
BAND_FINDINGS = (BAND_L, 408.18, BAND_R, 570.62)

BAND_TITLE_OFFSET = 15.5             # D7/D9: band titles lifted to band_top + 15.5
PROBLEM_TITLE_X   = 174.69
METHOD_TITLE_X    = 293.60           # D8: clear of the node diamond (right edge 284.62)
FINDINGS_TITLE_X  = 174.69
SUBTITLE_RIGHT_X  = 1014.50

HEADLINE_CX   = 590.40               # centred on the band midline
HEADLINE_BASE = (107.80, 129.05)

# --- node-flow diamond (unchanged from v1) --------------------------------
NODE_ESM2   = (187.52,  96.06, 284.62, 141.52)
NODE_TCGA   = (126.14, 154.27, 223.24, 199.73)
NODE_STRING = (248.90, 154.27, 346.00, 199.73)
NODE_HGT    = (187.52, 206.94, 284.62, 252.40)
NODE_EDGE_LW = 1.4
NODE_LABEL_SIZE = 9.0

DIAMOND_ARROWS = [
    ((236.07, 141.52), (180.90, 155.80)),
    ((236.07, 141.52), (290.70, 155.80)),
    ((174.69, 199.73), (231.50, 208.10)),
    ((297.45, 199.73), (240.70, 208.10)),
]

# --- three-column task grid ----------------------------------------------
TASK_BOX_TOP    = 250.73
TASK_BOX_BOTTOM = 394.00             # D4: raised from 381.57
TASK_COL1 = (202.31, TASK_BOX_TOP, 437.23, TASK_BOX_BOTTOM)
TASK_COL2 = (484.10, TASK_BOX_TOP, 719.02, TASK_BOX_BOTTOM)
TASK_COL3 = (760.31, TASK_BOX_TOP, 995.23, TASK_BOX_BOTTOM)
TASK_HEADER_BASE = 271.17
TASK_HEADER_SIZE = 12.5
TASK_ROW_BASES   = (287.0, 329.0, 371.0)
TASK_DESC_OFFSET = 12.0
TASK_PAD_R       = 8.0               # right inner padding used for the fit checks

# --- findings band: guideline cards (shifted +14.0) -----------------------
CARD_SHIFT = 14.0
CARD1 = (165.48, 430.50,  423.84, 495.92)
CARD2 = (461.22, 430.50,  719.58, 495.92)
CARD3 = (756.96, 430.50, 1015.32, 495.92)
CARD_EDGE_LW = 2.1
CARD_BODY_DX = 94.31
CARD_BODY_BASE = (460.82, 472.80)
CARD_BODY_SIZE = 10.5

PILL1 = (176.05, 450.10, 246.83, 476.32)
PILL2 = (471.79, 450.10, 542.57, 476.32)
PILL3 = (767.53, 450.10, 853.51, 476.32)
PILL_LABEL_BASE = (9.87, 21.28)      # offsets from pill top edge
PILL_LABEL_SIZE = 9.5

# --- findings band: scope / reproducibility boxes -------------------------
SCOPE_BOX = (236.63, 488.71, 589.84, 551.36)
REPRO_BOX = (616.07, 488.71, 969.28, 551.36)
CORNER_BOX_EDGE_LW = 1.8
CORNER_HEADER_DX = 13.39
CORNER_BODY_DX   = 18.97
CORNER_HEADER_BASE = 501.00          # D5: header on its own line
CORNER_HEADER_SIZE = 11.0
CORNER_BODY_BASES = (513.50, 526.00, 538.50)
CORNER_BODY_SIZE = 9.6

# --- inter-band flow arrows ----------------------------------------------
BAND_ARROW_X = 590.40
BAND_ARROW_SPANS = ((220.80, 224.57), (403.75, 410.29))
BAND_ARROW_LW = 1.9
BAND_ARROW_HALO_LW = 3.4
ARROW_MUT_SCALE = 8
CONN_LW = 1.3

Z_BAND_FILL = 0.5
Z_BOX       = 2.0
Z_BAND_EDGE = 3.0                    # D8: frames stroked above the node boxes
Z_ARROW     = 4.0
Z_TEXT      = 5.0

FIG_W_IN, FIG_H_IN = 16.0, 9.0


# ===========================================================================
# Section 5: Coordinate helpers
# ===========================================================================
def yup(y_down):
    """@brief Convert a y-down PDF coordinate to a matplotlib y-up data value."""
    return PAGE_H - y_down


def box_down(x0, y0d, x1, y1d):
    """@brief Convert a y-down rect into the (x, y, w, h) tuple FancyBboxPatch wants."""
    return (x0, yup(y1d), x1 - x0, y1d - y0d)


def rbox(ax, rect_down, face, edge, lw, z, radius=4.0):
    """@brief Add a rounded rectangle given a y-down rect."""
    x, y, w, h = box_down(*rect_down)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0_down, p1_down, color, lw, z, halo=False, mut=ARROW_MUT_SCALE):
    """@brief Add a straight arrow given y-down endpoints (optional white halo)."""
    p0 = (p0_down[0], yup(p0_down[1]))
    p1 = (p1_down[0], yup(p1_down[1]))
    if halo:
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=mut,
            linewidth=lw + BAND_ARROW_HALO_LW, color=C_WHITE, zorder=z - 0.1,
            shrinkA=0.0, shrinkB=0.0, capstyle="round",
        ))
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mut,
        linewidth=lw, color=color, zorder=z, shrinkA=0.0, shrinkB=0.0,
    ))


def text(ax, x, y_down, s, size, color, ha="left", weight="normal", style="normal", z=Z_TEXT):
    """@brief Place a baseline-anchored text string at a y-down coordinate."""
    return ax.text(
        x, yup(y_down), s, fontsize=size, color=color,
        ha=ha, va="baseline", fontweight=weight, fontstyle=style, zorder=z,
    )


# ===========================================================================
# Section 6: Width-fitting helpers (guarantee zero overflow)
# ===========================================================================
def _width_pt(fig, renderer, s, size, weight, style):
    """@brief Rendered width of a string in points (== data units on this canvas)."""
    probe = fig.text(0.5, 0.5, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = probe.get_window_extent(renderer=renderer).width / (fig.dpi / 72.0)
    probe.remove()
    return w


def pick_size(fig, renderer, strings, max_w_pt, sizes, weight="normal", style="normal"):
    """@brief Largest candidate size at which every string fits max_w_pt."""
    for s in sizes:
        if all(_width_pt(fig, renderer, t, s, weight, style) <= max_w_pt for t in strings):
            return s
    return sizes[-1]


# ===========================================================================
# Section 7: Figure builder
# ===========================================================================
def build(fig_dir):
    """
    @brief   Render the graphical abstract and write PDF + PNG into fig_dir.
    @return  None (side effects: two files on disk, a fit report on stdout).
    """
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, PAGE_W)
    ax.set_ylim(0.0, PAGE_H)
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()

    checks = []   # (label, x0, x1, max_x) -> verified after the canvas is drawn

    def record(label, x0, x1, max_x):
        checks.append((label, x0, x1, max_x))

    # -- 7.1 band fills and frames -----------------------------------------
    for name, rect in (("problem", BAND_PROBLEM),
                       ("method", BAND_METHOD),
                       ("findings", BAND_FINDINGS)):
        rbox(ax, rect, BAND_FILL[name], "none", 0.0, Z_BAND_FILL, radius=8.0)
    for rect in (BAND_PROBLEM, BAND_METHOD, BAND_FINDINGS):
        rbox(ax, rect, "none", C_BAND_TITLE, 2.2, Z_BAND_EDGE, radius=8.0)

    # -- 7.2 two-line headline ---------------------------------------------
    headline = [
        "Which graph SSL objective drives encoder quality",
        f"in the realistic PLM-initialized multi-cancer regime?",
    ]
    for line, base in zip(headline, HEADLINE_BASE):
        text(ax, HEADLINE_CX, base, line, 17.0, C_INK,
             ha="center", weight="bold")

    # -- 7.3 band titles and subtitles -------------------------------------
    band_titles = (
        ("PROBLEM",  BAND_PROBLEM[1]  + BAND_TITLE_OFFSET, PROBLEM_TITLE_X,
         f"Motivation {MIDDLE_DOT} Literature gap"),
        ("METHOD",   BAND_METHOD[1]   + BAND_TITLE_OFFSET, METHOD_TITLE_X,
         f"Equal-budget 5-seed ablation {MIDDLE_DOT} 3 tasks {MIDDLE_DOT} HGT encoder"),
        ("FINDINGS", BAND_FINDINGS[1] + BAND_TITLE_OFFSET, FINDINGS_TITLE_X,
         f"3 Guidelines + Scope Boundaries + {GA_NUM_CMD_STEPS}-command Reproduction"),
    )
    for title, base, x, subtitle in band_titles:
        t = text(ax, x, base, title, 14.0, C_BAND_TITLE, weight="bold")
        record(f"band-title {title}", x, x + _width_pt(fig, renderer, title, 14.0, "bold", "normal"), BAND_R)
        del t
        text(ax, SUBTITLE_RIGHT_X, base, subtitle, 11.0, C_BAND_TITLE,
             ha="right", style="oblique")

    # -- 7.4 PROBLEM band body ---------------------------------------------
    problem_body_x = 406.16
    problem_body = [
        f"PLM features (ESM-2) on graph nodes {EM_DASH} task ranking unknown",
        f"No protocol to diagnose task-loss {NOT_EQUAL} encoder-quality decoupling",
    ]
    for i, line in enumerate(problem_body):
        w = _width_pt(fig, renderer, line, 11.5, "normal", "normal")
        text(ax, problem_body_x, 159.10 + 15.53 * i, line, 11.5, C_PROBLEM_BODY)
        record(f"problem-body {i + 1}", problem_body_x, problem_body_x + w, BAND_R - 14.0)

    # -- 7.5 node-flow diamond ---------------------------------------------
    for rect in (NODE_ESM2, NODE_TCGA, NODE_STRING, NODE_HGT):
        rbox(ax, rect, C_WHITE, C_NODE_EDGE, NODE_EDGE_LW, Z_BOX, radius=3.0)

    node_labels = (
        (f"ESM-2",       236.07, 117.90, C_ESM),
        (f"8M / 650M",   236.07, 126.99, C_ESM),
        (f"TCGA",        174.69, 180.83, C_TCGA),
        (f"STRING",      297.45, 176.11, C_STRING),
        (f"v12",         297.45, 185.62, C_STRING),
        (f"HGT",         236.07, 228.78, C_HGT),
        (f"({GA_NUM_HGT_PARAMS})", 236.07, 237.77, C_HGT),
    )
    for s, cx, base, col in node_labels:
        text(ax, cx, base, s, NODE_LABEL_SIZE, col, ha="center", weight="bold")

    for p0, p1 in DIAMOND_ARROWS:
        arrow(ax, p0, p1, C_CONN, CONN_LW, Z_ARROW)

    # -- 7.6 three-column task grid ----------------------------------------
    task_cols = (
        (TASK_COL1, C_COL1, f"3 TASKS"),
        (TASK_COL2, C_COL2, f"5 x EQUAL-BUDGET"),
        (TASK_COL3, C_COL3, f"LEVEL-1/2/3"),
    )
    for (x0, _y0, x1, _y1), edge, header in task_cols:
        rbox(ax, (x0, TASK_BOX_TOP, x1, TASK_BOX_BOTTOM), C_WHITE, edge, 2.0, Z_BOX, radius=5.0)
        text(ax, (x0 + x1) / 2.0, TASK_HEADER_BASE, header, TASK_HEADER_SIZE, edge,
             ha="center", weight="bold")

    # 7.6a left column: bullet + label, grey description on the next line
    col1_x0, _, col1_x1, _ = TASK_COL1
    col1_x = col1_x0 + 12.0
    col1_rows = (
        (f"{WHITE_BUL} NAMR", "Denoising reconstruction (feature scaling mandatory)"),
        (f"{WHITE_BUL} CSP",  "PPI link prediction (-> collapse diagnostic)"),
        (f"{WHITE_BUL} C3L",  "Cross-cancer InfoNCE (-> redundancy elimination)"),
    )
    col1_max = col1_x1 - TASK_PAD_R - col1_x
    col1_desc_size = pick_size(
        fig, renderer, [d for _l, d in col1_rows], col1_max,
        [9.6, 9.25, 9.0, 8.75, 8.5, 8.25, 8.0],
    )
    for i, (label, desc) in enumerate(col1_rows):
        base = TASK_ROW_BASES[i]
        text(ax, col1_x, base, label, 11.5, C_COL1, weight="bold")
        w = _width_pt(fig, renderer, label, 11.5, "bold", "normal")
        record(f"task1-row{i + 1}-label", col1_x, col1_x + w, col1_x1 - TASK_PAD_R)
        text(ax, col1_x, base + TASK_DESC_OFFSET, desc, col1_desc_size, C_DESC)
        w = _width_pt(fig, renderer, desc, col1_desc_size, "normal", "normal")
        record(f"task1-row{i + 1}-desc", col1_x, col1_x + w, col1_x1 - TASK_PAD_R)

    # 7.6b middle column: ">" bullet + label, grey note on the next line
    col2_x0, _, col2_x1, _ = TASK_COL2
    col2_bullet_x = col2_x0 + 17.02
    col2_x = col2_x0 + 36.55
    col2_rows = (
        (f"Seeds {GA_NUM_SEEDS}", f"n = 5 same-code runs"),
        (f"{GA_NUM_EPOCHS} epochs per arm", "equal optimizer steps"),
        (f"Equal-volume {GA_NUM_EQV_EDGES}-edge subgraph", "FLOP parity vs BRCA-only"),
    )
    col2_max = col2_x1 - TASK_PAD_R - col2_x
    col2_main_size = pick_size(
        fig, renderer, [m for m, _g in col2_rows], col2_max,
        [10.2, 9.75, 9.5, 9.25, 9.0, 8.75],
        weight="bold",
    )
    for i, (main, grey) in enumerate(col2_rows):
        base = TASK_ROW_BASES[i]
        text(ax, col2_bullet_x, base, f">", 11.0, C_MID_MAIN)
        text(ax, col2_x, base, main, col2_main_size, C_MID_MAIN, weight="bold")
        w = _width_pt(fig, renderer, main, col2_main_size, "bold", "normal")
        record(f"task2-row{i + 1}-main", col2_x, col2_x + w, col2_x1 - TASK_PAD_R)
        text(ax, col2_x, base + TASK_DESC_OFFSET, grey, 9.3, C_MID_GREY)
        w = _width_pt(fig, renderer, grey, 9.3, "normal", "normal")
        record(f"task2-row{i + 1}-grey", col2_x, col2_x + w, col2_x1 - TASK_PAD_R)

    # 7.6c right column: numbered disc + level label, grey note on the next line
    col3_x0, _, col3_x1, _ = TASK_COL3
    disc_d = 11.96
    disc_cx = col3_x0 + 19.81
    col3_x = col3_x0 + 56.08
    col3_rows = (
        ("1", "Level-1  PPI AUC",         "SOLE ranking metric"),
        ("2", "Level-2  cluster accuracy", "Self-consistency only"),
        ("3", "Level-3  downstream",      "Scope boundaries only"),
    )
    col3_max = col3_x1 - TASK_PAD_R - col3_x
    for i, (num, label, grey) in enumerate(col3_rows):
        base = TASK_ROW_BASES[i]
        cy = base - 3.6
        ax.add_patch(Circle((disc_cx, yup(cy)), disc_d / 2.0, facecolor=C_LEVEL,
                            edgecolor=C_LEVEL, linewidth=1.0, zorder=Z_BOX))
        text(ax, disc_cx, cy + 3.7, num, 10.5, C_WHITE, ha="center", weight="bold")
        text(ax, col3_x, base, label, 10.5, C_LEVEL, weight="bold")
        w = _width_pt(fig, renderer, label, 10.5, "bold", "normal")
        record(f"task3-row{i + 1}-label", col3_x, col3_x + w, col3_x1 - TASK_PAD_R)
        text(ax, col3_x, base + TASK_DESC_OFFSET, grey, 9.3, C_LEVEL_GREY)
        w = _width_pt(fig, renderer, grey, 9.3, "normal", "normal")
        record(f"task3-row{i + 1}-grey", col3_x, col3_x + w, col3_x1 - TASK_PAD_R)

    # -- 7.7 findings band: guideline cards --------------------------------
    card_meta = (
        (CARD1, C_COL1, PILL1, "GUIDELINE 1", "[STRONG]",
         "NAMR reconstruction wins", "(when features are scaled)"),
        (CARD2, C_COL2, PILL2, "GUIDELINE 2", "[STRONG]",
         "Encoder collapse <> CSP loss", "(skip-decoder diagnostic)"),
        (CARD3, C_COL3, PILL3, "GUIDELINE 3", "[CONDITIONAL]",
         "ESM-2 => C3L redundancy", "(feature-first design rule)"),
    )
    for card, edge, pill, pill_l1, pill_l2, body_l1, body_l2 in card_meta:
        rbox(ax, card, C_WHITE, edge, CARD_EDGE_LW, Z_BOX, radius=5.0)
        rbox(ax, pill, edge, edge, 1.0, Z_BOX + 0.1, radius=3.0)
        pcx = (pill[0] + pill[2]) / 2.0
        for line, off in zip((pill_l1, pill_l2), PILL_LABEL_BASE):
            text(ax, pcx, pill[1] + off, line, PILL_LABEL_SIZE, C_WHITE,
                 ha="center", weight="bold")
        bx = card[0] + CARD_BODY_DX
        for line, base in zip((body_l1, body_l2), CARD_BODY_BASE):
            text(ax, bx, base, line, CARD_BODY_SIZE, C_INK)
            w = _width_pt(fig, renderer, line, CARD_BODY_SIZE, "normal", "normal")
            record(f"card-body {body_l1[:12]}", bx, bx + w, card[2] - TASK_PAD_R)

    # -- 7.8 findings band: scope + reproducibility boxes ------------------
    corner_meta = (
        (SCOPE_BOX, SCOPE_FILL, C_SCOPE_EDGE, C_SCOPE_TXT,
         "HONEST SCOPE BOUNDARIES (3 / 3)",
         ("Gene-level relational tasks GAIN",
          "Patient-level phenotype tasks (set-fingerprint) NOISE",
          "Drug response from target-only embeddings R^2 < 0")),
        (REPRO_BOX, REPRO_FILL, C_REPRO_EDGE, C_REPRO_TXT,
         f"REPRODUCIBILITY . {GA_NUM_CMD_STEPS}-command recipe",
         (f"R/RunMultiCancerSSL.R  .  {GA_NUM_R_VERBS} verbs",
          "Makefile (data/preprocess/train/evaluate/figures)",
          f"{GA_NUM_CKPT_ROWS}-row checkpoint provenance registry JSON")),
    )
    for box, fill, edge, txt_col, header, body in corner_meta:
        rbox(ax, box, fill, edge, CORNER_BOX_EDGE_LW, Z_BOX, radius=5.0)
        hx = box[0] + CORNER_HEADER_DX
        text(ax, hx, CORNER_HEADER_BASE, header, CORNER_HEADER_SIZE, txt_col, weight="bold")
        w = _width_pt(fig, renderer, header, CORNER_HEADER_SIZE, "bold", "normal")
        record(f"corner-header {header[:14]}", hx, hx + w, box[2] - TASK_PAD_R)
        bx = box[0] + CORNER_BODY_DX
        for line, base in zip(body, CORNER_BODY_BASES):
            text(ax, bx, base, line, CORNER_BODY_SIZE, txt_col)
            w = _width_pt(fig, renderer, line, CORNER_BODY_SIZE, "normal", "normal")
            record(f"corner-body {line[:14]}", bx, bx + w, box[2] - TASK_PAD_R)

    # -- 7.9 inter-band flow arrows ----------------------------------------
    for y0, y1 in BAND_ARROW_SPANS:
        arrow(ax, (BAND_ARROW_X, y0), (BAND_ARROW_X, y1), C_CONN,
              BAND_ARROW_LW, Z_ARROW, halo=True)

    # -- 7.10 save ---------------------------------------------------------
    stem = os.path.join(fig_dir, "graphical_abstract_v1")
    fs.save_fig(fig, stem)

    # -- 7.11 fit verification (rendered widths, data units == points) -----
    bad = 0
    for label, x0, x1, max_x in checks:
        slack = max_x - x1
        if slack < 0:
            bad += 1
        print(f"[fit] {label:34s} x0={x0:7.2f} x1={x1:7.2f} limit={max_x:7.2f} slack={slack:+6.2f}")
    print(f"[fit] {len(checks)} text spans checked, {bad} overflow(s)")
    if bad:
        raise SystemExit(f"{bad} text span(s) exceed their container - aborting")
    print(f"[ga] col1 desc size = {col1_desc_size:.2f} pt, "
          f"col2 main size = {col2_main_size:.2f} pt")


def main():
    parser = argparse.ArgumentParser(description="Build the submission graphical abstract.")
    parser.add_argument(
        "--fig-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"),
        help="output directory (default: <repo>/figures)",
    )
    args = parser.parse_args()
    fs.apply_style()
    os.makedirs(args.fig_dir, exist_ok=True)
    build(args.fig_dir)


if __name__ == "__main__":
    main()
