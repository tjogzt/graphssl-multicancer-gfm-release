#!/usr/bin/env python3
"""
@file    85_make_figures.py
@brief   Generate every manuscript figure from machine artifacts. One entry point,
         one style module (figure_style.py) and one data module (figure_data.py);
         no experimental value is hard-coded in the plotting code.

Figures:
  fig1  Architecture overview: (a) heterogeneous network, (b) 6-layer HGT with
        the three self-supervised tasks, (c) downstream evaluation outcomes.
        Horizontal three-band pipeline (four-way figure review, R2 consensus).
  fig2  Pretraining convergence of the full model (500 epochs, full.pt).
  fig3  Ablation: cluster accuracy +/- SD and PPI AUC (seed-42 evaluation).
  s1    NAMR-only standardization ablation over 100 epochs (namr_sweep).
  s2    Patient-level few-shot learning curve (log2 K axis).
  s3    Skip-connected versus plain-decoder NAMR diagnostic.

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (merged single figure pipeline, 2026-09)
@ingroup result_assembly_and_figures

Input:  config.RESULT_DIR artifacts + checkpoints (see figure_data.py)
Output: <FIG_DIR>/fig1_architecture, fig2_convergence_v2, fig3_ablation_v2,
        figS_namr_fix, figS_patient_learning, figS_skip_diagnostic (.png + .pdf)
Run:    python scripts/85_make_figures.py                 # all figures
        python scripts/85_make_figures.py --only fig1 fig3
"""
import argparse
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Cell Press figure compliance (CRM Methods-track 2026):  # noqa: E501
#   - Font family: Nimbus Sans (Helvetica-metric TeX default) fallback
#                   Arial, Helvetica, last-resort DejaVu Sans
#   - pdf/ps fonttype = 42 TrueType embedded (Type 3 FORBIDDEN by Cell Office)
#   - All explicit default sizes: titles 10pt / labels&ticks 9pt / legend 9pt
# ---------------------------------------------------------------------------
import matplotlib as _mpl_rc
_mpl_rc.rcParams["font.family"]     = "sans-serif"
_mpl_rc.rcParams["font.sans-serif"] = [
    "Nimbus Sans",    # TeX Live/TinyTeX native Helvetica clone
    "Arial",          # macOS/Office native
    "Helvetica",      # PostScript native
    "DejaVu Sans",    # LAST fallback (NOT Cell's preference)
]
_mpl_rc.rcParams["pdf.fonttype"]     = 42   # TrueType embedded (not Type 3)
_mpl_rc.rcParams["ps.fonttype"]      = 42
_mpl_rc.rcParams["font.size"]         = 10
_mpl_rc.rcParams["axes.titlesize"]    = 10
_mpl_rc.rcParams["axes.labelsize"]    = 9
_mpl_rc.rcParams["xtick.labelsize"]   = 9
_mpl_rc.rcParams["ytick.labelsize"]   = 9
_mpl_rc.rcParams["legend.fontsize"]   = 9
_mpl_rc.rcParams["mathtext.fontset"] = "custom"    # disable default dejavusans math fallback
_mpl_rc.rcParams["mathtext.rm"]  = "Arial"          # roman (upright) math
_mpl_rc.rcParams["mathtext.it"]  = "Arial:italic"   # italic math (e.g., $\pm$, $\Delta$)
_mpl_rc.rcParams["mathtext.bf"]  = "Arial:bold"     # bold math
_mpl_rc.rcParams["mathtext.sf"]  = "Arial"          # sans-serif math
_mpl_rc.rcParams["mathtext.tt"]  = "Courier New"    # monospace math

# Additional safety: force matplotlib.font_manager to drop the DejaVu Sans fallback
# if Arial is not found on system, Nimbus Sans will kick in. If no sans at all, will fallback.
import matplotlib.font_manager
try:
    for __f in list(matplotlib.font_manager.fontManager.ttflist):
        if "DejaVu" in __f.name and __f.name not in _mpl_rc.rcParams["font.sans-serif"]:
            # remove from list
            matplotlib.font_manager.fontManager.ttflist.remove(__f)
except Exception:
    pass


import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config
import figure_data as fd
import figure_style as fs

# ============================================================================
# Shared Layout Constants (presentation only - no experimental values here)
# ============================================================================

# --- Fig. 1 (Architecture Overview) Canvas & Style ---
FIG1_FIGSIZE = (10.0, 5.0)
FIG1_BANDS = {"a": (0.01, 0.335), "b": (0.365, 0.66), "c": (0.69, 0.99)}
FIG1_TINT_ENC = "#DCE9F5"
FIG1_TINT_DRUG = "#F5CCAD"
FIG1_BOX_FILL = "#F4F4F4"
FIG1_BOX_EDGE = "#8E8E8E"
FIG1_FRAME = "#8A8C8A"
FIG1_FRAME_LINEWIDTH = 1.4
FIG1_PANEL_TITLE_FONTSIZE = 10.5
FIG1_PANEL_TOP = 0.93
FIG1_PANEL_BOTTOM = 0.06
FIG1_PANEL_HEIGHT = 0.87
FIG1_INPUT_LABEL_FONTSIZE = 8.2
FIG1_INPUT_BOX_WIDTH = 0.34
FIG1_INPUT_BOX_HEIGHT = 0.17
FIG1_INPUT_BOX_LW = 0.8
FIG1_LEGEND_LABEL_FONTSIZE = 8.2
FIG1_LEGEND_SCATTER_SIZE = 34
FIG1_FOOTNOTE_FONTSIZE = 8.2
FIG1_CAPTION_FONTSIZE = 8.2
FIG1_NETWORK_BOX_FACE = "#FBFBFB"
FIG1_NETWORK_BOX_LW = 0.8
FIG1_NETWORK_EDGE_COLOR = "#B9B9B9"
FIG1_NETWORK_EDGE_LW = 0.4
FIG1_NETWORK_EDGE_ALPHA = 0.6
FIG1_NETWORK_MAX_EDGES_LOCAL = 90
FIG1_GENE_SCATTER_SIZE = 14
FIG1_CANCER_SCATTER_SIZE = 40
FIG1_PATIENT_SCATTER_SIZE = 36
FIG1_TASK_LABEL_FONTSIZE = 8.0
FIG1_TASK_DETAIL_FONTSIZE = 8.2
FIG1_ENCODER_LABEL_FONTSIZE = 8.2
FIG1_ENCODER_LAYER_FONTSIZE = 8.2
FIG1_LAYER_BOX_WIDTH = 0.26
FIG1_LAYER_BOX_HEIGHT = 0.085
FIG1_LAYER_BOX_LW = 0.7
FIG1_TASK_BOX_WIDTH = 0.53
FIG1_TASK_BOX_HEIGHT = 0.15
FIG1_TASK_BOX_LW = 0.8
FIG1_SHARED_ENCODER_BAND_X = 0.335
FIG1_SHARED_ENCODER_BAND_W = config.FIG1_SHARED_ENCODER_BAND_W  # 0.07 — centralised (config §6c)
FIG1_SHARED_ENCODER_BAND_Y = 0.085
FIG1_SHARED_ENCODER_BAND_H = 0.72
FIG1_CARD_TOP = 0.74
FIG1_CARD_HEIGHT = 0.205
FIG1_CARD_GAP = 0.055
FIG1_CARD_WIDTH = 0.94
FIG1_CARD_LW = 0.8
FIG1_CARD_LABEL_FONTSIZE = 8.2
FIG1_CARD_BADGE_FONTSIZE = 8.2
FIG1_FLOW_ARROW_MUT_SCALE = 16
FIG1_FLOW_ARROW_LW = 2.0
FIG1_INNER_ARROW_MUT_SCALE = 14
FIG1_INNER_ARROW_LW = 2.0
FIG1_INTER_PANEL_GAP = 0.006

# --- Canonical Canvas Widths (journal review 2026-09-17, R3) ---
# LaTeX include width is fixed to the canvas width so every figure prints at
# scale 1.0: type prints at its designed size (label 8 pt / tick 7 pt) and
# raster resolution stays at 300 dpi. Previous upscaling in LaTeX inflated
# printed type of fig3/S1/S2/S3 (up to 1.43x) and dropped to 210-279 dpi.
TEXTWIDTH_IN = 6.6929
MAIN_WIDTH_IN = TEXTWIDTH_IN
SUPP_WIDTH_RATIO = 0.8
SUPP_WIDTH_IN = SUPP_WIDTH_RATIO * TEXTWIDTH_IN

# --- Fig. 2 / Fig. 3 (Main Text) ---
FIG2_FIGSIZE = (MAIN_WIDTH_IN, 2.30)
FIG3_FIGSIZE = (MAIN_WIDTH_IN, 2.45)
FIG2_SUPTITLE_FONTSIZE = 10
FIG3_SUPTITLE_FONTSIZE = 10
FIG2_TOTAL_LOSS_FIGSIZE = (MAIN_WIDTH_IN, 2.1)

# --- Supplementary Figures ---
S1_FIGSIZE = (SUPP_WIDTH_IN, 2.80)
S2_FIGSIZE = (SUPP_WIDTH_IN, 2.95)
S3_FIGSIZE = (SUPP_WIDTH_IN, 2.20)
S_MECHANISM_FIGSIZE = (MAIN_WIDTH_IN, 2.6)

# --- Fig. 1 Content (Labels + Outcome Badges) ---
FIG1_TASK_LABELS = [
    ("NAMR", "mask & reconstruct\ngene features"),
    ("CSP", "real PPI pairs\nvs random pairs"),
    ("C3L", "contrast gene\u2013cancer\ntargets"),
]

# --- Fig. 3 Display Order + S3 Sample Epochs ---
FIG3_ORDER = [
    ("Full", "full"), ("NAMR-only", "namr_only"), ("CSP-only", "csp_only"),
    ("CSP+NAMR", "csp_namr"), ("Random-init", "random_init_hgt"),
    ("ESM-only", "esm_only"), ("PCA input", "pca_input"),
]
S3_SAMPLE_EPOCHS = [1, 100, 200, 300, 400, 500]

# --- Checkpoint Artifacts (relative to project root; resolved through config) ---
FULL_CKPT = os.path.join("ablation_real", "full.pt")
NAMR_ONLY_CKPT = os.path.join("ablation_real", "namr_only.pt")
SKIP_DECODER_CKPT = os.path.join("ablation_real", "namr_only_s7.pt")
BASE_REF_CKPT = os.path.join("namr_sweep", "base_ref.pt")
H1_STD_CKPT = os.path.join("namr_sweep", "h1_std.pt")

# --- Node Marker Map: (marker, color) per node type ---
NODE_MARKERS = {
    "gene": ("o", fs.COL_GENE),
    "cancer": ("s", fs.COL_CANCER),
    "patient": ("^", fs.COL_PATIENT),
}

# --- Fig. 1 Panel Titles (per band key) ---
FIG1_PANEL_TITLES = {
    "a": "(a) Multi-cancer network",
    "b": "(b) Pretraining",
    "c": "(c) Downstream evaluation",
}

# --- Fig. 1 Input Panel Items ---
FIG1_A_INPUTS = [
    "TCGA\nmulti-omics",
    "STRING v12\nPPI",
    "ESM-2 features\n256-dim",
]
FIG1_A_INPUT_ROW_Y_START = 0.78
FIG1_A_INPUT_ROW_Y_STEP = 0.22

# --- Fig. 1 Network Bounding Box ---
FIG1_A_NETWORK_BOX_X = 0.44
FIG1_A_NETWORK_BOX_Y = 0.40
FIG1_A_NETWORK_BOX_W = 0.54
FIG1_A_NETWORK_BOX_H = 0.46
FIG1_A_NETWORK_SCALE_X = 0.80
FIG1_A_NETWORK_SCALE_Y = 0.80
FIG1_A_NETWORK_ORIGIN_X_RATIO = 0.10
FIG1_A_NETWORK_ORIGIN_Y_RATIO = 0.10

# --- Fig. 1 Legend Rows ---
FIG1_A_LEGEND_ROWS = [
    ("o", "#0470AA", "Gene 13,881"),
    ("s", "#D35E00", "Cancer type 21"),
    ("^", "#009E73", "Patient 6,273"),
]
FIG1_A_LEGEND_Y_START = 0.20
FIG1_A_LEGEND_Y_STEP = config.FIG1_A_LEGEND_Y_STEP         # 0.07 — centralised (config §6c)
FIG1_A_LEGEND_X_SCATTER = 0.06
FIG1_A_LEGEND_X_TEXT = 0.11

# --- Fig. 1 Panel (b) Geometry ---
FIG1_B_HEADER_Y = 0.82
FIG1_B_HEADER_FONTSIZE = 8.5
FIG1_B_LAYER_Y_START = 0.10
FIG1_B_LAYER_Y_STEP = 0.115
FIG1_B_LAYER_X = 0.05
FIG1_B_NUM_LAYERS = 6
FIG1_B_TASK_Y_START = 0.58
FIG1_B_TASK_Y_STEP = 0.185
FIG1_B_TASK_X = 0.44
FIG1_B_ENCODER_BAND_FACE = "#E7EDF4"
FIG1_B_ENCODER_BAND_EDGE = "#B9CFE8"
FIG1_B_ENCODER_BAND_LW = 0.6
FIG1_B_ENCODER_BAND_LABEL_COLOR = "#1F4E79"
FIG1_B_TASK_ABBREV_FONTSIZE = 8
FIG1_B_ARROW_COLOR = "#5B7C99"
FIG1_B_ARROW_MUT_SCALE = 7
FIG1_B_ARROW_LW = 0.6

# --- Fig. 1 Panel (c) Card Status -> Color Map ---
FIG1_C_STATUS_COLOR_MAP = {
    "negative": "#B3261E",
    "warn": "#111111",
    "ok": "#111111",
}
FIG1_C_STATUS_BADGE_PREFIX_NEG = "\u2717 "
FIG1_C_STATUS_TINT_MAP = {
    "negative": FIG1_TINT_DRUG,
    "warn": FIG1_TINT_DRUG,
    "ok": FIG1_BOX_FILL,
}
FIG1_C_CARD_LABEL_RATIO_Y = 0.70
FIG1_C_CARD_BADGE_RATIO_Y = 0.28
FIG1_C_CARD_PAD_X = 0.06

# --- Fig. 2 Stylistic Constants ---
FIG2_ARMS_COL_RATIO = 0.34
FIG2_ARMS_COL_LABEL_FONTSIZE = 8.0
FIG2_ARMS_SCATTER_SIZE = 26
FIG2_ARMS_SCATTER_EDGE_WIDTH = 0.5
FIG2_ARMS_ANNOT_XYOFFSET = (6, 4)
FIG2_LEGEND_FONTSIZE = 8.0
FIG2_LEGEND_NCOL = 1
FIG2_LEGEND_HANDLELENGTH = 1.6
FIG2_LEGEND_LABELSPACING = 0.30
FIG2_LEGEND_BBOX = (0.5, -0.20)
FIG2_REF_LINE_LW = 1.0
FIG2_SMOOTH_LINE_LW = 1.2
FIG2_WSPACE = 0.34
FIG2_BOTTOM_PAD = 0.42
FIG2_SMOOTH_REF_TEXT_OFFSET_X = 0.015
FIG2_SMOOTH_REF_TEXT_OFFSET_Y = 0.028
FIG2_SMOOTH_REF_TEXT_FONTSIZE = 8.0

# --- Fig. 3 Stylistic Constants ---
FIG3_JITTER_RADIUS = 0.10
FIG3_SCATTER_SIZE = 18
FIG3_SCATTER_EDGE_WIDTH = 0.4
FIG3_ERRORBAR_FMT = "o"
FIG3_ERRORBAR_MS = 5.0
FIG3_ERRORBAR_CAPSIZE = 3.5
FIG3_ERRORBAR_LW = 1.4
FIG3_ARM_LABEL_FONTSIZE = 8.0
FIG3_PAIRED_LINE_LW = 0.7
FIG3_REF_SCATTER_SIZE = 17
FIG3_REF_SCATTER_GAP_X = 0.15
FIG3_REF_SCATTER_STEP_X = 0.42
FIG3_REF_LABEL_FONTSIZE = 8.0
FIG3_REF_LABEL_OFFSET_Y = 0.030
FIG3_CHANCE_LINE_LW = 1.0
FIG3_SMOOTH_LINE_LW = 1.2
FIG3_SMOOTH_TEXT_OFFSET_Y = 0.004
FIG3_SMOOTH_TEXT_FONTSIZE = 8.0
FIG3_XTICK_FONTSIZE = 8.0
FIG3_XTICK_ROT = 20
FIG3_YLIM_LOW = 0.46
FIG3_YLIM_HIGH = 1.02

# --- Fig. S1 / S2 / S3 Stylistic Constants ---
S2_LEGEND_FONTSIZE = 8.0
S2_LEGEND_NCOL = 2
S2_LEGEND_COLSPACING = 0.8
S2_ANNOT_XYOFFSET = (0, 6)
S2_ANNOT_FONTSIZE = 8.0
S2_YLIM_LOW = 0
S2_YLIM_HIGH = 0.92
S2_XLIM_LOW = 0.8
S2_XLIM_HIGH = 40
S2_SCATTER_MS = 5
S3_LOSS_DASH_Y = 1.32
S3_SCATTER_MS_PLAIN = 3
S3_SCATTER_MS_SKIP = 3
S3_XLIM_LOW = 0
S3_XLIM_HIGH = 560
S3_YLIM_LOSS_LOW = 0
S3_YLIM_LOSS_HIGH = 1.45
S3_CHANCE_LINE_LW = 0.8
S3_AUC_YLIM_LOW = 0.5
S3_AUC_YLIM_HIGH = 0.95
S3_BAR_WIDTH = 0.62
S3_BAR_VALUE_OFFSET_Y = 0.008
S3_BAR_VALUE_FONTSIZE = 8.0
S3_TICK_LABEL_FONTSIZE = 8.0
S3_SUPTITLE_FONTSIZE_A = "(a) Reconstruction loss"
S3_SUPTITLE_FONTSIZE_B = "(b) Representation quality"
S3_REPLACE_WS_PATTERN = r"\s*\n\s*"
S3_REPLACE_SEED_PATTERN = "(seed "
S3_REPLACE_SEED_REPLACEMENT = "(s"

# --- Mechanism Figure (S) ---
S_MECH_ARM_LABEL_FONTSIZE = 8.0
S_MECH_TITLE_FONTSIZE = 9
S_MECH_TIGHT_PAD = 1.6
S_MECH_SCATTER_ALPHA = 0.75
S_MECH_SCATTER_LW = 1.0
S_MECH_MARKER = "o"
S_MECH_MARKER_MS = 3.2
S_MECH_LINE_LW = 1.2

# --- Shared Geometry Helper Constants ---
FIT_TEXT_MIN_FONTSIZE = 8.0
FIT_TEXT_STEP = 0.25
ROUNDED_BOX_DEFAULT_PAD = 0.004
ROUNDED_BOX_DEFAULT_RADIUS = 0.02
ROUNDED_BOX_DEFAULT_LW = 0.75
ROUNDED_BOX_DEFAULT_FACE = "#FFFFFF"
ROUNDED_BOX_DEFAULT_EDGE = "#333333"
BAND_ARROW_Y = 0.5
BAND_ARROW_MUT_SCALE = 14
BAND_ARROW_LINEWIDTH = 2.0

# --- Random State Seed for Fig. 3 Jitter ---
FIG3_JITTER_RNG_SEED = 0


# ============================================================================
# Shared Helper Functions (geometry utilities)
# ============================================================================

def fit_text(ax, x, y, text, width_in, fontsize, **kwargs):
    """
    @brief   Draw text, shrinking the font (>= FIT_TEXT_MIN_FONTSIZE pt) until
             it fits within the given width in figure inches.

    @details Guards the class of defect where a fixed-size box clips or garbles
             its label (review finding: clipped head dimension labels).

    @param   ax         Axes to draw the text onto.
    @param   x          X coordinate in axes data coordinates.
    @param   y          Y coordinate in axes data coordinates.
    @param   text       Text string to render.
    @param   width_in   Maximum allowed width in figure inches.
    @param   fontsize   Starting (maximum) font size.
    @param   kwargs     Additional kwargs passed to ax.text().
    @return  The created matplotlib.text.Text instance.
    @note    Binary-steps font size down by FIT_TEXT_STEP per check.
    """
    figure = ax.figure
    renderer = figure.canvas.get_renderer()
    size = fontsize
    while size > FIT_TEXT_MIN_FONTSIZE:
        probe = ax.text(x, y, text, fontsize=size, **kwargs)
        box = probe.get_window_extent(renderer=renderer)
        probe.remove()
        if box.width / figure.dpi <= width_in:
            break
        size -= FIT_TEXT_STEP
    return ax.text(x, y, text, fontsize=size, **kwargs)


def rounded_box(ax, x, y, w, h,
                face=ROUNDED_BOX_DEFAULT_FACE, edge=ROUNDED_BOX_DEFAULT_EDGE,
                lw=ROUNDED_BOX_DEFAULT_LW, radius=ROUNDED_BOX_DEFAULT_RADIUS):
    """
    @brief   Draw a rounded-corner rectangle patch on the given axes.

    @param   ax      Target axes.
    @param   x       Lower-left X (axes coordinates).
    @param   y       Lower-left Y (axes coordinates).
    @param   w       Box width.
    @param   h       Box height.
    @param   face    Face fill color.
    @param   edge    Edge stroke color.
    @param   lw      Edge line width.
    @param   radius  Corner rounding radius parameter for FancyBboxPatch.
    @return  The created FancyBboxPatch instance (already added to axes).
    @note    Default zorder=1 so text/scatters above default plots are on top.
    """
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={ROUNDED_BOX_DEFAULT_PAD},rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=1
    )
    ax.add_patch(patch)
    return patch


def band_arrow(fig, x0, x1, label):
    """
    @brief   Thick main-flow arrow between two bands, drawn in figure coordinates.

    @param   fig     Figure instance to draw the overlay axes onto.
    @param   x0      Start X coordinate (figure normalized 0..1).
    @param   x1      End X coordinate (figure normalized 0..1).
    @param   label   Flow label string (currently unused, kept for signature
                     compatibility; review consensus removed rotated labels;
                     band titles already carry the semantics).
    @return  None (overlay axes + patch are side effects on fig).
    @note    Creates a full-figure overlay axes that spans 0..1 both axes.
    """
    overlay = fig.add_axes([0, 0, 1, 1], zorder=5)
    overlay.set_axis_off()
    y = BAND_ARROW_Y
    overlay.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=BAND_ARROW_MUT_SCALE,
            linewidth=BAND_ARROW_LINEWIDTH, color="black"
        )
    )
    # Flow labels removed per review consensus (no rotated text).
    # The band titles already carry the semantics. Keep label unused.
    _ = label


# ============================================================================
# Fig. 1 - Architecture Overview (Horizontal Three-Band Pipeline)
# ============================================================================
def make_fig1(out_dir):
    """
    @brief   Architecture overview - designed three-panel style (aligned, framed).

    @details Panel (a) multi-cancer network: deterministic TP53-centred PPI sample.
             Panel (b) pretraining: 6-layer HGT encoder + three SSL task boxes.
             Panel (c) downstream evaluation: outcome cards with status coloring.
             Inter-panel arrows connect bands in main flow direction.

    @param   out_dir   Output directory for fig1_architecture (.pdf/.png).
    @return  None (figure is saved through fs.save_fig()).
    """
    layout = fd.load_fig1_layout()
    fig = plt.figure(figsize=FIG1_FIGSIZE, facecolor="white")
    # Thin grey frame around the whole canvas (matches the approved render).
    fig.patch.set_edgecolor(FIG1_FRAME)
    fig.patch.set_linewidth(FIG1_FRAME_LINEWIDTH)

    band = FIG1_BANDS
    titles = FIG1_PANEL_TITLES
    top, bottom, height = FIG1_PANEL_TOP, FIG1_PANEL_BOTTOM, FIG1_PANEL_HEIGHT

    def new_ax(key):
        ax = fig.add_axes([band[key][0], bottom, band[key][1] - band[key][0], height])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.0, top, titles[key], transform=ax.transAxes, fontsize=FIG1_PANEL_TITLE_FONTSIZE,
                fontweight="bold", va="top", color="#111111")
        return ax

    # ---- panel (a): inputs + network + legend -----------------------------
    ax_a = new_ax("a")
    inputs = FIG1_A_INPUTS
    for row, label in enumerate(inputs):
        y = FIG1_A_INPUT_ROW_Y_START - row * FIG1_A_INPUT_ROW_Y_STEP
        rounded_box(ax_a, 0.02, y, FIG1_INPUT_BOX_WIDTH, FIG1_INPUT_BOX_HEIGHT,
                    face=FIG1_BOX_FILL, edge=FIG1_BOX_EDGE, lw=FIG1_INPUT_BOX_LW)
        fit_text(ax_a, 0.19, y + FIG1_INPUT_BOX_HEIGHT / 2, label,
                 0.30, FIG1_INPUT_LABEL_FONTSIZE,
                 ha="center", va="center")
        if row < 2:
            ax_a.add_patch(
                FancyArrowPatch(
                    (0.365, y + FIG1_INPUT_BOX_HEIGHT / 2),
                    (0.43, y + FIG1_INPUT_BOX_HEIGHT / 2),
                    arrowstyle="-|>", mutation_scale=9, lw=0.7, color="#666666"
                )
            )

    # Network sample (real TP53 neighbourhood, single grey edges).
    box_x, box_y, box_w, box_h = (
        FIG1_A_NETWORK_BOX_X, FIG1_A_NETWORK_BOX_Y,
        FIG1_A_NETWORK_BOX_W, FIG1_A_NETWORK_BOX_H,
    )
    rounded_box(ax_a, box_x, box_y, box_w, box_h,
                face=FIG1_NETWORK_BOX_FACE, edge=FIG1_BOX_EDGE, lw=FIG1_NETWORK_BOX_LW)
    positions = layout["pos"]
    gene_ids, cancer_ids, patient_ids = (
        layout["gene_ids"], layout["cancer_ids"], layout["patient_ids"]
    )
    gene_raw = np.array([positions[str(i)] for i in gene_ids], dtype=float)
    cancer_raw = np.array([positions[str(i)] for i in cancer_ids], dtype=float)
    patient_raw = np.array([positions[str(i)] for i in patient_ids], dtype=float)
    all_raw = np.vstack([gene_raw, cancer_raw, patient_raw])
    low, high = all_raw.min(axis=0), all_raw.max(axis=0)
    span = np.where((high - low) == 0, 1.0, high - low)
    scale = np.array([box_w * FIG1_A_NETWORK_SCALE_X, box_h * FIG1_A_NETWORK_SCALE_Y]) / span
    origin = np.array([box_x + box_w * FIG1_A_NETWORK_ORIGIN_X_RATIO,
                       box_y + box_h * FIG1_A_NETWORK_ORIGIN_Y_RATIO])

    def project(points):
        return (np.asarray(points, dtype=float) - low) * scale + origin

    gene_xy, cancer_xy, patient_xy = (
        project(gene_raw), project(cancer_raw), project(patient_raw)
    )
    for start, end in layout["edges_gene_local"][:FIG1_NETWORK_MAX_EDGES_LOCAL]:
        ax_a.plot([gene_xy[start, 0], gene_xy[end, 0]],
                  [gene_xy[start, 1], gene_xy[end, 1]],
                  color=FIG1_NETWORK_EDGE_COLOR, lw=FIG1_NETWORK_EDGE_LW,
                  alpha=FIG1_NETWORK_EDGE_ALPHA, zorder=1)
    cl = {n: project([positions[str(n)]])[0] for n in cancer_ids}
    pl = {n: project([positions[str(n)]])[0] for n in patient_ids}
    for start, node in layout["edges_cancer_local"]:
        ax_a.plot([gene_xy[start, 0], cl[node][0]], [gene_xy[start, 1], cl[node][1]],
                  color=FIG1_NETWORK_EDGE_COLOR, lw=FIG1_NETWORK_EDGE_LW,
                  alpha=FIG1_NETWORK_EDGE_ALPHA, zorder=1)
    for start, node in layout["edges_patient_local"]:
        ax_a.plot([gene_xy[start, 0], pl[node][0]], [gene_xy[start, 1], pl[node][1]],
                  color=FIG1_NETWORK_EDGE_COLOR, lw=FIG1_NETWORK_EDGE_LW,
                  alpha=FIG1_NETWORK_EDGE_ALPHA, zorder=1)
    ax_a.scatter(gene_xy[:, 0], gene_xy[:, 1], s=FIG1_GENE_SCATTER_SIZE,
                 marker="o", color="#0470AA", zorder=3)
    ax_a.scatter(cancer_xy[:, 0], cancer_xy[:, 1], s=FIG1_CANCER_SCATTER_SIZE,
                 marker="s", color="#D35E00", zorder=3)
    ax_a.scatter(patient_xy[:, 0], patient_xy[:, 1], s=FIG1_PATIENT_SCATTER_SIZE,
                 marker="^", color="#009E73", zorder=3)
    ax_a.text(0.02, 0.32,
              "Deterministic TP53-centred sample of the real edge set.\n"
              "Shown: 5,000-gene ESM subgraph (544,908 PPI edges);\n"
              "full graph: 13,881 genes / 770,440 PPI edges.",
              fontsize=FIG1_FOOTNOTE_FONTSIZE, va="top", color="#444444")
    for row, (marker, color, label) in enumerate(FIG1_A_LEGEND_ROWS):
        y = FIG1_A_LEGEND_Y_START - row * FIG1_A_LEGEND_Y_STEP
        ax_a.scatter([FIG1_A_LEGEND_X_SCATTER], [y], s=FIG1_LEGEND_SCATTER_SIZE,
                     marker=marker, color=color, zorder=3)
        ax_a.text(FIG1_A_LEGEND_X_TEXT, y, label, fontsize=FIG1_LEGEND_LABEL_FONTSIZE,
                  va="center", color="#111111")
    ax_a.text(0.02, 0.015,
              "Patient nodes carry no incident edges and are excluded from\n"
              "gene-level pretraining (used only in patient-level experiments).",
              fontsize=FIG1_CAPTION_FONTSIZE, va="bottom", color="#666666")

    # ---- panel (b): pretraining -------------------------------------------
    ax_b = new_ax("b")
    header_y = FIG1_B_HEADER_Y
    fit_text(ax_b, 0.5, header_y, "6-layer HGT \u00b7 8 heads \u00b7 256-dim", 0.9,
             FIG1_B_HEADER_FONTSIZE,
             ha="center", fontweight="bold", color="#111111")
    # 6 layer bars (left) + shared-encoder band + task boxes (right).
    for layer in range(FIG1_B_NUM_LAYERS):
        y = FIG1_B_LAYER_Y_START + layer * FIG1_B_LAYER_Y_STEP
        rounded_box(ax_b, FIG1_B_LAYER_X, y, FIG1_LAYER_BOX_WIDTH, FIG1_LAYER_BOX_HEIGHT,
                    face=FIG1_B_ENCODER_BAND_FACE, edge=FIG1_BOX_EDGE,
                    lw=FIG1_LAYER_BOX_LW)
        ax_b.text(FIG1_B_LAYER_X + FIG1_LAYER_BOX_WIDTH / 2, y + FIG1_LAYER_BOX_HEIGHT / 2,
                  f"HGT {layer + 1}", fontsize=FIG1_ENCODER_LAYER_FONTSIZE,
                  ha="center", va="center")
    band_x, band_w = FIG1_SHARED_ENCODER_BAND_X, FIG1_SHARED_ENCODER_BAND_W
    band_patch = FancyBboxPatch(
        (band_x, FIG1_SHARED_ENCODER_BAND_Y), band_w, FIG1_SHARED_ENCODER_BAND_H,
        boxstyle="round,pad=0.004",
        facecolor=FIG1_TINT_ENC, edgecolor=FIG1_B_ENCODER_BAND_EDGE,
        linewidth=FIG1_B_ENCODER_BAND_LW, zorder=1
    )
    ax_b.add_patch(band_patch)
    ax_b.text(band_x + band_w / 2, 0.44, "shared encoder", rotation=90,
              fontsize=FIG1_ENCODER_LABEL_FONTSIZE,
              ha="center", va="center", color=FIG1_B_ENCODER_BAND_LABEL_COLOR, zorder=2)
    for row, (abbrev, detail) in enumerate(FIG1_TASK_LABELS):
        y = FIG1_B_TASK_Y_START - row * FIG1_B_TASK_Y_STEP
        rounded_box(ax_b, FIG1_B_TASK_X, y, FIG1_TASK_BOX_WIDTH, FIG1_TASK_BOX_HEIGHT,
                    face="white", edge=FIG1_BOX_EDGE, lw=FIG1_TASK_BOX_LW)
        ax_b.text(0.48, y + 0.095, abbrev, fontsize=FIG1_B_TASK_ABBREV_FONTSIZE,
                  fontweight="bold", va="center")
        fit_text(ax_b, 0.48, y + 0.04, detail, 0.46, FIG1_TASK_DETAIL_FONTSIZE,
                 va="center", color="#222222")
        ax_b.add_patch(
            FancyArrowPatch(
                (band_x + band_w, 0.085 + layer * 0.115 + 0.0425),
                (0.435, y + 0.075),
                arrowstyle="-|>", mutation_scale=FIG1_B_ARROW_MUT_SCALE,
                lw=FIG1_B_ARROW_LW, color=FIG1_B_ARROW_COLOR, zorder=2
            )
        )

    # ---- panel (c): downstream evaluation ----------------------------------
    ax_c = new_ax("c")
    rows = fd.load_eval_summary()
    namr, csp_namr, esm = rows["namr_only"], rows["csp_namr"], rows["esm_only"]
    patient = fd.load_patient_acc()
    k_index = patient["K"].index(5)
    cards = [
        ("Known-edge recovery (cosine probe)",
         f"AUC {namr['auc']:.3f} (NAMR-only)", "ok"),
        ("Trivial graph smoothing",
         "AUC 0.985 \u2014 exceeds trained model", "warn"),
        ("Drug response prediction (GDSC)",
         "R\u00b2 < 0 at all K (protocol-scoped boundary)", "negative"),
    ]
    # Two lines per card (task name on top, outcome on the second line) so the
    # long metric strings can never collide with the task label.
    card_top, card_h, gap = FIG1_CARD_TOP, FIG1_CARD_HEIGHT, FIG1_CARD_GAP
    for index, (label, badge, status) in enumerate(cards):
        y = card_top - index * (card_h + gap)
        face = FIG1_C_STATUS_TINT_MAP.get(status, FIG1_BOX_FILL)
        rounded_box(ax_c, 0.03, y, FIG1_CARD_WIDTH, card_h,
                    face=face, edge=FIG1_BOX_EDGE, lw=FIG1_CARD_LW)
        fit_text(ax_c, FIG1_C_CARD_PAD_X, y + card_h * FIG1_C_CARD_LABEL_RATIO_Y,
                 label, 0.88, FIG1_CARD_LABEL_FONTSIZE,
                 va="center", color="#111111")
        badge_color = FIG1_C_STATUS_COLOR_MAP.get(status, "#111111")
        prefix = FIG1_C_STATUS_BADGE_PREFIX_NEG if status == "negative" else ""
        fit_text(ax_c, FIG1_C_CARD_PAD_X, y + card_h * FIG1_C_CARD_BADGE_RATIO_Y,
                 prefix + badge, 0.88, FIG1_CARD_BADGE_FONTSIZE,
                 va="center", fontweight="bold", color=badge_color)

    # Inter-panel arrows (black, thick, at mid height).
    for x0, x1 in [
        (band["a"][1] + FIG1_INTER_PANEL_GAP, band["b"][0] - FIG1_INTER_PANEL_GAP),
        (band["b"][1] + FIG1_INTER_PANEL_GAP, band["c"][0] - FIG1_INTER_PANEL_GAP),
    ]:
        fig.add_artist(
            FancyArrowPatch(
                (x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=FIG1_FLOW_ARROW_MUT_SCALE,
                lw=FIG1_FLOW_ARROW_LW, color="#111111", zorder=6
            )
        )

    fs.save_fig(fig, os.path.join(out_dir, "fig1_architecture"))


# ============================================================================
# Fig. 2 / 3 - Main Text Figures
# ============================================================================
def make_fig2(out_dir, ckpt=None):
    """
    @brief   Objective losses and their disagreement with the encoder-level endpoint.

    @details Review item G2: the total-loss curve carried little information for
             the paper's claim and occupied main-text space, so it moves to the
             supplementary material (make_fig_s_total_loss) and panel (b) now
             shows the contrast the title is about: across arms, a lower
             reconstruction loss does not order the encoder endpoint.

    @param   out_dir   Output directory.
    @param   ckpt      Optional full checkpoint path override.
    @return  None (figure saved through fs.save_fig()).
    """
    ckpt = os.path.join(config.RESULT_DIR, FULL_CKPT) if ckpt is None else ckpt
    history = fd.load_loss_history(ckpt if os.path.isabs(ckpt) else config.resolve(ckpt))
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(MAIN_WIDTH_IN, 2.95))
    ax = axes[0]
    ax.plot(epochs, history["namr"], color=fs.COL_OK, label="NAMR (full model)")
    ax.plot(epochs, history["csp"], color=fs.COL_ALT, label="CSP")
    if "c3l" in history:
        ax.plot(epochs, history["c3l"], color=fs.COL_EXTRA, label="C3L (InfoNCE)")
        # Centralised analytical baseline (ln N_CANCER_TYPES = ln 21) from config §6b.
        ax.axhline(float(config.C3L_RANDOM_LEVEL_LN), ls=":", color=fs.COL_REF, lw=FIG2_REF_LINE_LW,
                   label=r"$\ln N_{cancer}$ (C3L uniform-prototype level)")
    ax.axhline(1.065, ls="--", color=fs.COL_REF, lw=FIG2_REF_LINE_LW,
               label="constant-mean reference (standardized NAMR target)")
    ax.set_title("(a, fontsize=10) Objective losses during training", fontsize=FIG2_SUPTITLE_FONTSIZE)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="y", left=False, which="both", labelsize=9)
    axes[0].tick_params(axis="x", bottom=False, which="both", labelsize=9)
    ax.legend(fontsize=FIG2_LEGEND_FONTSIZE, frameon=False, loc="upper center",
              bbox_to_anchor=FIG2_LEGEND_BBOX, ncol=FIG2_LEGEND_NCOL,
              handlelength=FIG2_LEGEND_HANDLELENGTH,
              labelspacing=FIG2_LEGEND_LABELSPACING)

    # panel (b): the disagreement - final NAMR loss versus known-edge recovery
    rows = fd.load_eval_summary()
    arms = [
        ("NAMR-only", "namr_only.pt", fs.COL_OK, rows["namr_only"]["auc"]),
        ("CSP+NAMR", "csp_namr_legacy_s42.pt", fs.COL_EXTRA, rows["csp_namr"]["auc"]),
        ("Full", "full.pt", fs.COL_ALT, rows["full"]["auc"]),
        ("CSP-only", "csp_only.pt", fs.COL_MAIN, rows["csp_only"]["auc"]),
        ("skip-decoder NAMR", "namr_only_s7.pt", fs.COL_REF, 0.601),
    ]
    ax = axes[1]
    for label, fname, colour, auc in arms:
        try:
            h = fd.load_loss_history(
                os.path.join(config.RESULT_DIR, "ablation_real", fname)
            )
            loss = (
                float(h["namr"][-1])
                if np.any(np.asarray(h.get("namr", []), dtype=float))
                else float(h["total"][-1])
            )
        except Exception:
            continue
        ax.scatter(loss, auc, s=FIG2_ARMS_SCATTER_SIZE, color=colour, zorder=3,
                   edgecolor="white", linewidth=FIG2_ARMS_SCATTER_EDGE_WIDTH)
        ax.annotate(label, (loss, auc), textcoords="offset points",
                    xytext=FIG2_ARMS_ANNOT_XYOFFSET,
                    fontsize=FIG2_ARMS_COL_LABEL_FONTSIZE,
                    ha="left", va="bottom", annotation_clip=False)
    ax.axhline(fs.CHANCE_LEVEL_AUC, ls=":", color="black", lw=FIG2_REF_LINE_LW)
    smooth = _smoothing_auc()
    if smooth is not None:
        ax.axhline(smooth, ls="--", color=fs.COL_GENE, lw=FIG2_SMOOTH_LINE_LW)
        # Panel (b) reference line label: inline to avoid legend collision with
        # x-axis label below the axes.
        ax.text(
            FIG2_SMOOTH_REF_TEXT_OFFSET_X, smooth - FIG2_SMOOTH_REF_TEXT_OFFSET_Y,
            f"trivial smoothing, no training ({smooth:.3f})",
            transform=ax.get_yaxis_transform(),
            fontsize=FIG2_SMOOTH_REF_TEXT_FONTSIZE,
            color=fs.COL_GENE, ha="left", va="top"
        )
    ax.set_xscale("log")
    ax.set_xlabel("Final reconstruction (NAMR, fontsize=9) loss, log scale")
    ax.set_ylabel("Known-edge recovery AUC", fontsize=9)
    ax.set_title("(b, fontsize=10) Lower loss does not order the endpoint", fontsize=FIG2_SUPTITLE_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    # Both panels carry their legend below the axes, so reserve space for it.
    fig.subplots_adjust(bottom=FIG2_BOTTOM_PAD, wspace=FIG2_WSPACE)
    fs.save_fig(fig, os.path.join(out_dir, "fig2_convergence_v2"))


def make_fig_s_total_loss(out_dir, ckpt=None):
    """
    @brief   Total training loss, moved out of the main text (review item G2).

    @param   out_dir   Output directory.
    @param   ckpt      Optional full checkpoint path override.
    @return  None (figure saved through fs.save_fig()).
    """
    ckpt = os.path.join(config.RESULT_DIR, FULL_CKPT) if ckpt is None else ckpt
    history = fd.load_loss_history(ckpt if os.path.isabs(ckpt) else config.resolve(ckpt))
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=FIG2_TOTAL_LOSS_FIGSIZE)
    ax.plot(epochs, history["total"], color=fs.COL_MAIN)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Total loss", fontsize=9)
    ax.set_title("Total training loss of the full model (all three objectives, fontsize=10)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fs.save_fig(fig, os.path.join(out_dir, "figS_total_loss"))


def _smoothing_auc():
    """
    @brief   Best trivial graph-smoothing known-edge AUC (no training), for reference.

    @return  Float max known-edge AUC across smoothing variants; None if the
             control JSON is missing or unreadable.
    @note    Reads results/control/graph_smoothing_baseline.json.
    """
    path = os.path.join(config.RESULT_DIR, "control", "graph_smoothing_baseline.json")
    try:
        with open(path) as handle:
            payload = json.load(handle)
        return max(float(v) for v in payload["known_edge_auc"].values())
    except Exception:
        return None


def _multiseed_known_edge():
    """
    @brief   Per-seed known-edge AUC per arm, from results/known_edge_multiseed.json.

    @return  "per_arm" dict from the multiseed JSON payload.
    """
    path = os.path.join(config.RESULT_DIR, "known_edge_multiseed.json")
    with open(path) as handle:
        return json.load(handle)["per_arm"]


def make_fig3(out_dir):
    """
    @brief   Known-edge recovery per training seed, against the no-training references.

    @details Review item G3: a bar chart of single seed-42 evaluations understated
             the evidence. Each trained arm is now shown as all of its per-seed values,
             with the sample mean and SD, and the two multi-task arms are joined by
             seed so that the seed-dependent C3L effect is visible rather than
             averaged away.

    @param   out_dir   Output directory.
    @return  None (figure saved through fs.save_fig()).
    """
    per_arm = _multiseed_known_edge()
    rows = fd.load_eval_summary()
    arms = [
        ("Full", "full", fs.COL_ALT),
        ("NAMR-only", "namr_only", fs.COL_OK),
        ("CSP-only", "csp_only", fs.COL_MAIN),
        ("CSP+NAMR", "csp_namr", fs.COL_EXTRA),
    ]
    fig, ax = plt.subplots(1, 1, figsize=FIG3_FIGSIZE)
    rng = np.random.RandomState(FIG3_JITTER_RNG_SEED)
    paired = {}
    for pos, (label, key, colour) in enumerate(arms):
        d = per_arm[key]
        seeds, vals = d["seeds"], d["values"]
        jitter = rng.uniform(-FIG3_JITTER_RADIUS, FIG3_JITTER_RADIUS, size=len(vals))
        ax.scatter(pos + jitter, vals, s=FIG3_SCATTER_SIZE, color=colour, zorder=3,
                   edgecolor="white", linewidth=FIG3_SCATTER_EDGE_WIDTH)
        ax.errorbar(pos, d["mean"], yerr=d["sd_ddof1"],
                    fmt=FIG3_ERRORBAR_FMT, ms=FIG3_ERRORBAR_MS,
                    color=colour, capsize=FIG3_ERRORBAR_CAPSIZE,
                    lw=FIG3_ERRORBAR_LW, zorder=4)
        right_side = pos < len(arms) - 1
        ax.text(
            pos + (0.28 if right_side else -0.30), d["mean"],
            f"{d['mean']:.3f}\n$\\pm${d['sd_ddof1']:.3f}\n$n$={len(vals)}",
            fontsize=FIG3_ARM_LABEL_FONTSIZE,
            va="center", ha="left" if right_side else "right", color=colour
        )
        paired[label] = {s: v for s, v in zip(seeds, vals)}
    # Join the two multi-task arms seed by seed: the C3L effect is seed-dependent.
    shared = sorted(set(paired["Full"]) & set(paired["CSP+NAMR"]))
    for seed in shared:
        ax.plot([0, 3], [paired["Full"][seed], paired["CSP+NAMR"][seed]],
                color=fs.COL_PAIRED, lw=FIG3_PAIRED_LINE_LW, zorder=1)
    # Single-evaluation reference conditions.
    refs = [
        ("random-init", rows["random_init_hgt"]["auc"]),
        ("ESM-2 only", rows["esm_only"]["auc"]),
        ("PCA input", rows["pca_input"]["auc"]),
    ]
    for k, (name, val) in enumerate(refs):
        ax.scatter(len(arms) + FIG3_REF_SCATTER_GAP_X + FIG3_REF_SCATTER_STEP_X * k,
                   val, marker="s", s=FIG3_REF_SCATTER_SIZE,
                   color=fs.COL_REF, zorder=3)
    ax.text(len(arms) + 0.36, min(v for _, v in refs) - FIG3_REF_LABEL_OFFSET_Y,
            "references (single run)", fontsize=FIG3_REF_LABEL_FONTSIZE,
            color=fs.COL_REF, ha="center", va="top")
    ax.axhline(fs.CHANCE_LEVEL_AUC, ls=":", color="black", lw=FIG3_CHANCE_LINE_LW)
    smooth = _smoothing_auc()
    if smooth is not None:
        ax.axhline(smooth, ls="--", color=fs.COL_GENE, lw=FIG3_SMOOTH_LINE_LW)
        ax.text(-0.42, smooth + FIG3_SMOOTH_TEXT_OFFSET_Y,
                f"trivial smoothing, no training ({smooth:.3f})",
                fontsize=FIG3_SMOOTH_TEXT_FONTSIZE,
                color=fs.COL_GENE, ha="left", va="bottom")
    ax.set_xticks(
        list(range(len(arms))) +
        [len(arms) + FIG3_REF_SCATTER_GAP_X + FIG3_REF_SCATTER_STEP_X * k for k in range(len(refs))]
    )
    ax.set_xticklabels(
        [a[0] for a in arms] + [r[0] for r in refs],
        rotation=FIG3_XTICK_ROT, ha="right", fontsize=FIG3_XTICK_FONTSIZE
    )
    ax.set_xlim(-0.5, len(arms) + 1.15)
    ax.set_ylabel("Known-edge recovery AUC", fontsize=9)
    # Lowest arm (CSP-only, mean - SD = 0.505) would sit on the axis at ylim=0.5,
    # so the axis starts below chance and the chance line stays visible.
    ax.set_ylim(FIG3_YLIM_LOW, FIG3_YLIM_HIGH)
    ax.set_title("Known-edge recovery by training seed (mean $\\pm$ SD, fontsize=10)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fs.save_fig(fig, os.path.join(out_dir, "fig3_ablation_v2"))


# ============================================================================
# Supplementary Figures
# ============================================================================
def make_fig_s1(out_dir, base_ref=None, h1_std=None):
    """
    @brief   NAMR-only standardization ablation (100 epochs, namr_sweep).

    @details Compares unstandardized-feature vs standardized-feature NAMR loss
             trajectories.

    @param   out_dir    Output directory.
    @param   base_ref   Optional path to unstandardized baseline checkpoint.
    @param   h1_std     Optional path to standardized-feature checkpoint.
    @return  None (figure saved through fs.save_fig()).
    """
    base_ref = os.path.join(config.RESULT_DIR, BASE_REF_CKPT) if base_ref is None else base_ref
    h1_std = os.path.join(config.RESULT_DIR, H1_STD_CKPT) if h1_std is None else h1_std
    unstandardized = fd.load_loss_history(config.resolve(base_ref))
    standardized = fd.load_loss_history(config.resolve(h1_std))
    fig, ax = plt.subplots(figsize=S1_FIGSIZE)
    ax.plot(np.arange(1, len(unstandardized["namr"]) + 1), unstandardized["namr"],
            color=fs.COL_ALT,
            label="no standardization, 100 epochs (predict-mean level \u22481.32)")
    ax.plot(np.arange(1, len(standardized["namr"]) + 1), standardized["namr"],
            color=fs.COL_MAIN, label="standardized features")
    ax.set_xlabel("Epoch (100-epoch NAMR-only sweep, fontsize=9)")
    ax.set_ylabel("NAMR loss", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fs.save_fig(fig, os.path.join(out_dir, "figS_namr_fix"))


def make_fig_s2(out_dir):
    """
    @brief   Patient-level few-shot learning curve (log2 K axis).

    @details Compares P-NAMR against expression PCA, top-gene multi-hot,
             ESM-2-only aggregation, untrained patient graph, GFM-only, and
             the 1/21 uniform chance baseline.

    @param   out_dir   Output directory.
    @return  None (figure saved through fs.save_fig()).
    """
    payload = fd.load_patient_acc()
    fig, ax = plt.subplots(figsize=S2_FIGSIZE)
    ax.plot(payload["K"], payload["acc"], "o-", color=fs.COL_MAIN, ms=S2_SCATTER_MS,
            label="Patient-level pretraining (P-NAMR)")
    ax.axhline(0.844, ls="--", color=fs.COL_EXTRA, lw=1.4,
               label="Raw expression PCA (50d) \u2014 local rebuild")
    ax.axhline(0.727, ls=":", color=fs.COL_ORANGE, lw=1.4,
               label="Raw top-gene-set multi-hot \u2014 local rebuild")
    ax.axhline(0.67, ls="--", color=fs.COL_ALT, lw=1.2,
               label="ESM-2-only aggregation (K=5)")
    ax.axhline(0.546, ls="-.", color=fs.COL_SKY, lw=1.2,
               label="Untrained patient graph \u2014 local rebuild")
    ax.axhline(0.29, ls=":", color=fs.COL_OK, lw=1.2,
               label="GFM gene aggregation (K=5)")
    # Uniform-chance accuracy computed centrally from config.N_CANCER_TYPES (=21 TCGA
    # cancer types; 1/N chance for a randomly-guessing classifier).
    ax.axhline(1.0 / config.N_CANCER_TYPES, ls="-", color="0.6", lw=1.0,
               label=f"Uniform chance (1/N_cancer = {1.0/config.N_CANCER_TYPES:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(payload["K"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Training patients per cancer type (K; log2, fontsize=9)")
    ax.set_ylabel("Few-shot accuracy", fontsize=9)
    ax.set_ylim(S2_YLIM_LOW, S2_YLIM_HIGH)
    ax.set_xlim(S2_XLIM_LOW, S2_XLIM_HIGH)
    ax.legend(fontsize=S2_LEGEND_FONTSIZE, loc="lower right",
              frameon=False, ncol=S2_LEGEND_NCOL,
              columnspacing=S2_LEGEND_COLSPACING)
    ax.grid(alpha=0.3, which="both")
    for k, value in zip(payload["K"], payload["acc"]):
        ax.annotate(f"{value:.3f}", (k, value), textcoords="offset points",
                    xytext=S2_ANNOT_XYOFFSET, ha="center", fontsize=S2_ANNOT_FONTSIZE)
    fig.tight_layout()
    fs.save_fig(fig, os.path.join(out_dir, "figS_patient_learning"))


def make_fig_s3(out_dir, plain_ckpt=None, skip_ckpt=None):
    """
    @brief   Skip-connected versus plain-decoder NAMR diagnostic.

    @details Panel (a): NAMR reconstruction loss per sample epoch for both decoders.
             Panel (b): known-edge recovery AUC bar chart per provenance variant.

    @param   out_dir      Output directory.
    @param   plain_ckpt   Optional plain-decoder NAMR-only checkpoint path.
    @param   skip_ckpt    Optional skip-connected NAMR diagnostic checkpoint path.
    @return  None (figure saved through fs.save_fig()).
    """
    plain_ckpt = os.path.join(config.RESULT_DIR, NAMR_ONLY_CKPT) if plain_ckpt is None else plain_ckpt
    skip_ckpt = os.path.join(config.RESULT_DIR, SKIP_DECODER_CKPT) if skip_ckpt is None else skip_ckpt
    provenance = fd.load_auc_provenance()
    plain = fd.sample_at_epochs(
        fd.load_loss_history(config.resolve(plain_ckpt)), "namr", S3_SAMPLE_EPOCHS
    )
    skip = fd.sample_at_epochs(
        fd.load_loss_history(config.resolve(skip_ckpt)), "namr", S3_SAMPLE_EPOCHS
    )
    fig, axes = plt.subplots(1, 2, figsize=S3_FIGSIZE)
    epochs = np.asarray(S3_SAMPLE_EPOCHS)
    axes[0].plot(epochs, plain, "-o", ms=S3_SCATTER_MS_PLAIN,
                 color=fs.COL_MAIN, label="plain decoder (method)")
    axes[0].plot(epochs, skip, "-s", ms=S3_SCATTER_MS_SKIP,
                 color=fs.COL_ALT, label="skip-connected decoder (diagnostic)")
    axes[0].axhline(S3_LOSS_DASH_Y, ls="--", color=fs.COL_REF, lw=FIG2_REF_LINE_LW)
    axes[0].set_xlabel("epoch", fontsize=9)
    axes[0].set_ylabel("NAMR loss", fontsize=9)
    axes[0].set_xlim(S3_XLIM_LOW, S3_XLIM_HIGH)
    axes[0].set_ylim(S3_YLIM_LOSS_LOW, S3_YLIM_LOSS_HIGH)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title(S3_SUPTITLE_FONTSIZE_A, fontsize=10)
    # Single-line tick labels (review 2026-09-17, R5): the two-line
    # "plain NAMR\n(seed 42)" form produced adjacent labels only ~4 px apart at
    # 300 dpi. The run/seed identity is stated in the caption instead.
    names = provenance["bar_names"]
    tick_labels = [
        re.sub(S3_REPLACE_WS_PATTERN, " ", n).replace(
            S3_REPLACE_SEED_PATTERN, S3_REPLACE_SEED_REPLACEMENT
        )
        for n in names
    ]
    values = [float(provenance["bars"][name]) for name in names]
    colors = [provenance["colors"][name] for name in names]
    positions = np.arange(len(names))
    axes[1].bar(positions, values, color=colors, width=S3_BAR_WIDTH)
    axes[1].axhline(fs.CHANCE_LEVEL_AUC, ls=":", color="black", lw=S3_CHANCE_LINE_LW,
                    label="Chance (0.5)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(tick_labels, fontsize=S3_TICK_LABEL_FONTSIZE)
    axes[1].set_ylabel("known-edge recovery AUC (0.5\u20131.0, fontsize=9)")
    axes[1].set_ylim(S3_AUC_YLIM_LOW, S3_AUC_YLIM_HIGH)
    for index, value in enumerate(values):
        axes[1].text(index, value + S3_BAR_VALUE_OFFSET_Y,
                     f"{value:.3f}", ha="center", fontsize=S3_BAR_VALUE_FONTSIZE)
    axes[1].set_title(S3_SUPTITLE_FONTSIZE_B, fontsize=10)
    fig.tight_layout()
    fs.save_fig(fig, os.path.join(out_dir, "figS_skip_diagnostic"))


def make_fig_s_mechanism(out_dir):
    """
    @brief   Mechanism diagnostics (review item G5): spectral collapse and prototype decay.

    @details Panel (a): normalised singular-value spectrum per arm (top 20 shares),
             the evidence for the CSP-collapse claim.
             Panel (b): cancer-node prototype minimum pairwise Euclidean distance
             layer by layer, the degeneracy mechanism behind the C3L failure.

    @param   out_dir   Output directory.
    @return  None (figure saved through fs.save_fig()).
    """
    geo_path = os.path.join(config.RESULT_DIR, "control", "geometry_cross_seed.json")
    proto_path = os.path.join(config.RESULT_DIR, "control", "c3l_prototype_layers.json")
    geo = json.load(open(geo_path))
    proto = json.load(open(proto_path))
    fig, axes = plt.subplots(1, 2, figsize=S_MECHANISM_FIGSIZE)

    ax = axes[0]
    for arm, colour, label in [
        ("namr_only", fs.COL_OK, "NAMR-only"),
        ("csp_only", fs.COL_MAIN, "CSP-only"),
    ]:
        for seed, rec in sorted(geo["per_seed"][arm].items(), key=lambda kv: int(kv[0])):
            shares = rec["top20_shares"]
            ax.plot(range(1, len(shares) + 1), shares,
                    color=colour, lw=S_MECH_SCATTER_LW, alpha=S_MECH_SCATTER_ALPHA,
                    label=f"{label} (seed {seed})"
                    if seed == sorted(geo["per_seed"][arm])[0] else None)
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index (top 20, fontsize=9)")
    ax.set_ylabel("Share of $\\sum s_i$", fontsize=9)
    ax.set_title("(a, fontsize=10) Spectrum: CSP collapse, all seeds", fontsize=S_MECH_TITLE_FONTSIZE)
    ax.legend(fontsize=S_MECH_ARM_LABEL_FONTSIZE, frameon=False, loc="upper right")

    ax = axes[1]
    layers = [0] + [int(r["layer"]) for r in proto["layers"]]
    dist = [proto["cancer_input"]["min_pairwise_euclidean"]] + \
           [float(r["min_pairwise_euclidean"]) for r in proto["layers"]]
    ax.plot(layers, dist, marker=S_MECH_MARKER, ms=S_MECH_MARKER_MS,
            color=fs.COL_ALT, lw=S_MECH_LINE_LW)
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("HGT layer (0 = one-hot input, fontsize=9)")
    ax.set_ylabel("Min. pairwise distance (cancer nodes, fontsize=9)")
    ax.set_title("(b, fontsize=10) Prototype distance with depth", fontsize=S_MECH_TITLE_FONTSIZE)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout(pad=S_MECH_TIGHT_PAD)
    fs.save_fig(fig, os.path.join(out_dir, "figS_mechanism"))


# ============================================================================
# Figure Builder Registry & Main Entry Point
# ============================================================================

BUILDERS = {
    # fig1 is intentionally NOT regenerated here: the shipped Fig. 1 is the
    # author-approved Nano Banana Pro bitmap (figures/fig1_architecture.png),
    # upscaled to 300 dpi. make_fig1() remains below only as a reference; running
    # this script must never overwrite the approved figure.
    "fig2": make_fig2,
    "fig3": make_fig3,
    "figS_total_loss": make_fig_s_total_loss,
    "figS_mechanism": make_fig_s_mechanism,
    "s1": make_fig_s1,
    "s2": make_fig_s2,
    "s3": make_fig_s3,
}


def main():
    """
    @brief   CLI entry point: build selected (or all) manuscript figures.

    @details Applies shared matplotlib style, ensures output directory exists,
             then dispatches to each registered figure builder in order.

    @return  None (figures saved to args.fig_dir via fs.save_fig()).
    """
    parser = argparse.ArgumentParser(description="Generate all manuscript figures.")
    parser.add_argument(
        "--only", nargs="+", choices=sorted(BUILDERS), default=None,
        help="subset of figures to build (default: all)"
    )
    parser.add_argument(
        "--fig-dir", default=config.FIG_DIR,
        help="output directory (default: config.FIG_DIR)"
    )
    args = parser.parse_args()

    fs.apply_style()
    os.makedirs(args.fig_dir, exist_ok=True)
    for name in (args.only or sorted(BUILDERS)):
        BUILDERS[name](args.fig_dir)


if __name__ == "__main__":
    main()
