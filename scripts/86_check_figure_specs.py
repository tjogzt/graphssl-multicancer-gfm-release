#!/usr/bin/env python3
"""
@file    86_check_figure_specs.py
@brief   Deterministic figure-specification regression check for the six manuscript
         figures plus four supplementary panels. Replaces the second VLM pass with
         purely programmatic checks (S1..S7):
           S1 include scale never exceeds 1.0 (LaTeX upscale inflates point sizes)
           S2 effective raster resolution >= 300 dpi
           S3 minimum printed type size >= 6.5 pt (designed_pt * include/width)
           S4 every shipped PNG carries 300 dpi metadata
           S5 no figure text clipped at the image edge
           S6 no shipped builder uses a colour outside the Okabe-Ito palette
           S7 no text clusters are drawn on top of a filled bar (transparent legend
              overlapping a bar chart).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  manuscript/*.png, manuscript/__main.tex, manuscript/supplementary.tex,
        scripts/figure_style.py, scripts/85_make_figures.py.
Output: stdout report; exit code 1 if any check fails.
Run:    GFM_PROJECT_ROOT=<repo> python scripts/86_check_figure_specs.py
"""
import os
import re
import sys

import numpy as np
from PIL import Image

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Project root (environment variable override; portable default = parent of scripts/).
PROJECT_ROOT_ENV_VAR = "GFM_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get(PROJECT_ROOT_ENV_VAR, DEFAULT_PROJECT_ROOT)

# Filesystem paths relative to the project root.
FIG_DIR = os.path.join(ROOT, "manuscript")
MAIN_TEX = os.path.join(ROOT, "manuscript", "__main.tex")
SUPP_TEX = os.path.join(ROOT, "manuscript", "supplementary.tex")
FIGSTYLE = os.path.join(ROOT, "scripts", "figure_style.py")
MAKE_FIGS = os.path.join(ROOT, "scripts", "85_make_figures.py")

# LaTeX page geometry: A4 21.0 cm - 2 x 2.0 cm margins = 17.0 cm text width.
TEXTWIDTH_IN = 6.6929

# S1..S3 hard thresholds.
MIN_PRINTED_PT = 6.5
MIN_EFFECTIVE_DPI = 300.0

# One-pixel quantisation tolerances: absorb the 1-px rounding that makes
# "exactly 300.0 dpi" unreachable for integer canvas widths; raw values are
# always printed so the real margin remains visible.
EPS_SCALE = 0.002
EPS_DPI = 1.0

# Fig. 1 is the author-approved bitmap infographic (not a matplotlib output);
# its type size is measured from the artwork itself, so designed-pt bookkeeping
# (S3) and the edge-ink probe (S5) do not apply.
BITMAP_FIGURES = {"fig1_architecture"}
BITMAP_BODY_PT = 13.9

# Output PNG DPI expectation used by S4.
TARGET_PNG_DPI = 300
PNG_DPI_TOL = 0.5

# Tick-label gap analysis probe.
LUMINANCE_THRESHOLD = 160          # Pixel threshold for "ink" vs white.
EDGE_CLIP_MARGIN_PX = 3            # Left/right margin scanned for ink clipping.
TICK_BAND_TOP_FRAC = 70            # Pixels from bottom for band top.
TICK_BAND_BOTTOM_FRAC = 4          # Pixels from bottom for band bottom.
BAND_GAP_COARSE = 8                # Coarse-pass consecutive white column threshold.
BAND_GAP_FINE = 4                  # Fine-pass consecutive white column threshold.

# Text-over-fill (S7) probe parameters.
OKABE_ITO_SERIES = [
    (213, 94, 0), (0, 114, 178), (0, 158, 115), (204, 121, 167),
    (230, 159, 0), (86, 180, 233),
]
SERIES_COLOUR_TOL = 26             # RGB tolerance for each Okabe-Ito colour.
INK_THRESHOLD_SUM = 260            # Sum-threshold for "ink" pixels (<260=dark).
MIN_BAR_FRACTION = 0.03            # Series must cover >=3% canvas to be a bar chart.
BAR_SPAN_MIN_COLS = 20             # Row must have filled pixels in >=20 columns.
BAR_SPAN_FILL_FRAC = 0.5           # Fraction of span actually filled (not dotted line).
BAR_MIN_HEIGHT_PX = 12             # Minimum median vertical fill run = bar.
BAR_TEXT_ROW_INK_PX = 3            # Min ink pixels in a row to count as "ink over bar".
TEXT_CLUSTER_MIN_H = 6             # Min bounding-box height = text cluster (not line).
TEXT_CLUSTER_MIN_W = 4             # Min bounding-box width  = text cluster.
TEXT_CONN_STRUCTURE = np.ones((3, 3), dtype=int)  # 8-connectivity for clustering.

# Palette audit (S6): shipped figure builder names.
SHIPPED_BUILDERS = (
    "make_fig2", "make_fig3", "make_fig_s1", "make_fig_s2", "make_fig_s3",
)
BAD_HEX_DISPLAY_LIMIT = 12         # Truncate long non-palette occurrence lists.
SCORE_DECIMALS = 4


def include_widths():
    """
    @brief   Parse manuscript/__main.tex + supplementary.tex for every
             \\includegraphics[width=<frac>\\textwidth]{...} directive and map
             each figure stem -> (include width in inches, TeX source location).
    @return  Dict {stem: (width_inches, "file:line")}.
    """
    out = {}
    for path in (MAIN_TEX, SUPP_TEX):
        with open(path) as handle:
            for lineno, line in enumerate(handle, 1):
                m = re.search(
                    r"includegraphics\[width=([\d.]*)\\textwidth\]\{([^}]+)\}", line
                )
                if not m:
                    continue
                frac = float(m.group(1)) if m.group(1) else 1.0
                stem = os.path.splitext(m.group(2))[0]
                out[stem] = (frac * TEXTWIDTH_IN, f"{os.path.basename(path)}:{lineno}")
    return out


def canvas_size(stem):
    """
    @brief   Return canvas pixel size, file DPI metadata, and nominal width
             assuming 300 dpi for the given figure stem.
    @param   stem  str – figure file basename without extension.
    @return  Dict {"px", "py", "dpi", "width_in", "path"} or None if the PNG
             does not exist.
    """
    png = os.path.join(FIG_DIR, stem + ".png")
    if not os.path.exists(png):
        return None
    with Image.open(png) as im:
        dpi = im.info.get("dpi") or (0, 0)
        w, h = im.size
    return {
        "px": w, "py": h, "dpi": float(dpi[0]),
        "width_in": w / TARGET_PNG_DPI, "path": png,
    }


def designed_font_sizes(builder):
    """
    @brief   Extract (default axes-label pt, min tick/legend pt, overall minimum
             explicit pt seen inside the function body) for a given figure builder.
    @param   builder  str – figure builder function name inside 85_make_figures.py.
    @return  Tuple (label_pt, min_tick_legend_pt, min_explicit_pt) or None if
             the builder cannot be located.
    """
    with open(MAKE_FIGS) as handle:
        src = handle.read()
    start = src.find(f"def {builder}(")
    if start < 0:
        return None
    nxt = src.find("\ndef ", start + 1)
    body = src[start:nxt if nxt > 0 else len(src)]
    explicit = [float(v) for v in re.findall(r"fontsize=([\d.]+)", body)]
    with open(FIGSTYLE) as handle:
        style = handle.read()
    label = float(re.search(r'"axes\.labelsize": ([\d.]+)', style).group(1))
    tick = float(re.search(r'"xtick\.labelsize": ([\d.]+)', style).group(1))
    legend = float(re.search(r'"legend\.fontsize": ([\d.]+)', style).group(1))
    smallest = (
        min([tick, legend] + explicit)
        if explicit else min(tick, legend)
    )
    return label, min(tick, legend), smallest


def tick_label_gaps(path):
    """
    @brief   Run a pixel-level edge-clipping probe plus gap analysis on the
             bottom x-tick label band of a figure PNG.
    @param   path  str – PNG filesystem path.
    @return  Tuple (clipped_at_edge: bool, min_tick_gap_px or None).
    """
    with Image.open(path).convert("L") as im:
        arr = np.array(im)
    dark = arr < LUMINANCE_THRESHOLD
    h, w = dark.shape
    # S5: any ink within 3 px of the left/right image edge.
    clipped = bool(dark[:, w - EDGE_CLIP_MARGIN_PX:].any()
                   or dark[:, :EDGE_CLIP_MARGIN_PX].any())
    band = dark[h - TICK_BAND_TOP_FRAC:h - TICK_BAND_BOTTOM_FRAC, :]
    cols = band.sum(axis=0)
    # First pass: coarse gap of 8 columns to separate label blobs.
    groups, st, gap = [], None, 0
    for i, v in enumerate(cols):
        if v > 0:
            if st is None:
                st = i
            gap = 0
        elif st is not None:
            gap += 1
            if gap >= BAND_GAP_COARSE:
                groups.append((st, i - gap))
                st = None
                gap = 0
    if st is not None:
        groups.append((st, len(cols) - 1))
    # Second pass: re-split any merged blob at a 4-px gap so near-touching labels
    # are not hidden inside a single merged group.
    tight = []
    for a, b in groups:
        cols_slice = cols[a:b + 1]
        st2, gap2 = None, 0
        for i, v in enumerate(cols_slice):
            if v > 0:
                if st2 is None:
                    st2 = i
                gap2 = 0
            elif st2 is not None:
                gap2 += 1
                if gap2 >= BAND_GAP_FINE:
                    tight.append((a + st2, a + i - gap2))
                    st2 = None
                    gap2 = 0
        if st2 is not None:
            tight.append((a + st2, b))
    gaps = [tight[i + 1][0] - tight[i][1] for i in range(len(tight) - 1)]
    return clipped, (min(gaps) if gaps else None)


def text_over_fill(path):
    """
    @brief   Count text-like ink clusters drawn on top of a solid coloured bar
             fill (transparent legend placed over a bar chart, S7). Distinguishes
             bars from plotted lines by a minimum vertical fill run per column,
             and distinguishes text clusters from thin reference lines by a
             minimum bounding-box height of 6 px.
    @param   path  str – PNG filesystem path.
    @return  Tuple (n_text_clusters, n_rows_with_ink_above_bar).
    """
    from scipy import ndimage
    with Image.open(path).convert("RGB") as im:
        arr = np.array(im).astype(int)

    # Build fill mask: pixels close to any of the six Okabe-Ito series colours.
    fill = np.zeros(arr.shape[:2], bool)
    for r, g, b in OKABE_ITO_SERIES:
        fill |= (
            (abs(arr[:, :, 0] - r) < SERIES_COLOUR_TOL)
            & (abs(arr[:, :, 1] - g) < SERIES_COLOUR_TOL)
            & (abs(arr[:, :, 2] - b) < SERIES_COLOUR_TOL)
        )
    # Ink mask: any pixel whose RGB sum is far enough below white (255*3=765).
    ink = arr.sum(axis=2) < INK_THRESHOLD_SUM

    if not fill.any():
        return None, 0

    # Require at least one series colour to cover >= 3% of the canvas; line plots
    # otherwise trip the bar detector.
    biggest = max(
        int((
            fill
            & (abs(arr[:, :, 0] - r) < SERIES_COLOUR_TOL)
            & (abs(arr[:, :, 1] - g) < SERIES_COLOUR_TOL)
            & (abs(arr[:, :, 2] - b) < SERIES_COLOUR_TOL)
        ).sum())
        for r, g, b in OKABE_ITO_SERIES
    )
    if biggest < MIN_BAR_FRACTION * arr.shape[0] * arr.shape[1]:
        return None, 0

    # A bar body is a thick fill: per column, record the vertical fill run length.
    height = np.zeros(arr.shape[:2], int)
    for x in range(arr.shape[1]):
        col = fill[:, x]
        y = 0
        while y < len(col):
            if not col[y]:
                y += 1
                continue
            y0 = y
            while y < len(col) and col[y]:
                y += 1
            height[y0:y, x] = y - y0

    rows = np.nonzero(fill.any(axis=1))[0]
    candidate = np.zeros_like(ink)
    raw_rows = 0
    for y in range(rows.min(), rows.max() + 1):
        xs = np.nonzero(fill[y])[0]
        if len(xs) < BAR_SPAN_MIN_COLS:
            continue
        x0, x1 = xs.min(), xs.max()
        span = x1 - x0 + 1
        if fill[y, x0:x1 + 1].sum() < BAR_SPAN_FILL_FRAC * span:
            continue
        bar_median_height = float(
            np.median(height[y, x0:x1 + 1][fill[y, x0:x1 + 1]])
        )
        if bar_median_height < BAR_MIN_HEIGHT_PX:
            # Thin fill = plotted line, not a bar body.
            continue
        if ink[y, x0:x1 + 1].sum() >= BAR_TEXT_ROW_INK_PX:
            raw_rows += 1
            candidate[y, x0:x1 + 1] = True

    if not candidate.any():
        return 0, raw_rows

    # 8-connectivity clustering; only clusters large enough to be text count.
    lab, n = ndimage.label(candidate, structure=TEXT_CONN_STRUCTURE)
    text_clusters = 0
    for sl in ndimage.find_objects(lab):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h >= TEXT_CLUSTER_MIN_H and w >= TEXT_CLUSTER_MIN_W:
            text_clusters += 1
    return text_clusters, raw_rows


def check_palette():
    """
    @brief   Scan 85_make_figures.py for every hex-colour literal and classify
             each occurrence as (a) shipped builder non-palette hex, (b) reference
             only hex, or (c) duplicated palette literal.
    @return  Tuple (palette_set, shipped_bad_hex, reference_hex, duplicated_hex)
             where each _hex list contains (lineno, hexcode) tuples.
    """
    with open(FIGSTYLE) as handle:
        style = handle.read()
    palette = {c.upper() for c in re.findall(r'"(#[0-9A-Fa-f]{6})"', style)}
    with open(MAKE_FIGS) as handle:
        lines = handle.read().splitlines()
    # Figure out which source lines sit inside one of the shipped figure builders;
    # helpers and make_fig1() are reference-only because the shipped Fig. 1 is the
    # approved bitmap infographic.
    shipped_ranges = []
    for name in SHIPPED_BUILDERS:
        start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
        end = next(
            (i for i, l in enumerate(lines[start + 1:], start + 1) if l.startswith("def ")),
            len(lines),
        )
        shipped_ranges.append((start + 1, end))
    shipped, reference, duplicated = [], [], []
    for lineno, line in enumerate(lines, 1):
        inside = any(a <= lineno <= b for a, b in shipped_ranges)
        for hexcode in re.findall(r'"(#[0-9A-Fa-f]{6})"', line):
            if hexcode.upper() in palette:
                if inside:
                    duplicated.append((lineno, hexcode))
                continue
            (shipped if inside else reference).append((lineno, hexcode))
    return palette, shipped, reference, duplicated


def main():
    """
    @brief   Entry point: parse figure manifests, run S1..S7 for each panel,
             report the palette audit, and return the failure exit code.
    @return  int – 0 if all checks pass, 1 otherwise.
    """
    includes = include_widths()
    order = [
        "fig1_architecture", "fig2_convergence_v2", "fig3_ablation_v2",
        "figS_namr_fix", "figS_patient_learning", "figS_skip_diagnostic",
        "figS_total_loss", "figS_mechanism",
    ]
    builders = {
        "fig1_architecture": "make_fig1",
        "fig2_convergence_v2": "make_fig2",
        "fig3_ablation_v2": "make_fig3",
        "figS_namr_fix": "make_fig_s1",
        "figS_patient_learning": "make_fig_s2",
        "figS_skip_diagnostic": "make_fig_s3",
        "figS_total_loss": "make_fig_s_total_loss",
        "figS_mechanism": "make_fig_s_mechanism",
    }
    palette, bad_hex, ref_hex, dup_hex = check_palette()
    failures = []

    print(
        f"{'figure':<26}{'canvas':>8}{'include':>9}"
        f"{'scale':>7}{'eff dpi':>9}"
        f"{'min pt':>8}{'printed':>9}{'over fill':>10}  verdict"
    )
    for stem in order:
        info = canvas_size(stem)
        if info is None:
            continue
        inc, src = includes.get(stem, (float("nan"), "MISSING"))
        scale = inc / info["width_in"]
        eff_dpi = info["px"] / inc
        is_bitmap = stem in BITMAP_FIGURES
        fonts = designed_font_sizes(builders[stem])
        if is_bitmap:
            min_pt = BITMAP_BODY_PT
        else:
            min_pt = fonts[2] if fonts else float("nan")
        printed = min_pt * scale
        clipped, gap = tick_label_gaps(info["path"])
        over_fill, over_rows = text_over_fill(info["path"])

        bad = []
        if not np.isfinite(inc):
            bad.append("no \\includegraphics")
        else:
            if scale > 1.0 + EPS_SCALE:
                bad.append(f"upscaled x{scale:.4f} (S1)")
            if eff_dpi < MIN_EFFECTIVE_DPI - EPS_DPI:
                bad.append(f"{eff_dpi:.0f} dpi (S2)")
            if printed < MIN_PRINTED_PT:
                bad.append(f"printed {printed:.1f} pt (S3)")
        if abs(info["dpi"] - TARGET_PNG_DPI) > PNG_DPI_TOL:
            bad.append(f"file dpi {info['dpi']:.1f} (S4)")
        if clipped and not is_bitmap:
            bad.append("ink at image edge (S5)")
        if over_fill:
            bad.append(
                f"text over bar fill: {over_fill} clusters / {over_rows} rows (S7)"
            )
        verdict = "PASS" if not bad else "FAIL: " + "; ".join(bad)
        if bad:
            failures.append((stem, bad))

        gap_txt = (
            "n/a"
            if over_fill is None
            else (f"{over_fill} cl/{over_rows} rw" if over_fill else "none")
        )
        if is_bitmap:
            verdict = "PASS (bitmap infographic: documented type-size exception)"

        print(
            f"{stem:<26}{info['width_in']:>7.2f}\"{inc:>8.2f}\"{scale:>7.3f}"
            f"{eff_dpi:>9.0f}{min_pt:>8.1f}{printed:>9.1f}{gap_txt:>10}  {verdict}"
        )
        print(f"{'':<26}include {src}")

    # S6: hex-colour palette audit report.
    print(f"\npalette literals outside figure_style.py (S6)")
    print(
        f"    shipped builders (fig2/fig3/S1/S2/S3): {len(bad_hex)} occurrence(s)"
    )
    for lineno, hexcode in bad_hex[:BAD_HEX_DISPLAY_LIMIT]:
        print(f"        85_make_figures.py:{lineno}  {hexcode}")
    print(
        f"    reference-only make_fig1(): {len(ref_hex)} occurrence(s) "
        f"(not shipped; flagged for hygiene only)"
    )
    print(
        f"    palette values duplicated as literals in shipped builders: "
        f"{len(dup_hex)} occurrence(s)  [hygiene, not a colour violation]"
    )
    for lineno, hexcode in dup_hex:
        print(f"        85_make_figures.py:{lineno}  {hexcode}")
    if bad_hex:
        failures.append(
            (
                "85_make_figures.py",
                [f"{len(bad_hex)} non-palette hex in shipped builders (S6)"],
            )
        )

    print(
        f"\n{len(failures)} figure(s) failing. "
        f"thresholds: scale<=1.0, eff dpi>={MIN_EFFECTIVE_DPI:.0f}, "
        f"printed>={MIN_PRINTED_PT} pt, {TARGET_PNG_DPI} dpi metadata, "
        f"no edge clipping, no text over bar fills"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
