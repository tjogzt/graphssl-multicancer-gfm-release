#!/usr/bin/env python3
"""
@file    87_build_fig1_composite.py
@brief   Rebuild panel (c) of Fig. 1 with typography that matches the author-
         approved bitmap panels (a) and (b). The original panel (c) was rendered
         in DejaVu Sans at cap height 29 px while (a)/(b) headings are a condensed
         medium-weight grotesque at cap height 41.5 px; this script re-renders
         panel (c) in Arial Narrow with a calibrated stroke-width boost so heading
         cap height, median stroke, per-glyph width, and body x-height all match
         panels (a)/(b) to within 5%. Card geometry, fills, border colour, corner
         radius, the red failure accent and every word of text are carried over
         verbatim from the approved panel (c) copy.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  archive/fig1_architecture_bitmap_prev.png  (clean base with no panel (c) residue)
Output: manuscript/fig1_architecture.png + .pdf    (2400 x 1187 px @ 300 dpi)
        reviews/figure_review_2026/figures_2026/fig1_panel_c_typography.png
Run:    GFM_PROJECT_ROOT=<repo> python scripts/87_build_fig1_composite.py
"""
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

PROJECT_ROOT_ENV_VAR = "GFM_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get(PROJECT_ROOT_ENV_VAR, DEFAULT_PROJECT_ROOT)

# Source / output artifact paths.
BASE = os.path.join(ROOT, "archive", "fig1_architecture_bitmap_prev.png")
OUT_PNG = os.path.join(ROOT, "manuscript", "fig1_architecture.png")
OUT_PDF = os.path.join(ROOT, "manuscript", "fig1_architecture.pdf")
OUT_PREVIEW = os.path.join(
    ROOT, "reviews", "figure_review_2026", "figures_2026", "fig1_panel_c_typography.png"
)

# --- Panel (c) geometry measured from the shipped composite ---
PATCH = (1712, 0, 2400, 1187)       # Panel (c) extents inside the 2400x1187 canvas.
HEAD_X, HEAD_BASELINE = 1767, 68    # Heading position; baseline matches panels (a)/(b).
BOX_X0, BOX_X1 = 1747, 2362         # Card border centrelines.
BOX_TOP, BOX_BOTTOM_LIMIT = 125, 1105
BOX_GAP = 36                        # Vertical gap between consecutive cards.
CORNER_R = 13                       # Rounded-rectangle corner radius (px).
BORDER_W = 2.6                      # Border line width (2.6 px approx 3 px dark core).
PAD_L, PAD_R = 34, 20               # Horizontal text insets inside a card.
MIN_PAD_Y = 20                      # Minimum vertical inset top/bottom inside card.

# --- Typography calibrated against panels (a)/(b) ---
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Narrow.ttf"
FONT_INDEX = 0
HEAD_SIZE = 58                       # Cap height ~42 px, matching panels (a)/(b).
CARD_SIZE_MAX = 52                   # Upper bound for the card-size solver.
CARD_SIZE_TARGET = 51                # x-height 26 px target, matching panels (a)/(b).
LINE_PITCH = 1.20                    # Baseline-to-baseline spacing as multiple of size.
ELEM_GAP = 0.62                      # Gap between title and value, size multiple.

MARKER = "\u2717"                     # Ballot X; not present in Avenir Next Condensed.
MARKER_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
MARKER_FONT_FALLBACK = "/System/Library/Fonts/Supplemental/Arial.ttf"

# Ink / accent palette (hex, matches base artwork).
INK = "#111111"
FAIL = "#B3261E"
BORDER = "#8E8E8E"
FILLS = ["#F4F4F4", "#F5CCAD", "#F5CCAD"]  # Card fills: neutral / warning / warning.

# Panel (c) cards. Every word is carried over verbatim from the approved copy.
CARDS = [
    [
        ("Known-edge recovery (cosine probe)", INK),
        ("AUC 0.858 (NAMR-only)", INK),
    ],
    [
        ("Trivial graph smoothing (no training)", INK),
        ("AUC 0.985", INK),
        ("exceeds all trained arms", INK),
    ],
    [
        ("Drug response prediction (GDSC)", INK),
        (MARKER + " R\u00b2 < 0 at all K", FAIL),
        ("(protocol-scoped boundary)", FAIL),
    ],
]

SS = 6                               # Supersampling factor for sub-pixel smooth edges.
HEAD_STROKE = 2                      # Heading stroke width at SS resolution.
CARD_STROKE = 3                      # Card text stroke width at SS resolution.

# Output PNG DPI metadata target.
OUTPUT_DPI = (300, 300)
PDF_RESOLUTION_DPI = 300.0

# Expected base image pixel dimensions.
EXPECTED_BASE_SIZE = (2400, 1187)

# Metric-print precision for card-size comparison message.
BODY_SIZE_REFERENCE_PX = 26.5
CARD_BODY_COMPARE_PRECISION = 1


def load_font(size):
    """
    @brief   Load the Arial Narrow TrueType face at the given pixel size, scaled
             to the current supersampling (SS) grid.
    @param   size  int – nominal 1x point size (pixels).
    @return  PIL ImageFont instance.
    """
    return ImageFont.truetype(FONT_PATH, int(round(size * SS)), index=FONT_INDEX)


def wrap(text, font, max_w):
    """
    @brief   Balanced (optimal) text wrap against real rendered glyph widths.
             Greedy wrap produces ugly breaks before parentheses; this function
             prefers a parenthesis break when both halves fit, and otherwise
             minimises the sum of squared line slack.
    @param   text    str – text to wrap.
    @param   font    ImageFont – font used to measure real glyph advances.
    @param   max_w   float – maximum line width in 1x pixels.
    @return  List[str] of wrapped lines.
    """
    text = text.replace(MARKER + " ", "").strip()
    if font.getlength(text) / SS <= max_w:
        return [text]
    # Parenthetical qualifier reads best on its own line: take that break first
    # when both sub-strings fit within the width.
    if " (" in text:
        head, tail = text.split(" (", 1)
        tail = "(" + tail
        if (font.getlength(head) / SS <= max_w
                and font.getlength(tail) / SS <= max_w):
            return [head, tail]
    words = text.split()
    n = len(words)
    width = lambda i, j: font.getlength(" ".join(words[i:j])) / SS
    best = [0.0] + [float("inf")] * n
    prev = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            w = width(i, j)
            if w > max_w:
                continue
            cost = best[i] + (max_w - w) ** 2
            if cost < best[j]:
                best[j], prev[j] = cost, i
    if best[n] == float("inf"):
        # Unbreakable very-long word: return the original string as a single line.
        return [text]
    lines, j = [], n
    while j:
        lines.append(" ".join(words[prev[j]:j]))
        j = prev[j]
    return lines[::-1]


def marker_font(size):
    """
    @brief   Return a TrueType face that actually contains the ballot-X glyph
             (U+2717); Avenir Next Condensed does not, so fall back to Arial.
    @param   size  int – nominal 1x pixel size.
    @return  PIL ImageFont instance.
    """
    for cand in (MARKER_FONT, MARKER_FONT_FALLBACK):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, int(round(size * SS)))
    raise SystemExit("No fallback TrueType face available for the ballot X glyph.")


def marker_adv(size):
    """
    @brief   Horizontal advance (px, 1x) reserved for the ballot-X marker before
             the rest of the card line.
    @param   size  int – nominal 1x card pixel size.
    @return  float – advance in 1x pixels.
    """
    return 1.60 * cap_px(size)


def cap_px(size):
    """
    @brief   Cap height for Arial Narrow (measured: 42 px at size 58 => 0.724).
    @param   size  int – nominal 1x pixel size.
    @return  float – cap height in 1x pixels.
    """
    return 0.724 * size


def layout(card_size):
    """
    @brief   Wrap every card element and compute the vertical box height required
             by each card at the given text size.
    @param   card_size  int – nominal 1x card text size in pixels.
    @return  Tuple (blocks, heights) where:
             - blocks: per-card list of per-element (wrapped_lines, raw_text, colour)
             - heights: per-card total vertical box extent in 1x px.
    """
    font = load_font(card_size)
    max_w = (BOX_X1 - PAD_R) - (BOX_X0 + PAD_L)
    pitch = LINE_PITCH * card_size * SS
    gap = ELEM_GAP * card_size * SS
    blocks, heights = [], []
    for card in CARDS:
        elems = []
        for t, c in card:
            lm = max_w - (marker_adv(card_size) if t.startswith(MARKER) else 0.0)
            elems.append(
                (
                    wrap(t, font, max_w if not t.startswith(MARKER) else lm),
                    t, c,
                )
            )
        content = (
            sum(len(ls) * pitch for ls, _, _ in elems) + gap * (len(elems) - 1)
        )
        heights.append(
            (content - pitch) / SS
            + cap_px(card_size)
            + 0.30 * card_size
            + 2 * MIN_PAD_Y
        )
        blocks.append(elems)
    return blocks, heights


def solve_size():
    """
    @brief   Find the largest card text size such that all three cards (plus
             gaps) fit inside the vertical column available for panel (c).
    @return  Tuple (card_size, blocks, heights).
    """
    available = BOX_BOTTOM_LIMIT - BOX_TOP - 2 * BOX_GAP
    for size in range(CARD_SIZE_MAX, 20, -1):
        blocks, heights = layout(size)
        if (sum(heights) <= available
                and all(len(e) == len(c) for e, c in zip(blocks, CARDS))):
            return size, blocks, heights
    raise SystemExit("Cannot fit panel (c) content inside the allocated column.")


def draw_panel(card_size, blocks, heights):
    """
    @brief   Render panel (c) at the supersampled resolution then downsample to
             1x with LANCZOS for anti-aliased edges.
    @param   card_size  int – 1x card pixel size.
    @param   blocks     – layout() return value.
    @param   heights    – layout() return value.
    @return  PIL Image at 1x resolution.
    """
    scale = SS
    W_patch = PATCH[2] - PATCH[0]
    canvas = Image.new("RGB", (W_patch * scale, PATCH[3] * scale), "white")
    d = ImageDraw.Draw(canvas)

    def X(x):
        return (x - PATCH[0]) * scale

    def Y(y):
        return y * scale

    # Panel (c) heading.
    d.text(
        (X(HEAD_X), Y(HEAD_BASELINE)),
        "(c) Downstream evaluation",
        font=load_font(HEAD_SIZE), fill=INK, anchor="ls",
        stroke_width=HEAD_STROKE, stroke_fill=INK,
    )

    font = load_font(card_size)
    pitch = LINE_PITCH * card_size * scale
    gap = ELEM_GAP * card_size * scale

    y = BOX_TOP
    for elems, fill, box_h in zip(blocks, FILLS, heights):
        # Card rounded rectangle with fine border.
        d.rounded_rectangle(
            [X(BOX_X0), Y(y), X(BOX_X1), Y(y + box_h)],
            radius=CORNER_R * scale, fill=fill, outline=BORDER,
            width=int(round(BORDER_W * scale)),
        )
        cursor = Y(y + MIN_PAD_Y + cap_px(card_size))
        for ls, _text, colour in elems:
            has_marker = _text.startswith(MARKER)
            for line in ls:
                x_text = X(BOX_X0 + PAD_L)
                if has_marker:
                    # Glyph height scaled to match cap height of surrounding text.
                    mf = marker_font(card_size * 0.954)
                    d.text((x_text, cursor), MARKER, font=mf, fill=colour, anchor="ls")
                    x_text += marker_adv(card_size) * scale
                d.text(
                    (x_text, cursor), line, font=font, fill=colour, anchor="ls",
                    stroke_width=CARD_STROKE, stroke_fill=colour,
                )
                cursor += pitch
            cursor += gap
        y += box_h + BOX_GAP
    return canvas.resize((W_patch, PATCH[3]), Image.LANCZOS)


def runstats(mask, x0, x1, y0, y1):
    """
    @brief   Median and mean horizontal dark-run length within the given band
             (used as a pixel-level stroke-weight proxy).
    @param   mask  np.ndarray – binary dark-ink mask.
    @param   x0,x1,y0,y1  int – band extents (inclusive of x0,y0, exclusive of x1,y1).
    @return  Tuple (median_run, mean_run) both as Python floats.
    """
    runs = []
    for row in mask[y0:y1, x0:x1]:
        c = 0
        for v in row:
            if v:
                c += 1
            elif c:
                runs.append(c)
                c = 0
        if c:
            runs.append(c)
    r = np.array(runs)
    r = r[r > 1]
    return float(np.median(r)), float(r.mean())


def glyph_metrics(mask, x0, x1, y0, y1):
    """
    @brief   Compute x-height mode, cap-height max, parenthesis-height max, and
             glyph count for a heading-type band.
    @param   mask  np.ndarray – binary dark-ink mask.
    @param   x0,x1,y0,y1  int – band extents.
    @return  Tuple (x_height_mode, cap_height, paren_height, n_glyphs).
    """
    sub = mask[y0:y1, x0:x1]
    cols = sub.sum(axis=0)
    segs, st, gp = [], None, 0
    for i, v in enumerate(cols):
        if v > 0:
            if st is None:
                st = i
            gp = 0
        else:
            if st is not None:
                gp += 1
                if gp >= 1:
                    segs.append((st, i - gp))
                    st = None
                    gp = 0
    if st is not None:
        segs.append((st, len(cols) - 1))
    hs = []
    for a, b in segs:
        r = sub[:, a:b + 1].sum(axis=1)
        nz = np.nonzero(r)[0]
        hs.append(int(nz[-1] - nz[0] + 1))
    c = Counter(hs)
    xh = c.most_common(1)[0][0]
    cap = max([h for h in hs if 40 <= h <= 46] or [0])
    par = max([h for h in hs if h > 46] or [0])
    return xh, cap, par, len(segs)


def verify(path):
    """
    @brief   Print typography comparison table plus pixel-level defect scan so
             the operator can confirm (a)/(b) vs (c) calibration visually.
    @param   path  str – PNG filesystem path.
    @return  None
    """
    from collections import Counter  # noqa: F401  (re-used by glyph_metrics)
    im = Image.open(path).convert("RGB")
    g = np.array(im.convert("L"))
    mask = g < LUMINANCE_THRESHOLD

    base = np.array(Image.open(BASE).convert("RGB")).astype(int)

    print("\n--- heading metrics (target: panels (a)/(b)) ---")
    print(
        f"{'panel':<8}{'x-height':>9}{'cap':>5}"
        f"{'paren':>7}{'glyphs':>8}{'run med':>9}{'run mean':>10}"
    )
    for name, (x0, x1) in [
        ("(a)", (2, 934)),
        ("(b)", (1004, 1706)),
        ("(c)", (1751, 2359)),
    ]:
        xh, cap, par, n = glyph_metrics(mask, x0, x1, 20, 90)
        md, mean = runstats(mask, x0, x1, 20, 90)
        print(f"{name:<8}{xh:>9}{cap:>5}{par:>7}{n:>8}{md:>9.1f}{mean:>10.2f}")

    print("--- card text stroke (panels A/B body vs panel (c) cards) ---")
    for name, box in [
        ("A body", (2, 934, 998, 1042)),
        ("B body", (1004, 1706, 138, 185)),
        ("C cards", (1751, 2359, 140, 190)),
    ]:
        md, mean = runstats(mask, *box)
        print(f"    {name:<9} run med {md:.1f}  run mean {mean:.2f}")

    diff = np.abs(base - np.array(im).astype(int)).sum(axis=2)
    print("--- defect scan ---")
    print(
        f"    changed px outside the patch (x < {PATCH[0]}): "
        f"{int((diff[:, :PATCH[0]] > 0).sum())}"
    )
    print(
        f"    residual dark px in the old panel-c gutter: "
        f"{int(mask[:, PATCH[0]:BOX_X0 - 1].sum())}"
    )
    # Text-only overflow probe: mid-box rows, excluding the border columns.
    band = mask[300:340, BOX_X0:BOX_X1 - 4]
    ink = np.nonzero(band.sum(axis=0))[0]
    right = BOX_X0 + ink.max() if len(ink) else BOX_X0
    print(
        f"    rightmost card ink x: {right} (text limit "
        f"{BOX_X1 - PAD_R}; border at {BOX_X1})"
    )


LUMINANCE_THRESHOLD = 160


def main():
    """
    @brief   Entry point: solve card size, print layout summary, paste the panel
             (c) rendering onto the clean base bitmap, save outputs and run the
             verify() typography comparison.
    @return  None
    """
    card_size, blocks, heights = solve_size()
    print(
        f"card text size {card_size} px (x-height "
        f"{BODY_SIZE_REFERENCE_PX * card_size / CARD_SIZE_TARGET:.{CARD_BODY_COMPARE_PRECISION}f} px "
        f"vs (a)/(b) body {BODY_SIZE_REFERENCE_PX} px)"
    )
    print(
        "wrapped lines per card: "
        + ", ".join(str(sum(len(l) for l, _, _ in e)) for e in blocks)
    )
    print(
        "box heights: "
        + ", ".join(f"{h:.0f}" for h in heights)
        + f" (total {sum(heights):.0f} + gaps {2 * BOX_GAP} <= "
          f"{BOX_BOTTOM_LIMIT - BOX_TOP})"
    )

    base = Image.open(BASE).convert("RGB")
    if base.size != EXPECTED_BASE_SIZE:
        raise SystemExit(
            f"Unexpected base image size {base.size}; expected {EXPECTED_BASE_SIZE}."
        )
    panel = draw_panel(card_size, blocks, heights)
    blank_patch_size = (PATCH[2] - PATCH[0], PATCH[3] - PATCH[1])
    base.paste(Image.new("RGB", blank_patch_size, "white"), PATCH[:2])
    base.paste(panel, PATCH[:2])

    base.save(OUT_PNG, dpi=OUTPUT_DPI)
    base.save(OUT_PDF, "PDF", resolution=PDF_RESOLUTION_DPI)
    base.crop((PATCH[0], 0, PATCH[2], PATCH[3])).save(OUT_PREVIEW, dpi=OUTPUT_DPI)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PREVIEW}")
    verify(OUT_PNG)


if __name__ == "__main__":
    main()
