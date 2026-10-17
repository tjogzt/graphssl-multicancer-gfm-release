# FIGURE_REVIEW.md — figure provenance and review decisions

All manuscript figures are produced by `85_make_figures.py` (300-dpi PNG + vector
PDF with live text, Okabe-Ito colourblind-safe palette, shared rcParams from
`figure_style.py`). No plotted value is typed into the plotting code.

## 1. Figure-by-figure data provenance

| Figure | Builder | Values come from | Notes |
|---|---|---|---|
| Fig. 1 architecture overview | `make_fig1` | `results/fig1_subgraph_layout.json` (real TP53-neighbourhood sample of the 13,881-gene graph, 770,440 PPI edges, spring layout seed 42) + `eval_summary.txt` + `patient_fewshot_acc.json` for the outcome badges | layout cached by `84_extract_fig1_layout.py` |
| Fig. 2 convergence | `make_fig2` | `loss_history` inside `results/ablation_real/full.pt` (500 epochs, seed 42) | dashed stall line 1.32 documents the separate 100-epoch NAMR-only sweep (Fig. S1) |
| Fig. 3 ablation | `make_fig3` | `results/ablation_real/eval_summary.txt` (7 arms) | cluster accuracy mean ± SD over 30 repeats; PPI AUC from the seed-42 evaluation |
| Fig. S1 standardization | `make_fig_s1` | `loss_history` of `results/namr_sweep/base_ref.pt` (unstandardized) and `h1_std.pt` (standardized), 100 epochs, seed 42 | NAMR-only run, distinct from Fig. 2's 500-epoch full model |
| Fig. S2 patient learning curve | `make_fig_s2` | `results/patient_fewshot_acc.json` (provenance: Supplementary Table S7) | log2 K axis; K = 1, 2, 5, 10, 20, 30 |
| Fig. S3 skip-decoder diagnostic | `make_fig_s3` | trajectories from `namr_only.pt` (plain decoder, seed 42) and `namr_only_s7.pt` (skip-connected decoder, seed 7); bars from `results/figS3_auc_provenance.json` | AUC axis starts at chance 0.5 |

## 2. Review decisions (four-way review, R1 → R2 debate → lead adjudication)

1. **No hard-coded values in figure code** (blocking finding). Fig. 3 previously
   carried pre-audit literals (random-init cluster accuracy 0.5755 ± 0.0186 and
   AUC 0.6620, plus 3rd/4th-decimal drift on four other arms). All arms now read
   `eval_summary.txt`; the post-audit values are 0.4882 ± 0.0220 / 0.6625.
2. **Caption honesty.** The Fig. 3 caption previously claimed AUC error bars
   ("bootstrap over 2,000 pairs") that the code never drew; it now states that the
   AUC panel is a single seed-42 evaluation and points to the multi-seed tables.
3. **Error-bar policy.** Bar panels: mean ± SD over 30 repeats (clustering).
   AUC panel: single evaluation, no error bars. Trajectory panels: single run
   (seed 42, or seed 7 for the skip diagnostic) — stated in each caption.
4. **Axis honesty.** AUC panels start at chance (0.5) with a dotted chance line
   and an explicit `(0.5–1.0)` axis label; the patient learning curve uses a log2
   K axis because K is a count with multiplicative steps.
5. **Numerical consistency.** Every citation of the random-init HGT PPI AUC in the
   manuscript (tables, captions, prose) was aligned to the post-audit 0.6625;
   Supplementary Table S1 rows were aligned to `eval_summary.txt` at 4 decimals.
6. **Reproducibility gap closed.** The patient learning-curve generator was
   migrated out of the working tree into the released figure pipeline, and its
   values now live in a provenance JSON instead of a script literal.
7. **Fig. 1 redesign (R2 consensus: horizontal three-band pipeline).** The first
   rebuild attempt was rejected by the author as hard to read (text clipping,
   flat hierarchy, no main flow). The converged design is: (a) inputs + a real
   sampled network with an in-panel type legend (colour *and* shape, counts
   inline), (b) a grouped box holding the 6-layer HGT trunk and the three
   self-supervised task bars, (c) four downstream tasks with honest outcome
   badges, thick inter-band arrows for the main flow, rotated gap labels, and a
   `fit_text` guard that shrinks (never truncates) labels.
8. **Caption/content alignment.** The Fig. 1 caption now lists the fourth
   downstream task (patient cancer classification) and states that drug-response
   prediction is a negative (boundary) result, matching the Results text.

## 3. Outstanding human check

* Fig. 1 aesthetics were verified against the previous version by the author
  before submission (the reviewing agents are text-only and cannot judge
  appearance); the underlying panel composition and every label string are
  reproducible from the script.

## 4. Fig. 1 final (author-approved Nano Banana Pro bitmap)

The author generated the final Fig. 1 with Nano Banana Pro
(`Downloads/generated_fig_1.png`, 1456 x 720) and approved it over any
matplotlib rebuild. The bitmap was upscaled to 300 dpi (2400 x 1187) and placed as
`figures/fig1_architecture.png` + a raster `fig1_architecture.pdf` for the LaTeX
`\includegraphics`. Four-way vision cross-check (three VLM eyes) confirmed the nine
ground-truth numbers verbatim: 13,881 / 21 / 6,273 / 770,440 / 0.8578 /
0.7049 -> 0.8799 / R2<0 / 0.390. `85_make_figures.py` no longer regenerates fig1
(`make_fig1` kept only as reference) so a pipeline re-run cannot overwrite the
approved figure.

## 5. Fig. 1 v2 replica (superseded)


The author rejected the first matplotlib rebuild, generated a reference design with
Nano Banana Pro (`Downloads/generated_fig_1.png`, 1456x720), and asked for a faithful
vector replica. The reference was read with the four-way vision layer; its exact
colours were then extracted from pixels (PIL) rather than guessed:
  canvas white + thin grey frame #8A8C8A; neutral boxes #F4F4F4/#8E8E8E;
  gene node #0470AA (circle), cancer node #D35E00 (square), patient #009E73 (triangle);
  shared-encoder band #DCE9F5; drug-response card #F5CCAD with red #B3261E cross.
The replica (`85_make_figures.py make_fig1`) keeps: three equal-height aligned panels
with bold titles on one baseline, single-grey subgraph edges, a vertical "shared
encoder" band, two-line result cards (task name / outcome) to guarantee no overlap,
and all nine ground-truth numbers read from artifacts. QA via modlens confirmed:
panels aligned, no overlap/clipping, numbers verbatim (0.8578 / 0.7049->0.8799 /
R2<0 / 0.390 / 13,881 / 21 / 6,273 / 770,440). Unlike the PNG reference, the replica
is vector PDF with live, searchable text and is regenerated from the script.
