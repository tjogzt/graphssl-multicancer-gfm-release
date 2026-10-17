# Cross-cancer graph-pretrained clustering: robustness and task-boundary diagnostics in protein language model features

Code and reproducible pipeline accompanying the manuscript submitted to
***Cell Reports Methods*** (Methods Track, IF ~14.3).

> 🌐 **[中文 README · Chinese version](./README_zh-CN.md)** | 📖 This is the English document  
> ☝️ 点击上方「中文 README」切换到中文版 / Click the link for the Chinese translation

This repository ships five top-level entry-points for reviewers:

1. **R-language wrapper** `R/RunMultiCancerSSL.R` — five high-level verbs (`build_graph / pretrain / evaluate_ppi / evaluate_patient / make_figures`) wrap the Python pipeline for R-language reviewers.
2. **Makefile** at repository root — five canonical targets mirror the five verbs: `data / preprocess / train / evaluate / figures`.
3. **10-command shell recipe** (below, `§10-command recipe`) — regenerates every figure and table of the manuscript end-to-end from raw public downloads.
4. **Key Resources Table artifact** — STAR Methods first subsection of the manuscript PDF, 30 rows × 9 categories (Chemicals / Assays / 6 Deposited datasets / Cell lines / Oligos / 7 Software packages / 2 Other categories).
5. **Checkpoint provenance registry** — `results/checkpoint_registry.json` (96 rows × 12 fields: checkpoint id → file path → checkpoint SHA256 12-char prefix → training run-ID → seed → decoder architecture).

We pretrain a heterogeneous graph transformer (HGT) on multi-cancer molecular
networks built from TCGA (UCSC Xena), STRING v12 PPI and ESM-2 protein language
model embeddings, and compare three self-supervised tasks — NAMR (denoising
reconstruction), CSP (PPI link prediction), C3L (cross-cancer contrastive
learning) — alone and in combination, across five equal-budget seeds
42/7/123/21/99, and across two hardware platforms (NVIDIA CUDA and Apple MPS).

## Repository layout

```
<repo root>                      ← **GitHub submission root (git init here)**
├── Makefile                      5 targets: data / preprocess / train /
│                                  evaluate / figures
├── Dockerfile                    CUDA container (PyTorch + torch-geometric)
├── README.md                     this file (English; GitHub default view)
├── README_zh-CN.md               Chinese version of this README
├── PIPELINE.md                   end-to-end execution guide (stage by stage)
├── requirements.txt              pinned Python environment
├── project.md                    project log (Chinese-only file allowed; see §Language)
├── CITATION.cff                  "Cite this repository" metadata (7 authors)
├── CODE_OF_CONDUCT.md            Contributor Covenant v2.1
├── LICENSE                       MIT
├── R/RunMultiCancerSSL.R         R wrapper: 5 verbs (build_graph / pretrain /
│                                  evaluate_ppi / evaluate_patient / make_figures)
├── docs/                         DATA_LAYOUT / FIGURE_REVIEW / RENAME_MAP /
│                                  RENAME_MAP.v2 / REORGANIZATION_REPORT
├── figures/                      publication figures (300-dpi PNG + vector PDF):
│                                  Fig.1–3, Fig.S1–S5, graphical_abstract_v1
├── scripts/                      81 pipeline files (76 Python + 1 R + 4 shell drivers)
│   ├── config.py                 all paths + protocol constants (GFM_* overrides)
│   ├── gfm_losses.py             losses / negative samplers / C3L InfoNCE
│   ├── figure_data.py            figure-data loader (reads real artifacts)
│   ├── figure_style.py           palette + rcParams (PDF+Cairo, 300dpi PNG)
│   ├── 10–11_*                   data acquisition (STRING v12, TCGA Xena)
│   ├── 20–39_*                   heterograph construction (ESM / TCGA / patient)
│   ├── 30_embed_esm2             ESM-2 8m/650m embeddings
│   ├── 40–41_*                   HGT pretraining (CSP+NAMR+C3L 3-task joint)
│   ├── 50–56_*                   1° PPI AUC + cluster + pathway evaluation
│   ├── 60–66_*                   external validation cohorts + drug response
│   ├── 70–79_*                   functional enrichment, probes, hold-out graphs
│   ├── 80–82_*                   loss aggregation + result assembly + tables
│   ├── 84–87_*                   Fig.1 layout cache + figure generation
│   ├── 88–90_*                   manuscript / number / reference audit suite
│   ├── 93–95_*                   geometry, tissue-PPI and path-normalisation audits
│   ├── 100–103_*                 cohort, cross-seed and reproduction-status audits
│   ├── 120–137_*                 reviewer-response diagnostics and baselines
│   └── run_*.sh                  4 shell drivers (ablation / holdout / NAMR sweep)
├── tests/                        regression tests (3 modules, 43 checks)
└── .gitignore                    data/ results/ models/ cloud_results/ excluded
                                  (hosted on Zenodo/SWH/OSF separately)
```

*Note on large binaries:* `data/`, `results/`, `models/` and `cloud_results/` are intentionally gitignored; download/migrate them from the Zenodo/SWH/OSF deposits referenced in the manuscript **Data and Code Availability** section.

## Script naming

Every executable follows `NN_verb_object.py`; `NN` orders the file inside its
stage. Shared libraries are imported by name and therefore carry no number
(Python identifiers cannot start with a digit):

| Stage | Prefix | Files |
|---|---|---|
| Download | `10`–`11` | `10_download_string_ppi.py`, `11_download_tcga_xena.R` |
| Graphs / features | `20`–`39` | `30_embed_esm2.py` (`--model 8m\|650m`), `36_build_graph_variants.py` (`--variant esm650_full\|esm20c\|esm650_20c`) |
| Pretraining | `40`–`41` | `40_pretrain.py`, `41_pretrain_single_cancer.py` |
| Evaluation | `50`–`56` | `50_eval_checkpoint.py` … `56_center_check_diag.py` |
| Downstream / external validation | `60`–`66` | `60_metabric_validation.py`, `63_paad_transfer_eval.py`, `64_drug_response_gdsc.py` |
| Functional analysis / probes | `70`–`76` | `70_go_enrichment.py`, `74_c3l_permutation_test.py` |
| Hold-out & control graphs | `77`–`79` | `77_build_holdout_graph.py`, `79_build_control_graphs.py` |
| Result assembly / tables | `80`–`82` | `80_collect_losses.py`, `81_assemble_results.py`, `82_gen_final_tables.py` |
| Figures & figure-spec checks | `84`–`87` | `84_extract_fig1_layout.py`, `85_make_figures.py`, `86_check_figure_specs.py` |
| Manuscript / number / reference audit | `88`–`90` | `88_audit_manuscript.py`, `89_audit_numbers.py`, `90_audit_references.py` |
| Reviewer-response diagnostics | `93`–`95`, `100`–`103`, `120`–`137` | `125_namr_denoiser_diagnostic.py`, `129_graph_smoothing_baseline.py`, `134_c3l_elimination_ablation.py` |
| Benchmark | `122` | `122_bench_device.py` |

Imported libraries: `config.py` (all paths and protocol constants, environment
overridable), `gfm_losses.py` (losses and negative samplers),
`figure_data.py` (reads every plotted value from an artifact),
`figure_style.py` (palette, rcParams, 300-dpi PNG + vector PDF export).

## How to run

All scripts resolve paths through `config.py`, so the repository can run in two
layouts:

**A. Data outside the repository (recommended for a laptop / cluster checkout)**

```bash
# directory holding data/, results/, models/, cloud_results/
export GFM_PROJECT_ROOT=/path/to/project
# example: evaluate the full-joint pretrained checkpoint
python scripts/50_eval_checkpoint.py \
  --ckpt results/ablation_real/full.pt --tag full
```

**B. Self-contained checkout** — place `data/`, `results/`, `models/` and
`cloud_results/` directly under the repository root; no environment variable is
needed because `config.PROJECT_ROOT` then equals the repository root.

Minimal end-to-end walkthrough (details in `PIPELINE.md`).

## 10-command reproduction recipe (Cell Reports Methods Methods Track)

Run these 10 commands on a clean machine to regenerate every figure and table
in the manuscript from raw public downloads. Prerequisites: Python 3.11,
Python venv, R 4.3+, GNU Make.

```bash
# (1) clone the repository and enter the flat submission root
git clone <repo-url> gfm-crm && cd gfm-crm

# (2) create + activate the pinned Python virtual environment
python3 -m venv .venv && source .venv/bin/activate

# (3) upgrade pip and install the pinned PyTorch / PyG / ESM2 stack
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# (4) verify the R wrapper (5 verbs; needs R 4.3+, base-R only)
Rscript R/RunMultiCancerSSL.R

# (5) TARGET 1 — download every raw public input (STRING v12, TCGA Xena, ...)
make data

# (6) TARGET 2 — ESM-2 embeddings + all graph variants
make preprocess

# (7) TARGET 3 — equal-budget pretraining (5 seeds × 500 epochs, 5 arms)
make train

# (8) TARGET 4 — downstream evaluation (PPI / cluster / pathway / METABRIC / PAAD)
make evaluate

# (9) TARGET 5 — regenerate every manuscript figure (PDF + 300-dpi PNG)
make figures

# (10) optional (non-required for results): run the 43 regression tests
python tests/test_gfm_losses.py && python tests/test_new_scripts.py && python tests/test_figure_data.py
```

Environment variables available to path overrides: `GFM_PROJECT_ROOT`,
`GFM_DATA_RAW`, `GFM_DATA_PROC`, `GFM_MODEL_DIR`, `GFM_RESULT_DIR`,
`GFM_CLOUD_DIR`, `GFM_FIG_DIR`, `GFM_STRING_DIR`, `GFM_SEED`, `GFM_PYTHON`
(see `scripts/config.py`).

## Data and code availability

All preprocessing, pretraining, evaluation and figure code is in this
repository. Trained checkpoints, derived embeddings and the raw public
downloads are archived separately (Data availability statement of the
manuscript); without them the pipeline rebuilds the data from the public
sources but does not reproduce the paper's exact numbers.

## Figures

```bash
# cache the Fig.1(a) network layout precomputed from the real heterograph
python scripts/84_extract_fig1_layout.py
# regenerate fig1 / fig2 / fig3 / S1–S5 + GraphicalAbstract from real artifacts
python scripts/85_make_figures.py
# pick a subset:
python scripts/85_make_figures.py --only fig1 fig3
```

Every figure is written as 300-dpi PNG plus vector PDF with live text into
`GFM_FIG_DIR` (default `<PROJECT_ROOT>/manuscript`). No plotted number is typed
into the plotting code: values are read from `results/ablation_real/eval_summary.txt`,
checkpoint loss histories and the provenance JSON files listed in
`docs/FIGURE_REVIEW.md`. `tests/test_figure_data.py` asserts that the plotted
values still match those artifacts after any re-run or audit.

## Tests

```bash
# 9 loss / sampler unit tests
python tests/test_gfm_losses.py
# 5 statistics / new-script helper tests
python tests/test_new_scripts.py
# 29 checks: every plotted value matches the referenced artifact file
python tests/test_figure_data.py
```

## Language

- **Code, comments, printed output and figures:** English only. A CJK scan over
  `scripts/`, `tests/` and `R/` returns nothing.
- **`project.md`:** Chinese — the project's single Chinese-language
  source-of-truth log (Rule 13).
- **This `README.md`, `PIPELINE.md` and `docs/`:** English.
- **`README_zh-CN.md`:** Chinese — the counterpart of this file. Together with
  `project.md` it is the only CJK-bearing file in the release.

## Citation

(manuscript DOI / citation placeholder — fill at submission)
