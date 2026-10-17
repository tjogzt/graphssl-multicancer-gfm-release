# Reorganization report (2026-09-04)

> **Scope note.** This is a dated historical log of the organization passes
> applied to the code base (2026-09-04 and the 2026-09 pass below); the file
> counts stated in each section are as-of that date. Current file counts, layout
> and entry points are authoritative in `README.md` and `PIPELINE.md`.

Scope: manuscript-adjacent code base of the GFM / cross-cancer graph-pretraining
study. The external working tree is left intact; this repository is the
GitHub-ready release snapshot.

## 1. Code conventions and comments
- Every released script now carries a normalized English header:
  `Author` (manuscript author list), `Date` (2026-08), purpose paragraph,
  `Input`, `Output`, `Arguments`, and `Run` fields; standalone pipeline
  scripts are explicitly annotated as run-directly scripts.
- Verified: 0 Chinese characters anywhere in the release (scripts + shell + R).
- Cross-file references (importlib `spec_from_file_location`, docstring
  mentions) were rewritten to the new names; a loader-target audit reports
  zero missing files.
- Scripts are named `NN_verb_object.py` by pipeline stage; the full old-name →
  new-name table is in `docs/RENAME_MAP.md`.

## 2. Data/configuration conventions
- Removed hard-coded absolute paths (previously OS-specific `/Volumes/...`
  literals in `10_download_string_ppi.py`, `22_build_test_graph_ci.py`, the R downloader).
- Output/data paths now derive from `config.py`, which resolves paths from the
  repository root and remains env-overridable (`GFM_PROJECT_ROOT`, `GFM_DATA_RAW`,
  `GFM_DATA_PROC`, `GFM_MODEL_DIR`, `GFM_RESULT_DIR`, `GFM_SEED`, …).
- Constants that already existed in `config.py` (embedding dim, PPI score
  threshold, seed) are referenced, not re-declared, in rewritten scripts.
- The random seed remains the only in-code stochastic constant, via
  `config.set_seed()`.

## 3. Folder standardization and GitHub readiness
- `<repo root>/` layout:
  - `scripts/`  — 45 Python + 3 shell + 1 R (renamed & consolidated)
  - `figures/`  — 6 publication figures actually referenced by the manuscript
  - `docs/`     — `RENAME_MAP.md`, `REORGANIZATION_REPORT.md` (this file)
  - `tests/`    — 2 unit-test files, adapted to the new names (14 tests, all pass)
  - `README.md`, `PIPELINE.md`, `requirements.txt`, `Dockerfile`, `.gitignore`
- Superseded files removed from the release (not deleted from the working tree):
  `08_p1_patient_pretrain.py` (deprecated), `09_pretrain_fixed.py` /
  `09_pretrain_legacy.py` (shims → single `40_pretrain.py`),
  `fix_patient_edge.py` (one-off), `launch_job.sh`, `_post_batch1.sh`.
- Root-directory hygiene of the working tree: 1,529 AppleDouble files and
  1,462 stray UUID temp directories (~6.4 GB of real TCGA STAR files) moved to
  `data/raw/tcga/_gdc_root_leftovers/`; root-level scratch scripts, duplicate
  figure copies, LaTeX aux files and backups moved to `archive_root_*/`.

## Verification performed on the release
- `py_compile` on all 45 Python scripts: PASS
- `--help` smoke on guarded entrypoints: PASS
- `tests/` unit tests: 14/14 PASS
- Loader-target audit: zero missing modules
- Chinese-character scan: 0; absolute-path scan: 0
- `import config`/`import gfm_losses` resolve inside the release tree


---

## 2026-09 pass — comments, path normalization, merges, submission packaging

### 1. English documentation
Every released file (39 pipeline scripts, 4 libraries, 3 shell drivers, 1 R
script) now carries the standard header (Author / Date / purpose / Input /
Output / Run) and English inline comments on data loading, identifier mapping,
metric computation and file writes. All comments, docstrings and printed strings
are English; a CJK scan over `scripts/*` returns nothing.

### 2. Script merges (redundancy removed)
| New canonical file | Absorbed |
|---|---|
| `30_embed_esm2.py --model 8m\|650m` | `30_embed_esm2_8m.py`, `31_embed_esm2_650m.py` |
| `36_build_graph_variants.py --variant esm650_full\|esm20c\|esm650_20c` | `36_build_graph_esm650_full.py`, `37_build_graph_20c.py`, `38_build_graph_esm650_20c.py` |
| `81_assemble_results.py` | `81_assemble_results.py`, `82_assemble_final_results.py` |
| `85_make_figures.py --only …` | `84_make_figures.py`, `85_make_fig_s3.py`, `86_make_patient_fig.py`, `87_make_fig1.py` |

`82_gen_final_tables.py` keeps its position in the numbering after the merge.

### 3. Hard-coded data removed
* Every figure/table value is now read from a machine artifact; provenance files
  (`fig1_subgraph_layout.json`, `figS3_auc_provenance.json`,
  `patient_fewshot_acc.json`) record the run identity for values that have no
  other machine source.
* `81_assemble_results.py` previously printed transcribed METABRIC AUROCs
  (0.667 / 0.671 / 0.641 / 0.692 / 0.514). Those literals are gone; the report now
  computes mean ± SD over the 30 per-split `.npy` arrays (e.g. METABRIC GFM
  K=50 AUROC 0.6483 ± 0.0072), which also corrected the stale numbers.
* `80_collect_losses.py`, `82_gen_final_tables.py`, `55_eval_supervised_ppi.py`
  and the analysis scripts no longer contain path literals or re-typed dataset
  facts; seeds exist only in `config.py`.
* Path literals reduced to zero across `scripts/*.py`; shell drivers read
  `GFM_PROJECT_ROOT` / `GFM_DATA_PROC` / `GFM_RESULT_DIR` the same way `config.py` does.

### 4. Repository / submission packaging
* Root directory cleaned: AppleDouble files, LaTeX intermediates, duplicate
  figure copies, duplicated `scripts/`+`tests/` trees, the stale
  `manuscript/main.*` pair and the duplicated server tarball were removed or
  consolidated into `archive/` (backups, docs, tmp, legacy_scripts,
  legacy_tests, manuscript_stale, root_duplicates, server_package).
* `manuscript/__main.tex` gained `\graphicspath{{manuscript/}}`, so the single
  canonical figure copy under `manuscript/` is used (the root-level duplicates
  are gone).
* Submission directory = `<repo root>/`: script files, 3 test modules,
  figure files (PDF + PNG), `README.md`, `PIPELINE.md`, `requirements.txt`,
  `Dockerfile`, `.gitignore` and `docs/` (DATA_LAYOUT, FIGURE_REVIEW, RENAME_MAP,
  this report).
* Path references verified in both layouts (external `GFM_PROJECT_ROOT` and
  self-contained checkout).

### 5. Verification executed
| Check | Result |
|---|---|
| `py_compile` over all scripts and tests | pass |
| Hardcoded-path scan (`"results/…"`, `"data/processed/…"`, …) | 0 matches |
| CJK scan over scripts/*.py, *.sh, *.R | 0 matches |
| Literal-seed scan | 0 matches (only `config.py`, env-overridable) |
| `tests/test_gfm_losses.py` | 9/9 pass |
| `tests/test_new_scripts.py` | 5/5 pass |
| `tests/test_figure_data.py` | 29 checks pass |
| `85_make_figures.py` (all six figures) | 6/6 written, 300-dpi PNG + PDF |
| `pdflatex manuscript/__main.tex` / `supplementary.tex` (2 passes each) | 0 errors |
