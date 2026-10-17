# RENAME_MAP.md — legacy names → released names

Two normalization passes were applied to the working tree: **2026-08** (initial
release packaging) and **2026-09** (comment/path normalization + script merges).
This file is the authoritative mapping; it also records which legacy files were
merged into a single canonical entry point and which one-offs were dropped.

## 1. Current released names (2026-09, canonical)

Executables follow `NN_verb_object.py`. Shared libraries are imported by name and
therefore have no numeric prefix (Python identifiers cannot start with a digit).

| Stage | Released file | Replaces (legacy working-tree names) |
|---|---|---|
| Download | `10_download_string_ppi.py` | `01_download_string.py` |
| Download | `11_download_tcga_xena.R` | `01_download_tcga.R` |
| Graph | `20_build_graph_esm.py` | `02_build_graph_esm.py`, `02_build_graph.py` |
| Graph | `21_build_graph_xena_large.py` | `02_build_graph_v3_large.py` |
| Graph | `22_build_test_graph_ci.py` | `02_build_test_graph.py` |
| Features | `30_embed_esm2.py` | **merged** `30_embed_esm2_8m.py` (`07_esm_embeddings.py`) + `31_embed_esm2_650m.py` (`13_esm_650m.py`) → `--model 8m\|650m` |
| Targets | `32_precompute_c3l_targets.py` | `precompute_gene_cancer_specificity.py` |
| Patient | `33_build_patient_gene_edges.py` | `build_patient_gene_edges.py` |
| Patient | `34_build_patient_esm_graph.py` | `21_build_patient_esm_graph.py` |
| Patient | `35_build_patient_paad_graph.py` | `22_build_patient_paad.py` |
| Graph | `36_build_graph_variants.py` | **merged** `36_build_graph_esm650_full.py` (`25_build_large_esm650.py`) + `37_build_graph_20c.py` (`26_build_20c.py`) + `38_build_graph_esm650_20c.py` (`29_build_650_20c.py`) → `--variant esm650_full\|esm20c\|esm650_20c` |
| Graph | `39_build_random_edge_controls.py` | `_build_random_edge_controls.py` |
| Pretrain | `40_pretrain.py` | `09_pretrain.py`, `09_pretrain_fixed.py` |
| Pretrain | `41_pretrain_single_cancer.py` | `10_equal_data_single_cancer.py` |
| Eval | `50_eval_checkpoint.py` | `15_eval_gfm.py`, `15_eval_final.py` |
| Eval | `51_eval_gene_embeddings.py` | `15b_eval_single.py` helpers |
| Eval | `52_eval_single_cancer.py` | `15b_eval_single.py` |
| Eval | `53_eval_esm_baseline.py` | `11_esm_baseline.py` |
| Eval | `54_eval_esm650.py` | `28_eval_650.py` |
| Eval | `55_eval_supervised_ppi.py` | `_eval_supervised.py` |
| Eval | `56_center_check_diag.py` | `19_center_check.py` |
| Validation | `60_metabric_validation.py` | `21_metabric_validation.py` |
| Validation | `61_patient_eval.py` | `24_patient_eval.py` |
| Validation | `62_patient_pretrain_eval.py` | `22_patient_pretrain_eval.py` |
| Validation | `63_paad_transfer_eval.py` | `27_paad_transfer_eval.py` |
| Validation | `64_drug_response_gdsc.py` | `16_drug_response_gdsc.py` |
| Validation | `65_drug_response_stats.py` | `12_drug_response_stats.py` |
| Validation | `66_tissue_specific_ppi.py` | `14_tissue_specific_ppi.py` |
| Analysis | `70_go_enrichment.py` | `20_go_enrichment.py` |
| Analysis | `71_pathway_membership.py` | `30_pathway_membership.py` |
| Analysis | `72_c3l_probe.py` | `31_c3l_probe.py` |
| Analysis | `73_feature_scan_probe.py` | `32_feature_scan_probe.py` |
| Analysis | `74_c3l_permutation_test.py` | `c3l_permutation_test.py` |
| Analysis | `75_patient_setfingerprint.py` | `_patient_setfingerprint.py` |
| Assembly | `80_collect_losses.py` | `17_collect_results.py` |
| Assembly | `81_assemble_results.py` | **merged** `81_assemble_results.py` + `82_assemble_final_results.py` (`18_assemble_results.py`, `23_assemble_final.py`) |
| Tables | `82_gen_final_tables.py` | `19_gen_final_tables.py` (was `83_…` in the 2026-08 pass) |
| Figures | `84_extract_fig1_layout.py` | **new** — caches the Fig. 1(a) network layout from the real graph |
| Figures | `85_make_figures.py` | **merged** `84_make_figures.py` (`make_figs_v2.py`) + `85_make_fig_s3.py` (`_gen_figS3.py`) + `86_make_patient_fig.py` (`_gen_patient_fig.py`) + `87_make_fig1.py` → one entry point, `--only` selector |
| Benchmark | `90_bench_device.py` | `bench_device.py` |
| Library | `config.py` | `config.py` (unchanged name; imported everywhere) |
| Library | `gfm_losses.py` | `gfm_losses.py` (unchanged name) |
| Library | `figure_data.py` | `fig_data.py` (2026-09 rename) |
| Library | `figure_style.py` | `figure_common.py` (2026-09 rename) |
| Driver | `run_ablation_evals.sh` | `run_ablation_evals.sh` |
| Driver | `run_namr_sweep_local.sh` | `run_namr_sweep_local.sh` |
| Driver | `run_namr_sweep_parallel.sh` | `run_namr_sweep_parallel.sh` |

## 2. Dropped one-off files (superseded, unreferenced, or single-use)

| Legacy file | Reason |
|---|---|
| `_patch1a.py`, `_tmp_*.py`, `_tmp_ck.py`, `_tmp_g.py`, `_tmp_meta.py`, `_tmp_path.py`, `_tmp_sb.py`, `_tmp_t.py` | ad-hoc inspection/patch scratch files |
| `09_pretrain_legacy.py` | superseded by `40_pretrain.py` |
| `fix_patient_edge.py` | one-off data repair, superseded by `33_build_patient_gene_edges.py` |
| `_post_batch1.sh`, `launch_job.sh` | one-off server launchers |
| `_gen_report.py` | superseded by `81_assemble_results.py` |
| `namr5_skip_decoder_DISCARDED` | discarded diagnostic checkpoint set |

## 3. Header convention applied to every released file

```
"""Author: Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
Date:   2026-08 (normalized 2026-09)

<what the script does and why>

Input:  <artifacts/arguments, with the config symbol or CLI flag that supplies them>
Output: <files written, or "stdout only">
Run:    python scripts/<file> [key flags]
"""
```

All comments, docstrings and printed/logged messages are English. Business
configuration (paths, seeds, protocol constants, dataset facts) is read from
`config.py`/CLI arguments or from the loaded data — never re-typed as a literal.
Random seeds exist only in `config.py` (`GFM_SEED`, default 42).

## 4. Known deliberate constants

| Location | Constant | Why it stays |
|---|---|---|
| `config.py` | `SEED=42`, `EMBEDDING_DIM=256`, `PPI_SCORE_THRESHOLD=700`, `NAMR_MASK_RATIO=0.15`, `NAMR_NOISE_SIGMA=0.5`, `C3L_TEMPERATURE=0.07`, `C3L_PROJ_DIM=128`, `CSP_BATCH_SIZE=256`, `N_PPI_PAIRS`, `CLUSTER_K`, `FEWSHOT_K`, `FEWSHOT_REPEATS`, `PERMUTATION_N` | single source of truth for protocol settings (env-overridable) |
| analysis scripts | `RANDOM_BASELINE_SEED=0`, `BINARY_AUC_SEED=7` | deterministic control/split seeds kept distinct from `config.SEED` so published numbers reproduce |
| model code | HGT geometry (`hidden=256`, `layers=6`, `heads=8`), loss weights, learning rate | must match the archived checkpoints; documented in each header |
| `figure_style.py` | Okabe-Ito hex palette, `CHANCE=0.5` | presentation constants (colourblind-safe palette, AUC chance level) |
