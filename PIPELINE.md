# PIPELINE.md — end-to-end execution guide

Run every script from the repository root (or set `GFM_PROJECT_ROOT`). Paths
resolve through `scripts/config.py`; see `docs/DATA_LAYOUT.md` for the artifact
contract and `docs/FIGURE_REVIEW.md` for figure provenance.

## Stage 1 — Download public data
| Script | Output |
|---|---|
| `10_download_string_ppi.py` | `$DATA_RAW/string/*` (STRING v12 sequences, info, links) |
| `11_download_tcga_xena.R` | `$DATA_RAW/tcga/...` (GDC alternative; the manuscript uses the UCSC Xena export) |

The manuscript's TCGA matrices come from the UCSC Xena pan-cancer export and are
referenced through `GFM_XENA_DIR`; `21_build_graph_xena_large.py` exits with a
clear error when it is unset.

## Stage 2 — Embeddings, graphs, targets
| Script | Output |
|---|---|
| `30_embed_esm2.py --model 8m` | `$DATA_PROC/esm2_gene_embeddings.pt` (320-dim) |
| `30_embed_esm2.py --model 650m` | `$DATA_PROC/esm2_650m_gene_embeddings.pt` (1,280-dim) |
| `20_build_graph_esm.py` | `$DATA_PROC/xena_graph_esm.pt` (5,000 genes, 21 cancers) |
| `21_build_graph_xena_large.py` | `$DATA_PROC/xena_graph_large.pt` (13,881 genes) |
| `22_build_test_graph_ci.py` | CI-only toy graph (never used for analyses) |
| `32_precompute_c3l_targets.py` | `$DATA_PROC/gene_cancer_targets*.pt` |
| `33_build_patient_gene_edges.py` | patient–gene edge tables |
| `34_build_patient_esm_graph.py` | `$DATA_PROC/xena_graph_patient_esm.pt` |
| `35_build_patient_paad_graph.py` | `$DATA_PROC/xena_graph_patient_paad.pt` |
| `36_build_graph_variants.py --variant esm650_full` | 13,881-gene 650M graph + targets |
| `36_build_graph_variants.py --variant esm20c` | 20-cancer graph (held-out PAAD) |
| `36_build_graph_variants.py --variant esm650_20c` | 13,881-gene / 20-cancer 650M graph |
| `39_build_random_edge_controls.py` | equal-volume and rewired control graphs |

## Stage 3 — Pretraining
| Script | Purpose |
|---|---|
| `40_pretrain.py` | Canonical pretraining. `--decoder plain` (study default) or `skip` (diagnostic); `--namr_weight/--csp_weight/--c3l_weight` set the task mix; `--seed` for multi-seed runs |
| `41_pretrain_single_cancer.py` | Equal-budget single-cancer (BRCA) control |

Checkpoints are written to `$MODEL_DIR` (or `--out`); every checkpoint stores the
full per-epoch `loss_history`, which is the source for Figs. 2, S1 and S3.

Convenience drivers: `run_namr_sweep_local.sh`, `run_namr_sweep_parallel.sh`
(standardization sweep) and `run_ablation_evals.sh` (evaluation of every
ablation checkpoint into `$RESULT_DIR/ablation_real/eval_summary.txt`).

## Stage 4 — Evaluation
| Script | Purpose |
|---|---|
| `50_eval_checkpoint.py` | Canonical downstream evaluation: PPI link prediction (2,000 pos/neg pairs) + few-shot cluster accuracy (30 repeats); writes one `[tag] …` row |
| `51_eval_gene_embeddings.py` | Embedding-space diagnostics for a checkpoint |
| `52_eval_single_cancer.py` | Single-cancer control evaluation |
| `53_eval_esm_baseline.py` | Raw ESM-2 baseline under the same protocol |
| `54_eval_esm650.py` | 650M-feature evaluation |
| `55_eval_supervised_ppi.py` | Supervised link-prediction protocol (literature-comparable) |
| `56_center_check_diag.py` | Embedding-centre / isotropy diagnostic |

## Stage 5 — Downstream and external validation
| Script | Purpose |
|---|---|
| `60_metabric_validation.py` | METABRIC ER-status transfer (per-split AUROC arrays) |
| `61_patient_eval.py`, `62_patient_pretrain_eval.py` | Patient-level (P-NAMR) evaluation and learning curve |
| `63_paad_transfer_eval.py` | Held-out PAAD expression transfer (Spearman rho arrays) |
| `64_drug_response_gdsc.py`, `65_drug_response_stats.py` | GDSC2 drug-response probe (negative result) and its statistics |
| `66_tissue_specific_ppi.py` | GTEx-filtered tissue-specific PPI edge loss |

## Stage 6 — Functional analysis and probes
| Script | Purpose |
|---|---|
| `70_go_enrichment.py` | Per-cluster GO over-representation for NAMR-only embeddings |
| `71_pathway_membership.py` | KEGG pathway-membership AUROC |
| `72_c3l_probe.py`, `73_feature_scan_probe.py` | C3L-target recoverability probes |
| `74_c3l_permutation_test.py` | Permutation test on the C3L loss |
| `120_patient_setfingerprint.py` | Set-fingerprint mechanism control |

## Stage 7 — Result assembly, tables, figures
| Script | Output |
|---|---|
| `80_collect_losses.py` | `$RESULT_DIR/ablation_real/loss_summary.json` |
| `81_assemble_results.py` | `$RESULT_DIR/final_results_summary.md` (every number re-read from artifacts) |
| `82_gen_final_tables.py` | `$RESULT_DIR/final_tables.tex` |
| `84_extract_fig1_layout.py` | `$RESULT_DIR/fig1_subgraph_layout.json` (Fig. 1a network layout) |
| `85_make_figures.py` | every main + supplementary figure as 300-dpi PNG + vector PDF into `$FIG_DIR` |

## Stage 8 — Benchmark
| Script | Purpose |
|---|---|
| `122_bench_device.py` | Device/BLAS benchmark used for the compute discussion |

## Reproducing the manuscript numbers

1. `run_ablation_evals.sh` → `$RESULT_DIR/ablation_real/eval_summary.txt` (seed-42 rows, pinned stack).
2. `80_collect_losses.py` → `loss_summary.json` (per-run first/last/min losses).
3. `81_assemble_results.py` → `final_results_summary.md` (all derived statistics).
4. `82_gen_final_tables.py` → `final_tables.tex` (Table 1 + Supplementary Table S1).
5. `84_extract_fig1_layout.py` → Fig. 1(a) layout; `85_make_figures.py` → all figures.
6. `python tests/test_figure_data.py` → asserts that every figure value still matches
   `eval_summary.txt` and the provenance JSON files.

Platform note: CUDA numbers are primary; the Apple MPS replication used the same
code and seeds (`Supplementary Table S12`). Clustering accuracy is
platform-robust at matched seeds; PPI-derived AUC varies with platform and is
reported as a supporting metric.
