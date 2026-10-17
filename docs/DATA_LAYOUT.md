# DATA_LAYOUT.md — directories, environment variables, artifact contract

## 1. Directory layout

```
<PROJECT_ROOT>/                     # GFM_PROJECT_ROOT; defaults to the repository root
├── data/
│   ├── raw/                        # public downloads (STRING v12, Xena/GDC, GDSC2, GTEx, KEGG)
│   └── processed/                  # graphs, embeddings, targets (.pt)
├── models/checkpoints/             # training checkpoints (loss_history inside)
├── results/                        # evaluation summaries, .npy per-split arrays, JSON provenance
│   ├── ablation_real/              # eval_summary.txt, loss_summary.json, *.pt checkpoints used for figures
│   ├── tissue_ppi/                 # per-tissue filtered PPI summaries
│   ├── fig1_subgraph_layout.json   # Fig. 1(a) network layout (from the real graph)
│   ├── figS3_auc_provenance.json   # Supplementary Fig. S3(b) bar values + their run provenance
│   └── patient_fewshot_acc.json    # Supplementary Fig. S2 accuracy curve (Table S7 provenance)
├── cloud_results/                  # GPU-server artifacts (e.g. ESM-2 650M embeddings, C3L targets)
├── manuscript/                     # LaTeX sources + figure PDFs/PNGs (GFM_FIG_DIR)
└── scripts/ tests/ figures/ docs/  # this code repository (paths resolved via §2)
```

## 2. Environment variables (all optional; `scripts/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `GFM_PROJECT_ROOT` | parent of `scripts/` | root that holds `data/`, `results/`, `models/`, `cloud_results/` |
| `GFM_DATA_RAW` | `<root>/data/raw` | public downloads |
| `GFM_DATA_PROC` | `<root>/data/processed` | processed graphs/embeddings |
| `GFM_MODEL_DIR` | `<root>/models/checkpoints` | checkpoints |
| `GFM_RESULT_DIR` | `<root>/results` | evaluation outputs and provenance files |
| `GFM_CLOUD_DIR` | `<root>/cloud_results` | GPU-server artifacts |
| `GFM_FIG_DIR` | `<root>/manuscript` | figure output directory |
| `GFM_STRING_DIR` | `<DATA_RAW>/string` | STRING v12 files |
| `GFM_XENA_DIR` | unset | UCSC Xena pan-cancer export (required by `21_build_graph_xena_large.py`) |
| `GFM_GTEX_DIR`, `GFM_HPA_DIR` | unset | optional external expression references |
| `GFM_SEED` | `42` | single seed used by `config.set_seed()` |

## 3. Artifact contract (who writes / who reads)

| Artifact | Written by | Read by |
|---|---|---|
| `$DATA_PROC/esm2_gene_embeddings.pt` | `30_embed_esm2.py --model 8m` | `20_build_graph_esm.py`, baselines, probes |
| `$DATA_PROC/xena_graph_esm.pt` | `20_build_graph_esm.py` | `40_pretrain.py`, `50_eval_checkpoint.py`, `74_*`, `84_*` |
| `$DATA_PROC/xena_graph_large.pt` / `xena_graph_esm650_large.pt` | `21_build_graph_xena_large.py`, `36_build_graph_variants.py` | `71_pathway_membership.py`, `84_extract_fig1_layout.py` |
| `$DATA_PROC/gene_cancer_targets*.pt` | `32_precompute_c3l_targets.py`, `36_build_graph_variants.py` | `40_pretrain.py`, `72/73_*` probes |
| `$MODEL_DIR/*.pt` (checkpoints, with `loss_history`) | `40_pretrain.py`, `41_pretrain_single_cancer.py` | `50–56_*`, `80_collect_losses.py`, `85_make_figures.py` |
| `$RESULT_DIR/ablation_real/eval_summary.txt` | `run_ablation_evals.sh` (calls `50_eval_checkpoint.py`) | `81_assemble_results.py`, `82_gen_final_tables.py`, `figure_data.py`, `tests/test_figure_data.py` |
| `$RESULT_DIR/ablation_real/loss_summary.json` | `80_collect_losses.py` | manuscript text, `figure_data.py` |
| `$RESULT_DIR/metabric_*_auc_k50.npy`, `paad_rho_*.npy` | `60_metabric_validation.py`, `63_paad_transfer_eval.py` | `81_assemble_results.py` |
| `$RESULT_DIR/fig1_subgraph_layout.json` | `84_extract_fig1_layout.py` | `85_make_figures.py` (Fig. 1a) |
| `$RESULT_DIR/figS3_auc_provenance.json` | produced by the 2026-09 re-evaluations (documented in `docs/FIGURE_REVIEW.md`) | `85_make_figures.py` (Fig. S3b) |
| `$RESULT_DIR/patient_fewshot_acc.json` | transcribed from Supplementary Table S7 with its provenance field | `85_make_figures.py` (Fig. S2) |
| `$FIG_DIR/*.pdf`, `*.png` | `85_make_figures.py` | manuscript LaTeX |

## 4. Reproducibility rules enforced in code

1. **No transcribed numbers in plotting or table code.** `figure_data.py` and
   `81_assemble_results.py` read every value from an artifact; where a value
   cannot be derived from a machine artifact (patient few-shot curve, Fig. S3
   bars) a JSON provenance file records the source and the run identity.
2. **No fabricated/mock data.** The only synthetic artifact is
   `22_build_test_graph_ci.py` (a CI smoke-test graph, explicitly labelled and
   never used for analyses).
3. **Single seed source.** `config.SEED`/`config.set_seed()`; control seeds that
   must differ from the training seed are named constants with a comment.
4. **Path independence.** Absolute paths never appear in scripts; every path is
   composed from `config` symbols or is a module-level filename constant.
