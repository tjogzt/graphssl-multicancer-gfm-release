# Makefile — CRM Methods Track reproducible pipeline (5 targets)
# ----------------------------------------------------------------
# All targets inject GFM_PROJECT_ROOT=$(CURDIR) so scripts/config.py picks
# up the flattened repository layout (scripts/ lives one level below the
# Makefile, there is NO github_submit/ prefix in this repository).
#
# Usage:
#   make data        # (1) Download every raw public dataset.
#   make preprocess  # (2) ESM-2 embeddings + build every graph variant.
#   make train       # (3) 5 seeds × 500-epoch HGT pretraining (5 ablation arms).
#   make evaluate    # (4) Downstream evaluation: PPI AUC, cluster, pathway, METABRIC, PAAD, GDSC2.
#   make figures     # (5) Generate all manuscript figures: PDF + 300-dpi PNG.
# ----------------------------------------------------------------
SHELL := /bin/bash
export GFM_PROJECT_ROOT := $(CURDIR)

PY ?= python3
RSCRIPT ?= Rscript
SCRIPTDIR = $(CURDIR)/scripts

.PHONY: all data preprocess train evaluate figures

# ---------------- all (5-target convenience wrapper) ----------------
all: data preprocess train evaluate figures

# ---------------- (1) data ----------------
data:
	@echo "[Makefile::data] Download STRING PPI + TCGA Xena (public HTTPS sources)."
	$(PY) $(SCRIPTDIR)/10_download_string_ppi.py
	$(RSCRIPT) $(SCRIPTDIR)/11_download_tcga_xena.R

# ---------------- (2) preprocess ----------------
preprocess:
	@echo "[Makefile::preprocess] ESM-2 embeddings + graph build (8m default variant)."
	$(PY) $(SCRIPTDIR)/30_embed_esm2.py --model 8m
	$(PY) $(SCRIPTDIR)/20_build_graph_esm.py  --model 8m --variant esm_full
	$(PY) $(SCRIPTDIR)/32_precompute_c3l_targets.py
	$(PY) $(SCRIPTDIR)/33_build_patient_gene_edges.py

# ---------------- (3) train ----------------
train:
	@echo "[Makefile::train] 5 equal-budget seeds (42 7 123 21 99) — 500 epochs each."
	@for seed in 42 7 123 21 99; do \
	  echo "  → training seed $$seed"; \
	  $(PY) $(SCRIPTDIR)/40_pretrain.py --epochs 500 --seed $$seed; \
	done

# ---------------- (4) evaluate ----------------
evaluate:
	@echo "[Makefile::evaluate] PPI-link / cluster / pathway / METABRIC / PAAD / GDSC2 downstream."
	$(PY) $(SCRIPTDIR)/50_eval_checkpoint.py --ckpt results/ablation_real/full.pt --tag full
	$(PY) $(SCRIPTDIR)/60_metabric_validation.py
	$(PY) $(SCRIPTDIR)/63_paad_transfer_eval.py
	$(PY) $(SCRIPTDIR)/64_drug_response_gdsc.py
	$(PY) $(SCRIPTDIR)/70_go_enrichment.py
	$(PY) $(SCRIPTDIR)/71_pathway_membership.py
	bash $(SCRIPTDIR)/run_ablation_evals.sh

# ---------------- (5) figures ----------------
figures:
	@echo "[Makefile::figures] Regenerate every figure (PDF + PNG @300 dpi)."
	$(PY) $(SCRIPTDIR)/84_extract_fig1_layout.py
	$(PY) $(SCRIPTDIR)/85_make_figures.py
	$(PY) $(SCRIPTDIR)/86_check_figure_specs.py
