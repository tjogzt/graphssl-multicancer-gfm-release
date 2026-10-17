#!/bin/bash
# @file    run_namr_sweep_local.sh
# @brief   Local-CPU NAMR convergence sweep: runs the OFFICIAL ARCHITECTURE (6-layer HGT,
#          256 hidden, 8 attention heads) on the ESM graph with real gene-cancer targets for
#          100 epochs per configuration, then summarises the NAMR/CSP convergence captured
#          in each run's log. This is the Mac-workstation (non-GPU) mirror of the GPU-side
#          run_namr_sweep.sh script.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
# @date    2026-08 (normalized 2026-09)
# @ingroup shell_scheduler
#
# Input:  <DATA_PROC>/xena_graph_esm.pt and <DATA_PROC>/gene_cancer_targets_5000.pt
#         (env-overridable via GFM_PROJECT_ROOT / GFM_DATA_PROC / GFM_RESULT_DIR);
#         positional argument $1 = OMP thread count per run (default = 4).
# Output: <RESULT_DIR>/namr_sweep/<tag>.pt + <tag>.log for every configuration, plus a
#         summary_local.txt table that is also printed directly to stdout.
# Run:    bash scripts/run_namr_sweep_local.sh [threads]

set -u

# Project/data/result roots mirror scripts/config.py so the GFM_* env overrides apply.
ROOT="${GFM_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_PROC="${GFM_DATA_PROC:-$ROOT/data/processed}"
RESULT_DIR="${GFM_RESULT_DIR:-$ROOT/results}"

# Per-run artifact filenames (module-level constants; rename in one place).
XENA_GRAPH="xena_graph_esm.pt"            # ESM-2 graph with 5,000-gene PCA features
GENE_TARGETS="gene_cancer_targets_5000.pt" # real gene-cancer specificity targets
SUMMARY_FILE="summary_local.txt"           # convergence table written by this sweep
SWEEP_DIR_NAME="namr_sweep"                # subdir under RESULT_DIR for sweep outputs

# Protocol settings: the epoch budget and log cadence are named, not re-typed inline.
EPOCHS=100
LOG_EVERY=100

DATA="$DATA_PROC/$XENA_GRAPH"
TGT="$DATA_PROC/$GENE_TARGETS"
OUTDIR="$RESULT_DIR/$SWEEP_DIR_NAME"
THREADS="${1:-4}"

mkdir -p "$OUTDIR"
echo "config | final_NAMR | first_NAMR | final_CSP | final_total" | tee "$OUTDIR/$SUMMARY_FILE"

run() {
  local tag=$1; shift
  local ckpt="$OUTDIR/${tag}.pt"
  echo "=== [$tag] $* ==="
  # Single run with OMP pinned to THREADS; the full log is kept for metric extraction.
  OMP_NUM_THREADS=$THREADS python3 -u scripts/40_pretrain.py --data "$DATA" --targets "$TGT" \
    --epochs $EPOCHS --log_every $LOG_EVERY --out "$ckpt" "$@" > "$OUTDIR/${tag}.log" 2>&1
  # Convergence readouts: first/last NAMR and last CSP/Tot values printed to the log.
  local first=$(grep -oP 'NAMR \K[0-9.]+' "$OUTDIR/${tag}.log" | head -1)
  local final=$(grep -oP 'NAMR \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  local csp=$(grep -oP 'CSP \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  local tot=$(grep -oP 'Tot \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  echo "$tag | $final | $first | $csp | $tot" | tee -a "$OUTDIR/$SUMMARY_FILE"
}

run base_ref

run h1_std --standardize_features --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
run h2_s025 --namr_sigma 0.25 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
run h2_s010 --namr_sigma 0.10 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
run h3_mask --namr_mode mask --namr_sigma 0.5 --namr_weight 1.0 --lr 1e-3
run h4_w3 --namr_sigma 0.5 --namr_mode denoise --namr_weight 3.0 --lr 1e-3
run h4_w10 --namr_sigma 0.5 --namr_mode denoise --namr_weight 10.0 --lr 1e-3
run h5_lr3e3 --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 3e-3
run h5_lr5e4 --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 5e-4

echo ""
echo "=== SWEEP SUMMARY ==="
cat "$OUTDIR/$SUMMARY_FILE"
