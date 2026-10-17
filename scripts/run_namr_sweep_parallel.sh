#!/bin/bash
# @file    run_namr_sweep_parallel.sh
# @brief   Conservative parallel NAMR convergence sweep: dispatches each convergence
#          configuration through a bounded worker pool instead of running them
#          serially. Uses a fixed PAR-background × OMP-threads-per-job safety budget
#          tuned for M4 Max class machines so the sweep saturates cores without
#          pushing the system into swap. Behaviourally equivalent to the serial
#          run_namr_sweep_local.sh so results are byte-identical to the baseline.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
# @date    2026-08 (normalized 2026-09)
# @ingroup shell_scheduler
#
# Input:  <DATA_PROC>/xena_graph_esm.pt + <DATA_PROC>/gene_cancer_targets_5000.pt
#         (GFM_PROJECT_ROOT / GFM_DATA_PROC / GFM_RESULT_DIR overrides honoured);
#         PAR env variable = number of parallel jobs dispatched (default = 2).
# Output: <RESULT_DIR>/namr_sweep/<tag>.pt + <tag>.log per config, a generated
#         configs.txt enumerating the grid, and summary_local.txt also printed to stdout.
# Run:    PAR=4 bash scripts/run_namr_sweep_parallel.sh

set -u

# Project/data/result roots mirror scripts/config.py so the GFM_* env overrides apply.
ROOT="${GFM_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_PROC="${GFM_DATA_PROC:-$ROOT/data/processed}"
RESULT_DIR="${GFM_RESULT_DIR:-$ROOT/results}"

# Per-run artifact filenames (module-level constants; rename in one place).
XENA_GRAPH="xena_graph_esm.pt"            # ESM-2 graph with 5,000-gene PCA features
GENE_TARGETS="gene_cancer_targets_5000.pt" # real gene-cancer specificity targets
SUMMARY_FILE="summary_local.txt"           # convergence table written by this sweep
CONFIGS_FILE="configs.txt"                 # generated "tag [flags...]" grid read by the loop
SWEEP_DIR_NAME="namr_sweep"                # subdir under RESULT_DIR for sweep outputs

# Protocol settings: epoch budget, log cadence, and the per-run OMP thread count.
EPOCHS=100
LOG_EVERY=100
OMP_THREADS=2

DATA="$DATA_PROC/$XENA_GRAPH"
TGT="$DATA_PROC/$GENE_TARGETS"
OUTDIR="$RESULT_DIR/$SWEEP_DIR_NAME"
PARALLEL="${PAR:-2}"

mkdir -p "$OUTDIR"
echo "config | final_NAMR | first_NAMR | final_CSP | final_total" > "$OUTDIR/$SUMMARY_FILE"

run_one() {
  local tag=$1; shift
  local ckpt="$OUTDIR/${tag}.pt"
  # One job per config; OMP is pinned so PAR jobs do not oversubscribe the machine.
  OMP_NUM_THREADS=$OMP_THREADS python3 -u scripts/40_pretrain.py --data "$DATA" --targets "$TGT" \
    --epochs $EPOCHS --log_every $LOG_EVERY --out "$ckpt" "$@" > "$OUTDIR/${tag}.log" 2>&1
  # Convergence readouts: first/last NAMR and last CSP/Tot values printed to the log.
  local first=$(grep -oP 'NAMR \K[0-9.]+' "$OUTDIR/${tag}.log" | head -1)
  local final=$(grep -oP 'NAMR \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  local csp=$(grep -oP 'CSP \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  local tot=$(grep -oP 'Tot \K[0-9.]+' "$OUTDIR/${tag}.log" | tail -1)
  echo "$tag | $final | $first | $csp | $tot" >> "$OUTDIR/$SUMMARY_FILE"
}

# Write the config grid once (one "tag [flags...]" line per config) so the dispatch
# loop below iterates a stable file rather than a duplicated inline list.
cat > "$OUTDIR/$CONFIGS_FILE" <<EOF
base_ref
h1_std --standardize_features --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
h2_s025 --namr_sigma 0.25 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
h2_s010 --namr_sigma 0.10 --namr_mode denoise --namr_weight 1.0 --lr 1e-3
h3_mask --namr_mode mask --namr_sigma 0.5 --namr_weight 1.0 --lr 1e-3
h4_w3 --namr_sigma 0.5 --namr_mode denoise --namr_weight 3.0 --lr 1e-3
h4_w10 --namr_sigma 0.5 --namr_mode denoise --namr_weight 10.0 --lr 1e-3
h5_lr3e3 --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 3e-3
h5_lr5e4 --namr_sigma 0.5 --namr_mode denoise --namr_weight 1.0 --lr 5e-4
EOF

while IFS= read -r line; do
  [ -z "$line" ] && continue
  # eval splits the line into run_one's "tag" plus its flag arguments.
  eval "run_one $line" &
  # Bound the worker pool: wait while the running-job count is at PARALLEL.
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do sleep 10; done
done < "$OUTDIR/$CONFIGS_FILE"
wait   # join all background runs before printing the summary
echo "=== SWEEP SUMMARY ==="
cat "$OUTDIR/$SUMMARY_FILE"
