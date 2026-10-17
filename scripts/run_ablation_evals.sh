#!/bin/bash
# @file    run_ablation_evals.sh
# @brief   Driver script that evaluates every real ablation checkpoint PLUS the three
#          reference baselines (ESM-2, PCA, random) and appends all results to a single
#          tab-separated metric table. One rerun reproduces the full ablation comparison
#          table used in the manuscript without dropping any baseline rows.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
# @date    2026-08 (normalized 2026-09)
# @ingroup shell_scheduler
#
# Input:  Checkpoint files under <RESULT_DIR>/ablation_real/ (full.pt, namr_only.pt,
#         csp_only.pt, csp_namr_legacy_s42.pt) plus the graph data that 50_eval_checkpoint.py
#         loads from config.DATA_PROC. All locations honour the GFM_PROJECT_ROOT /
#         GFM_RESULT_DIR environment overrides.
# Output: <RESULT_DIR>/ablation_real/eval_summary.txt (tab-separated metric table);
#         each row is also echoed to stdout for Makefile-level log capture.
# Run:    bash scripts/run_ablation_evals.sh

# Project/result roots mirror scripts/config.py so the GFM_* env overrides apply.
ROOT="${GFM_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RESULT_DIR="${GFM_RESULT_DIR:-$ROOT/results}"
ABLATION_DIR="$RESULT_DIR/ablation_real"

# Per-run artifact filenames (module-level constants; rename in one place).
EVAL_SUMMARY="eval_summary.txt"     # metric table written by this script
# Ablation checkpoints evaluated in this run (basenames inside ABLATION_DIR).
# NOTE: csp_namr.pt is a skip-decoder diagnostic artifact; the manuscript ablation
# table uses the plain-decoder checkpoint csp_namr_legacy_s42.pt (see REVISION_LOG #5).
CKPTS="full.pt namr_only.pt csp_only.pt csp_namr_legacy_s42.pt"
# Reference baselines: no checkpoint needed, evaluated by --baseline.
BASELINES="esm pca random"

cd "$ROOT"
OUT="$ABLATION_DIR/$EVAL_SUMMARY"
echo "tag | PPI ratio | pos | neg | AUC | cluster acc | std" > "$OUT"

for ck in $CKPTS; do
  ck_path="$ABLATION_DIR/$ck"
  [ -f "$ck_path" ] || continue   # skip missing checkpoints so partial reruns still finish
  # Tag is the checkpoint basename with the legacy seed suffix stripped (stable row label).
  tag=$(basename "$ck" .pt | sed 's/_legacy_s42//')
  # Single-threaded BLAS keeps the eval deterministic and avoids CPU oversubscription.
  # grep keeps only the "[tag] ..." summary line that 50_eval_checkpoint.py prints last.
  line=$(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/50_eval_checkpoint.py --ckpt "$ck_path" --tag "$tag" 2>/dev/null | grep "^\[" | tail -1)
  echo "$line" | tee -a "$OUT"
done

# Baselines are regenerated on every run so a rerun never drops the reference rows.
for b in $BASELINES; do
  line=$(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/50_eval_checkpoint.py --baseline "$b" 2>/dev/null | grep "^\[" | tail -1)
  echo "$line" | tee -a "$OUT"
done
echo "=== DONE ==="
