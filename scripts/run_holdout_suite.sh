#!/usr/bin/env bash
# @file    run_holdout_suite.sh
# @brief   End-to-end driver for the held-out link-prediction + structural-control reframe
#          (reframe 3 of 4 in the Level-3 scope-exploration evaluation). Dispatches the
#          held-out edge split builder, the four structural-control graph builders (ppi_only,
#          dummy21, expr_real, expr_shuffled), the 6 NAMR-only retrainings, and the
#          final held-out evaluation so a single command reproduces the entire Supplementary
#          section on hold-out edge-prediction generalisability.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
# @date    2026-09
# @ingroup shell_scheduler
#
# Input:  None explicitly required on the CLI (all data/result paths are resolved through
#         GFM_PROJECT_ROOT / GFM_DATA_PROC / GFM_RESULT_DIR env overrides, mirroring
#         scripts/config.py). Run on the same CUDA GPU server as the original experiments.
# Output: Intermediate graph builders + 6 NAMR-only checkpoints under results/holdout/,
#         plus a final holdout_eval_summary.txt echoed to stdout.
# Run:    bash scripts/run_holdout_suite.sh    (from the repository root)
set -euo pipefail
ROOT="${GFM_PROJECT_ROOT:-$(pwd)}"
cd "$ROOT"
PY=python3   # GPU server interpreter (CUDA)

echo "=== 1. held-out edge split + control graphs ==="
$PY scripts/77_build_holdout_graph.py
$PY scripts/79_build_control_graphs.py --variant ppi_only
$PY scripts/79_build_control_graphs.py --variant dummy21
$PY scripts/79_build_control_graphs.py --variant expr_real
$PY scripts/79_build_control_graphs.py --variant expr_shuffled

echo "=== 2. retrain (6 runs, NAMR-only protocol unless noted) ==="
common=(--targets data/processed/gene_cancer_targets_5000.pt --epochs 500 --seed 42 --standardize_features --save_every 100)
$PY scripts/40_pretrain.py --data results/holdout/train_graph.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 0 --c3l_weight 0 --out results/holdout/namr_only_heldout.pt
$PY scripts/40_pretrain.py --data results/holdout/train_graph.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 1 --c3l_weight 0 --out results/holdout/csp_namr_heldout.pt
$PY scripts/40_pretrain.py --data results/control/ppi_only.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 0 --c3l_weight 0 --out results/control/namr_ppi_only.pt
$PY scripts/40_pretrain.py --data results/control/dummy21.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 0 --c3l_weight 0 --out results/control/namr_dummy21.pt
$PY scripts/40_pretrain.py --data results/control/expr_real.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 0 --c3l_weight 0 --out results/control/namr_expr_real.pt
$PY scripts/40_pretrain.py --data results/control/expr_shuffled.pt "${common[@]}" \
    --namr_weight 1 --csp_weight 0 --c3l_weight 0 --out results/control/namr_expr_shuffled.pt

echo "=== 3. held-out evaluation (3 negative schemes) ==="
$PY scripts/78_eval_holdout.py --ckpt results/holdout/namr_only_heldout.pt --tag namr_only_heldout
$PY scripts/78_eval_holdout.py --ckpt results/holdout/csp_namr_heldout.pt --tag csp_namr_heldout

echo "=== done: see results/holdout/eval_*.json and results/control/*.pt ==="
