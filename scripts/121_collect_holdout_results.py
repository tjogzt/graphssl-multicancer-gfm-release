#!/usr/bin/env python3
"""
@file    121_collect_holdout_results.py
@brief   Aggregate held-out link-prediction + structural-control results for
         manuscript back-fill (review items 3 & 4). Collects:
           1. heldout eval JSONs (3 negative-sampling schemes for NAMR-only and
              CSP+NAMR);
           2. each control checkpoint's final NAMR reconstruction loss;
         then writes a single summary.json that downstream number-lock audits
         can treat as a primary source.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  results/holdout/eval_*.json, results/control/namr_*.pt
Output: results/holdout/summary.json + printed summary table
Run:    GFM_PROJECT_ROOT=<repo> python scripts/121_collect_holdout_results.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
# Result path tags used by config.result_path().
HOLDOUT_TAG = "holdout"
CONTROL_TAG = "control"

# Heldout tags: exactly the two ablation arms evaluated under the heldout
# link-prediction protocol.
HOLDOUT_ARM_TAGS = ("namr_only_heldout", "csp_namr_heldout")
EVAL_JSON_PREFIX = "eval_"
EVAL_JSON_SUFFIX = ".json"

# Control checkpoints: four structural-control variants produced by
# 79_build_control_graphs.py + the NAMR pretrain pipeline.
CONTROL_CHECKPOINTS = [
    ("namr_ppi_only",     "namr_ppi_only.pt"),
    ("namr_dummy21",      "namr_dummy21.pt"),
    ("namr_expr_real",    "namr_expr_real.pt"),
    ("namr_expr_shuffled", "namr_expr_shuffled.pt"),
]

# Output summary artefact basename (resolved against holdout result dir).
SUMMARY_JSON_FILENAME = "summary.json"

# Display precision for control loss values in the printed summary.
LOSS_DISPLAY_DIGITS = 4

# Summary JSON top-level keys (kept stable for downstream readers).
SUMMARY_KEY_HELDOUT = "heldout"
SUMMARY_KEY_CONTROL = "controls"
CONTROL_KEY_FINAL_NAMR_LOSS = "final_namr_loss"

# torch.load() map device; loaded weights are immediately converted to numpy so
# CPU is sufficient and keeps memory pressure predictable.
LOAD_MAP_DEVICE = "cpu"


def final_namr(ckpt_path):
    """Return the final NAMR reconstruction loss stored inside a checkpoint.

    @brief  Extract the last entry of the checkpoint's ``loss_history["namr"]``
            trajectory (the NAMR loss at the end of training).
    @param  ckpt_path   Absolute path to a ``*.pt`` checkpoint produced by the
                        40_pretrain.py pipeline.
    @return Final NAMR loss cast to a Python float (not numpy scalar).
    @note   ``weights_only=False`` is required because legacy checkpoints may
            carry extra state (optimiser / rng) that the restricted loader would
            reject. The caller is responsible for trusting the file origin.
    """
    ck = torch.load(ckpt_path, map_location=LOAD_MAP_DEVICE, weights_only=False)
    return float(np.asarray(ck["loss_history"]["namr"], dtype=float)[-1])


def main():
    """Collect heldout + control artefacts and serialise a single summary.json.

    @brief  Iterate over each heldout arm tag and each control checkpoint,
            load the artefact if present, fold the result into the summary dict,
            pretty-print the summary, and write the JSON payload.
    @return None (exit code managed by the __main__ guard).
    @note   Missing artefacts are silently skipped (the printed summary will
            simply omit the key). This lets the script be run on a partial
            checkout without crashing.
    """
    holdout_dir = config.result_path(HOLDOUT_TAG)
    control_dir = config.result_path(CONTROL_TAG)
    summary = {SUMMARY_KEY_HELDOUT: {}, SUMMARY_KEY_CONTROL: {}}

    # --- 1. Heldout eval JSONs -------------------------------------------------
    for tag in HOLDOUT_ARM_TAGS:
        json_name = f"{EVAL_JSON_PREFIX}{tag}{EVAL_JSON_SUFFIX}"
        path = os.path.join(holdout_dir, json_name)
        if os.path.exists(path):
            with open(path) as fh:
                summary[SUMMARY_KEY_HELDOUT][tag] = json.load(fh)

    # --- 2. Control checkpoints (final NAMR loss only) ------------------------
    for tag, ckpt_basename in CONTROL_CHECKPOINTS:
        path = os.path.join(control_dir, ckpt_basename)
        if os.path.exists(path):
            loss_val = round(final_namr(path), LOSS_DISPLAY_DIGITS)
            summary[SUMMARY_KEY_CONTROL][tag] = {CONTROL_KEY_FINAL_NAMR_LOSS: loss_val}

    out_path = os.path.join(holdout_dir, SUMMARY_JSON_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=1)
    # --- English-only stdout ---------------------------------------------------
    print(json.dumps(summary, indent=1))
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
