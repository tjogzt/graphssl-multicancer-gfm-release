#!/usr/bin/env python3
"""
@file    80_collect_losses.py
@brief   Collect the final ablation loss histories into a single JSON summary.
         After the ablation runs finish, this script loads each collected checkpoint,
         reads its per-epoch loss_history (csp / namr / c3l / total), and records the
         first, last and minimum value of every term. The aggregate is written to one
         JSON file that downstream figure/table scripts read, so reporting code never
         has to re-open the large checkpoint files itself.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup result_assembly_and_figures

Input:  <RESULT_DIR>/ablation_real/*.pt checkpoints (torch loss_history + args)
Output: <RESULT_DIR>/ablation_real/loss_summary.json plus one progress line per
        checkpoint on stdout
Run:    python scripts/80_collect_losses.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (checkpoint tags, artifact names)
# ============================================================================

# Results subdirectory that holds the ablation checkpoints and their summaries.
ABLATION_SUBDIR = "ablation_real"

# Per-run checkpoint basenames collected here (each maps to a table tag).
CKPT_FULL = "full.pt"
CKPT_NAMR_ONLY = "namr_only.pt"
CKPT_CSP_ONLY = "csp_only.pt"
CKPT_CSP_NAMR = "csp_namr_legacy_s42.pt"

# Aggregate output filename written next to the checkpoints.
LOSS_SUMMARY_JSON = "loss_summary.json"

# Loss history terms to extract and summarize (in reporting order).
LOSS_TERMS = ("csp", "namr", "c3l", "total")

# Statistical summary keys extracted per loss term.
SUMMARY_KEYS = ("first", "last", "min")

# Checkpoint dictionary keys.
CKPT_KEY_LOSS_HISTORY = "loss_history"
CKPT_KEY_ARGS = "args"

# Rounding precision for reported loss summary values.
LOSS_VALUE_PRECISION = 4

# Missing checkpoint console tag.
MISSING_TAG = "MISSING"

# JSON serialization indentation.
JSON_INDENT = 1


# ============================================================================
# Checkpoint tags to collect. The "csp_namr" tag maps to the plain-decoder
# legacy checkpoint (manuscript Table S1/S3); the skip-decoder diagnostic
# checkpoint (csp_namr_skipdiag) is intentionally not collected here.
# ============================================================================
CKPTS = {
    "full": config.result_path(ABLATION_SUBDIR, CKPT_FULL),
    "namr_only": config.result_path(ABLATION_SUBDIR, CKPT_NAMR_ONLY),
    "csp_only": config.result_path(ABLATION_SUBDIR, CKPT_CSP_ONLY),
    "csp_namr": config.result_path(ABLATION_SUBDIR, CKPT_CSP_NAMR),
}


# ============================================================================
# Core Functions
# ============================================================================

def parse_losses(path):
    """
    @brief   Extract first/last/min of each tracked loss term from a checkpoint.

    @details Iterates over LOSS_TERMS; only summarizes terms that the trainer
             actually recorded. An empty history is skipped so first/last/min are
             never computed over a zero-length array. Also attaches the run args
             as provenance metadata.

    @param   path   Path to a pretraining checkpoint .pt file.
    @return  Dictionary with per-term {first, last, min} and the 'args' provenance.
    @note    Values are rounded to LOSS_VALUE_PRECISION decimal places.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    history = ck.get(CKPT_KEY_LOSS_HISTORY, {})
    out = {}

    # Only summarize terms the trainer actually recorded.
    for term in LOSS_TERMS:
        values = np.asarray(history.get(term, []), dtype=float)
        if len(values):
            out[term] = {
                SUMMARY_KEYS[0]: round(float(values[0]), LOSS_VALUE_PRECISION),
                SUMMARY_KEYS[1]: round(float(values[-1]), LOSS_VALUE_PRECISION),
                SUMMARY_KEYS[2]: round(float(values.min()), LOSS_VALUE_PRECISION),
            }

    # Keep the run arguments as provenance (plain data, not metrics).
    out[CKPT_KEY_ARGS] = ck.get(CKPT_KEY_ARGS, {})
    return out


def main():
    """
    @brief   Main entry point: iterate over registered ablation checkpoints,
             collect loss summaries, and write a single JSON aggregate.
    @return  None (outputs per-checkpoint progress to stdout; writes JSON summary).
    @note    Missing checkpoints are reported with MISSING_TAG and do not abort.
    """
    # Set global random seed for full reproducibility.
    config.set_seed()
    results = {}

    # Process each registered checkpoint in tag order.
    for tag, path in CKPTS.items():
        if os.path.exists(path):
            results[tag] = parse_losses(path)
            # Drop "args" from the console echo so printed metrics stay readable.
            print(tag, "losses:", {k: v for k, v in results[tag].items() if k != CKPT_KEY_ARGS})
        else:
            print(tag, MISSING_TAG)

    # Ensure output directory exists and write the JSON summary.
    out_path = config.result_path(ABLATION_SUBDIR, LOSS_SUMMARY_JSON)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=JSON_INDENT, default=str)
    print("saved", out_path)


if __name__ == "__main__":
    main()
