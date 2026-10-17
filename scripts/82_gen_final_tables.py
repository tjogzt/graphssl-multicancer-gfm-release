#!/usr/bin/env python3
"""
@file    82_gen_final_tables.py
@brief   Generate the final LaTeX results table from the machine artifacts.
         Every number in the ablation table is read back from the evaluation
         summary written by the eval scripts and from the checkpoint loss
         histories; nothing is hand-transcribed. Emits a replacement block
         for Table 1 (ablation + baselines) plus a comment block listing
         loss convergence, and writes it to a single .tex file for the manuscript.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup result_assembly_and_figures

Input:  <RESULT_DIR>/ablation_real/eval_summary.txt (rows written by eval scripts)
        <RESULT_DIR>/ablation_real/*.pt loss histories
Output: <RESULT_DIR>/final_tables.tex plus the same text on stdout
Run:    python scripts/82_gen_final_tables.py
"""
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact paths, table structure, regex patterns)
# ============================================================================

# Results subdirectory holding the ablation evaluation summary + checkpoints.
ABLATION_SUBDIR = "ablation_real"
EVAL_SUMMARY_NAME = "eval_summary.txt"
FINAL_TABLES_TEX = "final_tables.tex"

# Ablation checkpoint tags whose loss histories appear in the convergence comment.
ABLATION_TAGS = ["full", "namr_only", "csp_only", "csp_namr"]

# Per-tag checkpoint basename; csp_namr = plain-decoder legacy checkpoint.
# (The skip-decoder diagnostic is intentionally not part of Table 1.)
CKPT_FILES = {
    "full": "full.pt",
    "namr_only": "namr_only.pt",
    "csp_only": "csp_only.pt",
    "csp_namr": "csp_namr_legacy_s42.pt",
}

# Table 1 display order and human-readable labels (eval_summary tag -> label).
TABLE_ROWS = [
    ("full", "full"),
    ("namr_only", "namr_only"),
    ("csp_only", "csp_only"),
    ("csp_namr", "csp_namr"),
    ("random_init_hgt", "Random-init HGT"),
    ("esm_only", "ESM-2 only"),
    ("pca_input", "PCA input"),
]

# Eval summary row regex. Ratio may be negative for degenerate baselines (sign captured).
_EVAL_ROW_PATTERN = (
    r"^\[([^\]]+)\] PPI ratio=(-?[0-9.]+)x.*\| AUC=([0-9.]+) "
    r"\| cluster acc=([0-9.]+) \+/- ([0-9.]+)"
)
_EVAL_ROW = re.compile(_EVAL_ROW_PATTERN)

# Eval summary regex capture group indices (named for clarity).
RE_GROUP_TAG = 1
RE_GROUP_RATIO = 2
RE_GROUP_AUC = 3
RE_GROUP_ACC = 4
RE_GROUP_STD = 5

# Loss history comment block: rendered loss terms in first->last order.
LOSS_COMMENT_TERMS = ("csp", "namr", "c3l", "total")

# Loss history checkpoint dict key.
CKPT_KEY_LOSS_HISTORY = "loss_history"
CKPT_KEY_ARGS = "args"
CKPT_ARG_EPOCHS = "epochs"

# LaTeX template strings.
TEX_TABLE1_AUTOGEN_MARKER = "% ==== Table 1: Ablation (auto-generated) ===="
TEX_LOSS_CONVERGENCE_MARKER = "% ==== Loss convergence ===="
TEX_RATIO_NEGATIVE_REPLACEMENT = "n/a"  # Degenerate negative ratios: render as n/a.

# Loss convergence value display precision.
LOSS_COMMENT_PRECISION = 4

# LaTeX caption formatting.
TEX_CAPTION_TEMPLATE = (
    "\\caption{{Ablation study. {epochs} epochs, standardized ESM-2 features, "
    "seed {seed}. Cluster acc = few-shot logistic regression on K-means labels "
    "({repeats} repeats). PPI AUC over {pairs:,} pos / {pairs:,} neg pairs.}}"
)
TEX_TABLE_LABEL = "\\label{tab:ablation}"


# ============================================================================
# Core Functions
# ============================================================================

def parse_eval():
    """
    @brief   Parse eval_summary.txt into {tag: {ratio, auc, acc, std}}.

    @details Values are kept as their raw text (not re-parsed as floats) so the
             table reproduces the eval script's own formatting exactly, avoiding
             double-rounding artifacts.

    @return  Dictionary mapping row tag to dict of raw text values.
    @note    Returns empty dict if the eval summary file is missing on disk.
    """
    rows = {}
    path = config.result_path(ABLATION_SUBDIR, EVAL_SUMMARY_NAME)
    if not os.path.exists(path):
        return rows

    for line in open(path):
        match = _EVAL_ROW.match(line.strip())
        if match:
            rows[match.group(RE_GROUP_TAG)] = {
                "ratio": match.group(RE_GROUP_RATIO),
                "auc": match.group(RE_GROUP_AUC),
                "acc": match.group(RE_GROUP_ACC),
                "std": match.group(RE_GROUP_STD),
            }
    return rows


def losses(tag):
    """
    @brief   Return {loss_term: (first, last)} from a collected checkpoint.

    @param   tag   Checkpoint tag key in CKPT_FILES mapping.
    @return  Dict of {term_name: (float_first, float_last)} for each non-empty
             recorded loss term; None if the checkpoint file is missing on disk.
    """
    path = config.result_path(ABLATION_SUBDIR, CKPT_FILES[tag])
    if not os.path.exists(path):
        return None

    ck = torch.load(path, map_location="cpu", weights_only=False)
    history = ck.get(CKPT_KEY_LOSS_HISTORY, {})
    return {
        term: (float(np.asarray(v)[0]), float(np.asarray(v)[-1]))
        for term, v in history.items() if len(v)
    }


def epoch_budget():
    """
    @brief   Extract the epoch budget from the full checkpoint's recorded run args.

    @return  Integer number of training epochs used for the full ablation run.
    @note    Raises RuntimeError if the full checkpoint or its 'epochs' arg is missing.
    """
    path = config.result_path(ABLATION_SUBDIR, CKPT_FILES["full"])
    if not os.path.exists(path):
        raise RuntimeError(f"full checkpoint not found: {path}")

    args = torch.load(
        path, map_location="cpu", weights_only=False
    ).get(CKPT_KEY_ARGS, {})
    if CKPT_ARG_EPOCHS not in args:
        raise RuntimeError("full checkpoint args missing 'epochs'")
    return int(args[CKPT_ARG_EPOCHS])


def main():
    """
    @brief   Main entry point: parse eval summary + loss histories, emit
             a LaTeX replacement block for Table 1 and a loss-convergence
             comment block, write to final_tables.tex under RESULT_DIR.
    @return  None (prints LaTeX to stdout and writes .tex file).
    """
    ev = parse_eval()
    epochs = epoch_budget()

    out = []

    # --- Table 1 (ablation + baselines) replacement block ---
    out.append(TEX_TABLE1_AUTOGEN_MARKER)
    out.append("\\begin{table}[h]")
    out.append("\\centering")

    # Caption: epoch budget + seed from config, PPI pair count from config.
    out.append(TEX_CAPTION_TEMPLATE.format(
        epochs=epochs, seed=config.SEED, repeats=config.FEWSHOT_REPEATS,
        pairs=config.N_PPI_PAIRS,
    ))
    out.append(TEX_TABLE_LABEL)
    out.append("\\small")
    out.append("\\begin{tabular}{@{}lccc@{}}")
    out.append("\\toprule")
    out.append(
        "\\textbf{Variant} & \\textbf{Cluster Acc.} & "
        "\\textbf{PPI Ratio} & \\textbf{PPI AUC} \\\\"
    )
    out.append("\\midrule")

    # Render each ablation/baseline row in configured order.
    for tag, label in TABLE_ROWS:
        row = ev.get(tag)
        if row:
            # Negative ratios indicate degenerate baselines; show "n/a" not a
            # misleading negative multiplier that readers would misinterpret.
            ratio = (
                TEX_RATIO_NEGATIVE_REPLACEMENT
                if row["ratio"].startswith("-") else row["ratio"]
            )
            out.append(
                f"{label} & {row['acc']} $\\pm$ {row['std']} & "
                f"{ratio}$\\times$ & {row['auc']} \\\\"
            )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    out.append("")

    # --- Loss convergence comment block (LaTeX % comments) ---
    out.append(TEX_LOSS_CONVERGENCE_MARKER)
    for tag in ABLATION_TAGS:
        summary = losses(tag)
        if summary:
            parts = "; ".join(
                f"{term} {first:.{LOSS_COMMENT_PRECISION}f}->{last:.{LOSS_COMMENT_PRECISION}f}"
                for term, (first, last) in summary.items()
                if term in LOSS_COMMENT_TERMS
            )
            out.append(f"% {tag}: {parts}")

    # Write to disk and echo to stdout.
    text = "\n".join(out)
    target = config.result_path(FINAL_TABLES_TEX)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
