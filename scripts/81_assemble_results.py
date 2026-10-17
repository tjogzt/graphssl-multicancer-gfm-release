#!/usr/bin/env python3
"""
@file    81_assemble_results.py
@brief   Assemble the final results summary (Markdown) from the machine artifacts
         in config.RESULT_DIR. Every number in the report is computed from a file
         on disk; nothing is transcribed by hand, so the summary cannot drift away
         from the runs. Sections include ablation evaluation, loss convergence,
         controls, GO enrichment, KEGG control, METABRIC validation, tissue PPI,
         PAAD transfer, and patient-level pretraining.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (merged from 81_assemble_results.py + 82_assemble_final_results.py, 2026-09)
@ingroup result_assembly_and_figures

Input:  config.RESULT_DIR artifacts (see section list in file docstring above)
Output: <RESULT_DIR>/final_results_summary.md and the same text on stdout
Run:    python scripts/81_assemble_results.py
"""
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact paths and regex patterns)
# ============================================================================

# Input/output filenames (resolved relative to config.RESULT_DIR).
EVAL_SUMMARY = "ablation_real/eval_summary.txt"
SUMMARY_OUT = "final_results_summary.md"

# Eval summary row regex pattern. Captures tag, pos/neg ratio components,
# PPI AUC, and cluster accuracy mean/SD. Group indices are hard-coded below.
EVAL_ROW_PATTERN = (
    r"^\[([^\]]+)\] PPI ratio=([0-9.\-]+)x \(pos ([0-9.\-]+) / neg ([0-9.\-]+)\)"
    r" \| AUC=([0-9.]+) \| cluster acc=([0-9.]+) \+/- ([0-9.]+)"
)
EVAL_ROW = re.compile(EVAL_ROW_PATTERN)

# Eval summary regex capture group indices (for clarity, avoid magic numbers).
RE_GROUP_TAG = 1
RE_GROUP_RATIO = 2
RE_GROUP_POS = 3
RE_GROUP_NEG = 4
RE_GROUP_AUC = 5
RE_GROUP_ACC = 6
RE_GROUP_ACC_STD = 7

# NPY statistics loader parameters.
NPY_STATS_DDOF_PAIRED = 1
NPY_STATS_DDOF_UNPAIRED = 0
NPY_MIN_ARRAY_SIZE = 2

# Result section filenames (glob patterns relative to RESULT_DIR).
GO_ENRICHMENT_GLOB = "go_enrichment_*.json"
METABRIC_AUC_GLOB = "metabric_*_auc_k50.npy"
PAAD_RHO_GLOB = "paad_rho_*.npy"

# Single-artifact filenames under RESULT_DIR.
KEGG_CONTROL_JSON = "kegg_cluster_control.json"
PATIENT_FEWSHOT_JSON = "patient_fewshot_acc.json"
TISSUE_PPI_SUBDIR = "tissue_ppi"
TISSUE_PPI_SUMMARY_TXT = "summary.txt"
PATIENT_LOG_FILENAME = "patient_paad.log"

# Checkpoint artifact glob for loss convergence section.
ABLATION_CKPT_GLOB = os.path.join("ablation_real", "*.pt")

# Control tag search substrings (any match marks the row as a control).
CONTROL_TAG_HINTS = ("single", "eqvol", "rewired", "control")

# KEGG JSON read truncation length for report inclusion.
KEGG_JSON_DISPLAY_LEN = 300

# Patient log search keywords for convergence tail line.
PATIENT_LOG_SEARCH_KEYWORDS = ("conv", "loss")

# Report formatting: Markdown table header fields.
MD_TABLE_ABLATION_COLS = [
    "tag", "cluster acc (mean +/- SD)", "PPI ratio", "PPI AUC"
]

# Console skipped-file prefix for the npy_stats() diagnostic echo.
STATS_SKIPPED_PREFIX = "  [skipped]"

# Final summary markdown title.
MD_TITLE = "# Final results summary"

# Artifacts disclaimer substring in the generated metadata line.
ARTIFACTS_DISCLAIMER = "(no hand-transcribed numbers)"

# Datetime isoformat timespec for generated timestamp.
DATETIME_TIMESPEC = "seconds"


# ============================================================================
# Utility Functions
# ============================================================================

def result_path(*parts):
    """
    @brief   Resolve a path or path segments under config.RESULT_DIR.

    @param   parts   One or more path parts (strings) joined with os.path.join.
    @return  Full absolute path under config.RESULT_DIR.
    @note    Thin wrapper around os.path.join(config.RESULT_DIR, *parts).
    """
    return os.path.join(config.RESULT_DIR, *parts)


# ============================================================================
# Section Loader Functions
# ============================================================================

def read_eval_summary():
    """
    @brief   Parse eval_summary.txt into a {tag: metrics_dict} mapping.

    @details Parses each line matching EVAL_ROW regex; preserves insertion order
             via dict iteration in Python 3.7+.

    @return  Dictionary mapping each row tag to a dict of parsed metrics.
    @note    Returns empty dict if the eval summary file is missing.
    """
    path = result_path(EVAL_SUMMARY)
    if not os.path.exists(path):
        return {}

    rows = {}
    with open(path) as handle:
        for line in handle:
            match = EVAL_ROW.match(line.strip())
            if match:
                rows[match.group(RE_GROUP_TAG)] = {
                    "ratio": float(match.group(RE_GROUP_RATIO)),
                    "pos": float(match.group(RE_GROUP_POS)),
                    "neg": float(match.group(RE_GROUP_NEG)),
                    "auc": float(match.group(RE_GROUP_AUC)),
                    "acc": float(match.group(RE_GROUP_ACC)),
                    "acc_std": float(match.group(RE_GROUP_ACC_STD)),
                }
    return rows


def loss_summary(ckpt_path):
    """
    @brief   Extract per-loss-term first/last/min values from a checkpoint.

    @param   ckpt_path   Path to a pretraining checkpoint .pt file.
    @return  Dict {loss_term: (first, last, min)} for each non-empty term recorded.
    @note    Loads entire checkpoint into memory; intended for small count of runs.
    """
    import torch
    history = torch.load(
        ckpt_path, map_location="cpu", weights_only=False
    ).get("loss_history", {})
    out = {}
    for key, values in history.items():
        if len(values):
            arr = np.asarray(values, dtype=float)
            out[key] = (float(arr[0]), float(arr[-1]), float(arr.min()))
    return out


def npy_stats(pattern, paired=True):
    """
    @brief   mean +/- SD over the per-split arrays matching a results/ glob pattern.

    @details For each matched file loads the npy array with allow_pickle=False
             first, retries allow_pickle=True on ValueError, coerces to flat
             float array. Arrays that fail interpretation or are scalar/empty
             are reported as skipped rather than silently contributing wrong stats.

    @param   pattern   Glob pattern relative to config.RESULT_DIR.
    @param   paired    If True use sample SD (ddof=1); if False population SD (ddof=0).
    @return  List of (basename, mean, sd, n) tuples. Skipped files are appended
             with (name, NaN, NaN, 0) and echoed to stdout with skip reason.
    @note    For paired splits (same data across methods) use paired=True for
             the unbiased sample estimate.
    """
    rows, skipped = [], []

    for path in sorted(glob.glob(result_path(pattern))):
        # --- Primary load path: allow_pickle=False for security ---
        try:
            arr = np.load(path, allow_pickle=False)
        except ValueError:
            # Fallback: local artifacts may have been pickled; trusted path.
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception as exc:
                skipped.append((os.path.basename(path), f"unreadable: {exc}"))
                continue

        # --- Numeric coercion ---
        try:
            arr = np.asarray(arr, dtype=float).ravel()
        except (TypeError, ValueError):
            skipped.append((os.path.basename(path), "non-numeric array"))
            continue

        # --- Size check ---
        if arr.size < NPY_MIN_ARRAY_SIZE:
            skipped.append(
                (os.path.basename(path), f"scalar/empty (size={arr.size})")
            )
            continue

        # Compute mean + SD with appropriate ddof.
        ddof = NPY_STATS_DDOF_PAIRED if paired else NPY_STATS_DDOF_UNPAIRED
        rows.append(
            (os.path.basename(path), float(arr.mean()),
             float(arr.std(ddof=ddof)), int(arr.size))
        )

    # Echo skipped entries to stdout so the report stays honest about gaps.
    for name, reason in skipped:
        rows.append((name, float("nan"), float("nan"), 0))
        print(f"{STATS_SKIPPED_PREFIX} {name}: {reason}")

    return rows


def go_enrichment_counts():
    """
    @brief   Count significantly enriched clusters per GO enrichment JSON file.

    @return  List of (json_basename, n_significant, n_total_clusters) tuples.
             On parse error the middle element is None and the last holds the
             error message prefixed with "unreadable: ".
    @note    Uses glob to discover all go_enrichment_*.json files in RESULT_DIR.
    """
    rows = []
    for path in sorted(glob.glob(result_path(GO_ENRICHMENT_GLOB))):
        try:
            payload = json.load(open(path))
            n_total = len(payload)
            n_sig = sum(
                1 for v in payload.values()
                if isinstance(v, dict) and v.get("n_sig", 0) > 0
            )
            rows.append((os.path.basename(path), n_sig, n_total))
        except Exception as exc:
            rows.append((os.path.basename(path), None, f"unreadable: {exc}"))
    return rows


def kegg_control():
    """
    @brief   Load and summarize the KEGG cluster control JSON (if present).

    @return  Short summary string with first ~300 chars or unreadable error;
             None if the file does not exist on disk.
    """
    path = result_path(KEGG_CONTROL_JSON)
    if not os.path.exists(path):
        return None
    try:
        payload = json.load(open(path))
        return f"{os.path.basename(path)}: {json.dumps(payload)[:KEGG_JSON_DISPLAY_LEN]}"
    except Exception as exc:
        return f"{os.path.basename(path)} unreadable: {exc}"


def tissue_summary():
    """
    @brief   Load the tissue-specific PPI summary.txt lines.

    @return  List of non-empty stripped strings; empty list if file missing.
    """
    path = result_path(TISSUE_PPI_SUBDIR, TISSUE_PPI_SUMMARY_TXT)
    if not os.path.exists(path):
        return []
    return [line.strip() for line in open(path) if line.strip()]


def patient_fewshot():
    """
    @brief   Format the patient few-shot accuracy series for Markdown display.

    @return  Formatted "K=value:accuracy" comma-separated string; None if file missing.
    """
    path = result_path(PATIENT_FEWSHOT_JSON)
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    pairs = ", ".join(
        f"K={k}:{a:.3f}" for k, a in zip(payload["K"], payload["acc"])
    )
    return pairs


def patient_loss_tail():
    """
    @brief   Return the last convergence/loss line of the PAAD patient-level log.

    @details Prefers the last line reporting a convergence or loss summary;
             falls back to the last non-empty line so the report never invents
             a number.

    @return  String line content; None if log is missing or completely empty.
    """
    path = result_path("ablation_real", PATIENT_LOG_FILENAME)
    if not os.path.exists(path):
        return None
    lines = [line.strip() for line in open(path) if line.strip()]
    # Reverse-scan for a loss/convergence line.
    for line in reversed(lines):
        lowered = line.lower()
        if any(kw in lowered for kw in PATIENT_LOG_SEARCH_KEYWORDS):
            return line
    return lines[-1] if lines else None


# ============================================================================
# Main Assembly
# ============================================================================

def main():
    """
    @brief   Assemble the final Markdown results report and write to disk.
    @return  None (prints report to stdout and saves final_results_summary.md).
    """
    out = []

    # --- Title and provenance header ---
    out.append(MD_TITLE)
    out.append("")
    out.append(
        f"Generated: {datetime.datetime.now().isoformat(timespec=DATETIME_TIMESPEC)} | "
        f"artifacts under `{config.RESULT_DIR}` {ARTIFACTS_DISCLAIMER}."
    )
    out.append("")

    # --- Section 1: Ablation + baseline evaluation ---
    ev = read_eval_summary()
    out.append("## 1. Ablation + baseline evaluation")
    if ev:
        out.append(
            "| tag | cluster acc (mean +/- SD) | PPI ratio | PPI AUC |"
        )
        out.append("|---|---|---|---|")
        for tag, m in ev.items():
            out.append(
                f"| {tag} | {m['acc']:.4f} +/- {m['acc_std']:.4f} | "
                f"{m['ratio']:.3f}x | {m['auc']:.4f} |"
            )
    else:
        out.append(f"- NOT FOUND: {result_path(EVAL_SUMMARY)}")
    out.append("")

    # --- Section 2: Loss convergence ---
    out.append("## 2. Loss convergence (first -> last, min)")
    ckpts = sorted(glob.glob(result_path(ABLATION_CKPT_GLOB)))
    if ckpts:
        for path in ckpts:
            try:
                summary = loss_summary(path)
            except Exception as exc:
                out.append(f"- {os.path.basename(path)}: unreadable ({exc})")
                continue
            rendered = ", ".join(
                f"{k}: {tuple(round(x, 4) for x in v)}"
                for k, v in summary.items()
            )
            out.append(f"- {os.path.basename(path)}: {rendered}")
    else:
        out.append("- NOT FOUND: no checkpoints under results/ablation_real/")
    out.append("")

    # --- Section 3: Controls (single-cancer / equal-volume / rewired) ---
    out.append("## 3. Controls (single-cancer / equal-volume / rewired)")
    control_tags = [
        t for t in ev if any(hint in t for hint in CONTROL_TAG_HINTS)
    ]
    if control_tags:
        for tag in control_tags:
            m = ev[tag]
            out.append(
                f"- {tag}: cluster acc {m['acc']:.4f} +/- {m['acc_std']:.4f} | "
                f"PPI AUC {m['auc']:.4f} | ratio {m['ratio']:.3f}x"
            )
    else:
        out.append("- no control tags found in eval_summary.txt")
    out.append("")

    # --- Section 4: GO cluster enrichment ---
    out.append("## 4. GO cluster enrichment")
    go_rows = go_enrichment_counts()
    if go_rows:
        for name, n_sig, n_total in go_rows:
            out.append(f"- {name}: {n_sig}/{n_total} clusters significantly enriched")
    else:
        out.append("- NOT FOUND: results/go_enrichment_*.json")
    out.append("")

    # --- Section 5: KEGG cluster control ---
    out.append("## 5. KEGG cluster control")
    out.append(f"- {kegg_control() or 'NOT FOUND: results/kegg_cluster_control.json'}")
    out.append("")

    # --- Section 6: METABRIC external validation ---
    out.append("## 6. METABRIC external validation (per-split AUROC, K=50)")
    metabric = npy_stats(METABRIC_AUC_GLOB)
    if metabric:
        for name, mean, sd, n in metabric:
            if n == 0:
                out.append(f"- {name}: not usable as a numeric array")
            else:
                out.append(f"- {name}: {mean:.4f} +/- {sd:.4f} (n={n} paired splits)")
    else:
        out.append("- NOT FOUND: results/metabric_*_auc_k50.npy")
    out.append("")

    # --- Section 7: Tissue-specific PPI ---
    out.append("## 7. Tissue-specific PPI (GTEx v8)")
    tissue = tissue_summary()
    if tissue:
        out.extend(f"- {line}" for line in tissue)
    else:
        out.append("- NOT FOUND: results/tissue_ppi/summary.txt")
    out.append("")

    # --- Section 8: PAAD held-out transfer ---
    out.append("## 8. PAAD held-out transfer (per-split Spearman rho)")
    paad = npy_stats(PAAD_RHO_GLOB)
    if paad:
        for name, mean, sd, n in paad:
            if n == 0:
                out.append(f"- {name}: not usable as a numeric array")
            else:
                out.append(f"- {name}: {mean:.4f} +/- {sd:.4f} (n={n} splits)")
    else:
        out.append("- NOT FOUND: results/paad_rho_*.npy")
    out.append("")

    # --- Section 9: Patient-level pretraining ---
    out.append("## 9. Patient-level pretraining")
    fewshot = patient_fewshot()
    out.append(
        f"- few-shot accuracy curve (from patient_fewshot_acc.json): "
        f"{fewshot or 'NOT FOUND'}"
    )
    tail = patient_loss_tail()
    out.append(f"- PAAD patient-level training log tail: {tail or 'NOT FOUND'}")

    # --- Write report to disk and echo to stdout ---
    text = "\n".join(out) + "\n"
    target = result_path(SUMMARY_OUT)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as handle:
        handle.write(text)
    print(text)
    print(f"[written] {target}")


if __name__ == "__main__":
    main()
