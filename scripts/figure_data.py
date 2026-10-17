#!/usr/bin/env python3
"""
@file    figure_data.py
@brief   Central data-access module for every value plotted in the GFM manuscript
         figures. Resolves all plotted scalars, vectors, and structured layouts
         from on-disk machine artifacts (evaluation summaries, checkpoint loss
         histories, provenance JSON files, pre-computed graph layouts). Figure
         plotting scripts MUST NOT restate any plotted number as a literal —
         this module is the single place where plotted values enter the
         plotting code, guaranteeing that figure updates automatically track
         the latest evaluation run.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalised 2026-09)
@ingroup core_visualization

Input:  Artifact files on disk under config.RESULT_DIR (or a user-supplied
        override path). Specifically the module reads eval_summary.txt,
        loss_summary.json, figS3_auc_provenance.json, patient_fewshot_acc.json,
        fig1_subgraph_layout.json, and arbitrary training checkpoint .pth files.
Output: Plain Python dicts and NumPy arrays — no plotting and no I/O side
        effects besides reading the inputs. Callers (85_make_figures.py and
        ad-hoc analysis scripts) are responsible for rendering.
Run:    Library module. Typical import:
            import figure_data as fd
            rows = fd.load_eval_summary()
            layout = fd.load_fig1_layout()
"""

import json
import os
import re
import sys

import numpy as np
import torch

# Resolve the sibling config module by prepending this script's directory to
# sys.path. This keeps imports working whether the module is invoked from the
# repo root or from inside scripts/ during an interactive session.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# Section 1: Centralised artifact file names (relative to config.RESULT_DIR)
# =============================================================================
# Every result file read by the figure pipeline is declared here as a named
# constant so the file-name contract between the evaluation scripts and the
# plotting scripts lives in exactly one place. Filenames follow the project's
# "00-description.ext" pattern (albeit without a numeric prefix because these
# are downstream output artifacts rather than pipeline steps).

# Main ablation-study outputs produced by 50_eval_checkpoint.py.
ARTIFACT_EVAL_SUMMARY = os.path.join("ablation_real", "eval_summary.txt")
ARTIFACT_LOSS_SUMMARY = os.path.join("ablation_real", "loss_summary.json")

# Supplementary-figure provenance files.
ARTIFACT_AUC_PROVENANCE = "figS3_auc_provenance.json"
ARTIFACT_PATIENT_ACC = "patient_fewshot_acc.json"

# Pre-computed graph layout for the heterogeneous-network schematic.
ARTIFACT_FIG1_LAYOUT = "fig1_subgraph_layout.json"

# =============================================================================
# Section 2: Parsing regular expressions (eval_summary.txt row format)
# =============================================================================
# The on-disk eval_summary.txt format written by 50_eval_checkpoint.py is:
#   [tag_string] ... | AUC=X.XXXXX | cluster acc=Y.YYYYY +/- Z.ZZZZZ
# The regex below extracts the four fields in a single match; we anchor to
# the start of the stripped line and allow arbitrary text between the tag
# and the first pipe so human-readable notes can be appended without breaking
# parsing.
EVAL_ROW_PATTERN = re.compile(
    r"^\[([^\]]+)\].*\| AUC=([0-9.]+) \| cluster acc=([0-9.]+) \+/- ([0-9.]+)"
)

# Named capture-group indices for EVAL_ROW_PATTERN (kept in sync with regex).
_EVAL_GROUP_TAG = 1
_EVAL_GROUP_AUC = 2
_EVAL_GROUP_ACC = 3
_EVAL_GROUP_STD = 4

# Loss-history key name inside checkpoint dictionaries. Must match the key
# written by the training loop (see trainer.save_checkpoint in scripts/train.py).
CKPT_LOSS_HISTORY_KEY = "loss_history"


def _default(name):
    """
    @brief   Private helper that resolves a result-directory artifact name to
             an absolute path via config.result_path(). Centralising this tiny
             wrapper avoids repeating "path or _default(FOO)" in every loader's
             signature and lets future refactors (e.g. adding a download-fallback
             for missing artifacts) touch one function.
    @param   name   Relative artifact path under RESULT_DIR.
    @return  Absolute file path string as produced by config.result_path(name).
    @note    This function intentionally does NOT check existence — the caller
             is responsible for raising a clear error if the file is missing,
             because different loaders want different error messages.
    """
    return config.result_path(name)


def load_eval_summary(path=None):
    """
    @brief   Parse the ablation-study eval_summary.txt flat-file produced by
             50_eval_checkpoint.py into a structured Python dictionary.

             Each row in the input encodes one experimental condition (tagged
             with a bracketed run-name) and its associated PPI-link AUC,
             cluster-classification accuracy, and few-shot SD.
    @param   path   Optional absolute path override. If None (default) the path
                    is resolved from RESULT_DIR / ARTIFACT_EVAL_SUMMARY via the
                    _default helper, allowing the GFM_RESULT_DIR env var to
                    redirect input.
    @return  Dict of the form {tag_string: {"auc": float, "acc": float, "std": float}}
             with one entry per successfully matched input row.
    @note    Raises RuntimeError if zero rows parse successfully — this usually
             means the file path is wrong, the file was produced by an older
             version of 50_eval_checkpoint.py with incompatible formatting, or
             the regex above (EVAL_ROW_PATTERN) has drifted from the writer.
    """
    # Resolve the input file path, falling back to the centralised default.
    path = path or _default(ARTIFACT_EVAL_SUMMARY)
    rows = {}
    # Read line-by-line; the file is small (~dozens of rows) so streaming is fine.
    with open(path) as handle:
        for line in handle:
            # Try to match the expected row format on the whitespace-stripped line.
            match = EVAL_ROW_PATTERN.match(line.strip())
            if match:
                # Extract the four captured fields; numeric fields converted to float.
                rows[match.group(_EVAL_GROUP_TAG)] = {
                    "auc": float(match.group(_EVAL_GROUP_AUC)),
                    "acc": float(match.group(_EVAL_GROUP_ACC)),
                    "std": float(match.group(_EVAL_GROUP_STD)),
                }
    # Defensive check: an empty result is almost always a user error, so fail
    # loudly rather than returning an empty dict that produces empty figures.
    if not rows:
        raise RuntimeError(f"no evaluation rows parsed from {path}")
    return rows


def load_loss_summary(path=None):
    """
    @brief   Load the JSON-encoded loss summary produced by the ablation
             driver. The summary records first / last / min values per loss
             term (NAMR, CSP, C3L) plus the run's CLI arguments so the plotting
             code can cross-reference hyper-parameters without re-running training.
    @param   path   Optional absolute path override. If None, loads RESULT_DIR
                    / ARTIFACT_LOSS_SUMMARY.
    @return  De-serialised JSON object (typically a dict with keys "per_run",
             "loss_terms", "args", etc. — the exact schema is defined by the
             ablation driver that writes the file).
    """
    with open(path or _default(ARTIFACT_LOSS_SUMMARY)) as handle:
        return json.load(handle)


def load_loss_history(ckpt_path):
    """
    @brief   Extract the per-epoch loss-history arrays from a saved training
             checkpoint .pth file. The checkpoint is loaded onto the CPU to
             avoid GPU memory pressure during figure generation (which usually
             runs on the same GPU server as training).
    @param   ckpt_path   Absolute path to a torch.save()d checkpoint dictionary.
                         The dictionary must contain the key CKPT_LOSS_HISTORY_KEY
                         ("loss_history") whose value is a dict mapping loss-term
                         names to plain Python lists of per-epoch scalar values.
    @return  Dict of the form {loss_term_name: 1D numpy float array}. Each array
             has length equal to the number of completed epochs recorded in the
             checkpoint.
    @note    weights_only=False is required because the checkpoint stores not
             only model weights but also the loss_history plain-Python list;
             torch.load with weights_only=True would reject non-tensor payloads.
             The caller is responsible for trusting ckpt_path as a local file.
    """
    # Load the checkpoint on CPU to keep figure-gpu memory usage at zero.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Extract the loss-history sub-dict; a missing key is a hard error because
    # callers depend on every entry to draw a curve.
    history = ckpt.get(CKPT_LOSS_HISTORY_KEY)
    if history is None:
        raise KeyError(f"no '{CKPT_LOSS_HISTORY_KEY}' key in checkpoint {ckpt_path}")
    # Convert each term's Python list to a NumPy float array so downstream
    # plotting code can use fancy indexing and broadcasting directly.
    return {key: np.asarray(values, dtype=float) for key, values in history.items()}


def sample_at_epochs(history, key, epochs):
    """
    @brief   Sub-sample a single loss-term time-series at a user-supplied list
             of 1-based epoch numbers, clamping out-of-range requests to the
             last available epoch. Used by the figure pipeline to align
             loss-history x-ticks with the specific checkpoint checkpoints
             that were actually evaluated (e.g. every 10 epochs rather than
             plotting all epochs of a 500-epoch run).
    @param   history   Dict of loss-term -> numpy arrays, as returned by
                       load_loss_history().
    @param   key       String key selecting which loss-term series to sample
                       from history.
    @param   epochs    Iterable of 1-based epoch integers. Order is preserved
                       in the output; the caller typically passes a sorted list.
    @return  1D numpy float array of the same length as epochs, with
             result[i] == history[key][epochs[i] - 1] (epochs clamped to the
             array bounds so a request for epoch 9999 on a 500-epoch run
             returns the value at epoch 500 rather than raising IndexError).
    """
    # Grab the raw time-series array for the requested loss term.
    series = history[key]
    # For each requested epoch index: convert to 0-based by subtracting 1,
    # then clamp to [0, len-1] to silently handle out-of-range epochs.
    return np.asarray([
        series[min(int(epoch) - 1, len(series) - 1)]
        for epoch in epochs
    ])


def load_auc_provenance(path=None):
    """
    @brief   Load the per-bar AUC provenance dictionary used by the
             skip-decoder diagnostic supplementary figure (Fig. S3). Each entry
             records not only the final AUC but also the checkpoint path and
             random seed used to produce it, so the figure caption can cross-
             reference specific training runs.
    @param   path   Optional absolute path override. Default: RESULT_DIR /
                    ARTIFACT_AUC_PROVENANCE.
    @return  De-serialised JSON object (dict-of-dicts; outer keys are the
             experimental condition tags matching load_eval_summary()).
    """
    with open(path or _default(ARTIFACT_AUC_PROVENANCE)) as handle:
        return json.load(handle)


def load_patient_acc(path=None):
    """
    @brief   Load the patient-level few-shot classification accuracy curve
             that backs Table S7 and the associated supplementary panel.
             Records accuracy as a function of shot count K so the paper can
             report how performance scales with the number of support patients.
    @param   path   Optional absolute path override. Default: RESULT_DIR /
                    ARTIFACT_PATIENT_ACC.
    @return  De-serialised JSON object containing the K-sweep accuracy vector,
             SD vector, and per-run provenance.
    """
    with open(path or _default(ARTIFACT_PATIENT_ACC)) as handle:
        return json.load(handle)


def load_fig1_layout(path=None):
    """
    @brief   Load the pre-computed 2D node layout for the heterogeneous-network
             schematic panel Fig. 1(a). The layout bundles {gene, cancer, patient}
             node positions, edge lists, and grouping metadata so the figure
             script can render the schematic reproducibly WITHOUT needing to
             load the multi-gigabyte full STRING graph.

             Layouts are generated offline by scripts/84_extract_fig1_layout.py
             which samples the TP53 one-hop neighbourhood, runs a spring layout
             at the configured SEED, and serialises the result as JSON. Storing
             the layout (rather than recomputing it every time) guarantees
             pixel-identical figures across rebuilds.
    @param   path   Optional absolute path override. Default: RESULT_DIR /
                    ARTIFACT_FIG1_LAYOUT.
    @return  De-serialised JSON object. Expected top-level keys: "nodes" (list
             of node dicts with "x", "y", "type"), "edges" (list of index pairs),
             and "meta" (source graph parameters and seed for reproducibility).
    """
    with open(path or _default(ARTIFACT_FIG1_LAYOUT)) as handle:
        return json.load(handle)
