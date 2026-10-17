#!/usr/bin/env python3
"""
@file    102_known_edge_per_seed.py
@brief   Per-arm, per-seed known-edge AUC with dual SD-convention manifest.

Produces the machine-readable artifact backing Table 1 and Supplementary
Table S12 (known-edge AUROC across every archived training seed).  Why this
script exists: the published table had no regenerable source, which is how
one arm (CSP-only "0.514 +/- 0.007") ended up on the *population* SD
convention (ddof=0) while every other arm used the *sample* convention
(ddof=1).  The script here:

  * re-runs the identical probe from 50_eval_checkpoint.py (load_pairs +
    eval_embeddings) for each checkpoint,
  * records the per-seed AUC / PPI ratio / mean cosines,
  * emits both sd_ddof=1 AND sd_ddof=0 in the summary block, and
  * names the convention used in the manuscript explicitly so there is no
    future ambiguity.

Protocol is imported from 50_eval_checkpoint.py so the probe cannot drift
from the published evaluation.

Outputs:
  results/known_edge_multiseed.json

Run:
  GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/102_known_edge_per_seed.py

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics
"""

# ============================================================================
# File-level constants
# ============================================================================

# Basenames of the two sibling scripts that supply (a) the known-edge probe
# and (b) the HGT encoder + data-dict builder. Loaded dynamically because
# scripts/ is not an importable package.
EVAL_CHECKPOINT_MODULE_BASENAME = "50_eval_checkpoint.py"
PRETRAIN_MODULE_BASENAME = "40_pretrain.py"

# Basename of the ESM-augmented graph used by both the probe and the encoder.
GRAPH_ESM_BASENAME = "xena_graph_esm.pt"

# ---- HGT encoder hyperparameters (must match the pretraining config) ----
# NOTE: dimension is taken LIVE from config.EMBEDDING_DIM so this script
# tracks a config.py upgrade automatically; layers/heads are fixed because
# the published checkpoints were produced with exactly this architecture.
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# ---- PyTorch load() behaviour (single call site to audit) ----
TORCH_LOAD_MAP_LOCATION = "cpu"
TORCH_LOAD_WEIGHTS_ONLY = False
EMBEDDING_DEVICE = "cpu"  # CPU is deterministic; known-edge probe doesn't need GPU.

# ---- rounding precision (must match the digits shown in manuscript tables) ----
AUC_ROUND_DIGITS = 4
PPI_RATIO_ROUND_DIGITS = 3
COSINE_ROUND_DIGITS = 4
SUMMARY_MEAN_ROUND_DIGITS = 4
SUMMARY_SD_ROUND_DIGITS = 4

# ---- SD convention called out in the manuscript text ----
# ddof=1 = sample standard deviation (divisor n-1), used everywhere
# *except* the accidentally-different CSP-only line that motivated this audit.
MANUSCRIPT_SD_CONVENTION_TEXT = (
    "ddof=1 (sample standard deviation, denominator n-1)"
)
SUMMARY_SD_DDOF1 = 1  # sample
SUMMARY_SD_DDOF0 = 0  # population

# Protocol string printed into the payload so consumers can cross-check
# against 50_eval_checkpoint.py without opening that file.
PROTOCOL_DESCRIPTION = (
    "50_eval_checkpoint.py load_pairs/eval_embeddings "
    "(2,000 pos / 2,000 neg, seed 42)"
)

# Output basename for the machine-readable artifact.
OUTPUT_JSON_BASENAME = "known_edge_multiseed.json"

# JSON indentation: 1 space keeps the file diff-friendly.
JSON_OUTPUT_INDENT = 1

# Console / per-record template used when a checkpoint exists on disk but
# cannot be loaded by the 5,000-gene HGT encoder (e.g. 650M-wide
# checkpoints, or a mismatched graph realisation).
SKIP_NOTE_PREFIX = "not loadable into the 5,000-gene encoder: "
SKIP_LOG_TEMPLATE = "  {arm:14s} seed {seed:3d}  SKIPPED ({exc_name})"
MISSING_LOG_TEMPLATE = "  MISSING {rel}"

# ---- Archived checkpoints, grouped by ablation arm ----
# arm name -> (seed -> relpath under results/)
# Any rename here must be synchronised with results/checkpoint_registry.json.
ARMS = {
    "full": {
        42:  "ablation_real/full.pt",
        7:   "ablation_real/full_s7o.pt",
        123: "ablation_real/full_legacy_s123.pt",
        21:  "ablation_real/full_legacy_s21.pt",
        99:  "ablation_real/full_legacy_s99.pt",
    },
    "namr_only": {
        42:  "ablation_real/namr_only.pt",
        7:   "ablation_real/namr_only_s7o.pt",
        123: "server_cuda_namr5/namr_only_s123.pt",
        21:  "server_cuda_namr5/namr_only_s21.pt",
        99:  "server_cuda_namr5/namr_only_s99.pt",
    },
    "csp_only": {
        42:  "ablation_real/csp_only.pt",
        7:   "ablation_real/csp_only_s7.pt",
        123: "ablation_real/csp_only_s123.pt",
    },
    "csp_namr": {
        42:  "ablation_real/csp_namr_legacy_s42.pt",
        7:   "ablation_real/csp_namr_legacy_s7.pt",
        123: "ablation_real/csp_namr_legacy_s123.pt",
        21:  "ablation_real/csp_namr_legacy_s21.pt",
        99:  "ablation_real/csp_namr_legacy_s99.pt",
    },
    "namr_only_mps": {
        42:  "repro_plain/namr_only_plain_s42.pt",
        7:   "repro_plain/namr_only_plain_s7.pt",
        123: "repro_plain/namr_only_plain_s123.pt",
        21:  "repro_plain/namr_only_plain_s21.pt",
        99:  "repro_plain/namr_only_plain_s99.pt",
    },
    "full_mps": {
        42:  "repro_plain/full_plain_s42a.pt",
        7:   "repro_plain/full_plain_s7.pt",
        123: "repro_plain/full_plain_s123.pt",
    },
}

# ============================================================================
# Imports + dynamic sibling-module load
# ============================================================================
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_HERE = os.path.dirname(os.path.abspath(__file__))

# 50_eval_checkpoint.py :: ec.load_pairs / ec.eval_embeddings
_ec_spec = importlib.util.spec_from_file_location(
    "ec",
    os.path.join(_HERE, EVAL_CHECKPOINT_MODULE_BASENAME),
)
ec = importlib.util.module_from_spec(_ec_spec)
_ec_spec.loader.exec_module(ec)

# 40_pretrain.py :: pf.build_data_dict / pf.HGTEncoder
_pf_spec = importlib.util.spec_from_file_location(
    "pf",
    os.path.join(_HERE, PRETRAIN_MODULE_BASENAME),
)
pf = importlib.util.module_from_spec(_pf_spec)
_pf_spec.loader.exec_module(pf)

import torch


def embeddings_for(checkpoint_path):
    """
    @brief   Re-build the 5,000-gene HGT encoder and emit gene embeddings.

    Steps (byte-identical to the evaluation path in 50_eval_checkpoint.py):
      1. load xena_graph_esm.pt metadata + node features,
      2. build the HGT-friendly data-dict via pf.build_data_dict,
      3. instantiate HGTEncoder with config.EMBEDDING_DIM, 6 layers, 8 heads,
      4. load state_dict from ``checkpoint_path``,
      5. forward-pass on CPU and return the ``gene`` slice as numpy float32.

    @param  checkpoint_path  (str)  absolute path to a pretraining .pt
                                     checkpoint whose state_dict key is
                                     ``"model"``.
    @return numpy.ndarray  shape (n_genes, config.EMBEDDING_DIM) on CPU.
    @note   This function hard-requires the 5,000-gene ESM graph. 650M
            sub-graph checkpoints or patient-graph checkpoints will fail
            here with a shape-mismatch RuntimeError, which is caught by the
            caller and logged as SKIP.
    """
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_ESM_BASENAME),
        map_location=TORCH_LOAD_MAP_LOCATION,
        weights_only=TORCH_LOAD_WEIGHTS_ONLY,
    )
    dev = torch.device(EMBEDDING_DEVICE)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)

    x_dict = {k: data[k]["x"] for k in node_types}
    edge_index = {e: data[e]["edge_index"] for e in edge_types}

    # Architecture mirrors 40_pretrain.py exactly; dim tracks config live,
    # layers/heads pinned to the values used for every published checkpoint.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM,
        HGT_NUM_LAYERS,
        HGT_NUM_HEADS,
    ).to(dev)

    ckpt = torch.load(
        checkpoint_path,
        map_location=TORCH_LOAD_MAP_LOCATION,
        weights_only=TORCH_LOAD_WEIGHTS_ONLY,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # No grad during probe forward pass.
    with torch.no_grad():
        return model(x_dict, edge_index)["gene"].cpu().numpy()


def main():
    """
    @brief   Known-edge AUC for every archived checkpoint + cross-seed summary.

    Loads the 2,000-positive / 2,000-negative known-edge pairs once, then
    iterates arm -> seed:
      * missing file on disk -> "MISSING" console line, skip
      * checkpoint unloadable by the 5,000-gene encoder -> "SKIPPED" with
        exception class, payload entry with AUC=None and a human-readable note
      * success -> eval_embeddings, store 4 rounded scalars + file relpath

    Summary block per arm stores the sorted seed list, raw values, the
    arithmetic mean, and BOTH standard-deviation conventions so downstream
    consumers can match whichever reporting style they need.  The
    ``per_checkpoint`` sub-dict is emitted with stringified seed keys so
    JSON round-trips preserve seed order across tools that mangle numeric
    object keys.

    @return  None. Writes results/known_edge_multiseed.json.
    @note    The script never raises on a per-checkpoint failure; a single
             bad 650M checkpoint must not prevent the rest of the audit
             from completing.
    """
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_ESM_BASENAME),
        map_location=TORCH_LOAD_MAP_LOCATION,
        weights_only=TORCH_LOAD_WEIGHTS_ONLY,
    )
    pos_pairs, neg_pairs = ec.load_pairs(graph)

    # ---------- per-seed collection ----------
    per_seed_out = {}
    for arm_name, seed_map in ARMS.items():
        per_seed_out[arm_name] = {}
        for seed, rel_path in seed_map.items():
            full_path = os.path.join(config.PROJECT_ROOT, "results", rel_path)
            if not os.path.exists(full_path):
                print(MISSING_LOG_TEMPLATE.format(rel=rel_path))
                continue

            try:
                gene_embeddings = embeddings_for(full_path)
            except Exception as exc:
                # Shape-mismatch / bad-graph checkpoints land here. We
                # record the failure in the payload rather than aborting so
                # the final audit has a contiguous row for every seed.
                per_seed_out[arm_name][seed] = {
                    "file": rel_path,
                    "auc": None,
                    "note": SKIP_NOTE_PREFIX + type(exc).__name__,
                }
                print(SKIP_LOG_TEMPLATE.format(
                    arm=arm_name, seed=seed, exc_name=type(exc).__name__,
                ))
                continue

            ppi_ratio, pos_cos, neg_cos, auc = ec.eval_embeddings(
                gene_embeddings, pos_pairs, neg_pairs,
            )

            per_seed_out[arm_name][seed] = {
                "file": rel_path,
                "auc":     round(float(auc),     AUC_ROUND_DIGITS),
                "ppi_ratio": round(float(ppi_ratio), PPI_RATIO_ROUND_DIGITS),
                "pos_cos": round(float(pos_cos), COSINE_ROUND_DIGITS),
                "neg_cos": round(float(neg_cos), COSINE_ROUND_DIGITS),
            }
            print(f"  {arm_name:14s} seed {seed:3d}  AUC={auc:.4f}")

    # ---------- cross-seed summary block ----------
    arm_summary = {}
    for arm_name, per_seed_dict in per_seed_out.items():
        auc_values = [
            x["auc"] for x in per_seed_dict.values()
            if x.get("auc") is not None
        ]
        if not auc_values:
            continue  # no usable seeds for this arm -> skip the summary row

        arm_summary[arm_name] = {
            "seeds": sorted(int(k) for k in per_seed_dict),
            "values": auc_values,
            "mean": round(float(np.mean(auc_values)), SUMMARY_MEAN_ROUND_DIGITS),
            # ddof=1 (sample std): manuscript convention (n-1 denominator)
            "sd_ddof1": (
                round(float(np.std(auc_values, ddof=SUMMARY_SD_DDOF1)),
                      SUMMARY_SD_ROUND_DIGITS)
                if len(auc_values) > 1 else None
            ),
            # ddof=0 (population std): kept for the CSP-only "0.514 +/- 0.007"
            # line that originally used this convention; preserved so the
            # table reconstruction can match BOTH versions.
            "sd_ddof0": (
                round(float(np.std(auc_values, ddof=SUMMARY_SD_DDOF0)),
                      SUMMARY_SD_ROUND_DIGITS)
                if len(auc_values) > 1 else None
            ),
        }

    payload = {
        "protocol": PROTOCOL_DESCRIPTION,
        "sd_convention_used_in_manuscript": MANUSCRIPT_SD_CONVENTION_TEXT,
        "per_arm": arm_summary,
        # Stringify seed keys for JSON portability (YAML/XML bridges don't
        # always preserve numeric dict keys).
        "per_checkpoint": {
            arm: {str(seed): record for seed, record in per_seed_dict.items()}
            for arm, per_seed_dict in per_seed_out.items()
        },
    }

    output_path = config.result_path(OUTPUT_JSON_BASENAME)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=JSON_OUTPUT_INDENT)

    # ---- human-readable console summary ----
    print()
    for arm_name, s in arm_summary.items():
        print(
            f"{arm_name:14s} n={len(s['values'])} {s['values']} "
            f"mean={s['mean']} sd1={s['sd_ddof1']} sd0={s['sd_ddof0']}"
        )
    print("[written]", output_path)


if __name__ == "__main__":
    main()
