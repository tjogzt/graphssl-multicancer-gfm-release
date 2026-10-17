#!/usr/bin/env python3
"""
@file    130_kegg_per_seed.py
@brief   Per-arm, per-seed KEGG pathway-membership AUROC, closing review issue
         N1: main-text Table 1 quoted KEGG values whose seed support differed per
         arm while the caption implied a uniform multi-seed protocol.  The
         protocol here is BYTE-IDENTICAL to 76_kegg_membership_table.py (z-scored
         gene input, same pathway size filter, same K=20 seed-per-pathway draws)
         so per-seed values are directly comparable with the published seed-42
         table.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  results/ablation_real/*.pt + data/raw/kegg_2021_human.json + graph/ESM artifacts
Output: results/kegg_per_seed.json (per-checkpoint) + results/kegg_per_seed_by_arm.json
Run:    GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/130_kegg_per_seed.py
"""
import importlib.util
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
config.set_seed()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHWAY_SCRIPT_BASENAME = "71_pathway_membership.py"
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Artefact paths (resolved via config module -- no hard-coded absolute dirs).
GRAPH_BASENAME = "xena_graph_esm.pt"
ESM_BASENAME = "esm2_gene_embeddings.pt"
ABLATION_TAG = "ablation_real"
REPRO_PLAIN_TAG = "repro_plain"
PER_CKPT_OUT_REL = "kegg_per_seed.json"
PER_ARM_OUT_REL = "kegg_per_seed_by_arm.json"

# KEGG protocol matchers (values identical to 76_kegg_membership_table.py;
# constants are referenced via the helper module when applicable, but the
# explicit K=20 round precision is local because it is display-only).
KEGG_PROTOCOL_K = 20
KEGG_DISPLAY_PRECISION = 4
SD_ROUND_DIGITS = 4

# HGT architecture (shared with every checkpoint evaluated here; the helper
# function hard-codes these because 40_pretrain.py signatures require them).
HGT_HIDDEN_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Standardisation epsilon (per-dim std floor).
STANDARDIZE_EPS = 1e-8

# --- Arm definitions -------------------------------------------------------
# arm -> {seed: checkpoint file}.  Names follow Supplementary Table S23 where
# the decoder of every file was MEASURED on the checkpoint itself rather than
# inferred from the file name (see 132_audit_checkpoints.py for the audit).
# The five NAMR-only CUDA seeds behind Table 1's 0.831 +/- 0.038 live in
# results/server_cuda_namr5/ via relative paths; the same-named file in
# namr5_skip_decoder_DISCARDED/ is a DIFFERENT model and is never used.
ARMS = {
    # CUDA five-seed sets: server_cuda_namr5/ (2026-08-29 sequential CUDA replication)
    "namr_only": {
        42:  "namr_only.pt",
        7:   "namr_only_s7o.pt",
        123: "../server_cuda_namr5/namr_only_s123.pt",
        21:  "../server_cuda_namr5/namr_only_s21.pt",
        99:  "../server_cuda_namr5/namr_only_s99.pt",
    },
    "full": {
        42:  "full.pt",
        7:   "full_s7o.pt",
        123: "full_legacy_s123.pt",
        21:  "full_legacy_s21.pt",
        99:  "full_legacy_s99.pt",
    },
    "csp_only": {
        42:  "csp_only.pt",
        7:   "csp_only_s7.pt",
        123: "csp_only_s123.pt",
    },
    "csp_namr": {
        42:  "csp_namr_legacy_s42.pt",
        7:   "csp_namr_legacy_s7.pt",
        123: "csp_namr_legacy_s123.pt",
        21:  "csp_namr_legacy_s21.pt",
        99:  "csp_namr_legacy_s99.pt",
    },
    # Alternative same-arm runs (no headline value; included for full transparency).
    "namr_only_skipdiag": {7: "namr_only_s7.pt"},
    "csp_namr_alt":       {7: "csp_namr_s7.pt", 123: "csp_namr_s123.pt"},
    "full_alt":           {7: "full_s7.pt"},
    # Same-code MPS replication of NAMR-only (Supplementary Table S12 platform
    # comparison; not part of the headline CUDA seed set above).
    "namr_only_mps": {
        42:  "../repro_plain/namr_only_plain_s42.pt",
        7:   "../repro_plain/namr_only_plain_s7.pt",
        123: "../repro_plain/namr_only_plain_s123.pt",
        21:  "../repro_plain/namr_only_plain_s21.pt",
        99:  "../repro_plain/namr_only_plain_s99.pt",
    },
}

# Per-checkpoint dict stable keys.
PCK_FILE = "file"
PCK_KEGG = "kegg_k20"
PCK_N_PW = "n_pathways"
PCK_NOTE = "note"
NOTE_MISSING = "checkpoint not archived"

# Per-arm aggregate dict stable keys.
ARM_SEEDS_EVAL = "seeds_evaluated"
ARM_VALUES = "values"
ARM_MEAN = "mean"
ARM_SD1 = "sd_ddof1"
ARM_SD0 = "sd_ddof0"

# JSON output indentation.
JSON_INDENT_PER_CKPT = 1
JSON_INDENT_PER_ARM = 1

# Print formatting widths.
PRINT_ARM_WIDTH = 14
PRINT_SEED_WIDTH = 3
PRINT_FNAME_WIDTH = 26


def _load_helper_modules():
    """Import the pathway-membership helper and the HGT pretrain helper.

    @brief  Internal helper: scripts/ is not a Python package; we load siblings
            via importlib.util to keep the audit script self-contained.
    @return Tuple (pm_module, pf_module).
    """
    spec_pm = importlib.util.spec_from_file_location(
        "pm", os.path.join(SCRIPT_DIR, PATHWAY_SCRIPT_BASENAME),
    )
    pm = importlib.util.module_from_spec(spec_pm)
    spec_pm.loader.exec_module(pm)
    spec_pf = importlib.util.spec_from_file_location(
        "pf", os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf = importlib.util.module_from_spec(spec_pf)
    spec_pf.loader.exec_module(pf)
    return pm, pf


def embeddings_for(ckpt_path, pf):
    """Forward pass with z-scored gene input (IDENTICAL to pm.get_embeddings_8m).

    @brief  Load graph -> build data dict -> standardize gene features per-dim
            -> run HGT forward on CPU -> return gene embeddings numpy array.
    @param  ckpt_path   Absolute checkpoint path.
    @param  pf          Preloaded 40_pretrain module.
    @return float64 numpy array of shape ``[n_genes, HGT_HIDDEN_DIM]``.
    """
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_BASENAME),
        map_location="cpu", weights_only=False,
    )
    dev = torch.device("cpu")
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZE_EPS)
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy()


def main():
    """Evaluate every arm+seed pair, compute per-arm aggregates, write 2 JSON files.

    @brief  1. load helpers, load KEGG pathways (size filter via pm constants);
            2. for each (arm, seed, checkpoint): if file is archived, run the
               protocol and store the mean KEGG AUROC and n_pathways; else record
               a note; 3. compute per-arm mean + ddof=0/1 SDs over evaluated
               seeds; 4. write per_checkpoint JSON and per_arm JSON.
    @return None.
    """
    pm, pf = _load_helper_modules()
    kegg = json.load(open(os.path.join(config.DATA_RAW, pm.KEGG_FILE)))
    pathways = [
        (k, v) for k, v in kegg.items()
        if pm.PATHWAY_MIN_SIZE <= len(v) <= pm.PATHWAY_MAX_SIZE
    ]
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_BASENAME),
        map_location="cpu", weights_only=False,
    )
    gene_list = list(graph["meta"]["gene_list"])
    ablation_dir = config.result_path(ABLATION_TAG)

    per_ckpt = {}
    for arm, seedmap in ARMS.items():
        for seed, fname in seedmap.items():
            path = os.path.join(ablation_dir, fname)
            key = f"{arm}|s{seed}"
            if not os.path.exists(path):
                per_ckpt[key] = {
                    PCK_FILE: fname, PCK_KEGG: None, PCK_NOTE: NOTE_MISSING,
                }
                print(f"  [MISSING] {fname}")
                continue
            E = embeddings_for(path, pf)
            r, n = pm.eval_pathways(E, gene_list, pathways)
            auc_val = round(float(np.mean(r[KEGG_PROTOCOL_K])), KEGG_DISPLAY_PRECISION)
            per_ckpt[key] = {
                PCK_FILE: fname,
                PCK_KEGG: auc_val,
                PCK_N_PW: int(n),
            }
            print(
                f"  {arm:<{PRINT_ARM_WIDTH}} "
                f"seed {seed:>{PRINT_SEED_WIDTH}}  "
                f"{fname:<{PRINT_FNAME_WIDTH}} "
                f"KEGG K={KEGG_PROTOCOL_K} = {auc_val}"
            )

    by_arm = {}
    for arm in ARMS:
        values = [
            v[PCK_KEGG] for k, v in per_ckpt.items()
            if k.startswith(arm + "|") and v.get(PCK_KEGG) is not None
        ]
        if not values:
            continue
        seeds_eval = sorted([
            int(k.split("|s")[1]) for k, v in per_ckpt.items()
            if k.startswith(arm + "|") and v.get(PCK_KEGG) is not None
        ])
        aggregate = {
            ARM_SEEDS_EVAL: seeds_eval,
            ARM_VALUES: values,
            ARM_MEAN: round(float(np.mean(values)), KEGG_DISPLAY_PRECISION),
        }
        if len(values) > 1:
            aggregate[ARM_SD1] = round(float(np.std(values, ddof=1)), SD_ROUND_DIGITS)
            aggregate[ARM_SD0] = round(float(np.std(values, ddof=0)), SD_ROUND_DIGITS)
        else:
            aggregate[ARM_SD1] = None
            aggregate[ARM_SD0] = None
        by_arm[arm] = aggregate

    payload = {
        "K": KEGG_PROTOCOL_K,
        "n_pathways": int(n),
        "protocol": "identical to 76_kegg_membership_table.py",
        "per_checkpoint": per_ckpt,
        "per_arm": by_arm,
    }
    with open(config.result_path(PER_CKPT_OUT_REL), "w") as fh:
        json.dump(payload, fh, indent=JSON_INDENT_PER_CKPT)
    with open(config.result_path(PER_ARM_OUT_REL), "w") as fh:
        json.dump(by_arm, fh, indent=JSON_INDENT_PER_ARM)
    # English-only stdout summary.
    print(f"\npathways used: {n}")
    for arm, d in by_arm.items():
        print(
            f"  {arm:<{PRINT_ARM_WIDTH}} n={len(d[ARM_VALUES])} "
            f"seeds={d[ARM_SEEDS_EVAL]} mean={d[ARM_MEAN]} sd1={d[ARM_SD1]}"
        )
    print(f"[written] {config.result_path(PER_CKPT_OUT_REL)}")


if __name__ == "__main__":
    main()
