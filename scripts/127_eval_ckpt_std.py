#!/usr/bin/env python3
"""
@file    127_eval_ckpt_std.py
@brief   Evaluate a single GFM checkpoint's known-edge cosine probe and K-means
         recoverability with an EXPLICIT choice of gene-feature standardisation
         at inference. A model trained on standardised features can therefore be
         evaluated in-distribution (--standardize) or out-of-distribution (raw)
         without guessing the training-time space from the file name.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  --ckpt <path>; --data <graph> (default xena_graph_esm.pt);
        --standardize (flag) to z-score gene features before the forward pass;
        --tag <string> overrides the printed tag (default: checkpoint basename).
Output: stdout "[tag] ... AUC= ... cluster acc= ..." single summary line.
Run:    python scripts/127_eval_ckpt_std.py --ckpt <path> [--standardize]
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

import config

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"
EVAL_SCRIPT_BASENAME = "50_eval_checkpoint.py"

# Default graph basename (resolved against config.DATA_PROC when --data is not
# supplied by the user on the CLI).
DEFAULT_GRAPH_BASENAME = "xena_graph_esm.pt"

# Standardisation epsilon (per-dim std floor; matches training protocol).
STANDARDIZE_EPS = 1e-8

# HGT architecture (matches the checkpoints this script is used with; if a
# future checkpoint uses a different HGT size, pass the matching architecture
# by editing these constants -- CLI flags are deliberately NOT added to keep
# the audit footprint minimal and reproducible).
HGT_HIDDEN_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# torch.load map location (all eval runs are serialised on CPU to keep the
# cosine probe deterministic across backends).
LOAD_MAP_DEVICE = "cpu"

# Print / display numeric precision (kept short so the summary line fits within
# the manuscript audit logs).
RATIO_DISPLAY_DIGITS = 3
AUC_DISPLAY_DIGITS = 4
CLUSTER_DISPLAY_DIGITS = 4
DEFAULT_CKPT_TAG_REPLACEMENTS = ((".pt", ""),)


def _load_helper_modules():
    """Dynamically load 40_pretrain and 50_eval_checkpoint as modules.

    @brief  Internal helper: scripts/ is not a Python package, so each audit
            script uses importlib.util to treat sibling scripts as ad-hoc
            modules. Returns the tuple (pf_module, ev_module).
    @return (40_pretrain module, 50_eval_checkpoint module)
    """
    spec_pf = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf = importlib.util.module_from_spec(spec_pf)
    spec_pf.loader.exec_module(pf)

    spec_ev = importlib.util.spec_from_file_location(
        "ev",
        os.path.join(SCRIPT_DIR, EVAL_SCRIPT_BASENAME),
    )
    ev = importlib.util.module_from_spec(spec_ev)
    spec_ev.loader.exec_module(ev)
    return pf, ev


def main():
    """CLI entry point: parse args, evaluate, print a single-line summary.

    @brief  1. build argparse with 4 optional args; 2. load graph and PPI pairs
               via 50_eval_checkpoint.load_pairs(); 3. optionally standardize
               gene-node features; 4. load checkpoint weights into an HGT;
               5. run a single forward pass on CPU; 6. report (a) known-edge
               cosine-probe AUC via eval_embeddings and (b) K-means cluster
               recoverability via cluster_acc.
    @return None (process exits 0 via the __main__ guard).
    """
    pf, ev = _load_helper_modules()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint .pt file.")
    ap.add_argument(
        "--data",
        default=os.path.join(config.DATA_PROC, DEFAULT_GRAPH_BASENAME),
        help="Processed heterograph .pt file (default: %(default)s).",
    )
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score gene-node features (matching standardized-feature training runs).",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Printed tag (default: checkpoint basename with .pt stripped).",
    )
    args = ap.parse_args()

    graph = torch.load(args.data, map_location=LOAD_MAP_DEVICE, weights_only=False)
    pos_pairs, neg_pairs = ev.load_pairs(graph)
    dev = torch.device(LOAD_MAP_DEVICE)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    if args.standardize and "gene" in x_dict:
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZE_EPS)
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    )
    ckpt = torch.load(args.ckpt, map_location=LOAD_MAP_DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        X_gene = model(x_dict, ei_dict)["gene"].cpu().numpy()

    ratio, pos_mean, neg_mean, auc = ev.eval_embeddings(X_gene, pos_pairs, neg_pairs)
    acc_mean, acc_std = ev.cluster_acc(X_gene)

    tag = args.tag
    if tag is None:
        tag = os.path.basename(args.ckpt)
        for old, new in DEFAULT_CKPT_TAG_REPLACEMENTS:
            tag = tag.replace(old, new)
    # Single-line English summary used by manuscript audit logs.
    print(
        f"[{tag}] "
        f"PPI ratio={ratio:.{RATIO_DISPLAY_DIGITS}f}x "
        f"(pos {pos_mean:.{AUC_DISPLAY_DIGITS}f} / "
        f"neg {neg_mean:.{AUC_DISPLAY_DIGITS}f}) | "
        f"AUC={auc:.{AUC_DISPLAY_DIGITS}f} | "
        f"cluster acc={acc_mean:.{CLUSTER_DISPLAY_DIGITS}f} +/- "
        f"{acc_std:.{CLUSTER_DISPLAY_DIGITS}f}"
    )


if __name__ == "__main__":
    main()
