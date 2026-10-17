#!/usr/bin/env python3
"""
@file    52_eval_single_cancer.py
@brief   Evaluate an equal-data single-cancer control checkpoint by loading it
         into a fresh HGT encoder and scoring the gene embeddings on the same
         PPI and cluster probes used for the multi-cancer GFM, so the
         single-cancer control can be compared against the full model on
         identical metrics (raw large-graph or ESM-graph regimes).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --ckpt <path> (required); --graph <path> (required); --cancer (default BRCA);
        --standardize (flag, per-dimension standardize gene features);
        --multi (flag, skip single-cancer node filtering)
Output: stdout only
Run:    python scripts/52_eval_single_cancer.py --ckpt PATH --graph PATH
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.set_seed()

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# HGT encoder geometry must match the pretraining run behind the checkpoint.
HGT_LAYERS = 6                                      # Default number of HGT layers
HGT_HEADS = 8                                       # Default number of attention heads

# Default CLI parameter defaults.
DEFAULT_CANCER_TYPE = "BRCA"                        # Default cancer type

# Feature preprocessing constants.
FEATURE_STD_EPSILON = 1e-8                          # Numerical stability epsilon


def load_mod(name, path):
    """
    @brief   Dynamically load a Python module by absolute file path (code-reuse helper).
    @param   name : str
             Module name (used for importlib bookkeeping).
    @param   path : str
             Absolute path to the .py file to load.
    @return  module
             Loaded Python module object.
    @note    Uses importlib.util so metric/model functions can be shared between
             scripts without creating a rigid import-time dependency graph.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    """
    @brief   Entry point: parse CLI args, load single-cancer checkpoint and graph,
             optionally filter the graph to a single-cancer subgraph, run the
             shared PPI ratio/AUC and cluster accuracy probes, and print the
             evaluation summary line.
    @return  None (side effect: one evaluation summary line printed to stdout)
    @note    The --multi flag skips single-cancer graph filtering so this script
             can also evaluate multi-cancer checkpoints using the same evaluation
             pipeline for side-by-side comparison.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--cancer", default=DEFAULT_CANCER_TYPE)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--multi", action="store_true",
                    help="multi-cancer model (no filtering)")
    args = ap.parse_args()

    # Dynamically load sibling scripts for shared metric/model functions
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    ev = load_mod("ev", os.path.join(scripts_dir, "50_eval_checkpoint.py"))
    s10 = load_mod("s10", os.path.join(scripts_dir, "41_pretrain_single_cancer.py"))
    pf = load_mod("pf", os.path.join(scripts_dir, "40_pretrain.py"))

    # Load the source heterogeneous graph (may be full or single-cancer)
    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    if not args.multi:
        # Restrict graph to the single cancer's gene subgraph for equal-data control eval
        graph = s10.filter_single_cancer(graph, args.cancer)

    # Use CPU for evaluation to ensure reproducibility across hardware
    dev = torch.device("cpu")
    # Build data dictionaries and extract graph metadata
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Optional per-dimension gene feature standardization (protocol H1)
    if args.standardize and "gene" in x_dict:
        g = x_dict["gene"]
        # Per-dimension z-score standardization of gene features before encoder
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + FEATURE_STD_EPSILON)

    # Instantiate HGT encoder with matching geometry and load checkpoint weights
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_LAYERS, HGT_HEADS
    ).to(dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    # One full-graph forward pass in eval mode to extract gene embeddings
    with torch.no_grad():
        z = model(x_dict, ei_dict)["gene"].cpu().numpy()

    # Sample balanced PPI pairs and run the shared downstream probes
    pos, neg = ev.load_pairs(graph)
    ratio, pc, nc, auc = ev.eval_embeddings(z, pos, neg)
    acc, acc_std = ev.cluster_acc(z)
    tag = os.path.basename(args.ckpt).replace(".pt", "")
    # Print consolidated evaluation summary line (tag + PPI + AUC + cluster metrics)
    print(f"[{tag}] PPI ratio={ratio:.3f}x (pos {pc:.4f} / neg {nc:.4f}) "
          f"| AUC={auc:.4f} | cluster acc={acc:.4f} +/- {acc_std:.4f}")


if __name__ == "__main__":
    main()
