#!/usr/bin/env python3
"""
@file    54_eval_esm650.py
@brief   Evaluate full-scale 650M GFM checkpoints by scoring gene embeddings on
         the PPI and cluster probes. Mode subset extracts the ESM-graph genes for
         a like-for-like comparison with the main ablation table; mode full
         evaluates the large graph's own gene pairs, so both embedding scales are
         covered.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --ckpt <path> (required); --mode {subset,full} (default subset); graphs from
        config.DATA_PROC (xena_graph_esm650_large.pt, xena_graph_esm.pt)
Output: stdout only
Run:    python scripts/54_eval_esm650.py --ckpt PATH [--mode full]
"""
import sys
import os
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import config
config.set_seed()

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Per-run artifact filenames (resolved under config.DATA_PROC).
XENA_GRAPH_ESM650_LARGE = "xena_graph_esm650_large.pt"  # 650M-regime large graph
XENA_GRAPH_ESM = "xena_graph_esm.pt"                    # ESM-regime graph

# HGT encoder geometry must match the pretraining run behind the checkpoint.
HGT_LAYERS = 6                                      # Default number of HGT layers
HGT_HEADS = 8                                       # Default number of attention heads

# Default CLI parameter defaults.
DEFAULT_EVAL_MODE = "subset"                        # Default evaluation mode
EVAL_MODE_SUBSET = "subset"                         # Subset mode identifier
EVAL_MODE_FULL = "full"                             # Full mode identifier

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
    @brief   Entry point: parse CLI args, load 650M GFM checkpoint and large graph,
             extract gene embeddings, then evaluate either (a) on the subset of
             genes present in the main ESM graph for apples-to-apples comparison
             (mode=subset) or (b) on the full 650M large-graph gene set
             (mode=full). Prints the consolidated evaluation summary line.
    @return  None (side effect: one evaluation summary line printed to stdout)
    @note    Gene features are always per-dimension z-scored before the encoder to
             match the pretraining preprocessing protocol used for the main GFM.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=[EVAL_MODE_SUBSET, EVAL_MODE_FULL],
                    default=DEFAULT_EVAL_MODE)
    args = ap.parse_args()

    # Dynamically load sibling scripts for shared metric/model functions
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    ev = load_mod("ev", os.path.join(scripts_dir, "50_eval_checkpoint.py"))
    pf = load_mod("pf", os.path.join(scripts_dir, "40_pretrain.py"))

    # Load the 650M-regime large heterogeneous graph
    lg = torch.load(
        os.path.join(config.DATA_PROC, XENA_GRAPH_ESM650_LARGE),
        map_location="cpu", weights_only=False
    )
    # Use CPU for evaluation to ensure reproducibility across hardware
    dev = torch.device("cpu")
    # Build data dictionaries and extract graph metadata
    data, node_types, edge_types, in_dims = pf.build_data_dict(lg, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Always per-dimension z-score gene features (standard pretraining protocol H1)
    g = x_dict["gene"]
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
    gl = list(lg["meta"]["gene_list"])
    tag = os.path.basename(args.ckpt).replace(".pt", "")

    if args.mode == EVAL_MODE_SUBSET:
        # --- Mode subset: align to main ESM graph for ablation-table comparison ---
        # Align the ESM graph onto the large graph's gene ordering, then probe only
        # that subset so results match the main ablation table numbers exactly.
        g2 = torch.load(
            os.path.join(config.DATA_PROC, XENA_GRAPH_ESM),
            map_location="cpu", weights_only=False
        )
        gl2 = list(g2["meta"]["gene_list"])
        # Map each ESM-graph gene to its position in the large-graph embedding matrix
        idx = [gl.index(x) for x in gl2]
        Z = z[idx]
        # Sample PPI pairs from the ESM graph and run both probes
        pos, neg = ev.load_pairs(g2)
        ratio, pc, nc, auc = ev.eval_embeddings(Z, pos, neg)
        acc, acc_std = ev.cluster_acc(Z)
        print(f"[{tag}_subset{len(gl2)}] PPI ratio={ratio:.3f}x "
              f"(pos {pc:.4f} / neg {nc:.4f}) | AUC={auc:.4f} | "
              f"cluster acc={acc:.4f} +/- {acc_std:.4f}")
    else:
        # --- Mode full: evaluate on the entire 650M large-graph gene set ---
        pos, neg = ev.load_pairs(lg, n_pairs=config.N_PPI_PAIRS)
        ratio, pc, nc, auc = ev.eval_embeddings(z, pos, neg)
        acc, acc_std = ev.cluster_acc(z)
        print(f"[{tag}_full{len(gl)}] PPI ratio={ratio:.3f}x "
              f"(pos {pc:.4f} / neg {nc:.4f}) | AUC={auc:.4f} | "
              f"cluster acc={acc:.4f} +/- {acc_std:.4f}")


if __name__ == "__main__":
    main()
