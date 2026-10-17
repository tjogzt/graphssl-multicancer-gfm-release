#!/usr/bin/env python3
"""
@file    76_kegg_membership_table.py
@brief   Per-arm KEGG pathway-membership AUROC table (external fixed-functional-
         annotation endpoint). Reuses 71_pathway_membership.py's exact protocol –
         z-scored gene input features, min_members=10, randn random baseline – so
         the numbers are directly comparable against the manuscript's published
         baselines of NAMR-only=0.839 / ESM-2=0.599 / PCA=0.779 / random=0.504.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup functional_analysis

Input:  data/raw/kegg_2021_human.json + checkpoints + graph/ESM artifacts.
Output: results/kegg_membership_table.json + printed per-arm AUROC table.
Run:    python scripts/76_kegg_membership_table.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Source / output artifact paths.
GRAPH = os.path.join(config.DATA_PROC, "xena_graph_esm.pt")
ESM = os.path.join(config.DATA_PROC, "esm2_gene_embeddings.pt")
OUT = config.result_path("kegg_membership_table.json")

# HGT encoder architecture hyper-parameters (mirrors 40_pretrain.py defaults).
HGT_EMBEDDING_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# KEGG pathway membership K-value (AUROC averaged over K nearest neighbours).
KEGG_K = 20

# Score rounding decimal places for the printed JSON table.
SCORE_DECIMALS = 4

# Output JSON indentation spaces.
JSON_INDENT = 1

# Epsilon for z-score normalisation of gene features (avoid div-by-zero).
Z_SCORE_EPS = 1e-8

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def load_helper_modules():
    """
    @brief   Dynamically load the 71_pathway_membership.py and 40_pretrain.py
             scripts as Python modules so their helpers can be reused without
             copy-pasting code.
    @return  Tuple (pm_module, pf_module).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec_pm = importlib.util.spec_from_file_location(
        "pm", os.path.join(here, "71_pathway_membership.py")
    )
    pm = importlib.util.module_from_spec(spec_pm)
    spec_pm.loader.exec_module(pm)

    spec_pf = importlib.util.spec_from_file_location(
        "pf", os.path.join(here, "40_pretrain.py")
    )
    pf = importlib.util.module_from_spec(spec_pf)
    spec_pf.loader.exec_module(pf)

    return pm, pf


def embeddings_for(ckpt_path, pf):
    """
    @brief   Forward pass of the HGT encoder with z-scored gene input features
             (exactly the same preprocessing as pm.get_embeddings_8m).
    @param   ckpt_path  str – filesystem path of the trained checkpoint to load.
    @param   pf         module – imported 40_pretrain.py namespace.
    @return  np.ndarray of shape (n_genes, d) with dtype float, HGT gene embeddings.
    """
    graph = torch.load(GRAPH, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    dev = torch.device(LOAD_MAP_LOCATION)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # z-score gene features (exact same preprocessing as 71_pathway_membership.py).
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + Z_SCORE_EPS)

    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)
    ck = torch.load(ckpt_path, map_location=LOAD_MAP_LOCATION,
                    weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy()


def random_init_hgt_embeddings(pf):
    """
    @brief   Forward pass of a FRESH (untrained) HGT encoder with z-scored gene
             inputs. This gives the random-init null baseline for KEGG AUROC.
    @param   pf  module – imported 40_pretrain.py namespace.
    @return  np.ndarray of shape (n_genes, d).
    """
    graph = torch.load(GRAPH, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    dev = torch.device(LOAD_MAP_LOCATION)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + Z_SCORE_EPS)
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)  # Fresh random init: do NOT load any state dict.
    model.eval()
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy()


def main():
    """
    @brief   Entry point: load helper modules, build the per-arm gene embedding
             matrix collection, run KEGG membership evaluation on every arm, and
             write the JSON report plus printed summary.
    @return  None
    """
    pm, pf = load_helper_modules()

    # Load KEGG pathways filtered by size constraints (reuses 71's constants).
    kegg = json.load(open(os.path.join(config.DATA_RAW, pm.KEGG_FILE)))
    pathways = [
        (k, v) for k, v in kegg.items()
        if pm.PATHWAY_MIN_SIZE <= len(v) <= pm.PATHWAY_MAX_SIZE
    ]

    graph = torch.load(GRAPH, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    gl = list(graph["meta"]["gene_list"])

    # Baseline feature sets: ESM-2 raw embeddings and PCA graph input features.
    e8 = torch.load(ESM, map_location=LOAD_MAP_LOCATION,
                    weights_only=TORCH_LOAD_WEIGHTS_ONLY)["embeddings"]
    esm = np.stack([e8[g] for g in gl]).astype(np.float64)
    pca = graph["gene"]["x"].numpy()

    # Checkpoint-based arms: each tag maps to a checkpoint file under results/.
    arms = {
        "namr_only": config.result_path("ablation_real", "namr_only.pt"),
        "full": config.result_path("ablation_real", "full.pt"),
        "csp_only": config.result_path("ablation_real", "csp_only.pt"),
        "csp_namr": config.result_path("ablation_real", "csp_namr_legacy_s42.pt"),
    }

    sets = {}
    for tag, ckpt in arms.items():
        sets[tag] = embeddings_for(ckpt, pf)

    # Random-init HGT baseline: fresh (untrained) encoder forward pass.
    sets["random_init_hgt"] = random_init_hgt_embeddings(pf)

    # Pure random gaussian baseline (matches 71_pathway_membership.py protocol).
    rng = np.random.RandomState(config.SEED)
    sets["random"] = rng.randn(*sets["namr_only"].shape)

    # Reference feature baselines.
    sets["esm_only"] = esm
    sets["pca_input"] = pca

    # Evaluate every embedding set via the shared KEGG protocol.
    table = {}
    for name, E in sets.items():
        r, n = pm.eval_pathways(E, gl, pathways)
        table[name] = round(float(np.mean(r[KEGG_K])), SCORE_DECIMALS)

    with open(OUT, "w") as fh:
        json.dump({"K": KEGG_K, "n_pathways": n, "rows": table}, fh, indent=JSON_INDENT)
    print(f"pathways used (size-filtered): {n}")
    for name, auc in table.items():
        print(f"  {name:16s} KEGG K={KEGG_K} AUROC = {auc}")
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
