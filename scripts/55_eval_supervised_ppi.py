#!/usr/bin/env python3
"""
@file    55_eval_supervised_ppi.py
@brief   Supervised link-prediction evaluation (literature-comparable protocol):
         train a logistic regression on Hadamard (elementwise-product) pair features
         with 5-fold CV and report ROC-AUC plus AUPRC over the same positive/negative
         PPI pairs. The zero-shot cosine probe is the paper's primary metric; this
         supervised run is the absolute-level cross-check.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  checkpoints and graphs for each ablation row (JOBS list below); ESM-2 gene
        embeddings and the ESM-regime graph under config.DATA_PROC
Output: stdout only
Run:    python scripts/55_eval_supervised_ppi.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.set_seed()

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Results/ subdirectory that holds the ablation checkpoints.
ABLATION_SUBDIR = "ablation_real"                    # Ablation experiment subfolder

# Ablation checkpoint filenames under config.result_path(ABLATION_SUBDIR, ...).
NAMRO_CKPT = "namr_only.pt"                          # NAMR-only ablation checkpoint
FULL_CKPT = "full.pt"                                # Full GFM checkpoint (all objectives)
CSP_CKPT = "csp_only.pt"                             # CSP-only ablation checkpoint
CSP_NAMR_CKPT = "csp_namr_legacy_s42.pt"             # CSP+NAMR legacy checkpoint (s42)
CSP_NAMR_EQVOL_CKPT = "csp_namr_eqvol_s42.pt"        # CSP+NAMR equal-volume control
CSP_NAMR_REWIRED_CKPT = "csp_namr_rewired_s42.pt"    # CSP+NAMR rewired-graph control
SINGLE_BRCA_CKPT = "single_BRCA_esm.pt"              # Single-cancer BRCA control checkpoint

# Graph / embedding filenames under config.DATA_PROC.
XENA_GRAPH_ESM = "xena_graph_esm.pt"                 # ESM-regime graph (main ablation)
XENA_GRAPH_ESM_EQVOL = "xena_graph_esm_eqvol.pt"     # Equal-volume control graph
XENA_GRAPH_ESM_REWIRED = "xena_graph_esm_rewired.pt" # Rewired control graph
ESM2_GENE_EMBEDDINGS = "esm2_gene_embeddings.pt"     # Raw ESM-2 gene embeddings

# HGT encoder geometry must match the pretraining run behind each checkpoint.
HGT_LAYERS = 6                                       # Default number of HGT layers
HGT_HEADS = 8                                        # Default number of attention heads

# Supervised probe hyperparameters.
LR_MAX_ITER = 1000                                   # Max iterations for logistic regression
LR_REGULARIZATION_C = 1.0                            # Inverse regularization strength
CV_FOLDS = 5                                         # Number of cross-validation folds

# Feature preprocessing constants.
FEATURE_STD_EPSILON = 1e-8                           # Numerical stability epsilon


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


def get_emb(ckpt, graph_path):
    """
    @brief   Load a GFM checkpoint and extract its gene embeddings.
             Instantiates a fresh HGT encoder, loads pretrained weights, runs one
             full-graph forward pass in eval mode, and returns CPU numpy embeddings.
    @param   ckpt : str
             Path to the pretrained GFM checkpoint .pt file.
    @param   graph_path : str
             Path to the heterogeneous graph .pt file matching the checkpoint.
    @return  tuple of (embeddings, graph)
             - embeddings : numpy.ndarray of shape [num_genes, embedding_dim]
             - graph      : loaded raw heterogeneous graph dictionary
    @note    Gene features are always per-dimension z-scored before the encoder to
             match the pretraining preprocessing protocol.
    """
    # Dynamically load model builder from the pretraining script
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    ev = load_mod("pf", os.path.join(scripts_dir, "40_pretrain.py"))
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    # Build data dictionaries and extract graph metadata
    data, node_types, edge_types, in_dims = ev.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Per-dimension z-score gene features (standard pretraining protocol H1)
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + FEATURE_STD_EPSILON)

    # Instantiate HGT encoder with matching geometry and load checkpoint weights
    model = ev.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_LAYERS, HGT_HEADS
    ).to(dev)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    # One full-graph forward pass in eval mode (no gradients)
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy(), graph


def sup_eval(X, pos, neg, name):
    """
    @brief   Supervised PPI link-prediction evaluation: Hadamard-pair features
             -> 5-fold CV logistic regression -> report ROC-AUC + AUPRC.
             The Hadamard product (elementwise multiplication) of the two node
             embeddings serves as the per-pair feature vector; a linear classifier
             trained on this representation yields literature-comparable numbers.
    @param   X : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   pos : list
             True PPI edge tuples (gene_id pairs).
    @param   neg : list
             Non-edge (negative) gene_id pairs (same length as pos).
    @param   name : str
             Display name for the evaluated model (used in the printed line).
    @return  None (side effect: one supervised-metric line printed to stdout)
    @note    Class probabilities are produced via cross_val_predict so every pair
             appears in the held-out fold exactly once; the output probabilities
             are fed directly to roc_auc_score and average_precision_score.
    """
    # Build Hadamard-product pair features for positive and negative PPI pairs
    A = X[np.array([a for a, _ in pos])] * X[np.array([b for _, b in pos])]
    B = X[np.array([a for a, _ in neg])] * X[np.array([b for _, b in neg])]
    # Stack features and build balanced binary class labels (1 = PPI edge, 0 = no edge)
    F = np.vstack([A, B])
    y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
    # Standardize Hadamard features for stable logistic regression
    Fs = StandardScaler().fit_transform(F)
    # 5-fold CV: obtain out-of-fold class probability estimates
    p = cross_val_predict(
        LogisticRegression(max_iter=LR_MAX_ITER, C=LR_REGULARIZATION_C),
        Fs, y, cv=CV_FOLDS, method="predict_proba"
    )[:, 1]
    # Compute and print supervised PPI link-prediction metrics
    auc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    print(f"{name:26s} supervised LR AUC={auc:.4f} | AUPRC={auprc:.4f}", flush=True)


def main():
    """
    @brief   Entry point: load the ESM-regime graph and sample balanced PPI pairs
             once, then iterate over all ablation rows (checkpoints and baselines),
             extract embeddings per row, and print supervised LR AUC + AUPRC for
             each. Includes three checkpoint-free baselines (random-init HGT,
             ESM-2-only, PCA input) alongside ablation checkpoints.
    @return  None (side effect: one supervised-metric line per ablation row printed
             to stdout in order, flushed immediately)
    @note    All rows use the same pre-sampled pos/neg pair set for strict pairwise
             comparability; checkpoint-free baselines skip get_emb() and instead
             build embeddings from scratch.
    """
    # Load main ESM-regime graph and sample shared pos/neg PPI pair set
    graph = torch.load(
        os.path.join(config.DATA_PROC, XENA_GRAPH_ESM),
        map_location="cpu", weights_only=False
    )
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    ev = load_mod("ev", os.path.join(scripts_dir, "50_eval_checkpoint.py"))
    pos, neg = ev.load_pairs(graph)

    # Evaluation job table: each row is (display_name, ckpt_path or None, graph_path or None)
    # Checkpoint-free baselines use None for both paths to trigger inline embedding generation.
    jobs = [
        ("NAMR-only (s42)",
         config.result_path(ABLATION_SUBDIR, NAMRO_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM)),
        ("Full (s42)",
         config.result_path(ABLATION_SUBDIR, FULL_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM)),
        ("CSP-only (s42)",
         config.result_path(ABLATION_SUBDIR, CSP_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM)),
        ("CSP+NAMR (s42)",
         config.result_path(ABLATION_SUBDIR, CSP_NAMR_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM)),
        ("Random-init HGT", None, None),
        ("ESM-2 only", None, None),
        ("PCA input", None, None),
        ("EqVol control (s42)",
         config.result_path(ABLATION_SUBDIR, CSP_NAMR_EQVOL_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM_EQVOL)),
        ("Rewired control (s42)",
         config.result_path(ABLATION_SUBDIR, CSP_NAMR_REWIRED_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM_REWIRED)),
        ("Single-cancer (s42)",
         config.result_path(ABLATION_SUBDIR, SINGLE_BRCA_CKPT),
         os.path.join(config.DATA_PROC, XENA_GRAPH_ESM)),
    ]

    # Evaluate each ablation row in table order
    for name, ckpt, gp in jobs:
        if ckpt is None:
            # --- Checkpoint-free baselines: build embeddings from scratch ---
            if name.startswith("Random-init"):
                # Random-initialized HGT encoder (no training, no checkpoint)
                spec = importlib.util.spec_from_file_location(
                    "pf2", os.path.join(scripts_dir, "40_pretrain.py"))
                pf2 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pf2)
                dev = torch.device("cpu")
                data, node_types, edge_types, in_dims = pf2.build_data_dict(graph, dev)
                x_dict = {nt: data[nt]["x"] for nt in node_types}
                ei_dict = {et: data[et]["edge_index"] for et in edge_types}
                # Fresh random HGT encoder (random weights, never trained)
                m2 = pf2.HGTEncoder(
                    node_types, edge_types, in_dims,
                    config.EMBEDDING_DIM, HGT_LAYERS, HGT_HEADS
                ).to(dev)
                m2.eval()
                with torch.no_grad():
                    X = m2(x_dict, ei_dict)["gene"].cpu().numpy()
            elif name.startswith("ESM-2"):
                # Raw PLM embeddings: index each graph gene by its gene_list entry
                e = torch.load(
                    os.path.join(config.DATA_PROC, ESM2_GENE_EMBEDDINGS),
                    map_location="cpu", weights_only=False
                )["embeddings"]
                X = np.stack([e[g] for g in graph["meta"]["gene_list"]]).astype(np.float64)
            else:
                # PCA-style input baseline: raw graph gene features (no model at all)
                X = graph["gene"]["x"].numpy()
        else:
            # --- Checkpoint-based row: load checkpoint via get_emb helper ---
            X, _ = get_emb(ckpt, gp)
        # Run supervised PPI link-prediction evaluation for this row
        sup_eval(X, pos, neg, name)


if __name__ == "__main__":
    main()
