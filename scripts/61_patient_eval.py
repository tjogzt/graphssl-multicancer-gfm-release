#!/usr/bin/env python3
"""
@file    61_patient_eval.py
@brief   Patient-level evaluation: run the patient-pretrained HGT encoder on the
         patient-gene graph and extract the learned patient embeddings. Reports
         the correlation between the first principal component of the learned
         patient embeddings and that of the input patient features; saves the
         embeddings for downstream few-shot cancer classification.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  config.DATA_PROC/xena_graph_patient_paad.pt          (patient-gene graph)
        config.RESULT_DIR/ablation_real/patient_paad.pt      (patient-pretrained checkpoint)
Output: stdout (embedding shape, NaN count, PC1 correlation) and
        config.RESULT_DIR/patient_paad_embeddings.npy        (saved patient embeddings)
Run:    python scripts/61_patient_eval.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved through config so paths stay env-overridable).
GRAPH_NAME = "xena_graph_patient_paad.pt"
CKPT_RELPATH = os.path.join("ablation_real", "patient_paad.pt")
EMBED_OUT_NAME = "patient_paad_embeddings.npy"

# HGT encoder architecture parameters (must match checkpoint exactly).
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Feature standardization epsilon for numerical stability.
STANDARDIZATION_EPS = 1e-8

# Correlation computation denominator epsilon (prevents division by zero).
CORRELATION_EPS = 1e-9

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Module-level pretraining module loader (shared across all functions).
# ============================================================================

# Load the pretraining module (HGTEncoder / build_data_dict) without importing
# it as a package. Executed once at import time for efficiency.
_spec = importlib.util.spec_from_file_location(
    "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


# ============================================================================
# Core Functions
# ============================================================================

def pc1(X):
    """
    @brief   Compute the first principal component of the input data matrix.

    @details Extracts the eigenvector of the covariance matrix corresponding
             to the largest eigenvalue via eigendecomposition.

    @param   X   Input data matrix [n_samples x n_features].
    @return  Projection of each sample onto the first PC [n_samples].
    @note    Assumes the covariance matrix is full rank; uses mean-centered data.
    """
    # Mean-center the data before eigendecomposition.
    Xc = X - X.mean(0)
    # Extract the eigenvector with the largest eigenvalue (last column from eigh).
    w = np.linalg.eigh(Xc.T @ Xc)[1][:, -1]
    # Project data onto the first principal component direction.
    return Xc @ w


def main():
    """
    @brief   Main entry point: extract patient embeddings, compute PC1
             correlation with input features, and save results.
    @return  None (outputs to stdout and saves .npy file).
    @note    Uses CPU-only inference to avoid GPU memory overhead for eval.
    """
    # Load patient-gene graph artifact.
    graph_path = os.path.join(config.DATA_PROC, GRAPH_NAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    # Inference on CPU for reproducibility and reduced memory footprint.
    dev = torch.device("cpu")

    # Build heterogeneous graph data dictionaries for encoder input.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Re-standardize gene features per dimension to match training preprocessing.
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

    # Initialize HGT encoder with checkpoint-matching architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(dev)

    # Load pretrained checkpoint weights.
    ckpt_path = config.result_path(CKPT_RELPATH)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])

    # Set model to evaluation mode and run forward pass without gradients.
    model.eval()
    with torch.no_grad():
        z = model(x_dict, ei_dict)

    # Extract patient embeddings and convert to numpy array.
    pz = z["patient"].cpu().numpy()
    print("patient embeddings:", pz.shape, "nan:", int(np.isnan(pz).sum()))

    # Extract raw input patient features for correlation comparison.
    px = graph["patient"]["x"].numpy()

    # Correlation between learned and input patient embeddings' first PC.
    # Center both PC projections before computing Pearson correlation.
    a, b = pc1(pz), pc1(px)
    a = a - a.mean()
    b = b - b.mean()

    # Compute normalized Pearson correlation coefficient.
    corr = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + CORRELATION_EPS))
    print("patient emb PC1 vs input PC1 corr:", round(corr, 4))

    # Save patient embeddings for downstream few-shot classification tasks.
    out = config.result_path(EMBED_OUT_NAME)
    np.save(out, pz)
    print("saved patient embeddings", pz.shape)


if __name__ == "__main__":
    main()
