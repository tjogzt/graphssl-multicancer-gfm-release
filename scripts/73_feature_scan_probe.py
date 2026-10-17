#!/usr/bin/env python3
"""
@file    73_feature_scan_probe.py
@brief   Feature-content scan probe: measures how C3L-target recoverability degrades
         as feature information content is reduced. Variants span from raw ESM-2
         320-dim (standardised) through PCA-256 and PCA-64 down to a random 256-dim
         no-content control. Each variant is scored with a multinomial logistic
         regression over 30 paired 70/30 stratified splits.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  config.DATA_PROC/xena_graph_esm.pt, gene_cancer_targets.pt,
        esm2_gene_embeddings.pt (features, targets, raw ESM-2 embeddings).
Output: config.result_path(OUT_FILE) JSON plus the same payload on stdout.
Run:    python scripts/73_feature_scan_probe.py
"""
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
GRAPH_FILE = "xena_graph_esm.pt"         # Graph gene features (PCA-256 encoder input)
TARGETS_FILE = "gene_cancer_targets.pt"  # Top-1 cancer-type index per gene
ESM_FILE = "esm2_gene_embeddings.pt"     # Raw ESM-2 mean-pooled gene embeddings
OUT_FILE = "feature_scan_probe.json"     # Recoverability result per feature variant

# Base seed for paired stratified splits (each repeat adds its offset).
RNG = config.SEED

# Reduced-content PCA endpoint dimension.
PCA64_DIM = 64

# Number of repeated 70/30 stratified split evaluations.
REPEATS = 30

# Train fraction for the stratified split protocol.
TRAIN_FRACTION = 0.7

# Logistic regression regularisation strength (C).
LR_C = 1.0

# Logistic regression max iterations (high because feature dim can be large).
LR_MAX_ITER = 2000

# Output JSON indentation spaces.
JSON_INDENT = 2

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def run_lr(X, y, repeats=REPEATS):
    """
    @brief   Evaluate mean and standard deviation of classification accuracy for
             a multinomial logistic regression on the given features over N
             paired stratified 70/30 splits.
    @param   X        np.ndarray – (n_samples, n_features) feature matrix.
    @param   y        np.ndarray – (n_samples,) integer class labels.
    @param   repeats  int – number of stratified splits to average over
                      (default REPEATS).
    @return  Tuple (mean_accuracy, std_accuracy) as Python floats.
    @note    Splits are paired across feature variants: repeat i uses seed
             RNG + i, so the same train/test indices are reused for every
             variant, making the comparison more powerful.
    """
    accs = []
    for i in range(repeats):
        # Deterministic paired split: offset base seed by repeat index.
        tr, te = train_test_split(
            np.arange(len(y)),
            train_size=TRAIN_FRACTION,
            stratify=y,
            random_state=RNG + i,
        )
        clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER).fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    """
    @brief   Entry point: load graph features, targets, and raw ESM embeddings;
             build four feature variants of decreasing information content;
             evaluate LR recoverability for each; write the JSON report.
    @return  None
    """
    # Load source artefacts.
    g = torch.load(os.path.join(config.DATA_PROC, GRAPH_FILE),
                   map_location=LOAD_MAP_LOCATION,
                   weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    genes = g['meta']['gene_list']

    y = torch.load(os.path.join(config.DATA_PROC, TARGETS_FILE),
                   map_location=LOAD_MAP_LOCATION,
                   weights_only=TORCH_LOAD_WEIGHTS_ONLY)['targets'].numpy()

    e = torch.load(os.path.join(config.DATA_PROC, ESM_FILE),
                   map_location=LOAD_MAP_LOCATION,
                   weights_only=TORCH_LOAD_WEIGHTS_ONLY)

    # Stack raw 320-dim ESM-2 vectors in the graph's gene order, then standardise.
    X320 = np.stack([e['embeddings'][gene] for gene in genes])
    X320s = StandardScaler().fit_transform(X320)

    # Reduced-content endpoint: project the standardised 320-dim features to 64 dims.
    pca = PCA(n_components=PCA64_DIM).fit(StandardScaler().fit_transform(X320))
    X64 = StandardScaler().fit_transform(
        pca.transform(StandardScaler().fit_transform(X320))
    )

    # No-content endpoint: random gaussian features shaped like the encoder input.
    rng = np.random.RandomState(RNG)
    Xr = StandardScaler().fit_transform(
        rng.randn(X320.shape[0], g['gene']['x'].shape[1])
    )

    # Build the comparison matrix: four feature variants of decreasing information.
    out = {}
    for name, X in [
        ("esm320_raw_std", X320s),
        ("pca256_std", g['gene']['x'].numpy()),
        ("pca64_std", X64),
        ("random_std", Xr),
    ]:
        m, sd = run_lr(X, y)
        out[name] = {"lr_acc_mean": m, "lr_acc_sd": sd}

    # Chance-level baseline for the multi-class C3L target.
    out["chance"] = 1.0 / config.CLUSTER_K
    out["protocol"] = (
        f"{REPEATS} paired {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} splits "
        f"(seed {RNG}); LR C={LR_C}; features standardized; "
        f"target = C3L top-1 cancer type"
    )

    with open(config.result_path(OUT_FILE), "w") as f:
        json.dump(out, f, indent=JSON_INDENT)
    print(json.dumps(out, indent=JSON_INDENT))


if __name__ == "__main__":
    main()
