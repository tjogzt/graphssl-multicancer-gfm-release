#!/usr/bin/env python3
"""
@file    72_c3l_probe.py
@brief   C3L redundancy probe: checks whether the ESM-2 input features already
         encode each gene's top-1 cancer target, so the encoder task is merely
         recovering signal present in its input. Multinomial logistic regression
         and kNN classifiers are trained on standardized graph features (and raw
         ESM-2 features) to predict the top-1 cancer type, with a column-permuted
         random-feature control and a 1/n-class chance baseline.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  config.DATA_PROC/xena_graph_esm.pt, gene_cancer_targets.pt,
        esm2_gene_embeddings.pt (graph features, targets, raw ESM-2 embeddings)
Output: config.result_path(OUT_FILE) JSON plus the same payload on stdout
Run:    python scripts/72_c3l_probe.py
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
GRAPH_FILE = "xena_graph_esm.pt"
TARGETS_FILE = "gene_cancer_targets.pt"
ESM_FILE = "esm2_gene_embeddings.pt"
OUT_FILE = "c3l_probe.json"

# Base random seed (derived from config) for CV folds and paired splits.
RNG_SEED = config.SEED

# Cross-validation protocol parameters.
CV_KFOLD_N_SPLITS = 5
REPEATED_SPLIT_N_REPEATS = 30
REPEATED_SPLIT_TRAIN_RATIO = 0.7

# Classifier hyperparameters.
LR_REGULARIZATION_C = 1.0
LR_MAX_ITER_CV = 2000
KNN_NEIGHBORS_SMALL = 5
KNN_NEIGHBORS_LARGE = 15
KNN_METRIC_COSINE = "cosine"
KNN_METRIC_DEFAULT = "minkowski"  # default: Euclidean distance in sklearn

# Classifier name keys in the results dictionary.
CLF_KEY_LR = "lr"
CLF_KEY_KNN5 = "knn5"
CLF_KEY_KNN15 = "knn15"
CLF_KEY_KNN15_COSINE = "knn15cos"
RES_KEY_PREFIX_CV = "cv_"
RES_KEY_PREFIX_REP30 = "rep30_"
RES_KEY_RANDOM_CONTROL = "random_control_lr"
RES_KEY_CHANCE = "chance"
RES_KEY_CHANCE_PRESENT = "chance_among_present"
RES_KEY_N_GENES = "n_genes"
RES_KEY_N_CLASSES = "n_classes_present"

# Output JSON and stdout payload keys.
OUT_KEY_GRAPH_INPUT = "features_graph_encoder_input"
OUT_KEY_ESM_RAW = "features_esm320_raw_standardized"
OUT_KEY_PROTOCOL = "protocol"

# Protocol description string (appears in the output JSON payload for provenance).
PROTOCOL_DESCRIPTION_TEMPLATE = (
    "{kfold}-fold CV (seed {seed}) + {nrep} paired {train_pct} splits (seed {seed}); "
    "LR C={lr_c}; kNN cosine-free Euclidean on standardized features; "
    "random control = column-permuted features"
)

# JSON output indentation.
JSON_INDENT = 2

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Core Functions
# ============================================================================

def load():
    """
    @brief   Load graph features, ESM-2 raw embeddings, and cancer target labels.

    @return  Tuple of (X_graph, X_esm, y):
             - X_graph: standardized graph gene input features [n_genes x graph_dim]
             - X_esm: standardized raw ESM-2 320-dim embeddings [n_genes x 320]
             - y: integer array of top-1 cancer-type target per gene [n_genes]
    @note    Assumes graph gene_list order is consistent across all loaded artifacts;
             standardizes both feature sets for comparable classifier performance.
    """
    # Load graph: gene input features (PCA of ESM-2, the exact encoder input).
    g = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_FILE),
        map_location="cpu", weights_only=False
    )
    X_graph = g["gene"]["x"].numpy()

    # Load top-1 cancer-type targets per gene (class labels).
    t = torch.load(
        os.path.join(config.DATA_PROC, TARGETS_FILE),
        map_location="cpu", weights_only=False
    )
    y = t["targets"].numpy()

    # Load raw ESM-2 mean-pooled gene embeddings and stack in graph gene order.
    e = torch.load(
        os.path.join(config.DATA_PROC, ESM_FILE),
        map_location="cpu", weights_only=False
    )
    emb = e["embeddings"]
    genes = g["meta"]["gene_list"]
    X_esm = np.stack([emb[gene] for gene in genes])

    # Standardize raw ESM-2 features (z-score per dimension).
    X_esm = StandardScaler().fit_transform(X_esm)
    return X_graph, X_esm, y


def run_probe(X, y, name, n_repeats=REPEATED_SPLIT_N_REPEATS, kfold=CV_KFOLD_N_SPLITS):
    """
    @brief   Run the full C3L redundancy probe evaluation on a feature set.

    @details Evaluates three classifier types via two protocols:
             (1) 5-fold stratified CV with LR, kNN-5, kNN-15 (Euclidean).
             (2) 30 paired 70/30 holdout splits with LR, kNN-5, kNN-15 (cosine).
             (3) Random-feature control (row-permuted features) LR via both protocols.
             (4) Uniform chance baselines over all classes and present classes.

    @param   X          Feature matrix [n_samples x n_features].
    @param   y          Integer class label vector [n_samples].
    @param   name       Human-readable feature set name (currently unused internally).
    @param   n_repeats  Number of paired 70/30 split repeats.
    @param   kfold      Number of stratified CV folds.
    @return  Dictionary of results with accuracy mean/SD per classifier/protocol.
    @note    Uses deterministic paired split seeds (RNG_SEED + i) across feature sets.
    """
    rng = np.random.RandomState(RNG_SEED)
    res = {}

    # --- Protocol 1: K-fold stratified cross-validation ---
    # Stratified folds preserve the per-class distribution within each fold.
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=RNG_SEED)
    for clf_name, clf in [
        (CLF_KEY_LR, LogisticRegression(C=LR_REGULARIZATION_C, max_iter=LR_MAX_ITER_CV)),
        (CLF_KEY_KNN5, KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_SMALL)),
        (CLF_KEY_KNN15, KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_LARGE)),
    ]:
        accs = []
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X[te])))
        res[f"{RES_KEY_PREFIX_CV}{clf_name}"] = {
            "acc_mean": float(np.mean(accs)),
            "acc_sd": float(np.std(accs)),
        }

    # --- Protocol 2: 30 paired 70/30 splits, each with a fresh split seed ---
    for clf_name, clf in [
        (CLF_KEY_LR, LogisticRegression(C=LR_REGULARIZATION_C, max_iter=LR_MAX_ITER_CV)),
        (CLF_KEY_KNN5, KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS_SMALL)),
        (CLF_KEY_KNN15_COSINE, KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS_LARGE, metric=KNN_METRIC_COSINE)),
    ]:
        accs = []
        for i in range(n_repeats):
            tr, te = train_test_split(
                np.arange(len(y)), train_size=REPEATED_SPLIT_TRAIN_RATIO,
                stratify=y, random_state=RNG_SEED + i
            )
            clf.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X[te])))
        res[f"{RES_KEY_PREFIX_REP30}{clf_name}"] = {
            "acc_mean": float(np.mean(accs)),
            "acc_sd": float(np.std(accs)),
        }

    # --- Protocol 3: Random-feature control (row permutation destroys gene-feature link) ---
    perm = rng.permutation(X.shape[0])
    Xr = X[perm]

    # CV on permuted features (LR only, sufficient for control).
    accs_cv = []
    skf2 = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=RNG_SEED)
    for tr, te in skf2.split(Xr, y):
        clf = LogisticRegression(
            C=LR_REGULARIZATION_C, max_iter=LR_MAX_ITER_CV
        ).fit(Xr[tr], y[tr])
        accs_cv.append(accuracy_score(y[te], clf.predict(Xr[te])))

    # Repeated splits on permuted features.
    accs_rep = []
    for i in range(n_repeats):
        tr, te = train_test_split(
            np.arange(len(y)), train_size=REPEATED_SPLIT_TRAIN_RATIO,
            stratify=y, random_state=RNG_SEED + i
        )
        clf = LogisticRegression(
            C=LR_REGULARIZATION_C, max_iter=LR_MAX_ITER_CV
        ).fit(Xr[tr], y[tr])
        accs_rep.append(accuracy_score(y[te], clf.predict(Xr[te])))

    res[RES_KEY_RANDOM_CONTROL] = {
        "cv_mean": float(np.mean(accs_cv)),
        "cv_sd": float(np.std(accs_cv)),
        "rep30_mean": float(np.mean(accs_rep)),
        "rep30_sd": float(np.std(accs_rep)),
    }

    # --- Protocol 4: Uniform-chance baselines ---
    res[RES_KEY_CHANCE] = 1.0 / config.CLUSTER_K
    res[RES_KEY_CHANCE_PRESENT] = 1.0 / len(np.unique(y))
    res[RES_KEY_N_GENES] = int(len(y))
    res[RES_KEY_N_CLASSES] = int(len(np.unique(y)))
    return res


def main():
    """
    @brief   Main entry point: run C3L probe on graph input features and raw
             ESM-2 embeddings, save results JSON, and echo to stdout.
    @return  None (outputs JSON to stdout and writes results file).
    @note    Evaluates two feature sets in parallel: graph-encoder-input PCA features
             and raw 320-dim ESM-2 mean-pooled embeddings.
    """
    # Load all probe inputs.
    X_graph, X_esm, y = load()

    # Build provenance-aware protocol description string.
    protocol_desc = PROTOCOL_DESCRIPTION_TEMPLATE.format(
        kfold=CV_KFOLD_N_SPLITS, seed=RNG_SEED, nrep=REPEATED_SPLIT_N_REPEATS,
        train_pct=f"{int(REPEATED_SPLIT_TRAIN_RATIO * 100)}/{100 - int(REPEATED_SPLIT_TRAIN_RATIO * 100)}",
        lr_c=LR_REGULARIZATION_C,
    )

    # Assemble full output payload.
    out = {
        OUT_KEY_GRAPH_INPUT: run_probe(X_graph, y, "graph"),
        OUT_KEY_ESM_RAW: run_probe(X_esm, y, "esm"),
        OUT_KEY_PROTOCOL: protocol_desc,
    }

    # Write JSON to disk and echo to stdout.
    with open(config.result_path(OUT_FILE), "w") as f:
        json.dump(out, f, indent=JSON_INDENT)
    print(json.dumps(out, indent=JSON_INDENT))


if __name__ == "__main__":
    main()
