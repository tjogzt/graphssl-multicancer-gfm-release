#!/usr/bin/env python3
"""
@file    60_metabric_validation.py
@brief   External validation on METABRIC cohort: classify patient ER status
         (Positive/Negative) from expression-weighted GFM gene embeddings.
         METABRIC is an independent cohort on the Illumina microarray platform,
         so this tests whether the GFM gene embeddings (and the ESM-2 /
         random-feature baselines) transfer to a non-TCGA, non-RNA-seq platform
         via few-shot logistic regression over repeated splits.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  CLI arg 1: GFM checkpoint (default config.RESULT_DIR/ablation_real/full.pt)
        config.DATA_PROC/xena_graph_esm.pt                 (ESM-regime gene graph)
        config.DATA_PROC/esm2_gene_embeddings.pt           (ESM-2 baseline embeddings)
        config.DATA_RAW/metabric/data_mrna_illumina_microarray.txt  (expression matrix)
        config.DATA_RAW/metabric/data_clinical_sample.txt           (ER status labels)
Output: stdout (matched-gene count and per-method few-shot AUROC table) and
        config.RESULT_DIR/metabric_{gfm,esm,random}_auc_k50.npy  (per-split K=50 AUROC)
Run:    python scripts/60_metabric_validation.py [checkpoint.pt]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved through config so paths stay env-overridable).
CKPT_RELPATH = os.path.join("ablation_real", "full.pt")
GRAPH_NAME = "xena_graph_esm.pt"
ESM_EMB_NAME = "esm2_gene_embeddings.pt"
METABRIC_DIR = "metabric"
EXPRESSION_NAME = "data_mrna_illumina_microarray.txt"
CLINICAL_NAME = "data_clinical_sample.txt"
AUC_OUT_PREFIX = "metabric"

# Fixed seed for the random-features control. Kept distinct from config.SEED so the
# deterministic Gaussian baseline reproduces the published numbers unchanged.
RANDOM_BASELINE_SEED = 0

# HGT encoder architecture parameters (must match checkpoint).
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Feature standardization epsilon for numerical stability.
STANDARDIZATION_EPS = 1e-8

# Patient vector aggregation denominator epsilon (prevents division by zero).
PATIENT_VEC_EPS = 1e-9

# Logistic regression hyperparameters for few-shot classification.
LR_MAX_ITER = 500
LR_REGULARIZATION_C = 1.0

# Few-shot evaluation shot budgets (K values).
FEWSHOT_K_VALUES = (5, 20, 50)

# ER status label mapping (clinical annotations to binary integer labels).
ER_STATUS_MAPPING = {"Positive": 1, "Negative": 0}

# Clinical file header marker line prefix.
CLINICAL_HEADER_MARKER = "#Patient"


# ============================================================================
# Core Functions
# ============================================================================

def get_gene_embeddings(ckpt_path, graph_path):
    """
    @brief   Load the trained HGT encoder and extract gene embeddings.

    @param   ckpt_path   Path to the pretrained HGT checkpoint file (.pt).
    @param   graph_path   Path to the processed gene graph file (.pt).
    @return  Tuple of (embeddings_array, graph_object):
             - embeddings_array: numpy array of shape [n_genes, embedding_dim]
             - graph_object: loaded torch graph dictionary with metadata
    @note    Imports pretraining module dynamically to avoid circular dependencies;
             architecture dims (layers/heads) must match the checkpoint exactly.
    """
    import importlib.util
    # Dynamically load the pretraining module (HGTEncoder / build_data_dict)
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "40_pretrain.py"))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    # Load graph onto CPU for inference regardless of training device.
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")

    # Build data dictionaries from the heterogeneous graph.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Re-standardize gene features per dimension to match training preprocessing.
    if "gene" in x_dict:
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

    # Initialize HGT encoder with matching checkpoint architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(dev)

    # Load checkpoint weights and set model to evaluation mode.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # Forward pass: extract gene embeddings without gradient computation.
    with torch.no_grad():
        z = model(x_dict, ei_dict)["gene"].cpu().numpy()
    return z, graph


def patient_vectors(expr_df, emb, gene_list):
    """
    @brief   Build one embedding vector per patient sample using
             expression-weighted mean aggregation of gene embeddings.

    @details Each sample vector is computed as:
             sum_i(z_i * E_i) / sum_i |z_i|,
             where z_i is the z-scored expression of gene i in that sample
             and E_i its embedding. The |z| denominator keeps the result
             invariant to the expression dynamic range.

    @param   expr_df     DataFrame of expression values (genes x samples).
    @param   emb         Gene embedding matrix (n_graph_genes x embedding_dim).
    @param   gene_list   Ordered list of gene symbols matching embedding rows.
    @return  Tuple of (patient_matrix, common_genes):
             - patient_matrix: numpy array [n_samples x embedding_dim]
             - common_genes: list of gene symbols present in both inputs
    @note    Expression values are z-scored per sample (row-wise); zero-variance
             rows produce NaN which are replaced with 0.0.
    """
    # Build gene-to-index mapping for fast lookup.
    g2i = {g: i for i, g in enumerate(gene_list)}

    # Find intersection of genes present in both expression matrix and graph gene list.
    common = [g for g in expr_df.index if g in g2i]
    print(f"genes matched: {len(common)} / {len(gene_list)}")

    # Transpose to samples x genes matrix, cast to float64 for numerical precision.
    Xe = expr_df.loc[common].T.astype(np.float64)

    # Z-score each sample (row); zero-variance rows become NaN, set to 0.0.
    Xz = pd.DataFrame(
        np.apply_along_axis(lambda r: zscore(r, nan_policy="omit"), 1, Xe.values),
        index=Xe.index, columns=Xe.columns
    ).fillna(0.0)

    # Subset embedding matrix to matched genes.
    E = emb[np.array([g2i[g] for g in common])]

    # Weighted aggregation: z-scored expression weighted mean of gene embeddings.
    # Denominator uses absolute z-score sum to ensure invariance to dynamic range.
    P = (Xz.values @ E) / (np.abs(Xz.values).sum(axis=1, keepdims=True) + PATIENT_VEC_EPS)
    return P, common


def few_shot_auc(X, y, K, n_splits=config.FEWSHOT_REPEATS, seed=config.SEED):
    """
    @brief   Compute K-shot balanced logistic-regression AUROC averaged
             over repeated random train/test splits.

    @param   X           Feature matrix [n_samples x n_features].
    @param   y           Binary label vector [n_samples] (0/1).
    @param   K           Number of shots per class (train set size per class).
    @param   n_splits    Number of repeated random splits to average over.
    @param   seed        Random seed for reproducible split generation.
    @return  Numpy array of AUROC scores, one per split [n_splits].
    @note    Uses balanced sampling: K samples drawn without replacement from each class
             for training set; remaining samples form the test set.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_splits):
        # Identify positive and negative sample indices.
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]

        # Cap shot count at the smaller class size for balanced training set.
        k = min(K, len(pos), len(neg))

        # Sample balanced training set: K positive + K negative samples.
        tr = np.concatenate([
            rng.choice(pos, k, replace=False),
            rng.choice(neg, k, replace=False)
        ])

        # Build boolean train/test mask.
        mask = np.zeros(len(y), dtype=bool)
        mask[tr] = True

        # Train logistic regression classifier.
        clf = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_REGULARIZATION_C)
        clf.fit(X[mask], y[mask])

        # Predict probabilities on held-out set and compute AUROC.
        p = clf.predict_proba(X[~mask])[:, 1]
        aucs.append(roc_auc_score(y[~mask], p))

    return np.array(aucs)


def main():
    """
    @brief   Main entry point: run METABRIC ER status classification
             few-shot validation pipeline.
    @return  None (outputs results to stdout and saves .npy AUROC arrays.
    @note    Evaluates three embedding methods: GFM, ESM-2, and random Gaussian
             features across multiple K-shot budgets.
    """
    # Resolve checkpoint path from CLI argument or use default ablation model.
    ckpt = sys.argv[1] if len(sys.argv) > 1 else config.result_path(CKPT_RELPATH)
    graph_path = os.path.join(config.DATA_PROC, GRAPH_NAME)

    # Extract GFM gene embeddings and graph metadata.
    emb, graph = get_gene_embeddings(ckpt, graph_path)
    gl = list(graph["meta"]["gene_list"])

    # Load METABRIC expression: gene-symbol index, uppercase, first duplicate wins.
    expr_path = os.path.join(config.DATA_RAW, METABRIC_DIR, EXPRESSION_NAME)
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr.index = expr.index.str.upper()
    expr = expr[~expr.index.duplicated(keep='first')]

    # Clinical file has a comment header; locate the data header row.
    clin_path = os.path.join(config.DATA_RAW, METABRIC_DIR, CLINICAL_NAME)
    with open(clin_path) as _f:
        _lines = _f.readlines()
    _hdr = next(i for i, _l in enumerate(_lines) if _l.startswith(CLINICAL_HEADER_MARKER))
    clin = pd.read_csv(clin_path, sep='\t', skiprows=_hdr)
    clin.columns = [c.lstrip('#').strip() for c in clin.columns]

    # Build patient-level embedding vectors from GFM embeddings.
    P, common = patient_vectors(expr, emb, gl)

    # Map ER status to binary target and keep samples present in the expression matrix.
    clin["ER"] = clin["ER Status"].map(ER_STATUS_MAPPING)
    c = clin.dropna(subset=["ER"]).set_index("Sample Identifier")
    samples = [s for s in c.index if s in expr.columns]
    y = c.loc[samples, "ER"].values.astype(int)
    Xgfm = P[[list(expr.columns).index(s) for s in samples]]

    # Build baselines: i.i.d. Gaussian features (random control).
    rng = np.random.RandomState(RANDOM_BASELINE_SEED)
    Xrand = rng.randn(*Xgfm.shape)
    Xrand /= (np.linalg.norm(Xrand, axis=1, keepdims=True) + PATIENT_VEC_EPS)

    # Build ESM-2 baseline embeddings.
    esm_path = os.path.join(config.DATA_PROC, ESM_EMB_NAME)
    e = torch.load(esm_path, map_location="cpu", weights_only=False)
    emb_esm = np.stack([e["embeddings"][g] for g in common]).astype(np.float64)
    Xe = expr.loc[common].T.astype(np.float64)
    # Z-score ESM expression values per sample with NaN handling.
    Xz = pd.DataFrame(
        np.apply_along_axis(lambda r: zscore(r, nan_policy="omit"), 1, Xe.values),
        index=Xe.index, columns=Xe.columns
    ).fillna(0.0)
    # Weighted aggregation for ESM-2 patient vectors.
    P_esm = (Xz.values @ emb_esm) / (np.abs(Xz.values).sum(axis=1, keepdims=True) + PATIENT_VEC_EPS)
    Xesm = P_esm[[list(expr.columns).index(s) for s in samples]]

    # Print dataset summary statistics.
    print(f"METABRIC ER classification: n={len(y)} (pos={y.sum()}, neg={(1-y).sum()})")

    # Evaluate each embedding method across K-shot budgets.
    for name, X in [("gfm", Xgfm), ("esm", Xesm), ("random", Xrand)]:
        for K in FEWSHOT_K_VALUES:
            a = few_shot_auc(X, y, K)
            print(f"  {name:6s} K={K:3d}: AUROC={a.mean():.4f} +/- {a.std():.4f}")
        # Save per-split K=50 AUROC values for downstream statistical analysis.
        out = config.result_path(f"{AUC_OUT_PREFIX}_{name}_auc_k50.npy")
        np.save(out, few_shot_auc(X, y, 50))


if __name__ == "__main__":
    main()
