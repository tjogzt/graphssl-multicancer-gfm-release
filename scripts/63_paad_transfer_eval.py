#!/usr/bin/env python3
"""
@file    63_paad_transfer_eval.py
@brief   Held-out cancer (PAAD) few-shot transfer evaluation. The model was
         trained on 20 cancers and never saw PAAD; this script checks whether
         the learned gene embeddings let a few-shot regressor recover PAAD
         gene specificity (mean log1p TPM across patients) from real PAAD
         expression, reporting few-shot Spearman rho and a binary top/bottom
         AUROC against ESM-2 / raw-input / random baselines.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  CLI arg 1: GFM checkpoint (default config.RESULT_DIR/ablation_real/namr_only_20c.pt)
        CLI arg 2: graph (default config.DATA_PROC/xena_graph_esm_20c.pt)
        CLI arg 3: run tag (default "gfm20c")
        config.DATA_RAW/tcga/expression/TCGA-PAAD/.../Gene_Expression_Quantification
        config.DATA_PROC/esm2_gene_embeddings.pt           (ESM-2 8M baseline embeddings)
        config.CLOUD_DIR/esm2_650m_gene_embeddings.pt      (ESM-2 650M baseline embeddings)
Output: stdout only (matched-gene count, few-shot Spearman rho and binary AUROC tables)
Run:    python scripts/63_paad_transfer_eval.py [checkpoint.pt] [graph.pt] [tag]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved through config so paths stay env-overridable).
CKPT_RELPATH = os.path.join("ablation_real", "namr_only_20c.pt")
GRAPH_NAME = "xena_graph_esm_20c.pt"
ESM8M_EMB_NAME = "esm2_gene_embeddings.pt"
ESM650M_EMB_NAME = "esm2_650m_gene_embeddings.pt"

# PAAD per-sample expression TSVs relative path under config.DATA_RAW.
PAAD_EXPR_RELPATH = os.path.join(
    "tcga", "expression", "TCGA-PAAD", "Transcriptome_Profiling",
    "Gene_Expression_Quantification")

# Default run label for the GFM column in output tables.
DEFAULT_TAG = "gfm20c"

# Fixed seeds for deterministic baselines/splits (kept distinct from config.SEED
# so published control numbers reproduce unchanged).
RANDOM_BASELINE_SEED = 0
BINARY_AUROC_SEED = 7

# HGT encoder architecture parameters (must match checkpoint).
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8
HGT_EMBEDDING_DIM = 256

# Feature standardization epsilon for numerical stability.
STANDARDIZATION_EPS = 1e-8

# L2-normalization denominator epsilon (prevents division by zero).
NORM_EPS = 1e-9

# TSV file pattern for PAAD expression samples.
PAAD_TSV_GLOB_PATTERN = os.path.join("*", "*.tsv")

# Sample ID extraction: filename split delimiter and index.
SAMPLE_ID_SPLIT_DELIM = "."
SAMPLE_ID_SPLIT_INDEX = 0

# TSV comment character for header skipping.
TSV_COMMENT_CHAR = "#"

# TSV column names for gene name and expression value.
COL_GENE_NAME = "gene_name"
COL_TPM_UNSTRANDED = "tpm_unstranded"

# Log1p transform target name and standardization parameters.
SPECIFICITY_STANDARDIZE = True

# Few-shot Spearman regression evaluation K values.
FEWSHOT_SPEARMAN_K_VALUES = (10, 50, 200)

# Binary AUROC evaluation K (shots per class).
BINARY_AUROC_K = 50

# Binary AUROC logistic regression training iterations.
BINARY_LR_MAX_ITER = 500

# Ridge regression hyperparameters.
RIDGE_ALPHA = 1.0
RIDGE_SOLVER = "lsqr"

# Specificity threshold quantile for binary split (0.5 = median split).
SPECIFICITY_MEDIAN_Q = 0.5

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Resolve full paths at module load time.
# ============================================================================
GENE_TSV = os.path.join(config.DATA_RAW, PAAD_EXPR_RELPATH)


# ============================================================================
# Core Functions
# ============================================================================

def load_paad_expr():
    """
    @brief   Load PAAD per-sample TPM expression and aggregate into genes x patients matrix.

    @details Collapses multiple transcript rows per gene by taking max TPM,
             then assembles a single DataFrame across all samples.

    @return  DataFrame with genes as index (unique gene symbols) and samples as columns.
    @note    Uses max TPM aggregation per gene; keeps first occurrence on duplicate indices.
    """
    # Discover all per-sample TSV files under the PAAD expression directory.
    files = sorted(glob.glob(os.path.join(GENE_TSV, PAAD_TSV_GLOB_PATTERN)))
    mats = {}

    for f in files:
        # Extract sample ID from filename (portion before first dot delimiter).
        sid = os.path.basename(f).split(SAMPLE_ID_SPLIT_DELIM)[SAMPLE_ID_SPLIT_INDEX]

        # Load TSV with comment skipping and low-memory disabled for large files.
        df = pd.read_csv(f, sep="\t", comment=TSV_COMMENT_CHAR, low_memory=False)
        df = df[df[COL_GENE_NAME].notna()]

        # Collapse multiple transcript rows per gene by taking the max TPM value.
        mats[sid] = df.groupby(COL_GENE_NAME)[COL_TPM_UNSTRANDED].max()

    # Assemble per-sample series into a genes x samples DataFrame.
    expr = pd.DataFrame(mats)
    return expr[~expr.index.duplicated(keep="first")]


def get_embeddings(ckpt, graph_path, standardize=True):
    """
    @brief   Run the trained HGT encoder on the graph and return gene embeddings.

    @param   ckpt          Path to pretrained checkpoint file (.pt).
    @param   graph_path    Path to processed gene graph file (.pt).
    @param   standardize   Whether to re-standardize gene input features per dimension.
    @return  Tuple of (embeddings_array, graph_object):
             - embeddings_array: shape [n_genes, embedding_dim]
             - graph_object: loaded torch graph with metadata
    @note    Imports pretraining module dynamically; uses CPU for inference.
    """
    import importlib.util
    # Dynamically load pretraining module (HGTEncoder / build_data_dict).
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    # Load graph onto CPU for inference.
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")

    # Build heterogeneous graph input dictionaries.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Optionally re-standardize gene features per dimension to match training preprocessing.
    if standardize and "gene" in x_dict:
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

    # Initialize HGT encoder with checkpoint-matching architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(dev)

    # Load checkpoint weights and set evaluation mode.
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # Forward pass without gradient computation.
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy(), graph


def few_shot_spearman(X, y, K, n_splits=config.FEWSHOT_REPEATS, seed=config.SEED):
    """
    @brief   K-shot Ridge regression -> Spearman rho on held-out genes,
             averaged over repeated random splits.

    @param   X         Feature matrix [n_samples x n_features].
    @param   y         Target vector [n_samples].
    @param   K         Number of training samples (shots) per split.
    @param   n_splits  Number of repeated random splits to average over.
    @param   seed      Random seed for reproducible split generation.
    @return  Numpy array of Spearman rho values, one per split [n_splits].
    @note    Uses Ridge regression with fixed solver; test set is complement of training.
    """
    rng = np.random.RandomState(seed)
    rhos = []

    for _ in range(n_splits):
        # Sample K training indices without replacement.
        tr = rng.choice(len(y), min(K, len(y)), replace=False)
        # Test set = all indices not in training set.
        te = np.setdiff1d(np.arange(len(y)), tr)

        # Fit Ridge regressor on training split.
        clf = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER).fit(X[tr], y[tr])
        pred = clf.predict(X[te])

        # Compute Spearman rank correlation on held-out predictions.
        rhos.append(spearmanr(pred, y[te]).statistic)

    return np.array(rhos)


def main():
    """
    @brief   Main entry point: PAAD held-out transfer evaluation with
             multiple baselines across few-shot K values and binary AUROC.
    @return  None (outputs results table to stdout).
    @note    Evaluates GFM, ESM-8M (if available), ESM-650M, PCA, and random baselines.
    """
    # Parse CLI arguments with sensible defaults.
    ckpt = sys.argv[1] if len(sys.argv) > 1 else config.result_path(CKPT_RELPATH)
    graph_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(config.DATA_PROC, GRAPH_NAME)
    tag = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TAG

    # Extract GFM gene embeddings and graph metadata.
    emb, graph = get_embeddings(ckpt, graph_path)
    gl = list(graph["meta"]["gene_list"])

    # Load PAAD expression data and find shared gene universe.
    expr = load_paad_expr()
    common = [g for g in gl if g in expr.index]

    # Further restrict to genes covered by ESM-2 650M embeddings (shared baseline universe).
    e650keys = set(torch.load(
        os.path.join(config.CLOUD_DIR, ESM650M_EMB_NAME),
        map_location="cpu", weights_only=False
    )["embeddings"].keys())
    common = [g for g in common if g in e650keys]
    print("PAAD expr genes matched (all sources):", len(common))

    # Compute PAAD gene specificity: per-gene mean log1p TPM across all patients.
    Xe = np.log1p(expr.loc[common].values.astype(np.float64))
    spec = Xe.mean(axis=1)
    # Standardize specificity target for regression stability.
    spec = (spec - spec.mean()) / (spec.std() + STANDARDIZATION_EPS)

    # Map common genes back to graph-ordered indices for embedding lookup.
    idx = [gl.index(g) for g in common]
    E = emb[idx]

    # Build random baseline: unit-norm i.i.d. Gaussian features.
    rng = np.random.RandomState(RANDOM_BASELINE_SEED)
    Er = rng.randn(*E.shape)
    Er /= (np.linalg.norm(Er, axis=1, keepdims=True) + NORM_EPS)

    # Load ESM-8M baseline embeddings (skip if gene set exceeds coverage).
    e8 = torch.load(
        os.path.join(config.DATA_PROC, ESM8M_EMB_NAME),
        map_location="cpu", weights_only=False
    )["embeddings"]
    if all(g in e8 for g in common):
        E8 = np.stack([e8[g] for g in common]).astype(np.float64)
    else:
        E8 = None
        print("(esm8m baseline skipped: gene set exceeds 8M coverage)")

    # Load and project ESM-650M embeddings down to GFM dimension for fair comparison.
    e650 = torch.load(
        os.path.join(config.CLOUD_DIR, ESM650M_EMB_NAME),
        map_location="cpu", weights_only=False
    )["embeddings"]
    E650raw = np.stack([e650[g].astype(np.float64) for g in common])
    E650 = PCA(n_components=config.EMBEDDING_DIM, random_state=config.SEED).fit_transform(
        StandardScaler().fit_transform(E650raw))

    # Raw graph input features (PCA baseline).
    Epca = graph["gene"]["x"].numpy()[idx]

    # Binary target: above-median specificity genes (top half) vs. the rest.
    q = np.quantile(spec, [SPECIFICITY_MEDIAN_Q])
    yb = (spec > q[0]).astype(int)
    print(f"PAAD-specificity ranking: {len(common)} genes; binary split pos={yb.sum()}")

    # Assemble baseline list in evaluation order (GFM first, then baselines).
    base_list = [(tag, E), ("esm650", E650), ("pca", Epca), ("random", Er)]
    if E8 is not None:
        base_list.insert(1, ("esm8m", E8))

    # Evaluate each feature set across K values and binary AUROC.
    for name, X in base_list:
        # Standardize features for fair cross-method comparison.
        Xs = StandardScaler().fit_transform(X)

        # Few-shot Spearman rho across K values.
        for K in FEWSHOT_SPEARMAN_K_VALUES:
            rho = few_shot_spearman(Xs, spec, K)
            print(f"  {name:8s} K={K:4d}: Spearman rho={rho.mean():.4f} +/- {rho.std():.4f}")

        # Binary AUROC at K=50: balanced 50-shot logistic fit on specificity halves.
        aucs = []
        rng2 = np.random.RandomState(BINARY_AUROC_SEED)
        for _ in range(config.FEWSHOT_REPEATS):
            pos_i = np.where(yb == 1)[0]
            neg_i = np.where(yb == 0)[0]
            k = min(BINARY_AUROC_K, len(pos_i), len(neg_i))

            # Balanced training set: k positive + k negative samples.
            tr = np.concatenate([
                rng2.choice(pos_i, k, replace=False),
                rng2.choice(neg_i, k, replace=False)
            ])
            te = np.setdiff1d(np.arange(len(yb)), tr)

            # Train logistic regression and compute AUROC on held-out set.
            clf = LogisticRegression(max_iter=BINARY_LR_MAX_ITER).fit(Xs[tr], yb[tr])
            aucs.append(roc_auc_score(yb[te], clf.predict_proba(Xs[te])[:, 1]))

        print(f"  {name:8s} binary AUROC (K=50): {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
