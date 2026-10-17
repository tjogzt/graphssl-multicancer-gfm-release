#!/usr/bin/env python3
"""
@file    64_drug_response_gdsc.py
@brief   Drug response prediction on GDSC2 dataset. For each drug, target genes
         are parsed from PUTATIVE_TARGET and mapped to the ESM graph gene list;
         the drug feature is the mean embedding of its mapped target genes (GFM /
         ESM-only / random / raw PCA input). Few-shot Ridge regression predicts
         LN_IC50 over repeated random splits per K. Per-split R^2 arrays are saved
         for scripts/65_drug_response_stats.py, which computes absolute R^2,
         a bootstrap CI, and a paired permutation test.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  --ckpt        GFM checkpoint (None = random-init HGT)
        --data        config.DATA_PROC/xena_graph_esm.pt           (ESM-regime gene graph)
        --gdsc        config.DATA_RAW/gdsc/GDSC2_fitted_dose_response_24Jul22.csv
        --out_prefix  config.RESULT_DIR/drug2                       (output filename prefix)
        --K / --n_splits                                           (shot budgets / split count)
Output: stdout (mapped-drug and pair counts, per-K mean R^2 table) and
        <out_prefix>_{gfm,esm,random,pca}_r2_k50.npy               (per-split K=50 R^2)
Run:    python scripts/64_drug_response_gdsc.py [--ckpt ... --out_prefix ...]
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved through config so paths stay env-overridable).
GRAPH_NAME = "xena_graph_esm.pt"
ESM_EMB_NAME = "esm2_gene_embeddings.pt"
GDSC_RELPATH = os.path.join("gdsc", "GDSC2_fitted_dose_response_24Jul22.csv")
OUT_PREFIX = "drug2"

# Fixed seed for the random-features control (distinct from config.SEED so the
# deterministic Gaussian baseline reproduces the published numbers unchanged).
RANDOM_BASELINE_SEED = 0

# HGT encoder architecture default parameters.
HGT_EMBEDDING_DIM_DEFAULT = 256
HGT_NUM_LAYERS_DEFAULT = 6
HGT_NUM_HEADS_DEFAULT = 8

# L2-normalization denominator epsilon (prevents division by zero).
NORM_EPS = 1e-9

# PUTATIVE_TARGET string tokenization regex pattern.
TARGET_TOKEN_PATTERN = r"[;,/ ]+"

# GDSC drug target synonym normalization mappings.
GDSC_SYNONYM_ALIASES = {
    "C-MET": "MET",
    "C-MET.": "MET",
    "RAF": "RAF1",
    "MEK": "MAP2K1",
}

# Ridge regression hyperparameters.
RIDGE_ALPHA_DEFAULT = 1.0
RIDGE_SOLVER = "lsqr"

# Maximum test set size for each split (subsample for speed on large datasets).
MAX_TEST_SAMPLES_DEFAULT = 20000

# Minimum valid train/test set size (require at least 2 samples each for R^2).
MIN_SET_SIZE = 2

# CLI argument defaults.
CLI_DEFAULT_K_VALUES = "5,20,50,100,200"
CLI_K_SPLIT_DELIM = ","

# K value for which per-split R^2 arrays are saved to disk.
SAVED_K_VALUE = 50

# Saved output file suffix pattern.
OUTPUT_R2_SUFFIX = "_r2_k50.npy"

# GDSC environment variable for debugging with truncated dataset.
ENV_GDSC_MAX_PAIRS = "GDSC_MAX_PAIRS"

# GDSC CSV column names.
COL_DRUG_NAME = "DRUG_NAME"
COL_PUTATIVE_TARGET = "PUTATIVE_TARGET"
COL_LN_IC50 = "LN_IC50"

# Print formatting: table header separator length multiplier.
TABLE_SEP_MULTIPLIER = 14
TABLE_HEADER_PAD = 6

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Core Functions
# ============================================================================

def extract_gene_embeddings(
    graph,
    ckpt_path,
    hidden=HGT_EMBEDDING_DIM_DEFAULT,
    layers=HGT_NUM_LAYERS_DEFAULT,
    heads=HGT_NUM_HEADS_DEFAULT
):
    """
    @brief   Run the HGT encoder (trained or random-init) on the ESM graph
             and extract gene-level embeddings.

    @param   graph       Loaded torch graph object with heterogeneous data.
    @param   ckpt_path   Path to checkpoint (.pt); if None, use random init weights.
    @param   hidden      Embedding dimension (must match checkpoint if provided).
    @param   layers      Number of HGT layers (must match checkpoint).
    @param   heads       Number of attention heads (must match checkpoint).
    @return  Numpy array of gene embeddings [n_genes x hidden_dim].
    @note    Uses CPU-only inference for reproducibility.
    """
    import importlib.util
    # Dynamically load pretraining module (HGTEncoder / build_data_dict).
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    # Inference on CPU for reproducibility and cross-platform behavior.
    device = torch.device("cpu")

    # Build heterogeneous graph data dictionaries.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, device)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Initialize HGT encoder with specified architecture.
    model = pf.HGTEncoder(node_types, edge_types, in_dims, hidden, layers, heads).to(device)

    # Load pretrained weights if checkpoint provided; otherwise use random init.
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])

    # Set evaluation mode and run forward pass without gradients.
    model.eval()
    with torch.no_grad():
        z = model(x_dict, ei_dict)["gene"].cpu().numpy()
    return z


def parse_targets(s, g2i):
    """
    @brief   Parse a PUTATIVE_TARGET string and return graph gene indices for
             all mapped target genes (aliases resolved to canonical symbols).

    @param   s     Raw PUTATIVE_TARGET string from GDSC CSV.
    @param   g2i   Gene-symbol-to-index mapping dictionary.
    @return  List of unique integer gene graph indices (order-preserving).
    @note    Applies GDSC-specific synonym normalization before lookup.
    """
    idx = []
    # Split target string on standard delimiters (;, /, comma, space).
    for tok in re.split(TARGET_TOKEN_PATTERN, str(s).strip().upper()):
        tok = tok.strip()
        # Resolve known GDSC synonym aliases to canonical gene symbols.
        if tok in GDSC_SYNONYM_ALIASES:
            tok = GDSC_SYNONYM_ALIASES[tok]
        # Record index only if gene exists in graph universe.
        if tok in g2i:
            idx.append(g2i[tok])
    # Deduplicate while preserving first-occurrence order.
    return list(dict.fromkeys(idx))


def few_shot_ridge(
    X,
    y,
    K,
    n_splits=config.FEWSHOT_REPEATS,
    seed=config.SEED,
    alpha=RIDGE_ALPHA_DEFAULT,
    max_test=MAX_TEST_SAMPLES_DEFAULT
):
    """
    @brief   K-shot Ridge regression R^2 on held-out pairs, averaged over
             repeated random splits.

    @param   X         Feature matrix [n_samples x n_features].
    @param   y         Target vector [n_samples].
    @param   K         Number of training samples (shots) per split.
    @param   n_splits  Number of repeated random splits to evaluate.
    @param   seed      Random seed for reproducible split generation.
    @param   alpha     Ridge regularization strength (L2 penalty).
    @param   max_test  Maximum held-out test set size (subsample for speed).
    @return  Numpy array of R^2 values, one per valid split [n_valid_splits].
    @note    Skips splits with degenerate train/test sizes (< 2 samples each);
             R^2 is computed on sub-sampled test set for large datasets.
    """
    rng = np.random.RandomState(seed)
    r2s = []
    n = len(y)

    for _ in range(n_splits):
        # Sample K training indices without replacement.
        idx = rng.choice(n, min(K, n), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True

        # Skip degenerate splits where train or test set has fewer than 2 samples.
        if mask.sum() < MIN_SET_SIZE or (~mask).sum() < MIN_SET_SIZE:
            continue

        # Identify test indices; subsample if too large for efficiency.
        te_idx = np.where(~mask)[0]
        if len(te_idx) > max_test:
            te_idx = rng.choice(te_idx, max_test, replace=False)

        # Standardize features using only training set statistics.
        scaler = StandardScaler().fit(X[mask])
        Xtr, Xte = scaler.transform(X[mask]), scaler.transform(X[te_idx])

        # Fit Ridge regressor on standardized training split.
        clf = Ridge(alpha=alpha, solver=RIDGE_SOLVER).fit(Xtr, y[mask])
        pred = clf.predict(Xte)
        yte = y[te_idx]

        # Compute coefficient of determination (R^2) manually.
        ss_res = np.sum((yte - pred) ** 2)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    return np.array(r2s)


def main():
    """
    @brief   Main entry point: GDSC drug response few-shot prediction pipeline
             with multiple embedding baselines.
    @return  None (prints results table and saves per-split R^2 .npy files).
    @note    Supports CLI arguments for checkpoint, data paths, K values, splits.
    """
    # Parse command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="GFM checkpoint; None = random-init HGT")
    ap.add_argument("--data", default=os.path.join(config.DATA_PROC, GRAPH_NAME))
    ap.add_argument("--gdsc", default=os.path.join(config.DATA_RAW, GDSC_RELPATH))
    ap.add_argument("--out_prefix", default=os.path.join(config.RESULT_DIR, OUT_PREFIX))
    ap.add_argument("--K", default=CLI_DEFAULT_K_VALUES)
    ap.add_argument("--n_splits", type=int, default=config.FEWSHOT_REPEATS)
    args = ap.parse_args()

    # Set global random seed for full reproducibility.
    config.set_seed()

    # Load ESM graph and build gene-to-index mapping.
    graph = torch.load(args.data, map_location="cpu", weights_only=False)
    gene_list = list(graph["meta"]["gene_list"])
    g2i = {g: i for i, g in enumerate(gene_list)}

    # Load GDSC dose-response CSV; optionally truncate for debugging via env var.
    df = pd.read_csv(args.gdsc, low_memory=False)
    if os.environ.get(ENV_GDSC_MAX_PAIRS):
        df = df.head(int(os.environ[ENV_GDSC_MAX_PAIRS]))

    # Parse numeric LN_IC50 values and drop rows with missing response.
    df[COL_LN_IC50] = pd.to_numeric(df[COL_LN_IC50], errors="coerce")
    df = df.dropna(subset=[COL_LN_IC50])

    # Build drug -> target gene graph indices mapping.
    drug_targets = {}
    for drug, grp in df.groupby(COL_DRUG_NAME):
        idx = []
        for s in grp[COL_PUTATIVE_TARGET].dropna().unique():
            idx += parse_targets(s, g2i)
        drug_targets[drug] = list(dict.fromkeys(idx))

    # Filter to drugs with at least one mapped target gene.
    n_total_drugs = df[COL_DRUG_NAME].nunique()
    df = df[df[COL_DRUG_NAME].map(lambda d: len(drug_targets[d]) > 0)].copy()
    drugs = sorted(df[COL_DRUG_NAME].unique())
    print(f"drugs with mapped targets: {len(drugs)} / {n_total_drugs} (graph has {len(gene_list)} genes)")
    print(f"pairs: {len(df)}  (log-IC50 mean {df[COL_LN_IC50].mean():.3f}, std {df[COL_LN_IC50].std():.3f})")

    # Extract GFM embeddings and build baselines.
    X_gfm = extract_gene_embeddings(graph, args.ckpt)
    rng = np.random.RandomState(RANDOM_BASELINE_SEED)
    X_rand = rng.randn(*X_gfm.shape)
    X_rand = X_rand / (np.linalg.norm(X_rand, axis=1, keepdims=True) + NORM_EPS)

    # Build ESM-2 baseline embeddings.
    esm_emb = torch.load(
        os.path.join(config.DATA_PROC, ESM_EMB_NAME),
        map_location="cpu", weights_only=False
    )["embeddings"]
    X_esm = np.stack([esm_emb[g] for g in gene_list]).astype(np.float64)

    # Build PCA raw-input baseline from graph gene features.
    X_pca = graph["gene"]["x"].float().numpy()

    # Compute drug-level mean features for each embedding type.
    def drug_feats(X):
        return np.array([
            np.mean(X[drug_targets[d]], axis=0)
            if len(drug_targets[d]) else np.zeros(X.shape[1])
            for d in drugs
        ])

    # Assemble all feature sets.
    F = {
        "gfm": drug_feats(X_gfm),
        "esm": drug_feats(X_esm),
        "random": drug_feats(X_rand),
        "pca": drug_feats(X_pca),
    }

    # Build (drug, LN_IC50) pair vectors and drug-index mapping.
    pairs = [(r[COL_DRUG_NAME], float(r[COL_LN_IC50])) for _, r in df.iterrows()]
    y = np.array([v for _, v in pairs])
    D = np.array([drugs.index(d) for d, _ in pairs])

    # Parse K values from CLI string.
    Ks = [int(k) for k in args.K.split(CLI_K_SPLIT_DELIM)]

    # Run few-shot Ridge evaluation for all feature sets and K values.
    results = {name: {} for name in F}
    for name, fmat in F.items():
        X = fmat[D]
        for K in Ks:
            results[name][K] = few_shot_ridge(X, y, K, n_splits=args.n_splits)

    # Print formatted results table.
    print("\nK      | " + " | ".join(f"{name:>10}" for name in F))
    print("-" * (TABLE_HEADER_PAD + TABLE_SEP_MULTIPLIER * len(F)))
    for K in Ks:
        row = " | ".join(f"{results[name][K].mean():>10.4f}" for name in F)
        print(f"{K:5d}  | {row}")

    # Save K=50 per-split R^2 arrays for downstream statistical analysis.
    for name in F:
        np.save(f"{args.out_prefix}_{name}{OUTPUT_R2_SUFFIX}", results[name][SAVED_K_VALUE])
    print(f"\nsaved {args.out_prefix}_*{OUTPUT_R2_SUFFIX} (K={SAVED_K_VALUE}, n_splits={args.n_splits})")


if __name__ == "__main__":
    main()
