#!/usr/bin/env python3
"""
@file    71_pathway_membership.py
@brief   Pathway membership prediction (gene-level functional benchmark). For each
         KEGG pathway a few-shot seed of members defines a centroid, every gene is
         ranked by cosine similarity to it, and the AUROC of held-out members vs
         non-members is reported. GFM (NAMR-only at 8M and 650M scales) is compared
         against ESM-8M, ESM-650, PCA, and random baselines.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  <mode> (positional arg: "8m" or "650m")
        8m: config.DATA_PROC/xena_graph_esm.pt, esm2_gene_embeddings.pt,
            config.result_path("ablation_real", CKPT_8M)
        650m: config.DATA_PROC/xena_graph_esm650_large.pt, config.CLOUD_DIR/esm2_650m_gene_embeddings.pt,
            config.result_path("ablation_real", CKPT_650M)
        KEGG gene sets config.DATA_RAW/kegg_2021_human.json
Output: stdout only (per-model AUROC mean +/- SD at K=5 and K=20)
Run:    python scripts/71_pathway_membership.py [8m|650m]
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (artifact names and protocol thresholds)
# ============================================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
KEGG_FILE = "kegg_2021_human.json"
GRAPH_FILE = "xena_graph_esm.pt"
GRAPH650_FILE = "xena_graph_esm650_large.pt"
ESM8M_FILE = "esm2_gene_embeddings.pt"
ESM650_FILE = "esm2_650m_gene_embeddings.pt"
CKPT_8M = "namr_only.pt"
CKPT_650M = "namr_only_650.pt"

# Protocol thresholds (kept as named constants; values unchanged).
PATHWAY_MIN_SIZE = 20    # Lower bound on pathway member count.
PATHWAY_MAX_SIZE = 300   # Upper bound on pathway member count.

# HGT encoder architecture parameters (8M and 650M scales share architecture).
HGT_EMBEDDING_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Feature standardization epsilon.
STANDARDIZATION_EPS = 1e-8

# L2-normalization denominator epsilon (prevents division by zero).
NORM_EPS = 1e-9

# Pathway evaluation default parameters.
EVAL_K_VALUES = (5, 20)
EVAL_N_SPLITS_DEFAULT = 20
EVAL_MIN_MEMBERS = 10  # Require min_members + max(K) to ensure split feasibility.

# Default mode string when no CLI argument is provided.
DEFAULT_MODE = "8m"
MODE_8M = "8m"
MODE_650M = "650m"

# Checkpoint subdirectory under RESULT_DIR.
CKPT_SUBDIR = "ablation_real"

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Core Functions
# ============================================================================

def get_embeddings_8m(ckpt, graph_path=None):
    """
    @brief   Run the 8M NAMR-only encoder and return gene embeddings plus gene list.

    @param   ckpt         Path to the 8M NAMR-only checkpoint.
    @param   graph_path   Path to 8M-scale graph file; uses default if None.
    @return  Tuple of (embeddings_array, gene_list):
             - embeddings_array: shape [n_genes x embedding_dim]
             - gene_list: ordered list of gene symbols
    @note    Imports pretraining module dynamically; CPU-only inference.
    """
    # Resolve default graph path if not explicitly provided.
    graph_path = graph_path or os.path.join(config.DATA_PROC, GRAPH_FILE)

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

    # Z-score gene features per dimension to match training preprocessing.
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

    # Initialize HGT encoder with checkpoint-matching architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(dev)

    # Load checkpoint weights and set evaluation mode.
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # Forward pass without gradient computation.
    with torch.no_grad():
        z = model(x_dict, ei_dict)["gene"].cpu().numpy()
    return z, list(graph["meta"]["gene_list"])


def eval_pathways(
    emb,
    gl,
    pathways,
    K_values=EVAL_K_VALUES,
    n_splits=EVAL_N_SPLITS_DEFAULT,
    seed=config.SEED,
    min_members=EVAL_MIN_MEMBERS
):
    """
    @brief   Mean AUROC of held-out pathway members vs non-members, averaged
             over multiple few-shot seed splits per pathway.

    @details For each pathway:
             (1) K seed members define a centroid by mean pooling.
             (2) All remaining members (held-out positives) + all non-members
                 (negatives) are scored by cosine similarity to the centroid.
             (3) AUROC is computed on held-out positives vs negatives.
             Results are averaged across splits and then across pathways.

    @param   emb           Embedding matrix [n_genes x n_dims].
    @param   gl            Ordered gene-symbol list matching embedding rows.
    @param   pathways      Iterable of (term_name, [gene_symbols]) tuples.
    @param   K_values      Tuple of few-shot seed sizes to evaluate.
    @param   n_splits      Number of random seed splits per pathway per K.
    @param   seed          Random seed for reproducible sampling.
    @param   min_members   Minimum pathway size (after graph overlap) required.
    @return  Tuple of (results_dict, n_evaluated_pathways):
             - results_dict: {K: [auroc_mean_per_pathway, ...]}
             - n_evaluated_pathways: int count of pathways that passed size filters
    @note    Row-normalizes embeddings so dot products equal cosine similarities.
    """
    # Build gene-name -> embedding-row-index mapping for fast lookup.
    g2i = {g: i for i, g in enumerate(gl)}

    # Row-normalize so dot products equal cosine similarities directly.
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + NORM_EPS)

    rng = np.random.RandomState(seed)
    results = {K: [] for K in K_values}
    n_path = 0

    for term, genes in pathways:
        # Keep only pathway members present in the embedding universe.
        mem = [g for g in genes if g in g2i]
        # Skip pathway if too few members to seed + evaluate any K value.
        if len(mem) < min_members + max(K_values):
            continue
        n_path += 1

        # Build member and non-member index arrays.
        mi = np.array([g2i[g] for g in mem])
        nonmem = np.setdiff1d(np.arange(len(gl)), mi)

        # Evaluate each K value independently.
        for K in K_values:
            aucs = []
            for _ in range(n_splits):
                # Sample K seed members; the rest are held-out positives.
                seed_idx = rng.choice(len(mi), K, replace=False)
                held = np.setdiff1d(mi, mi[seed_idx])

                # Compute few-shot centroid as mean of seed embeddings.
                centroid = embn[mi[seed_idx]].mean(0)

                # Score non-members (negatives) and held-out members (positives).
                scores_neg = embn[nonmem] @ centroid
                scores_pos = embn[held] @ centroid

                # Assemble labels and scores for AUROC computation.
                y = np.concatenate([np.ones(len(held)), np.zeros(len(nonmem))])
                s = np.concatenate([scores_pos, scores_neg])
                aucs.append(roc_auc_score(y, s))

            # Store the mean per-pathway AUROC across splits.
            results[K].append(np.mean(aucs))

    return results, n_path


def main():
    """
    @brief   Main entry point: run pathway membership benchmark for 8M or 650M
             scale across all baselines and print AUROC summary table.
    @return  None (outputs results table to stdout).
    @note    Mode selection via CLI arg 1: "8m" (default) or "650m".
    """
    # Parse mode from CLI argument.
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE

    # Load KEGG pathway gene sets.
    with open(os.path.join(config.DATA_RAW, KEGG_FILE)) as fh:
        kegg = json.load(fh)

    # Restrict to pathways within the valid size range for meaningful evaluation.
    pathways = [
        (k, v) for k, v in kegg.items()
        if PATHWAY_MIN_SIZE <= len(v) <= PATHWAY_MAX_SIZE
    ]
    print("pathways in range [20,300]:", len(pathways))

    if mode == MODE_8M:
        # --- 8M scale: GFM + ESM-8M + PCA baselines ---
        Z, gl = get_embeddings_8m(config.result_path(CKPT_SUBDIR, CKPT_8M))

        # Build ESM-8M baseline embeddings.
        e8 = torch.load(
            os.path.join(config.DATA_PROC, ESM8M_FILE),
            map_location="cpu", weights_only=False
        )["embeddings"]
        baselines = {
            "esm8m": np.stack([e8[g] for g in gl]).astype(np.float64),
            "pca": torch.load(
                os.path.join(config.DATA_PROC, GRAPH_FILE),
                map_location="cpu", weights_only=False
            )["gene"]["x"].numpy(),
        }
    else:
        # --- 650M scale: GFM + ESM-650M (PCA-projected) + PCA baselines ---
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
        pf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pf)

        # Load 650M-scale graph and build model.
        lg = torch.load(
            os.path.join(config.DATA_PROC, GRAPH650_FILE),
            map_location="cpu", weights_only=False
        )
        dev = torch.device("cpu")
        data, node_types, edge_types, in_dims = pf.build_data_dict(lg, dev)
        x_dict = {nt: data[nt]["x"] for nt in node_types}
        ei_dict = {et: data[et]["edge_index"] for et in edge_types}

        # Standardize gene features.
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

        # Initialize and load 650M-scale checkpoint.
        model = pf.HGTEncoder(
            node_types, edge_types, in_dims,
            HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
        ).to(dev)
        ck = torch.load(
            config.result_path(CKPT_SUBDIR, CKPT_650M),
            map_location="cpu", weights_only=False
        )
        model.load_state_dict(ck["model"])
        model.eval()

        # Forward pass: extract GFM gene embeddings.
        with torch.no_grad():
            Z = model(x_dict, ei_dict)["gene"].cpu().numpy()
        gl = list(lg["meta"]["gene_list"])

        # Build ESM-650M baseline; project down to GFM dimension for fair comparison.
        e650 = torch.load(
            os.path.join(config.CLOUD_DIR, ESM650_FILE),
            map_location="cpu", weights_only=False
        )["embeddings"]
        raw = np.stack([e650[g].astype(np.float64) for g in gl])
        baselines = {
            "esm650": PCA(n_components=HGT_EMBEDDING_DIM, random_state=config.SEED
                         ).fit_transform(StandardScaler().fit_transform(raw)),
            "pca": lg["gene"]["x"].numpy(),
        }

    # Build random baseline (chance-level feature set).
    rng = np.random.RandomState(config.SEED)
    rand = rng.randn(*Z.shape)

    # Assemble all feature sets for evaluation.
    sets = {"gfm": Z, **baselines, "random": rand}
    print(f"mode={mode} | genes={len(gl)} | embeddings={Z.shape}")

    # Run pathway membership evaluation for each feature set.
    for name, E in sets.items():
        r, n = eval_pathways(E, gl, pathways)
        print(
            f"  {name:6s} | K=5  AUROC {np.mean(r[5]):.4f} +/- {np.std(r[5]):.4f}"
            f" | K=20 AUROC {np.mean(r[20]):.4f} +/- {np.std(r[20]):.4f}"
            f" | pathways={n}"
        )


if __name__ == "__main__":
    main()
