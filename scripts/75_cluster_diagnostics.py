#!/usr/bin/env python3
"""
@file    75_cluster_diagnostics.py
@brief   Cluster-label self-consistency diagnostics for the K-means assignment
         recoverability metric. Quantifies three key reviewer-style controls:
           1. Cross-training-seed partition stability (ARI / NMI between two seeds).
           2. K-sensitivity: recoverability at K in {15, 21, 27}.
           3. Matched-size random-partition control: recoverability after shuffling
              K-means labels while preserving cluster sizes (the null for any equal-
              size partition).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup functional_analysis

Input:  Trained checkpoints + baselines (same evaluation protocol as 50_eval_checkpoint.py).
Output: results/cluster_diagnostics.json + a printed summary table.
Run:    python scripts/75_cluster_diagnostics.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Source / output artifact paths.
GRAPH = os.path.join(config.DATA_PROC, "xena_graph_esm.pt")
ESM = os.path.join(config.DATA_PROC, "esm2_gene_embeddings.pt")
OUT = config.result_path("cluster_diagnostics.json")

# Primary K for K-means recoverability (same as manuscript default).
K = config.CLUSTER_K

# K-sensitivity sweep values evaluated on the primary arm.
K_SENSITIVITY_VALUES = (15, 21, 27)

# Few-shot support size per cluster for the recoverability probe.
K_SHOT = 5

# Number of few-shot repeats averaged per recoverability call.
RECOVERABILITY_REPEATS = 30

# Logistic regression parameters used by the recoverability probe.
LR_C = 1.0
LR_MAX_ITER = 500

# K-means parameters.
KMEANS_N_INIT = 10

# HGT encoder architecture hyper-parameters (mirrors 40_pretrain.py defaults).
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Output JSON indentation spaces.
JSON_INDENT = 1

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False

# Sub-directory under results for cached per-arm gene embeddings.
EMBEDDINGS_CACHE_SUBDIR = "embeddings"

# Recovery score rounding decimal places.
SCORE_DECIMALS = 4


def load_model_module():
    """
    @brief   Dynamically import 40_pretrain.py as a module so that the HGT encoder
             architecture and graph-builder helpers can be reused without copy-paste.
    @return  Module namespace for 40_pretrain.py (aliased as `pf` inside this script).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(here, "40_pretrain.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def embeddings_for(graph, gene_list, pf, ckpt_path=None, baseline=None):
    """
    @brief   Return the (n_genes x d) gene embedding matrix either from a baseline
             source (ESM raw embeddings, PCA graph features, random HGT init) or by
             a forward pass of a trained HGT checkpoint.
    @param   graph       dict – HeteroData-style graph dictionary.
    @param   gene_list   list – ordered gene symbols (used to align ESM embeddings).
    @param   pf          module – imported 40_pretrain.py namespace.
    @param   ckpt_path   str or None – path to trained checkpoint, used when no
                         baseline tag is provided.
    @param   baseline    str or None – one of {"esm", "pca", "random", None}.
    @return  np.ndarray of shape (len(gene_list), d) with dtype float64.
    """
    dev = torch.device(LOAD_MAP_LOCATION)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)

    # Baseline arm: ESM-2 raw embeddings, standardised elsewhere by preprocess().
    if baseline == "esm":
        e = torch.load(ESM, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
        return np.stack([e["embeddings"][g] for g in gene_list]).astype(np.float64)

    # Baseline arm: PCA-256 graph input features.
    if baseline == "pca":
        return graph["gene"]["x"].float().numpy()

    # HGT forward pass: fresh random init or loaded from a trained checkpoint.
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims, config.EMBEDDING_DIM,
        HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)

    if baseline == "random":
        # Fresh random init: do not load any state dict.
        pass
    else:
        ck = torch.load(ckpt_path, map_location=LOAD_MAP_LOCATION,
                        weights_only=TORCH_LOAD_WEIGHTS_ONLY)
        model.load_state_dict(ck["model"])

    model.eval()
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy()


def preprocess(X):
    """
    @brief   Standardise rows across dimensions and sanitise any non-finite values
             that can arise from zero-variance columns or overflow.
    @param   X  np.ndarray – (n_genes, d) raw embedding matrix.
    @return  np.ndarray – (n_genes, d) z-scored matrix with NaN/Inf replaced by 0.
    """
    X = StandardScaler().fit_transform(X.astype(np.float64))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def kmeans_labels(X, k, seed=config.SEED):
    """
    @brief   Fit K-means and return the cluster assignment vector.
    @param   X     np.ndarray – (n_genes, d) preprocessed features.
    @param   k     int – number of clusters.
    @param   seed  int – random state for the K-means initialiser.
    @return  np.ndarray of shape (n_genes,) containing int cluster labels.
    """
    return KMeans(
        n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed
    ).fit_predict(X)


def recoverability(X, labels, k_shot=K_SHOT, repeats=RECOVERABILITY_REPEATS,
                   seed=config.SEED):
    """
    @brief   Few-shot K-means recoverability: average few-shot classification
             accuracy when using k_shot labelled samples per cluster as support.
    @param   X        np.ndarray – (n_genes, d) preprocessed features.
    @param   labels   np.ndarray – (n_genes,) K-means cluster assignments.
    @param   k_shot   int – number of support points per cluster.
    @param   repeats  int – number of independent random draws to average over.
    @param   seed     int – master seed for the random draws.
    @return  float – mean classification accuracy over the repeats.
    """
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(repeats):
        tr, te = [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < k_shot + 1:
                # Cluster too small to have at least one test point: skip.
                continue
            ch = rng.choice(idx, k_shot, replace=False)
            tr.extend(ch)
            te.extend([i for i in idx if i not in set(ch)])
        if not te:
            continue
        clf = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C)
        clf.fit(X[tr], labels[tr])
        accs.append(clf.score(X[te], labels[te]))
    return float(np.mean(accs))


def random_partition_control(X, labels, k_shot=K_SHOT, repeats=RECOVERABILITY_REPEATS,
                             seed=config.SEED):
    """
    @brief   Null-model recoverability: the same few-shot recoverability metric but
             after a random permutation of the cluster labels. This preserves the
             cluster-size distribution but destroys any geometric structure.
    @param   X        np.ndarray – (n_genes, d) preprocessed features.
    @param   labels   np.ndarray – (n_genes,) cluster labels to be permuted.
    @param   k_shot   int – number of support points per cluster.
    @param   repeats  int – number of independent random draws.
    @param   seed     int – master seed.
    @return  float – mean recoverability of the shuffled-label partition.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(labels))
    shuffled = labels[perm]  # Preserves cluster sizes, destroys geometric structure.
    return recoverability(X, shuffled, k_shot, repeats, seed)


def main():
    """
    @brief   Entry point: load graph and model module, evaluate each ablation arm's
             recoverability + random-partition null, measure cross-seed stability,
             write the JSON diagnostics report.
    @return  None
    """
    graph = torch.load(GRAPH, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    gene_list = list(graph["meta"]["gene_list"])
    pf = load_model_module()

    # Definition of the arms: (tag, checkpoint_path, baseline_tag).
    arms = [
        ("full", os.path.join(config.RESULT_DIR, "ablation_real", "full.pt"), None),
        ("namr_only", os.path.join(config.RESULT_DIR, "ablation_real", "namr_only.pt"), None),
        ("csp_only", os.path.join(config.RESULT_DIR, "ablation_real", "csp_only.pt"), None),
        ("csp_namr", os.path.join(config.RESULT_DIR, "ablation_real", "csp_namr_legacy_s42.pt"), None),
        ("esm_only", None, "esm"),
        ("pca_input", None, "pca"),
        ("random_init_hgt", None, "random"),
    ]

    os.makedirs(config.result_path(EMBEDDINGS_CACHE_SUBDIR), exist_ok=True)
    rows = {}
    for tag, ckpt, baseline in arms:
        raw = embeddings_for(graph, gene_list, pf, ckpt, baseline)
        # Cache for downstream re-use by the KEGG table script.
        np.save(config.result_path(EMBEDDINGS_CACHE_SUBDIR, f"{tag}.npy"), raw)
        X = preprocess(raw)
        labels = kmeans_labels(X, K)
        rows[tag] = {
            "recoverability_K21": round(recoverability(X, labels), SCORE_DECIMALS),
            "random_partition_control": round(
                random_partition_control(X, labels), SCORE_DECIMALS
            ),
        }
        # K-sensitivity sweep only on the primary arm (full) to save compute.
        if tag == "full":
            rows[tag]["K_sensitivity"] = {
                str(k): round(
                    recoverability(X, kmeans_labels(X, k)), SCORE_DECIMALS
                )
                for k in K_SENSITIVITY_VALUES
            }

    # Cross-training-seed stability: compare K-means partitions of seed-42 vs seed-7
    # models for the full and namr_only arms.
    stability = {}
    for name, a_ckpt, b_ckpt in [
        ("full_s42_vs_s7", "full.pt", "full_s7.pt"),
        ("namr_s42_vs_s7", "namr_only.pt", "namr_only_s7o.pt"),
    ]:
        base = os.path.join(config.RESULT_DIR, "ablation_real")
        Xa = preprocess(
            embeddings_for(graph, gene_list, pf, os.path.join(base, a_ckpt))
        )
        Xb = preprocess(
            embeddings_for(graph, gene_list, pf, os.path.join(base, b_ckpt))
        )
        la, lb = kmeans_labels(Xa, K), kmeans_labels(Xb, K)
        stability[name] = {
            "ARI": round(adjusted_rand_score(la, lb), SCORE_DECIMALS),
            "NMI": round(normalized_mutual_info_score(la, lb), SCORE_DECIMALS),
        }

    result = {"rows": rows, "cross_seed_stability": stability}
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=JSON_INDENT)
    print(json.dumps(result, indent=JSON_INDENT))
    print(f"[written] {OUT}")


if __name__ == "__main__":
    main()
