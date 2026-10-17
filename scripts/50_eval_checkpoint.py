#!/usr/bin/env python3
"""
@file    50_eval_checkpoint.py
@brief   Evaluate a GFM checkpoint (or an ESM / PCA / random-init baseline) on
         two downstream probes so pretrained gene embeddings can be compared
         apples-to-apples across ablation runs: (1) PPI link prediction via the
         cosine-similarity ratio (mean positive / mean negative cosine) plus
         ROC-AUC over equal numbers of positive and negative pairs, and (2)
         gene-cluster self-consistency via K-means followed by few-shot logistic
         regression (averaged over repeats).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --data <path> (default config.DATA_PROC/xena_graph_esm.pt); --ckpt <path> or
        --baseline {esm,pca,random}; --tag <name> (optional output label)
Output: stdout only (one PPI / AUC / cluster-accuracy line)
Run:    python scripts/50_eval_checkpoint.py --ckpt <path> [--tag NAME]
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.set_seed()

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Per-run artifact filenames (resolved under config.DATA_PROC).
XENA_GRAPH_ESM = "xena_graph_esm.pt"               # ESM-regime graph
ESM2_GENE_EMBEDDINGS = "esm2_gene_embeddings.pt"   # Raw ESM-2 gene embeddings

# HGT encoder geometry must match the pretraining run behind each checkpoint.
HGT_LAYERS = 6                                      # Default number of HGT layers
HGT_HEADS = 8                                       # Default number of attention heads

# Supervised probe hyperparameters.
LR_MAX_ITER = 500                                   # Max iterations for logistic regression
LR_REGULARIZATION_C = 1.0                           # Inverse regularization strength

# Numerical stability constants.
COSINE_EPSILON = 1e-9                               # Epsilon for L2-normalization


def load_pairs(graph, n_pairs=config.N_PPI_PAIRS, seed=config.SEED):
    """
    @brief   Sample balanced positive/negative PPI pairs for the link-prediction probe.
             Sample `n_pairs` true PPI edges plus an equal number of non-edges.
    @param   graph : dict
             Heterogeneous graph dictionary with PPI edge index and meta information.
    @param   n_pairs : int, optional
             Maximum number of positive/negative pairs to sample (default: config.N_PPI_PAIRS).
    @param   seed : int, optional
             Random seed for reproducible pair sampling (default: config.SEED).
    @return  tuple of (pos, neg)
             - pos : list of (gene_id_a, gene_id_b) true PPI edge tuples
             - neg : list of (gene_id_a, gene_id_b) non-edge tuples (same length as pos)
    @note    Gene indices are the integer node ids in the graph edge_index, so a pair is
             simply (src_id, dst_id). Negatives are drawn uniformly over all gene-id pairs
             while excluding self-loops and any pair already in the PPI set, guaranteeing
             the two classes are balanced for the downstream ROC-AUC.
    """
    # Extract full PPI edge list from the graph
    ppi = graph[("gene", "ppi", "gene")]["edge_index"]
    edges = [(int(ppi[0, i]), int(ppi[1, i])) for i in range(ppi.size(1))]
    # Use local RandomState to ensure reproducibility independent of global RNG state
    rng = np.random.RandomState(seed)
    # Sample positive PPI pairs without replacement (cap at n_pairs)
    pos = [edges[i] for i in rng.choice(len(edges), min(n_pairs, len(edges)), replace=False)]
    # Build undirected deduplicated PPI set for efficient negative pair exclusion
    ppi_set = set((min(a, b), max(a, b)) for a, b in edges)
    neg = []
    n = graph["meta"]["n_genes"]
    # Sample balanced negative non-edge pairs (exclude self-loops and existing PPIs)
    while len(neg) < len(pos):
        a, b = int(rng.randint(n)), int(rng.randint(n))
        if a != b and (min(a, b), max(a, b)) not in ppi_set:
            neg.append((a, b))
    return pos, neg


def eval_embeddings(X, pos, neg):
    """
    @brief   Evaluate gene embeddings on the cosine-similarity PPI probe.
             Returns ratio, per-class mean cosine, and ROC-AUC.
    @param   X : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   pos : list
             True PPI edge tuples (gene_id pairs).
    @param   neg : list
             Non-edge (negative) gene_id pairs (same length as pos).
    @return  tuple of (ratio, pos_mean, neg_mean, auc)
             - ratio    : float, mean positive cosine / mean negative cosine (headline score)
             - pos_mean : float, average cosine similarity over true PPI pairs
             - neg_mean : float, average cosine similarity over negative non-edge pairs
             - auc      : float, ROC-AUC over balanced class labels
    @note    Rows are L2-normalized first so the dot product is the cosine; the ratio of
             mean positive to mean negative cosine is the paper's headline separation score.
    """
    # Cast to float64 for numerical stability in cosine computation
    X = X.astype(np.float64)
    # L2-normalize each embedding row so dot product equals cosine similarity
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + COSINE_EPSILON)
    # Compute cosine similarities for positive and negative pairs
    pos_cos = np.array([float(Xn[a] @ Xn[b]) for a, b in pos])
    neg_cos = np.array([float(Xn[a] @ Xn[b]) for a, b in neg])
    # Ratio of mean positive to mean negative cosine (headline separation score)
    ratio = pos_cos.mean() / (neg_cos.mean() + COSINE_EPSILON)
    # Build balanced class labels and compute ROC-AUC
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos_cos, neg_cos])
    auc = roc_auc_score(y, s)
    return ratio, pos_cos.mean(), neg_cos.mean(), auc


def cluster_acc(emb, n_clusters=config.CLUSTER_K, k_shot=config.FEWSHOT_K,
                n_repeats=config.FEWSHOT_REPEATS, seed=config.SEED):
    """
    @brief   K-means clustering followed by few-shot logistic-regression
             self-consistency accuracy.
             K-means cluster ids become pseudo-labels; per cluster, `k_shot` members
             train a linear classifier and the rest are held out. Results are averaged
             over `n_repeats` random train/test folds.
    @param   emb : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   n_clusters : int, optional
             Number of K-means clusters (default: config.CLUSTER_K).
    @param   k_shot : int, optional
             Number of training examples per cluster (default: config.FEWSHOT_K).
    @param   n_repeats : int, optional
             Number of random split repeats to average over (default: config.FEWSHOT_REPEATS).
    @param   seed : int, optional
             Random seed for reproducible clustering and splits (default: config.SEED).
    @return  tuple of (acc_mean, acc_std)
             - acc_mean : float, mean few-shot classification accuracy over repeats
             - acc_std  : float, standard deviation over repeats
    @note    Splits are pre-generated with a local RandomState so serial and joblib-parallel
             runs use identical folds for reproducibility.
    """
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Use local RandomState for deterministic train/test split generation
    rng = np.random.RandomState(seed)
    # Standardize features before clustering for stable K-means
    X = StandardScaler().fit_transform(emb.astype(np.float64))
    # Handle non-finite values that may arise from extreme embeddings
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Generate K-means pseudo-labels with fixed seed for reproducibility
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)

    # Pre-generate all train/test splits (main-process RNG; order identical to serial
    # execution) so parallelism never changes the folds used in the evaluation.
    splits = []
    for _ in range(n_repeats):
        tr, te = [], []
        for c in range(n_clusters):
            idx = np.where(labels == c)[0]
            # Skip clusters that are too small to support a train/test split
            if len(idx) < k_shot + 1:
                continue
            # Sample k_shot training points without replacement
            ch = rng.choice(idx, k_shot, replace=False)
            tr.extend(ch)
            te.extend([i for i in idx if i not in set(ch)])
        splits.append((tr, te))

    # Optionally evaluate folds in parallel when the environment variable is set
    if os.environ.get('EVAL_PARALLEL') == '1':
        from joblib import Parallel, delayed
        def _one(tr, te):
            """Evaluate a single train/test split."""
            clf = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_REGULARIZATION_C)
            clf.fit(X[tr], labels[tr])
            return clf.score(X[te], labels[te])
        accs = Parallel(n_jobs=-1)(delayed(_one)(tr, te) for tr, te in splits)
    else:
        # Serial evaluation of all splits
        accs = []
        for tr, te in splits:
            clf = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_REGULARIZATION_C)
            clf.fit(X[tr], labels[tr])
            accs.append(clf.score(X[te], labels[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    """
    @brief   Entry point: parse CLI arguments, load graph and embeddings (from checkpoint
             or baseline encoder), and print PPI ratio/AUC and cluster accuracy metrics.
    @return  None (side effect: one evaluation line printed to stdout)
    @note    Supports four embedding sources: (1) ESM-2 raw PLM embeddings,
             (2) raw input graph features (PCA baseline), (3) randomly initialized HGT
             encoder, or (4) a trained GFM checkpoint.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", choices=["esm", "pca", "random"], default=None)
    ap.add_argument("--data", default=os.path.join(config.DATA_PROC, XENA_GRAPH_ESM))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # Load heterogeneous graph and extract gene list for embedding alignment
    graph = torch.load(args.data, map_location="cpu", weights_only=False)
    gene_list = list(graph["meta"]["gene_list"])
    # Sample balanced positive/negative PPI pairs once (shared by all evaluation modes)
    pos, neg = load_pairs(graph)

    if args.baseline == "esm":
        # --- Baseline 1: Raw PLM embeddings (ESM-2 only, no graph pretraining) ---
        # Index each graph gene by its gene_list entry in the embedding dictionary
        e = torch.load(
            os.path.join(config.DATA_PROC, ESM2_GENE_EMBEDDINGS),
            map_location="cpu", weights_only=False
        )
        X = np.stack([e["embeddings"][g] for g in gene_list]).astype(np.float64)
        tag = "esm_only"
    elif args.baseline == "pca":
        # --- Baseline 2: Raw graph input features (no model, PCA-style baseline) ---
        X = graph["gene"]["x"].float().numpy()
        tag = "pca_input"
    elif args.baseline == "random":
        # --- Baseline 3: Randomly initialized HGT encoder (no training at all) ---
        import importlib.util
        # Dynamically load the pretraining script for model/builder reuse
        spec = importlib.util.spec_from_file_location(
            "pf",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "40_pretrain.py")
        )
        pf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pf)
        dev = torch.device("cpu")
        data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
        x_dict = {nt: data[nt]["x"] for nt in node_types}
        ei_dict = {et: data[et]["edge_index"] for et in edge_types}
        # Instantiate fresh random HGT encoder (random weights, no training)
        model = pf.HGTEncoder(
            node_types, edge_types, in_dims,
            config.EMBEDDING_DIM, HGT_LAYERS, HGT_HEADS
        ).to(dev)
        model.eval()
        with torch.no_grad():
            X = model(x_dict, ei_dict)["gene"].cpu().numpy()
        tag = "random_init_hgt"
    else:
        # --- Default mode: Load a trained GFM checkpoint into a fresh HGT encoder ---
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pf",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "40_pretrain.py")
        )
        pf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pf)
        dev = torch.device("cpu")
        data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
        x_dict = {nt: data[nt]["x"] for nt in node_types}
        ei_dict = {et: data[et]["edge_index"] for et in edge_types}
        # Instantiate HGT encoder with matching geometry and load checkpoint weights
        model = pf.HGTEncoder(
            node_types, edge_types, in_dims,
            config.EMBEDDING_DIM, HGT_LAYERS, HGT_HEADS
        ).to(dev)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            X = model(x_dict, ei_dict)["gene"].cpu().numpy()
        tag = args.tag or os.path.basename(args.ckpt).replace(".pt", "")

    # --- Run both downstream probes and print the summary line ---
    ratio, pc, nc, auc = eval_embeddings(X, pos, neg)
    acc, acc_std = cluster_acc(X)
    print(f"[{tag}] PPI ratio={ratio:.3f}x (pos {pc:.4f} / neg {nc:.4f}) "
          f"| AUC={auc:.4f} | cluster acc={acc:.4f} +/- {acc_std:.4f}")


if __name__ == "__main__":
    main()
