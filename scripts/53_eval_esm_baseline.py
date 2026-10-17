#!/usr/bin/env python3
"""
@file    53_eval_esm_baseline.py
@brief   ESM-2-only baseline (review item 3): evaluate RAW protein-language-model
         embeddings without any graph pretraining on the same PPI ratio and
         gene-cluster probes as the GFM. This closes the "the gain comes entirely
         from ESM-2, not graph pretraining" loop - if the GFM beats this baseline,
         the graph pretraining is doing real work.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --emb (default config.DATA_PROC/esm2_gene_embeddings.pt); --graph (default
        config.DATA_PROC/xena_graph_esm.pt); --dim, --n_ppi_pairs
Output: stdout only
Run:    python scripts/53_eval_esm_baseline.py
"""
import argparse
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Per-run artifact filenames (resolved under config.DATA_PROC).
ESM2_GENE_EMBEDDINGS = "esm2_gene_embeddings.pt"   # Raw ESM-2 gene embeddings
XENA_GRAPH_ESM = "xena_graph_esm.pt"               # ESM-regime graph

# Cluster probe hyperparameters.
KMEANS_N_INIT = 10                                  # Number of K-means restarts
LR_MAX_ITER = 500                                   # Max iterations for logistic regression
LR_REGULARIZATION_C = 1.0                           # Inverse regularization strength

# Dimension adjustment constants.
NARROW_EMB_PAD_STD = 0.01                           # Std dev of padding noise for narrow embeddings

# Numerical stability constants.
COSINE_EPSILON = 1e-9                               # Epsilon for L2-normalization


def ppi_ratio(emb, ppi_pairs, random_pairs):
    """
    @brief   Compute the mean cosine similarity ratio: true PPI pairs / random pairs.
    @param   emb : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   ppi_pairs : list
             True PPI edge tuples (gene_id pairs).
    @param   random_pairs : list
             Non-edge (negative) gene_id pairs (same length as ppi_pairs).
    @return  tuple of (ratio, pos_mean, neg_mean)
             - ratio    : float, pos_mean / neg_mean (headline separation score; >1 is better)
             - pos_mean : float, average cosine similarity over true PPI pairs
             - neg_mean : float, average cosine similarity over negative non-edge pairs
    @note    Rows are L2-normalized so the dot product is the cosine; the ratio is the
             headline separation score (a ratio above 1 means true pairs are closer).
    """
    # L2-normalize each embedding row so dot product equals cosine similarity
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + COSINE_EPSILON)
    # Compute per-pair cosine similarities and take class-wise averages
    pos = np.mean([float(emb[a] @ emb[b]) for a, b in ppi_pairs])
    neg = np.mean([float(emb[a] @ emb[b]) for a, b in random_pairs])
    return pos / (neg + COSINE_EPSILON), pos, neg


def gene_cluster_accuracy(emb, n_clusters=config.CLUSTER_K, k_shot=config.FEWSHOT_K,
                          n_repeats=config.FEWSHOT_REPEATS, seed=config.SEED):
    """
    @brief   K-means cluster labels -> few-shot logistic regression self-consistency
             accuracy. K-means ids become pseudo-labels; per cluster `k_shot` members
             train a linear classifier and the rest are held out. Results are averaged
             over `n_repeats` random folds.
    @param   emb : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   n_clusters : int, optional
             Number of K-means clusters (default: config.CLUSTER_K).
    @param   k_shot : int, optional
             Number of training examples per cluster (default: config.FEWSHOT_K).
    @param   n_repeats : int, optional
             Number of random split repeats to average over (default: config.FEWSHOT_REPEATS).
    @param   seed : int, optional
             Random seed for reproducibility (default: config.SEED).
    @return  float
             Mean few-shot classification accuracy over all valid repeats, or NaN if no
             valid splits could be formed.
    @note    Clusters with fewer than k_shot+1 members are skipped per split to avoid
             degenerate train/test splits.
    """
    # Use local RandomState for deterministic train/test split generation
    rng = np.random.RandomState(seed)
    # Standardize features for stable K-means clustering and logistic regression
    scaler = StandardScaler()
    X = scaler.fit_transform(emb)
    # Run K-means with multiple restarts to find a stable pseudo-label assignment
    labels = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT, random_state=seed).fit_predict(X)
    accs = []
    for _ in range(n_repeats):
        train_idx, test_idx = [], []
        for c in range(n_clusters):
            idx = np.where(labels == c)[0]
            # Skip clusters that are too small to support a train/test split
            if len(idx) < k_shot + 1:
                continue
            # Sample k_shot training points without replacement
            chosen = rng.choice(idx, k_shot, replace=False)
            train_idx.extend(chosen)
            test_idx.extend([i for i in idx if i not in set(chosen)])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        # Skip this repeat if the split is degenerate (empty train or test set)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        # Train and evaluate few-shot logistic regression classifier
        clf = LogisticRegression(
            max_iter=LR_MAX_ITER, C=LR_REGULARIZATION_C, random_state=seed)
        clf.fit(X[train_idx], labels[train_idx])
        accs.append(clf.score(X[test_idx], labels[test_idx]))
    return float(np.mean(accs)) if accs else float('nan')


def main():
    """
    @brief   Entry point: parse CLI args, load ESM-2 embeddings and graph, align
             embeddings to graph genes, optionally adjust embedding dimension (PCA
             projection or padding), sample balanced PPI pairs, run both downstream
             probes, and print the evaluation summary with interpretation guidance.
    @return  None (side effect: evaluation summary with interpretation guidance printed
             to stdout)
    @note    Dimension adjustment ensures fair comparison between models of different
             embedding widths: wide embeddings are projected via PCA, narrow embeddings
             are padded with small Gaussian noise to reach the target dimension.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default=os.path.join(config.DATA_PROC, ESM2_GENE_EMBEDDINGS))
    ap.add_argument('--graph', default=os.path.join(config.DATA_PROC, XENA_GRAPH_ESM))
    ap.add_argument('--dim', type=int, default=config.EMBEDDING_DIM)
    ap.add_argument('--n_ppi_pairs', type=int, default=config.N_PPI_PAIRS)
    args = ap.parse_args()

    # Fix global random seed for full reproducibility of the evaluation run
    config.set_seed()
    # Load ESM-2 embedding dictionary and the heterogeneous graph
    emb_dict = torch.load(args.emb, map_location='cpu', weights_only=False)['embeddings']
    graph = torch.load(args.graph, map_location='cpu')
    gene_list = graph['meta']['gene_list']
    ppi_ei = graph[('gene', 'ppi', 'gene')]['edge_index']

    # Build the aligned embedding matrix over genes present in the embedding dict,
    # then map graph gene ids back onto the contiguous kept-gene index range.
    kept = [g for g in gene_list if g in emb_dict]
    X = np.array([emb_dict[g] for g in kept], dtype=np.float32)
    g2i = {g: i for i, g in enumerate(kept)}
    print(f"genes with ESM-2 embeddings: {len(kept):,}")

    # --- Adjust embedding dimension to match the target probe dimension ---
    if X.shape[1] > args.dim:
        # Wide embeddings: project down via PCA to the target dimension
        X = PCA(
            n_components=args.dim, random_state=config.SEED
        ).fit_transform(StandardScaler().fit_transform(X))
    elif X.shape[1] < args.dim:
        # Narrow embeddings: pad with small Gaussian noise to reach target dimension
        pad_rng = np.random.RandomState(config.SEED)
        pad = (pad_rng.randn(X.shape[0], args.dim - X.shape[1]).astype(np.float32)
               * NARROW_EMB_PAD_STD)
        X = np.concatenate([X, pad], axis=1)

    # --- Sample balanced PPI pairs restricted to genes with ESM-2 embeddings ---
    edges = []
    for i in range(ppi_ei.size(1)):
        a, b = ppi_ei[0, i].item(), ppi_ei[1, i].item()
        ga, gb = gene_list[a], gene_list[b]
        # Only keep PPI edges where both endpoints have an ESM-2 embedding
        if ga in g2i and gb in g2i:
            edges.append((g2i[ga], g2i[gb]))
    # Use local RandomState for reproducible positive pair sampling
    rng = np.random.RandomState(config.SEED)
    pos = [edges[i] for i in rng.choice(len(edges), min(args.n_ppi_pairs, len(edges)), replace=False)]
    # Build undirected deduplicated PPI set for efficient negative pair exclusion
    ppi_set = set((min(a, b), max(a, b)) for a, b in edges)
    neg = []
    n = len(kept)
    # Sample balanced negative non-edge pairs (exclude self-loops and existing PPIs)
    while len(neg) < len(pos):
        a, b = int(rng.randint(n)), int(rng.randint(n))
        if a != b and (min(a, b), max(a, b)) not in ppi_set:
            neg.append((a, b))

    # Run both downstream probes (PPI ratio probe + cluster self-consistency probe)
    ratio, p, q = ppi_ratio(X, pos, neg)
    acc = gene_cluster_accuracy(X)

    # Print the full ESM-2-only baseline evaluation summary with interpretation guidance
    print("\n=== ESM-2-only (NO graph pretraining) baseline ===")
    print(f"PPI ratio: {ratio:.3f}x  (pos {p:.4f} / neg {q:.4f})")
    print(f"Gene cluster accuracy: {acc:.4f}")
    print("\nReference values for the pretrained GFM model are in scripts/README.md; "
          "run the pretrained-checkpoint eval for a direct comparison.")
    print("If GFM > these values, graph pretraining adds value beyond ESM-2 alone.")


if __name__ == '__main__':
    main()
