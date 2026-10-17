#!/usr/bin/env python3
"""
@file    56_center_check_diag.py
@brief   Diagnostic for whether mean-centering the gene embeddings explains the
         PPI cosine-similarity ratio: compute the ratio before and after
         subtracting the per-dimension mean, so a ratio that collapses after
         centering implicates the mean-centering preprocessing step rather than
         the learned embedding geometry.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --ckpt (default config.result_path("ablation_real", "namr_only.pt"));
        graph from config.DATA_PROC/xena_graph_esm.pt
Output: stdout only
Run:    python scripts/56_center_check_diag.py [--ckpt PATH]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_here = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Results/ subdirectory and default checkpoint for the manuscript-standard model.
ABLATION_SUBDIR = "ablation_real"                    # Ablation experiment subfolder
NAMRO_CKPT = "namr_only.pt"                          # Default checkpoint to diagnose
XENA_GRAPH_ESM = "xena_graph_esm.pt"                 # ESM-regime graph

# Diagnostic reference: a centered ratio above this points at the centering step.
CENTER_RATIO_THRESHOLD = 1.11                        # Heuristic threshold

# Numerical stability constants.
COSINE_EPSILON = 1e-9                                # Epsilon for L2-normalization


def _load(name, path):
    """
    @brief   Dynamically load a Python module by path relative to the scripts folder
             (code-reuse helper for evaluation utilities).
    @param   name : str
             Module name (used for importlib bookkeeping).
    @param   path : str
             Path relative to the scripts directory of the .py file to load.
    @return  module
             Loaded Python module object.
    @note    Uses importlib.util so metric/model functions can be shared between
             scripts without creating an import-time package dependency.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Reuse GFM embedding extraction from the gene-embeddings evaluation script
_ev = _load("eval", "51_eval_gene_embeddings.py")


def ppi_ratio(emb, pos_pairs, neg_pairs):
    """
    @brief   Compute the mean cosine similarity ratio of true PPI pairs over random
             non-edge pairs. Rows are L2-normalized so dot product equals cosine.
    @param   emb : numpy.ndarray
             Gene embedding matrix of shape [num_genes, embedding_dim].
    @param   pos_pairs : list
             True PPI edge tuples (gene_id pairs).
    @param   neg_pairs : list
             Non-edge (negative) gene_id pairs (same length as pos_pairs).
    @return  tuple of (ratio, pos_mean, neg_mean)
             - ratio    : float, mean pos cosine / mean neg cosine (headline score)
             - pos_mean : float, average cosine similarity over true PPI pairs
             - neg_mean : float, average cosine similarity over negative non-edge pairs
    """
    # L2-normalize each embedding row so dot product equals cosine similarity
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + COSINE_EPSILON)
    # Compute per-pair cosine similarities and take class-wise averages
    pos = np.mean([float(emb[a] @ emb[b]) for a, b in pos_pairs])
    neg = np.mean([float(emb[a] @ emb[b]) for a, b in neg_pairs])
    return pos / (neg + COSINE_EPSILON), pos, neg


def main():
    """
    @brief   Entry point: parse CLI args, load graph and checkpoint, extract GFM
             gene embeddings, sample balanced PPI pairs, then compute and print the
             PPI cosine ratio (a) on raw L2-normalized embeddings and (b) after
             per-dimension mean-centering. The diagnostic interpretation guidance is
             printed alongside the numbers.
    @return  None (side effect: two ratio lines + diagnostic interpretation printed
             to stdout)
    @note    If the ratio remains above CENTER_RATIO_THRESHOLD after mean-centering,
             the separation is attributable to the mean-centering preprocessing step
             rather than the learned relative embedding geometry.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=config.result_path(ABLATION_SUBDIR, NAMRO_CKPT))
    args = ap.parse_args()

    # Auto-select best available compute device: CUDA > MPS > CPU
    device = torch.device(
        'cuda:0' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    # Load the heterogeneous graph and extract pretrained GFM gene embeddings
    graph = torch.load(
        os.path.join(config.DATA_PROC, XENA_GRAPH_ESM),
        map_location='cpu'
    )
    X = _ev.extract_gene_embeddings(graph, args.ckpt)
    ppi_ei = graph[('gene', 'ppi', 'gene')]['edge_index']
    n = X.shape[0]

    # Build the full PPI edge list from the graph's edge_index tensor
    edges = [(ppi_ei[0, i].item(), ppi_ei[1, i].item()) for i in range(ppi_ei.size(1))]
    # Use local RandomState for reproducible positive pair sampling
    rng = np.random.RandomState(config.SEED)
    # Balanced sample of true edges and random non-edges (same procedure as 51)
    pos = [edges[i] for i in rng.choice(len(edges), config.N_PPI_PAIRS, replace=False)]
    # Build undirected deduplicated PPI set for efficient negative pair exclusion
    ppi_set = set((min(a, b), max(a, b)) for a, b in edges)
    neg = []
    # Sample balanced negative non-edge pairs (exclude self-loops and existing PPIs)
    while len(neg) < len(pos):
        a, b = int(rng.randint(n)), int(rng.randint(n))
        if a != b and (min(a, b), max(a, b)) not in ppi_set:
            neg.append((a, b))

    # --- PPI ratio BEFORE mean-centering (raw L2-normalized embeddings) ---
    r1, p1, q1 = ppi_ratio(X, pos, neg)
    print(f"Raw (L2-normalized, not mean-centered): ratio {r1:.3f}x  "
          f"(pos {p1:.4f} / neg {q1:.4f})")

    # --- PPI ratio AFTER mean-centering (per-dimension mean subtracted first) ---
    Xc = X - X.mean(axis=0, keepdims=True)  # Mean-center each embedding dimension
    r2, p2, q2 = ppi_ratio(Xc, pos, neg)
    print(f"After mean-centering:      ratio {r2:.3f}x  "
          f"(pos {p2:.4f} / neg {q2:.4f})")
    # Print diagnostic interpretation guidance for the user
    print(f"\nIf the ratio after mean-centering is clearly > {CENTER_RATIO_THRESHOLD}, "
          "the separation is attributable to the mean-centering preprocessing step.")


if __name__ == '__main__':
    main()
