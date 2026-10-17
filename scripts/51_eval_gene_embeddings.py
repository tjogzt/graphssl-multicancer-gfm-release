#!/usr/bin/env python3
"""
@file    51_eval_gene_embeddings.py
@brief   Extract GFM-pretrained gene embeddings and evaluate them on the same
         downstream probes as the ESM-2-only baseline (53_eval_esm_baseline.py),
         reusing that script's metric functions so both models go through
         identical code for a fair apples-to-apples comparison of graph
         pretraining against raw protein-language-model embeddings.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_evaluation

Input:  --data (default config.DATA_PROC/xena_graph_esm.pt); --ckpt (default
        config.MODEL_DIR/gfm_fixed_e500.pt); --n_ppi_pairs, --dim, --layers
Output: stdout only
Run:    python scripts/51_eval_gene_embeddings.py [--ckpt PATH]
"""
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

# Per-run artifact filenames (resolved under config.DATA_PROC / config.MODEL_DIR).
XENA_GRAPH_ESM = "xena_graph_esm.pt"               # ESM-regime graph
GFM_CKPT = "gfm_fixed_e500.pt"                      # Default pretrained checkpoint name

# HGT encoder geometry must match the pretraining run behind the checkpoint.
HGT_LAYERS = 6                                      # Default number of HGT layers
HGT_HEADS = 8                                       # Default number of attention heads

# Reference ESM-only numbers documented in scripts/README.md; printed for comparison
# only and never used in the computation.
ESM_REF_POS_COS = 0.2004                            # Reference pos cosine (ESM-only)
ESM_REF_NEG_COS = 0.0008                            # Reference neg cosine (ESM-only)
ESM_REF_CLUSTER = 0.4774                            # Reference cluster acc (ESM-only)

# Numerical stability constants.
COSINE_EPSILON = 1e-9                               # Epsilon for L2-normalization


def _load(name, path):
    """
    @brief   Dynamically load a Python module by file path (helper for code reuse).
    @param   name : str
             Module name (used for importlib bookkeeping).
    @param   path : str
             Absolute or relative (to scripts/) path to the .py file to load.
    @return  module
             Loaded Python module object.
    @note    Uses importlib.util so metric/model functions can be shared between
             scripts without creating an import-time dependency graph.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import reusable metric and model functions from sibling scripts
_eb = _load("esm_base", "53_eval_esm_baseline.py")
_pf = _load("pretrain_fixed", "40_pretrain.py")
ppi_ratio = _eb.ppi_ratio
gene_cluster_accuracy = _eb.gene_cluster_accuracy
HGTEncoder = _pf.HGTEncoder


def extract_gene_embeddings(graph, ckpt_path, hidden=config.EMBEDDING_DIM,
                            layers=HGT_LAYERS, heads=HGT_HEADS, dim=config.EMBEDDING_DIM):
    """
    @brief   Load a GFM checkpoint into a fresh HGT encoder and extract its gene
             embeddings via one forward pass over the input graph.
    @param   graph : dict
             Heterogeneous graph dictionary (see build_data_dict for structure).
    @param   ckpt_path : str
             Path to the pretrained GFM checkpoint .pt file.
    @param   hidden : int, optional
             Hidden dimension of the HGT encoder (must match checkpoint) (default: config.EMBEDDING_DIM).
    @param   layers : int, optional
             Number of HGT layers in the encoder (must match checkpoint) (default: HGT_LAYERS).
    @param   heads : int, optional
             Number of attention heads per layer (must match checkpoint) (default: HGT_HEADS).
    @param   dim : int, optional
             Unused placeholder parameter retained for API compatibility (default: config.EMBEDDING_DIM).
    @return  numpy.ndarray
             Gene embedding matrix of shape [num_genes, hidden], on CPU as float32/64 numpy array.
    @note    Inference is performed in eval mode with torch.no_grad() to avoid
             autograd overhead and ensure deterministic output.
    """
    # Auto-select best available compute device: CUDA > MPS > CPU
    device = torch.device(
        'cuda:0' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    # Build data dictionaries from the raw graph
    data, node_types, edge_types, in_dims = _pf.build_data_dict(graph, device)
    x_dict = {nt: data[nt]['x'] for nt in node_types}
    ei_dict = {et: data[et]['edge_index'] for et in edge_types}

    # Instantiate HGT encoder with matching geometry and load pretrained weights
    model = HGTEncoder(node_types, edge_types, in_dims, hidden, layers, heads).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    # One full-graph forward pass to extract final node embeddings
    with torch.no_grad():
        z = model(x_dict, ei_dict)
    return z['gene'].cpu().numpy()  # [N, hidden]


def main():
    """
    @brief   Entry point: parse CLI args, load graph + checkpoint, extract gene
             embeddings, optionally project dimension, sample PPI pairs, run both
             downstream probes, and print the evaluation summary with reference
             ESM-only numbers.
    @return  None (side effect: evaluation summary printed to stdout)
    @note    If the extracted embedding dimension exceeds the requested --dim,
             PCA projection is used to reduce the dimension for a fair comparison
             with narrower baselines.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(config.DATA_PROC, XENA_GRAPH_ESM))
    ap.add_argument('--ckpt', default=os.path.join(config.MODEL_DIR, GFM_CKPT))
    ap.add_argument('--n_ppi_pairs', type=int, default=config.N_PPI_PAIRS)
    ap.add_argument('--dim', type=int, default=config.EMBEDDING_DIM)
    ap.add_argument('--layers', type=int, default=HGT_LAYERS)
    args = ap.parse_args()

    # Fix global random seed for reproducibility of the entire evaluation run
    config.set_seed()
    # Load heterogeneous graph and extract gene list + PPI edge index
    graph = torch.load(args.data, map_location='cpu')
    gene_list = graph['meta']['gene_list']
    ppi_ei = graph[('gene', 'ppi', 'gene')]['edge_index']

    # Extract GFM-pretrained gene embeddings from the checkpoint
    X = extract_gene_embeddings(graph, args.ckpt, layers=args.layers, dim=args.dim)
    if X.shape[1] > args.dim:
        # Project (possibly wider) embeddings down to the requested dimension via PCA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = PCA(
            n_components=args.dim,
            random_state=config.SEED
        ).fit_transform(StandardScaler().fit_transform(X))

    # Sample true PPI edges plus an equal number of non-edges (same procedure as
    # the ESM baseline) so the ratio is computed over a balanced, reproducible pair set.
    edges = []
    for i in range(ppi_ei.size(1)):
        a, b = ppi_ei[0, i].item(), ppi_ei[1, i].item()
        edges.append((a, b))
    # Use local RandomState for reproducible pair sampling
    rng = np.random.RandomState(config.SEED)
    pos = [edges[i] for i in rng.choice(len(edges), min(args.n_ppi_pairs, len(edges)), replace=False)]
    # Build undirected deduplicated PPI set for efficient negative pair exclusion
    ppi_set = set((min(a, b), max(a, b)) for a, b in edges)
    neg = []
    n = X.shape[0]
    # Sample balanced negative non-edge pairs (exclude self-loops and existing PPIs)
    while len(neg) < len(pos):
        a, b = int(rng.randint(n)), int(rng.randint(n))
        if a != b and (min(a, b), max(a, b)) not in ppi_set:
            neg.append((a, b))

    # Run both downstream probes (shared metric functions with the ESM baseline)
    ratio, p, q = ppi_ratio(X, pos, neg)
    acc = gene_cluster_accuracy(X)

    # Print the full evaluation summary with reference ESM-only comparison numbers
    print("\n=== GFM-pretrained embeddings ===")
    print(f"PPI ratio: {ratio:.3f}x  (pos cosine {p:.4f} / neg cosine {q:.4f})")
    print(f"Gene cluster accuracy: {acc:.4f}")
    print(f"\nReference (ESM-only, no graph): pos {ESM_REF_POS_COS:.4f} / "
          f"neg {ESM_REF_NEG_COS:.4f}, clustering {ESM_REF_CLUSTER:.4f}")
    print("Reference values for the pretrained GFM model are in scripts/README.md.")


if __name__ == '__main__':
    main()
