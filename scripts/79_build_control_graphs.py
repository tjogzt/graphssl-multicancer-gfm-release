#!/usr/bin/env python3
"""
@file    79_build_control_graphs.py
@brief   Build the four structural-control graphs used to validate the contribution
         of each graph component to the learned gene embeddings. Supported variants:
           --variant ppi_only      : gene nodes + PPI edges only (no gene-cancer edges).
           --variant dummy21       : replace the 21 real cancer feature nodes with 21
                                     one-hot dummy nodes (gene-cancer topology unchanged).
           --variant expr_real     : gene-cancer edges thresholded by REAL expression
                                     specificity (top-K per gene, from targets file).
           --variant expr_shuffled : same edge count / degree per gene as expr_real, but
                                     the per-gene specificity profile is PERMUTED so that
                                     edge identities do not carry biological signal.
         All four control graphs are trained with NAMR-only loss (CSP / C3L weights 0).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup functional_analysis

Input:  data/processed/xena_graph_esm.pt (+ gene_cancer_targets_5000.pt for expr_*).
Output: results/control/<variant>.pt
Run:    python scripts/79_build_control_graphs.py --variant ppi_only
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Source artifact relative paths.
SRC_GRAPH_RELATIVE = "xena_graph_esm.pt"
TARGETS_RELATIVE = "gene_cancer_targets_5000.pt"

SRC = os.path.join(config.DATA_PROC, SRC_GRAPH_RELATIVE)
TARGETS = os.path.join(config.DATA_PROC, TARGETS_RELATIVE)

# Output directory under config.result_path(...).
OUTPUT_SUBDIR = "control"
OUTDIR = config.result_path(OUTPUT_SUBDIR)

# Top-K per-gene specificity edges used for the expr_real / expr_shuffled variants.
# K was manually tuned to approximately match the original gene-cancer edge density.
EXPR_EDGE_TOP_K = 5

# Edge index dtype for PyG compatibility.
EDGE_TORCH_DTYPE = torch.int64

# Valid CLI variant names.
VARIANT_PPI_ONLY = "ppi_only"
VARIANT_DUMMY21 = "dummy21"
VARIANT_EXPR_REAL = "expr_real"
VARIANT_EXPR_SHUFFLED = "expr_shuffled"
ALL_VARIANTS = (
    VARIANT_PPI_ONLY,
    VARIANT_DUMMY21,
    VARIANT_EXPR_REAL,
    VARIANT_EXPR_SHUFFLED,
)

# Output file extension.
OUTPUT_FILE_EXT = ".pt"

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def load_graph():
    """
    @brief   Load the base source graph from the canonical path under DATA_PROC.
    @return  dict – HeteroData-style graph dictionary.
    """
    return torch.load(SRC, map_location=LOAD_MAP_LOCATION,
                      weights_only=TORCH_LOAD_WEIGHTS_ONLY)


def ppi_only():
    """
    @brief   Build the ppi_only control: remove every gene-cancer edge so that the
             cancer nodes become isolated and contribute no message-passing path to
             the gene encoder. The cancer node type is still retained (with its
             original node count) so the total parameter count stays aligned with
             the full graph for fair ablations.
    @return  Modified graph dictionary with empty gene<->cancer edge index tensors.
    """
    g = load_graph()
    # Number of genes kept for logging parity (currently unused in return).
    _ = g["meta"]["n_genes"]
    empty = torch.empty((2, 0), dtype=EDGE_TORCH_DTYPE)
    g[("gene", "expressed_in", "cancer")] = {"edge_index": empty}
    g[("cancer", "expresses", "gene")] = {"edge_index": empty}
    return g


def dummy21():
    """
    @brief   Build the dummy21 control: replace the real cancer feature vectors with
             one-hot dummy features while keeping exactly the same gene-cancer edge
             topology as the original graph. This isolates whether the graph's
             signal comes from the feature content of cancer nodes vs. just their
             connectivity structure.
    @return  Modified graph dictionary with one-hot cancer features.
    """
    g = load_graph()
    n_cancers = g["cancer"]["x"].shape[0]
    # One-hot dummy nodes, same tensor shape as the real cancer features.
    g["cancer"] = {"x": torch.eye(n_cancers), "num_nodes": n_cancers}
    # Keep the existing complete-bipartite gene-cancer edges exactly as-is.
    return g


def expr_edges(shuffle):
    """
    @brief   Build either expr_real or expr_shuffled variant: replace the complete-
             bipartite gene-cancer edge set with a top-K per-gene edge set derived
             from the C3L specificity matrix.
    @param   shuffle  bool – if True, permute each gene's specificity profile so
                       each gene keeps exactly the same degree K but its edge
                       partners are randomised; if False, use the real top-K.
    @return  Modified graph dictionary with new gene<->cancer edges.
    """
    g = load_graph()
    t = torch.load(TARGETS, map_location=LOAD_MAP_LOCATION,
                   weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    # Shape: [n_genes, n_cancers], C3L expression-specificity scores.
    spec = t["specificity"].numpy()
    n_genes, n_cancers = spec.shape

    # For each gene pick the EXPR_EDGE_TOP_K cancer types with highest specificity.
    order = np.argsort(-spec, axis=1)[:, :EXPR_EDGE_TOP_K]

    if shuffle:
        rng = np.random.RandomState(config.SEED)
        # Permute the per-gene specificity profile; this preserves each gene's
        # degree exactly (K) while destroying the gene-to-cancer mapping signal.
        for i in range(n_genes):
            perm = rng.permutation(n_cancers)
            order[i] = perm[:EXPR_EDGE_TOP_K]

    # Build the directed gene -> cancer edge_index tensor.
    rows = np.repeat(np.arange(n_genes), EXPR_EDGE_TOP_K)
    cols = order.reshape(-1)
    ei = torch.tensor(np.stack([rows, cols]), dtype=EDGE_TORCH_DTYPE)

    # Store both directions (gene -> cancer and cancer -> gene).
    g[("gene", "expressed_in", "cancer")] = {"edge_index": ei}
    g[("cancer", "expresses", "gene")] = {"edge_index": ei.flip(0)}
    return g


def main():
    """
    @brief   Entry point: parse CLI variant flag, dispatch to the correct builder,
             persist the resulting control graph under OUTDIR, and print a short
             summary including the resulting gene-cancer edge count when present.
    @return  None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", required=True, choices=ALL_VARIANTS,
        help="Which structural-control graph variant to build.",
    )
    args = parser.parse_args()

    # Route each variant to its builder.
    builders = {
        VARIANT_PPI_ONLY: ppi_only,
        VARIANT_DUMMY21: dummy21,
        VARIANT_EXPR_REAL: lambda: expr_edges(False),
        VARIANT_EXPR_SHUFFLED: lambda: expr_edges(True),
    }
    graph = builders[args.variant]()

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{args.variant}{OUTPUT_FILE_EXT}")
    torch.save(graph, out)

    # Report the gene-cancer edge count when the edge type exists.
    gc_edge_entry = graph.get(("gene", "expressed_in", "cancer"), {})
    gc_edge_index = gc_edge_entry.get("edge_index", None)
    gc_edge_count = gc_edge_index.size(1) if gc_edge_index is not None else "none"
    print(f"saved {out} | gene-cancer edges: {gc_edge_count}")


if __name__ == "__main__":
    main()
