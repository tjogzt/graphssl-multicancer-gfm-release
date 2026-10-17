#!/usr/bin/env python3
"""
@file    36_build_graph_variants.py
@brief   Build the three graph variants used for the scale-up and held-out-cancer
         experiments through a single parameterised entry point:
           (1) esm650_full  – 13,881 genes with ESM-2 650M features reduced via PCA-256.
           (2) esm20c       – ESM-8M graph with one cancer type (default PAAD) held out.
           (3) esm650_20c   – ESM-650M graph with one cancer type held out.
         The *20c variants remap cancer indices, filter gene-cancer edges in both
         directions, and rebuild C3L classification targets so class ids align with
         the retained cancer subset.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Input:  Processed graphs and embeddings under config.DATA_PROC / config.CLOUD_DIR.
Output: esm650_full: xena_graph_esm650_large.pt + gene_cancer_targets_13881.pt
        esm20c     : xena_graph_esm_20c.pt     + gene_cancer_targets_20c.pt
        esm650_20c : xena_graph_esm650_20c.pt  + gene_cancer_targets_13881_20c.pt
Run:    python scripts/36_build_graph_variants.py --variant esm650_full
        python scripts/36_build_graph_variants.py --variant esm20c --exclude-cancer PAAD
        python scripts/36_build_graph_variants.py --variant esm650_20c
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Default artifact names (overridable through CLI flags).
BASE_LARGE = "xena_graph_large.pt"                  # Large-graph base (13,881 genes)
BASE_ESM = "xena_graph_esm.pt"                      # 8M-feature base graph
EMB_650M = "esm2_650m_gene_embeddings.pt"           # ESM-2 650M gene embeddings
TARGETS_5000 = "c3l_targets.pt"                     # C3L targets for the esm20c variant
TARGETS_FULL = "gene_cancer_targets.pt"             # Full C3L targets (13881-gene order)
OUT_650_FULL = "xena_graph_esm650_large.pt"         # esm650_full output graph
OUT_TARGETS_13881 = "gene_cancer_targets_13881.pt"  # esm650_full output targets
OUT_20C_ESM = "xena_graph_esm_20c.pt"               # esm20c output graph
OUT_20C_650 = "xena_graph_esm650_20c.pt"            # esm650_20c output graph
OUT_TARGETS_20C = "gene_cancer_targets_20c.pt"              # esm20c output targets
OUT_TARGETS_13881_20C = "gene_cancer_targets_13881_20c.pt"  # esm650_20c output targets

# Cancer type held out by the *20c variants when no override is given.
DEFAULT_EXCLUDE_CANCER = "PAAD"

# ESM-2 650M model identifiers written into the output graph metadata.
ESM_650_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_650_RAW_DIM = 1280

# Explained variance ratio print precision for PCA summary.
PCA_EXPLAINED_VAR_PRECISION = 3

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def load(path):
    """
    @brief   Load a pickled graph / target bundle from disk.
    @param   path  str – filesystem path of the .pt file to load.
    @return  The deserialised Python object (typically a HeteroData-style dict).
    @note    Uses weights_only=False because HeteroData dicts contain custom
             containers that are not pure tensors.
    """
    return torch.load(path, map_location=LOAD_MAP_LOCATION,
                      weights_only=TORCH_LOAD_WEIGHTS_ONLY)


def build_esm650_full(args):
    """
    @brief   Build the 13,881-gene graph where gene features are ESM-2 650M
             embeddings reduced to PCA-256, and realign the full C3L targets
             to the large-gene gene ordering.
    @param   args  argparse.Namespace – parsed CLI flags (pca_dim, base_graph).
    @return  None
    @note    Both graph and targets artefacts are written to config.DATA_PROC.
    """
    base = load(args.base_graph or os.path.join(config.DATA_PROC, BASE_LARGE))
    gene_list = list(base["meta"]["gene_list"])
    print(f"Large graph genes: {len(gene_list)}")

    # Load full ESM-2 650M embeddings and stack them in the same order as gene_list.
    embeddings = load(os.path.join(config.CLOUD_DIR, EMB_650M))["embeddings"]
    matrix = np.stack([embeddings[g].astype(np.float64) for g in gene_list])
    print(f"650M feature matrix shape: {matrix.shape}")

    # Standardise per dimension before PCA so heterogeneous variances do not dominate.
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=args.pca_dim, random_state=config.SEED).fit(scaled)
    reduced = pca.transform(scaled).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"PCA-{args.pca_dim}: shape {reduced.shape} | "
          f"explained variance {explained:.{PCA_EXPLAINED_VAR_PRECISION}f}")

    # Assemble the full heterogenous graph structure reusing base edges and metadata.
    graph = {
        "gene": {"x": torch.tensor(reduced), "num_nodes": len(gene_list)},
        "cancer": base["cancer"],
        "patient": base["patient"],
        ("gene", "ppi", "gene"): base[("gene", "ppi", "gene")],
        ("gene", "expressed_in", "cancer"): base[("gene", "expressed_in", "cancer")],
        ("cancer", "expresses", "gene"): base[("cancer", "expresses", "gene")],
        ("patient", "belongs_to", "cancer"): base[("patient", "belongs_to", "cancer")],
        "meta": {
            "gene_list": gene_list,
            "cancer_list": base["meta"]["cancer_list"],
            "cancer_counts": base["meta"]["cancer_counts"],
            "n_genes": len(gene_list),
            "n_cancers": base["meta"]["n_cancers"],
            "n_samples": base["meta"]["n_samples"],
            "embedding_dim": args.pca_dim,
            "esm_model": ESM_650_MODEL_NAME,
            "esm_dim": ESM_650_RAW_DIM,
        },
    }
    graph_path = os.path.join(config.DATA_PROC, OUT_650_FULL)
    torch.save(graph, graph_path)
    print(f"Saved graph: {graph_path}")

    # Align C3L targets with the large-graph gene order by re-indexing per gene.
    targets = load(os.path.join(config.CLOUD_DIR, TARGETS_FULL))
    gene_to_index = {g: i for i, g in enumerate(targets["gene_list"])}
    index = [gene_to_index[g] for g in gene_list if g in gene_to_index]
    print(f"Targets matched to graph genes: {len(index)}")

    torch.save(
        {
            "targets": targets["targets"][index].long(),
            "gene_list": gene_list,
            "cancer_list": targets["cancer_list"],
            "specificity": targets["specificity"][index],
        },
        os.path.join(config.DATA_PROC, OUT_TARGETS_13881),
    )
    print("Saved re-aligned targets for 13881 genes")


def drop_cancer(graph, cancer_list, exclude):
    """
    @brief   Remove one cancer type from a graph and remap the remaining cancer
             indices so they stay dense (0..len(keep)-1). Gene-cancer edges are
             filtered and remapped consistently in both directions.
    @param   graph        dict – HeteroData-style graph dict (modified in place).
    @param   cancer_list  list – ordered list of cancer-type names prior to removal.
    @param   exclude      str  – the cancer-type name to remove.
    @return  Tuple (graph, keep, filtered): the updated graph, the list of original
             cancer indices that were kept, and the remapped gene->cancer edge_index
             tensor (so the caller can print stats).
    @note    This function mutates the input graph dictionary in place.
    """
    # Determine which cancer indices survive.
    keep = [i for i, c in enumerate(cancer_list) if c != exclude]
    index_map = {old: new for new, old in enumerate(keep)}

    # Subset the cancer node features to the retained cancers only.
    graph["cancer"] = {
        "x": graph["cancer"]["x"][keep],
        "num_nodes": len(keep),
    }

    # Filter gene-cancer edges where the cancer end belongs to a retained type,
    # then remap the cancer-side indices so they remain dense 0..len(keep)-1.
    edge_index = graph[("gene", "expressed_in", "cancer")]["edge_index"]
    mask = torch.tensor([int(i) in index_map for i in edge_index[1].tolist()])
    filtered = edge_index[:, mask]
    filtered[1] = torch.tensor([index_map[int(i)] for i in filtered[1].tolist()])

    # Write back filtered edge indices in both directions.
    graph[("gene", "expressed_in", "cancer")] = {"edge_index": filtered}
    graph[("cancer", "expresses", "gene")] = {"edge_index": filtered.flip(0)}

    # Update graph metadata to reflect the post-drop state.
    graph["meta"]["cancer_list"] = [cancer_list[i] for i in keep]
    graph["meta"]["n_cancers"] = len(keep)

    return graph, keep, filtered


def build_20c(args):
    """
    @brief   Drop one cancer type from a base graph and rebuild multi-class C3L
             targets based only on the retained cancer columns.
    @param   args  argparse.Namespace – parsed CLI flags (variant, exclude_cancer,
                   base_graph).
    @return  None
    @note    Behaviour is parameterised by args.variant: "esm650_20c" uses the
             esm650_full artefacts as its base; "esm20c" uses the 8M ESM graph.
    """
    is_650 = args.variant == "esm650_20c"
    base_path = args.base_graph or os.path.join(
        config.DATA_PROC, OUT_650_FULL if is_650 else BASE_ESM
    )
    graph = load(base_path)
    cancer_list = list(graph["meta"]["cancer_list"])
    print(f"Cancers before drop: {len(cancer_list)} | excluding: {args.exclude_cancer}")

    # Remove the held-out cancer and remap remaining cancer indices.
    graph, keep, filtered = drop_cancer(graph, cancer_list, args.exclude_cancer)
    out_graph = os.path.join(config.DATA_PROC, OUT_20C_650 if is_650 else OUT_20C_ESM)
    torch.save(graph, out_graph)
    print(f"Saved graph: {out_graph} | gene-cancer edges retained: {filtered.size(1)}")

    # Rebuild top-1 classification targets over the retained cancer columns only.
    targets_file = OUT_TARGETS_13881 if is_650 else TARGETS_5000
    targets = load(os.path.join(config.DATA_PROC, targets_file))
    specificity = targets["specificity"][:, keep]
    top1 = specificity.argmax(1)

    out_targets = os.path.join(
        config.DATA_PROC,
        OUT_TARGETS_13881_20C if is_650 else OUT_TARGETS_20C,
    )
    torch.save(
        {
            "targets": top1.long(),
            "gene_list": targets["gene_list"],
            "cancer_list": [cancer_list[i] for i in keep],
            "specificity": specificity,
        },
        out_targets,
    )
    print(f"Saved targets: {out_targets} | distinct classes: {top1.unique().numel()}")


def main():
    """
    @brief   Entry point: parse CLI flags, set seeds, and dispatch to the
             appropriate variant builder.
    @return  None
    """
    parser = argparse.ArgumentParser(description="Build a molecular graph variant.")
    parser.add_argument(
        "--variant", required=True,
        choices=["esm650_full", "esm20c", "esm650_20c"],
    )
    parser.add_argument(
        "--exclude-cancer", default=DEFAULT_EXCLUDE_CANCER,
        help="Cancer type held out by the *20c variants",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=config.EMBEDDING_DIM,
        help="Target PCA dimensionality for esm650_full gene features",
    )
    parser.add_argument(
        "--base-graph", default=None,
        help="Override the default input graph path",
    )
    args = parser.parse_args()

    config.set_seed()

    if args.variant == "esm650_full":
        build_esm650_full(args)
    else:
        build_20c(args)


if __name__ == "__main__":
    main()
