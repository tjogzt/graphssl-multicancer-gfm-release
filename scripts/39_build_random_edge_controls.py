#!/usr/bin/env python3
"""
@file    39_build_random_edge_controls.py
@brief   Build two random gene-cancer edge control graphs (R-A equal-volume
         random-subset and R-B volume-matched rewired) for ablation experiments,
         preserving gene features and PPI edges while isolating the effect of
         gene-cancer topology.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Build random gene-cancer edge control graphs (equal-data-volume control). R-A keeps
the same total gene-cancer edge count as the single-cancer control by giving each
cancer a random gene subset, and R-B keeps the multi-cancer edge count but rewires
each edge's cancer endpoint uniformly at random. Both preserve gene features and PPI
edges, isolating the effect of the gene-cancer structure on downstream metrics.

Input:  config.DATA_PROC / SRC_GRAPH_FILE (from 20_build_graph_esm.py).
Output: config.DATA_PROC / OUT_EQVOL_FILE and OUT_REWIRED_FILE plus stdout logs.
Run:    python scripts/39_build_random_edge_controls.py
"""
import os

import torch
import numpy as np

import config

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================

# --- Input / output artifact filenames (resolved under config.DATA_PROC) ---
# Source graph produced by 20_build_graph_esm.py (ESM-featured heterograph).
SOURCE_GRAPH_FILENAME = "xena_graph_esm.pt"
# R-A: Equal-volume random-subset control graph output.
OUTPUT_EQVOL_FILENAME = "xena_graph_esm_eqvol.pt"
# R-B: Volume-matched rewired control graph output.
OUTPUT_REWIRED_FILENAME = "xena_graph_esm_rewired.pt"

# --- R-A design constant ---
# Number of genes connected to each cancer in the R-A (equal-volume) control.
# Chosen so that the total gene-cancer edge count
#   = n_cancers * GENES_PER_CANCER_RANDOM_SUBSET
# matches the single-cancer baseline control's edge count.
GENES_PER_CANCER_RANDOM_SUBSET = 238

# Edge-type relation keys shared between controls and source graph.
REL_GENE_TO_CANCER = ("gene", "expressed_in", "cancer")
REL_CANCER_TO_GENE = ("cancer", "expresses", "gene")


# =========================================================================
# Source-loading helpers
# =========================================================================

def load_source_graph(data_proc_dir, filename):
    """
    @brief   Load the source ESM-featured graph from disk.

    @param   data_proc_dir  Absolute path to config.DATA_PROC.
    @param   filename       Basename of the source graph .pt.
    @return  Tuple (graph_dict, n_genes, n_cancers). n_genes is resolved from
             graph['gene']['num_nodes'] with a fallback to meta['n_genes'].
    """
    full_path = os.path.join(data_proc_dir, filename)
    graph = torch.load(full_path, map_location="cpu", weights_only=False)
    # Resolve node counts: prefer explicit num_nodes field, fall back to meta.
    n_genes = (
        graph["gene"]["num_nodes"]
        if "num_nodes" in graph["gene"]
        else graph["meta"]["n_genes"]
    )
    n_cancers = graph["meta"]["n_cancers"]
    print(f"[Load] Source graph: {full_path}")
    print(f"  Genes: {n_genes:,} | Cancers: {n_cancers}")
    return graph, n_genes, n_cancers


# =========================================================================
# R-A: Equal-volume, random subset per cancer
# =========================================================================

def build_equal_volume_control(source_graph, n_genes, n_cancers, rng,
                               genes_per_cancer, seed):
    """
    @brief   Build control R-A: each cancer links to a fresh random gene subset
             of fixed size, making total edge count == n_cancers * genes_per_cancer.

    @param   source_graph     Dict loaded from the source ESM graph.
    @param   n_genes          Total number of gene nodes (graph order).
    @param   n_cancers        Total number of cancer nodes.
    @param   rng              np.random.RandomState used for reproducible draws.
    @param   genes_per_cancer Number of distinct genes sampled per cancer.
    @param   seed             Integer seed value stored in metadata for traceability.
    @return  Modified graph dict copy with R-A gene<->cancer edges and metadata updated.
    @note    All other graph data (features, PPI edges, patient info) is copied as-is.
    """
    # Build edge_index: for each cancer sample `genes_per_cancer` genes without replacement.
    rows_gene_idx, rows_cancer_idx = [], []
    for c in range(n_cancers):
        # Draw a distinct random gene subset of the requested size for each cancer.
        sampled_genes = rng.choice(n_genes, genes_per_cancer, replace=False)
        rows_gene_idx.extend(sampled_genes.tolist())
        rows_cancer_idx.extend([c] * genes_per_cancer)

    edge_index_a = torch.tensor(
        [rows_gene_idx, rows_cancer_idx], dtype=torch.long
    )

    # Shallow-copy the source graph dict to avoid mutating the original in caller.
    graph_a = source_graph.copy()
    # Overwrite only the two gene<->cancer relation tensors.
    graph_a[REL_GENE_TO_CANCER] = {"edge_index": edge_index_a}
    graph_a[REL_CANCER_TO_GENE] = {"edge_index": edge_index_a.flip(0)}
    # Preserve original metadata but append a human-readable control note.
    graph_a["meta"] = dict(source_graph["meta"])
    graph_a["meta"]["edge_note"] = (
        f"R-A control: equal-volume random gene-cancer edges "
        f"({edge_index_a.size(1)} total; {genes_per_cancer} genes/cancer); "
        f"seed={seed}"
    )
    return graph_a, edge_index_a.size(1)


# =========================================================================
# R-B: Same total volume, cancer-endpoint rewired
# =========================================================================

def build_rewired_volume_control(source_graph, n_cancers, rng, seed):
    """
    @brief   Build control R-B: preserve the full multi-cancer edge volume but
             randomly rewire each edge's CANCER endpoint uniformly at random.
             Gene endpoints (row 0 of edge_index) are kept FIXED, isolating the
             cancer-assignment structure as the only varying factor.

    @param   source_graph  Dict loaded from the source ESM graph.
    @param   n_cancers     Total number of cancer nodes (rewire range).
    @param   rng           np.random.RandomState used for reproducible draws.
    @param   seed          Integer seed value stored in metadata for traceability.
    @return  Modified graph dict copy with R-B gene<->cancer edges and metadata updated.
    """
    # Clone the source gene->cancer edge_index so we don't mutate the source tensor.
    edge_index_b = source_graph[REL_GENE_TO_CANCER]["edge_index"].clone()
    total_edges = edge_index_b.size(1)
    # Rewire ONLY the cancer endpoint (row index 1); keep gene endpoints (row 0) unchanged.
    rewired_cancer_ids = rng.randint(0, n_cancers, size=total_edges)
    edge_index_b[1] = torch.from_numpy(rewired_cancer_ids).long()

    # Shallow-copy the source graph dict and overwrite the two relations.
    graph_b = source_graph.copy()
    graph_b[REL_GENE_TO_CANCER] = {"edge_index": edge_index_b}
    graph_b[REL_CANCER_TO_GENE] = {"edge_index": edge_index_b.flip(0)}
    graph_b["meta"] = dict(source_graph["meta"])
    graph_b["meta"]["edge_note"] = (
        f"R-B control: rewired random gene-cancer edges "
        f"({edge_index_b.size(1)} total, volume-matched to multi-cancer source); "
        f"seed={seed}"
    )
    return graph_b, edge_index_b.size(1)


# =========================================================================
# Save helpers
# =========================================================================

def save_control_graph(data_proc_dir, filename, graph_dict, control_label):
    """
    @brief   Persist a control graph dict and print a summary line.

    @param   data_proc_dir  Output directory.
    @param   filename       Basename of the saved .pt file.
    @param   graph_dict     Graph dict to serialize.
    @param   control_label  Short human-readable label (e.g. "R-A", "R-B") for logs.
    @return  Full output path string.
    """
    out_path = os.path.join(data_proc_dir, filename)
    torch.save(graph_dict, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[{control_label}] Saved {out_path} ({size_mb:.1f} MB)")
    return out_path


# =========================================================================
# Main entry
# =========================================================================

def main():
    """
    @brief   Driver: load source graph, build R-A and R-B controls, save both.

    @return  None. Prints summary of each control.
    """
    print("=" * 60)
    print("Phase 3.9: Building random gene-cancer edge controls (R-A / R-B)")
    print("=" * 60)

    # --- Load source graph ---
    source_graph, n_genes, n_cancers = load_source_graph(
        config.DATA_PROC, SOURCE_GRAPH_FILENAME
    )

    # --- Deterministic seeded random state ---
    rng = np.random.RandomState(config.SEED)
    print(f"  Random seed: {config.SEED} (reproducible draws)")

    # --- R-A: equal-volume random subsets ---
    print("\n[R-A] Building equal-volume random-subset control ...")
    graph_a, n_edges_a = build_equal_volume_control(
        source_graph, n_genes, n_cancers, rng,
        GENES_PER_CANCER_RANDOM_SUBSET, config.SEED
    )
    save_control_graph(
        config.DATA_PROC, OUTPUT_EQVOL_FILENAME, graph_a, "R-A"
    )
    print(f"  Gene->Cancer edges: {n_edges_a:,} "
          f"({GENES_PER_CANCER_RANDOM_SUBSET} genes/cancer x {n_cancers} cancers)")

    # --- R-B: same volume, rewired cancer endpoints ---
    print("\n[R-B] Building volume-matched rewired control ...")
    graph_b, n_edges_b = build_rewired_volume_control(
        source_graph, n_cancers, rng, config.SEED
    )
    save_control_graph(
        config.DATA_PROC, OUTPUT_REWIRED_FILENAME, graph_b, "R-B"
    )
    print(f"  Gene->Cancer edges: {n_edges_b:,} "
          f"(volume-matched, rewired cancer endpoints)")

    print("\n" + "=" * 60)
    print("Phase 3.9 complete: both random controls saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
