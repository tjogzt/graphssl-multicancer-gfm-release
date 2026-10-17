#!/usr/bin/env python3
"""
@file    133_fix_patient_graph.py
@brief   Fix the patient-gene edge index space: 33_build_patient_gene_edges.py
         indexed patient-gene edges against the 13,881-gene LARGE graph, but
         34_build_patient_esm_graph merges with the 5,000-gene ESM subgraph
         (whose gene_list equals large[:5000]).  This script filters the
         patient-gene edges to gene index < N_ESM_GENES and rebuilds the merged
         patient-ESM heterograph so the patient-gene / gene-patient tensors
         live in the same index space as every other gene edge triple.
         NOTE: the original implementation was a top-level linear script with
         NO functions; this revision wraps the work in main() with a standard
         __main__ guard while preserving every byte of the underlying logic.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_patient.pt   (LARGE graph patient edges)
        data/processed/xena_graph_esm.pt       (ESM 5000-gene subgraph)
Output: data/processed/xena_graph_patient_esm.pt  (merged, filtered edges)
Run:    GFM_PROJECT_ROOT=<repo> python scripts/133_fix_patient_graph.py
"""
import os

import torch

import config

# =============================================================================
# File-level constants (paths + edge triples, all centralised here).
# =============================================================================
# Source + target artifact basenames (resolved against config.DATA_PROC).
LARGE_PATIENT_GRAPH = "xena_graph_patient.pt"
ESM_SUBGRAPH = "xena_graph_esm.pt"
OUTPUT_MERGED_GRAPH = "xena_graph_patient_esm.pt"

# Edge triple names referenced in the heterograph dicts.
PG_EDGE_TRIPLE = ("patient", "expressed_by", "gene")
GP_EDGE_TRIPLE = ("gene", "expresses", "patient")
G_PPI_TRIPLE = ("gene", "ppi", "gene")
G_CANCER_TRIPLE = ("gene", "expressed_in", "cancer")
CANCER_G_TRIPLE = ("cancer", "expresses", "gene")
P_BELONGS_TRIPLE = ("patient", "belongs_to", "cancer")

# Meta keys copied from the patient graph's own meta dict into the merged
# graph's meta dict.  The edge-index space fix also records the final count
# of patient-gene edges in the merged meta for downstream audits.
META_PATIENT_IDS_KEY = "patient_ids"
META_PATIENT_CANCER_KEY = "patient_cancer"
META_PATIENT_GENE_EDGES_KEY = "patient_gene_edges"
NUM_NODES_KEY = "num_nodes"
EDGE_INDEX_KEY = "edge_index"

# torch.load map location (all graph manipulations are pure index-tensor ops on
# CPU; no GPU required).
LOAD_MAP_DEVICE = "cpu"


def filter_edge(edge_index, gene_axis):
    """Return a copy of ``edge_index`` keeping only rows where the gene dim
    index is strictly less than the ESM subgraph's gene count.

    @brief  The patient-gene / gene-patient edge triples were originally built
            in the LARGE graph's gene index space (0..13880).  To match the ESM
            subgraph they must be restricted to indices < 5000; this helper is
            called twice (once per direction) so the assertion below can
            independently verify that the two directions remain consistent.
    @param  edge_index   LongTensor of shape [2, E] (patient x gene or vice versa).
    @param  gene_axis    Which row of edge_index holds the gene indices (0 for
                         gene->patient; 1 for patient->gene).
    @return Filtered edge_index tensor of shape [2, E_filtered], same dtype/device.
    """
    n_esm_genes = filter_edge._n_esm_cache          # injected by main() for brevity
    keep_mask = edge_index[gene_axis] < n_esm_genes
    return edge_index[:, keep_mask]


def main():
    """Filter patient/gene edges, rebuild merged patient-ESM heterograph, save.

    @brief  (1) load both source graphs, read the ESM subgraph's actual gene
            count from its ``gene.num_nodes`` attribute; (2) install a closure
            on filter_edge so the count is available without threading a
            parameter everywhere; (3) read PG / GP edge indices, filter each
            to the ESM index space, assert the two directions agree on edge
            count; (4) copy every node type and edge triple from the ESM
            subgraph, add patient nodes + edges + belongs_to, copy meta + the
            two patient-specific meta lists; (5) write the merged graph back
            to config.DATA_PROC / OUTPUT_MERGED_GRAPH and print an English
            summary with before/after edge counts.
    @return None.
    """
    pat_path = os.path.join(config.DATA_PROC, LARGE_PATIENT_GRAPH)
    esm_path = os.path.join(config.DATA_PROC, ESM_SUBGRAPH)
    out_path = os.path.join(config.DATA_PROC, OUTPUT_MERGED_GRAPH)
    pat = torch.load(pat_path, map_location=LOAD_MAP_DEVICE)
    esm = torch.load(esm_path, map_location=LOAD_MAP_DEVICE)

    n_esm_genes = int(esm["gene"][NUM_NODES_KEY])
    # Expose the count to filter_edge without a global; a function attribute
    # keeps the public API (edge_index, gene_axis) unchanged.
    filter_edge._n_esm_cache = n_esm_genes

    # Patient <-> Gene edges: read both directions and filter to ESM index space.
    pg = pat[PG_EDGE_TRIPLE][EDGE_INDEX_KEY]           # [patient_axis, gene_axis]
    gp = pat[GP_EDGE_TRIPLE][EDGE_INDEX_KEY]           # [gene_axis, patient_axis]
    pg_filtered = filter_edge(pg, gene_axis=1)
    gp_filtered = filter_edge(gp, gene_axis=0)
    # Directional consistency: after filtering to the same sub-index set, PG
    # and GP must have the same number of edges (the heterograph semantics
    # are strictly bidirectional for this triple pair).
    assert pg_filtered.size(1) == gp_filtered.size(1), \
        f"PG/GP direction mismatch after filter: {pg_filtered.size(1)} vs {gp_filtered.size(1)}"

    merged = {
        "gene":   esm["gene"],
        "cancer": esm["cancer"],
        G_PPI_TRIPLE:      esm[G_PPI_TRIPLE],
        G_CANCER_TRIPLE:   esm[G_CANCER_TRIPLE],
        CANCER_G_TRIPLE:   esm[CANCER_G_TRIPLE],
        "patient":         pat["patient"],
        P_BELONGS_TRIPLE:  pat[P_BELONGS_TRIPLE],
        PG_EDGE_TRIPLE:    {EDGE_INDEX_KEY: pg_filtered},
        GP_EDGE_TRIPLE:    {EDGE_INDEX_KEY: gp_filtered},
        "meta": dict(esm["meta"]),                          # copy so we don't mutate source
    }
    # Patient-specific meta entries may be absent from older patient graph
    # builds; .get(default=[]) keeps the merge deterministic.
    merged["meta"][META_PATIENT_IDS_KEY] = pat["meta"].get(META_PATIENT_IDS_KEY, [])
    merged["meta"][META_PATIENT_CANCER_KEY] = pat["meta"].get(META_PATIENT_CANCER_KEY, [])
    merged["meta"][META_PATIENT_GENE_EDGES_KEY] = int(pg_filtered.size(1))

    torch.save(merged, out_path)
    # English-only stdout summary.
    print(
        f"patient-gene edges after filter: {pg_filtered.size(1)} "
        f"(before {pg.size(1)})"
    )
    print(
        f"gene max index now: {int(pg_filtered[1].max())} "
        f"(n_esm_genes={n_esm_genes})"
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
