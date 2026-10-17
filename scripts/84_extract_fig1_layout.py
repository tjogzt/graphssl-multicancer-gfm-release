#!/usr/bin/env python3
"""
@file    84_extract_fig1_layout.py
@brief   Extract the heterogeneous-network layout used by Fig. 1(a). A publication
         figure cannot render the full graph legibly; this script caches an honest,
         deterministic sample: the one-hop neighbourhood of the highest-confidence
         hub gene (default TP53), its strongest neighbours by PPI degree, all PPI
         edges induced by that set, plus representative cancer-type and patient
         nodes attached to the same gene set. The spring layout is computed once
         (at the configured seed) and cached as JSON so the figure renderer needs
         no graph file and produces identical output on any machine.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup audit_and_diagnostics

Input:  <DATA_PROC>/xena_graph_esm650_large.pt (the full heterogeneous graph).
Output: <RESULT_DIR>/fig1_subgraph_layout.json with keys:
        {hub, gene_names, edges_gene, edges_cancer, edges_patient, pos, provenance}.
Run:    python scripts/84_extract_fig1_layout.py [--hub TP53] [--neighbours 24]
"""
import argparse
import datetime
import json
import os
import sys

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Default input / output artifact names.
FULL_GRAPH = "xena_graph_esm650_large.pt"   # Input: full heterogeneous graph
OUT_JSON = "fig1_subgraph_layout.json"      # Output: cached Fig. 1(a) layout

# Default sampling parameters (CLI-overridable).
DEFAULT_HUB_GENE = "TP53"
DEFAULT_NEIGHBOURS = 24
DEFAULT_CANCER_NODES = 6
DEFAULT_PATIENT_NODES = 4

# Representative-node id offset families so synthetic node ids cannot collide
# with gene indices (which typically start at 0 and go up to ~14,000).
CANCER_ID_BASE_OFFSET = 100000
PATIENT_ID_BASE_OFFSET = 200000

# Edges per representative node connecting to the sampled gene set.
CANCER_EDGES_PER_NODE = 3
PATIENT_EDGES_PER_NODE = 2

# NetworkX spring-layout hyper-parameters.
SPRING_K = 0.9
SPRING_ITERATIONS = 200

# Output JSON indentation spaces.
JSON_INDENT = 1

# provenance ISO-8601 second precision.
ISO_TIMESPEC = "seconds"

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def main():
    """
    @brief   Entry point: parse CLI flags, sample the Fig. 1(a) subgraph, compute
             a deterministic NetworkX spring layout, serialise positions and
             provenance metadata as JSON.
    @return  None
    """
    parser = argparse.ArgumentParser(description="Cache the Fig. 1(a) network layout.")
    parser.add_argument(
        "--graph",
        default=os.path.join(config.DATA_PROC, FULL_GRAPH),
        help="Path of the full heterogeneous graph to sample from.",
    )
    parser.add_argument(
        "--hub",
        default=DEFAULT_HUB_GENE,
        help="Hub gene symbol that centres the one-hop neighbourhood sample.",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=DEFAULT_NEIGHBOURS,
        help="Number of strongest PPI neighbours of the hub to include.",
    )
    parser.add_argument(
        "--cancer-nodes",
        type=int,
        default=DEFAULT_CANCER_NODES,
        help="Representative cancer-type nodes to attach to the sampled gene set.",
    )
    parser.add_argument(
        "--patient-nodes",
        type=int,
        default=DEFAULT_PATIENT_NODES,
        help="Representative patient nodes to attach to the sampled gene set.",
    )
    args = parser.parse_args()

    config.set_seed()

    # Load full graph and build gene symbol -> index lookup.
    graph = torch.load(args.graph, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    gene_list = [str(g) for g in graph["meta"]["gene_list"]]
    symbol_to_index = {symbol: i for i, symbol in enumerate(gene_list)}
    ppi = graph[("gene", "ppi", "gene")]["edge_index"].numpy()
    print(f"Full graph: {len(gene_list):,} genes | {ppi.shape[1]:,} PPI edges")

    if args.hub not in symbol_to_index:
        raise KeyError(f"Hub gene {args.hub} is not present in the graph.")

    # PPI degree table used to rank hub neighbours deterministically.
    degree = np.bincount(ppi.reshape(-1), minlength=len(gene_list))
    hub_index = symbol_to_index[args.hub]

    # Collect one-hop neighbours of the hub.
    neighbours = set()
    for source, target in ppi.T:
        if source == hub_index:
            neighbours.add(int(target))
        elif target == hub_index:
            neighbours.add(int(source))

    # Sort neighbours by (-degree, gene_symbol) so the selection is reproducible.
    ranked = sorted(neighbours, key=lambda i: (-int(degree[i]), gene_list[i]))
    keep = [hub_index] + ranked[:args.neighbours]
    keep_set = set(keep)
    print(f"Hub {args.hub}: PPI degree {int(degree[hub_index])} | "
          f"neighbours kept {len(keep) - 1}")

    # Edges induced by the sampled gene set: stored once as undirected canonical pairs.
    gene_edges = [
        (int(a), int(b))
        for a, b in ppi.T
        if a in keep_set and b in keep_set and a < b
    ]
    print(f"Induced PPI edges inside sample: {len(gene_edges)}")

    # Representative cancer-type and patient nodes, each linked to RANDOM sampled genes.
    rng = np.random.RandomState(config.SEED)
    cancer_ids = list(range(
        CANCER_ID_BASE_OFFSET,
        CANCER_ID_BASE_OFFSET + args.cancer_nodes,
    ))
    patient_ids = list(range(
        PATIENT_ID_BASE_OFFSET,
        PATIENT_ID_BASE_OFFSET + args.patient_nodes,
    ))
    cancer_edges = [
        (int(rng.choice(keep)), c)
        for c in cancer_ids
        for _ in range(CANCER_EDGES_PER_NODE)
    ]
    patient_edges = [
        (int(rng.choice(keep)), p)
        for p in patient_ids
        for _ in range(PATIENT_EDGES_PER_NODE)
    ]

    # Assemble NetworkX graph and run deterministic spring layout.
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(keep, kind="gene")
    layout_graph.add_nodes_from(cancer_ids, kind="cancer")
    layout_graph.add_nodes_from(patient_ids, kind="patient")
    layout_graph.add_edges_from(gene_edges)
    layout_graph.add_edges_from(cancer_edges)
    layout_graph.add_edges_from(patient_edges)
    positions = nx.spring_layout(
        layout_graph,
        seed=config.SEED,
        k=SPRING_K,
        iterations=SPRING_ITERATIONS,
    )

    # Build JSON payload with localised indices for the figure renderer.
    payload = {
        "hub": args.hub,
        "gene_names": [gene_list[i] for i in keep],
        "gene_ids": keep,
        "cancer_ids": cancer_ids,
        "patient_ids": patient_ids,
        "edges_gene_local": [
            (keep.index(a), keep.index(b)) for a, b in gene_edges
        ],
        "edges_cancer_local": [
            (keep.index(a), c) for a, c in cancer_edges
        ],
        "edges_patient_local": [
            (keep.index(a), p) for a, p in patient_edges
        ],
        "pos": {
            str(node): [float(xy[0]), float(xy[1])]
            for node, xy in positions.items()
        },
        "provenance": {
            "source_graph": os.path.basename(args.graph),
            "sampling": (
                f"one-hop neighbourhood of {args.hub} "
                f"(top {args.neighbours} by PPI degree) + "
                f"{args.cancer_nodes} cancer nodes + "
                f"{args.patient_nodes} patient nodes"
            ),
            "layout": (
                f"networkx.spring_layout(seed={config.SEED}, "
                f"k={SPRING_K}, iterations={SPRING_ITERATIONS})"
            ),
            "generated": datetime.datetime.now().isoformat(timespec=ISO_TIMESPEC),
        },
    }

    out_path = config.result_path(OUT_JSON)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=JSON_INDENT)
    print(f"Saved layout cache: {out_path}")


if __name__ == "__main__":
    main()
