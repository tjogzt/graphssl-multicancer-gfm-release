#!/usr/bin/env python3
"""
@file    123_report_control_graphs.py
@brief   Report structural properties of the gene--cancer control graphs
         (original, equal-volume, rewired random-endpoint multigraph). The
         rewired control must be described honestly as a *multigraph* that
         keeps duplicate edges rather than as a clean simple-graph control;
         this script quantifies: raw edge entries, deduplicated edge count,
         per-gene unique-cancer-neighbour count, per-gene and per-cancer
         degree distributions (multiplicity-inclusive), and the number of
         duplicate edge entries.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt (original),
        data/processed/xena_graph_esm_eqvol.pt (equal-volume),
        data/processed/xena_graph_esm_rewired.pt (rewired).
Output: stdout table + results/control/graph_properties.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/123_report_control_graphs.py
"""
import json
import os

import numpy as np
import torch

import config

# =============================================================================
# File-level constants
# =============================================================================
# Canonical edge triple inspected by the structural summary: gene -> expressed_in
# -> cancer (the only gene-cancer relation in the heterograph).
EDGE_TRIPLE = ("gene", "expressed_in", "cancer")
EDGE_INDEX_KEY = "edge_index"
META_N_GENES_KEY = "n_genes"
META_N_CANCERS_KEY = "n_cancers"

# Graph variant -> basename mapping; basenames are resolved against
# config.DATA_PROC so a change in the processed-data directory only touches
# the config module.
GRAPH_VARIANTS = (
    ("original",  "xena_graph_esm.pt"),
    ("eqvol",     "xena_graph_esm_eqvol.pt"),
    ("rewired",   "xena_graph_esm_rewired.pt"),
)

# Result-path tag consumed by config.result_path() for the JSON artefact.
RESULT_TAG = "control"
JSON_FILENAME = "graph_properties.json"

# Print formatting.
HEADER_FORMAT = (
    f"{{:<10}}{{:>9}}{{:>8}}{{:>7}} | "
    f"{{:>28}} | {{:>26}} | {{:>28}}"
)
HEADER_FIELDS = (
    "graph", "entries", "dedup", "dup",
    "gene-uniq-neigh (min/med/max)",
    "gene-deg (min/med/max)",
    "cancer-deg (min/med/max)",
)
CELL_FORMAT = "{:.0f}/{:.0f}/{:.0f}"
JSON_INDENT = 2

# torch.load map location: CPU is sufficient because we only inspect index
# tensors and do not run any forward passes.
LOAD_MAP_DEVICE = "cpu"


def _summarise(x):
    """Return a 4-number summary for a 1-D numeric array.

    @brief  Internal helper used for every degree-like vector we report.
    @param  x   1-D numpy array of numerics.
    @return Dict with keys ``{min, median, max, mean}``, each cast to plain
            Python ``float`` so the result is JSON-serialisable.
    """
    return dict(
        min=float(x.min()),
        median=float(np.median(x)),
        max=float(x.max()),
        mean=float(x.mean()),
    )


def describe(name, path):
    """Produce a structural summary dict for a single heterograph file.

    @brief  Load the graph, inspect the ``gene -> expressed_in -> cancer``
            edge_index tensor, and compute:
              - raw edge entry count (includes duplicates);
              - deduplicated edge count and duplicate residual;
              - per-gene unique-cancer-neighbour distribution;
              - per-gene and per-cancer degree (multiplicity-inclusive).
    @param  name   Stable human-readable variant name (e.g. ``"rewired"``).
    @param  path   Absolute path to the ``*.pt`` heterograph file.
    @return Dict suitable for JSON serialisation with keys: ``name``,
            ``edge_entries``, ``dedup_edges``, ``duplicate_entries``,
            ``gene_unique_cancer_neighbors``, ``gene_degree``,
            ``cancer_degree``.
    """
    graph = torch.load(path, map_location=LOAD_MAP_DEVICE, weights_only=False)
    edge_index = graph[EDGE_TRIPLE][EDGE_INDEX_KEY].numpy()   # shape: [2, E]
    n_genes = graph["meta"][META_N_GENES_KEY]
    n_cancers = graph["meta"][META_N_CANCERS_KEY]
    n_entries = edge_index.shape[1]

    unique_pairs = set()
    for i in range(n_entries):
        unique_pairs.add((int(edge_index[0, i]), int(edge_index[1, i])))
    n_unique = len(unique_pairs)
    n_duplicate = n_entries - n_unique

    # --- Per-gene unique cancer neighbours (set semantics per gene) ----------
    uniq_by_gene = {}
    for gid, cid in unique_pairs:
        uniq_by_gene.setdefault(gid, set()).add(cid)
    gene_uniq_vec = np.array(
        [len(uniq_by_gene.get(g, set())) for g in range(n_genes)],
        dtype=float,
    )

    # --- Degree distributions (include multiplicity; multigraph semantics) ---
    gene_degree = np.bincount(edge_index[0], minlength=n_genes).astype(float)
    cancer_degree = np.bincount(edge_index[1], minlength=n_cancers).astype(float)

    return {
        "name": name,
        "edge_entries": int(n_entries),
        "dedup_edges": int(n_unique),
        "duplicate_entries": int(n_duplicate),
        "gene_unique_cancer_neighbors": _summarise(gene_uniq_vec),
        "gene_degree": _summarise(gene_degree),
        "cancer_degree": _summarise(cancer_degree),
    }


def main():
    """Summarise every graph variant, print a table, and write JSON.

    @brief  Iterate over GRAPH_VARIANTS, build the row list, write the JSON
            payload, then emit a fixed-width human-readable table to stdout.
    @return None.
    """
    proc_dir = config.DATA_PROC
    rows = [
        describe(name, os.path.join(proc_dir, basename))
        for name, basename in GRAPH_VARIANTS
    ]
    # Drop the "name" key from the dict representation inside JSON (the outer
    # dict is already keyed by the variant name).
    json_payload = {
        row["name"]: {k: v for k, v in row.items() if k != "name"}
        for row in rows
    }
    out_dir = config.result_path(RESULT_TAG)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, JSON_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(json_payload, fh, indent=JSON_INDENT)
    # --- English-only stdout ---------------------------------------------------
    print(f"[written] {out_path}\n")
    header = HEADER_FORMAT.format(*HEADER_FIELDS)
    print(header)
    print("-" * len(header))
    for row in rows:
        gu = row["gene_unique_cancer_neighbors"]
        gd = row["gene_degree"]
        cd = row["cancer_degree"]
        gu_cell = CELL_FORMAT.format(gu["min"], gu["median"], gu["max"]).rjust(28)
        gd_cell = CELL_FORMAT.format(gd["min"], gd["median"], gd["max"]).rjust(26)
        cd_cell = CELL_FORMAT.format(cd["min"], cd["median"], cd["max"]).rjust(28)
        print(
            f"{row['name']:<10}"
            f"{row['edge_entries']:>9}"
            f"{row['dedup_edges']:>8}"
            f"{row['duplicate_entries']:>7} | "
            f"{gu_cell} | {gd_cell} | {cd_cell}"
        )


if __name__ == "__main__":
    main()
