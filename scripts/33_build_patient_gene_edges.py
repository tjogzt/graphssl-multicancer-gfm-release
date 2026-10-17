#!/usr/bin/env python3
"""
@file    33_build_patient_gene_edges.py
@brief   Attach patient-gene expression edges to the pan-cancer heterogeneous graph
         so that patient nodes receive message-passing signals from the gene layer.
         Each patient links to its top-K expressed genes via bidirectional
         'expressed_by' / 'expresses' edges.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Input:  --graph (default config.DATA_PROC / SRC_GRAPH_FILE, must carry meta.patient_ids),
        --expr (Xena expression gz at config.XENA_DIR / XENA_EXPR_FILE),
        --out (default config.DATA_PROC / OUT_GRAPH_FILE),
        --top_k, --expr_floor.
Output: Augmented graph at --out plus stdout progress logs.
Run:    python scripts/33_build_patient_gene_edges.py --graph <graph.pt> [--out <o> --top_k K]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Per-run artifact names inside config.XENA_DIR / config.DATA_PROC
XENA_EXPR_FILE = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
SRC_GRAPH_FILE = "xena_graph_large.pt"
OUT_GRAPH_FILE = "xena_graph_patient.pt"

# Default thresholds / hyper-parameters used if not overridden via CLI
DEFAULT_TOP_K = 100               # Number of top-expressed genes linked per patient
DEFAULT_EXPR_FLOOR = 0.0          # Expression values below this are treated as zero/unexpressed
DEFAULT_NAN_REPLACEMENT = 0.0     # Value used to fill NaN entries in expression matrix
FILE_SIZE_MB_DIVISOR = 1024 * 1024  # Divisor for converting raw bytes to megabytes


def build_patient_gene_edges(expr_sub, patient_ids, gene_list,
                             top_k=DEFAULT_TOP_K, expr_floor=DEFAULT_EXPR_FLOOR):
    """
    @brief   Construct edge-index arrays connecting every patient node to its
             top-K expressed genes.
    @param   expr_sub     pd.DataFrame – genes x samples expression matrix (rows aligned to a
                          subset of `gene_list`).
    @param   patient_ids  list – ordered list of sample/patient node identifiers used to
                          align expression columns with patient node indices.
    @param   gene_list    list – ordered gene symbols that define the gene node ordering
                          in the graph.
    @param   top_k        int – maximum number of expressed genes to link per patient
                          (default DEFAULT_TOP_K).
    @param   expr_floor   float – expression values strictly below this threshold are
                          treated as absent (set to 0) before the top-K selection
                          (default DEFAULT_EXPR_FLOOR).
    @return  Tuple (src, dst) of lists containing patient-indices (src) and
             gene-indices (dst) for the 'patient -> gene' direction.
    @note    Only genes present in both `gene_list` and `expr_sub.index` are used.
             Patients whose sample id is missing from the expression columns are skipped.
    """
    # Build gene symbol -> index lookup using the canonical graph ordering.
    g2i = {g: i for i, g in enumerate(gene_list)}
    src, dst = [], []

    # Restrict the expression matrix to genes that exist in the graph's gene list.
    common_genes = [g for g in gene_list if g in expr_sub.index]
    sub = expr_sub.loc[common_genes]
    arr = sub.to_numpy(dtype=np.float32)  # Shape: [n_common_genes, n_samples]

    for p_idx, sid in enumerate(patient_ids):
        if sid not in sub.columns:
            # Patient sample id is not present in the expression matrix: skip.
            continue

        # Locate the column for this patient and sanitise NaNs.
        col = np.nan_to_num(arr[:, sub.columns.get_loc(sid)], nan=DEFAULT_NAN_REPLACEMENT)

        # Apply expression floor: values below threshold are considered unexpressed.
        if expr_floor > 0:
            col[col < expr_floor] = 0.0

        # Select top-K genes by descending expression; stop iteration early when
        # we hit a non-positive value (remaining entries after sort will be <= 0).
        order = np.argsort(-col)[:top_k]
        for g_row in order:
            if col[g_row] <= 0:
                continue
            g_idx = g2i[common_genes[g_row]]
            src.append(p_idx)
            dst.append(g_idx)

    return src, dst


def main():
    """
    @brief   Entry point: load pan-cancer graph and Xena expression matrix,
             build patient<->gene edges, store the augmented graph on disk.
    @return  None
    @note    The script aborts with SystemExit if config.XENA_DIR is unset or
             the loaded graph lacks the required `patient_ids` metadata field.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default=os.path.join(config.DATA_PROC, SRC_GRAPH_FILE))
    ap.add_argument('--out', default=os.path.join(config.DATA_PROC, OUT_GRAPH_FILE))
    ap.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    ap.add_argument('--expr_floor', type=float, default=DEFAULT_EXPR_FLOOR)
    args = ap.parse_args()

    if not config.XENA_DIR:
        raise SystemExit("Set GFM_XENA_DIR to the Xena pan-cancer directory.")
    expr_path = os.path.join(config.XENA_DIR, XENA_EXPR_FILE)

    # Load source graph and extract required metadata fields.
    graph = torch.load(args.graph, map_location='cpu')
    gene_list = graph['meta']['gene_list']
    patient_ids = graph['meta'].get('patient_ids', [])
    n_patients = graph['patient']['num_nodes']

    if not patient_ids:
        # Patient ids are required to align expression columns to node order;
        # the upstream graph builder stores them under meta.patient_ids.
        raise SystemExit(
            "graph meta lacks 'patient_ids'; re-run the graph builder storing sample ids, "
            "or pass --patient_ids_csv with the ordered sample list."
        )

    print(f"Loading expression {expr_path} ...")
    expr = pd.read_csv(expr_path, sep='\t', index_col=0,
                       compression='gzip', low_memory=False)

    src, dst = build_patient_gene_edges(expr, patient_ids, gene_list,
                                        args.top_k, args.expr_floor)
    print(f"patient-gene edges: {len(src):,}")

    # Store both directions of the patient<->gene relation as PyG edge_index tensors.
    pg_ei = torch.tensor([src, dst], dtype=torch.long)
    gp_ei = torch.tensor([dst, src], dtype=torch.long)

    graph[('patient', 'expressed_by', 'gene')] = {'edge_index': pg_ei}
    graph[('gene', 'expresses', 'patient')] = {'edge_index': gp_ei}
    graph['meta']['patient_gene_edges'] = pg_ei.shape[1]
    graph['meta']['patient_ids'] = patient_ids

    torch.save(graph, args.out)
    file_size_mb = os.path.getsize(args.out) / FILE_SIZE_MB_DIVISOR
    print(f"Saved {args.out} ({file_size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
