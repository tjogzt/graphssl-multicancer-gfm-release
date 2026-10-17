#!/usr/bin/env python3
"""
@file    34_build_patient_esm_graph.py
@brief   Merge the ESM-feature gene/cancer graph with the patient-level graph into
         one unified heterogeneous graph. The resulting graph combines protein-
         sequence gene features (from the ESM graph) with patient expression
         edges and patient-cancer membership edges for downstream patient-level
         objectives such as P-NAMR and P-C3L.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Input:  config.DATA_PROC / SRC_ESM_FILE (output of 20_build_graph_esm.py)
        and config.DATA_PROC / SRC_PATIENT_FILE (output of 33_build_patient_gene_edges.py).
Output: config.DATA_PROC / OUT_GRAPH_FILE plus a stdout summary.
Run:    python scripts/34_build_patient_esm_graph.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Per-run artifact names inside config.DATA_PROC
SRC_ESM_FILE = "xena_graph_esm.pt"            # ESM-feature graph (from 20_build_graph_esm.py)
SRC_PATIENT_FILE = "xena_graph_patient.pt"    # Patient-edge graph (from 33_build_patient_gene_edges.py)
OUT_GRAPH_FILE = "xena_graph_patient_esm.pt"  # Final merged heterogeneous graph

# Map location for torch.load
LOAD_MAP_LOCATION = 'cpu'


def main():
    """
    @brief   Entry point: load the two source graphs, perform the structural merge,
             carry over patient-level metadata, and persist the merged graph.
    @return  None
    @note    The merged graph reuses ESM graph metadata as its base and injects
             patient_ids / patient_cancer / patient_gene_edges from the patient
             graph's meta dictionary.
    """
    # Load both source graphs onto CPU to avoid GPU memory issues.
    esm = torch.load(os.path.join(config.DATA_PROC, SRC_ESM_FILE),
                     map_location=LOAD_MAP_LOCATION)
    pat = torch.load(os.path.join(config.DATA_PROC, SRC_PATIENT_FILE),
                     map_location=LOAD_MAP_LOCATION)

    # Structural merge: gene/cancer nodes + gene-centric edges come from the ESM graph;
    # patient nodes and patient-centric edges come from the patient graph.
    merged = {
        'gene': esm['gene'],
        'cancer': esm['cancer'],
        ('gene', 'ppi', 'gene'): esm[('gene', 'ppi', 'gene')],
        ('gene', 'expressed_in', 'cancer'): esm[('gene', 'expressed_in', 'cancer')],
        ('cancer', 'expresses', 'gene'): esm[('cancer', 'expresses', 'gene')],
        'patient': pat['patient'],
        ('patient', 'belongs_to', 'cancer'): pat[('patient', 'belongs_to', 'cancer')],
        ('patient', 'expressed_by', 'gene'): pat[('patient', 'expressed_by', 'gene')],
        ('gene', 'expresses', 'patient'): pat[('gene', 'expresses', 'patient')],
        'meta': esm['meta'],
    }

    # Carry patient-level metadata from the patient graph into the merged graph,
    # ensuring downstream patient-level objectives have access to these fields.
    merged['meta']['patient_ids'] = pat['meta']['patient_ids']
    merged['meta']['patient_cancer'] = pat['meta']['patient_cancer']
    merged['meta']['patient_gene_edges'] = pat['meta'].get('patient_gene_edges', 0)

    out = os.path.join(config.DATA_PROC, OUT_GRAPH_FILE)
    torch.save(merged, out)
    print(f"Saved {out}")
    print(f"  gene {merged['gene']['num_nodes']} | "
          f"cancer {merged['cancer']['num_nodes']} | "
          f"patient {merged['patient']['num_nodes']}")
    print(f"  edges: ppi {merged[('gene', 'ppi', 'gene')]['edge_index'].size(1)}, "
          f"patient-gene {merged[('patient', 'expressed_by', 'gene')]['edge_index'].size(1)}")


if __name__ == '__main__':
    main()
