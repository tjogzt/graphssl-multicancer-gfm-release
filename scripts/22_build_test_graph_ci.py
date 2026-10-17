#!/usr/bin/env python3
"""
@file    22_build_test_graph_ci.py
@brief   Build a small synthetic test graph for CI/CD smoke tests and feasibility
         experiments (memory footprint), preferring BioGRID physical PPI when
         available and falling back to a synthetic small-world network.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

CI/TEST-ONLY: build a small test graph for feasibility and smoke-test experiments
(memory footprint of the model). It prefers real BioGRID physical PPI; when BioGRID
is unavailable it falls back to a synthetic small-world network. All node features
are random tensors by design and must never feed the analysis pipeline -- published
results use the real data/processed graphs built by the 20/21 scripts.

Input:  BioGRID human PPI (downloaded under config.DATA_RAW / "biogrid", or a
        synthetic fallback) plus the CANCER_TYPES / N_GENES / EMBEDDING_DIM
        protocol constants.
Output: config.DATA_PROC / OUT_GRAPH_FILE plus stdout progress logs.
Run:    python scripts/22_build_test_graph_ci.py
"""
import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
import urllib.request
import gzip
from tqdm import tqdm

# Add scripts directory to import path so config.py is discoverable from any CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================
# Base paths (from env-aware config module).
DATA_RAW  = config.DATA_RAW
DATA_PROC = config.DATA_PROC
os.makedirs(DATA_PROC, exist_ok=True)

# --- BioGRID source parameters ---
BIOGRID_SUBDIR   = "biogrid"
BIOGRID_FILENAME = "BIOGRID-ORGANISM-Homo_sapiens-4.4.240.tab3.txt"
BIOGRID_ARCHIVE_URL = (
    "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/"
    "BIOGRID-ORGANISM-LATEST.tab3.zip"
)
# BioGRID-specific organism identifier for Homo sapiens (NOT the NCBI taxonomy id).
BIOGRID_ORGANISM_ID_HUMAN = "2697049"

# --- Test graph output filename ---
OUT_GRAPH_FILE = "test_graph.pt"

# --- Protocol constants for the small synthetic feasibility graph ---
TEST_CANCER_TYPES = ["BRCA", "LUAD", "COAD"]   # 3 representative cancers
TEST_N_GENES      = 5000                       # Top-N genes by connectivity
TEST_EMBEDDING_DIM = config.EMBEDDING_DIM      # Canonical feature width
TEST_PPI_MIN_SCORE = config.PPI_SCORE_THRESHOLD

# Fallback synthetic network parameters (Watts-Strogatz small-world model).
SYNTH_NEIGHBOURS_K = 20    # Each node connected to K nearest neighbours
SYNTH_REWIRE_P     = 0.1   # Edge re-wiring probability

# Minimum number of PPIs before we accept the BioGRID load as successful;
# falls back to synthetic if fewer PPIs were found.
MIN_BIOGRID_ACCEPTABLE_PPIS = 10000

# Patients per cancer type in the synthetic graph (smoke test sample count).
N_PATIENTS_PER_CANCER = 100


# =========================================================================
# Step 1: BioGRID PPI download + parse (with synthetic fallback)
# =========================================================================

def download_biogrid_ppi():
    """
    @brief   Attempt to download and parse the human BioGRID physical PPI dataset.

    @return  Set of canonicalized undirected edges (gene_a, gene_b) with
             a <= b lex order; returns None on any failure (network error,
             missing human file in archive, parse error, edge count too low).
    @note    The download is idempotent: if the extracted TAB3 file already
             exists under DATA_RAW/biogrid/ we skip the download and just parse.
    """
    bio_dir = os.path.join(DATA_RAW, BIOGRID_SUBDIR)
    os.makedirs(bio_dir, exist_ok=True)
    ppi_path = os.path.join(bio_dir, BIOGRID_FILENAME)

    # Download the ZIP only if the extracted TAB3 is not already cached locally.
    if not os.path.exists(ppi_path):
        print(f"[BioGRID] Downloading latest organism archive from {BIOGRID_ARCHIVE_URL} ...")
        zip_path = ppi_path + ".zip"
        try:
            urllib.request.urlretrieve(BIOGRID_ARCHIVE_URL, zip_path)
            # Extract only the human TAB3 file from the multi-organism ZIP archive.
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                human_tab3_found = False
                for entry_name in zf.namelist():
                    if 'Homo_sapiens' in entry_name and entry_name.endswith('.tab3.txt'):
                        zf.extract(entry_name, bio_dir)
                        os.rename(os.path.join(bio_dir, entry_name), ppi_path)
                        human_tab3_found = True
                        break
                if not human_tab3_found:
                    print("[BioGRID] ERROR: No Homo_sapiens TAB3 file found in archive.")
                    return None
            os.remove(zip_path)
        except Exception as exc:
            print(f"[BioGRID] Download/extract failed: {exc}")
            return None

    # --- Parse TAB3: keep only human-human physical interactions ---
    print(f"[BioGRID] Parsing physical PPI interactions from: {ppi_path}")
    ppi_edges = set()
    with open(ppi_path, 'r') as handle:
        # First line is the column header; we skip it but also skip any '#' comment lines.
        header = handle.readline()
        for line in tqdm(handle, desc="  Reading TAB3", unit=" lines"):
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            # Guard against malformed / truncated lines with fewer than 17 fields.
            if len(cols) < 17:
                continue
            # Columns 15 and 16 are the two organism IDs (BioGRID id format).
            # We keep only interactions where BOTH endpoints are Homo sapiens.
            org_a, org_b = cols[15], cols[16]
            if BIOGRID_ORGANISM_ID_HUMAN not in org_a or BIOGRID_ORGANISM_ID_HUMAN not in org_b:
                continue
            # Column 11 is the interaction-type string; require keyword "physical"
            # so we keep only direct physical binding events, not genetic interactions.
            inter_type = cols[11]
            if 'physical' not in inter_type.lower():
                continue
            # Columns 7 and 8 are the HGNC gene symbols for the two interactors.
            gene_a = cols[7].strip()
            gene_b = cols[8].strip()
            # Skip self-loops and rows with empty gene symbols.
            if gene_a and gene_b and gene_a != gene_b:
                # Canonicalize edge by sorted order so each undirected pair stored once.
                a, b = sorted([gene_a, gene_b])
                ppi_edges.add((a, b))

    print(f"[BioGRID] Loaded {len(ppi_edges):,} unique physical PPIs")
    return ppi_edges


def generate_synthetic_ppi(n_genes, seed):
    """
    @brief   Fallback: generate a synthetic PPI network using a Watts-Strogatz
             small-world graph, with dummy GENE#### node labels.

    @param   n_genes  Target number of nodes (genes).
    @param   seed     Integer random seed for reproducibility.
    @return  Set of undirected edges (gene_a, gene_b), a <= b.
    @note    Requires `networkx` package to be installed at runtime.
    """
    print("[Synthetic] Generating Watts-Strogatz small-world PPI graph ...")
    import networkx as nx
    G = nx.watts_strogatz_graph(
        n=n_genes, k=SYNTH_NEIGHBOURS_K, p=SYNTH_REWIRE_P, seed=seed
    )
    ppi_edges = set()
    for u, v in G.edges():
        # Lexicographic order canonicalization; zero-padded for deterministic sort.
        ppi_edges.add((f"GENE{u:04d}", f"GENE{v:04d}"))
    print(f"[Synthetic] Generated {len(ppi_edges):,} edges")
    return ppi_edges


# =========================================================================
# Step 2: Assemble the small test heterograph with simulated features
# =========================================================================

def build_test_graph(ppi_edges, n_genes=TEST_N_GENES,
                     n_patients_per_cancer=N_PATIENTS_PER_CANCER):
    """
    @brief   Build a PyTorch dict-style HeteroData with simulated features for
             CI / memory feasibility tests.

    @param   ppi_edges             Set of (gene_a, gene_b) tuples (undirected).
    @param   n_genes               Number of genes to keep (top-N by PPI degree).
    @param   n_patients_per_cancer Simulated patient count per cancer type.
    @return  The assembled graph dict (same schema as production graph builders).
    @note    All node feature tensors are drawn from N(0, 1) by design: this is
             a SMOKE-TEST only artifact and MUST NOT be used for real analysis.
    """
    # --- Select top-N most connected genes by unweighted PPI degree ---
    gene_degree = {}
    for a, b in ppi_edges:
        gene_degree[a] = gene_degree.get(a, 0) + 1
        gene_degree[b] = gene_degree.get(b, 0) + 1
    top_genes = sorted(gene_degree, key=gene_degree.get, reverse=True)[:n_genes]
    top_gene_set = set(top_genes)

    # --- Filter edges to the selected gene subset ---
    filtered_ppi = [
        (a, b) for a, b in ppi_edges
        if a in top_gene_set and b in top_gene_set
    ]
    gene_to_idx = {g: i for i, g in enumerate(top_genes)}

    n_total_patients = n_patients_per_cancer * len(TEST_CANCER_TYPES)

    print("\n[Build] Test graph statistics:")
    print(f"  Gene nodes          : {len(top_genes):,}")
    print(f"  PPI edges (undir.)  : {len(filtered_ppi):,}")
    print(f"  Cancer types        : {len(TEST_CANCER_TYPES)} -> {TEST_CANCER_TYPES}")
    print(f"  Patient nodes (sim.): {n_total_patients}")

    # --- PPI edge index (symmetrized) ---
    ppi_src, ppi_dst = [], []
    for a, b in filtered_ppi:
        ppi_src.append(gene_to_idx[a])
        ppi_dst.append(gene_to_idx[b])
    # Add both directions so the edge_index tensor represents an undirected graph.
    ppi_edge_index = torch.tensor(
        [ppi_src + ppi_dst, ppi_dst + ppi_src], dtype=torch.long
    )

    # --- Simulated node features (RANDOM, for CI smoke test only) ---
    # Gene features: placeholder for PCA-reduced multi-omics (replaced in prod graphs).
    gene_x = torch.randn(len(top_genes), TEST_EMBEDDING_DIM)
    # Cancer features: placeholder (prod uses one-hot; here random for simplicity).
    cancer_x = torch.randn(len(TEST_CANCER_TYPES), TEST_EMBEDDING_DIM)
    # Patient features: placeholder (prod uses standardized omics summaries).
    patient_x = torch.randn(n_total_patients, TEST_EMBEDDING_DIM)

    # --- Patient -> Cancer edges (each block of n_patients_per_cancer belongs to one cancer) ---
    pc_src = list(range(n_total_patients))
    pc_dst = [i // n_patients_per_cancer for i in range(n_total_patients)]
    pc_edge_index = torch.tensor([pc_src, pc_dst], dtype=torch.long)

    # --- Gene <-> Cancer full bipartite edges (same pattern as production C3L setup) ---
    gc_src, gc_dst = [], []
    for gi in range(len(top_genes)):
        for ci in range(len(TEST_CANCER_TYPES)):
            gc_src.append(gi)
            gc_dst.append(ci)
    gc_edge_index = torch.tensor([gc_src, gc_dst], dtype=torch.long)
    cg_src = list(gc_dst)
    cg_dst = list(gc_src)
    cg_edge_index = torch.tensor([cg_src, cg_dst], dtype=torch.long)

    # --- Assemble final graph dict in the same heterogeneous schema as 20/21 builders ---
    graph_data = {
        'gene':    {'x': gene_x,    'num_nodes': len(top_genes)},
        'cancer':  {'x': cancer_x,  'num_nodes': len(TEST_CANCER_TYPES)},
        'patient': {'x': patient_x, 'num_nodes': n_total_patients},
        ('gene',    'ppi',          'gene'):   {'edge_index': ppi_edge_index},
        ('patient', 'belongs_to',   'cancer'): {'edge_index': pc_edge_index},
        ('gene',    'expressed_in', 'cancer'): {'edge_index': gc_edge_index},
        ('cancer',  'expresses',    'gene'):   {'edge_index': cg_edge_index},
        'meta': {
            'gene_list':     top_genes,
            'cancer_list':   TEST_CANCER_TYPES,
            'n_genes':       len(top_genes),
            'n_cancers':     len(TEST_CANCER_TYPES),
            'n_patients':    n_total_patients,
            'embedding_dim': TEST_EMBEDDING_DIM,
        }
    }

    # --- Persist test graph to disk ---
    out_path = os.path.join(DATA_PROC, OUT_GRAPH_FILE)
    torch.save(graph_data, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n[Build] Test graph saved to: {out_path}")
    print(f"  File size: {size_mb:.1f} MB")
    return graph_data


# =========================================================================
# Main entry
# =========================================================================

def main():
    """
    @brief   Main driver: try BioGRID first, fall back to synthetic PPI, then build the graph.

    @return  None. Prints summary banners on stdout.
    """
    print("=" * 60)
    print("Phase 1.2: Building small test graph for CI / feasibility experiment")
    print("=" * 60)

    # Deterministic random state shared across synthetic graph & features.
    config.set_seed()

    # --- Obtain PPI edges: BioGRID (preferred) or synthetic fallback ---
    ppi_edges = download_biogrid_ppi()
    if ppi_edges is None or len(ppi_edges) < MIN_BIOGRID_ACCEPTABLE_PPIS:
        print("\n[Fallback] BioGRID unavailable or too few edges; "
              "generating synthetic PPI network ...")
        ppi_edges = generate_synthetic_ppi(TEST_N_GENES, seed=config.SEED)

    # --- Build and persist the small test heterograph ---
    build_test_graph(ppi_edges)

    print("\n" + "=" * 60)
    print("Phase 1.2 complete. Test graph ready for feasibility experiments.")
    print("=" * 60)


if __name__ == '__main__':
    main()
