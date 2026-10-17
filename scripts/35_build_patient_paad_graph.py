#!/usr/bin/env python3
"""
@file    35_build_patient_paad_graph.py
@brief   Build a single-cancer (TCGA-PAAD) patient-level heterogeneous graph from
         raw STAR expression files. Gene nodes reuse the ESM-2 feature set; patient
         node features are PCA-256 of log1p(TPM); each patient links to its top-K
         expressed genes to give patient nodes real gradient flow for patient-level
         objectives.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Input:  The ESM graph at config.DATA_PROC / SRC_ESM_FILE, and per-sample STAR
        gene-expression TSV files under config.DATA_RAW / TCGA_PAAD_EXPR_SUBDIR.
Output: config.DATA_PROC / OUT_GRAPH_FILE plus stdout progress logs.
Run:    python scripts/35_build_patient_paad_graph.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Source / output artifact names
SRC_ESM_FILE = "xena_graph_esm.pt"            # ESM-feature graph (from 20_build_graph_esm.py)
OUT_GRAPH_FILE = "xena_graph_patient_paad.pt" # Output single-cancer patient graph

# Nested directory layout under DATA_RAW where PAAD STAR files live
TCGA_PAAD_EXPR_SUBDIR = os.path.join(
    "tcga", "expression", "TCGA-PAAD",
    "Transcriptome_Profiling", "Gene_Expression_Quantification",
)

# Glob pattern used to discover STAR count files per sample
STAR_FILE_GLOB_PATTERN = os.path.join("*", "*.rna_seq.augmented_star_gene_counts.tsv")

# Column names inside the STAR gene-count TSV
STAR_GENE_NAME_COL = "gene_name"
STAR_TPM_COL = "tpm_unstranded"

# Top-K expressed-gene edges per patient
TOP_K = 100

# Embedding dimension for patient PCA features
DIM = config.EMBEDDING_DIM

# Single-cancer graph uses exactly one cancer node; cancer feature tensor is identity(1)
SINGLE_CANCER_NODE_COUNT = 1
SINGLE_CANCER_INDEX = 0

# Comment character that prefixes STAR TSV header lines
TSV_COMMENT_CHAR = "#"

# Load map location for torch.load
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def load_paad_expression():
    """
    @brief   Load per-sample STAR TSV files from the TCGA-PAAD directory and
             combine them into a single genes x samples DataFrame of TPM values.
    @return  pd.DataFrame with genes as index and sample ids as columns; values
             are TPM (float). Duplicate genes are resolved by keeping the first
             occurrence after row-level max aggregation per gene symbol.
    @note    One TSV is loaded per sample; duplicate gene rows inside a single
             sample are collapsed by taking the max TPM value per gene symbol.
    """
    gene_tsv_root = os.path.join(config.DATA_RAW, TCGA_PAAD_EXPR_SUBDIR)
    # Glob the nested directory layout: each sample has its own sub-directory.
    files = sorted(glob.glob(os.path.join(gene_tsv_root, STAR_FILE_GLOB_PATTERN)))
    print(f"PAAD samples found: {len(files)}")

    matrices = {}
    for f in files:
        # Sample id is the first dot-separated token of the file basename.
        sid = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f, sep="\t", comment=TSV_COMMENT_CHAR, low_memory=False)
        # Drop rows that have no gene symbol annotation.
        df = df[df[STAR_GENE_NAME_COL].notna()]
        # One TPM value per gene symbol: max collapses duplicate gene rows.
        tpm = df.groupby(STAR_GENE_NAME_COL)[STAR_TPM_COL].max()
        matrices[sid] = tpm

    expr = pd.DataFrame(matrices)  # Shape: genes x samples
    # Drop any duplicate gene indices that remain after per-sample collapse.
    expr = expr[~expr.index.duplicated(keep="first")]
    return expr


def main():
    """
    @brief   Entry point: load PAAD expression, derive patient features via PCA
             on log1p(TPM), build patient-gene top-K edges, assemble the single-
             cancer heterogeneous graph, and serialise it to disk.
    @return  None
    """
    expr = load_paad_expression()
    print(f"Expression matrix shape (genes x samples): {expr.shape}")

    # Load ESM graph to reuse gene features, PPI edges, and the canonical gene list.
    g = torch.load(os.path.join(config.DATA_PROC, SRC_ESM_FILE),
                   map_location=LOAD_MAP_LOCATION,
                   weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    gene_list = list(g["meta"]["gene_list"])
    g2i = {x: i for i, x in enumerate(gene_list)}

    # Intersect genes between the graph and the PAAD expression matrix.
    common = [x for x in gene_list if x in expr.index]
    print(f"Genes matched: {len(common)} | Total expression genes: {len(expr.index)}")

    # Build samples x genes matrix and apply log1p TPM transform.
    X = expr.loc[common].T.astype(np.float64)  # Shape: samples x genes
    Xl = np.log1p(X.values)

    # Patient features: standardised log1p TPM reduced to PCA-256.
    scaler = StandardScaler().fit(Xl)
    Xs = scaler.transform(Xl)
    n_comp = min(DIM, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=config.SEED).fit(Xs)
    Xp = pca.transform(Xs)
    print(f"Patient feature matrix shape after PCA: {Xp.shape}")

    # Patient-gene edges: top-K expressed genes by log1p TPM per patient.
    src, dst = [], []
    for p_idx in range(Xl.shape[0]):
        order = np.argsort(-Xl[p_idx])[:TOP_K]
        for gr in order:
            # Only keep edges for genes that have strictly positive log1p expression.
            if Xl[p_idx, gr] > 0:
                src.append(p_idx)
                dst.append(g2i[common[gr]])
    pg = torch.tensor([src, dst], dtype=torch.long)
    print(f"Patient-gene edges created: {pg.size(1)}")

    n_pat = Xp.shape[0]

    # Edge indices that link every gene / patient node to the single cancer node.
    all_gene_indices = [i for i in range(len(gene_list))]
    gene_to_cancer_dst = [SINGLE_CANCER_INDEX] * len(gene_list)
    all_patient_indices = [i for i in range(n_pat)]
    patient_to_cancer_dst = [SINGLE_CANCER_INDEX] * n_pat

    graph = {
        "gene": {
            "x": g["gene"]["x"],
            "num_nodes": len(gene_list),
        },
        "cancer": {
            "x": torch.eye(SINGLE_CANCER_NODE_COUNT),
            "num_nodes": SINGLE_CANCER_NODE_COUNT,
        },
        "patient": {
            "x": torch.tensor(Xp, dtype=torch.float32),
            "num_nodes": n_pat,
        },
        ("gene", "ppi", "gene"): g[("gene", "ppi", "gene")],
        # Single cancer node: every gene links to cancer index 0 (both directions).
        ("gene", "expressed_in", "cancer"): {
            "edge_index": torch.tensor(
                [all_gene_indices, gene_to_cancer_dst], dtype=torch.long
            )
        },
        ("cancer", "expresses", "gene"): {
            "edge_index": torch.tensor(
                [gene_to_cancer_dst, all_gene_indices], dtype=torch.long
            )
        },
        # Single cancer node: every patient links to cancer index 0.
        ("patient", "belongs_to", "cancer"): {
            "edge_index": torch.tensor(
                [all_patient_indices, patient_to_cancer_dst], dtype=torch.long
            )
        },
        # Patient-gene bidirectional edges.
        ("patient", "expressed_by", "gene"): {"edge_index": pg},
        ("gene", "expresses", "patient"): {"edge_index": pg.flip(0)},
        # Metadata.
        "meta": {
            "gene_list": gene_list,
            "cancer_list": ["PAAD"],
            "n_genes": len(gene_list),
            "n_cancers": SINGLE_CANCER_NODE_COUNT,
            "n_patients": n_pat,
            "patient_ids": list(expr.columns),
            "top_k": TOP_K,
        },
    }

    out = os.path.join(config.DATA_PROC, OUT_GRAPH_FILE)
    torch.save(graph, out)
    print(f"Saved graph to: {out}")


if __name__ == "__main__":
    main()
