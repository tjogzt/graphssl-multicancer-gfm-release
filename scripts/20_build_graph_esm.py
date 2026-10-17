#!/usr/bin/env python3
"""
@file    20_build_graph_esm.py
@brief   Build the pan-cancer heterogeneous graph using ESM-2 protein embeddings
         as gene node features, preserving PPI and gene-cancer topology.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Rebuild the pan-cancer graph with ESM-2 protein embeddings as gene node features.
The PPI and gene-cancer topology of the large graph is kept unchanged, but each
gene's omics summary is replaced by its standardized (optionally PCA-reduced)
ESM-2 vector, so NAMR denoising starts from protein-sequence-informed features.

Input:  ESM-2 embeddings (config.DATA_PROC / EMBEDDINGS_FILE, written by
        30_embed_esm2.py) and the large graph (config.DATA_PROC / LARGE_GRAPH_FILE,
        written by 21_build_graph_xena_large.py).
Output: config.DATA_PROC / OUT_GRAPH_FILE plus stdout progress logs.
Run:    python scripts/20_build_graph_esm.py
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# --- Configure structured logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger()

import config as _cfg

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================
# Base path for all input/output artifacts (mirrors config.DATA_PROC).
OUTPUT_DIR = _cfg.DATA_PROC

# Per-run artifact names, all resolved under OUTPUT_DIR.
EMBEDDINGS_FILE = "esm2_gene_embeddings.pt"   # Produced by 30_embed_esm2.py
LARGE_GRAPH_FILE = "xena_graph_large.pt"      # Produced by 21_build_graph_xena_large.py
OUT_GRAPH_FILE = "xena_graph_esm.pt"          # Output of this script

# ESM-2 checkpoint identifier recorded in graph metadata for traceability.
ESM_MODEL_CHECKPOINT = "esm2_t6_8M_UR50D"

# Env flag: when set to '1', re-standardize gene features AFTER PCA reduction
# so NAMR additive-noise denoising (sigma=0.5) is uniformly well-posed per dimension.
ENV_POST_PCA_STD = 'GFM_POST_PCA_STD'


def load_esm_embeddings(data_dir, embeddings_filename):
    """
    @brief   Load the precomputed ESM-2 per-gene embedding dictionary from disk.

    @param   data_dir             Base directory containing the embeddings file.
    @param   embeddings_filename  Filename (basename) of the embeddings .pt archive.
    @return  Tuple (embeddings_dict, embedding_dimension) where embeddings_dict
             maps gene_symbol -> np.ndarray, and dimension is the raw ESM dim.
    @note    Uses map_location="cpu" so this function works on GPU-free hosts.
    """
    log.info("Loading ESM-2 embeddings ...")
    full_path = os.path.join(data_dir, embeddings_filename)
    esm_bundle = torch.load(full_path, map_location="cpu", weights_only=False)
    embeddings = esm_bundle["embeddings"]   # Dict: {gene_symbol: np.array(320,)}
    esm_dim = esm_bundle["dim"]
    log.info(f"  Loaded {len(embeddings):,} genes with raw dim={esm_dim}")
    return embeddings, esm_dim


def load_source_graph(data_dir, graph_filename):
    """
    @brief   Load the upstream large graph (omics-featured) and unpack its components.

    @param   data_dir        Base directory containing the graph file.
    @param   graph_filename  Filename (basename) of the source graph .pt.
    @return  Tuple (gene_list, cancer_list, ppi_ei, gc_ei, cg_ei, cancer_x_data)
             where *ei are edge-index tensors and cancer_x_data is the
             one-hot cancer feature matrix.
    """
    log.info("Loading existing source graph ...")
    graph = torch.load(os.path.join(data_dir, graph_filename), map_location="cpu")
    gene_list   = graph["meta"]["gene_list"]
    cancer_list = graph["meta"]["cancer_list"]
    ppi_ei      = graph[("gene", "ppi", "gene")]["edge_index"]
    gc_ei       = graph[("gene", "expressed_in", "cancer")]["edge_index"]
    cg_ei       = graph[("cancer", "expresses", "gene")]["edge_index"]
    cancer_x_data = graph["cancer"]["x"]
    log.info(f"  Source graph: {len(gene_list):,} genes, {len(cancer_list)} cancers")
    return gene_list, cancer_list, ppi_ei, gc_ei, cg_ei, cancer_x_data


def match_genes_to_embeddings(gene_list, embeddings):
    """
    @brief   Match graph genes to available ESM-2 embeddings and build an index remap.

    @param   gene_list   Ordered list of gene symbols from the source graph.
    @param   embeddings  Dict of gene_symbol -> embedding vector.
    @return  Tuple (matched_genes, matched_embeddings_list, gene_index_remap)
             where gene_index_remap maps old_graph_idx -> new_graph_idx (only
             for genes that had an embedding).
    @note    Gene order is preserved from the source graph for genes that matched;
             unmatched genes are simply dropped (no imputation).
    """
    log.info("Matching genes with ESM-2 embeddings ...")
    matched_genes = []
    matched_emb = []
    gene_to_new_idx = {}

    # Iterate in source-graph order to preserve topology semantics.
    for i, gene in enumerate(gene_list):
        if gene in embeddings:
            gene_to_new_idx[i] = len(matched_genes)
            matched_genes.append(gene)
            matched_emb.append(embeddings[gene])

    n_matched = len(matched_genes)
    log.info(f"  Matched: {n_matched:,} / {len(gene_list):,} genes "
             f"({100.0 * n_matched / len(gene_list):.1f}%)")
    return matched_genes, matched_emb, gene_to_new_idx


def build_and_normalize_gene_features(matched_emb, target_dim, seed, post_pca_std_env):
    """
    @brief   Standardize embeddings, reduce/expand them to target_dim, and optionally
             re-standardize after PCA.

    @param   matched_emb       List of raw per-gene ESM-2 embedding arrays.
    @param   target_dim        Target feature width (from config.EMBEDDING_DIM).
    @param   seed              Random seed for deterministic PCA.
    @param   post_pca_std_env  Env var name; if its value is '1' re-std after PCA.
    @return  torch.Tensor of shape (n_genes, target_dim) with dtype float32.
    @note    When raw dim < target_dim the tail is zero-padded (no information
             added); when raw dim > target_dim PCA with the fixed seed is used.
    """
    # Stack into a single [n_genes, raw_dim] matrix for vectorized processing.
    gene_feats = np.array(matched_emb)
    # First standardization (before PCA): zero-mean, unit-variance per dim.
    gene_feats = StandardScaler().fit_transform(gene_feats)
    n_genes = gene_feats.shape[0]

    if gene_feats.shape[1] > target_dim:
        # Dimensionality reduction: keep the top target_dim principal components.
        gene_feats = PCA(n_components=target_dim, random_state=seed).fit_transform(gene_feats)
    elif gene_feats.shape[1] < target_dim:
        # Zero-padding to reach the canonical embedding width.
        assert gene_feats.shape[1] <= target_dim, \
            "gene features exceed target dim (should be unreachable)"
        pad = np.zeros((n_genes, target_dim - gene_feats.shape[1]), dtype=np.float32)
        gene_feats = np.concatenate([gene_feats, pad], axis=1)

    # Optional second standardization step (H1 convergence fix).
    if os.environ.get(post_pca_std_env) == '1':
        log.info("  Applying post-PCA standardization (NAMR convergence fix)")
        gene_feats = StandardScaler().fit_transform(gene_feats)

    gene_x = torch.tensor(gene_feats, dtype=torch.float32)
    log.info(f"  Final gene feature matrix shape: {tuple(gene_x.shape)}")
    return gene_x


def remap_ppi_edges(ppi_ei_old, old_to_new_map):
    """
    @brief   Rebuild PPI edge_index, keeping only edges where both endpoints matched.

    @param   ppi_ei_old     Edge-index tensor (2, N_edges) from the source graph.
    @param   old_to_new_map Dict mapping old gene index -> new gene index.
    @return  New edge-index tensor (2, M_edges) with M <= N after endpoint filtering.
    """
    old_to_new = old_to_new_map
    ppi_src, ppi_dst = [], []
    for i in range(ppi_ei_old.size(1)):
        old_a = ppi_ei_old[0, i].item()
        old_b = ppi_ei_old[1, i].item()
        # Drop any edge whose endpoint was dropped for lack of embedding.
        if old_a in old_to_new and old_b in old_to_new:
            ppi_src.append(old_to_new[old_a])
            ppi_dst.append(old_to_new[old_b])
    ppi_new = torch.tensor([ppi_src, ppi_dst], dtype=torch.long)
    log.info(f"  PPI edges after remap: {ppi_new.size(1):,} "
             f"(was {ppi_ei_old.size(1):,})")
    return ppi_new


def remap_gene_cancer_edges(gc_ei_old, cg_ei_old, old_to_new_map):
    """
    @brief   Rebuild both directions of the gene<->cancer bipartite edge sets.

    @param   gc_ei_old      Edge-index (gene->cancer direction) from source graph.
    @param   cg_ei_old      Edge-index (cancer->gene direction) from source graph.
    @param   old_to_new_map Dict mapping old gene index -> new gene index.
    @return  Tuple (gc_new, cg_new) of edge-index tensors.
    """
    # Direction 1: gene -> expressed_in -> cancer
    gc_src, gc_dst = [], []
    for i in range(gc_ei_old.size(1)):
        old_g = gc_ei_old[0, i].item()
        if old_g in old_to_new_map:
            gc_src.append(old_to_new_map[old_g])
            gc_dst.append(gc_ei_old[1, i].item())
    gc_new = torch.tensor([gc_src, gc_dst], dtype=torch.long)

    # Direction 2: cancer -> expresses -> gene
    cg_src, cg_dst = [], []
    for i in range(cg_ei_old.size(1)):
        old_g = cg_ei_old[1, i].item()
        if old_g in old_to_new_map:
            cg_src.append(cg_ei_old[0, i].item())
            cg_dst.append(old_to_new_map[old_g])
    cg_new = torch.tensor([cg_src, cg_dst], dtype=torch.long)

    log.info(f"  Gene<->Cancer edges after remap: {gc_new.size(1):,} each direction")
    return gc_new, cg_new


def assemble_and_save_graph(data_dir, out_filename,
                            gene_x, n_genes, matched_gene_list,
                            cancer_x, cancer_list,
                            ppi_new, gc_new, cg_new,
                            target_dim, esm_dim, esm_model_ckpt):
    """
    @brief   Assemble all components into the final HeteroData-style dict and persist.

    @param   data_dir          Destination directory for the saved graph.
    @param   out_filename      Output filename (basename).
    @param   gene_x            Gene feature tensor (n_genes, target_dim).
    @param   n_genes           Number of gene nodes (int).
    @param   matched_gene_list Ordered list of gene symbols.
    @param   cancer_x          Cancer feature tensor (n_cancers, target_dim).
    @param   cancer_list       Ordered list of cancer abbreviations.
    @param   ppi_new           PPI edge-index tensor.
    @param   gc_new            Gene->Cancer edge-index tensor.
    @param   cg_new            Cancer->Gene edge-index tensor.
    @param   target_dim        Canonical embedding dim (post PCA/pad).
    @param   esm_dim           Raw ESM-2 output dim.
    @param   esm_model_ckpt    ESM-2 model identifier string.
    @return  None. Writes graph to disk and logs final summary.
    """
    new_graph = {
        'gene':   {'x': gene_x,   'num_nodes': n_genes},
        'cancer': {'x': cancer_x, 'num_nodes': len(cancer_list)},
        ('gene',   'ppi',          'gene'):   {'edge_index': ppi_new},
        ('gene',   'expressed_in', 'cancer'): {'edge_index': gc_new},
        ('cancer', 'expresses',    'gene'):   {'edge_index': cg_new},
        'meta': {
            'gene_list':     matched_gene_list,
            'cancer_list':   cancer_list,
            'n_genes':       n_genes,
            'n_cancers':     len(cancer_list),
            'embedding_dim': target_dim,
            'esm_dim':       esm_dim,
            'esm_model':     esm_model_ckpt,
        }
    }

    out_path = os.path.join(data_dir, out_filename)
    torch.save(new_graph, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    log.info(f"Graph saved: {out_path} ({size_mb:.1f} MB)")
    log.info(f"  Genes: {n_genes:,} | "
             f"Cancers: {len(cancer_list)} | "
             f"PPI edges: {ppi_new.size(1):,}")


def main():
    """
    @brief   Top-level driver: load artifacts, match genes, build features,
             remap edges, save output graph.
    @return  None.
    """
    # ----- Step 1: Load source artifacts -----
    embeddings, esm_dim = load_esm_embeddings(OUTPUT_DIR, EMBEDDINGS_FILE)
    gene_list, cancer_list, ppi_ei, gc_ei, cg_ei, cancer_x_data = \
        load_source_graph(OUTPUT_DIR, LARGE_GRAPH_FILE)

    # ----- Step 2: Match genes to ESM-2 embeddings and build index remap -----
    matched_genes, matched_emb, gene_to_new_idx = \
        match_genes_to_embeddings(gene_list, embeddings)
    n_genes = len(matched_genes)

    # ----- Step 3: Standardize and project gene features to target dim -----
    target_dim = _cfg.EMBEDDING_DIM
    gene_x = build_and_normalize_gene_features(
        matched_emb, target_dim, seed=_cfg.SEED,
        post_pca_std_env=ENV_POST_PCA_STD
    )

    # ----- Step 4: Remap edges to the new gene index space -----
    ppi_new = remap_ppi_edges(ppi_ei, gene_to_new_idx)
    gc_new, cg_new = remap_gene_cancer_edges(gc_ei, cg_ei, gene_to_new_idx)

    # ----- Step 5: Keep cancer features untouched (clone for isolation) -----
    cancer_x = cancer_x_data.clone()

    # ----- Step 6: Assemble graph dict and serialize to disk -----
    assemble_and_save_graph(
        data_dir=OUTPUT_DIR,
        out_filename=OUT_GRAPH_FILE,
        gene_x=gene_x,
        n_genes=n_genes,
        matched_gene_list=matched_genes,
        cancer_x=cancer_x,
        cancer_list=cancer_list,
        ppi_new=ppi_new,
        gc_new=gc_new,
        cg_new=cg_new,
        target_dim=target_dim,
        esm_dim=esm_dim,
        esm_model_ckpt=ESM_MODEL_CHECKPOINT,
    )


if __name__ == "__main__":
    main()
