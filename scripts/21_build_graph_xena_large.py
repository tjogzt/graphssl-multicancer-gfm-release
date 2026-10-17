#!/usr/bin/env python3
"""
@file    21_build_graph_xena_large.py
@brief   Build the large pan-cancer heterogeneous graph from UCSC Xena multi-omics
         (expression, mutation, CNV) aligned with STRING PPI network topology.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup graph_construction

Build the large pan-cancer heterogeneous graph from UCSC Xena data: expression,
mutation, and (when present) CNV matrices are aligned with the STRING PPI network,
the top genes by PPI degree are selected, and patient-, gene-, and cancer-level
node features plus PPI / patient-cancer / gene-cancer edges are assembled into
the artifact consumed by training and the patient/edge post-processing scripts.

Input:  STRING PPI tables (config.STRING_DIR / STRING_INFO_FILE and
        STRING_LINKS_FILE), Xena omics and clinical tables (config.XENA_DIR /
        XENA_EXPR_FILE, XENA_CNV_FILE, XENA_MUT_FILE, XENA_CLIN_FILE), and the
        N_GENES / PPI_SCORE_THRESHOLD / EMBEDDING_DIM protocol constants.
Output: config.DATA_PROC / OUT_GRAPH_FILE plus stdout progress logs.
Run:    python scripts/21_build_graph_xena_large.py
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# --- Configure structured logging with timestamp and severity ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger()

import config as _cfg

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================
# Base directories (from env-aware config module; absolute paths resolved there).
XENA_DIR   = _cfg.XENA_DIR      # UCSC Xena pan-cancer data root
STRING_DIR = _cfg.STRING_DIR    # STRING v12 downloaded data root
OUTPUT_DIR = _cfg.DATA_PROC     # Processed-artifact destination

# Ensure the output directory exists before attempting any writes.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guard against an empty/missing Xena root: without omics matrices the graph
# cannot be built, so we fail fast with a clear remediation message.
if not XENA_DIR:
    raise SystemExit(
        "FATAL: XENA_DIR is not configured. "
        "Set GFM_XENA_DIR environment variable to the UCSC Xena pan-cancer directory."
    )

# --- Input / output artifact filenames (all basenames, resolved under dirs above) ---
# STRING PPI files (produced by 10_download_string_ppi.py).
STRING_INFO_FILE  = "9606.protein.info.v12.0.txt"          # protein id -> gene symbol
STRING_LINKS_FILE = "9606.protein.links.full.v12.0.txt"   # PPI links + combined_score
# UCSC Xena pan-cancer matrices (externally acquired; config.XENA_DIR points at them).
XENA_EXPR_FILE = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
XENA_CNV_FILE  = "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
XENA_MUT_FILE  = "mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"
XENA_CLIN_FILE = "TCGA_phenotype_denseDataOnlyDownload.tsv.gz"
# Output graph artifact (consumed by downstream scripts 20/32/39/40+).
OUT_GRAPH_FILE = "xena_graph_large.pt"

# --- Protocol / model hyperparameters ---
DIM                  = _cfg.EMBEDDING_DIM       # Canonical node feature width
N_GENES              = 15000                    # Top-N genes kept by PPI degree
PPI_THRESHOLD        = _cfg.PPI_SCORE_THRESHOLD # Minimum combined_score for PPIs
MIN_SAMPLES_PER_CANCER = 50                     # Drop rare cancers with fewer samples
# Minimum size (bytes) for the optional CNV file to be considered present and usable.
MIN_CNV_FILE_SIZE_BYTES = 1000

# --- Xena disease -> TCGA abbreviation mapping (comprehensive, 32 TCGA types) ---
# Keys are the exact lowercase strings found in the Xena phenotype
# '_primary_disease' column; values are the standard TCGA TSS abbreviations.
CANCER_TYPE_MAP = {
    'breast invasive carcinoma': 'BRCA',
    'lung adenocarcinoma': 'LUAD',
    'lung squamous cell carcinoma': 'LUSC',
    'colon adenocarcinoma': 'COAD',
    'prostate adenocarcinoma': 'PRAD',
    'head and neck squamous cell carcinoma': 'HNSC',
    'kidney renal clear cell carcinoma': 'KIRC',
    'stomach adenocarcinoma': 'STAD',
    'bladder urothelial carcinoma': 'BLCA',
    'liver hepatocellular carcinoma': 'LIHC',
    'uterine corpus endometrial carcinoma': 'UCEC',
    'ovarian serous cystadenocarcinoma': 'OV',
    'glioblastoma multiforme': 'GBM',
    'thyroid carcinoma': 'THCA',
    'sarcoma': 'SARC',
    'acute myeloid leukemia': 'LAML',
    'pancreatic adenocarcinoma': 'PAAD',
    'cervical squamous cell carcinoma and endocervical adenocarcinoma': 'CESC',
    'kidney renal papillary cell carcinoma': 'KIRP',
    'brain lower grade glioma': 'LGG',
    'skin cutaneous melanoma': 'SKCM',
    'esophageal carcinoma': 'ESCA',
    'pheochromocytoma and paraganglioma': 'PCPG',
    'rectum adenocarcinoma': 'READ',
    'testicular germ cell tumors': 'TGCT',
    'thymoma': 'THYM',
    'uterine carcinosarcoma': 'UCS',
    'uveal melanoma': 'UVM',
    'adrenocortical carcinoma': 'ACC',
    'mesothelioma': 'MESO',
    'cholangiocarcinoma': 'CHOL',
    'diffuse large b-cell lymphoma': 'DLBC',
    'kidney chromophobe': 'KICH',
}


# =========================================================================
# Step 1: STRING PPI loading and filtering
# =========================================================================

def load_string_ppi(string_dir, info_file, links_file, score_threshold):
    """
    @brief   Load STRING PPI tables, apply confidence filter, and map protein ids to gene symbols.

    @param   string_dir       Directory containing the STRING .txt files.
    @param   info_file        Basename of the protein-info mapping file.
    @param   links_file       Basename of the protein-links (PPI scores) file.
    @param   score_threshold  Minimum combined_score for an interaction to be kept.
    @return  Tuple (ppi_df, ppi_gene_set) where ppi_df has columns ['ga','gb']
             with uppercase gene symbols, and ppi_gene_set is the union of all
             genes appearing in any kept PPI.
    """
    log.info("Step 1/8: Loading STRING PPI data ...")

    # Protein-id -> preferred-gene-symbol lookup table.
    info_df = pd.read_csv(os.path.join(string_dir, info_file), sep='\t')
    ensembl_to_gene = dict(zip(
        info_df['#string_protein_id'],
        info_df['preferred_name'].str.upper()
    ))

    # Full PPI link table with per-channel and combined confidence scores.
    ppi_df = pd.read_csv(os.path.join(string_dir, links_file), sep=' ')

    # Keep only high-confidence interactions at or above the protocol threshold.
    ppi_df = ppi_df[ppi_df['combined_score'] >= score_threshold]

    # Map both STRING protein endpoints to uppercase HGNC gene symbols.
    ppi_df['ga'] = ppi_df['protein1'].map(ensembl_to_gene)
    ppi_df['gb'] = ppi_df['protein2'].map(ensembl_to_gene)
    # Drop rows where either endpoint failed to map to a gene symbol.
    ppi_df = ppi_df.dropna(subset=['ga', 'gb'])

    log.info(f"  High-confidence PPIs (score >= {score_threshold}): {len(ppi_df):,}")
    ppi_gene_set = set(ppi_df['ga']) | set(ppi_df['gb'])
    log.info(f"  Unique genes across kept PPIs: {len(ppi_gene_set):,}")
    return ppi_df, ppi_gene_set


# =========================================================================
# Step 2: Omics matrices loading (expression + CNV + mutation)
# =========================================================================

def load_omics_matrices(xena_dir, expr_file, cnv_file, mut_file, min_cnv_bytes):
    """
    @brief   Load three omics matrices from their Xena gzip-compressed TSV files.

    @param   xena_dir       Root directory containing the Xena files.
    @param   expr_file      Basename of the RNA-seq expression matrix.
    @param   cnv_file       Basename of the Gistic2 thresholded CNV matrix.
    @param   mut_file       Basename of the MC3 non-silent mutation matrix.
    @param   min_cnv_bytes  Minimum file size (bytes) for the CNV file to be loaded.
    @return  Tuple (expr_df, cnv_df, mut_df) where cnv_df may be None if the
             file is missing or too small; CNV is treated as an optional modality.
    """
    log.info("Step 2/8: Loading omics matrices ...")

    # Expression is mandatory: no fallback exists.
    log.info("  Loading gene expression matrix (RNA-seq) ...")
    expr_df = pd.read_csv(
        os.path.join(xena_dir, expr_file),
        sep='\t', index_col=0, compression='gzip', low_memory=False
    )
    log.info(f"    Expression shape: {expr_df.shape} (genes x samples)")

    # CNV is optional (may be absent on some hosts): degrade gracefully.
    cnv_path = os.path.join(xena_dir, cnv_file)
    if os.path.exists(cnv_path) and os.path.getsize(cnv_path) > min_cnv_bytes:
        log.info("  Loading CNV matrix (Gistic2 thresholded) ...")
        cnv_df = pd.read_csv(
            cnv_path, sep='\t', index_col=0, compression='gzip', low_memory=False
        )
        log.info(f"    CNV shape: {cnv_df.shape} (genes x samples)")
    else:
        log.warning(
            "  CNV matrix (Gistic2) not available or too small; "
            "proceeding with expression + mutation (2 omics) only."
        )
        cnv_df = None

    # Mutation is mandatory: no fallback exists.
    log.info("  Loading mutation matrix (MC3 non-silent) ...")
    mut_df = pd.read_csv(
        os.path.join(xena_dir, mut_file),
        sep='\t', index_col=0, compression='gzip', low_memory=False
    )
    log.info(f"    Mutation shape: {mut_df.shape} (genes x samples)")
    return expr_df, cnv_df, mut_df


# =========================================================================
# Step 3: Clinical -> sample-to-cancer mapping
# =========================================================================

def build_sample_to_cancer(xena_dir, clin_file, cancer_map):
    """
    @brief   Build a dict mapping each TCGA sample barcode prefix to its cancer type.

    @param   xena_dir    Root directory containing the Xena files.
    @param   clin_file   Basename of the dense phenotype TSV (clinical).
    @param   cancer_map  Dict of lowercase disease name -> TCGA abbreviation.
    @return  Dict: sample_id_prefix[:15] -> cancer_abbreviation.
    """
    log.info("Step 3/8: Loading clinical data and building sample->cancer map ...")
    clin_df = pd.read_csv(
        os.path.join(xena_dir, clin_file),
        sep='\t', compression='gzip', low_memory=False
    )
    # TCGA sample barcodes: first 15 chars = patient-level unique id that
    # matches the first 15 chars of all omics matrix column barcodes.
    clin_df['sid'] = clin_df['sample'].str[:15]

    sample_to_cancer = {}
    for _, row in clin_df.iterrows():
        disease = str(row.get('_primary_disease', '')).lower()
        abbr = cancer_map.get(disease, '')
        if abbr:
            sample_to_cancer[row['sid']] = abbr
    log.info(f"  Mapped samples with known cancer type: {len(sample_to_cancer):,}")
    return sample_to_cancer


# =========================================================================
# Step 4: Gene alignment and top-N selection by PPI degree
# =========================================================================

def align_and_select_genes(expr_df, mut_df, cnv_df, ppi_gene_set, ppi_df, n_genes):
    """
    @brief   Find the omics ∩ PPI gene intersection and select the top-N by PPI degree.

    @param   expr_df       Expression data frame (genes x samples; index = gene symbols).
    @param   mut_df        Mutation data frame (genes x samples; index = gene symbols).
    @param   cnv_df        Optional CNV data frame (or None).
    @param   ppi_gene_set  Set of genes present in the filtered PPI network.
    @param   ppi_df        Filtered PPI data frame (columns 'ga' and 'gb').
    @param   n_genes       Number of top genes to keep by PPI degree.
    @return  Tuple (top_genes_list, top_genes_set, gene_to_idx, ppi_sub_df)
             where ppi_sub_df is restricted to the pre-selection common genes.
    """
    log.info("Step 4/8: Aligning genes across omics modalities and selecting top-N ...")

    # Compute the intersection of genes measured across ALL omics matrices.
    common_omic_genes = set(expr_df.index) & set(mut_df.index)
    if cnv_df is not None:
        common_omic_genes &= set(cnv_df.index)
    # Restrict further to genes that appear in at least one PPI.
    common_genes = sorted(common_omic_genes & ppi_gene_set)
    log.info(f"  Common genes (omics ∩ PPI): {len(common_genes):,}")

    # Select top N_GENES by unweighted PPI degree computed on the
    # common-gene-only subgraph (ensures degree reflects retained topology).
    ppi_sub_df = ppi_df[ppi_df['ga'].isin(common_genes) & ppi_df['gb'].isin(common_genes)]
    gene_degree = {}
    for _, r in ppi_sub_df.iterrows():
        gene_degree[r['ga']] = gene_degree.get(r['ga'], 0) + 1
        gene_degree[r['gb']] = gene_degree.get(r['gb'], 0) + 1

    top_genes_list = sorted(gene_degree, key=gene_degree.get, reverse=True)[:n_genes]
    top_genes_set = set(top_genes_list)
    gene_to_idx = {g: i for i, g in enumerate(top_genes_list)}
    log.info(f"  Selected top-{len(top_genes_list):,} genes by PPI degree")
    return top_genes_list, top_genes_set, gene_to_idx, ppi_sub_df


# =========================================================================
# Step 5: Sample filtering (valid samples + cancer cohort thresholding)
# =========================================================================

def filter_valid_samples(expr_df, mut_df, cnv_df, sample_to_cancer, min_samples_per_cancer):
    """
    @brief   Keep samples present in all omics and whose cancer meets the size threshold.

    @param   expr_df              Expression frame (columns = sample barcodes).
    @param   mut_df               Mutation frame (columns = sample barcodes).
    @param   cnv_df               Optional CNV frame or None.
    @param   sample_to_cancer     Dict sample_id[:15] -> cancer_abbr.
    @param   min_samples_per_cancer  Drop cancer types with fewer samples.
    @return  Tuple (valid_samples, target_cancers, cancer_counts) where
             valid_samples is an ordered list of sample barcodes,
             target_cancers is the sorted list of kept cancer abbreviations,
             and cancer_counts maps cancer -> integer sample count.
    """
    log.info("Step 5/8: Filtering samples by omics overlap and cancer cohort size ...")

    # Start with samples present in every omics matrix.
    common_samples = sorted(set(expr_df.columns) & set(mut_df.columns))
    if cnv_df is not None:
        common_samples = sorted(set(common_samples) & set(cnv_df.columns))

    # First pass: count samples per mapped cancer type.
    preliminary_valid = []
    preliminary_counts = {}
    for s in common_samples:
        ct = sample_to_cancer.get(s[:15], '')
        if ct:
            preliminary_valid.append(s)
            preliminary_counts[ct] = preliminary_counts.get(ct, 0) + 1

    # Second pass: keep only cancers whose sample count meets the protocol minimum.
    target_cancers = sorted(
        [ct for ct, c in preliminary_counts.items() if c >= min_samples_per_cancer]
    )
    target_cancer_set = set(target_cancers)
    valid_samples = [
        s for s in preliminary_valid
        if sample_to_cancer.get(s[:15], '') in target_cancer_set
    ]
    # Recompute counts over the final kept sample list (stable ordering).
    cancer_counts = {
        ct: sum(1 for s in valid_samples if sample_to_cancer.get(s[:15]) == ct)
        for ct in target_cancers
    }

    log.info(f"  Cancer types kept (>= {min_samples_per_cancer} samples each): "
             f"{len(target_cancers)}")
    for ct, c in sorted(cancer_counts.items()):
        log.info(f"    {ct}: {c}")
    log.info(f"  Total patient samples kept: {len(valid_samples):,}")
    return valid_samples, target_cancers, cancer_counts


# =========================================================================
# Step 6: Build patient-level and gene-level feature matrices (PCA padded)
# =========================================================================

def mean_across_samples(omics_df, samples, top_gene_set):
    """
    @brief   Compute per-sample mean over a subset of genes (patient-level omic summary).

    @param   omics_df      Omics matrix (genes x samples).
    @param   samples       Ordered list of sample barcodes (defines output order).
    @param   top_gene_set  Set of genes to include in the mean.
    @return  np.ndarray of shape (len(samples),) with dtype float32; NaNs replaced by 0.
    """
    # Keep only columns that actually exist in this omics matrix.
    cols = [s for s in samples if s in omics_df.columns]
    series = omics_df.loc[omics_df.index.isin(top_gene_set), cols].mean(axis=0)
    # Reindex to the canonical sample order; fill NaNs with 0 (missing columns -> 0).
    return series.reindex(samples).to_numpy(dtype=np.float32)


def mean_across_genes(omics_df, samples, top_gene_set):
    """
    @brief   Compute per-gene mean across all samples (gene-level omic summary).

    @param   omics_df      Omics matrix (genes x samples).
    @param   samples       List of sample barcodes to include in the mean.
    @param   top_gene_set  Ordered iterable of genes (defines output order).
    @return  np.ndarray of shape (len(top_gene_set),) with dtype float32.
    """
    cols = [s for s in samples if s in omics_df.columns]
    series = omics_df.loc[omics_df.index.isin(top_gene_set), cols].mean(axis=1)
    # Reindex to the canonical gene order; fill NaNs with 0.
    return series.reindex(list(top_gene_set)).to_numpy(dtype=np.float32)


def build_patient_and_gene_features(
    expr_df, mut_df, cnv_df,
    top_gene_set, top_genes_list,
    valid_samples, target_dim
):
    """
    @brief   Stack omic summaries into feature matrices and standardize/pad to target dim.

    @param   expr_df       Expression matrix (genes x samples).
    @param   mut_df        Mutation matrix (genes x samples).
    @param   cnv_df        Optional CNV matrix or None.
    @param   top_gene_set  Set of selected genes.
    @param   top_genes_list Ordered list of selected genes (defines row order).
    @param   valid_samples Ordered list of sample barcodes (defines row order for patient x).
    @param   target_dim    Canonical feature width; pad with zeros if feature dim < target_dim.
    @return  Tuple (patient_x, gene_x, cancer_x) as torch.float32 tensors.
    """
    log.info("Step 6/8: Building patient / gene / cancer feature matrices ...")
    n_samples = len(valid_samples)

    # --- Stack per-omic summaries for patients and genes ---
    patient_omic_summaries = [
        mean_across_samples(expr_df, valid_samples, top_gene_set),
        mean_across_samples(mut_df, valid_samples, top_gene_set),
    ]
    gene_omic_summaries = [
        mean_across_genes(expr_df, valid_samples, top_gene_set),
        mean_across_genes(mut_df, valid_samples, top_gene_set),
    ]
    if cnv_df is not None:
        # CNV is thresholded copy-number (sign is arbitrary), use abs(|CNV|).
        patient_omic_summaries.append(
            np.abs(mean_across_samples(cnv_df, valid_samples, top_gene_set))
        )
        gene_omic_summaries.append(
            np.abs(mean_across_genes(cnv_df, valid_samples, top_gene_set))
        )

    # --- Assemble patient features ---
    patient_feats = np.nan_to_num(
        np.stack(patient_omic_summaries, axis=1), nan=0.0
    )
    patient_feats_norm = StandardScaler().fit_transform(patient_feats)
    if patient_feats_norm.shape[1] < target_dim:
        assert patient_feats_norm.shape[1] <= target_dim
        pad = np.zeros((n_samples, target_dim - patient_feats_norm.shape[1]),
                       dtype=np.float32)
        patient_feats_norm = np.concatenate([patient_feats_norm, pad], axis=1)
    patient_x = torch.tensor(patient_feats_norm, dtype=torch.float32)

    # --- Assemble gene features ---
    gene_feats = np.nan_to_num(
        np.stack(gene_omic_summaries, axis=1), nan=0.0
    )
    gene_feats_norm = StandardScaler().fit_transform(gene_feats)
    if gene_feats_norm.shape[1] < target_dim:
        assert gene_feats_norm.shape[1] <= target_dim
        pad = np.zeros((len(top_genes_list), target_dim - gene_feats_norm.shape[1]),
                       dtype=np.float32)
        gene_feats_norm = np.concatenate([gene_feats_norm, pad], axis=1)
    gene_x = torch.tensor(gene_feats_norm, dtype=torch.float32)

    # --- Cancer features: simple one-hot encoding per cancer type ---
    cancer_x = torch.eye(len(target_cancers), dtype=torch.float32)

    log.info(f"  Gene   x shape: {tuple(gene_x.shape)}")
    log.info(f"  Patient x shape: {tuple(patient_x.shape)}")
    log.info(f"  Cancer  x shape: {tuple(cancer_x.shape)}")
    return patient_x, gene_x, cancer_x


# =========================================================================
# Step 7: Build edge indices (PPI / patient->cancer / gene<->cancer)
# =========================================================================

def build_edge_indices(
    ppi_sub_df, top_genes_set, gene_to_idx,
    valid_samples, sample_to_cancer, target_cancers,
    top_genes_list
):
    """
    @brief   Build all three heterogeneous edge sets as undirected/bipartite tensors.

    @param   ppi_sub_df       PPI subframe restricted to common genes.
    @param   top_genes_set    Set of final selected genes.
    @param   gene_to_idx      Dict gene_symbol -> row index.
    @param   valid_samples    Ordered list of patient sample barcodes.
    @param   sample_to_cancer Dict sample[:15] -> cancer abbreviation.
    @param   target_cancers   Sorted list of cancer abbreviations.
    @param   top_genes_list   Ordered list of selected genes.
    @return  Tuple (ppi_ei, pc_ei, gc_ei, cg_ei) each a torch.long edge index.
    """
    log.info("Step 7/8: Building heterogeneous edge indices ...")

    # --- PPI edges (gene-gene undirected): store both directions in edge_index ---
    ppi_filtered = ppi_sub_df[
        ppi_sub_df['ga'].isin(top_genes_set) &
        ppi_sub_df['gb'].isin(top_genes_set)
    ]
    src, dst = [], []
    for _, r in ppi_filtered.iterrows():
        if r['ga'] in gene_to_idx and r['gb'] in gene_to_idx:
            src.append(gene_to_idx[r['ga']])
            dst.append(gene_to_idx[r['gb']])
    # Duplicate edges in both directions to make the adjacency symmetric.
    ppi_ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    log.info(f"  PPI edges (symmetrized): {ppi_ei.shape[1]:,}")

    # --- Patient -> Cancer edges (each patient belongs to exactly one cancer) ---
    cancer_to_idx = {c: i for i, c in enumerate(target_cancers)}
    pc_src, pc_dst = [], []
    for i, s in enumerate(valid_samples):
        ct = sample_to_cancer.get(s[:15])
        if ct in cancer_to_idx:
            pc_src.append(i)
            pc_dst.append(cancer_to_idx[ct])
    pc_ei = torch.tensor([pc_src, pc_dst], dtype=torch.long)
    log.info(f"  Patient->Cancer edges: {pc_ei.shape[1]:,}")

    # --- Gene <-> Cancer edges (dense bipartite, used by C3L contrastive learning) ---
    # Every gene is linked to every cancer type (the contrastive loss then
    # pushes apart the non-expressing cancers during training).
    gc_src, gc_dst = [], []
    for gi in range(len(top_genes_list)):
        for ci in range(len(target_cancers)):
            gc_src.append(gi)
            gc_dst.append(ci)
    gc_ei = torch.tensor([gc_src, gc_dst], dtype=torch.long)
    cg_ei = torch.tensor([gc_dst, gc_src], dtype=torch.long)
    log.info(f"  Gene<->Cancer edges (full bipartite): {gc_ei.shape[1]:,} each direction")
    return ppi_ei, pc_ei, gc_ei, cg_ei


# =========================================================================
# Step 8: Assemble and save graph
# =========================================================================

def assemble_and_save_graph(
    output_dir, out_filename,
    gene_x, top_genes_list,
    cancer_x, target_cancers, cancer_counts,
    patient_x, valid_samples, sample_to_cancer,
    ppi_ei, pc_ei, gc_ei, cg_ei,
    gene_to_idx, target_dim
):
    """
    @brief   Assemble the final HeteroData-style dictionary and write it to disk.

    @param   output_dir         Directory where the graph .pt will be written.
    @param   out_filename       Output basename.
    @param   gene_x             Gene feature tensor (n_genes, dim).
    @param   top_genes_list     Ordered list of gene symbols.
    @param   cancer_x           Cancer feature tensor (n_cancers, dim).
    @param   target_cancers     Ordered list of cancer abbreviations.
    @param   cancer_counts      Dict cancer -> sample count.
    @param   patient_x          Patient feature tensor (n_samples, dim).
    @param   valid_samples      Ordered list of sample barcodes.
    @param   sample_to_cancer   Dict sample[:15] -> cancer abbreviation.
    @param   ppi_ei             PPI edge index tensor.
    @param   pc_ei              Patient->Cancer edge index tensor.
    @param   gc_ei              Gene->Cancer edge index tensor.
    @param   cg_ei              Cancer->Gene edge index tensor.
    @param   gene_to_idx        Dict gene_symbol -> node index (stored for downstream tools).
    @param   target_dim         Canonical embedding width (stored in meta).
    @return  None; writes graph to disk and prints final summary.
    """
    log.info("Step 8/8: Assembling graph object and saving to disk ...")

    n_genes = len(top_genes_list)
    n_cancers = len(target_cancers)
    n_samples = len(valid_samples)

    graph = {
        # --- Node-level data ---
        'gene':    {'x': gene_x,    'num_nodes': n_genes},
        'cancer':  {'x': cancer_x,  'num_nodes': n_cancers},
        'patient': {'x': patient_x, 'num_nodes': n_samples},
        # --- Edge-level data (heterogeneous relation types) ---
        ('gene',    'ppi',          'gene'):   {'edge_index': ppi_ei},
        ('patient', 'belongs_to',   'cancer'): {'edge_index': pc_ei},
        ('gene',    'expressed_in', 'cancer'): {'edge_index': gc_ei},
        ('cancer',  'expresses',    'gene'):   {'edge_index': cg_ei},
        # --- Provenance metadata ---
        'meta': {
            'gene_list':      top_genes_list,
            'cancer_list':    target_cancers,
            'cancer_counts':  cancer_counts,
            'n_genes':        n_genes,
            'n_cancers':      n_cancers,
            'n_samples':      n_samples,
            'embedding_dim':  target_dim,
            'gene_to_idx':    gene_to_idx,
            'ppi_edges':      ppi_ei.shape[1],
            'patient_ids':    list(valid_samples),
            'patient_cancer': [sample_to_cancer.get(s[:15], '') for s in valid_samples],
        }
    }

    full_out_path = os.path.join(output_dir, out_filename)
    torch.save(graph, full_out_path)
    size_mb = os.path.getsize(full_out_path) / 1024 / 1024

    log.info(f"Graph saved successfully: {full_out_path} ({size_mb:.1f} MB)")
    log.info(f"  Genes: {n_genes:,} | "
             f"Cancers: {n_cancers} | "
             f"PPI edges: {ppi_ei.shape[1]:,} | "
             f"Samples: {n_samples:,}")


def main():
    """
    @brief   Top-level driver for the Xena large pan-cancer graph builder.

    Steps:
        1. Load STRING PPI tables (filter, map to gene symbols).
        2. Load omics matrices (expression / CNV / mutation).
        3. Parse clinical table to sample->cancer mapping.
        4. Align genes across omics ∩ PPI; select top-N by degree.
        5. Filter samples and cancer cohorts by minimum size.
        6. Build and standardize patient / gene / cancer feature matrices.
        7. Build edge indices (PPI / patient->cancer / gene<->cancer).
        8. Assemble graph dict and persist to disk.
    @return  None.
    """
    # ---- Step 1: STRING PPI ----
    ppi_df, ppi_gene_set = load_string_ppi(
        STRING_DIR, STRING_INFO_FILE, STRING_LINKS_FILE, PPI_THRESHOLD
    )

    # ---- Step 2: Omics matrices ----
    expr_df, cnv_df, mut_df = load_omics_matrices(
        XENA_DIR, XENA_EXPR_FILE, XENA_CNV_FILE, XENA_MUT_FILE,
        MIN_CNV_FILE_SIZE_BYTES
    )

    # ---- Step 3: Sample->Cancer mapping ----
    sample_to_cancer = build_sample_to_cancer(
        XENA_DIR, XENA_CLIN_FILE, CANCER_TYPE_MAP
    )

    # ---- Step 4: Gene alignment and top-N selection ----
    top_genes_list, top_genes_set, gene_to_idx, ppi_sub_df = align_and_select_genes(
        expr_df, mut_df, cnv_df, ppi_gene_set, ppi_df, N_GENES
    )

    # ---- Step 5: Sample / cohort filtering ----
    valid_samples, target_cancers, cancer_counts = filter_valid_samples(
        expr_df, mut_df, cnv_df, sample_to_cancer, MIN_SAMPLES_PER_CANCER
    )

    # ---- Step 6: Feature matrices ----
    patient_x, gene_x, cancer_x = build_patient_and_gene_features(
        expr_df, mut_df, cnv_df,
        top_genes_set, top_genes_list,
        valid_samples, DIM
    )

    # ---- Step 7: Edge indices ----
    ppi_ei, pc_ei, gc_ei, cg_ei = build_edge_indices(
        ppi_sub_df, top_genes_set, gene_to_idx,
        valid_samples, sample_to_cancer, target_cancers,
        top_genes_list
    )

    # ---- Step 8: Assemble + save ----
    assemble_and_save_graph(
        OUTPUT_DIR, OUT_GRAPH_FILE,
        gene_x, top_genes_list,
        cancer_x, target_cancers, cancer_counts,
        patient_x, valid_samples, sample_to_cancer,
        ppi_ei, pc_ei, gc_ei, cg_ei,
        gene_to_idx, DIM
    )


if __name__ == "__main__":
    main()
