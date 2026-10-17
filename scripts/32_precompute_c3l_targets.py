#!/usr/bin/env python3
"""
@file    32_precompute_c3l_targets.py
@brief   Precompute biologically grounded C3L (gene-cancer contrastive) positive
         InfoNCE targets from Xena expression data: for each gene, find the cancer
         type with the highest mean log-normalized expression, saved as a tensor.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup feature_engineering

Precompute biologically grounded C3L (gene-cancer contrastive) targets: for every
gene, compute its mean log-normalized expression in each cancer type and take the
argmax cancer as the positive InfoNCE target, saved as a [n_genes] tensor of
cancer-type indices aligned to the graph's gene order. This replaces random or
simplified targets with real expression-derived supervision.

Input:  --graph (a .pt graph carrying meta.gene_list / meta.cancer_list),
        --expr (Xena gene-expression gz; default config.XENA_DIR / XENA_EXPR_FILE),
        --clin (Xena phenotype gz; default config.XENA_DIR / XENA_CLIN_FILE),
        --out (default config.DATA_PROC / OUT_TARGETS_FILE), --top_k.
Output: the targets/specificity .pt file at --out plus a stdout summary.
Run:    python scripts/32_precompute_c3l_targets.py --graph <graph.pt> [--expr <f> --clin <f> --out <o>]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import config

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================

# --- Xena input artifact names (resolved under config.XENA_DIR) ---
XENA_EXPR_FILENAME = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
XENA_CLIN_FILENAME = "TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

# --- Default output filename (resolved under config.DATA_PROC) ---
OUT_TARGETS_FILENAME = "gene_cancer_targets.pt"

# Length of the TCGA sample barcode prefix that uniquely identifies a patient
# (used to join the clinical table against omics matrices whose columns carry
# the full TCGA barcode with additional suffixes).
TCGA_PATIENT_BARCODE_PREFIX_LEN = 15

# --- Xena disease name -> TCGA abbreviation (mirrors the graph builders' map) ---
# Keys are the exact lowercase strings found in the Xena phenotype
# '_primary_disease' column; values are standard TCGA TSS abbreviations.
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

# Known marker genes printed as sanity-check examples after target computation.
SANITY_CHECK_MARKER_GENES = ('TP53', 'EGFR', 'KRAS')


# =========================================================================
# Core computation functions
# =========================================================================

def load_sample_to_cancer(clinical_path):
    """
    @brief   Parse the Xena phenotype TSV and build sample[:15] -> cancer abbr mapping.

    @param   clinical_path  Absolute path to the gzipped clinical TSV file.
    @return  Dict: sample_id_prefix[:15] -> cancer_abbreviation_string.
    """
    clin = pd.read_csv(
        clinical_path, sep='\t', compression='gzip', low_memory=False
    )
    # TCGA patient-level barcode: first 15 chars of the full sample barcode.
    clin['sid'] = clin['sample'].str[:TCGA_PATIENT_BARCODE_PREFIX_LEN]
    sample_to_cancer = {}
    for _, row in clin.iterrows():
        disease = str(row.get('_primary_disease', '')).lower()
        abbr = CANCER_TYPE_MAP.get(disease, '')
        if abbr:
            sample_to_cancer[row['sid']] = abbr
    return sample_to_cancer


def compute_gene_cancer_specificity(
    expression_df,
    sample_to_cancer,
    gene_list,
    cancer_list,
    top_k=1,
):
    """
    @brief   Compute per-gene mean expression per cancer type and return argmax targets.

    @param   expression_df     pd.DataFrame: genes x samples (log-normalized values).
    @param   sample_to_cancer  Dict: sample_id[:15] -> cancer_abbr.
    @param   gene_list         Ordered list of gene symbols defining the output row order.
    @param   cancer_list       Ordered list of cancer abbreviations defining columns.
    @param   top_k             Reserved parameter; currently only top-1 argmax is used.
    @return  Tuple (targets_tensor, specificity_matrix) where:
             - targets_tensor: torch.long [n_genes] index into cancer_list per gene.
             - specificity_matrix: np.float32 [n_genes, n_cancers] raw mean expression values.
    @note    Unmeasured gene-cancer pairs are encoded as NaN; nanargmax skips NaN columns.
    """
    cancer_to_idx = {c: i for i, c in enumerate(cancer_list)}
    n_genes, n_cancers = len(gene_list), len(cancer_list)

    # Restrict samples to those whose cancer type is one of the target cancers.
    usable_samples = [
        s for s in expression_df.columns
        if sample_to_cancer.get(s[:TCGA_PATIENT_BARCODE_PREFIX_LEN], '') in cancer_to_idx
    ]

    # NaN initialization: gene-cancer pairs with 0 supporting samples stay NaN
    # so nanargmax below correctly ignores them (avoids spuriously picking a
    # zero-filled column over real measured ones).
    specificity_matrix = np.full(
        (n_genes, n_cancers), np.nan, dtype=np.float32
    )
    for ci, cancer in enumerate(cancer_list):
        # Collect all sample columns belonging to this cancer type.
        cancer_samples = [
            s for s in usable_samples
            if sample_to_cancer[s[:TCGA_PATIENT_BARCODE_PREFIX_LEN]] == cancer
        ]
        if not cancer_samples:
            continue
        # Row-wise mean over the current cancer's sample columns, reindexed to
        # the canonical gene_list order (filling missing genes with NaN).
        sub_df = expression_df.loc[expression_df.index.isin(gene_list), cancer_samples]
        specificity_matrix[:, ci] = (
            sub_df.reindex(gene_list).mean(axis=1).to_numpy(dtype=np.float32)
        )

    # Argmax over available (non-NaN) cancer columns per gene.
    # np.nanargmax raises ValueError if an entire row is NaN; this is acceptable
    # because downstream code only uses targets for rows with at least one measurement.
    targets_np = np.nanargmax(specificity_matrix, axis=1)
    targets_tensor = torch.from_numpy(targets_np.astype(np.int64))
    return targets_tensor, specificity_matrix


# =========================================================================
# Main entry
# =========================================================================

def main():
    """
    @brief   Top-level driver: parse CLI, load graph metadata + Xena tables,
             compute targets, serialize checkpoint, print sanity report.

    @return  None. Writes output checkpoint and prints summary to stdout.
    """
    ap = argparse.ArgumentParser(
        description="Precompute C3L gene->cancer supervision targets from Xena expression data."
    )
    ap.add_argument(
        '--graph', required=True,
        help='Path to a saved graph .pt (xena_graph_esm.pt or xena_graph_large.pt); '
             'its meta.gene_list / meta.cancer_list define the output ordering.'
    )
    ap.add_argument(
        '--out',
        default=os.path.join(config.DATA_PROC, OUT_TARGETS_FILENAME),
        help='Output .pt checkpoint path.'
    )
    ap.add_argument(
        '--expr', default=None,
        help='Xena expression gz path (default: <GFM_XENA_DIR>/EB++AdjustPANCAN...xena.gz).'
    )
    ap.add_argument(
        '--clin', default=None,
        help='Xena phenotype gz path (default: <GFM_XENA_DIR>/TCGA_phenotype...gz).'
    )
    ap.add_argument(
        '--top_k', type=int, default=1,
        help='Reserved parameter: top-k target cancers per gene (currently top-1 only).'
    )
    args = ap.parse_args()

    # --- Resolve Xena input paths ---
    if not config.XENA_DIR and (not args.expr or not args.clin):
        raise SystemExit(
            "FATAL: No Xena input paths available. Set GFM_XENA_DIR env var or "
            "pass --expr/--clin explicitly pointing at the Xena pan-cancer data."
        )
    expr_path = args.expr or os.path.join(config.XENA_DIR, XENA_EXPR_FILENAME)
    clin_path = args.clin or os.path.join(config.XENA_DIR, XENA_CLIN_FILENAME)

    # --- Load graph metadata (defines node ordering) ---
    graph_bundle = torch.load(args.graph, map_location='cpu')
    gene_list   = graph_bundle['meta']['gene_list']
    cancer_list = graph_bundle['meta']['cancer_list']
    print(f"[Init] Graph metadata loaded: {len(gene_list):,} genes x "
          f"{len(cancer_list)} cancers")

    # --- Load Xena expression matrix ---
    print(f"[Load] Reading expression matrix: {expr_path}")
    expr_df = pd.read_csv(
        expr_path, sep='\t', index_col=0, compression='gzip', low_memory=False
    )
    print(f"  Expression shape: {expr_df.shape} (genes x samples)")

    # --- Build sample->cancer mapping from clinical table ---
    sample_to_cancer = load_sample_to_cancer(clin_path)
    print(f"  Sample->cancer mappings: {len(sample_to_cancer):,}")

    # --- Compute target tensor + specificity matrix ---
    print("[Compute] Gene-cancer specificity matrix and argmax targets ...")
    targets, specificity_mat = compute_gene_cancer_specificity(
        expr_df, sample_to_cancer, gene_list, cancer_list, args.top_k
    )

    # --- Serialize checkpoint ---
    checkpoint = {
        'targets':     targets,                # [n_genes] long tensor
        'gene_list':   gene_list,               # Ordered gene symbols
        'cancer_list': cancer_list,             # Ordered cancer abbreviations
        'specificity': torch.from_numpy(specificity_mat),  # [n_genes, n_cancers]
    }
    torch.save(checkpoint, args.out)

    # --- Post-save sanity report ---
    # Count genes that had at least one non-NaN mean-expression value.
    n_genes_with_data = int((~np.isnan(specificity_mat).all(axis=1)).sum())
    print(f"\n[Done] Checkpoint saved: {args.out}")
    print(f"  Total genes            : {len(gene_list):,}")
    print(f"  Total cancers          : {len(cancer_list)}")
    print(f"  Genes with any data    : {n_genes_with_data:,}")

    # Print a small set of well-known marker genes as manual quality check.
    print(f"  Sanity-check marker genes:")
    for symbol in SANITY_CHECK_MARKER_GENES:
        if symbol in gene_list:
            i = gene_list.index(symbol)
            top_cancer = cancer_list[targets[i].item()]
            print(f"    {symbol:6s} -> argmax cancer: {top_cancer}")


if __name__ == '__main__':
    main()
