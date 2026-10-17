#!/usr/bin/env python3
"""
@file    66_tissue_specific_ppi.py
@brief   Tissue-specific PPI filtering. Removes "phantom" PPI edges: in a given
         tissue/cancer an interaction between two proteins is biologically meaningful
         only if both proteins are expressed there. This script filters a STRING
         PPI edge list using tissue expression from GTEx or the Human Protein Atlas
         (HPA): an edge survives if BOTH genes have TPM >= threshold in the target
         tissue. `filter_ppi_by_expression` is pure pandas/numpy and unit-testable.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  --ppi            config.STRING_DIR/9606.protein.links.full.v12.0.txt (STRING scores)
        --info           config.STRING_DIR/9606.protein.info.v12.0.txt    (protein-id -> symbol)
        --tissue_expr    GTEx/HPA median TPM file (genes x tissues)       (required)
        --tissue         tissue column name to use                        (required)
        --threshold      TPM floor for both endpoints (default 1.0)
        --score_threshold STRING combined_score floor (default config.PPI_SCORE_THRESHOLD)
        --out            optional output TSV path
Output: stdout (edge counts before/after filtering) and, if --out is set, the filtered PPI TSV
Run:    python scripts/66_tissue_specific_ppi.py --tissue_expr <file> --tissue <name>
"""
import argparse
import os

import numpy as np
import pandas as pd

import config

# ============================================================================
# Configuration Constants (artifact names and default thresholds)
# ============================================================================

# Per-run artifact names (resolved through config so paths stay env-overridable).
PPI_LINKS_NAME = "9606.protein.links.full.v12.0.txt"
PPI_INFO_NAME = "9606.protein.info.v12.0.txt"

# Default tissue expression threshold (TPM floor for both edge endpoints).
DEFAULT_TPM_THRESHOLD = 1.0

# GCT file format parameters: skip rows, index column, description column index.
GCT_SKIP_ROWS = 2
GCT_GENE_INDEX_COL = 1
GCT_DESC_COL_INDEX = 1  # drop after index col

# TSV (non-GCT) file format parameters.
TSV_GENE_INDEX_COL = 0

# STRING PPI file column names.
PPI_COL_PROTEIN1 = "protein1"
PPI_COL_PROTEIN2 = "protein2"
PPI_COL_COMBINED_SCORE = "combined_score"

# STRING info file column names.
INFO_COL_PROTEIN_ID = "#string_protein_id"
INFO_COL_PREFERRED_NAME = "preferred_name"

# Filtered PPI output column names.
OUT_COL_GENE_A = "gene_a"
OUT_COL_GENE_B = "gene_b"

# Print formatting: thousands separator and percentage precision.
PRINT_PERCENT_PRECISION = 1
PRINT_THOUSANDS_SEP = ","


# ============================================================================
# Core Functions
# ============================================================================

def filter_ppi_by_expression(ppi, tissue_expr, threshold=DEFAULT_TPM_THRESHOLD):
    """
    @brief   Drop PPI rows where either endpoint gene has TPM < threshold.

    @details An edge is retained only if BOTH genes pass the tissue TPM threshold,
             implementing the "both-endpoints-expressed" biological criterion.

    @param   ppi           PPI DataFrame with columns ['gene_a', 'gene_b'].
    @param   tissue_expr   Series mapping gene symbol -> TPM (single tissue profile).
    @param   threshold     Minimum TPM required for BOTH genes to keep an edge.
    @return  Filtered copy of ppi DataFrame containing only edges where both genes pass.
    @note    Returns empty DataFrame unchanged; genes not in tissue_expr get NaN and fail.
    """
    if ppi.empty:
        return ppi

    # Build a TPM lookup for the union of all genes in the PPI edge list.
    gene_union = pd.Index(set(ppi[OUT_COL_GENE_A]) | set(ppi[OUT_COL_GENE_B]))
    expr = tissue_expr.reindex(gene_union)

    # Keep edges where BOTH endpoints meet or exceed the TPM threshold.
    keep_a = ppi[OUT_COL_GENE_A].map(expr) >= threshold
    keep_b = ppi[OUT_COL_GENE_B].map(expr) >= threshold
    return ppi[keep_a & keep_b].copy()


def load_ppi_with_genes(ppi_path, info_path, score_threshold=config.PPI_SCORE_THRESHOLD):
    """
    @brief   Load STRING PPI edges, map protein IDs to gene symbols, and
             apply combined_score threshold.

    @param   ppi_path          Path to STRING protein.links.full file.
    @param   info_path         Path to STRING protein.info ID-to-symbol mapping file.
    @param   score_threshold   Minimum combined_score required to retain an edge.
    @return  PPI DataFrame with columns ['gene_a', 'gene_b', ...] after filtering.
    @note    Drops rows where either endpoint failed to map to a gene symbol.
    """
    # Load protein-ID-to-gene-symbol mapping.
    info = pd.read_csv(info_path, sep='\t')
    e2g = dict(zip(info[INFO_COL_PROTEIN_ID], info[INFO_COL_PREFERRED_NAME].str.upper()))

    # Load PPI score matrix and apply combined_score minimum threshold.
    ppi = pd.read_csv(ppi_path, sep=' ')
    ppi = ppi[ppi[PPI_COL_COMBINED_SCORE] >= score_threshold]

    # Map STRING protein IDs to uppercase gene symbols.
    ppi[OUT_COL_GENE_A] = ppi[PPI_COL_PROTEIN1].map(e2g)
    ppi[OUT_COL_GENE_B] = ppi[PPI_COL_PROTEIN2].map(e2g)

    # Drop edges with unmapped gene symbols.
    return ppi.dropna(subset=[OUT_COL_GENE_A, OUT_COL_GENE_B])


def main():
    """
    @brief   Main entry point: load PPI, apply tissue-specific expression filter,
             report statistics, and optionally save filtered PPI.
    @return  None (outputs statistics to stdout; optionally writes filtered TSV).
    @note    Supports both GTEx .gct(.gz) and plain genes x tissues TSV formats.
    """
    # Parse command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppi', default=os.path.join(config.STRING_DIR, PPI_LINKS_NAME))
    ap.add_argument('--info', default=os.path.join(config.STRING_DIR, PPI_INFO_NAME))
    ap.add_argument('--tissue_expr', required=True,
                    help='GTEx/HPA median TPM file (genes x tissues)')
    ap.add_argument('--tissue', required=True, help='tissue column name to use')
    ap.add_argument('--threshold', type=float, default=DEFAULT_TPM_THRESHOLD)
    ap.add_argument('--score_threshold', type=int, default=config.PPI_SCORE_THRESHOLD)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # Load STRING PPI with gene-symbol mapping and score thresholding.
    ppi = load_ppi_with_genes(args.ppi, args.info, args.score_threshold)
    n_before = len(ppi)

    # Load tissue expression: handle GTEx .gct(.gz) or plain TSV format.
    if args.tissue_expr.endswith('.gct.gz') or args.tissue_expr.endswith('.gct'):
        expr = pd.read_csv(
            args.tissue_expr, sep='\t',
            skiprows=GCT_SKIP_ROWS, index_col=GCT_GENE_INDEX_COL,
            compression='gzip'
        )
        expr = expr.iloc[:, GCT_DESC_COL_INDEX:]
    else:
        expr = pd.read_csv(args.tissue_expr, sep='\t', index_col=TSV_GENE_INDEX_COL)

    # Validate requested tissue column exists in expression file.
    if args.tissue not in expr.columns:
        raise SystemExit(
            f"tissue '{args.tissue}' not found; available: {list(expr.columns)[:20]}..."
        )
    tissue_expr = expr[args.tissue]

    # Normalize gene symbols: uppercase and deduplicate (multiple Ensembl -> one gene).
    tissue_expr.index = tissue_expr.index.str.upper()
    tissue_expr = tissue_expr.groupby(level=0).max()

    # Apply tissue-expression PPI filter.
    filtered = filter_ppi_by_expression(ppi, tissue_expr, args.threshold)

    # Report before/after edge counts and removal percentage.
    n_removed = n_before - len(filtered)
    pct_removed = 100 * n_removed / max(n_before, 1)
    print(f"PPI before filter: {n_before:{PRINT_THOUSANDS_SEP}}")
    print(f"PPI after  filter: {len(filtered):{PRINT_THOUSANDS_SEP}}  "
          f"(tissue='{args.tissue}', TPM>={args.threshold})")
    print(f"removed: {n_removed:{PRINT_THOUSANDS_SEP}} "
          f"({pct_removed:.{PRINT_PERCENT_PRECISION}f}%)")

    # Optionally save filtered PPI edges to TSV.
    if args.out:
        filtered.to_csv(args.out, sep='\t', index=False)
        print(f"Saved filtered PPI to {args.out}")


if __name__ == '__main__':
    main()
