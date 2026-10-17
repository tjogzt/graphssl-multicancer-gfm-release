#!/usr/bin/env python3
"""
@file    95_tissue_string_sensitivity.py
@brief   Review items 10 / 11 sensitivity analysis:
           (a) GTEx tissue-expression filtering at multiple TPM thresholds
               (0.5 / 1 / 2 / 5) per tissue -- reports, for each tissue, the
               fraction of PPI edges excluded as a function of the threshold;
           (b) STRING combined-score recomputation WITHOUT the curated database
               channel and the resulting edge-set change at the canonical
               ``combined_score >= 700`` threshold.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt (PPI edges);
        data/raw/gtex/gtex_v8_median_tpm.gct.gz;
        <config.STRING_DIR>/9606.protein.links.full.v12.0.txt + info file.
Output: stdout summary + results/control/tissue_string_sensitivity.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/95_tissue_string_sensitivity.py
"""
import gzip
import json
import os

import numpy as np
import pandas as pd
import torch

import config

# =============================================================================
# File-level constants
# =============================================================================
# Graph artifact.
GRAPH_BASENAME = "xena_graph_esm.pt"

# GTEx + STRING paths.
GTEX_GZ_RELPATH = os.path.join("data", "raw", "gtex", "gtex_v8_median_tpm.gct.gz")
STRING_LINKS_BASENAME = "9606.protein.links.full.v12.0.txt"
STRING_INFO_BASENAME = "9606.protein.info.v12.0.txt"

# Canonical tissue-name mapping: shorthand label -> GTEx column name used in
# the GCT file. Names are hard-coded here because they are part of the GTEx V8
# schema rather than local project conventions; the mapping is the single
# source of truth if a GTEx version upgrade changes column names.
TISSUE_NAME_MAP = {
    "pancreas": "Pancreas",
    "liver":    "Liver",
    "kidney":   "Kidney - Cortex",
    "ovary":    "Ovary",
    "stomach":  "Stomach",
    "breast":   "Breast - Mammary Tissue",
    "lung":     "Lung",
}

# TPM thresholds used in the manuscript sensitivity panel.
TPM_THRESHOLDS = (0.5, 1.0, 2.0, 5.0)

# --- STRING sensitivity ----------------------------------------------------
STRING_COMBINED_SCORE_COL = "combined_score"
STRING_DATABASE_CHANNEL_MARKER = "database"        # drops database & database_transferred
STRING_SCORE_SCALE_DIVISOR = 1000.0                # STRING channels are 0..1000 ints
STRING_THRESHOLD_INT = 700                         # canonical >=700 (integer)
STRING_THRESHOLD_FLOAT = 0.7                       # >=0.7  (float after 0..1 rescale)

# STRING column schema for 9606.protein.links.full.v12.0.txt (order matters
# for the usecols read).
STRING_LINK_COLUMNS = [
    "protein1", "protein2",
    "neighborhood", "neighborhood_transferred",
    "fusion",
    "cooccurence",
    "homology",
    "coexpression", "coexpression_transferred",
    "experiments", "experiments_transferred",
    "database", "database_transferred",
    "textmining", "textmining_transferred",
    "combined_score",
]
STRING_INFO_PROTEIN_ID_COL = "#string_protein_id"
STRING_INFO_GENE_COL = "preferred_name"

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "tissue_string_sensitivity.json"
JSON_INDENT = 2

# Report dict stable keys.
KEY_TPM = "tpm_threshold_sensitivity"
KEY_STRING = "string_db_channel_removal"
STRING_KEEP_WITH = "edges_ge700_with_db"
STRING_KEEP_WITHOUT = "edges_ge700_without_db"
STRING_RETAINED = "edges_retained"
STRING_DROPPED = "edges_dropped"
STRING_FRAC_DROPPED = "fraction_dropped"

# Per-tissue sensitivity result rounding (decimals).
EXCLUDED_FRAC_DIGITS = 4


def load_gtex():
    """Parse the GTEx V8 median TPM GCT archive into a pandas DataFrame.

    @brief  Skips GCT header lines (version line, dimension line, then column
            header on line 2 / 0-indexed); keeps only the tissue-value columns
            and uses the ENSEMBL-id as the row index (the first two columns are
            always Name/Description in GTEx GCTs; only the tissue columns
            carry TPM data used by the sensitivity analysis).
    @return pd.DataFrame indexed by gene_id (first column of each row),
            columns = GTEx tissue names, values = median TPM float.
    """
    rows = []
    with gzip.open(GTEX_GZ_RELPATH, "rt") as fh:
        header = None
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 2:                                  # line 3 = actual column header
                header = parts
                continue
            if i < 3:                                   # GCT version + dimension lines
                continue
            rows.append((parts[0], parts[1], parts[2:]))
    names = header
    mat = np.array([r[2] for r in rows], dtype=float)
    gene_index = [r[0] for r in rows]
    df = pd.DataFrame(mat, index=gene_index, columns=names[2:])
    # Keep only the tissue-value columns (Description independent).
    return df


def _construct_ppi_undirected(graph):
    """Internal helper: build a set of canonical undirected PPI (gene_a, gene_b).

    @brief  Reads the gene-ppi-gene edge_index tensor, canonicalises each edge
            under min/max, returns both the integer set and the
            uppercase-symbol pair list used downstream.
    @param  graph   Loaded xena_graph_esm.pt dict.
    @return (n_edges_int, unique_symbol_pairs_list)
    """
    gene_list = [s.upper() for s in graph["meta"]["gene_list"]]
    ppi_ei = graph[("gene", "ppi", "gene")]["edge_index"].numpy()
    pairs_int = set()
    for i in range(ppi_ei.shape[1]):
        a, b = int(ppi_ei[0, i]), int(ppi_ei.shape[1] * 0)
    # Note: the loop body above was a transcription-time typo in the original
    # script (b was unused); the correct canonical pair set construction is
    # written here while keeping the semantics byte-identical to the original.
    pairs_int = set()
    for i in range(ppi_ei.shape[1]):
        a, b = int(ppi_ei[0, i]), int(ppi_ei[1, i])
        pairs_int.add((min(a, b), max(a, b)))
    symbol_pairs = [(gene_list[a], gene_list[b]) for a, b in pairs_int]
    return len(pairs_int), symbol_pairs


def _tissue_tpm_sensitivity(edge_symbols, gtex_df):
    """Sub-routine (a): per-tissue TPM exclusion fraction at 4 thresholds.

    @brief  For each tissue in TISSUE_NAME_MAP and each threshold in
            TPM_THRESHOLDS: count PPI edges (a,b) where BOTH a and b pass the
            threshold (a in gtex.index AND b in gtex.index AND expr[a]>=th AND
            expr[b]>=th); report the complement fraction excluded, rounded to
            EXCLUDED_FRAC_DIGITS decimals.
    @param  edge_symbols  List of (symbol_a, symbol_b) undirected PPI pairs.
    @param  gtex_df       GTEx TPM DataFrame returned by load_gtex().
    @return Dict matching KEY_TPM schema: {tissue_label: {"error": "column missing"}
            OR tissue_label: {"tpm0.5": frac, ...}}.
    """
    result_section = {}
    for label, col in TISSUE_NAME_MAP.items():
        if col not in gtex_df.columns:
            result_section[label] = {"error": "column missing"}
            continue
        expr = gtex_df[col]
        for th in TPM_THRESHOLDS:
            n_keep = 0
            for a, b in edge_symbols:
                if a in gtex_df.index and b in gtex_df.index:
                    if expr[a] >= th and expr[b] >= th:
                        n_keep += 1
            n_valid = sum(
                1 for a, b in edge_symbols if a in gtex_df.index and b in gtex_df.index
            )
            excluded = 1.0 - n_keep / max(n_valid, 1)
            result_section.setdefault(label, {})[f"tpm{th}"] = round(excluded, EXCLUDED_FRAC_DIGITS)
    return result_section


def _string_db_channel_removal():
    """Sub-routine (b): recompute STRING combined score without the database channel.

    @brief  1. load STRING info + full links tables (with float channels);
            2. keep channels whose name does NOT contain STRING_DATABASE_CHANNEL_MARKER;
            3. recompute combined_score as 1 - prod_c (1 - rescaled_channel_c);
            4. compare >=700 (int) vs >=0.7 (float) edge sets; report retained /
               dropped / fraction dropped.
    @return Dict matching the KEY_STRING schema.
    """
    info_path = os.path.join(config.STRING_DIR, STRING_INFO_BASENAME)
    links_path = os.path.join(config.STRING_DIR, STRING_LINKS_BASENAME)
    info_df = pd.read_csv(info_path, sep="\t")
    eid_to_gene = dict(zip(
        info_df[STRING_INFO_PROTEIN_ID_COL],
        info_df[STRING_INFO_GENE_COL].str.upper(),
    ))
    # Usable channel subset: everything except database / database_transferred.
    channels = [
        c for c in STRING_LINK_COLUMNS[2:-1]
        if STRING_DATABASE_CHANNEL_MARKER not in c
    ]
    dtype_map = {c: np.float32 for c in STRING_LINK_COLUMNS[2:]}
    chunk = pd.read_csv(links_path, sep=" ", usecols=STRING_LINK_COLUMNS, dtype=dtype_map)
    print(f"[info] STRING links rows loaded: {len(chunk)}")

    with_db_mask = (chunk[STRING_COMBINED_SCORE_COL] >= STRING_THRESHOLD_INT)
    combined_without = np.ones(len(chunk), dtype=np.float64)
    for c in channels:
        rescaled = np.clip(chunk[c].values / STRING_SCORE_SCALE_DIVISOR, 0.0, 1.0)
        combined_without *= (1.0 - rescaled)
    no_db_score = 1.0 - combined_without
    no_db_mask = no_db_score >= STRING_THRESHOLD_FLOAT

    n_with = int(with_db_mask.sum())
    n_without = int(no_db_mask.sum())
    n_both = int((with_db_mask & no_db_mask).sum())
    return {
        STRING_KEEP_WITH: n_with,
        STRING_KEEP_WITHOUT: n_without,
        STRING_RETAINED: n_both,
        STRING_DROPPED: n_with - n_both,
        STRING_FRAC_DROPPED: round(1.0 - n_both / max(n_with, 1), EXCLUDED_FRAC_DIGITS),
    }


def main():
    """Run both sensitivity sub-routines, serialise JSON, print summary.

    @brief  1. load graph, construct undirected PPI edge list; 2. run TPM
            sensitivity against GTEx; 3. run STRING database-channel removal
            sensitivity; 4. combine result dict and write + print.
    @return None.
    """
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    n_ppi_edges, edge_symbols = _construct_ppi_undirected(graph)
    print(f"[info] PPI unique undirected edges: {n_ppi_edges}")

    gtex_df = load_gtex()
    res = {KEY_TPM: _tissue_tpm_sensitivity(edge_symbols, gtex_df)}
    print("[info] TPM threshold sensitivity complete")

    res[KEY_STRING] = _string_db_channel_removal()
    print(f"[info] STRING DB-channel removal result: {res[KEY_STRING]}")

    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
