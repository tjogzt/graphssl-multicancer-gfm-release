#!/usr/bin/env python3
"""
@file    131_patient_baselines_2cancer.py
@brief   Review item 12: two-cancer patient baselines (PAAD vs SKCM; locally
         available because the full 21-cancer expression lives on the retired
         GPU server). On 182 + 104 patients (top-100 expressed genes per
         patient) reports 5-shot, 30-split logistic classification accuracy
         for three feature-only baselines plus a pair of set-fingerprint
         reference embeddings (random vs ESM-2 mean vectors):
           (a) raw top-gene-set multi-hot;
           (b) raw expression PCA (50 dims);
           (c) set-fingerprint mean embedding with random / ESM-2 vectors.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/raw/tcga/expression/TCGA-{PAAD,SKCM}/... + esm2_gene_embeddings.pt
Output: stdout summary + results/control/patient_baselines_2cancer.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/131_patient_baselines_2cancer.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
# TCGA expression path pieces (under config.DATA_RAW).
TCGA_SUBDIR = "tcga"
EXPRESSION_DIR = "expression"
TUMOUR_DIR = "Transcriptome_Profiling"
QUANT_DIR = "Gene_Expression_Quantification"
EXP_GLOB = "*.tsv"

# Cohort selection.
COHORTS = ("PAAD", "SKCM")

# Embedding artifact.
ESM_BASENAME = "esm2_gene_embeddings.pt"
ESM_EMBEDDINGS_KEY = "embeddings"

# Baseline protocol sizes.
TOP_K_GENES = 100
PCA_N_COMPONENTS = 50
RANDOM_EMB_DIM = 256

# Few-shot classification protocol.
FEWSHOT_K_SHOT = 5
FEWSHOT_N_REPEATS = 30
FEWSHOT_SEED = 42
CLF_MAX_ITER = 1000
CLF_C = 1.0

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "patient_baselines_2cancer.json"
JSON_INDENT = 2
DISPLAY_ROUND_DIGITS = 4

# TPM column name in the TCGA GCT-style tables.
COL_GENE_NAME = "gene_name"
COL_TPM = "tpm_unstranded"
TSV_COMMENT_CHAR = "#"

# Result dict stable keys.
RKEY_N_PATIENTS = "n_patients"
RKEY_N_COMMON_GENES = "n_common_genes"
RKEY_MULTIHOT = "top_gene_set_multihot"
RKEY_PCA = "expression_pca"
RKEY_SETFP = "setfingerprint_mean_embedding"
ACC_FIELD = "acc"
SD_FIELD = "sd"


def _tcga_glob(ct):
    """Return the glob pattern matching TCGA expression quantification files.

    @brief  Internal helper: construct the full path pattern under
            config.DATA_RAW / tcga/expression/TCGA-<ct>/... matching the
            directory layout used by GDC downloads.
    @param  ct   Cancer type abbreviation (PAAD, SKCM, ...).
    @return Absolute or config-relative glob string.
    """
    return os.path.join(
        config.DATA_RAW, TCGA_SUBDIR, EXPRESSION_DIR,
        f"TCGA-{ct}", TUMOUR_DIR, QUANT_DIR, "*", EXP_GLOB,
    )


def load_cohort(ct):
    """Load a single-cancer TCGA expression cohort (genes x patients DataFrame).

    @brief  For every matching quantification TSV: read with pd.read_csv, drop
            genes without a gene_name, aggregate duplicate gene names by max
            TPM (some ensembl IDs share HGNC names; taking max avoids zero rows
            from duplicated names), then combine across samples. Final index is
            de-duplicated by first occurrence for downstream join alignment.
    @param  ct   Cancer type string (PAAD / SKCM / ...).
    @return pd.DataFrame with index = gene_name, columns = sample_ids, values = TPM.
    """
    mats = {}
    for f in sorted(glob.glob(_tcga_glob(ct))):
        sample_id = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f, sep="\t", comment=TSV_COMMENT_CHAR, low_memory=False)
        df = df[df[COL_GENE_NAME].notna()]
        mats[sample_id] = df.groupby(COL_GENE_NAME)[COL_TPM].max()
    expr = pd.DataFrame(mats)
    return expr[~expr.index.duplicated(keep="first")]


def few_shot_acc(X, y,
                 k_shot=FEWSHOT_K_SHOT,
                 n_repeats=FEWSHOT_N_REPEATS,
                 seed=FEWSHOT_SEED):
    """Evaluate few-shot per-class classification accuracy (logistic + Z-score).

    @brief  Repeat n_repeats times: (a) for each unique class, shuffle and
            take k_shot training samples, rest test; (b) fit StandardScaler on
            training rows, train LogisticRegression(C=CLF_C, max_iter=CLF_MAX_ITER)
            on scaled training rows, and record mean test accuracy.  Returns
            the mean and the sample standard deviation of per-repeat accuracies.
    @param  X          2-D feature array (samples x features).
    @param  y          1-D class label vector (array-like).
    @param  k_shot     Training samples taken per class per repeat.
    @param  n_repeats  Number of independent random splits.
    @param  seed       NumPy RandomState seed.
    @return Tuple ``(mean_accuracy, sd_accuracy)`` as plain Python floats.
    """
    rng = np.random.RandomState(seed)
    accs = []
    classes = np.unique(y)
    for _ in range(n_repeats):
        tr_idx, te_idx = [], []
        for c in classes:
            cidx = np.where(np.asarray(y) == c)[0]
            rng.shuffle(cidx)
            tr_idx.extend(cidx[:k_shot])
            te_idx.extend(cidx[k_shot:])
        scaler = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(max_iter=CLF_MAX_ITER, C=CLF_C).fit(
            scaler.transform(X[tr_idx]), np.asarray(y)[tr_idx],
        )
        y_true = np.asarray(y)[te_idx]
        y_pred = clf.predict(scaler.transform(X[te_idx]))
        accs.append(float((y_pred == y_true).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    """Run all 3 baselines + 2 set-fingerprint references, write JSON + summary.

    @brief  1. load PAAD + SKCM cohorts; 2. restrict to genes shared between
            both cancers AND present in the ESM-2 embedding dict; 3. build the
            patient-by-gene raw TPM matrix and patient label vector; 4. build
            (a) multi-hot top-K features and (b) log1p PCA features and run
            few-shot evaluation for each; 5. compute set-fingerprint mean
            embedding features using ESM-2 vectors and random vectors; 6.
            aggregate results into the JSON dict and print a human summary.
    @return None.
    """
    paad = load_cohort("PAAD")
    skcm = load_cohort("SKCM")
    common = paad.index.intersection(skcm.index)
    expr_all = paad.loc[common].join(skcm.loc[common])  # genes x patients

    # Restrict to genes with ESM-2 embeddings (the set-fingerprint reference
    # needs embeddings; restricting once keeps all three baselines comparable
    # on the same gene universe).
    esm_path = os.path.join(config.DATA_PROC, ESM_BASENAME)
    esm_embeddings = torch.load(
        esm_path, map_location="cpu", weights_only=False,
    )[ESM_EMBEDDINGS_KEY]
    genes_with_emb = [g for g in expr_all.index if g in esm_embeddings]
    expr = expr_all.loc[genes_with_emb]
    genes = list(expr.index)
    y = np.array(["PAAD"] * paad.shape[1] + ["SKCM"] * skcm.shape[1])
    T = expr.values.T                              # patients x genes (raw TPM)

    # --- (a) raw top-K gene-set multi-hot -------------------------------------
    top_k_order = np.argsort(-T, axis=1)[:, :TOP_K_GENES]  # patients x TOP_K
    multihot = np.zeros((T.shape[0], len(genes)), dtype=np.float64)
    for p in range(T.shape[0]):
        multihot[p, top_k_order[p]] = 1.0
    a_mh, s_mh = few_shot_acc(multihot, y)

    # --- (b) raw expression log1p PCA ----------------------------------------
    log_T = np.log1p(T)
    pca = PCA(
        n_components=min(PCA_N_COMPONENTS, T.shape[1]),
        random_state=FEWSHOT_SEED,
    ).fit(log_T)
    X_pca = pca.transform(log_T)
    a_pca, s_pca = few_shot_acc(X_pca, y)

    # --- (c) set-fingerprint mean embedding references -----------------------
    E_esm = np.stack([esm_embeddings[g] for g in genes]).astype(np.float64)
    rng = np.random.RandomState(FEWSHOT_SEED)
    E_rand = rng.randn(len(genes), RANDOM_EMB_DIM)
    E_rand = E_rand / (np.linalg.norm(E_rand, axis=1, keepdims=True) + 1e-12)
    refs = {"esm": E_esm, "random": E_rand}
    ref_results = {}
    for name, E_mat in refs.items():
        feat = np.stack(
            [E_mat[top_k_order[p]].mean(axis=0) for p in range(T.shape[0])],
            axis=0,
        )
        acc, sd = few_shot_acc(feat, y)
        ref_results[name] = {
            ACC_FIELD: round(acc, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(sd, DISPLAY_ROUND_DIGITS),
        }

    res = {
        RKEY_N_PATIENTS: {
            "PAAD": int(paad.shape[1]),
            "SKCM": int(skcm.shape[1]),
        },
        RKEY_N_COMMON_GENES: int(len(genes)),
        RKEY_MULTIHOT: {
            ACC_FIELD: round(a_mh, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(s_mh, DISPLAY_ROUND_DIGITS),
        },
        RKEY_PCA: {
            ACC_FIELD: round(a_pca, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(s_pca, DISPLAY_ROUND_DIGITS),
        },
        RKEY_SETFP: ref_results,
    }
    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    # English-only stdout summary.
    print(f"[written] {out_path}")
    print(
        f"PAAD={res[RKEY_N_PATIENTS]['PAAD']} "
        f"SKCM={res[RKEY_N_PATIENTS]['SKCM']} "
        f"genes={res[RKEY_N_COMMON_GENES]}"
    )
    print(
        f"top-gene-set multi-hot : {a_mh:.{DISPLAY_ROUND_DIGITS}f} +/- "
        f"{s_mh:.{DISPLAY_ROUND_DIGITS}f}"
    )
    print(
        f"expression PCA ({PCA_N_COMPONENTS}d)   : {a_pca:.{DISPLAY_ROUND_DIGITS}f} +/- "
        f"{s_pca:.{DISPLAY_ROUND_DIGITS}f}"
    )
    print(
        f"set-fingerprint (ESM-2) : {ref_results['esm'][ACC_FIELD]} +/- "
        f"{ref_results['esm'][SD_FIELD]}"
    )
    print(
        f"set-fingerprint (random): {ref_results['random'][ACC_FIELD]} +/- "
        f"{ref_results['random'][SD_FIELD]}"
    )


if __name__ == "__main__":
    main()
