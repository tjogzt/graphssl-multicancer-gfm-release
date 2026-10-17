#!/usr/bin/env python3
"""
@file    120_patient_setfingerprint.py
@brief   Set-fingerprint demonstration on real patient data: classify PAAD vs SKCM
         patients based ONLY on the identity of the top-K expressed gene set per
         patient (no expression magnitude is used). Accuracy above chance (50%)
         therefore reflects the set-fingerprint mechanism rather than raw expression
         levels. Compares random 256-dim gene vectors against ESM-2 gene vectors.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  TCGA expression TSVs under config.DATA_RAW/tcga/expression/TCGA-{PAAD,SKCM}/...
        and config.DATA_PROC/esm2_gene_embeddings.pt.
Output: Stdout only: few-shot classification accuracy per gene-vector source.
Run:    GFM_PROJECT_ROOT=<repo> python scripts/120_patient_setfingerprint.py
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
ESM_FILE = "esm2_gene_embeddings.pt"  # ESM-2 mean-pooled gene embeddings

# Protocol settings (kept as named constants; values unchanged).
TOP_K = 100      # Top expressed genes per patient used as the set fingerprint
K_SHOT = 5       # Few-shot support size per class
N_REPEATS = 30   # Few-shot repeats averaged per accuracy

# TCGA expression root under raw data.
GENE_TSV_ROOT = os.path.join(config.DATA_RAW, "tcga", "expression")
# Directory fragment that completes the per-cohort STAR files path.
COHORT_PATH_FRAGMENT = os.path.join("Transcriptome_Profiling", "Gene_Expression_Quantification")
# Glob pattern for per-sample STAR TSVs inside the cohort directory.
PER_SAMPLE_GLOB = os.path.join("*", "*.tsv")

# Two cohorts compared for the set-fingerprint demo.
COHORT_A = "PAAD"
COHORT_B = "SKCM"
N_CLASSES = 2
CHANCE_ACCURACY = 1.0 / N_CLASSES

# Random control vector normalisation epsilon (avoid div-by-zero on zero-norm rows).
NORM_EPS = 1e-9

# Logistic regression parameters for the few-shot probe.
LR_C = 1.0
LR_MAX_ITER = 1000

# Column names inside the STAR gene-count TSV.
STAR_GENE_NAME_COL = "gene_name"
STAR_TPM_COL = "tpm_unstranded"

# STAR TSV comment prefix character.
TSV_COMMENT_CHAR = "#"

# Few-shot accuracy print precision.
PRINT_PRECISION = 3

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def load_cohort(ct):
    """
    @brief   Load per-sample TPM values from TCGA transcriptome-profiling TSVs and
             combine them into a single gene x sample DataFrame.
    @param   ct  str – TCGA cohort code, e.g. "PAAD", "SKCM".
    @return  pd.DataFrame with genes as index, sample ids as columns; values are
             TPM floats. Duplicate gene symbols are resolved by keeping the first
             occurrence after per-gene max aggregation.
    """
    cohort_root = os.path.join(GENE_TSV_ROOT, f"TCGA-{ct}", COHORT_PATH_FRAGMENT)
    files = sorted(glob.glob(os.path.join(cohort_root, PER_SAMPLE_GLOB)))
    mats = {}
    for f in files:
        # Sample id is the TSV basename before the first dot.
        sid = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f, sep="\t", comment=TSV_COMMENT_CHAR, low_memory=False)
        # Drop rows without a gene symbol annotation.
        df = df[df[STAR_GENE_NAME_COL].notna()]
        # Take max TPM per gene name in case a gene appears on multiple rows.
        mats[sid] = df.groupby(STAR_GENE_NAME_COL)[STAR_TPM_COL].max()
    expr = pd.DataFrame(mats)
    # Drop any duplicate gene indices that remain after per-sample collapse.
    return expr[~expr.index.duplicated(keep="first")]


def topk_feats(expr, emb, top_k=TOP_K):
    """
    @brief   Build a patient feature matrix by averaging the gene embeddings of the
             top-K expressed genes per patient. Crucially, the mean is uniform
             over the selected set; no TPM magnitude weighting is applied, so
             only the IDENTITY of the expressed gene set influences the feature.
    @param   expr    pd.DataFrame – genes x patients TPM matrix.
    @param   emb     dict – {gene_symbol: np.ndarray (D,)}, gene embedding lookup.
    @param   top_k   int – number of highest-expressed genes per patient.
    @return  np.ndarray of shape (n_patients, D) with dtype float.
    """
    genes = list(expr.index)
    E = np.stack([emb[g] for g in genes]).astype(np.float64)  # Shape: genes x D
    feats = []
    for col in expr.columns:
        tpm = expr[col].values.astype(np.float64)
        top = np.argsort(-tpm)[:top_k]  # Highest-expression genes form the set
        feats.append(E[top].mean(axis=0))
    return np.array(feats)


def few_shot_acc(X, y, k_shot=K_SHOT, n_repeats=N_REPEATS, seed=config.SEED):
    """
    @brief   Mean and standard deviation of few-shot classification accuracy:
             for each repeat, draw k_shot random support samples per class, train
             a standardised logistic regression on the support only, and evaluate
             on the held-out remainder.
    @param   X          np.ndarray – (n_patients, D) patient feature matrix.
    @param   y          np.ndarray – (n_patients,) class labels.
    @param   k_shot     int – number of support samples per class.
    @param   n_repeats  int – number of independent random draws.
    @param   seed       int – master random seed.
    @return  Tuple (mean_acc, std_acc) as numpy floats.
    @note    The StandardScaler is fit on the support only to avoid test-set leakage.
    """
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(n_repeats):
        tr, te = [], []
        for c in np.unique(y):
            cidx = np.where(y == c)[0]
            rng.shuffle(cidx)
            tr.extend(cidx[:k_shot])
            te.extend(cidx[k_shot:])
        # Fit the scaler on the support only to avoid leakage into the test set.
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C).fit(
            sc.transform(X[tr]), y[tr]
        )
        accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    return np.mean(accs), np.std(accs)


def main():
    """
    @brief   Entry point: load PAAD and SKCM expression cohorts, build two
             competing gene embedding dictionaries (random vs ESM-2), compute
             set-fingerprint patient features, and print the few-shot accuracy.
    @return  None
    """
    paad = load_cohort(COHORT_A)
    skcm = load_cohort(COHORT_B)

    # Restrict to genes observed in both cohorts so the set is comparable.
    common = paad.index.intersection(skcm.index)
    expr = paad.loc[common].join(skcm.loc[common])
    y = np.array([COHORT_A] * paad.shape[1] + [COHORT_B] * skcm.shape[1])
    print(f"patients: {COHORT_A} {paad.shape[1]} + {COHORT_B} {skcm.shape[1]} "
          f"| genes {len(common)}")

    rng = np.random.RandomState(config.SEED)
    # Random control: independent gaussian vectors normalised to unit norm.
    Erand = {
        g: rng.randn(config.EMBEDDING_DIM).astype(np.float32) for g in common
    }
    Erand = {g: v / (np.linalg.norm(v) + NORM_EPS) for g, v in Erand.items()}

    # ESM-2 embeddings: reuse the standard pre-computed dictionary.
    esm = torch.load(os.path.join(config.DATA_PROC, ESM_FILE),
                     map_location=LOAD_MAP_LOCATION,
                     weights_only=TORCH_LOAD_WEIGHTS_ONLY)["embeddings"]
    Eesm = {g: np.asarray(esm[g], dtype=np.float64) for g in common if g in esm}

    for name, emb in [
        ("random-vectors (set only)", Erand),
        ("ESM-2 vectors (set only)", Eesm),
    ]:
        sub = [g for g in expr.index if g in emb]
        X = topk_feats(expr.loc[sub], emb)
        acc, sd = few_shot_acc(X, y)
        print(f"{name}: {K_SHOT}-shot acc = {acc:.{PRINT_PRECISION}f} "
              f"+/- {sd:.{PRINT_PRECISION}f} "
              f"({N_CLASSES} classes; chance {CHANCE_ACCURACY:.1f})")


if __name__ == "__main__":
    main()
