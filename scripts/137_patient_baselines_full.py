#!/usr/bin/env python3
"""
@file    137_patient_baselines_full.py
@brief   Review item 12: full 21-cancer patient-level baselines (rebuilt from
         the local Xena archive because the retired GPU server holds the
         curated expression copy).  Evaluates the same three feature-only
         baselines as 131_patient_baselines_2cancer on the full 21-cancer
         patient-ESM heterograph:
           (a) raw top-gene-set multi-hot (binary top-100 membership only);
           (b) raw expression PCA (50 dims, log2(x+1) scale kept from Xena);
           (c) untrained patient graph (random-init HGT encoder on the merged
               patient-ESM graph).
         Protocol: 5-shot per-class logistic classification, 30 repeats, seed 42.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_patient_esm.pt + Xena EB++AdjustPANCAN expression gz
Output: stdout summary + results/control/patient_baselines_full.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/137_patient_baselines_full.py
"""
import gzip
import importlib.util
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Xena expression directory (fallback used only when the GFM_XENA_DIR config
# environment variable is unset). Derived from config.DATA_RAW so the path stays
# portable across checkouts instead of pointing at a developer-machine location.
DEFAULT_XENA_FALLBACK = os.path.join(config.DATA_RAW, "xena")
XENA_EXPRESSION_FILE = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
PATIENT_ESM_GRAPH = "xena_graph_patient_esm.pt"

# Baseline protocol sizes (kept consistent with 131_patient_baselines_2cancer).
TOP_K_GENES = 100
PCA_N_COMPONENTS = 50
FEWSHOT_K_SHOT = 5
FEWSHOT_N_REPEATS = 30
FEWSHOT_SEED = 42
CLF_MAX_ITER = 1000
CLF_C = 1.0

# HGT encoder config for baseline (c): a SMALL patient-graph encoder is used
# (128/3/4) because the untrained reference is meant to show structural
# inductive bias, not trained representational quality.
HGT_HIDDEN_DIM = 128
HGT_NUM_LAYERS = 3
HGT_NUM_HEADS = 4
STANDARDIZE_EPS = 1e-8

# Xena file format.
XENA_SEP = "\t"
XENA_INDEX_COL = 0
XENA_COMPRESSION = "gzip"

# Expression cleaning: EB++AdjustPANCAN is already log2(x+1) scaled; a small
# number of entries can be NaN/inf due to low-count suppression -- these are
# substituted with 0.0 (i.e. log2(1), unexpressed gene baseline) so downstream
# operations stay well-defined.
NAN_CLEAN_VALUE = 0.0

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "patient_baselines_full.json"
JSON_INDENT = 2
DISPLAY_ROUND_DIGITS = 4

# Graph meta field names.
META_PATIENT_IDS = "patient_ids"
META_PATIENT_CANCER = "patient_cancer"
META_GENE_LIST = "gene_list"
META_CANCER_LIST = "cancer_list"

# Result dict stable keys.
RKEY_N_PATIENTS = "n_patients"
RKEY_N_CLASSES = "n_classes"
RKEY_MULTIHOT = "top_gene_set_multihot"
RKEY_PCA = "expression_pca"
RKEY_UNTRAINED_HGT = "untrained_patient_graph"
ACC_FIELD = "acc"
SD_FIELD = "sd"


def _resolve_xena_expression():
    """Return the absolute path to the Xena pancan expression gzip file.

    @brief  Tries ``config.XENA_DIR`` first (the canonical config-driven path);
            if that is None, falls back to the hard-coded DEFAULT_XENA_FALLBACK
            absolute path.  The fallback is kept as a file-level constant so
            any future data move requires editing exactly one location.
    @return Absolute path string.
    """
    base = config.XENA_DIR or DEFAULT_XENA_FALLBACK
    return os.path.join(base, XENA_EXPRESSION_FILE)


def _load_pretrain_module():
    """Dynamically load the HGT pretrain helper via importlib.

    @brief  Internal helper: exposes build_data_dict and HGTEncoder used by
            baseline (c).
    @return Loaded module.
    """
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf_mod)
    return pf_mod


def few_shot_acc(X, y,
                 k_shot=FEWSHOT_K_SHOT,
                 n_repeats=FEWSHOT_N_REPEATS,
                 seed=FEWSHOT_SEED):
    """Evaluate few-shot per-class accuracy with standardised logistic regression.

    @brief  Identical protocol to 131_patient_baselines_2cancer.py: draw
            ``k_shot`` training samples per class for ``n_repeats`` independent
            splits, fit StandardScaler + LogisticRegression(C=CLF_C,
            max_iter=CLF_MAX_ITER), return mean and sample SD of test accuracy.
            Classes with fewer than k_shot + 1 samples are skipped in a split
            so degenerate (0 training OR 0 test) splits never happen.
    @param  X          2-D feature array (patients x features).
    @param  y          1-D integer class label vector.
    @param  k_shot     Train samples per class per repeat.
    @param  n_repeats  Number of independent random splits.
    @param  seed       NumPy RandomState seed.
    @return Tuple ``(mean_accuracy, sd_accuracy)`` plain Python floats.
    """
    rng = np.random.RandomState(seed)
    accs = []
    classes = np.unique(y)
    for _ in range(n_repeats):
        tr_idx, te_idx = [], []
        for c in classes:
            cidx = np.where(np.asarray(y) == c)[0]
            if len(cidx) < k_shot + 1:
                continue
            rng.shuffle(cidx)
            tr_idx.extend(cidx[:k_shot])
            te_idx.extend(cidx[k_shot:])
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr = scaler.transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        ytr = np.asarray(y)[tr_idx]
        yte = np.asarray(y)[te_idx]
        clf = LogisticRegression(max_iter=CLF_MAX_ITER, C=CLF_C).fit(Xtr, ytr)
        accs.append(float((clf.predict(Xte) == yte).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    """Run 3 patient baselines on the full 21-cancer cohort, write JSON + summary.

    @brief  (1) load patient-ESM heterograph to obtain (patient_ids,
            patient_cancer labels, gene_list, optional cancer_list); (2) load
            Xena EB++AdjustPANCAN expression matrix, subset columns to the
            patient_ids stored in the graph, restrict rows to graph genes
            present in the expression index, align columns to graph patient
            order, keep only patients that have expression data; (3) build
            multi-hot top-100 features and run few-shot evaluation; (4) run
            PCA on NaN-cleaned log-scale expression and run few-shot evaluation;
            (5) build a random-init small HGT encoder on the heterograph,
            extract patient embeddings under a standardized-gene forward pass,
            run few-shot evaluation; (6) write JSON with summary numbers +
            print an English comparison table.
    @return None.
    """
    pf = _load_pretrain_module()
    graph_path = os.path.join(config.DATA_PROC, PATIENT_ESM_GRAPH)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    patient_ids = list(graph["meta"][META_PATIENT_IDS])
    patient_cancer = list(graph["meta"][META_PATIENT_CANCER])
    gene_list = list(graph["meta"][META_GENE_LIST])
    cancer_list = (
        list(graph["meta"][META_CANCER_LIST])
        if META_CANCER_LIST in graph["meta"]
        else sorted(set(patient_cancer))
    )
    ci = {c: i for i, c in enumerate(cancer_list)}
    y_all = np.array([ci[c] for c in patient_cancer], dtype=np.int64)
    n_total_patients = len(patient_ids)

    # --- Load expression subset aligned to graph genes + patients. ----------
    exp_path = _resolve_xena_expression()
    expr = pd.read_csv(
        exp_path,
        sep=XENA_SEP,
        index_col=XENA_INDEX_COL,
        compression=XENA_COMPRESSION,
        low_memory=False,
    )
    expr = expr[[c for c in patient_ids if c in expr.columns]]
    common_genes = [g for g in gene_list if g in expr.index]
    expr = expr.loc[common_genes]
    expr = expr[[c for c in patient_ids if c in expr.columns]]
    kept_patient_ids = list(expr.columns)
    keep_idx = [patient_ids.index(p) for p in kept_patient_ids]
    y_kept = y_all[keep_idx]
    T = expr.values.T  # patients x genes (raw expression; EB++Adjust is log2(x+1))
    print(
        f"[info] patients: {len(kept_patient_ids)} (of {n_total_patients}) | "
        f"genes: {len(common_genes)} | classes: {len(cancer_list)}"
    )

    # --- (a) multi-hot of top-100 gene set -----------------------------------
    top_k_order = np.argsort(-T, axis=1)[:, :TOP_K_GENES]
    n_pat, n_genes_kept = T.shape
    multihot = np.zeros((n_pat, n_genes_kept), dtype=np.float64)
    for p in range(n_pat):
        multihot[p, top_k_order[p]] = 1.0
    a_mh, s_mh = few_shot_acc(multihot, y_kept)

    # --- (b) expression PCA (50 dims, NaN-cleaned input). --------------------
    logT = np.nan_to_num(
        T, nan=NAN_CLEAN_VALUE, posinf=NAN_CLEAN_VALUE, neginf=NAN_CLEAN_VALUE,
    )
    pca = PCA(
        n_components=min(PCA_N_COMPONENTS, logT.shape[1]),
        random_state=FEWSHOT_SEED,
    ).fit(logT)
    Xpca = pca.transform(logT)
    a_pca, s_pca = few_shot_acc(Xpca, y_kept)

    # --- (c) untrained patient HGT encoder (standardised gene features). -----
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {k: data[k]["x"] for k in node_types}
    ei = {e: data[e]["edge_index"] for e in edge_types}
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(dim=0)) / (g.std(dim=0) + STANDARDIZE_EPS)
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)
    model.eval()
    with torch.no_grad():
        z = model(x_dict, ei)
        P_emb = z["patient"].cpu().numpy()
    a_unt, s_unt = few_shot_acc(P_emb, y_kept)

    # --- Serialise + English-only stdout summary. ----------------------------
    res = {
        RKEY_N_PATIENTS: int(len(kept_patient_ids)),
        RKEY_N_CLASSES:  int(len(cancer_list)),
        RKEY_MULTIHOT: {
            ACC_FIELD: round(a_mh, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(s_mh, DISPLAY_ROUND_DIGITS),
        },
        RKEY_PCA: {
            ACC_FIELD: round(a_pca, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(s_pca, DISPLAY_ROUND_DIGITS),
        },
        RKEY_UNTRAINED_HGT: {
            ACC_FIELD: round(a_unt, DISPLAY_ROUND_DIGITS),
            SD_FIELD:  round(s_unt, DISPLAY_ROUND_DIGITS),
        },
    }
    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")
    width = 28
    print(
        f"{'multi-hot (top-100 set)':<{width}}: "
        f"{a_mh:.{DISPLAY_ROUND_DIGITS}f} +/- {s_mh:.{DISPLAY_ROUND_DIGITS}f}"
    )
    print(
        f"{'expression PCA (50d)':<{width}}: "
        f"{a_pca:.{DISPLAY_ROUND_DIGITS}f} +/- {s_pca:.{DISPLAY_ROUND_DIGITS}f}"
    )
    print(
        f"{'untrained patient graph':<{width}}: "
        f"{a_unt:.{DISPLAY_ROUND_DIGITS}f} +/- {s_unt:.{DISPLAY_ROUND_DIGITS}f}"
    )


if __name__ == "__main__":
    main()
