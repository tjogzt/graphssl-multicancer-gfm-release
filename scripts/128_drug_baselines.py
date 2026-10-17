#!/usr/bin/env python3
"""
@file    128_drug_baselines.py
@brief   Review item 12: three low-cost drug-response baselines under the EXACT
         same few-shot Ridge protocol used by 64_drug_response_gdsc.py:
           (a) DummyRegressor (train-set mean of LN_IC50);
           (b) drug identity one-hot;
           (c) raw target-set multi-hot (binary gene-membership vector).
         Reported at K in {5, 20, 50, 100, 200}, 30 paired random splits, seed 42.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt (gene list) + <DATA_RAW>/gdsc/GDSC2 CSV
Output: stdout table + results/control/drug_baselines.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/128_drug_baselines.py
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

import config

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_SCRIPT_BASENAME = "64_drug_response_gdsc.py"

# Artifact paths.
GRAPH_BASENAME = "xena_graph_esm.pt"
GDSC_SUBDIR = "gdsc"
GDSC_CSV = "GDSC2_fitted_dose_response_24Jul22.csv"

# Few-shot protocol: 30 independent splits per K, seed 42.
SPLIT_SEED = 42
NUM_SPLITS = 30
K_VALUES = (5, 20, 50, 100, 200)
MAX_TEST_SET_SIZE = 20000        # cap test-set size to keep the sweep tractable

# Ridge configuration (mirrors 64_drug_response_gdsc.py protocol exactly).
RIDGE_ALPHA = 1.0
RIDGE_SOLVER = "lsqr"

# Baseline section stable keys.
KEY_DUMMY = "dummy_regressor"
KEY_ONEHOT = "drug_onehot"
KEY_MULTIHOT = "target_multihot"

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "drug_baselines.json"
JSON_INDENT = 2
R2_ROUND_DIGITS = 4

# GDSC2 CSV required columns.
COL_DRUG_NAME = "DRUG_NAME"
COL_LN_IC50 = "LN_IC50"
COL_PUTATIVE_TARGET = "PUTATIVE_TARGET"


def _load_drug_module():
    """Dynamically import 64_drug_response_gdsc.py to reuse parse_targets().

    @brief  Internal helper: keeps the target-parsing logic byte-identical to
            the main GDSC pipeline rather than re-implementing it.
    @return Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "d64",
        os.path.join(SCRIPT_DIR, DRUG_SCRIPT_BASENAME),
    )
    d64 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d64)
    return d64


def few_shot_ridge_X(X, y, K,
                     n_splits=NUM_SPLITS,
                     seed=SPLIT_SEED,
                     alpha=RIDGE_ALPHA,
                     max_test=MAX_TEST_SET_SIZE):
    """Few-shot Ridge R^2 evaluation used by baselines (b) and (c).

    @brief  Run ``n_splits`` paired train/test draws: K rows drawn uniformly at
            random without replacement form the train split; the remainder form
            the test split (capped at ``max_test`` rows for tractability on the
            largest GDSC drugs). Fits a Ridge with the same solver/alpha used by
            the full GDSC pipeline and reports a mean R^2 over splits (0 if a
            split is degenerate).
    @param  X          2-D array of feature vectors (per sample).
    @param  y          1-D regression targets (LN_IC50 values).
    @param  K          Number of train samples drawn per class (not per drug: the
                       protocol draws K drug-response records, stratified on
                       nothing -- baseline (b) and (c) are feature-only, not
                       few-shot per-drug).
    @param  n_splits   Number of independent random draws.
    @param  seed       NumPy RandomState seed used across all splits (stabilises
                       the baseline comparison with the GFM checkpoints).
    @param  alpha      Ridge L2 penalty.
    @param  max_test   Maximum test rows per split (larger tests are sub-sampled
                       uniformly without replacement; this only affects speed).
    @return Plain Python float: the arithmetic mean of split-level R^2 values.
    @note   R^2 is clamped to 0 for degenerate splits (ss_tot == 0) so a single
            pathological draw does not collapse the overall mean to NaN.
    """
    rng = np.random.RandomState(seed)
    r2s = []
    n = len(y)
    for _ in range(n_splits):
        idx = rng.choice(n, min(K, n), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        if mask.sum() < 2 or (~mask).sum() < 2:
            continue
        te_idx = np.where(~mask)[0]
        if len(te_idx) > max_test:
            te_idx = rng.choice(te_idx, max_test, replace=False)
        Xtr, Xte = X[mask], X[te_idx]
        ytr, yte = y[mask], y[te_idx]
        clf = Ridge(alpha=alpha, solver=RIDGE_SOLVER).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        ss_res = np.sum((yte - pred) ** 2)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))


def main():
    """Run all three baselines, write JSON, print the per-K comparison table.

    @brief  1. load graph -> gene list -> gene-to-index map; 2. load GDSC CSV
            and extract drug->targets via 64_drug_response_gdsc.parse_targets();
            3. build drug one-hot and target-set multi-hot feature matrices;
            4. evaluate the dummy baseline directly, the other two via
               few_shot_ridge_X(); 5. output.
    @return None.
    """
    d64 = _load_drug_module()
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_BASENAME),
        map_location="cpu",
        weights_only=False,
    )
    gene_list = list(graph["meta"]["gene_list"])
    gene_to_idx = {s: i for i, s in enumerate(gene_list)}

    gdsc_path = os.path.join(config.DATA_RAW, GDSC_SUBDIR, GDSC_CSV)
    df = pd.read_csv(gdsc_path, low_memory=False)
    df[COL_LN_IC50] = pd.to_numeric(df[COL_LN_IC50], errors="coerce")
    df = df.dropna(subset=[COL_LN_IC50])

    # Drug -> sorted list of gene indices (union over its unique target strings).
    drug_targets = {}
    for drug, grp in df.groupby(COL_DRUG_NAME):
        target_gene_indices = []
        for s in grp[COL_PUTATIVE_TARGET].dropna().unique():
            target_gene_indices += d64.parse_targets(s, gene_to_idx)
        # De-duplicate while preserving first-seen order (dict.fromkeys trick).
        drug_targets[drug] = list(dict.fromkeys(target_gene_indices))
    # Drop drugs whose target list could not be mapped (they cannot participate
    # in the multi-hot baseline anyway and are also dropped by the full pipeline).
    df = df[df[COL_DRUG_NAME].map(lambda d: len(drug_targets[d]) > 0)].copy()
    drugs = sorted(df[COL_DRUG_NAME].unique())

    # Flatten the CSV into (drug, LN_IC50) pairs; D maps each pair -> drug id.
    pairs = [(r[COL_DRUG_NAME], float(r[COL_LN_IC50])) for _, r in df.iterrows()]
    y = np.array([v for _, v in pairs])
    pair_drug_id = np.array([drugs.index(d) for d, _ in pairs])

    # (b) Drug-identity one-hot feature matrix.
    X_onehot = np.eye(len(drugs), dtype=np.float64)
    # (c) Target-set multi-hot per-drug feature matrix (gene-membership binary).
    X_multihot = np.zeros((len(drugs), len(gene_list)), dtype=np.float64)
    for drug_idx, d in enumerate(drugs):
        for gene_idx in drug_targets[d]:
            X_multihot[drug_idx, gene_idx] = 1.0

    # --- Evaluate each baseline across all K values --------------------------
    res = {KEY_DUMMY: {}, KEY_ONEHOT: {}, KEY_MULTIHOT: {}}
    for K in K_VALUES:
        # (a) Dummy baseline: train-set mean predictor (no fitting required).
        rng = np.random.RandomState(SPLIT_SEED)
        r2s = []
        n = len(y)
        for _ in range(NUM_SPLITS):
            idx = rng.choice(n, min(K, n), replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            te = np.where(~mask)[0]
            if len(te) > MAX_TEST_SET_SIZE:
                te = rng.choice(te, MAX_TEST_SET_SIZE, replace=False)
            pred = np.full(len(te), y[mask].mean())
            ss_res = np.sum((y[te] - pred) ** 2)
            ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        res[KEY_DUMMY][str(K)] = round(float(np.mean(r2s)), R2_ROUND_DIGITS)

        # (b) + (c): Ridge baselines on per-drug features broadcasted to pairs.
        res[KEY_ONEHOT][str(K)] = round(
            few_shot_ridge_X(X_onehot[pair_drug_id], y, K), R2_ROUND_DIGITS,
        )
        res[KEY_MULTIHOT][str(K)] = round(
            few_shot_ridge_X(X_multihot[pair_drug_id], y, K), R2_ROUND_DIGITS,
        )

    # --- Serialise + English-only stdout --------------------------------------
    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")
    print("K   | dummy(mean) | drug-onehot | target-multihot")
    for K in K_VALUES:
        print(
            f"{K:3d} | "
            f"{res[KEY_DUMMY][str(K)]:.4f} | "
            f"{res[KEY_ONEHOT][str(K)]:.4f} | "
            f"{res[KEY_MULTIHOT][str(K)]:.4f}"
        )


if __name__ == "__main__":
    main()
