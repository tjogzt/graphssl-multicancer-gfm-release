#!/usr/bin/env python3
"""
@file    124_eval_probe_regimes.py
@brief   Review item C1 (2026-09-18): does the choice of inference-time
         preprocessing change the arm ranking? Re-scores every headline arm
         and reference condition under BOTH regimes on the exact same fixed
         PPI-positive / negative pair set:
           - regime A (raw)   : L2-normalise encoder output, cosine probe
           - regime B (z-score): standardise each dim then L2-normalise, probe
         The comparison is therefore unconfounded by any downstream change.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  results/ablation_real/*.pt + data/processed/xena_graph_esm.pt
Output: results/probe_regimes.json + stdout comparison table
Run:    GFM_PROJECT_ROOT=<repo> python scripts/124_eval_probe_regimes.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Artifact basenames resolved against the config paths.
GRAPH_BASENAME = "xena_graph_esm.pt"
ESM_EMB_BASENAME = "esm2_gene_embeddings.pt"

# HGT architecture (must match every checkpoint this script loads).
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Ablation arms (stable ordering for the printed table; the *legacy* CSP+NAMR
# path is used because that is the checkpoint stored against the baseline
# ablation_real directory used by the manuscript figures).
ABLATION_ARMS = [
    ("Full",        "full.pt"),
    ("NAMR-only",   "namr_only.pt"),
    ("CSP-only",    "csp_only.pt"),
    ("CSP+NAMR",    "csp_namr_legacy_s42.pt"),
]
ABLATION_REAL_TAG = "ablation_real"

# Reference (non-learned) conditions evaluated alongside the trained arms.
REF_ESM2_LABEL = "ESM-2 only"
REF_PCA_LABEL = "PCA input"
REF_RANDOM_HGT_LABEL = "Random-init HGT"

# Sampling protocol: PPI pairs are drawn from config.N_PPI_PAIRS using
# config.SEED so every run reuses the *exact* same pair set (critical for
# a paired comparison like review item C1).
N_PPI_PAIRS_DEFAULT = config.N_PPI_PAIRS
SEED_DEFAULT = config.SEED

# Standard / normalisation floors (prevent div-by-zero on constant columns).
STANDARDIZE_EPS = 1e-9
NORM_EPS = 1e-9

# Output artefact + result path key.
RESULT_JSON_FILENAME = "probe_regimes.json"
RESULT_JSON_INDENT = 1

# AUC display precision in the stdout table.
AUC_DISPLAY_DIGITS = 4


def _load_pretrain_module():
    """Dynamically load the 40_pretrain helper script as a module.

    @brief  Internal helper: importlib-based loader that exposes the
            ``build_data_dict`` and ``HGTEncoder`` symbols used throughout.
    @return Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf_mod)
    return pf_mod


pf = _load_pretrain_module()


def pairs(graph, n=N_PPI_PAIRS_DEFAULT, seed=SEED_DEFAULT):
    """Sample a matched, fixed set of positive PPI pairs + known-negatives.

    @brief  Sample ``n`` unique positive edges from the gene--gene PPI triple
            without replacement, then sample an equal number of random
            gene--gene pairs that are NOT present in the known PPI graph
            (pair-direction canonicalised under min/max so undirected
            semantics are respected).
    @param  graph   HeteroData-style graph dict loaded from disk.
    @param  n       Number of positive pairs to sample.
    @param  seed    NumPy random seed used for reproducibility.
    @return Tuple ``(pos, neg)`` where each is a list of ``(gene_a, gene_b)``
            int tuples (len(neg) == len(pos) by construction).
    """
    ppi = graph[("gene", "ppi", "gene")]["edge_index"]
    edges = [(int(a), int(b)) for a, b in zip(ppi[0].tolist(), ppi[1].tolist())]
    rng = np.random.RandomState(seed)
    pos = [edges[i] for i in rng.choice(len(edges), min(n, len(edges)), replace=False)]
    known = {(min(a, b), max(a, b)) for a, b in edges}
    n_genes = graph["meta"]["n_genes"]
    neg = []
    while len(neg) < len(pos):
        a, b = int(rng.randint(n_genes)), int(rng.randint(n_genes))
        if a != b and (min(a, b), max(a, b)) not in known:
            neg.append((a, b))
    return pos, neg


def auc(X, pos, neg, standardize):
    """Compute known-edge cosine-probe AUC under one preprocessing regime.

    @brief  (1) optionally z-score each column (regime B); (2) L2-normalise
            every row; (3) compute cosine similarity for each pos/neg pair;
            (4) return ``roc_auc_score`` against the 0/1 label vector.
    @param  X             Gene embedding matrix, shape ``[N_genes, D]``.
    @param  pos           Positive gene-gene pair list (int tuples).
    @param  neg           Negative gene-gene pair list (int tuples).
    @param  standardize   Boolean: True == regime B, False == regime A.
    @return AUROC float (single scalar).
    """
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        # z-score per dimension (regime B): expose column-wise structure.
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + STANDARDIZE_EPS)
    # L2-normalise rows (cosine probe) -- applies to both regimes.
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + NORM_EPS)
    scores_pos = np.array([float(X[a] @ X[b]) for a, b in pos], dtype=np.float64)
    scores_neg = np.array([float(X[a] @ X[b]) for a, b in neg], dtype=np.float64)
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, scores))


def encoder_embeddings(ckpt_path, graph):
    """Run the HGT encoder (with or without pretrained weights) on CPU.

    @brief  Build the HGT input dictionaries, instantiate the encoder (matching
            arch), load weights if a checkpoint path is provided, switch to
            eval(), and return the gene-node embeddings as a numpy array.
    @param  ckpt_path   Absolute checkpoint path; pass ``None`` to evaluate a
                        freshly-initialised (random-weight) HGT reference.
    @param  graph       Loaded heterograph dict.
    @return Numpy float array of shape ``[N_genes, config.EMBEDDING_DIM]``.
    """
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, torch.device("cpu"))
    x = {nt: data[nt]["x"] for nt in node_types}
    ei = {et: data[et]["edge_index"] for et in edge_types}
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        config.EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(torch.device("cpu"))
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        return model(x, ei)["gene"].cpu().numpy()


def main():
    """Run the regime-A vs regime-B comparison and emit the report table/JSON.

    @brief  1. Load graph + pair set; 2. evaluate every trained arm under both
            regimes; 3. add the three reference conditions (ESM-2 raw, PCA raw,
            random-init HGT raw); 4. compute arm rankings under each regime;
            5. print table and write probe_regimes.json.
    @return Process exit code (0 on success; the script never aborts mid-way).
    """
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    esm_path = os.path.join(config.DATA_PROC, ESM_EMB_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    gene_list = list(graph["meta"]["gene_list"])
    pos, neg = pairs(graph)

    rows = []
    for name, ckpt_basename in ABLATION_ARMS:
        ckpt_path = os.path.join(config.RESULT_DIR, ABLATION_REAL_TAG, ckpt_basename)
        X = encoder_embeddings(ckpt_path, graph)
        a_val = auc(X, pos, neg, False)
        b_val = auc(X, pos, neg, True)
        rows.append((name, a_val, b_val))

    # --- Reference conditions --------------------------------------------------
    esm_payload = torch.load(esm_path, map_location="cpu", weights_only=False)
    X_esm = np.stack([esm_payload["embeddings"][g] for g in gene_list]).astype(np.float64)
    rows.append((REF_ESM2_LABEL, auc(X_esm, pos, neg, False), auc(X_esm, pos, neg, True)))
    X_pca = graph["gene"]["x"].float().numpy()
    rows.append((REF_PCA_LABEL, auc(X_pca, pos, neg, False), auc(X_pca, pos, neg, True)))
    X_rnd = encoder_embeddings(None, graph)
    rows.append((REF_RANDOM_HGT_LABEL, auc(X_rnd, pos, neg, False), auc(X_rnd, pos, neg, True)))

    # --- Rankings (higher AUC = rank 1) ---------------------------------------
    rank_map_a = {
        name: i + 1 for i, (name, a, b) in enumerate(sorted(rows, key=lambda r: -r[1]))
    }
    rank_map_b = {
        name: i + 1 for i, (name, a, b) in enumerate(sorted(rows, key=lambda r: -r[2]))
    }

    # --- English-only stdout ---------------------------------------------------
    print(f"{'arm':<18}{'A raw output':>14}{'B z-scored':>13}{'rank A':>8}{'rank B':>8}")
    out_payload_rows = []
    for name, a_val, b_val in rows:
        print(
            f"{name:<18}"
            f"{a_val:>{14}.{AUC_DISPLAY_DIGITS}f}"
            f"{b_val:>{13}.{AUC_DISPLAY_DIGITS}f}"
            f"{rank_map_a[name]:>8}{rank_map_b[name]:>8}"
        )
        out_payload_rows.append({
            "arm": name,
            "auc_raw_output": a_val,
            "auc_zscored_output": b_val,
            "rank_raw": rank_map_a[name],
            "rank_zscored": rank_map_b[name],
        })
    ranking_identical = all(
        r["rank_raw"] == r["rank_zscored"] for r in out_payload_rows
    )
    print(f"\nranking identical under both regimes: {ranking_identical}")
    # --- JSON artefact ---------------------------------------------------------
    json_path = os.path.join(config.RESULT_DIR, RESULT_JSON_FILENAME)
    with open(json_path, "w") as fh:
        json.dump(
            {
                "pairs": len(pos),
                "rows": out_payload_rows,
                "ranking_identical": ranking_identical,
            },
            fh,
            indent=RESULT_JSON_INDENT,
        )
    print(f"[written] results/{RESULT_JSON_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
