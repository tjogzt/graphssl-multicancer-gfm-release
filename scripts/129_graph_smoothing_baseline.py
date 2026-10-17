#!/usr/bin/env python3
"""
@file    129_graph_smoothing_baseline.py
@brief   Review item 6 (second-priority): a plain graph-diffusion / smoothing
         baseline with NO trainable message passing. Answers whether simple
         feature smoothing over the symmetric-normalised PPI adjacency already
         recovers the known-edge / KEGG pathway signal that the trained HGT
         exploits.  Protocol: standardise ESM-2 PCA features, then apply K
         rounds of  X <- (1-alpha)*X + alpha * D^{-1/2} A D^{-1/2} X  with
         alpha=0.5, and evaluate (i) known-edge cosine AUC (ii) KEGG pathway-
         membership AUROC (K=20 seed genes per pathway, 30 draws).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt + data/raw/kegg_2021_human.json
Output: stdout table + results/control/graph_smoothing_baseline.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/129_graph_smoothing_baseline.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT_BASENAME = "50_eval_checkpoint.py"

# Artifact paths.
GRAPH_BASENAME = "xena_graph_esm.pt"
KEGG_JSON_RELPATH = os.path.join("data", "raw", "kegg_2021_human.json")

# Smoothing protocol.
SMOOTHING_ALPHA = 0.5                     # weight on D^{-1/2} A D^{-1/2} X term
SMOOTHING_STEPS = (0, 1, 2, 3, 5, 10)     # K values reported in the table
STANDARDIZE_EPS = 1e-8                    # per-dim std floor during standardisation

# Pathway-membership AUROC protocol (identical to 76_kegg_membership_table.py).
KEGG_K_SEEDS = 20
KEGG_N_DRAWS = 30
KEGG_SEED = 42
PATHWAY_MIN_GENES = 20
PATHWAY_MAX_GENES = 300
PATHWAY_HELD_MIN = 21                     # min number of held-out genes to evaluate
COSINE_NORM_EPS = 1e-9

# Reference values for the printed table (from the full NAMR-only run); these
# are reported alongside the smoothing sweep to contextualise the baseline.
REF_NAMR_KNOWN_EDGE = 0.858
REF_NAMR_KEGG = 0.839
REF_ESM_KNOWN_EDGE = 0.622
REF_ESM_KEGG = 0.599

# Display precision.
AUC_DISPLAY_DIGITS = 4
KEGG_ROUND_DIGITS = 4

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "graph_smoothing_baseline.json"
JSON_INDENT = 2

# Result dict stable keys.
RKEY_KNOWN_EDGE = "known_edge_auc"
RKEY_KEGG = "kegg_auc"
RKEY_REFERENCE = "reference"
REF_SUB_KEYS = ("namr_only_known_edge", "namr_only_kegg",
                "raw_esm_known_edge", "raw_esm_kegg")


def _load_eval_module():
    """Dynamically load 50_eval_checkpoint as a helper module.

    @brief  Internal helper: exposes load_pairs() (fixed PPI pos/neg pair set)
            and eval_embeddings() (cosine-known-edge AUC) used by the baseline.
    @return Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "ev",
        os.path.join(SCRIPT_DIR, EVAL_SCRIPT_BASENAME),
    )
    ev_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev_mod)
    return ev_mod


def build_norm_adjacency(gene_list, ppi_ei, n_genes):
    """Build the symmetric-normalised PPI adjacency used by the smoothing step.

    @brief  Convert the undirected gene--gene PPI edge_index into a scipy CSR
            matrix, symmetrize via ``A.maximum(A.T)``, drop self-loops, compute
            the degree-normalised form D^{-1/2} A D^{-1/2} and return as CSR.
    @param  gene_list   Gene symbol list (not used numerically; kept for API
                        parity with the graph-construction code).
    @param  ppi_ei      LongTensor of shape [2, E] with PPI edge indices.
    @param  n_genes     Number of genes (matrix dimension).
    @return scipy.sparse.csr_matrix of shape [n_genes, n_genes], symmetric
            normalised PPI adjacency.
    """
    del gene_list                                       # parameter retained for call-signature stability
    rows = ppi_ei[0].numpy()
    cols = ppi_ei[1].numpy()
    n_edges = len(rows)
    A = sp.coo_matrix(
        (np.ones(n_edges, dtype=np.float64), (rows, cols)),
        shape=(n_genes, n_genes),
    ).tocsr()
    # Symmetrize (the stored PPI is stored directed per PyG convention but
    # the semantic graph is strictly undirected).
    A = A.maximum(A.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    degree = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D = sp.diags(d_inv_sqrt)
    return (D @ A @ D).tocsr()


def kegg_auc(X, gene_list,
             k=KEGG_K_SEEDS, n_draws=KEGG_N_DRAWS, seed=KEGG_SEED):
    """KEGG pathway-membership AUROC protocol (identical to 76 / 130_kegg_per_seed).

    @brief  For every KEGG pathway with PATHWAY_MIN_GENES <= len(P) <=
            PATHWAY_MAX_GENES: draw n_draws independent seed sets of k genes,
            compute the cosine similarity between the mean-seed centroid and
            (held-out pathway members) vs (all other graph genes), then report
            the mean + SD of per-pathway AUROCs.
    @param  X          2-D array of shape ``[n_genes, D]``.
    @param  gene_list  List of gene symbols (index-aligned with rows of X).
    @param  k          Number of seed genes per pathway draw.
    @param  n_draws    Independent random draws per pathway.
    @param  seed       NumPy RandomState seed (reproducible centroid draws).
    @return Tuple ``(mean_auroc, sd_auroc)``, plain Python floats.
    """
    g2i = {g: i for i, g in enumerate(gene_list)}
    kegg = json.load(open(os.path.join(config.DATA_RAW, "kegg_2021_human.json")))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + COSINE_NORM_EPS)
    rng = np.random.RandomState(seed)
    aucs = []
    for _pw_name, pw_genes in kegg.items():
        if not (PATHWAY_MIN_GENES <= len(pw_genes) <= PATHWAY_MAX_GENES):
            continue
        pw_idx = [g2i[g] for g in pw_genes if g in g2i]
        if len(pw_idx) < PATHWAY_HELD_MIN:
            continue
        # Non-members = every graph gene NOT in this pathway.
        non_idx = [i for i in range(len(gene_list)) if i not in set(pw_idx)]
        pw_set = set(pw_idx)
        for _ in range(n_draws):
            seeds = rng.choice(pw_idx, min(k, len(pw_idx)), replace=False)
            centroid = Xn[seeds].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + COSINE_NORM_EPS)
            held_idx = [i for i in pw_idx if i not in set(seeds)]
            pos = Xn[held_idx] @ centroid
            neg = Xn[non_idx] @ centroid
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            s = np.concatenate([pos, neg])
            aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    """Run the K-step smoothing sweep, print comparison table, write JSON.

    @brief  1. load graph, standardize ESM-2 PCA features; 2. build D^{-1/2}
            A D^{-1/2} PPI adjacency; 3. for each K in SMOOTHING_STEPS, apply
            (1-alpha) X + alpha A_norm X smoothing for K steps; 4. evaluate
            known-edge AUC + KEGG AUROC at each K; 5. append reference numbers
            and serialise.
    @return None.
    """
    ev = _load_eval_module()
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    gene_list = list(graph["meta"]["gene_list"])
    n_genes = len(gene_list)
    X0 = graph["gene"]["x"].float().numpy()
    # Standardise (identical to standardized-feature training runs).
    X0 = (X0 - X0.mean(axis=0)) / (X0.std(axis=0) + STANDARDIZE_EPS)
    ppi_ei = graph[("gene", "ppi", "gene")]["edge_index"]
    A_norm = build_norm_adjacency(gene_list, ppi_ei, n_genes)
    pos, neg = ev.load_pairs(graph)       # 2000 pos / 2000 neg by config default

    res = {RKEY_KNOWN_EDGE: {}, RKEY_KEGG: {}}
    X_smooth = X0
    for K in SMOOTHING_STEPS:
        if K > 0:
            # One smoothing step per K: matches the iterative protocol used
            # in graph signal-processing baselines (no closed form used).
            for _ in range(K if len(SMOOTHING_STEPS) == len(SMOOTHING_STEPS) else 1):
                pass
    # Re-implement the step loop correctly (kept byte-identical with the
    # original script: accumulate smoothing for successive K values, applying
    # ONE step per increment).
    X_smooth = X0
    for K in SMOOTHING_STEPS:
        # Apply the *remaining* number of steps from the previous K.
        if K == 0:
            steps_this_round = 0
        else:
            prev_K = SMOOTHING_STEPS[max(0, SMOOTHING_STEPS.index(K) - 1)]
            steps_this_round = K - prev_K if K > 0 else 0
        for _ in range(steps_this_round):
            X_smooth = (1.0 - SMOOTHING_ALPHA) * X_smooth + SMOOTHING_ALPHA * (A_norm @ X_smooth)
        _, _, _, auc = ev.eval_embeddings(X_smooth, pos, neg)
        k_auc, k_sd = kegg_auc(X_smooth, gene_list)
        res[RKEY_KNOWN_EDGE][str(K)] = round(float(auc), AUC_DISPLAY_DIGITS)
        res[RKEY_KEGG][str(K)] = round(k_auc, KEGG_ROUND_DIGITS)
        print(
            f"K={K:2d}: known-edge AUC={auc:.{AUC_DISPLAY_DIGITS}f} | "
            f"KEGG AUROC={k_auc:.{AUC_DISPLAY_DIGITS}f} +/- {k_sd:.{AUC_DISPLAY_DIGITS}f}"
        )

    res[RKEY_REFERENCE] = {
        REF_SUB_KEYS[0]: REF_NAMR_KNOWN_EDGE,
        REF_SUB_KEYS[1]: REF_NAMR_KEGG,
        REF_SUB_KEYS[2]: REF_ESM_KNOWN_EDGE,
        REF_SUB_KEYS[3]: REF_ESM_KEGG,
    }
    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
