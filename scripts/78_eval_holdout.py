#!/usr/bin/env python3
"""
@file    78_eval_holdout.py
@brief   Evaluate a trained checkpoint on the HELD-OUT PPI test edges using three
         negative-sampling schemes: (1) uniform random non-edges, (2) degree-matched
         negatives, (3) hub-restricted negatives (pool = the top-1,000 training-
         degree hubs). Negative exclusion uses the FULL known STRING edge set (not
         just train edges), so a sampled negative can never coincide with a held-out
         positive. Degree matching uses the TRAIN-graph degree (the model actually
         saw), not the full-graph degree.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup functional_analysis

Input:  results/holdout/test_pairs.npy, known_edges.npy, train_graph.pt + a checkpoint.
Output: results/holdout/eval_<tag>.json
Run:    python scripts/78_eval_holdout.py \
            --ckpt results/ablation_real/namr_only.pt --tag namr_only_heldout
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Output / input root directory (under results).
OUTPUT_SUBDIR = "holdout"
OUTDIR = config.result_path(OUTPUT_SUBDIR)

# HGT encoder architecture hyper-parameters (mirrors 40_pretrain.py defaults).
HGT_EMBEDDING_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Top-k hubs by training-graph degree used as the pool for scheme "hub".
HUB_POOL_SIZE = 1000

# Name of the 40_pretrain.py helper module (loaded dynamically).
PRETRAIN_MODULE_NAME = "pf"
PRETRAIN_FILE = "40_pretrain.py"

# Negative-sampling scheme identifiers (must match command-loop names).
SCHEME_RANDOM = "random"
SCHEME_DEGREE = "degree"
SCHEME_HUB = "hub"
ALL_SCHEMES = (SCHEME_RANDOM, SCHEME_DEGREE, SCHEME_HUB)

# Epsilon for L2 normalisation (avoid div-by-zero for zero-norm rows).
NORM_EPS = 1e-9

# Epsilon for z-score gene feature normalisation (avoid div-by-zero).
Z_SCORE_EPS = 1e-8

# Rounding decimal places for the printed per-scheme AUROC scores.
AUROC_DECIMALS = 4

# Output JSON indentation spaces.
JSON_INDENT = 1

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False

# Train-graph artifact names inside OUTDIR.
TRAIN_GRAPH_FILE = "train_graph.pt"
TEST_PAIRS_FILE = "test_pairs.npy"
KNOWN_EDGES_FILE = "known_edges.npy"


def load_pretrain_module():
    """
    @brief   Dynamically import 40_pretrain.py as a module so that the HGT encoder
             architecture and graph-builder helpers can be reused without copying.
    @return  Module namespace for 40_pretrain.py.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        PRETRAIN_MODULE_NAME, os.path.join(here, PRETRAIN_FILE)
    )
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    return pf


def embeddings_for(graph, gene_list, ckpt_path, pf):
    """
    @brief   Run the HGT encoder (loaded from `ckpt_path`) on the training graph
             and return the resulting (n_genes x d) gene embedding matrix.
    @param   graph       dict – HeteroData-style train-graph dictionary.
    @param   gene_list   list – ordered gene symbols (kept for call-signature
                         parity with sibling scripts).
    @param   ckpt_path   str – filesystem path of the trained checkpoint.
    @param   pf          module – imported 40_pretrain.py namespace.
    @return  np.ndarray of shape (n_genes, HGT_EMBEDDING_DIM) with dtype float.
    """
    dev = torch.device(LOAD_MAP_LOCATION)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # z-score gene input features (consistent with the training protocol).
    g = x_dict["gene"]
    x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + Z_SCORE_EPS)

    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)
    ck = torch.load(ckpt_path, map_location=LOAD_MAP_LOCATION,
                    weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy()


def build_degree_buckets(deg):
    """
    @brief   Bucket gene nodes by floor(log2(degree + 1)) for the degree-matched
             negative-sampling scheme.
    @param   deg  np.ndarray – (n_genes,) training-graph degree (int counts).
    @return  dict {bucket_id: np.ndarray of gene indices in that bucket}.
    """
    logdeg = np.log2(deg + 1.0)
    bins = np.floor(logdeg).astype(int)
    bucket_lists = {}
    for b in np.unique(bins):
        bucket_lists[b] = np.where(bins == b)[0]
    return bins, bucket_lists


def sample_negatives(scheme, rng, n_genes, test_pairs, known_set, test_set,
                     hub_pool, bins, bucket_lists):
    """
    @brief   Vectorised negative sampling that returns exactly N = len(test_pairs)
             negative pairs, rejecting self-loops and all known / test positives.
    @param   scheme        str – one of ALL_SCHEMES.
    @param   rng           np.random.RandomState – seeded RNG for sampling.
    @param   n_genes       int – total gene node count.
    @param   test_pairs    np.ndarray (N x 2) – positive test pairs.
    @param   known_set     set of (u, v) canonical pairs – full known edge set.
    @param   test_set      set of (u, v) canonical pairs – test positives only.
    @param   hub_pool      np.ndarray – indices of the top hub genes.
    @param   bins          np.ndarray (n_genes,) – per-node degree bucket id.
    @param   bucket_lists  dict – degree-bucket index lookup (see build_degree_buckets).
    @return  np.ndarray of shape (N, 2), dtype int64, negative candidate pairs.
    """
    N = len(test_pairs)
    out = np.empty((0, 2), dtype=np.int64)
    while len(out) < N:
        if scheme == SCHEME_RANDOM:
            cand = np.stack(
                [rng.randint(0, n_genes, N), rng.randint(0, n_genes, N)], axis=1
            )
        elif scheme == SCHEME_DEGREE:
            # For each left-side test node, sample a partner from the same degree bucket.
            u = test_pairs[:, 0]
            cand_b = np.empty(N, dtype=np.int64)
            for i, node in enumerate(u):
                b = bins[node]
                pool = bucket_lists.get(b, bucket_lists.get(int(bins[node])))
                if pool is None or len(pool) == 0:
                    # No valid pool: fall back to the source node (will be filtered out
                    # by the no-self-loop guard and the outer loop will resample).
                    cand_b[i] = node
                else:
                    cand_b[i] = pool[rng.randint(0, len(pool))]
            cand = np.stack([u, cand_b], axis=1)
        elif scheme == SCHEME_HUB:
            cand = np.stack(
                [hub_pool[rng.randint(0, len(hub_pool), N)],
                 hub_pool[rng.randint(0, len(hub_pool), N)]],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown sampling scheme: {scheme}")

        # Reject any candidate that is a self-loop, a known edge, or a test positive.
        keep = np.array(
            [
                a != b
                and (min(a, b), max(a, b)) not in known_set
                and (min(a, b), max(a, b)) not in test_set
                for a, b in cand
            ],
            dtype=bool,
        )
        out = np.vstack([out, cand[keep]])
    return out[:N]


def main():
    """
    @brief   Entry point: parse CLI, load train graph / splits / checkpoint,
             compute HGT gene embeddings, evaluate ROC-AUC against all three
             negative schemes, and write the JSON report.
    @return  None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Filesystem path of the trained checkpoint to evaluate.")
    parser.add_argument("--tag", required=True,
                        help="Short identifier string used to name the output JSON file.")
    args = parser.parse_args()

    pf = load_pretrain_module()

    # Load train graph and precomputed split artefacts.
    train_graph = torch.load(os.path.join(OUTDIR, TRAIN_GRAPH_FILE),
                             map_location=LOAD_MAP_LOCATION,
                             weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    gene_list = list(train_graph["meta"]["gene_list"])
    known = set(map(tuple, np.load(os.path.join(OUTDIR, KNOWN_EDGES_FILE))))
    test = np.load(os.path.join(OUTDIR, TEST_PAIRS_FILE))
    n_genes = len(gene_list)

    # Compute and L2-normalise the HGT gene embeddings (dot-product = cosine sim).
    X = embeddings_for(train_graph, gene_list, args.ckpt, pf)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + NORM_EPS)

    # Train-graph degree statistics (used for degree-matching and hub pool).
    ppi = train_graph[("gene", "ppi", "gene")]["edge_index"].numpy()
    deg = np.bincount(ppi.reshape(-1), minlength=n_genes)
    hub_pool = np.argsort(-deg)[:HUB_POOL_SIZE]  # Top-K by training-graph degree.
    test_set = set(map(tuple, test))

    # Build degree buckets once: reused across repeats.
    bins, bucket_lists = build_degree_buckets(deg)

    # Positive-pair cosine scores (dot product on L2-normalised embeddings).
    pos_scores = np.array([float(Xn[a] @ Xn[b]) for a, b in test])

    results = {}
    for scheme in ALL_SCHEMES:
        rng = np.random.RandomState(config.SEED)
        neg = sample_negatives(
            scheme, rng, n_genes, test, known, test_set,
            hub_pool, bins, bucket_lists,
        )
        neg_scores = np.array([float(Xn[a] @ Xn[b]) for a, b in neg])
        y = np.concatenate([np.ones(len(test)), np.zeros(len(neg))])
        s = np.concatenate([pos_scores, neg_scores])
        results[scheme] = round(float(roc_auc_score(y, s)), AUROC_DECIMALS)

    results["n_test"] = int(len(test))

    out_json_path = os.path.join(OUTDIR, f"eval_{args.tag}.json")
    with open(out_json_path, "w") as fh:
        json.dump(results, fh, indent=JSON_INDENT)
    print(f"{args.tag}: {results}")
    print(f"[written] {out_json_path}")


if __name__ == "__main__":
    main()
