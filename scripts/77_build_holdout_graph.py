#!/usr/bin/env python3
"""
@file    77_build_holdout_graph.py
@brief   Build the held-out PPI graph for the strict link-prediction experiment.
         The script enforces three correctness-critical contracts:
           1. Edges are split on the CANONICAL UNDIRECTED pair {min, max}, so a
              stored edge (a->b) and its mirror (b->a) never land in different splits.
           2. The training graph is REBUILT with only the train edges (in their
              original stored orientation), so test/val edges are removed from BOTH
              message passing and the CSP positive sampler.
           3. The negative-exclusion set is the FULL known STRING edge set, so a
              sampled negative can never coincide with a held-out positive.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup functional_analysis

Input:  data/processed/xena_graph_esm.pt
Output: results/holdout/train_graph.pt, val_pairs.npy, test_pairs.npy,
        known_edges.npy (canonical pairs), split_meta.json
Run:    python scripts/77_build_holdout_graph.py [--val-frac 0.1 --test-frac 0.1]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

# ============================================================
# File-level constants (extracted from hard-coded literals)
# ============================================================

# Source graph file under config.DATA_PROC.
SRC_GRAPH_RELATIVE = "xena_graph_esm.pt"
SRC = os.path.join(config.DATA_PROC, SRC_GRAPH_RELATIVE)

# Output subdirectory under config.result_path(...).
OUTPUT_SUBDIR = "holdout"
OUTDIR = config.result_path(OUTPUT_SUBDIR)

# Default validation / test split fractions (can be overridden by CLI flags).
DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1

# Canonicalised edge dtype (int64 for PyG edge_index compatibility).
EDGE_DTYPE = np.int64

# Output file names inside OUTDIR.
TRAIN_GRAPH_FILE = "train_graph.pt"
VAL_PAIRS_FILE = "val_pairs.npy"
TEST_PAIRS_FILE = "test_pairs.npy"
KNOWN_EDGES_FILE = "known_edges.npy"
SPLIT_META_FILE = "split_meta.json"

# Output JSON indentation spaces.
JSON_INDENT = 1

# Load map location for torch.load and weights_only flag.
LOAD_MAP_LOCATION = 'cpu'
TORCH_LOAD_WEIGHTS_ONLY = False


def canonical(edge_index):
    """
    @brief   Convert a directed edge_index tensor (2 x E) into an array of UNIQUE
             canonical undirected pairs (u < v).
    @param   edge_index  torch.LongTensor – (2, E) directed edge index (which may
                         store both (u,v) and (v,u) for undirected graphs).
    @return  np.ndarray of shape (E_unique, 2), dtype int64, sorted rows, with
             rows satisfying row[0] < row[1] and no duplicates.
    """
    rows = edge_index.numpy()
    pairs = set()
    for source, target in rows.T:
        pairs.add((min(int(source), int(target)), max(int(source), int(target))))
    arr = np.array(sorted(pairs), dtype=EDGE_DTYPE)
    return arr


def main():
    """
    @brief   Entry point: parse CLI split fractions, load the full graph, split
             canonical PPI pairs into train/val/test, rebuild the training graph
             keeping only train edges (in original orientation), and persist all
             artefacts plus a machine-readable split metadata JSON.
    @return  None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                        help="Fraction of canonical pairs reserved for validation.")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                        help="Fraction of canonical pairs reserved for final test.")
    args = parser.parse_args()

    graph = torch.load(SRC, map_location=LOAD_MAP_LOCATION,
                       weights_only=TORCH_LOAD_WEIGHTS_ONLY)
    ppi = graph[("gene", "ppi", "gene")]["edge_index"]
    pairs = canonical(ppi)
    print(f"canonical undirected PPI pairs (pre-split): {len(pairs)}")

    # Deterministic random permutation of the full pair list.
    rng = np.random.RandomState(config.SEED)
    order = rng.permutation(len(pairs))

    # Split order prefix into test / val / train.
    n_test = int(len(pairs) * args.test_frac)
    n_val = int(len(pairs) * args.val_frac)
    test_idx = set(order[:n_test].tolist())
    val_idx = set(order[n_test:n_test + n_val].tolist())
    train_idx = order[n_test + n_val:]

    train_pairs = pairs[train_idx]
    val_pairs = pairs[np.array(sorted(val_idx))]
    test_pairs = pairs[np.array(sorted(test_idx))]
    print(f"split sizes: train {len(train_pairs)} | "
          f"val {len(val_pairs)} | test {len(test_pairs)}")

    # Rebuild the training graph: iterate over the original directed edge_index
    # and keep only those edges whose canonical pair is in the TRAIN set. This
    # preserves the original storage orientation (which may differ from u<v).
    train_set = {(a, b) for a, b in train_pairs}
    keep = []
    for source, target in ppi.numpy().T:
        if (min(int(source), int(target)), max(int(source), int(target))) in train_set:
            keep.append((int(source), int(target)))

    graph[("gene", "ppi", "gene")] = {
        "edge_index": torch.tensor(np.array(keep).T, dtype=torch.int64)
    }
    print(f"train-graph PPI directed edges retained: {len(keep)}")

    # Persist all artefacts.
    os.makedirs(OUTDIR, exist_ok=True)
    torch.save(graph, os.path.join(OUTDIR, TRAIN_GRAPH_FILE))
    np.save(os.path.join(OUTDIR, VAL_PAIRS_FILE), val_pairs)
    np.save(os.path.join(OUTDIR, TEST_PAIRS_FILE), test_pairs)
    np.save(os.path.join(OUTDIR, KNOWN_EDGES_FILE), pairs)  # Full set = negative-exclusion

    with open(os.path.join(OUTDIR, SPLIT_META_FILE), "w") as fh:
        json.dump(
            {
                "seed": config.SEED,
                "n_total": int(len(pairs)),
                "n_train": int(len(train_pairs)),
                "n_val": int(len(val_pairs)),
                "n_test": int(len(test_pairs)),
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
            },
            fh,
            indent=JSON_INDENT,
        )
    print(f"[written] {OUTDIR}/{TRAIN_GRAPH_FILE} + val/test/known .npy + {SPLIT_META_FILE}")


if __name__ == "__main__":
    main()
