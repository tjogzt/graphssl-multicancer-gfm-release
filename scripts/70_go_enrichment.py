#!/usr/bin/env python3
"""
@file    70_go_enrichment.py
@brief   GO enrichment validation of the learned gene clusters. Clusters the
         NAMR-only encoder embeddings with K-means and runs Enrichr
         GO_Biological_Process_2023 on each cluster (using the full gene list
         as the background universe) to test whether the learned clusters are
         functionally coherent rather than arbitrary graph partitions.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  <ckpt_path> (positional arg; default config.result_path("ablation_real", CKPT_FILE))
        graph artifact config.DATA_PROC/xena_graph_esm.pt (gene features + gene list)
Output: config.result_path(OUT_FILE) JSON plus cluster/enrichment progress on stdout
Run:    python scripts/70_go_enrichment.py [ckpt_path]
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import gseapy as gp

# ============================================================================
# Configuration Constants (artifact names and protocol thresholds)
# ============================================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
GRAPH_FILE = "xena_graph_esm.pt"
CKPT_FILE = "namr_only.pt"
OUT_FILE = "go_enrichment_namr_only.json"

# Protocol thresholds (kept as named constants; values unchanged).
MIN_GENES = 15      # Skip clusters with fewer than this many genes.
TOP_TERMS = 5       # Significant GO terms recorded per cluster.
SIG_P = 0.05        # Adjusted p-value cutoff for significance.

# HGT encoder architecture parameters (must match checkpoint).
HGT_EMBEDDING_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Feature standardization epsilon.
STANDARDIZATION_EPS = 1e-8

# K-means clustering hyperparameters.
KMEANS_N_INIT = 10  # Number of random restarts for stable solution.

# Enrichr API parameters.
ENRICHR_GENE_SET = "GO_Biological_Process_2023"
ENRICHR_ORGANISM = "human"
ENRICHR_RESULT_COL_TERM = "Term"
ENRICHR_RESULT_COL_ADJP = "Adjusted P-value"
ENRICHR_RESULT_COL_GENES = "Genes"

# Error message truncation length (for per-cluster error field).
ERROR_TRUNCATE_LEN = 120

# Enrichr request throttling (seconds between clusters) to stay within rate limits.
ENRICHR_THROTTLE_SEC = 0.5

# JSON output indentation.
JSON_INDENT = 1

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Core Functions
# ============================================================================

def get_embeddings(ckpt_path, graph_path=None, standardize=True):
    """
    @brief   Load the graph, run the NAMR-only encoder, and return gene
             embeddings plus the graph object.

    @param   ckpt_path     Path to the pretrained NAMR-only checkpoint.
    @param   graph_path    Path to processed graph file; uses default under DATA_PROC if None.
    @param   standardize   Whether to re-standardize gene input features per dimension.
    @return  Tuple of (embeddings_array, graph_object):
             - embeddings_array: [n_genes x embedding_dim]
             - graph_object: loaded torch graph with metadata
    @note    Imports pretraining module dynamically to avoid circular imports.
    """
    # Resolve default graph path if not explicitly provided.
    graph_path = graph_path or os.path.join(config.DATA_PROC, GRAPH_FILE)

    import importlib.util
    # Dynamically load pretraining module (HGTEncoder / build_data_dict).
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    # Load graph onto CPU for inference.
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")

    # Build heterogeneous graph input dictionaries.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Z-score gene features per column so clustering is not dominated by feature scale.
    if standardize and "gene" in x_dict:
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZATION_EPS)

    # Initialize HGT encoder with checkpoint-matching architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(dev)

    # Load checkpoint weights and set evaluation mode.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # Forward pass without gradient computation.
    with torch.no_grad():
        return model(x_dict, ei_dict)["gene"].cpu().numpy(), graph


def main():
    """
    @brief   Main entry point: cluster embeddings, run GO enrichment per cluster,
             and save results to JSON.
    @return  None (outputs progress to stdout and saves JSON file).
    @note    Uses Enrichr for GO Biological Process enrichment with full gene list
             as background universe; throttles requests to respect rate limits.
    """
    # Resolve checkpoint path from CLI argument or use default ablation checkpoint.
    ckpt = sys.argv[1] if len(sys.argv) > 1 else config.result_path("ablation_real", CKPT_FILE)
    emb, graph = get_embeddings(ckpt)
    gl = list(graph["meta"]["gene_list"])

    # Standardize embeddings and replace NaN (constant columns poison K-means otherwise).
    X = StandardScaler().fit_transform(emb.astype(np.float64))
    X = np.nan_to_num(X)

    # K = number of cancer types; multiple restarts average for stable clustering.
    lab = KMeans(
        n_clusters=config.CLUSTER_K, n_init=KMEANS_N_INIT,
        random_state=config.SEED
    ).fit_predict(X)
    sizes = Counter(lab.tolist())
    print("cluster sizes:", dict(sorted(sizes.items())))

    # Run GO enrichment for each cluster.
    results = {}
    for c in range(config.CLUSTER_K):
        # Map cluster indices back to gene names via graph gene list order.
        genes = [gl[i] for i in np.where(lab == c)[0]]

        # Skip clusters that are too small for meaningful enrichment testing.
        if len(genes) < MIN_GENES:
            results[c] = {"n": len(genes), "terms": []}
            continue

        try:
            # Background = full gene list so the test is against the true universe.
            enr = gp.enrichr(
                gene_list=genes, gene_sets=ENRICHR_GENE_SET,
                organism=ENRICHR_ORGANISM, background=gl, outdir=None
            )
            df = enr.results

            # Filter significant terms and sort by adjusted p-value (ascending).
            sig = df[df[ENRICHR_RESULT_COL_ADJP] < SIG_P].sort_values(ENRICHR_RESULT_COL_ADJP)
            terms = [
                {
                    "term": r[ENRICHR_RESULT_COL_TERM],
                    "adjp": float(r[ENRICHR_RESULT_COL_ADJP]),
                    "genes": r[ENRICHR_RESULT_COL_GENES],
                }
                for _, r in sig.head(TOP_TERMS).iterrows()
            ]
            results[c] = {
                "n": len(genes),
                "n_sig": int((df[ENRICHR_RESULT_COL_ADJP] < SIG_P).sum()),
                "terms": terms,
            }
            print(f"cluster {c}: n={len(genes)} significant terms={results[c]['n_sig']}",
                  flush=True)
        except Exception as e:
            # Record error (truncated) rather than aborting the whole run.
            results[c] = {"n": len(genes), "error": str(e)[:ERROR_TRUNCATE_LEN]}
            print(f"cluster {c}: ERROR {e}", flush=True)

        # Throttle Enrichr API requests to stay within published rate limits.
        time.sleep(ENRICHR_THROTTLE_SEC)

    # Write enrichment results to JSON on disk.
    with open(config.result_path(OUT_FILE), "w") as f:
        json.dump(results, f, indent=JSON_INDENT)

    # Report summary: how many clusters showed >=1 significant GO term.
    n_sig_clusters = sum(1 for v in results.values() if v.get("n_sig", 0) > 0)
    print(f"clusters with >=1 significant GO term: "
          f"{n_sig_clusters}/{config.CLUSTER_K}")


if __name__ == "__main__":
    main()
