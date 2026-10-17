#!/usr/bin/env python3
"""
@file    93_csp_geometry_shortcut.py
@brief   Review item 7 diagnostics:
           (1) CSP-only vs NAMR-only embedding geometry: effective rank, per-dim
               std, mean pairwise cosine / squared distance (does CSP collapse
               the manifold?);
           (2) Skip-decoder shortcut test: evaluate the skip-connected NAMR
               decoder loss with the encoder hidden state INTACT vs ZEROED. If
               the loss barely moves, the decoder copies the noisy input and
               ignores the encoder (shortcut pathology).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt + checkpoints csp_only.pt / namr_only.pt
        + namr_only_s123.pt (skip decoder variant).
Output: stdout summary + results/control/csp_geometry_shortcut.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/93_csp_geometry_shortcut.py
"""
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import config
import gfm_losses
from gfm_losses import NAMRMask

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Graph artifact.
GRAPH_BASENAME = "xena_graph_esm.pt"

# Result tags + checkpoint basenames.
ABLATION_TAG = "ablation_real"
CKPT_CSP_ONLY = "csp_only.pt"
CKPT_NAMR_ONLY = "namr_only.pt"
# Skip-decoder checkpoint lives in a retired sibling directory; the relative
# path is preserved so the audit references the actual artifact bytes that
# were produced (registry entry: see checkpoint_registry.json for full sha).
SKIP_DECODER_DIR = os.path.join("..", "namr5_skip_decoder_DISCARDED")
CKPT_SKIP_NAMR = "namr_only_s123.pt"

# Output artifact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "csp_geometry_shortcut.json"
JSON_INDENT = 2

# Run protocol constants.
RUN_SEED = 42
STANDARDIZE_EPS = 1e-8

# --- Geometry diagnostic ---------------------------------------------------
SVD_SPECTRUM_THRESH = 1e-12        # drop near-zero singular values
GEOM_PAIRS_CANDIDATE = 4000        # random (i,j) draws before same-pair dedup
GEOM_PAIRS_USED = 2000             # distinct pairs actually used after dedup
GEOM_COSINE_NORM_EPS = 1e-9

# --- Skip-decoder shortcut diagnostic --------------------------------------
NAMR_SIGMA = 0.5
HGT_HIDDEN_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Result dict keys (stable for downstream readers).
KEY_CSP_RAW = "csp_only_raw"
KEY_NAMR_RAW = "namr_only_raw"
KEY_SKIP = "skip_shortcut"
SKIP_MSE_WITH_H = "mse_with_h"
SKIP_MSE_ZERO_H = "mse_zero_h"
SKIP_DELTA = "delta"


def _load_pretrain_module():
    """Dynamically load 40_pretrain helper.

    @brief  Internal helper.
    @return Loaded module exposing build_data_dict / HGTEncoder / MLPHead.
    """
    spec = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf_mod)
    return pf_mod


def embeddings(ckpt_path, standardize=False):
    """Return the trained HGT's gene embeddings on CPU.

    @brief  Load the graph, optionally standardize gene features (matching the
            model's own training protocol), run a single forward pass and
            return the numpy float64 gene-embedding matrix.
    @param  ckpt_path     Absolute path to a pretrained *.pt checkpoint.
    @param  standardize   Boolean: True == z-score the gene features before
                          passing through the encoder (standardized runs).
    @return numpy array of shape ``[n_genes, HGT_HIDDEN_DIM]``, dtype float64.
    """
    pf = _load_pretrain_module()
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_dict = {k: data[k]["x"] for k in node_types}
    if standardize and "gene" in x_dict:
        g = x_dict["gene"]
        x_dict["gene"] = (g - g.mean(0)) / (g.std(0) + STANDARDIZE_EPS)
    ei = {e: data[e]["edge_index"] for e in edge_types}
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        X = model(x_dict, ei)["gene"].cpu().numpy()
    return X.astype(np.float64)


def geom(X):
    """Summarise the row-geometry of an embedding matrix X.

    @brief  For rows of X (gene embeddings) compute:
              - effective rank (exponential of the negative Shannon entropy of
                the normalised squared-singular-value distribution);
              - min / median / max per-dimension standard deviation;
              - mean pairwise cosine similarity over GEOM_PAIRS_USED random
                distinct pairs (seeded by RUN_SEED);
              - mean squared Euclidean distance over the same pairs;
              - count of singular values kept above SVD_SPECTRUM_THRESH.
    @param  X   2-D array of shape [N, D]; will be cast to float64 internally.
    @return JSON-serialisable dict with the keys listed above (plain floats).
    """
    X = np.asarray(X, dtype=np.float64)
    # Centre columns before the spectrum decomposition (effective rank is
    # always computed on centred data so translation of the whole cloud does
    # not artificially increase the rank).
    Xc = X - X.mean(axis=0)
    singular = np.linalg.svd(Xc, compute_uv=False)
    s_kept = singular[singular > SVD_SPECTRUM_THRESH]
    p_norm = s_kept / s_kept.sum()
    effective_rank = float(np.exp(-(p_norm * np.log(p_norm)).sum()))
    per_dim_std = np.std(X, axis=0)

    # Pairwise statistics on a fixed, seeded random subset of distinct pairs.
    rng = np.random.RandomState(RUN_SEED)
    n_rows = X.shape[0]
    i = rng.randint(0, n_rows, GEOM_PAIRS_CANDIDATE)
    j = rng.randint(0, n_rows, GEOM_PAIRS_CANDIDATE)
    ok = i != j
    i, j = i[ok][:GEOM_PAIRS_USED], j[ok][:GEOM_PAIRS_USED]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + GEOM_COSINE_NORM_EPS)
    cos_mean = float(((Xn[i] * Xn[j]).sum(axis=1)).mean())
    sqdist_mean = float((((X[i] - X[j]) ** 2).sum(axis=1)).mean())

    return dict(
        effective_rank=effective_rank,
        per_dim_std=dict(
            min=float(per_dim_std.min()),
            median=float(np.median(per_dim_std)),
            max=float(per_dim_std.max()),
        ),
        mean_pairwise_cosine=cos_mean,
        mean_pairwise_sqdist=sqdist_mean,
        n_singular=len(s_kept),
    )


def main():
    """Run geometry + shortcut diagnostics, serialise JSON, print summary.

    @brief  1. Run geom() on CSP-only and NAMR-only raw embeddings;
            2. Load the skip-decoder checkpoint, run a fixed NAMR mask, and
               compare decoder output with h present vs h zeroed;
            3. Write JSON and print each top-level key to stdout.
    @return None.
    """
    pf = _load_pretrain_module()
    ablation_dir = config.result_path(ABLATION_TAG)
    res = {}
    res[KEY_CSP_RAW] = geom(
        embeddings(os.path.join(ablation_dir, CKPT_CSP_ONLY), False)
    )
    res[KEY_NAMR_RAW] = geom(
        embeddings(os.path.join(ablation_dir, CKPT_NAMR_ONLY), False)
    )

    # --- Skip-decoder shortcut test -------------------------------------------
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    gene_dim = in_dims["gene"]
    x_dict = {k: data[k]["x"] for k in node_types}
    g_x = x_dict["gene"]
    x_std = (g_x - g_x.mean(0)) / (g_x.std(0) + STANDARDIZE_EPS)
    ei = {e: data[e]["edge_index"] for e in edge_types}

    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    )
    # Skip decoder's first linear input = HGT_HIDDEN_DIM (h) + gene_dim (x_noisy)
    skip_head = pf.MLPHead(HGT_HIDDEN_DIM + gene_dim, gene_dim)
    skip_ckpt_path = os.path.join(
        ablation_dir, SKIP_DECODER_DIR, CKPT_SKIP_NAMR,
    )
    skip_ckpt = torch.load(skip_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(skip_ckpt["model"])
    skip_head.load_state_dict(skip_ckpt["namr"])
    model.eval()
    skip_head.eval()

    torch.manual_seed(RUN_SEED)
    masker = NAMRMask(
        mask_ratio=config.NAMR_MASK_RATIO, sigma=NAMR_SIGMA, mode="denoise",
    )
    x_noisy, mask_idx = masker.apply(x_std, dev)
    xnd_gene = {k: (x_noisy if k == "gene" else x_dict[k]) for k in x_dict}
    with torch.no_grad():
        zn = model(xnd_gene, ei)
        h_hidden = zn["gene"][mask_idx]
        x_obs = x_noisy[mask_idx]
        tgt = x_std[mask_idx]
        full_input = torch.cat([h_hidden, x_obs], dim=-1)
        zero_input = torch.cat([torch.zeros_like(h_hidden), x_obs], dim=-1)
        pred_full = skip_head(full_input)
        pred_zero = skip_head(zero_input)
        mse_full = F.mse_loss(pred_full, tgt).item()
        mse_zero = F.mse_loss(pred_zero, tgt).item()
    res[KEY_SKIP] = {
        SKIP_MSE_WITH_H: mse_full,
        SKIP_MSE_ZERO_H: mse_zero,
        SKIP_DELTA: mse_zero - mse_full,
    }

    # --- Serialisation + English-only stdout ----------------------------------
    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")
    for k, v in res.items():
        print(f"{k} -> {v}")


if __name__ == "__main__":
    main()
