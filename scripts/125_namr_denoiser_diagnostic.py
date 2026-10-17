#!/usr/bin/env python3
"""
@file    125_namr_denoiser_diagnostic.py
@brief   NAMR denoising diagnostic (review item 6.2). On a FIXED masked-validation
         set, reports reconstruction MSE + cosine terms SEPARATELY for five
         predictors of increasing expressivity:
           (a) constant per-dimension mean
           (b) identity (output = noisy input)
           (c) optimal linear Wiener denoiser
           (d) no-graph MLP denoiser (trained on disjoint 60% split)
           (e) trained graph NAMR (plain decoder)
         Both standardized and unstandardized feature spaces are reported so
         the standardization effect is visible on a common scale.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt + results/ablation_real/namr_only.pt
Output: stdout table + results/control/namr_denoiser_diagnostic.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/125_namr_denoiser_diagnostic.py
"""
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import gfm_losses
from gfm_losses import NAMRMask

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Artifact basenames resolved against config paths.
GRAPH_BASENAME = "xena_graph_esm.pt"
CKPT_RESULT_TAG = "ablation_real"
CKPT_BASENAME = "namr_only.pt"
OUTPUT_RESULT_TAG = "control"
OUTPUT_JSON_FILENAME = "namr_denoiser_diagnostic.json"

# Noise model (matches NAMR training protocol).
NAMR_SIGMA = 0.5
NAMR_MASK_RATIO = config.NAMR_MASK_RATIO
RUN_SEED = 42
MLP_TRAIN_SPLIT_SEED = 1234

# NAMR loss weighting:  MSE + LOSS_COS_WEIGHT * (1 - cosine)
LOSS_COS_WEIGHT = 0.1

# Standardisation epsilon (per-dim std floor).
STANDARDIZE_EPS = 1e-8

# --- Predictor (d) no-graph MLP training hyper-parameters ------------------
MLP_TRAIN_FRACTION = 0.6          # fraction of genes used for MLP training (split)
MLP_LAYERS_D = None               # hidden dim matches feature dim
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_NUM_EPOCHS = 2000

# --- HGT architecture (must match the loaded checkpoint) --------------------
HGT_HIDDEN_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# --- Display / serialization ------------------------------------------------
LOSS_DISPLAY_DIGITS = 4
JSON_INDENT = 2
PRINT_HEADER = (
    "space        | predictor          |   MSE |   cos | "
    f"loss (MSE+{LOSS_COS_WEIGHT}(1-cos))"
)

# Feature-space labels used in the report dict.
SPACE_STANDARDIZED = "standardized"
SPACE_UNSTANDARDIZED = "unstandardized"

# Predictor labels used as stable dict keys in results[space].
KEY_CONSTANT_MEAN = "constant_mean"
KEY_IDENTITY = "identity"
KEY_WIENER_LINEAR = "wiener_linear"
KEY_MLP_NO_GRAPH = "mlp_no_graph"
KEY_GRAPH_NAMR = "graph_namr"
KEY_N_MASKED = "n_masked"


def _load_pretrain_module():
    """Dynamically load the 40_pretrain helper script via importlib.

    @brief  Internal helper exposing HGTEncoder + build_data_dict + MLPHead.
    @return Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf_mod)
    return pf_mod


def mse_cos(pred, tgt):
    """Compute (MSE, mean-cosine, combined NAMR-style loss) for a prediction pair.

    @brief  Evaluate the three scalars used by every denoiser entry of the
            diagnostic table. The combined loss uses the same MSE + 0.1(1-cos)
            weighting that the full NAMR training objective uses.
    @param  pred   Predicted tensor of shape ``[N_masked, D]``.
    @param  tgt    Clean target tensor (same shape).
    @return Tuple ``(mse_float, cosine_float, combined_loss_float)``.
    @note   Uses function-level torch API rather than class loss so we can
            recompute on arbitrary predictors without constructing an NAMRMask.
    """
    mse = F.mse_loss(pred, tgt).item()
    cos = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
    combined = mse + LOSS_COS_WEIGHT * (1.0 - cos)
    return mse, cos, combined


def main():
    """Run the 5-predictor x 2-space NAMR denoising diagnostic and write JSON.

    @brief  (1) load graph, build data dict, standardize features; (2) apply a
            *fixed* NAMR mask + noise (same seed across predictors); (3) run
            constant-mean / identity / Wiener baseline in both spaces; (4) train
            a no-graph MLP denoiser in standardized space on a disjoint 60%
            train split; (5) load the trained graph NAMR and evaluate its plain
            decoder on the same fixed mask; (6) pretty-print a table and write
            the machine-readable JSON artefact.
    @return None.
    """
    pf = _load_pretrain_module()
    config.set_seed(RUN_SEED)

    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    ckpt_path = os.path.join(config.result_path(CKPT_RESULT_TAG), CKPT_BASENAME)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    dev = torch.device("cpu")
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x_raw = data["gene"]["x"].clone()           # heterogeneous PCA-256 [N, D]
    n_genes, feat_dim = x_raw.shape

    # Per-dim standardization (identical to NAMR training protocol).
    x_std = (x_raw - x_raw.mean(0)) / (x_raw.std(0) + STANDARDIZE_EPS)

    # --- Fixed masked-validation set (one mask seed across all predictors) ---
    torch.manual_seed(RUN_SEED)
    masker = NAMRMask(mask_ratio=NAMR_MASK_RATIO, sigma=NAMR_SIGMA, mode="denoise")

    def fixed_noise(x_tensor):
        """Apply masker once and cache the (noisy, mask_idx) pair for reuse."""
        x_noisy, mask_idx = masker.apply(x_tensor, dev)
        return x_noisy, mask_idx

    results = {}
    for space, xc in ((SPACE_STANDARDIZED, x_std), (SPACE_UNSTANDARDIZED, x_raw)):
        x_noisy, midx = fixed_noise(xc)
        tgt = xc[midx]                        # clean targets on masked nodes
        obs = x_noisy[midx]                   # noisy observations on masked nodes

        # (a) constant per-dimension mean predictor
        cmean = xc.mean(0).expand_as(tgt)
        m_a, c_a, L_a = mse_cos(cmean, tgt)
        # (b) identity
        m_b, c_b, L_b = mse_cos(obs, tgt)
        # (c) Wiener optimal-linear denoiser: alpha = var_x / (var_x + sigma^2)
        wiener_alpha = 1.0 / (1.0 + NAMR_SIGMA ** 2)
        m_c, c_c, L_c = mse_cos(wiener_alpha * obs, tgt)

        results[space] = {
            KEY_CONSTANT_MEAN: {"mse": m_a, "cos": c_a, "loss": L_a},
            KEY_IDENTITY:      {"mse": m_b, "cos": c_b, "loss": L_b},
            KEY_WIENER_LINEAR: {"mse": m_c, "cos": c_c, "loss": L_c},
            KEY_N_MASKED:      int(midx.numel()),
        }

    # --- Predictor (d): no-graph MLP denoiser, standardized space ------------
    # Trained on a *separate* 60%/40% node split (MLP-train mask seed != the
    # fixed-validation mask seed), so the MLP cannot simply memorize the
    # validation masked nodes.
    torch.manual_seed(MLP_TRAIN_SPLIT_SEED)
    n_train = int(MLP_TRAIN_FRACTION * n_genes)
    tr_idx = torch.randperm(n_genes)[:n_train]
    tr_obs = (x_std[tr_idx] + torch.randn_like(x_std[tr_idx]) * NAMR_SIGMA)
    tr_tgt = x_std[tr_idx]

    # Re-create the fixed validation mask on standardized space.
    torch.manual_seed(RUN_SEED)
    x_noisy_std, midx_std = fixed_noise(x_std)
    va_tgt = x_std[midx_std]
    va_obs = x_noisy_std[midx_std]

    mlp = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.GELU(),
        nn.Linear(feat_dim, feat_dim),
    )
    opt = torch.optim.AdamW(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    for _ in range(MLP_NUM_EPOCHS):
        opt.zero_grad()
        loss = F.mse_loss(mlp(tr_obs), tr_tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        m_d, c_d, L_d = mse_cos(mlp(va_obs), va_tgt)
    results[SPACE_STANDARDIZED][KEY_MLP_NO_GRAPH] = {"mse": m_d, "cos": c_d, "loss": L_d}

    # --- Predictor (e): trained graph NAMR (plain decoder), standardized space
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    )
    namr_head = pf.MLPHead(HGT_HIDDEN_DIM, feat_dim)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    namr_head.load_state_dict(ck["namr"])
    model.eval()
    namr_head.eval()

    x_dict = {nt: data[nt]["x"] for nt in node_types}
    x_dict["gene"] = x_std
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Re-create the same fixed mask on standardized features.
    torch.manual_seed(RUN_SEED)
    x_noisy_e, midx_e = fixed_noise(x_std)
    xnd = {k: (x_noisy_e if k == "gene" else v) for k, v in x_dict.items()}
    with torch.no_grad():
        zn = model(xnd, ei_dict)
        pred = namr_head(zn["gene"][midx_e])
        m_e, c_e, L_e = mse_cos(pred, x_std[midx_e])
    results[SPACE_STANDARDIZED][KEY_GRAPH_NAMR] = {"mse": m_e, "cos": c_e, "loss": L_e}
    results[SPACE_STANDARDIZED][KEY_N_MASKED] = int(midx_e.numel())

    # --- Serialize + print (English-only stdout) ------------------------------
    os.makedirs(config.result_path(OUTPUT_RESULT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_RESULT_TAG), OUTPUT_JSON_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}\n")
    print(PRINT_HEADER)
    print("-" * len(PRINT_HEADER))
    for space in (SPACE_STANDARDIZED, SPACE_UNSTANDARDIZED):
        for k, v in results[space].items():
            if isinstance(v, dict) and "mse" in v:
                print(
                    f"{space:<12} | {k:<18} | "
                    f"{v['mse']:.{LOSS_DISPLAY_DIGITS}f} | "
                    f"{v['cos']:.{LOSS_DISPLAY_DIGITS}f} | "
                    f"{v['loss']:.{LOSS_DISPLAY_DIGITS}f}"
                )


if __name__ == "__main__":
    main()
