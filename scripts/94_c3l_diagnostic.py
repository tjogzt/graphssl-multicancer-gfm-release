#!/usr/bin/env python3
"""
@file    94_c3l_diagnostic.py
@brief   Review item 8 diagnostic: why does C3L not learn from a signal that a
         plain linear probe recovers? Tests are run on FROZEN standardised ESM-2
         PCA features (no HGT encoder) so the failure mode can be attributed to
         the C3L head itself rather than the representation:
           (a) plain cross-entropy linear classifier (sanity check matches S15);
           (b) the C3L prototype + InfoNCE head (gene projection + 21 learned
               cancer prototypes) trained for 300 epochs (does the loss drop
               below ln(21)?);
           (c) temperature sensitivity of (b) at tau in {0.03, 0.07, 0.2, 0.5};
           (d) gradient norms of gene projection / cancer prototypes at init.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  data/processed/xena_graph_esm.pt + gene_cancer_targets_5000.pt
Output: stdout summary + results/control/c3l_diagnostic.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/94_c3l_diagnostic.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# =============================================================================
# File-level constants
# =============================================================================
GRAPH_BASENAME = "xena_graph_esm.pt"
TARGETS_BASENAME = "gene_cancer_targets_5000.pt"

# Architectural / training sizes (kept in sync with the C3L head used inside
# 40_pretrain.py; these are fixed rather than CLI parameters so the diagnostic
# is reproducible without additional flags).
FEATURE_DIM = 256
NUM_CANCER_CLASSES = config.N_CANCER_TYPES            # 21 — centralised in config §6
C3L_TAU_DEFAULT = config.C3L_TEMPERATURE               # 0.07 — OSF preregistered value, config §6b
C3L_PROJECTION_HIDDEN = FEATURE_DIM
C3L_PROJECTION_OUT = 128
C3L_PROTO_INIT_STD = 0.01

# --- Training protocol -----------------------------------------------------
RUN_SEED = 42
CROSS_ENTROPY_EPOCHS = 300
C3L_EPOCHS = 300
LR = 1e-2
WEIGHT_DECAY = 1e-4
TEMPERATURE_SWEEP_VALUES = (0.03, 0.07, 0.2, 0.5)

# Standardisation epsilon (per-dim std floor; matches the training protocol).
STANDARDIZE_EPS = 1e-8

# Output artefact.
OUTPUT_TAG = "control"
OUTPUT_JSON = "c3l_diagnostic.json"
JSON_INDENT = 2

# Result dict stable keys.
RKEY_CE_LINEAR = "ce_linear"
RKEY_TAU_DEFAULT = "c3l_head_tau007"
RKEY_GRAD_NORMS = "grad_norms_init"
RKEY_LN21 = "ln21"
RKEY_ENTROPY = "empirical_class_entropy"
CE_FINAL_LOSS = "final_loss"
CE_TRAIN_ACC = "train_accuracy"


def load():
    """Load the frozen input features X and multi-class gene->cancer targets y.

    @brief  Reads the ESM-2 PCA-256 gene features from the heterograph and the
            curated 5000-gene gene_cancer_targets_5000.pt file; standardises the
            features per-dim (matching training protocol) and returns CPU tensors.
    @return Tuple ``(X_tensor, y_tensor)`` where:
            X is float32 of shape ``[n_genes, FEATURE_DIM]``;
            y is int64 of shape ``[n_genes]`` taking values in ``[0, NC)``.
    """
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    targets_path = os.path.join(config.DATA_PROC, TARGETS_BASENAME)
    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    X = g["gene"]["x"].float()
    # Per-dim standardisation: identical to the standardize_features=True runs.
    X = (X - X.mean(0)) / (X.std(0) + STANDARDIZE_EPS)
    t = torch.load(targets_path, map_location="cpu", weights_only=False)
    y = t["targets"].long()
    return X, y


def train_ce(X, y, epochs=CROSS_ENTROPY_EPOCHS):
    """Train a plain linear classifier (sanity baseline) on frozen X.

    @brief  Single linear layer ``W x + b`` trained with AdamW + standard CE for
            ``epochs`` iterations on the FULL dataset (no split; the diagnostic
            measures whether the head *can* memorise, not its generalisation).
    @param  X       Float tensor ``[N, FEATURE_DIM]`` (frozen features).
    @param  y       Long tensor ``[N]`` integer labels in ``[0, NC)``.
    @param  epochs  Number of full-batch optimisation steps.
    @return Tuple ``(final_loss_float, train_accuracy_float)``.
    """
    torch.manual_seed(RUN_SEED)
    lin = nn.Linear(FEATURE_DIM, NUM_CANCER_CLASSES)
    opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    final_loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(lin(X), y)
        loss.backward()
        opt.step()
        final_loss = loss.item()
    with torch.no_grad():
        train_acc = (lin(X).argmax(dim=1) == y).float().mean().item()
    return float(final_loss), train_acc


def train_c3l_head(X, y, tau=C3L_TAU_DEFAULT, epochs=C3L_EPOCHS):
    """Train the exact C3L prototype + InfoNCE head on the same frozen X.

    @brief  Projection MLP (D -> D -> 128 with GELU) + NC learnable prototype
            rows; InfoNCE loss over normalised (gp @ cp^T) / tau logits. Same
            optimiser / schedule as train_ce(). Reports (epoch-0 loss, final
            loss, min loss) so we can see whether the head ever drops below
            ln(21).
    @param  X       Float tensor ``[N, FEATURE_DIM]``.
    @param  y       Long labels.
    @param  tau     InfoNCE temperature (lower = sharper prototypes).
    @param  epochs  Number of full-batch steps.
    @return Tuple ``(epoch0_loss, final_loss, min_loss)``, three plain Python
            floats rounded to double precision.
    """
    torch.manual_seed(RUN_SEED)
    proj = nn.Sequential(
        nn.Linear(FEATURE_DIM, C3L_PROJECTION_HIDDEN),
        nn.GELU(),
        nn.Linear(C3L_PROJECTION_HIDDEN, C3L_PROJECTION_OUT),
    )
    proto = nn.Parameter(torch.randn(NUM_CANCER_CLASSES, C3L_PROJECTION_OUT) * C3L_PROTO_INIT_STD)
    opt = torch.optim.AdamW(
        list(proj.parameters()) + [proto], lr=LR, weight_decay=WEIGHT_DECAY,
    )
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        gp = F.normalize(proj(X), dim=-1)
        cp = F.normalize(proto, dim=-1)
        loss = F.cross_entropy(gp @ cp.t() / tau, y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history[0], history[-1], min(history)


def grad_norms():
    """Gradient norms of the C3L head at initialization (before any training).

    @brief  Seed a fresh C3L head, compute one forward + backward pass on the
            actual data, and report the Frobenius norm of (a) the gene-proj
            first-linear weight gradient and (b) the cancer-prototype param
            gradient -- a lopsided ratio indicates one component (usually the
            prototypes) sees vanishingly small gradients.
    @return Dict ``{gene_proj_grad, cancer_proto_grad}`` (plain floats).
    """
    torch.manual_seed(RUN_SEED)
    lin = nn.Linear(FEATURE_DIM, C3L_PROJECTION_OUT)
    proto = nn.Parameter(torch.randn(NUM_CANCER_CLASSES, C3L_PROJECTION_OUT) * C3L_PROTO_INIT_STD)
    X, y = load()
    gp = F.normalize(lin(X), dim=-1)
    cp = F.normalize(proto, dim=-1)
    loss = F.cross_entropy(gp @ cp.t() / C3L_TAU_DEFAULT, y)
    loss.backward()
    return dict(
        gene_proj_grad=float(lin.weight.grad.norm().item()),
        cancer_proto_grad=float(proto.grad.norm().item()),
    )


def _empirical_class_entropy(y):
    """Compute entropy of the empirical class distribution (natural log units).

    @brief  Internal helper: computes H[Y] = - sum_k p_k ln p_k on non-empty
            classes only, matching the reference entropy value ln(21) used in
            the manuscript.
    @param  y   Integer class labels (numpy int array or 1-D long tensor).
    @return Plain Python float entropy value.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=NUM_CANCER_CLASSES)
    mask = counts > 0
    probs = counts[mask] / float(len(y))
    return float(-(probs * np.log(probs)).sum())


def main():
    """Run every diagnostic sub-protocol, write JSON, print summary.

    @brief  1. run the sanity CE linear classifier; 2. run C3L head at the
            default temperature and at the full sweep; 3. measure init-time
            gradient norms; 4. record reference entropies; 5. serialise.
    @return None.
    """
    X, y = load()
    res = {}
    loss0, acc0 = train_ce(X, y)
    res[RKEY_CE_LINEAR] = {CE_FINAL_LOSS: loss0, CE_TRAIN_ACC: acc0}

    # Default-temperature run (kept as a dedicated stable key for plots).
    # Use the OSF-preregistered config.C3L_TEMPERATURE instead of a bare literal 0.07.
    h0d, hfd, hmd = train_c3l_head(X, y, config.C3L_TEMPERATURE)  # = 0.07 (see config §6b)
    res[RKEY_TAU_DEFAULT] = {
        "train_epoch0": h0d, "train_final": hfd, "min_loss": hmd,
    }
    for tau in TEMPERATURE_SWEEP_VALUES:
        t0, tf, tm = train_c3l_head(X, y, tau)
        res[f"c3l_head_tau{tau}"] = {"epoch0": t0, "final": tf, "min": tm}

    res[RKEY_GRAD_NORMS] = grad_norms()
    res[RKEY_LN21] = float(np.log(NUM_CANCER_CLASSES))
    res[RKEY_ENTROPY] = _empirical_class_entropy(y)

    os.makedirs(config.result_path(OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.result_path(OUTPUT_TAG), OUTPUT_JSON)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=JSON_INDENT)
    print(f"[written] {out_path}")
    for k, v in res.items():
        print(f"{k} -> {v}")


if __name__ == "__main__":
    main()
