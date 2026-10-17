#!/usr/bin/env python3
"""
@file    74_c3l_permutation_test.py
@brief   C3L permutation test: whether the C3L (gene-cancer-target InfoNCE) objective
         learned any real signal. The trained checkpoint's loss and top-1 accuracy are
         compared to a null distribution built by shuffling the gene-cancer target
         pairing B times and recomputing the same statistics; the one-sided p-value is
         the conservative (1 + count) / (1 + B) fraction of null values at least as
         good as the observed.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup functional_analysis

Input:  --ckpt (checkpoint .pt), --targets (gene_cancer_targets.pt),
        --data (graph .pt, default config.DATA_PROC/xena_graph_esm.pt),
        --B/--smoke (permutation count), --out (optional JSON path), --seed
Output: stdout JSON; also written to --out when supplied
Run:    python scripts/74_c3l_permutation_test.py --ckpt <CKPT> --targets <TARGETS> [--B 1000]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import config

# ============================================================================
# Configuration Constants (artifact names and protocol parameters)
# ============================================================================

# Per-run artifact names (resolved against config paths, not hardcoded dirs).
GRAPH_FILE = "xena_graph_esm.pt"

# Permutation test parameters.
SMOKE_PERMS = 50          # Reduced permutation count for the --smoke quick check.
OBSERVED_REPEATS = 3      # Repeats averaged for the observed statistic (stability).
GENE_SUBSAMPLE = 512      # Genes sampled per evaluation (keeps each pass fast).

# HGT encoder + C3L head architecture parameters.
HGT_EMBEDDING_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Logging prefix tag for console output.
LOG_PREFIX = "[c3l-perm]"

# Status print interval (print progress every N permutations).
PERM_LOG_INTERVAL = 200

# Significance cutoff for conclusion string.
SIGNIFICANCE_P_CUTOFF = 0.05

# Conclusion strings for result payload.
CONCLUSION_NO_SIGNAL = (
    "no learning signal (indistinguishable from null)"
)
CONCLUSION_SIGNAL_PRESENT = (
    "learning signal present (significantly different from null)"
)

# Output JSON indentation.
JSON_INDENT = 2

# Device selection priority order: CUDA > MPS (Apple Silicon) > CPU.
DEVICE_PRIORITY_CUDA = "cuda:0"
DEVICE_BACKEND_MPS = "mps"
DEVICE_FALLBACK_CPU = "cpu"

# Pretraining module filename for dynamic import.
PRETRAIN_MODULE_NAME = "40_pretrain.py"


# ============================================================================
# Core Functions
# ============================================================================

def build_model(graph, device):
    """
    @brief   Build the HGT encoder and C3L head from pretraining module.

    @param   graph    Loaded torch graph object with heterogeneous data.
    @param   device   Target torch device for model placement.
    @return  Tuple of (module, model, c3l_head, x_dict, ei_dict):
             - module: dynamically loaded 40_pretrain module reference
             - model: HGTEncoder instance on `device`
             - c3l_head: C3LHead instance on `device`
             - x_dict: node-type -> feature tensor dict on `device`
             - ei_dict: edge-type -> edge_index tensor dict on `device`
    @note    Imports pretraining module dynamically to avoid circular dependencies.
    """
    import importlib.util
    # Dynamically load pretraining module (HGTEncoder / C3LHead / build_data_dict).
    spec = importlib.util.spec_from_file_location(
        "pf", os.path.join(os.path.dirname(os.path.abspath(__file__)), PRETRAIN_MODULE_NAME))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    # Build heterogeneous graph data dictionaries on target device.
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, device)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}

    # Initialize HGT encoder with checkpoint-matching architecture.
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_EMBEDDING_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS
    ).to(device)

    # Initialize C3L contrastive head on target device.
    c3l_head = pf.C3LHead(HGT_EMBEDDING_DIM).to(device)
    return pf, model, c3l_head, x_dict, ei_dict


def main():
    """
    @brief   Main entry point: run C3L permutation test, print and save JSON result.
    @return  None (outputs JSON to stdout; optionally writes to --out path).
    @note    Supports --smoke flag for quick B=50 validation; one-sided conservative
             p-values computed as (1 + count) / (1 + B).
    """
    # Parse command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Trained checkpoint .pt file")
    ap.add_argument("--targets", required=True, help="gene_cancer_targets.pt file")
    ap.add_argument("--data", default=os.path.join(config.DATA_PROC, GRAPH_FILE),
                    help="Graph .pt artifact path")
    ap.add_argument("--B", type=int, default=config.PERMUTATION_N,
                    help="Number of permutations (B)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"Quick smoke check with B={SMOKE_PERMS}")
    ap.add_argument("--out", default=None,
                    help="Optional output JSON file path")
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="Random seed for reproducible permutations")
    args = ap.parse_args()

    # Apply smoke-test permutation count override if requested.
    B = SMOKE_PERMS if args.smoke else args.B

    # Select best available accelerator device; fall back to CPU.
    device = torch.device(
        DEVICE_PRIORITY_CUDA if torch.cuda.is_available()
        else DEVICE_BACKEND_MPS if torch.backends.mps.is_available()
        else DEVICE_FALLBACK_CPU
    )
    print(f"{LOG_PREFIX} device={device} B={B} ckpt={args.ckpt}")

    # Load graph and build model + data dicts.
    graph = torch.load(args.data, map_location="cpu", weights_only=False)
    _, model, c3l_head, x_dict, ei_dict = build_model(graph, device)

    # Load trained checkpoint weights onto target device.
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    c3l_head.load_state_dict(ck["c3l"])

    # Load gene-cancer target labels onto target device.
    targets = torch.load(
        args.targets, map_location="cpu", weights_only=False
    )["targets"].to(device)
    n_genes = targets.numel()
    n_cancer = len(graph["meta"]["cancer_list"])
    print(f"{LOG_PREFIX} n_genes={n_genes} n_cancer={n_cancer}")

    # Set both models to evaluation mode.
    model.eval()
    c3l_head.eval()

    # Seed both NumPy and PyTorch RNGs for reproducibility.
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Inner evaluation helper: subsample genes, compute loss + top-1 acc.
    # ------------------------------------------------------------------
    def evaluate(tgt, n_genes_sub=GENE_SUBSAMPLE):
        """
        @brief   Randomly subsample genes and evaluate C3L loss + top-1 accuracy.

        @param   tgt             Gene-to-cancer target tensor on device [n_genes].
        @param   n_genes_sub     Number of genes to subsample per evaluation.
        @return  Tuple of (cross_entropy_loss, top1_accuracy) as Python floats.
        @note    Uses torch.randperm for random gene subsetting; no gradients computed.
        """
        gidx = torch.randperm(n_genes, device=device)[:n_genes_sub]
        with torch.no_grad():
            # Full HGT forward pass over entire heterogeneous graph.
            z = model(x_dict, ei_dict)
            # Project gene and cancer embeddings via C3L contrastive head.
            gp = c3l_head(z["gene"])[gidx]
            cp = c3l_head(z["cancer"])
            # Compute InfoNCE-style logits at C3L temperature.
            logits = gp @ cp.t() / config.C3L_TEMPERATURE
            t = tgt[gidx]
            # Standard cross-entropy classification loss.
            loss = torch.nn.functional.cross_entropy(logits, t).item()
            # Top-1 classification accuracy.
            acc = (logits.argmax(1) == t).float().mean().item()
        return loss, acc

    # ------------------------------------------------------------------
    # Step 1: observed (true pairing) statistics, averaged over repeats.
    # ------------------------------------------------------------------
    real_loss, real_acc = [], []
    for _ in range(OBSERVED_REPEATS):
        l, a = evaluate(targets)
        real_loss.append(l)
        real_acc.append(a)
    real_loss = float(np.mean(real_loss))
    real_acc = float(np.mean(real_acc))

    # ------------------------------------------------------------------
    # Step 2: build permutation null distribution (B iterations).
    # ------------------------------------------------------------------
    t0 = time.time()
    null_loss, null_acc = [], []
    perm_targets = targets.clone()

    for i in range(B):
        # Randomly permute target labels to break the gene-cancer pairing.
        perm = torch.randperm(n_genes, device=device)
        perm_targets.copy_(targets[perm])

        # Evaluate loss + accuracy under the permuted null.
        l, a = evaluate(perm_targets)
        null_loss.append(l)
        null_acc.append(a)

        # Periodic progress logging so the user knows the job is alive.
        if (i + 1) % PERM_LOG_INTERVAL == 0:
            print(f"  perm {i+1}/{B} ({time.time()-t0:.0f}s)")

    null_loss = np.array(null_loss)
    null_acc = np.array(null_acc)

    # ------------------------------------------------------------------
    # Step 3: compute one-sided conservative p-values.
    #         - Loss: smaller is better (null_loss <= real_loss).
    #         - Accuracy: larger is better (null_acc >= real_acc).
    #         Conservative: +1 in both numerator and denominator.
    # ------------------------------------------------------------------
    p_loss = (1 + int((null_loss <= real_loss).sum())) / (1 + B)
    p_acc = (1 + int((null_acc >= real_acc).sum())) / (1 + B)

    # Build conclusion string based on significance cutoffs.
    signal_present = not (p_loss > SIGNIFICANCE_P_CUTOFF and p_acc > SIGNIFICANCE_P_CUTOFF)
    conclusion = CONCLUSION_SIGNAL_PRESENT if signal_present else CONCLUSION_NO_SIGNAL

    # Assemble full result payload.
    result = {
        "ckpt": args.ckpt,
        "device": str(device),
        "B": B,
        "real_loss": real_loss,
        "real_acc": real_acc,
        "null_loss_mean": float(null_loss.mean()),
        "null_loss_std": float(null_loss.std()),
        "null_acc_mean": float(null_acc.mean()),
        "null_acc_std": float(null_acc.std()),
        "p_loss": p_loss,
        "p_acc": p_acc,
        "conclusion": conclusion,
        "seconds": round(time.time() - t0, 1),
    }

    # Print JSON to stdout.
    print(json.dumps(result, indent=JSON_INDENT, ensure_ascii=False))

    # Optionally write JSON to disk.
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=JSON_INDENT, ensure_ascii=False)
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
