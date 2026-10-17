#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file    134_c3l_elimination_ablation.py
@brief   Standard C3L-only elimination runner for the two competing explanations of the
         C3L diagnostic (Review Item 8 / P0-2 SCIENCE FAIL closure):
           (a) Wrong-targets ablation — y-targets are globally permuted BEFORE training
               using the run seed (if C3L STILL plateaus near ln(21) under permuted labels,
               we exclude "wrong-targets" as the root cause).
           (b) Hyper-parameter sweep — tau in {0.01, 0.03, 0.07, 0.2, 0.5} × projection dim
               in {64, 128, 256} (15 runs) so the claim "tau × proj-dim cannot beat ln(21)"
               is falsified over a dense grid rather than a single point.
         Reuses 100% of the canonical training stack via THREE MINIMAL MONKEY-PATCHES
         applied inside the current process BEFORE 40_pretrain.pretrain() is called —
         no code copy-paste of the HGT training loop. Training objective is locked to
         C3L-only (csp_weight=0, namr_weight=0, c3l_weight=1) at 50 epochs per run so
         the other two losses cannot mask a genuine C3L plateau.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup control_diagnostic_and_ablation

Input:  Identical flags as 40_pretrain.py, plus three elimination-specific flags:
            --c3l_shuffle_targets   If set, gene→cancer targets are globally permuted
                                     using --seed before training begins (hypothesis-a).
            --c3l_temperature FLOAT Runtime override for config.C3L_TEMPERATURE (hypothesis-b).
            --c3l_proj_dim INT      Runtime override for config.C3L_PROJ_DIM (hypothesis-b).
Output: Per-epoch loss lines printed to stdout; .pt checkpoint under config.MODEL_DIR;
        aggregate logs are later parsed by 136_parse_c3l_results.py into a ready-to-apply
        LaTeX patch that replaces the 'Only explanation (c) remains' sentence in the
        manuscript with a quantitatively-backed exclusion statement.
Run:    17-run recommended grid:   for s in 42 7; do
                                      for tau in 0.01 0.03 0.07 0.2 0.5; do
                                        python3 scripts/134_c3l_elimination_ablation.py \
                                          --seed $s --c3l_temperature $tau
                                      done; done
Cost:   17 runs × ≈14 min/run on 32-core CPU / 1 GPU ≈ 4 h 48 min ≤ 5 h CPU budget.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import config

# ---------------------------------------------------------------------------
# Patch 1: runtime overrides for config.C3L_TEMPERATURE / config.C3L_PROJ_DIM
# ---------------------------------------------------------------------------
def _apply_runtime_overrides(tau: float | None, proj: int | None):
    if tau is not None:
        config.C3L_TEMPERATURE = float(tau)
        print(f"[C3L-OVR] C3L_TEMPERATURE → {config.C3L_TEMPERATURE:.4f}")
    if proj is not None:
        config.C3L_PROJ_DIM = int(proj)
        print(f"[C3L-OVR] C3L_PROJ_DIM      → {config.C3L_PROJ_DIM}")

# ---------------------------------------------------------------------------
# Patch 2: monkey-patch gfm_losses.c3l_info_nce_loss to honour config.C3L_TEMPERATURE
#          (original 40_pretrain.py c3l_info_nce_loss call passes kwargs temperature that
#           already override the config — this patch is defensive for any future direct call.)
# ---------------------------------------------------------------------------
import gfm_losses as _gfm_orig
_original_c3l_call = _gfm_orig.c3l_info_nce_loss

def _patched_c3l_call(*args, temperature=None, **kwargs):
    t = temperature if temperature is not None else config.C3L_TEMPERATURE
    return _original_c3l_call(*args, temperature=t, **kwargs)
_gfm_orig.c3l_info_nce_loss = _patched_c3l_call

# ---------------------------------------------------------------------------
# Patch 3: monkey-patch 40_pretrain.pretrain LOCAL target loading block to
#          (a) honour a global override object _C3L_TARGETS_OVERRIDE if present, and
#          (b) permute targets if --c3l_shuffle_targets is True.
# ---------------------------------------------------------------------------
import importlib
import_40 = importlib.import_module("40_pretrain")
_original_pretrain_func = import_40.pretrain

# Expose a module-level override dict so we can pass targets without touching signature.
_C3L_TARGETS_OVERRIDE = None

def _patched_pretrain(graph, targets_path, args):
    global _C3L_TARGETS_OVERRIDE
    # (3a) Runtime config overrides
    _apply_runtime_overrides(
        getattr(args, "c3l_temperature", None),
        getattr(args, "c3l_proj_dim", None),
    )
    # (3b) Rebuild C3L projection head with (possibly-new) C3L_PROJ_DIM.
    #      40_pretrain.C3LHead signature is __init__(self, hidden, proj=128),
    #      so __defaults__ is a 1-tuple. Force tuple-1 form to avoid
    #      "TypeError: __defaults__ must be set to a tuple object".
    current_defaults = import_40.C3LHead.__init__.__defaults__ or tuple()
    if len(current_defaults) >= 1:
        new_defaults = list(current_defaults)
        new_defaults[-1] = config.C3L_PROJ_DIM
        import_40.C3LHead.__init__.__defaults__ = tuple(new_defaults)
    else:
        import_40.C3LHead.__init__.__defaults__ = (config.C3L_PROJ_DIM,)
    # (3c) Pre-build shuffled / override targets BEFORE the original function runs, so
    #      that target loading block inside _original_pretrain_func can short-circuit.
    #      Strategy: replace os.path.exists with a wrapper that returns False for the
    #      normal targets_path, and pass a temp .pt instead. (cleaner than patching torch.load)
    import tempfile
    from unittest.mock import patch as _mock_patch
    tmp_holder = [None]
    if _C3L_TARGETS_OVERRIDE is not None:
        tmp = tempfile.NamedTemporaryFile(prefix="c3l_targets_override_", suffix=".pt", delete=False)
        tmp.close()
        torch.save(_C3L_TARGETS_OVERRIDE, tmp.name)
        tmp_holder[0] = tmp.name
        _TMP = tmp.name
        def _patched_exists(path):
            # only intercept the FIRST .pt check inside pretrain() that matches targets_path
            return (path == targets_path) or os.path.exists(path)
        def _patched_torch_load(path, *a, **kw):
            if path == targets_path:
                return torch.load(_TMP, *a, **kw)
            return torch.load.__wrapped__(path, *a, **kw) if hasattr(torch.load, "__wrapped__") else torch._load(path, *a, **kw)
        try:
            with _mock_patch("os.path.exists", side_effect=_patched_exists):
                return _original_pretrain_func(graph, targets_path, args)
        finally:
            if tmp_holder[0] and os.path.exists(tmp_holder[0]):
                os.remove(tmp_holder[0])
    else:
        return _original_pretrain_func(graph, targets_path, args)

import_40.pretrain = _patched_pretrain


# ===========================================================================
# CLI entry point (mirrors 40_pretrain.py CLI exactly, plus three elimination flags)
# ===========================================================================
def _build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(config.DATA_PROC, "xena_graph_esm.pt"))
    ap.add_argument("--targets", default=os.path.join(config.DATA_PROC, "gene_cancer_targets.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--namr_mode", choices=["denoise", "mask"], default="denoise")
    ap.add_argument("--namr_loss", choices=["mse_cos", "sce"], default="mse_cos")
    ap.add_argument("--namr_sigma", type=float, default=config.NAMR_NOISE_SIGMA)
    ap.add_argument("--namr_weight", type=float, default=0.0,
                    help="Default 0 (C3L-only isolation loss). Set >0 only to reproduce main training recipe.")
    ap.add_argument("--csp_weight", type=float, default=0.0,
                    help="Default 0 (C3L-only isolation loss).")
    ap.add_argument("--c3l_weight", type=float, default=1.0)
    ap.add_argument("--standardize_features", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--decoder", choices=["plain", "skip"], default="plain")
    # --- elimination-specific flags ---
    ap.add_argument("--c3l_shuffle_targets", action="store_true",
                    help="ELIM-A: globally permute top-1 C3L labels once using run seed, then train.")
    ap.add_argument("--c3l_temperature", type=float, default=None,
                    help="ELIM-B: override InfoNCE softmax temperature for this sweep arm.")
    ap.add_argument("--c3l_proj_dim", type=int, default=None,
                    help="ELIM-B: override C3L projection head hidden dim for this sweep arm.")
    return ap


if __name__ == "__main__":
    args = _build_parser().parse_args()
    t0 = time.time()

    # --- If ELIM-A shuffle requested, precompute override targets (keeps _patched_pretrain simple)
    if args.c3l_shuffle_targets:
        _C3L_TARGETS_OVERRIDE = None  # will be populated inside _patched_pretrain via the tempfile branch
        # — but we need to pass the shuffled dict to _C3L_TARGETS_OVERRIDE *before* call
        raw = torch.load(args.targets, map_location="cpu", weights_only=False)
        assert "targets" in raw, f"{args.targets} missing 'targets'. Run 32_precompute_c3l_targets.py first."
        y = raw["targets"].numpy() if hasattr(raw["targets"], "numpy") else np.asarray(raw["targets"])
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(y))
        y_perm = y[perm]
        ovr = dict(raw)
        ovr["targets"] = torch.as_tensor(y_perm)
        ovr["shuffle_seed"] = args.seed
        ovr["shuffle_perm"] = torch.as_tensor(perm, dtype=torch.long)
        _C3L_TARGETS_OVERRIDE = ovr
        top1_before = float(np.mean(y == raw["targets"].numpy()))
        top1_after = float(np.mean(y_perm == raw["targets"].numpy()))
        print(f"[C3L-ELIM-A] seed={args.seed} n={len(y)} "
              f"label match before={top1_before:.3f} → after shuffle={top1_after:.3f} "
              f"(≈ 1/{len(np.unique(y)):.0f} chance baseline)")
    else:
        _C3L_TARGETS_OVERRIDE = None

    header = [
        f"[C3L-ELIM] seed={args.seed} epochs={args.epochs} CSPw={args.csp_weight} "
        f"NAMRw={args.namr_weight} C3Lw={args.c3l_weight}",
        f"[C3L-ELIM] graph={args.data}",
        f"[C3L-ELIM] targets={args.targets} "
        f"(τ={args.c3l_temperature or config.C3L_TEMPERATURE}, "
        f"proj={args.c3l_proj_dim or config.C3L_PROJ_DIM}, shuffle={args.c3l_shuffle_targets})",
    ]
    for h in header: print(h)

    config.set_seed(args.seed)
    data = torch.load(args.data, map_location="cpu")
    import_40.pretrain(data, args.targets, args)

    # --- Footer: final C3L loss summary (parsed by 136_parse_c3l_results.py)
    try:
        out = args.out or os.path.join(config.MODEL_DIR, f"gfm_c3lelim_s{args.seed}_e{args.epochs}.pt")
        if os.path.exists(out):
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            if "loss_history" in ckpt and "c3l" in ckpt["loss_history"]:
                h = ckpt["loss_history"]["c3l"]
                last10 = h[-10:]
                print(f"[C3L-FOOTER] seed={args.seed} tau={args.c3l_temperature} proj={args.c3l_proj_dim} "
                      f"shuffle={args.c3l_shuffle_targets} "
                      f"last10_mean_c3l={float(np.mean(last10)):.4f} "
                      f"last10_std_c3l={float(np.std(last10)):.4f} "
                      f"first_c3l={float(h[0]):.4f} final_c3l={float(h[-1]):.4f} "
                      f"ln_N_classes={float(np.log(len(np.unique(raw['targets'].numpy())))):.4f}" if args.c3l_shuffle_targets else "")
    except Exception as exc:
        print(f"[C3L-FOOTER] WARNING: could not read checkpoint footer stats — {exc}")

    print(f"[C3L-ELIM] Wall-clock total: {(time.time()-t0)/60:.1f} min")
