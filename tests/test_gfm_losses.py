#!/usr/bin/env python3
# =============================================================================
# @file    tests/test_gfm_losses.py
# @brief   Unit tests for the pure-PyTorch Graph SSL losses, negative
#          samplers and the centralised config constants. 9 independent
#          assertions covering CSPSampler, the InfoNCE/C3L head and the
#          path-resolution helpers.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun,
#          Tao Zhu
# @date    2026-09-27
# @ingroup tests_and_quality_gates
#
# Input:  scripts/config.py + scripts/gfm_losses.py.
# Output: stdout PASS/FAIL summary + process exit code (0 on all 9 pass).
# Run:    python3 tests/test_gfm_losses.py   (also runnable under pytest).
# =============================================================================
"""Unit tests for the pure-PyTorch GFM losses and config (review item 7).

Run:  python3 -m pytest tests/ -q        # or:  python3 tests/test_gfm_losses.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from config import PROJECT_ROOT, DATA_PROC, SEED, set_seed  # noqa: E402
from gfm_losses import (  # noqa: E402
    CSPSampler,
    NAMRMask,
    c3l_info_nce_loss,
    csp_loss,
    namr_loss,
)


def test_config_paths_resolve():
    assert os.path.isdir(PROJECT_ROOT)
    assert DATA_PROC.endswith("data/processed")
    assert isinstance(SEED, int)


def test_namr_loss_shapes_and_reduction():
    pred = torch.randn(16, 256)
    tgt = torch.randn(16, 256)
    loss = namr_loss(pred, tgt)
    assert loss.ndim == 0
    # perfect reconstruction -> cosine term vanishes, mse ~0
    assert namr_loss(tgt, tgt).item() < 1e-6


def test_namr_mask_denoise_and_mask():
    x = torch.randn(100, 8)
    for mode in ("denoise", "mask"):
        m = NAMRMask(mask_ratio=0.15, sigma=0.5, mode=mode)
        x_noisy, idx = m.apply(x, device="cpu")
        assert x_noisy.shape == x.shape
        assert 0 < idx.numel() <= int(100 * 0.15) + 1
        # non-masked rows are untouched
        mask = torch.zeros(100, dtype=torch.bool)
        mask[idx] = True
        assert torch.allclose(x_noisy[~mask], x[~mask])
        if mode == "mask":
            assert torch.all(x_noisy[idx] == 0.0)


def test_csp_loss_range():
    pos = torch.randn(64)
    neg = torch.randn(64)
    l = csp_loss(pos, neg)
    assert l.ndim == 0 and l.item() > 0


def test_csp_sampler_never_returns_ppi_or_self_loop():
    n_genes = 200
    # build a small synthetic undirected PPI set
    edges = []
    for i in range(0, n_genes, 2):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    ei = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
    sampler = CSPSampler(ei, n_genes)
    ppi_set = set()
    for i in range(ei.size(1)):
        a, b = ei[0, i].item(), ei[1, i].item()
        ppi_set.add((min(a, b), max(a, b)))
    for _ in range(50):
        pa, pb, na, nb = sampler.sample(128, device="cpu")
        assert (pa != pb).all()
        assert (na != nb).all()
        for j in range(128):
            assert (na[j].item(), nb[j].item()) not in ppi_set


def test_c3l_info_nce_matches_manual_cross_entropy():
    set_seed(0)
    N, C, D, tau = 64, 21, 128, 0.07
    gene = torch.nn.functional.normalize(torch.randn(N, D), dim=-1)
    cancer = torch.nn.functional.normalize(torch.randn(C, D), dim=-1)
    targets = torch.randint(0, C, (N,))
    loss = c3l_info_nce_loss(gene, cancer, targets, temperature=tau)
    manual = torch.nn.functional.cross_entropy((gene @ cancer.t()) / tau, targets)
    assert torch.allclose(loss, manual, atol=1e-5)


def test_c3l_info_nce_random_baseline_is_lnC_when_logits_uniform():
    # If the model cannot discriminate (uniform logits), the InfoNCE baseline is ln(C).
    C, tau = 21, 0.07
    gene = torch.zeros(2, C)  # zero logits -> uniform softmax -> baseline ln(C)
    cancer = torch.eye(C)
    targets = torch.tensor([0, 1])
    loss = c3l_info_nce_loss(gene, cancer, targets, temperature=tau)
    expected = float(torch.tensor(C).log())  # ln(21) ~ 3.04
    assert abs(loss.item() - expected) < 0.01, (loss.item(), expected)


def test_c3l_info_nce_decreases_when_aligned():
    set_seed(1)
    C, D = 21, 128
    cancer = torch.nn.functional.normalize(torch.randn(C, D), dim=-1)
    targets = torch.arange(C)
    # random genes -> higher loss; aligned genes -> near-zero loss
    random_genes = torch.nn.functional.normalize(torch.randn(C, D), dim=-1)
    loss_random = c3l_info_nce_loss(random_genes, cancer, targets)
    loss_aligned = c3l_info_nce_loss(cancer, cancer, targets)
    assert loss_aligned.item() < loss_random.item()


def test_c3l_info_nce_learns_perfect_separation():
    # If genes are aligned one-to-one with their positive cancer (orthogonal classes),
    # the loss should be well below the random baseline.
    C, D = 21, 128
    basis = torch.eye(C)
    pad = torch.zeros(C, D - C)
    cancer = torch.nn.functional.normalize(torch.cat([basis, pad], dim=1), dim=-1)
    gene = cancer.clone()
    targets = torch.arange(C)
    loss = c3l_info_nce_loss(gene, cancer, targets)
    assert loss.item() < 0.5, loss.item()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
