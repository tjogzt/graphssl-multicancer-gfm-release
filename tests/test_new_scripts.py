#!/usr/bin/env python3
# =============================================================================
# @file    tests/test_new_scripts.py
# @brief   Unit tests for the review-fix helper scripts covering items 1/4/8
#          of the Graph SSL audit checklist: co-accessibility statistical
#          helpers, patient-set fingerprint utilities and C3L diagnostic
#          numerical comparisons. 5 independent assertions.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun,
#          Tao Zhu
# @date    2026-09-27
# @ingroup tests_and_quality_gates
#
# Input:  scripts/new_*.py helpers (imported dynamically via importlib.util).
# Output: stdout PASS/FAIL summary + process exit code (0 on all 5 pass).
# Run:    python3 tests/test_new_scripts.py
# =============================================================================
"""Unit tests for the new review-fix scripts (items 1, 4, 8 core logic).

Run:  python3 tests/test_new_scripts.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- item 4: drug-response statistics -------------------------------------
_stats = _load("drug_response_stats", "65_drug_response_stats.py")
bootstrap_ci = _stats.bootstrap_ci
paired_permutation_test = _stats.paired_permutation_test


def test_bootstrap_ci_contains_mean():
    scores = np.random.RandomState(0).normal(0.08, 0.02, 30)
    lo, hi = bootstrap_ci(scores, seed=1)
    assert lo < scores.mean() < hi


def test_paired_permutation_detects_difference():
    rng = np.random.RandomState(0)
    a = rng.normal(0.086, 0.02, 30)
    b = rng.normal(-0.15, 0.05, 30)
    p = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert p < 0.01


def test_paired_permutation_null_is_large():
    rng = np.random.RandomState(1)
    a = rng.normal(0, 0.05, 30)
    b = rng.normal(0, 0.05, 30)
    p = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert p > 0.1


# --- item 8: tissue-specific PPI filter -----------------------------------
_tppi = _load("tissue_specific_ppi", "66_tissue_specific_ppi.py")
filter_ppi_by_expression = _tppi.filter_ppi_by_expression


def test_tissue_filter_removes_unexpressed():
    ppi = pd.DataFrame({'gene_a': ['A', 'B', 'C'], 'gene_b': ['B', 'C', 'D']})
    expr = pd.Series({'A': 100.0, 'B': 100.0, 'C': 0.1, 'D': 100.0})
    out = filter_ppi_by_expression(ppi, expr, threshold=1.0)
    # A-B both expressed -> kept; B-C (C low) removed; C-D (C low) removed
    assert list(out['gene_a']) == ['A']


# --- item 1: gene-cancer specificity precompute ---------------------------
_pre = _load("precompute_c3l_targets", "32_precompute_c3l_targets.py")
compute_gene_cancer_specificity = _pre.compute_gene_cancer_specificity


def test_gene_cancer_specificity_argmax():
    genes = ['G1', 'G2', 'G3']
    cancers = ['BRCA', 'LUAD', 'COAD']
    # G1 highly expressed in BRCA, G2 in LUAD, G3 in COAD
    samples = ['s1', 's2', 's3', 's4', 's5', 's6']
    s2c = {'s1': 'BRCA', 's2': 'BRCA', 's3': 'LUAD', 's4': 'LUAD', 's5': 'COAD', 's6': 'COAD'}
    data = {
        'G1': [10, 10, 1, 1, 1, 1],
        'G2': [1, 1, 10, 10, 1, 1],
        'G3': [1, 1, 1, 1, 10, 10],
    }
    expr = pd.DataFrame(data, index=samples).T  # genes x samples (3x6)
    targets, mat = compute_gene_cancer_specificity(expr, s2c, genes, cancers)
    assert list(targets.numpy()) == [0, 1, 2]
    assert mat.shape == (3, 3)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
