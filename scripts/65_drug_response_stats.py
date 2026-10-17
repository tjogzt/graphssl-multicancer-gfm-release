#!/usr/bin/env python3
"""
@file    65_drug_response_stats.py
@brief   Rigorous drug-response statistical reporting. Replaces relative percentage
         framing with: (1) absolute R^2 with 95% bootstrap confidence interval over
         the splits; (2) a paired permutation test (GFM vs. baseline) for the
         difference in R^2; (3) a literature reference note on the GDSC
         cell-line-feature baseline (~0.3). Core statistics functions are pure
         numpy/scipy and unit-testable.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup downstream_validation

Input:  --gfm             .npy of per-split GFM R^2 values (from 64)
        --baseline        .npy of per-split baseline R^2 values (from 64)
        --baseline_label  label for the baseline column (default "random features")
Output: stdout only (absolute R^2, bootstrap CI, delta, permutation p-value)
Run:    python scripts/65_drug_response_stats.py --gfm <gfm.npy> --baseline <baseline.npy>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# Configuration Constants (protocol parameters and reference values)
# ============================================================================

# Literature reference: GDSC cell-line-feature baseline R^2.
# This work does NOT use cell-line features, so performance is expectedly lower.
REF_BASELINE_R2 = 0.3

# Bootstrap confidence interval parameters.
BOOTSTRAP_DEFAULT_CI = 0.95
BOOTSTRAP_DEFAULT_N = 2000

# Paired permutation test parameters.
PERMUTATION_DEFAULT_N = 10000

# Significance cutoff for p-value interpretation.
SIGNIFICANCE_P_CUTOFF = 0.05

# Permutation p-value conservative floor (ensures non-zero p-value even with B=0).
PERM_PVALUE_EPS_B = 1  # +1 in numerator and denominator for conservative estimate


# ============================================================================
# Core Statistical Functions
# ============================================================================

def bootstrap_ci(scores, ci=BOOTSTRAP_DEFAULT_CI, n_bootstrap=BOOTSTRAP_DEFAULT_N, seed=config.SEED):
    """
    @brief   Bootstrap percentile confidence interval for the population mean.

    @details Draws n_bootstrap resamples with replacement, computes the mean of
             each, and reports the (1-ci)/2 and (1+ci)/2 percentiles.

    @param   scores        Array of scalar scores (1-D numeric).
    @param   ci            Confidence level (0 < ci < 1, e.g., 0.95 for 95%).
    @param   n_bootstrap   Number of bootstrap resamples to draw.
    @param   seed          Random seed for reproducible resampling.
    @return  Tuple of (lower_bound, upper_bound) as Python floats.
    @note    Assumes i.i.d. observations; percentile method is used (not BCa).
    """
    rng = np.random.RandomState(seed)
    scores = np.asarray(scores, dtype=np.float64)
    means = np.empty(n_bootstrap)

    # Draw bootstrap resamples and compute the mean of each.
    for i in range(n_bootstrap):
        means[i] = rng.choice(scores, size=scores.size, replace=True).mean()

    # Compute percentile-based CI bounds.
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def paired_permutation_test(a, b, n_perm=PERMUTATION_DEFAULT_N, seed=config.SEED):
    """
    @brief   Two-sided paired permutation test on the mean difference (a - b).

    @details Implements the exchangeable-null permutation of paired differences
             via random sign flips: under the null of no difference, the sign of
             each paired difference is equally likely to be positive or negative.

    @param   a         First condition score array [n_pairs].
    @param   b         Second condition score array [n_pairs] (paired with a).
    @param   n_perm    Number of permutation iterations (B).
    @param   seed      Random seed for reproducible sign flips.
    @return  Conservative two-sided p-value as float: (count_extreme + 1) / (B + 1).
    @note    Arrays a and b MUST have identical shape (paired observations).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "Paired arrays must have identical shape"

    rng = np.random.RandomState(seed)
    obs = np.mean(a - b)
    diffs = a - b

    # Random sign flips implement the exchangeable-null distribution.
    signs = rng.choice([-1.0, 1.0], size=(n_perm, a.size))
    perm = signs * diffs[None, :]
    perm_means = perm.mean(axis=1)

    # Count permutations at least as extreme as observed (two-tailed).
    count = int((np.abs(perm_means) >= np.abs(obs)).sum())

    # Conservative estimate: add 1 to both numerator and denominator.
    return float((count + PERM_PVALUE_EPS_B) / (n_perm + PERM_PVALUE_EPS_B))


def summarize(gfm, baseline, baseline_label="random features", ref_baseline=REF_BASELINE_R2):
    """
    @brief   Produce a printable summary string with corrected metrics.

    @details Reports:
             (1) GFM absolute R^2 with 95% bootstrap CI
             (2) Baseline R^2
             (3) Absolute delta (GFM - baseline), NOT relative percentage
             (4) Paired permutation test p-value with significance annotation
             (5) Literature reference note on cell-line-feature baseline

    @param   gfm              Per-split GFM R^2 array [n_splits].
    @param   baseline         Per-split baseline R^2 array [n_splits, paired with gfm].
    @param   baseline_label   Human-readable label for the baseline method.
    @param   ref_baseline     Literature reference baseline R^2 value.
    @return  Multi-line summary string suitable for stdout or report inclusion.
    @note    Uses ABSOLUTE delta, never a relative percentage over a possibly-negative baseline.
    """
    gfm = np.asarray(gfm, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    # Compute bootstrap CI for the GFM mean R^2.
    lo, hi = bootstrap_ci(gfm)
    # Compute paired permutation test for GFM vs. baseline difference.
    p = paired_permutation_test(gfm, baseline)

    gfm_mean = gfm.mean()
    base_mean = baseline.mean()
    # Absolute difference (NOT relative percentage over potentially-negative baseline).
    delta = gfm_mean - base_mean

    # Build p-value annotation based on significance cutoff.
    p_annotation = "  (significant)" if p < SIGNIFICANCE_P_CUTOFF else "  (not significant)"

    lines = [
        f"GFM absolute R^2: {gfm_mean:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])",
        f"{baseline_label} R^2: {base_mean:.3f}",
        f"absolute delta (GFM - baseline): {delta:+.3f}",
        f"paired permutation p-value: {p:.4f}" + p_annotation,
        f"reference GDSC cell-line-feature baseline R^2 ~ {ref_baseline} "
        f"(this work does not use cell-line features)",
    ]
    return "\n".join(lines)


def main():
    """
    @brief   Main entry point: load per-split R^2 .npy files, run statistical
             tests, and print corrected summary report.
    @return  None (outputs to stdout).
    @note    Requires mandatory --gfm and --baseline CLI arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--gfm', required=True, help='.npy of per-split GFM R^2 values')
    ap.add_argument('--baseline', required=True, help='.npy of per-split baseline R^2 values')
    ap.add_argument('--baseline_label', default='random features')
    args = ap.parse_args()

    # Load per-split arrays from disk.
    gfm = np.load(args.gfm)
    baseline = np.load(args.baseline)

    # Compute and print the full summary report.
    print(summarize(gfm, baseline, args.baseline_label))


if __name__ == '__main__':
    main()
