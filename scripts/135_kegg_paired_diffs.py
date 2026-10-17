#!/usr/bin/env python3
"""
@file    135_kegg_paired_diffs.py
@brief   Augment results/kegg_per_seed.json with the paired per-seed KEGG
         differences used in the main text, so that every derived number quoted
         in the manuscript is backed by a reproducible result artifact (the
         number-lock audit L2 can therefore trace every claim back to a file).
         The four headline arm-vs-arm comparisons are stored under
         ``paired_seed_differences`` together with per-seed diff values, the
         set of shared seeds evaluated for both arms, and aggregate min / max /
         mean / all-positive / all-same-sign booleans.
         NOTE: the original implementation was a top-level linear script with
         no entry-point guard; this revision wraps the work in main() while
         preserving every comparison and rounding convention exactly.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  results/kegg_per_seed.json (produced by 130_kegg_per_seed.py)
Output: results/kegg_per_seed.json with an added "paired_seed_differences" block
Run:    GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/135_kegg_paired_diffs.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
KEGG_PER_SEED_REL = "kegg_per_seed.json"
PER_CKPT_KEY = "per_checkpoint"
DIFFS_KEY = "paired_seed_differences"
PC_KEGG_FIELD = "kegg_k20"

# --- Headline arm comparisons (exactly the four pairs quoted in the manuscript
# main text).  Each tuple is (numerator_arm, denominator_arm) -- i.e. diffs
# are computed as  arm_a[s] - arm_b[s]  for every shared seed s.
ARM_PAIRS = [
    ("namr_only", "csp_namr"),
    ("namr_only", "full"),
    ("namr_only", "csp_only"),
    ("namr_only", "namr_only_mps"),
]

# Display / rounding constants.
DIFF_ROUND_DIGITS = 4
JSON_INDENT = 1

# Per-pair result dict stable keys.
PAIR_SEEDS = "seeds"
PAIR_PER_SEED = "per_seed"
PAIR_ALL_POS = "all_positive"
PAIR_ALL_SAME_SIGN = "all_same_sign"
PAIR_MIN = "min"
PAIR_MAX = "max"
PAIR_MEAN = "mean"


def _vals_for_arm(per_ckpt_dict, arm):
    """Extract {seed_int: kegg_k20} for one arm, skipping missing entries.

    @brief  Internal helper: walks ``per_checkpoint`` dict, keeps entries whose
            ``f"{arm}|s"`` prefix matches, converts the seed suffix to int,
            and only retains entries whose kegg_k20 is not None (so arms with
            partially archived seed sets compare cleanly on the overlap).
    @param  per_ckpt_dict   ``payload[PER_CKPT_KEY]`` from kegg_per_seed.json.
    @param  arm             Arm label string (e.g. "namr_only").
    @return Dict ``{seed_int: kegg_value_float}``.
    """
    out = {}
    prefix = arm + "|s"
    for k, v in per_ckpt_dict.items():
        if k.startswith(prefix) and v.get(PC_KEGG_FIELD) is not None:
            seed = int(k[len(prefix):])
            out[seed] = v[PC_KEGG_FIELD]
    return out


def main():
    """Load kegg_per_seed.json, compute paired diffs, write file back in-place.

    @brief  (1) resolve the full path to results/kegg_per_seed.json and load
            the existing payload; (2) compute _vals_for_arm for each arm in
            ARM_PAIRS; (3) for each pair, intersect seeds, compute per-seed
            diff = a - b rounded to DIFF_ROUND_DIGITS, then compute min/max/
            mean / all-positive / all-same-sign aggregates; (4) attach the
            resulting block under ``payload[DIFFS_KEY]`` and overwrite the
            original JSON with JSON_INDENT indentation; (5) print an English
            per-pair summary line.
    @return None.  ``sys.exit`` is not used because the script never returns a
            non-zero status; a missing input file is surfaced as a JSON load
            exception, which is the correct behaviour for a post-processing
            step that assumes its input exists.
    """
    json_path = config.result_path(KEGG_PER_SEED_REL)
    with open(json_path) as fh:
        payload = json.load(fh)
    per_ckpt = payload[PER_CKPT_KEY]

    block = {}
    for arm_a, arm_b in ARM_PAIRS:
        va = _vals_for_arm(per_ckpt, arm_a)
        vb = _vals_for_arm(per_ckpt, arm_b)
        shared_seeds = sorted(set(va.keys()) & set(vb.keys()))
        if not shared_seeds:
            continue
        per_seed_diffs = {
            str(s): round(va[s] - vb[s], DIFF_ROUND_DIGITS) for s in shared_seeds
        }
        values = list(per_seed_diffs.values())
        signs = {v > 0 for v in values}
        block[f"{arm_a}_minus_{arm_b}"] = {
            PAIR_SEEDS: shared_seeds,
            PAIR_PER_SEED: per_seed_diffs,
            PAIR_ALL_POS: all(v > 0 for v in values),
            PAIR_ALL_SAME_SIGN: len(signs) == 1,
            PAIR_MIN: min(values),
            PAIR_MAX: max(values),
            PAIR_MEAN: round(sum(values) / len(values), DIFF_ROUND_DIGITS),
        }

    payload[DIFFS_KEY] = block
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=JSON_INDENT)

    # English-only stdout summary.
    for k, v in block.items():
        print(
            f"{k}: seeds={v[PAIR_SEEDS]} per_seed={v[PAIR_PER_SEED]} "
            f"all_positive={v[PAIR_ALL_POS]} min={v[PAIR_MIN]} max={v[PAIR_MAX]} "
            f"mean={v[PAIR_MEAN]}"
        )
    print(f"[written] {json_path}")


if __name__ == "__main__":
    main()
