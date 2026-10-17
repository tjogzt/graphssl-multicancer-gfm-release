#!/usr/bin/env python3
"""
@file    101_geometry_cross_seed.py
@brief   Cross-seed singular-value spectrum diagnostic (review issue M7 follow-up).

The effective-rank / mean-cosine collapse claim for the CSP-only arm was
originally demonstrated on a single seed-42 checkpoint.  This script repeats
the *identical* diagnostic - imported verbatim from
``93_csp_geometry_shortcut.py`` so the formula cannot drift - for every
archived training seed of the two arms, and saves the per-seed spectrum
summary so collapse can be verified on a per-seed basis rather than taken on
one reading.

Inputs:
  data/processed/xena_graph_esm.pt
  per-seed checkpoints enumerated in ARMS (resolved against
  config.PROJECT_ROOT/results)

Outputs:
  results/control/geometry_cross_seed.json   per-seed + cross-seed summary

Run:
  GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/101_geometry_cross_seed.py

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics
"""

# ============================================================================
# File-level constants
# ============================================================================

# Basename of the geometry helper that supplies ``embeddings()`` and
# ``geom()``. Imported dynamically because scripts/ is not a package.
GEOMETRY_MODULE_BASENAME = "93_csp_geometry_shortcut.py"

# Spectral threshold used when dropping near-zero singular values. Same
# value is hard-coded inside 93_csp_geometry_shortcut.py itself; replicated
# here so the spectrum_summary local function can be audited without
# cross-referencing a second file.
SVD_KEEP_THRESHOLD = 1e-12

# ---- per-field rounding precision (must match the manuscript table digits) ----
EFF_RANK_ROUND_DIGITS = 4
TOP20_SHARE_ROUND_DIGITS = 5      # top-N shares keep one extra digit of detail
TOP1_SHARE_ROUND_DIGITS = 4
TOP5_SHARE_ROUND_DIGITS = 4
TOP10_SHARE_ROUND_DIGITS = 4
S_MAX_ROUND_DIGITS = 4
S_MIN_KEPT_ROUND_DIGITS = 10      # smallest kept singular value is near 1e-12
GEOMETRY_SCALAR_ROUND_DIGITS = 4  # catch-all for every field from geo.geom()

# SD convention used in the summary block - ddof=1 (sample std, denominator n-1),
# matching every other diagnostic script in the project.
SUMMARY_SD_DDOF = 1
SUMMARY_MEAN_ROUND_DIGITS = 4
SUMMARY_SD_ROUND_DIGITS = 4
SUMMARY_LIST_ROUND_DIGITS = 4

# Subdirectory + basename under $RESULTS for the output artifact.
OUTPUT_SUBDIR_PARTS = ("control",)
OUTPUT_JSON_BASENAME = "geometry_cross_seed.json"

# Printed into the payload so a reader can re-derive the formula without
# opening the helper module. Must stay byte-identical to what
# 93_csp_geometry_shortcut.py actually computes.
PROTOCOL_DESCRIPTION = (
    "identical to 93_csp_geometry_shortcut.py (column-centred SVD, "
    "raw singular values, p = s/sum(s), eff_rank = exp(-sum p log p), "
    "threshold 1e-12)"
)

# JSON indentation for the output artifact (1 space is both readable and compact).
JSON_OUTPUT_INDENT = 1

# ---- Archived per-seed checkpoints (arm -> seed -> relpath under results/) ----
# These paths are also referenced from results/checkpoint_registry.json; any
# rename must be synchronised there as well.
ARMS = {
    "csp_only": {
        42:  "ablation_real/csp_only.pt",
        7:   "ablation_real/csp_only_s7.pt",
        123: "ablation_real/csp_only_s123.pt",
    },
    "namr_only": {
        42:  "ablation_real/namr_only.pt",
        7:   "ablation_real/namr_only_s7o.pt",
        123: "server_cuda_namr5/namr_only_s123.pt",
        21:  "server_cuda_namr5/namr_only_s21.pt",
        99:  "server_cuda_namr5/namr_only_s99.pt",
    },
}

# ============================================================================
# Imports + dynamic module load
# ============================================================================
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Dynamically import the geometry helper so we call the *exact* same
# embeddings/geom code that produced Fig. S5 / shortcut.json; no copy-paste.
_HERE = os.path.dirname(os.path.abspath(__file__))
_geo_spec = importlib.util.spec_from_file_location(
    "geo",
    os.path.join(_HERE, GEOMETRY_MODULE_BASENAME),
)
geo = importlib.util.module_from_spec(_geo_spec)
_geo_spec.loader.exec_module(geo)


def spectrum_summary(X):
    """
    @brief   Column-centred SVD spectrum summary (same formula as 93_csp*).

    Steps:
      1. cast to float64
      2. subtract per-column mean (column-centred)
      3. full SVD, discard u/v
      4. drop singular values <= 1e-12 (numerical noise floor)
      5. p_i = s_i / sum(s)
      6. return counts, shares, and boundary values rounded to the
         manuscript-matching precision defined in the constants above.

    @param  X  (numpy.ndarray)  n_sample x n_dim embedding matrix.
    @return dict with keys:
              n_singular, eff_rank, top20_shares, top1_share,
              top5_share, top10_share, s_max, s_min_kept
    @note   eff_rank = exp(-sum_i p_i log p_i) is the continuous effective
            rank introduced by Roy & Vetterli (2007); it is strictly smaller
            than the count of kept singular values and penalises spectra
            that concentrate in one direction.
    """
    # Work in float64 so the SVD of 5,000 x 256 matrices is numerically
    # stable; downstream rounding brings us back to the reported precision.
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(0)

    s = np.linalg.svd(Xc, compute_uv=False)
    # 1e-12 is the same floor used inside 93_csp_geometry_shortcut.py so
    # n_singular between the two scripts must match exactly.
    s = s[s > SVD_KEEP_THRESHOLD]
    p = s / s.sum()

    return {
        "n_singular": int(s.size),
        # Exponential of the negative Shannon entropy -> effective rank.
        # Rounded to 4 digits (manuscript: "eff. rank 38.2").
        "eff_rank": round(
            float(np.exp(-(p * np.log(p)).sum())),
            EFF_RANK_ROUND_DIGITS,
        ),
        # top-20 kept as a list with 5-digit precision so the reader can
        # eyeball the spectrum tail without the full array.
        "top20_shares": [
            round(float(x), TOP20_SHARE_ROUND_DIGITS) for x in p[:20]
        ],
        "top1_share": round(float(p[0]), TOP1_SHARE_ROUND_DIGITS),
        "top5_share": round(float(p[:5].sum()), TOP5_SHARE_ROUND_DIGITS),
        "top10_share": round(float(p[:10].sum()), TOP10_SHARE_ROUND_DIGITS),
        "s_max": round(float(s[0]), S_MAX_ROUND_DIGITS),
        # Smallest KEPT singular value; reports where the numerical floor
        # bit us so future readers can retune SVD_KEEP_THRESHOLD.
        "s_min_kept": round(float(s[-1]), S_MIN_KEPT_ROUND_DIGITS),
    }


def main():
    """
    @brief   Loop over arms -> seeds, collect geom + spectrum, write payload.

    Per seed:
      * geo.embeddings(path) -> raw gene embedding matrix
      * geo.geom(X)          -> {mean_pairwise_cosine, effective_rank, ...}
      * update with spectrum_summary(X) -> singular-value spectrum report

    Per arm: aggregate eff_rank, mean_pairwise_cosine, top1/top5 share
    across seeds with the sample-SD convention (ddof=1) when more than one
    seed is present.

    @return  None. Writes ``results/control/geometry_cross_seed.json``.
    @note    Seeds whose checkpoint file is absent on disk are printed with
             the "MISSING" prefix and simply omitted from the payload; the
             script does NOT hard-fail so partial checkouts can still
             summarise the seeds they have.
    """
    per_seed_out = {}

    for arm_name, seed_map in ARMS.items():
        per_seed_out[arm_name] = {}
        for seed, rel_path in seed_map.items():
            full_path = os.path.join(config.PROJECT_ROOT, "results", rel_path)
            if not os.path.exists(full_path):
                print(f"  MISSING {rel_path}")
                continue

            X = geo.embeddings(full_path)
            geom_record = geo.geom(X)
            geom_record.update(spectrum_summary(X))

            # geo.geom() mixes ints/floats/numpy scalars; coerce to plain
            # python floats rounded to GEOMETRY_SCALAR_ROUND_DIGITS so the
            # JSON artifact is byte-stable across numpy/PyTorch versions.
            per_seed_out[arm_name][seed] = {
                k: (
                    round(float(v), GEOMETRY_SCALAR_ROUND_DIGITS)
                    if isinstance(v, (int, float, np.floating))
                    else v
                )
                for k, v in geom_record.items()
            }

            print(
                f"  {arm_name:10s} seed {seed:3d}: "
                f"eff_rank={geom_record['effective_rank']:.4f} "
                f"mean_cos={geom_record['mean_pairwise_cosine']:.3f} "
                f"top1_share={geom_record['top1_share']:.3f} "
                f"top5_share={geom_record['top5_share']:.3f}"
            )

    # ---------- cross-seed summary block ----------
    arm_summary = {}
    for arm_name, per_seed_dict in per_seed_out.items():
        eff_rank_values = [v["eff_rank"] for v in per_seed_dict.values()]
        mean_cos_values = [
            float(v.get("mean_pairwise_cosine", np.nan))
            for v in per_seed_dict.values()
        ]

        arm_summary[arm_name] = {
            "seeds": sorted(int(k) for k in per_seed_dict),
            "eff_rank_values": [
                round(float(x), SUMMARY_LIST_ROUND_DIGITS) for x in eff_rank_values
            ],
            "eff_rank_mean": round(
                float(np.mean(eff_rank_values)), SUMMARY_MEAN_ROUND_DIGITS
            ),
            # ddof=1 (sample std) only when more than one seed exists;
            # single-seed arms expose None instead of pretending 0 variance.
            "eff_rank_sd": (
                round(float(np.std(eff_rank_values, ddof=SUMMARY_SD_DDOF)),
                      SUMMARY_SD_ROUND_DIGITS)
                if len(eff_rank_values) > 1 else None
            ),
            "mean_pairwise_cosine_values": [
                round(float(x), SUMMARY_LIST_ROUND_DIGITS) for x in mean_cos_values
            ],
            "mean_pairwise_cosine_mean": round(
                float(np.mean(mean_cos_values)), SUMMARY_MEAN_ROUND_DIGITS
            ),
            "mean_pairwise_cosine_sd": (
                round(float(np.std(mean_cos_values, ddof=SUMMARY_SD_DDOF)),
                      SUMMARY_SD_ROUND_DIGITS)
                if len(mean_cos_values) > 1 else None
            ),
            "top1_share_values": [
                round(float(v["top1_share"]), SUMMARY_LIST_ROUND_DIGITS)
                for v in per_seed_dict.values()
            ],
            "top5_share_values": [
                round(float(v["top5_share"]), SUMMARY_LIST_ROUND_DIGITS)
                for v in per_seed_dict.values()
            ],
        }

    payload = {
        "protocol": PROTOCOL_DESCRIPTION,
        "per_seed": per_seed_out,
        "summary": arm_summary,
    }

    output_path = config.result_path(*OUTPUT_SUBDIR_PARTS, OUTPUT_JSON_BASENAME)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=JSON_OUTPUT_INDENT)

    # ---- human-readable console summary ----
    print()
    for arm_name, s in arm_summary.items():
        print(
            f"{arm_name}: n={len(s['seeds'])} eff_rank {s['eff_rank_values']} "
            f"mean {s['eff_rank_mean']} sd {s['eff_rank_sd']} | "
            f"mean_cosine {s['mean_pairwise_cosine_values']} "
            f"mean {s['mean_pairwise_cosine_mean']} | "
            f"top1 {s['top1_share_values']} | top5 {s['top5_share_values']}"
        )
    print("[written]", output_path)


if __name__ == "__main__":
    main()
