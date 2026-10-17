#!/usr/bin/env python3
# =============================================================================
# @file    tests/test_figure_data.py
# @brief   Figure-review R2 regression tests: every plotted value returned by
#          figure_data.py must still match the authoritative on-disk artifacts
#          (ablation_real/eval_summary.txt, provenance JSONs and the Fig. S3
#          skip-decoder diagnostic output). 29 independent assertions.
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun,
#          Tao Zhu
# @date    2026-09-27
# @ingroup tests_and_quality_gates
#
# Input:  config.RESULT_DIR tree (Zenodo deposit, gitignored).
# Output: stdout PASS/FAIL summary + process exit code (0 on all 29 pass).
# Run:    python3 tests/test_figure_data.py
# =============================================================================
"""Figure-review R2 regression tests (items I2): figure data must match the
authoritative artifacts (eval_summary.txt, figS3 provenance JSON).

Run:  python3 tests/test_figure_data.py   (needs project results/ next to repo)
"""
import json
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
import figure_data  # noqa: E402

# project root = directory that contains results/ablation_real/eval_summary.txt
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = None
for cand in (os.environ.get("GFM_PROJECT_ROOT"),
             os.path.join(_HERE, "..", ".."),
             os.path.join(_HERE, "..", "..", "..")):
    if cand and os.path.exists(os.path.join(cand, "results/ablation_real/eval_summary.txt")):
        _PROJ = cand
        break

_fail = []


def check(name, cond, detail=""):
    if not cond:
        _fail.append(f"{name}: {detail}")


def run():
    if _PROJ is None:
        print("SKIP: results/ablation_real/eval_summary.txt not found next to repo")
        return
    ev = figure_data.load_eval_summary(os.path.join(_PROJ, "results/ablation_real/eval_summary.txt"))

    # authoritative post-audit values (Table S1 / eval_summary)
    expect = {
        "full":            (0.8519, 0.0193, 0.5752),
        "namr_only":       (0.8025, 0.0160, 0.8578),
        "csp_only":        (0.8466, 0.0234, 0.5237),
        "csp_namr":        (0.8799, 0.0126, 0.7136),
        "random_init_hgt": (0.4882, 0.0220, 0.6625),
        "esm_only":        (0.7049, 0.0183, 0.6221),
        "pca_input":       (0.4648, 0.0417, 0.7048),
    }
    for tag, (acc, std, auc) in expect.items():
        check(f"eval[{tag}] exists", tag in ev, f"missing {tag}")
        if tag in ev:
            check(f"eval[{tag}].acc", abs(ev[tag]["acc"] - acc) < 1e-9, f"{ev[tag]['acc']} != {acc}")
            check(f"eval[{tag}].std", abs(ev[tag]["std"] - std) < 1e-9, f"{ev[tag]['std']} != {std}")
            check(f"eval[{tag}].auc", abs(ev[tag]["auc"] - auc) < 1e-9, f"{ev[tag]['auc']} != {auc}")

    # figS3 provenance JSON must be consistent with eval_summary where they overlap
    prov_path = os.path.join(_PROJ, "results/figS3_auc_provenance.json")
    check("figS3 provenance exists", os.path.exists(prov_path), prov_path)
    if os.path.exists(prov_path):
        with open(prov_path) as fh:
            prov = json.load(fh)
        b = prov["bars"]
        check("figS3 plain(42) == namr_only auc",
              abs(b["plain NAMR\n(seed 42)"] - ev["namr_only"]["auc"]) < 1e-9)
        check("figS3 random == random_init_hgt auc",
              abs(b["random-init\nHGT"] - ev["random_init_hgt"]["auc"]) < 1e-9)
        check("figS3 esm == esm_only auc",
              abs(b["ESM-2\n(no graph)"] - ev["esm_only"]["auc"]) < 1e-9)
        check("figS3 plain(7) == 0.8262", abs(b["plain NAMR\n(seed 7)"] - 0.8262) < 1e-9)
        check("figS3 skip(7) == 0.6012", abs(b["skip NAMR\n(seed 7)"] - 0.6012) < 1e-9)

    # patient provenance JSON
    pat_path = os.path.join(_PROJ, "results/patient_fewshot_acc.json")
    check("patient json exists", os.path.exists(pat_path), pat_path)
    if os.path.exists(pat_path):
        with open(pat_path) as fh:
            p = json.load(fh)
        check("patient K list", p["K"] == [1, 2, 5, 10, 20, 30], str(p["K"]))
        check("patient acc list", p["acc"] == [0.238, 0.302, 0.390, 0.459, 0.520, 0.555], str(p["acc"]))

    if _fail:
        print(f"FAIL ({len(_fail)}):")
        for f in _fail:
            print("  -", f)
        sys.exit(1)
    print(f"OK: {sum(1 for _ in expect) * 3 + 5 + 3} checks passed (project root: {_PROJ})")


if __name__ == "__main__":
    run()
