#!/usr/bin/env python3
"""
@file    100_cohort_composition.py
@brief   Cohort composition and patient-node identity audit (review issue M3).

Reads the shipped graph metadata and reports, as a citable artifact:
  * per-cancer sample counts
  * sample-type (TCGA code) composition
  * whether any participant barcode contributes more than one node

Node identity in the released graph is the 15-character TCGA barcode
(participant + sample-type code); the participant prefix is its first 12 characters.

Outputs:
  results/cohort_composition.json   backs Supplementary Table S25 and the
                                    Methods cohort paragraph (regenerated,
                                    never hand-typed).

Run:
  GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/100_cohort_composition.py

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics
"""

# ============================================================================
# File-level constants
# ============================================================================

# TCGA sample-type code -> human-readable label mapping.
# Codes follow the official TCGA barcode specification.
SAMPLE_TYPE_CODE_LABELS = {
    "01": "primary solid tumour",
    "02": "recurrent solid tumour",
    "03": "primary blood-derived cancer",
    "06": "metastatic",
    "11": "solid tissue normal",
}

# Fallback label for sample-type codes not present in the official mapping.
SAMPLE_TYPE_UNMAPPED_LABEL = "unmapped"

# Patient-graph files whose metadata we inspect. Both are produced by the
# graph-construction pipeline and live under config.DATA_PROC.
GRAPH_FILE_BASENAMES = (
    "xena_graph_large.pt",
    "xena_graph_patient.pt",
)

# TCGA barcode slicing constants.
#   * participant prefix = first 12 characters (submitter + participant)
#   * full barcode length  = 15 characters
#   * sample-type code     = characters at positions 13:15 (0-based, right-exclusive)
TCGA_PARTICIPANT_PREFIX_LENGTH = 12
TCGA_FULL_BARCODE_LENGTH = 15
TCGA_SAMPLE_TYPE_CODE_START = 13
TCGA_SAMPLE_TYPE_CODE_END = 15

# Minimum samples per cancer type required by the cohort inclusion rule.
# Cited in the Methods cohort paragraph; kept as a named constant so the
# rule can be changed in a single place.
MIN_SAMPLES_PER_CANCER_RULE = 50

# Output basename for the machine-readable artifact.
OUTPUT_JSON_BASENAME = "cohort_composition.json"

# PyTorch load() options. Kept at file top so that any future move to
# weights_only=True (PyTorch 2.1+) has a single call site to audit.
TORCH_LOAD_MAP_LOCATION = "cpu"
TORCH_LOAD_WEIGHTS_ONLY = False

# JSON serialisation indentation used for the output artifact. 1 space keeps
# the file human-readable while remaining compact.
JSON_OUTPUT_INDENT = 1

# ============================================================================
# Imports
# ============================================================================
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    """
    @brief   Build and persist the cohort-composition manifest.

    Walks every patient graph listed in GRAPH_FILE_BASENAMES, extracts the
    patient-id list and per-cancer counts from its ``meta`` dict, then
    assembles:
      * id-length distribution       -> catches non-TCGA id leaks
      * 12-char participant uniques  -> population denominator
      * multi-node participants      -> reviewer flag M3
      * sample-type code counts      -> 01 primary / 11 normal / ...
      * per-cancer counts sorted     -> desc order + rule-50 threshold

    Missing graph files are silently skipped so the script can run against
    partial reproduction checkouts.

    @return  None; side effect: writes JSON to ``config.result_path(...)``.
    @note    The script does NOT fail when a graph file is absent; a
             continuous-integration checkout may legitimately only ship one
             of the two variants.
    """
    out = {}

    for graph_name in GRAPH_FILE_BASENAMES:
        graph_path = os.path.join(config.DATA_PROC, graph_name)

        # Gracefully skip graph variants that are not present in the current
        # checkout; downstream consumers simply check the dict keys.
        if not os.path.exists(graph_path):
            continue

        # weights_only=False because ``meta`` is a nested plain-python dict,
        # not a tensor; PyTorch 2.0+ would refuse to load it under the
        # default weights_only=True path.
        meta = torch.load(
            graph_path,
            map_location=TORCH_LOAD_MAP_LOCATION,
            weights_only=TORCH_LOAD_WEIGHTS_ONLY,
        )["meta"]

        patient_ids = list(meta.get("patient_ids") or [])
        cancer_counts = meta.get("cancer_counts") or {}

        record = {
            "n_nodes": len(patient_ids),
            "n_samples_meta": meta.get("n_samples"),
            # id_lengths catches any non-15-character ids that would break
            # the barcode slicing below; useful during graph rebuilds.
            "id_lengths": dict(collections.Counter(len(s) for s in patient_ids)),
            "n_unique_full_ids": len(set(patient_ids)),
            # participant prefix (12 chars) is the identity that a reviewer
            # cares about: one human being should correspond to one node.
            "n_unique_participants_12char": len({
                s[:TCGA_PARTICIPANT_PREFIX_LENGTH] for s in patient_ids
            }),
            "participants_with_multiple_nodes": {
                k: v for k, v in
                collections.Counter(
                    s[:TCGA_PARTICIPANT_PREFIX_LENGTH] for s in patient_ids
                ).items()
                if v > 1
            },
            # sample-type code occupies positions 13:15 in the 15-char TCGA
            # barcode (indices 13 and 14 inclusive).
            "sample_type_codes": {
                code: {
                    "n": n,
                    "label": SAMPLE_TYPE_CODE_LABELS.get(
                        code, SAMPLE_TYPE_UNMAPPED_LABEL
                    ),
                }
                for code, n in sorted(
                    collections.Counter(
                        s[TCGA_SAMPLE_TYPE_CODE_START:TCGA_SAMPLE_TYPE_CODE_END]
                        for s in patient_ids
                    ).items()
                )
            },
            "cancer_counts": dict(
                sorted(cancer_counts.items(), key=lambda kv: -kv[1])
            ),
            "cancer_counts_sum": sum(cancer_counts.values()) if cancer_counts else None,
            "n_cancer_types": len(cancer_counts) if cancer_counts else None,
            # The "at least 50 samples per cancer" inclusion rule, written
            # into the output so downstream readers can see the exact
            # threshold that was enforced when the cohort was built.
            "min_samples_per_cancer_rule": MIN_SAMPLES_PER_CANCER_RULE,
        }

        out[graph_name] = record

        # ---- console summary (English, machine-scannable "==" preamble) ----
        print(
            f"== {graph_name}: {record['n_nodes']} nodes | "
            f"unique participants(12ch) {record['n_unique_participants_12char']} | "
            f"participants with >1 node: "
            f"{len(record['participants_with_multiple_nodes'])}"
        )
        print(f"   sample types: {record['sample_type_codes']}")
        print(
            f"   cancer types: {record['n_cancer_types']} | "
            f"sum {record['cancer_counts_sum']}"
        )

    # config.RESULTS may be missing on trees that imported an older version
    # of config.py; fall back to $PROJECT_ROOT/results, which is the
    # canonical layout.
    config.RESULTS = getattr(
        config,
        "RESULTS",
        os.path.join(config.PROJECT_ROOT, "results"),
    )

    output_path = config.result_path(OUTPUT_JSON_BASENAME)
    with open(output_path, "w") as fh:
        json.dump(out, fh, indent=JSON_OUTPUT_INDENT)
    print("[written]", output_path)


if __name__ == "__main__":
    main()
