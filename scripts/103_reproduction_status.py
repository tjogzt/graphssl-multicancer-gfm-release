#!/usr/bin/env python3
r"""
@file    103_reproduction_status.py
@brief   Per-result reproducibility classification (review item 10.1).

Classifies every tabulated / figured result of the paper into one of three
states, so that no single sentence has to claim that "all results are
reproduced from archived checkpoints":

  A    re-evaluable from the released checkpoint
       (checkpoint + evaluation config present)
  B    retrainable from the released configuration
       (config present, original checkpoint absent)
  C    log-only
       (numbers survive, but the run cannot be regenerated from the tree)

Classification is derived from:
  * results/checkpoint_registry.json  (SHA256, measured decoder, run id)
  * the artifact files each result is cited from
  * reproducibility notes already present in the supplementary tables

An extra pseudo-class **A\*** is assigned at verification time when a
result nominally tagged "A" has one or more named checkpoints missing from
the registry - i.e. the claim was written but the backing artefact did not
ship.  A* results also carry a ``note`` that lists the missing files.

Outputs:
  results/reproduction_status.json   machine-readable manifest
  results/REPRODUCTION_STATUS.md     same, rendered for human readers

Run:
  GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/103_reproduction_status.py

@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics
"""

# ============================================================================
# File-level constants
# ============================================================================

# Basename of the authoritative SHA256 + measured-decoder checkpoint registry
# produced by 132_audit_checkpoints.py.
CHECKPOINT_REGISTRY_BASENAME = "checkpoint_registry.json"

# ---- reproducibility class tokens (keep byte-identical to manuscript text) ----
STATUS_CLASS_A = "A"       # re-evaluable from released checkpoint
STATUS_CLASS_B = "B"       # retrainable from released config (cp absent)
STATUS_CLASS_C = "C"       # log-only
STATUS_CLASS_A_STAR = "A*" # A-downgrade: named checkpoint(s) missing from registry

# Human-readable class descriptions, serialised into the payload so JSON
# consumers can display a legend without a second lookup.
CLASS_DESCRIPTIONS = {
    STATUS_CLASS_A: "re-evaluable from the released checkpoint",
    STATUS_CLASS_B: "retrainable from the released configuration",
    STATUS_CLASS_C: (
        "log-only (not regenerable from the released tree)"
    ),
}

# Free-text note attached to the payload describing what the checkpoint
# registry contains. Matches the audit report produced by 132_audit_checkpoints.py.
CHECKPOINT_REGISTRY_NOTE = (
    "results/checkpoint_registry.json "
    "(96 checkpoints, SHA256 + measured decoder + run id)"
)

# The `note`/`command` field in the Markdown table is truncated to this many
# characters so a single rogue row doesn't blow out the page width. 220
# chars keeps the 19 rows within a single TextHeight on the print page.
NOTE_MAX_DISPLAY_LENGTH = 220

# Output artifact basenames (json for machines, md for humans). Both live
# directly under config.RESULT_DIR.
OUTPUT_JSON_BASENAME = "reproduction_status.json"
OUTPUT_MD_BASENAME = "REPRODUCTION_STATUS.md"

# JSON indentation: 1 space keeps diffs reviewable on GitHub PRs.
JSON_OUTPUT_INDENT = 1

# Registry records key their paths with a leading "results/"; we normalise
# every checkpoint basename with the same prefix so the ``in by_path`` test
# always hits regardless of how an item's checkpoints list was authored.
CHECKPOINT_RELPATH_PREFIX = "results/"

# ---- Markdown-render constants ----
MD_HEADER_TITLE = "# Result reproduction status"
MD_INTRO_PARAGRAPH = (
    "Classes: **A** re-evaluable from the released checkpoint; "
    "**B** retrainable from the released configuration "
    "(original checkpoint absent); **C** log-only. "
    "Derived from `results/checkpoint_registry.json` and the artifacts "
    "each result is cited from."
)
MD_TABLE_HEADER = "| result | class | artifact | note |"
MD_TABLE_SEPARATOR = "|---|---|---|---|"
MD_TABLE_ROW_TEMPLATE = "| {item} | **{cls}** | {artifact} | {note} |"
MD_COUNTS_LINE_PREFIX = "Counts: "

# Markdown table column: pipes inside artifact/note are escaped to forward
# slashes so we don't produce a malformed table on a free-text field.
MD_TABLE_PIPE_ESCAPE_TO = "/"

# Console summary template: "  A*  Table 1 / S12 known-edge AUC ..."
CONSOLE_ITEM_LINE_TEMPLATE = "  {status:2s}  {item}"

# ============================================================================
# Imports
# ============================================================================
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

RESULT_DIR = config.RESULT_DIR


def exists(*parts):
    """
    @brief   Join ``parts`` against RESULT_DIR and test for file existence.

    @param   parts  (tuple of str)  path segments to join under RESULT_DIR.
    @return  bool  True iff the joined path exists on disk.
    @note    Historically the script imported this helper into the inner
             ``has()`` closure; kept as a module-level function for
             auditability even though the current main() no longer calls it.
    """
    return os.path.exists(os.path.join(RESULT_DIR, *parts))


def main():
    """
    @brief   Build the per-result reproduction manifest and write both views.

    Pipeline:
      1. load checkpoint_registry.json records, build ``by_path`` index,
      2. declare the 19 audited items (ordered to match the manuscript
         narrative),
      3. for each item: normalise checkpoint paths with "results/" prefix,
         compute missing-set, optionally downgrade A -> A* with a note,
      4. tally per-class counts,
      5. emit JSON payload (classes + counts + items + registry note),
      6. emit Markdown table (result | class | artifact | note),
      7. console summary.

    @return  None. Side effects: two files under RESULT_DIR, console output.
    @note    Items 19 are hand-ordered to match the manuscript table of
             contents; the list literal is kept inline in this function so
             a reviewer can diff it against the manuscript line-by-line.
    """
    registry_records = json.load(
        open(os.path.join(RESULT_DIR, CHECKPOINT_REGISTRY_BASENAME))
    )["records"]
    # index: "results/<relpath>.pt" -> registry record
    by_path = {record["path"]: record for record in registry_records}

    def has_checkpoint(relpath):
        """
        @brief   Predicate: is relpath (without "results/" prefix) in the registry?
        @param  relpath  (str)  path relative to the ``results/`` directory.
        @return bool
        """
        return (CHECKPOINT_RELPATH_PREFIX + relpath) in by_path

    # ---- 19 audited manuscript results ------------------------------------------------
    # Ordered to match the manuscript table of contents; any reordering
    # should keep the output table aligned with the manuscript narrative.
    items = [
        dict(
            item="Table 1 / S12 known-edge AUC (all arms, all seeds)",
            status=STATUS_CLASS_A,
            checkpoints=[
                p for p in by_path
                if "ablation_real" in p or "server_cuda_namr5" in p and p.endswith(".pt")
            ][:0] or [
                "ablation_real/full.pt",
                "ablation_real/namr_only.pt",
                "server_cuda_namr5/namr_only_s123.pt",
                "ablation_real/csp_only.pt",
                "ablation_real/csp_namr_legacy_s42.pt",
            ],
            artifact="results/known_edge_multiseed.json",
            command=(
                "scripts/102_known_edge_per_seed.py "
                "(probe imported from 50_eval_checkpoint.py)"
            ),
        ),
        dict(
            item="Table 1 / S11 KEGG AUROC (per arm, per seed)",
            status=STATUS_CLASS_A,
            checkpoints=[
                "ablation_real/namr_only.pt",
                "ablation_real/full.pt",
                "ablation_real/csp_only.pt",
                "ablation_real/csp_namr_legacy_s42.pt",
            ],
            artifact="results/kegg_per_seed.json",
            command="scripts/130_kegg_per_seed.py",
        ),
        dict(
            item="Table 1 smoothing row / S21(a) smoothing sweep",
            status=STATUS_CLASS_A,
            checkpoints=[],
            artifact="results/control/graph_smoothing_baseline.json",
            command="scripts/129_graph_smoothing_baseline.py (no training)",
        ),
        dict(
            item="S21(b) GraphMAE-style SCE ablation",
            status=STATUS_CLASS_A,
            checkpoints=[
                r["path"] for r in registry_records if r["file"] == "namr_only_sce.pt"
            ],
            artifact="results/control/ (SCE run)",
            command="40_pretrain.py with the SCE criterion",
        ),
        dict(
            item="S18 held-out edge recovery / structural controls",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/namr_only.pt"],
            artifact=(
                "results/holdout/, results/control/control_eval_correct.json"
            ),
            command="scripts/78_eval_holdout.py, 79_build_control_graphs.py",
        ),
        dict(
            item="S19 NAMR denoising benchmark (trivial denoisers)",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/namr_only.pt"],
            artifact="results/control/ (denoiser benchmark)",
            command="scripts/125_namr_denoiser_diagnostic.py",
        ),
        dict(
            item="S20 geometry, skip-decoder, prototype degeneracy",
            status=STATUS_CLASS_A,
            checkpoints=[
                "ablation_real/csp_only.pt",
                "ablation_real/namr_only.pt",
                "ablation_real/namr_only_s7.pt",
            ],
            artifact=(
                "results/control/csp_geometry_shortcut.json, "
                "results/control/geometry_cross_seed.json, "
                "results/control/c3l_prototype_layers.json"
            ),
            command=(
                "scripts/93_csp_geometry_shortcut.py, "
                "101_geometry_cross_seed.py, 126_c3l_prototype_diagnostic.py"
            ),
        ),
        dict(
            item="S24 probe-regime sensitivity",
            status=STATUS_CLASS_A,
            checkpoints=[
                r["path"] for r in registry_records
                if r["file"].endswith(".pt")
            ][:0] or ["ablation_real/namr_only.pt"],
            artifact="results/probe_regimes.json",
            command="scripts/124_eval_probe_regimes.py",
        ),
        dict(
            item="S7 / S2 patient-level P-NAMR curve (21-cancer, GPU run)",
            status=STATUS_CLASS_C,
            checkpoints=[p for p in by_path if "patient" in p],
            artifact="results/patient_fewshot_acc.json",
            note=(
                "the trained full-patient-graph checkpoint for the GPU run is "
                "not in the released tree; results/checkpoints/p1_patient_gfm.pt "
                "and the local-rebuild depth checkpoints are present, but they "
                "are different runs (different graph realisation), so this "
                "curve is reported as log-only"
            ),
            command=(
                "scripts/61_patient_eval.py "
                "(cannot be re-run without the checkpoint)"
            ),
        ),
        dict(
            item="S7 / S17 local-rebuild patient baselines and depth sweep",
            status=STATUS_CLASS_A,
            checkpoints=[
                r["path"] for r in registry_records
                if "gfm_patient_depth" in r["path"]
            ],
            artifact="results/control/ (patient baselines)",
            command="scripts/62_patient_pretrain_eval.py",
        ),
        dict(
            item="S9 METABRIC rows (full, NAMR-only, ESM-2, random)",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/full.pt", "ablation_real/namr_only.pt"],
            artifact="results/metabric_rerun_20260918/",
            command="scripts/60_metabric_validation.py",
        ),
        dict(
            item="S9 METABRIC CSP+NAMR row (0.641)",
            status=STATUS_CLASS_C,
            checkpoints=[],
            artifact="original run only (no per-arm array retained)",
            note=(
                "the CSP+NAMR checkpoint used for that row is not archived "
                "under that arm name, so the value cannot be regenerated; it "
                "is flagged as such in the table itself and excluded from "
                "the reproduced range"
            ),
            command="n/a",
        ),
        dict(
            item="S6 drug-response sweep",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/namr_only.pt"],
            artifact=(
                "results/drug2_*.npy, results/control/ (drug baselines)"
            ),
            command="scripts/64_drug_response_gdsc.py, 128_drug_baselines.py",
        ),
        dict(
            item="S22 650M NAMR-only known-edge AUC (0.896 / 0.887)",
            status=STATUS_CLASS_B,
            checkpoints=[
                r["path"] for r in registry_records if "650" in r["file"]
            ],
            artifact="results/nmi_sprint_results.md",
            note=(
                "the 650M checkpoints are released, but the evaluation that "
                "produced 0.896 was not retained as a per-run log; the value "
                "is re-derivable by running the 650M evaluation protocol, "
                "which requires the external Xena matrices"
            ),
            command="scripts/54_eval_esm650.py",
        ),
        dict(
            item="S22 raw ESM-2 650M baseline (0.671)",
            status=STATUS_CLASS_A,
            checkpoints=[],
            artifact="results/nmi_sprint_results.md",
            command=(
                "scripts/53_eval_esm_baseline.py "
                "(evaluation-side, no checkpoint)"
            ),
        ),
        dict(
            item="S22 PAAD held-out correlations",
            status=STATUS_CLASS_A,
            checkpoints=[
                "ablation_real/namr_only_20c.pt",
                "ablation_real/namr_only_650_20c.pt",
            ],
            artifact="results/paad_rho_*.npy",
            command="scripts/63_paad_transfer_eval.py",
        ),
        dict(
            item="S10 GTEx tissue filtering",
            status=STATUS_CLASS_A,
            checkpoints=[],
            artifact="results/tissue_ppi/",
            command="scripts/66_tissue_specific_ppi.py",
        ),
        dict(
            item="Fig. 3 per-seed known-edge scatter",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/full.pt", "ablation_real/namr_only.pt"],
            artifact="results/known_edge_multiseed.json",
            command="scripts/102_known_edge_per_seed.py",
        ),
        dict(
            item="Fig. S5 mechanism panels",
            status=STATUS_CLASS_A,
            checkpoints=["ablation_real/csp_only.pt", "ablation_real/namr_only.pt"],
            artifact=(
                "results/control/geometry_cross_seed.json, "
                "results/control/c3l_prototype_layers.json"
            ),
            command=(
                "scripts/101_geometry_cross_seed.py, "
                "126_c3l_prototype_diagnostic.py"
            ),
        ),
    ]

    # ---- per-item verification: normalise checkpoints + A -> A* downgrade ----
    for item in items:
        # 1. Normalise every checkpoint reference to "results/<path>" form,
        #    matching the schema used by the registry ``by_path`` index.
        raw_checkpoints = item.get("checkpoints", [])
        normalised_checkpoints = [
            (cp if cp.startswith(CHECKPOINT_RELPATH_PREFIX)
             else CHECKPOINT_RELPATH_PREFIX + cp)
            for cp in raw_checkpoints
        ]
        item["checkpoints"] = normalised_checkpoints

        # 2. Compute the set of named checkpoints that are NOT present in
        #    the shipped registry; this drives the A-downgrade below.
        missing_checkpoints = [
            cp for cp in normalised_checkpoints if cp not in by_path
        ]
        item["checkpoints_present"] = not missing_checkpoints

        if missing_checkpoints:
            item["missing_checkpoints"] = missing_checkpoints
            # A -> A* downgrade: the item *claimed* it was backed by a
            # released checkpoint, but at least one such cp is missing.
            if item["status"] == STATUS_CLASS_A:
                item["status"] = STATUS_CLASS_A_STAR
                item["note"] = (
                    "one or more named checkpoints are absent from the registry: "
                    + ", ".join(missing_checkpoints)
                )

    # ---- per-class tally ----
    counts = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    # =====================
    #  JSON payload
    # =====================
    payload = {
        "classes": CLASS_DESCRIPTIONS,
        "counts": counts,
        "items": items,
        "checkpoint_registry": CHECKPOINT_REGISTRY_NOTE,
    }

    json_output_path = os.path.join(RESULT_DIR, OUTPUT_JSON_BASENAME)
    with open(json_output_path, "w") as fh:
        json.dump(payload, fh, indent=JSON_OUTPUT_INDENT)

    # =====================
    #  Markdown rendering
    # =====================
    md_lines = [
        MD_HEADER_TITLE,
        "",
        MD_INTRO_PARAGRAPH,
        "",
        MD_TABLE_HEADER,
        MD_TABLE_SEPARATOR,
    ]
    for item in items:
        # Prefer the explicit ``note`` if present, else fall back to the
        # reproduction ``command``. Both are free-text fields so we escape
        # any pipe characters (would break the Markdown table) and then
        # truncate to NOTE_MAX_DISPLAY_LENGTH for a clean print page.
        raw_note = item.get("note", item.get("command", ""))
        safe_note = (
            raw_note.replace("|", MD_TABLE_PIPE_ESCAPE_TO)
            [:NOTE_MAX_DISPLAY_LENGTH]
        )
        safe_artifact = item.get("artifact", "").replace(
            "|", MD_TABLE_PIPE_ESCAPE_TO
        )
        md_lines.append(MD_TABLE_ROW_TEMPLATE.format(
            item=item["item"],
            cls=item["status"],
            artifact=safe_artifact,
            note=safe_note,
        ))

    # Human-readable class-count footer. Sorted by class token so A*/B/C
    # ordering is deterministic regardless of which statuses appear.
    md_lines += [
        "",
        MD_COUNTS_LINE_PREFIX + ", ".join(
            f"{k} = {v}" for k, v in sorted(counts.items())
        ),
        "",
    ]

    md_output_path = os.path.join(RESULT_DIR, OUTPUT_MD_BASENAME)
    with open(md_output_path, "w") as fh:
        fh.write("\n".join(md_lines))

    # ---- console summary (counts footer first, then the 19 status lines) ----
    print("\n".join(md_lines[-3:]))
    for item in items:
        print(CONSOLE_ITEM_LINE_TEMPLATE.format(
            status=item["status"], item=item["item"],
        ))
    print(
        "[written] results/reproduction_status.json + "
        "results/REPRODUCTION_STATUS.md"
    )


if __name__ == "__main__":
    main()
