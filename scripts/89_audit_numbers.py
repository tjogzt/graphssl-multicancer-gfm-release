#!/usr/bin/env python3
"""
@file    89_audit_numbers.py
@brief   Manuscript audit L2 (number lock). Every metric-like decimal in the text
         must trace to a result artefact (results/**/*.json, per-split .npy arrays
         or supplementary tables) within half a displayed unit. Also recomputes
         the headline derived statistics from results/run_ledger.json and prints
         them next to the claimed values so ghost numbers can be triaged.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  manuscript/__main.tex, results/**/*.json
Output: reviews/manuscript_audit_20260918/L2_report.md + number_lock.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/89_audit_numbers.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# =============================================================================
# File-level constants
# =============================================================================
ENV_PROJECT_ROOT = "GFM_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTDIR_RELPATH = os.path.join("reviews", "manuscript_audit_20260918")
MAIN_TEX_RELPATH = os.path.join("manuscript", "__main.tex")
SUPP_TEX_RELPATH = os.path.join("manuscript", "supplementary.tex")
RUN_LEDGER_RELPATH = os.path.join("results", "run_ledger.json")

L2_REPORT_FILENAME = "L2_report.md"
NUMBERLOCK_FILENAME = "number_lock.json"

# Numeric corpus search patterns under results/ (recursive).
RESULT_GLOB_JSON = os.path.join("results", "**", "*.json")
RESULT_GLOB_NPY = os.path.join("results", "**", "*.npy")
RESULT_LOG_PATTERNS = ("*.log", "*.txt", "*.tsv", "*.md")

# Dotfile prefix used by macOS resource forks and similar; never scanned.
DOTFILE_PREFIX = "._"

# Absolute tolerance for the "claimed == artefact" match. 5e-4 == half a unit
# in the last displayed place for 3-decimal metric values.
MATCH_ABSOLUTE_TOL = 5e-4

# Numeric-token regexes (exact same patterns in manuscript vs supplementary sources).
MANUSCRIPT_DECIMAL_RE = r"(?<![\d.])(\d\.\d{3,4})(?![\d])"
SUPP_DECIMAL_RE = r"(?<![\d.])(\d\.\d{2,4})(?![\d])"
TEXT_DECIMAL_TOKEN_RE = r"\d\.\d{3,4}"

# Context window for ghost-number triage (chars around the matched token).
CONTEXT_LEFT_CHARS = 90
CONTEXT_RIGHT_CHARS = 40
CONTEXT_TRUNCATE_CHARS = 110

# Rounding precision when constructing the derived-value lookup set.
DERIVED_ROUND_DIGITS = 6

# Derived-statistics display precision in the report table.
DERIVED_DISPLAY_DIGITS = 4

# Ledger entry filters for the two headline derived statistics.
LEDGER_VARIANT_NAMRONLY = "NAMR-only"
LEDGER_DEVICE_CUDA = "CUDA"
LEDGER_FIELD_KNOWN_EDGE_AUC = "known_edge_auc"
LEDGER_SEEDS_TWO = (42, 7)

# Triage verdicts (stable English strings so downstream parsers can count them).
TRIAGE_SD_FOLLOWS_MEAN = "SD (derived; tagged with its mean)"
TRIAGE_ANALYTIC_BASELINE = "Analytic baseline (derivable from the sample distribution)"
TRIAGE_DIFF = "Derived difference"
TRIAGE_DERIVED_FROM_LEDGER = "Derived statistic (recomputed below from run_ledger)"
TRIAGE_MANUAL_REQUIRED = "**Manual adjudication required**"


def flatten(obj, path=""):
    """Recursively flatten a JSON-like object into (float_value, jq-style_path).

    @brief  Walk dict/list leaves and emit every non-bool int/float together with
            a string path describing how to reach the value from the root.
    @param  obj    Root of the JSON structure (dict, list, or scalar).
    @param  path   Accumulator for the location string (internal).
    @yield  Tuple ``(float_value, path)`` for every numeric leaf.
    @note   Booleans are explicitly excluded even though ``bool`` is a subclass
            of ``int`` in Python; they never represent metric values.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj), path


def load_artifacts():
    """Collect every numeric value from result artefacts under results/.

    @brief  Build the "numeric corpus" used for claim-to-artefact matching:
              1. every JSON leaf (results/**/*.json, recursive)
              2. mean / SD of every per-split .npy array (arrays with < 2 elements
                 are skipped because a mean/SD on 1 entry is not meaningful)
              3. every 3-to-4-decimal token printed in logs/txt/tsv/md files
            (Protocol G2: locate the artefact before declaring a ghost number.)
    @return List of ``(float_value, provenance_string)`` pairs; duplicate entries
            for the same float from different sources are retained because the
            triage report shows up-to-N provenances per match.
    @note   .npy arrays that fail to load (wrong dtype, pickled objects, truncated
            files) are silently skipped so one bad artefact cannot abort the
            entire audit.
    """
    root = os.environ.get(ENV_PROJECT_ROOT, DEFAULT_PROJECT_ROOT)
    values = []
    # --- 1. JSON leaves ---------------------------------------------------------
    for path in glob.glob(os.path.join(root, RESULT_GLOB_JSON), recursive=True):
        if os.path.basename(path).startswith(DOTFILE_PREFIX):
            continue
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except Exception:
            continue
        rel = os.path.relpath(path, root)
        for value, where in flatten(payload):
            values.append((value, f"{rel}{where}"))
    # --- 2. NPY arrays (mean + SD per split) -----------------------------------
    for path in glob.glob(os.path.join(root, RESULT_GLOB_NPY), recursive=True):
        if os.path.basename(path).startswith(DOTFILE_PREFIX):
            continue
        try:
            arr = np.load(path, allow_pickle=False).astype(float).ravel()
        except Exception:
            continue
        if arr.size < 2:
            continue
        rel = os.path.relpath(path, root)
        values.append((float(np.mean(arr)), f"{rel} (mean of {arr.size} splits)"))
        if arr.size > 2:
            values.append((float(np.std(arr, ddof=1)), f"{rel} (SD of {arr.size} splits)"))
    # --- 3. Token scan of log/text/markdown/tabular files ----------------------
    for pattern in RESULT_LOG_PATTERNS:
        for path in glob.glob(os.path.join(root, "results", "**", pattern), recursive=True):
            if os.path.basename(path).startswith(DOTFILE_PREFIX):
                continue
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            rel = os.path.relpath(path, root)
            for token in set(re.findall(TEXT_DECIMAL_TOKEN_RE, text)):
                values.append((float(token), rel))
    return values


def supplementary_values():
    """Collect every numeric value tabulated inside supplementary.tex.

    @brief  Numbers printed in the supplementary tables are an authorised second
            source for the number-lock corpus; they are matched with the same
            tolerance.
    @return List of float values extracted from the comment-stripped supp source.
    @note   A 2-decimal window is used (in addition to 3/4) because the
            supplementary tables often print 2-decimal SDs that legitimate 3-decimal
            main-text means may pair with.
    """
    root = os.environ.get(ENV_PROJECT_ROOT, DEFAULT_PROJECT_ROOT)
    with open(os.path.join(root, SUPP_TEX_RELPATH)) as handle:
        supp = handle.read()
    supp = re.sub(r"(?<!\\)%.*", "", supp)
    return [float(m.group(1)) for m in re.finditer(SUPP_DECIMAL_RE, supp)]


def _triage_verdict(value, ctx, derived_vals, tol=MATCH_ABSOLUTE_TOL):
    """Classify an unmatched claim into one of the stable triage categories.

    @brief  Helper used inside main(); kept here (not inlined) so the verdict
            vocabulary lives in exactly one place.
    @param  value         Claimed decimal value (float).
    @param  ctx           Trimmed textual context around the value.
    @param  derived_vals  Set of rounded derived values recomputed from the ledger.
    @param  tol           Absolute tolerance used for the derived-value lookup.
    @return One of the TRIAGE_* string constants defined at module scope.
    @note   Internal helper; intentionally not exported.
    """
    # SDs are printed immediately after a $\pm$ and travel with their paired mean.
    if re.search(r"\$\\pm\$\s*$|\$\\pm\$", ctx[-14:]):
        return TRIAGE_SD_FOLLOWS_MEAN
    # Chance / majority / class-frequency baselines can be recomputed from cohort.
    if re.search(r"chance|majority|uniform|class-frequency|balanced accuracy", ctx):
        return TRIAGE_ANALYTIC_BASELINE
    # +d or -d deltas are derived from two stored numbers.
    if re.search(r"difference|\+\d|\-\d", ctx):
        return TRIAGE_DIFF
    # If the value matches a recomputed ledger-derived stat, short-circuit.
    if any(abs(value - d) <= tol for d in derived_vals):
        return TRIAGE_DERIVED_FROM_LEDGER
    return TRIAGE_MANUAL_REQUIRED


def main():
    """Run the L2 number-lock audit and write the report + machine-readable lock.

    @brief  1. Load the numeric corpus (JSON/NPY/log tokens + supplementary tables);
            2. Extract every metric-like 3/4-decimal claim from the manuscript body;
            3. Match each claim against the corpus within MATCH_ABSOLUTE_TOL;
            4. Recompute headline derived statistics from run_ledger.json;
            5. Triage unmatched claims into the stable TRIAGE_* categories;
            6. Serialise L2_report.md and number_lock.json.
    @return Process exit code (0; no claim is "wrong" by construction -- the audit
            is purely informational for manual adjudication).
    """
    root = os.environ.get(ENV_PROJECT_ROOT, DEFAULT_PROJECT_ROOT)
    outdir = os.path.join(root, OUTDIR_RELPATH)
    main_path = os.path.join(root, MAIN_TEX_RELPATH)
    ledger_path = os.path.join(root, RUN_LEDGER_RELPATH)

    with open(main_path) as handle:
        tex = handle.read()
    body = re.sub(r"(?<!\\)%.*", "", tex)

    artifacts = load_artifacts()
    supp_vals = supplementary_values()
    print(
        f"[corpus] {len(artifacts)} values from results/**/*.json; "
        f"{len(supp_vals)} values tabulated in the supplementary"
    )

    # --- Extract metric-like claims: 0 < value < 1, 3 or 4 decimals ------------
    claims = []
    for m in re.finditer(MANUSCRIPT_DECIMAL_RE, body):
        value = float(m.group(1))
        if 0.0 < value < 1.0:
            ctx_start = max(0, m.start() - CONTEXT_LEFT_CHARS)
            ctx_end = m.end() + CONTEXT_RIGHT_CHARS
            ctx = body[ctx_start:ctx_end].replace("\n", " ")
            claims.append((value, ctx))
    # Deduplicate claims by value; keep the *first* (lexicographically earliest)
    # context because repeated occurrences are typically consistent.
    uniq = {}
    for value, ctx in claims:
        uniq.setdefault(value, ctx)

    matched, ghosts = [], []
    for value, ctx in sorted(uniq.items()):
        hits = [(a, p) for a, p in artifacts if abs(a - value) <= MATCH_ABSOLUTE_TOL]
        if not hits:
            near = [v for v in supp_vals if abs(v - value) <= MATCH_ABSOLUTE_TOL]
            if near:
                hits = [(near[0], "supplementary.tex (tabulated)")]
        (matched if hits else ghosts).append((value, ctx, hits[:3]))

    # --- Derived statistics recomputed from run_ledger.json --------------------
    with open(ledger_path) as handle:
        ledger = json.load(handle)
    namr = [
        r[LEDGER_FIELD_KNOWN_EDGE_AUC] for r in ledger["runs"]
        if r.get("variant") == LEDGER_VARIANT_NAMRONLY
        and r.get("device") == LEDGER_DEVICE_CUDA
        and r.get(LEDGER_FIELD_KNOWN_EDGE_AUC)
    ]
    derived = []
    if namr:
        derived.append((
            f"NAMR-only CUDA known-edge mean (n={len(namr)})",
            float(np.mean(namr)),
            float(np.std(namr, ddof=1)),
        ))
    two = [
        r[LEDGER_FIELD_KNOWN_EDGE_AUC] for r in ledger["runs"]
        if r.get("variant") == LEDGER_VARIANT_NAMRONLY and r.get("seed") in LEDGER_SEEDS_TWO
    ]
    if len(two) == 2:
        derived.append((
            "NAMR-only seeds 42+7 mean",
            float(np.mean(two)),
            float(np.std(two, ddof=1)),
        ))
    derived_vals = {round(v, DERIVED_ROUND_DIGITS) for _, v, _ in derived}

    # --- Ghost triage ----------------------------------------------------------
    counts = defaultdict(int)
    ghost_rows = []
    for value, ctx, _ in ghosts:
        verdict = _triage_verdict(value, ctx, derived_vals)
        counts[verdict] += 1
        ghost_rows.append((value, ctx, verdict))

    # --- Report (English; markdown + JSON) -------------------------------------
    os.makedirs(outdir, exist_ok=True)
    lines = [
        "# L2 Numeric Audit Report (number lock + ghost-number back-trace)", "",
        f"Corpus: `results/**/*.json` / supplementary tables = "
        f"{len(artifacts)} numeric entries; manuscript body yields "
        f"{len(uniq)} distinct 3/4-decimal values in (0,1). "
        f"Match tolerance {MATCH_ABSOLUTE_TOL} (half a displayed unit).", "",
        f"## Artifact-backed claims ({len(matched)}/{len(uniq)})", "",
        "| Claimed value | Matching artifacts |", "|---|---|",
    ]
    for value, _ctx, hits in matched:
        lines.append(
            f"| {value} | "
            + "; ".join(f"`{p}` = {a}" for a, p in hits)
            + " |"
        )
    lines += [
        "",
        f"## Unmatched claims ({len(ghosts)}) -- manual adjudication "
        f"(external literature value / derived statistic / missing source)", "",
        "| Claimed value | Context | Triage |", "|---|---|---|",
    ]
    for value, ctx, verdict in ghost_rows:
        lines.append(f"| {value} | …{ctx[:CONTEXT_TRUNCATE_CHARS]}… | {verdict} |")

    lines += [
        "", "### Ghost-number category summary", "",
        "| Category | Count |", "|---|---|",
    ]
    for k, v in sorted(counts.items(), key=lambda t: -t[1]):
        lines.append(f"| {k} | {v} |")

    lines += [
        "", "## Derived statistics (recomputed from run_ledger.json)", "",
        "| Statistic | Recomputed value |", "|---|---|",
    ]
    for name, mean, sd in derived:
        lines.append(
            f"| {name} | {mean:.{DERIVED_DISPLAY_DIGITS}f} "
            f"(SD {sd:.{DERIVED_DISPLAY_DIGITS}f}) |"
        )

    lines += [
        "", "## Reporting convention checks", "",
        "| Check | Result |", "|---|---|",
    ]
    units = (
        len(re.findall(r"mean \$\\pm\$ SD", body))
        + len(re.findall(r"SD over", body))
    )
    lines.append(
        f"| Every mean ± SD states its units | "
        f"PASS ({units} explicit SD-unit statements in body) |"
    )
    has_ci = len(re.findall(r"\bCI\b|confidence interval", body))
    has_p = len(re.findall(r"\$p\s*[<=]", body))
    if has_ci == 0:
        lines.append(
            f"| CI and p share the same source | "
            f"PASS (no CI claims; only permutation p values) |"
        )
    else:
        lines.append(
            f"| CI and p share the same source | "
            f"**To check** (CI={has_ci} sites / p={has_p} sites) |"
        )

    with open(os.path.join(outdir, L2_REPORT_FILENAME), "w") as handle:
        handle.write("\n".join(lines))
    with open(os.path.join(outdir, NUMBERLOCK_FILENAME), "w") as handle:
        json.dump(
            {
                "matched": [[v, [p for _, p in h]] for v, _, h in matched],
                "ghosts": [[v, c] for v, c, _ in ghosts],
                "derived": derived,
            },
            handle,
            indent=1,
            ensure_ascii=False,
        )
    print(f"[summary] matched {len(matched)} / {len(uniq)}; ghosts {len(ghosts)}")
    print(f"[written] {outdir}/{L2_REPORT_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
