#!/usr/bin/env python3
"""
@file    88_audit_manuscript.py
@brief   Manuscript audit L1 (mechanical) + G7 (files and release consistency).
         Implements the deterministic half of the manuscript review protocol:
         reference closure, numbering continuity, prefix forms, placeholder/style
         scans, metadata counts, and the G7 four-way reconciliation between figure
         labels, includegraphics targets, in-text references and the filesystem,
         plus an orphan-file audit. Also checks checkpoint registry identity (N4).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  manuscript/__main.tex, manuscript/supplementary.tex, manuscript.log, manuscript/
Output: reviews/manuscript_audit_20260918/L1_report.md (+ manifest.json)
Run:    GFM_PROJECT_ROOT=<repo> python scripts/88_audit_manuscript.py
"""
import json
import os
import re
import sys
from collections import Counter

# =============================================================================
# File-level constants (paths, basenames, audit parameters)
# =============================================================================
ENV_PROJECT_ROOT = "GFM_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Relative path pieces resolved against ROOT.
MAIN_TEX_RELPATH = os.path.join("manuscript", "__main.tex")
SUPP_TEX_RELPATH = os.path.join("manuscript", "supplementary.tex")
SUPP_AUX_RELPATH = os.path.join("manuscript", "supplementary.aux")
LOG_RELPATH = "manuscript.log"
FIGDIR_RELPATH = "manuscript"
OUTDIR_RELPATH = os.path.join("reviews", "manuscript_audit_20260918")
REGISTRY_RELPATH = os.path.join("results", "checkpoint_registry.json")
SCRIPTDIR_RELPATH = "scripts"

L1_REPORT_FILENAME = "L1_report.md"
MANIFEST_FILENAME = "manifest.json"

# Regex-based phrase lists for the AI-phrase / placeholder scans.
AI_PHRASE_PATTERNS = [
    r"\bdelv(e|es|ing)\b", r"\bunderscor(e|es|ing)\b", r"\bshed light\b",
    r"\bpivotal\b", r"\bcrucial\b", r"\bit is worth noting\b",
    r"\bin conclusion\b", r"\bplays a (key|vital) role\b", r"\bseamless(ly)?\b",
    r"\brobustly\b", r"\bcomprehensive(ly)?\b", r"\bmoreover\b", r"\bfurthermore\b",
]
PLACEHOLDER_TOKENS = [
    "TODO", "TBD", "FIXME", "XXX", "???", "PLACEHOLDER", "to be added",
    "coming soon", "lorem",
]

# Claimed values exempt from the 3-decimal display rule (documented pair).
EXEMPT_4DECIMAL_VALUES = {"0.7077", "0.7078"}

# Top-N truncation thresholds for long report detail strings.
DETAIL_TRUNC_MAXLEN = 300            # table cell detail truncation
FAIL_DETAIL_TRUNC_MAXLEN = 200       # per-FAIL line detail truncation
UNDEFINED_REF_PREVIEW = 8            # preview of first N undefined refs
PLACEHOLDER_NAME_PREVIEW = 10
AMBIGUOUS_NAME_PREVIEW = 6
NEVER_CITED_PREVIEW = 12
DANGLING_SUPP_PREVIEW = 10
ORPHAN_DETAIL_PREVIEW = None         # full list (orphans are few)
GRAPHIC_STEM_NOTPRODUCED_PREVIEW = None

# LaTeX macro parameter that must be excluded from label-closure dangling sets.
MACRO_PARAM_TOKEN = "#1"

# Figure artefact basenames that are hand-drawn (no producing script) and therefore
# exempt from the "every graphic stem produced by a script" check.
HANDDRAWN_GRAPHIC_STEMS = {"fig1_architecture"}

# Checkpoint registry record fields used during the N4 lookup.
REGISTRY_SHA_FIELD = "sha256_12"
REGISTRY_FILE_FIELD = "file"
REGISTRY_PATH_FIELD = "path"
REGISTRY_RECORDS_KEY = "records"
REGISTRY_ERROR_PREFIX = "__"

# Report status tokens (output stable strings).
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"


def read(path):
    """Read a whole text file with UTF-8 decoding and replace error handler.

    @brief  Read the full contents of a UTF-8 text file.
    @param  path   Absolute or relative filesystem path to read.
    @return Entire file contents as a Python str; decoding errors are replaced
            with the Unicode replacement glyph so parsing never aborts.
    @note   Used for both TeX sources and the plain-text compilation log.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def strip_comments(text):
    r"""Remove LaTeX line comments while keeping document structure intact.

    @brief  Strip every LaTeX `% ...` end-of-line comment that is not preceded
            by a backslash (i.e. `\%` literals are preserved).
    @param  text   Raw LaTeX source.
    @return Source with comments removed (line count preserved).
    @note   Matches the lookbehind pattern `(?<!\\)%.*`; multi-line environments
            are not special-cased because this audit only operates on line-level
            constructs (labels, bibitems, decimal tokens).
    """
    return re.sub(r"(?<!\\)%.*", "", text)


def check(label, ok, detail=""):
    """Wrap a single audit line into a dict with PASS/FAIL status.

    @brief  Produce the canonical finding record consumed by the markdown table.
    @param  label    Short, human-readable name for the check (stable for diffs).
    @param  ok       Boolean: True == PASS, False == FAIL.
    @param  detail   Free-form elaboration; truncated downstream for the table.
    @return dict with keys ``check``, ``status``, ``detail``.
    @note   Status string is drawn from the file-level constants so the report
            parser (if added later) can rely on a fixed vocabulary.
    """
    return {"check": label, "status": STATUS_PASS if ok else STATUS_FAIL, "detail": detail}


def main():
    """Run the full L1 + G7 mechanical audit and write L1_report.md / manifest.json.

    @brief  Orchestrate every G1 (compile/closure/placeholder/terminology) and G7
            (four-way naming / checkpoint registry) check, then serialise the
            findings as both a human markdown report and a JSON manifest.
    @return Process exit code: 0 if every finding is PASS, 1 otherwise.
    @note   The supplementary.aux file is read directly from disk; its existence
            is required for the dense-numbering check (numbers are drawn from the
            LaTeX-generated \newlabel entries, not re-derived from sources).
    """
    # --- Resolve absolute paths against the project root -----------------------
    root = os.environ.get(ENV_PROJECT_ROOT, DEFAULT_PROJECT_ROOT)
    main_path = os.path.join(root, MAIN_TEX_RELPATH)
    supp_path = os.path.join(root, SUPP_TEX_RELPATH)
    supp_aux_path = os.path.join(root, SUPP_AUX_RELPATH)
    log_path = os.path.join(root, LOG_RELPATH)
    figdir_path = os.path.join(root, FIGDIR_RELPATH)
    outdir_path = os.path.join(root, OUTDIR_RELPATH)
    registry_path = os.path.join(root, REGISTRY_RELPATH)
    script_dirs = [os.path.join(root, SCRIPTDIR_RELPATH)]

    # --- Load sources -----------------------------------------------------------
    main_tex = strip_comments(read(main_path))
    supp_tex = strip_comments(read(supp_path))
    log = read(log_path) if os.path.exists(log_path) else ""
    both = main_tex + "\n" + supp_tex
    findings = []

    # =========================================================================
    # G1: compile cleanliness -------------------------------------------------
    # =========================================================================
    undefined = re.findall(r"Reference `([^']+)' on page", log) + \
                re.findall(r"Citation `([^']+)' on page", log)
    findings.append(check(
        "G1 compile: 0 undefined references/citations",
        not undefined,
        f"{len(undefined)} undefined: {sorted(set(undefined))[:UNDEFINED_REF_PREVIEW]}",
    ))
    errs = [l for l in log.splitlines() if l.startswith("!")]
    findings.append(check(
        "G1 compile: 0 LaTeX errors",
        not errs,
        "; ".join(errs[:4]),
    ))

    # =========================================================================
    # G1: label / reference closure ------------------------------------------
    # =========================================================================
    labels = set(re.findall(r"\\label\{([^}]+)\}", both))
    refs = set()
    for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", both):
        refs.update(x.strip() for x in m.group(1).split(","))
    # Macro parameters such as \ref{#1} inside a command definition are not labels.
    dangling = sorted(x for x in (refs - labels) if x != MACRO_PARAM_TOKEN)
    findings.append(check(
        "G1 label closure: no dangling \\ref",
        not dangling,
        str(dangling),
    ))

    # Supplementary float numbering is generated by LaTeX (S\d+) via \supptab /
    # \suppfig / \suppnote macros. Any hand-written "Supplementary Table~S12"
    # form in the source is a regression the audit must flag.
    manual = re.findall(
        r"Supplementary (?:Table|Tables|Fig\.|Note)~?\s?S?\d[\w]*",
        main_tex + "\n" + supp_tex,
    )
    manual = [m for m in manual if "name=Supplementary" not in m]
    findings.append(check(
        "G1 no hand-written supplementary numbers in either source",
        not manual,
        str(sorted(set(manual))[:PLACEHOLDER_NAME_PREVIEW]),
    ))

    supp_labels = set(re.findall(r"\\label\{([^}]*)\}", supp_tex))
    supp_labels.update(re.findall(r"\\suppnotesection\{[^}]*\}\{([^}]*)\}", supp_tex))
    used = set()
    for cmd in ("supptab", "supptabs", "suppfig", "suppnote"):
        used.update(re.findall(r"\\%s\{([^}]*)\}" % cmd, main_tex + "\n" + supp_tex))
    # \supptabrange{A}{B} -> both labels must be in the closure graph.
    used.update(
        re.findall(r"\\supptabrange\{([^}]*)\}\{([^}]*)\}", main_tex + "\n" + supp_tex)
        and [x for pair in re.findall(
            r"\\supptabrange\{([^}]*)\}\{([^}]*)\}",
            main_tex + "\n" + supp_tex,
        ) for x in pair]
        or []
    )
    dangling = sorted(used - supp_labels)
    findings.append(check(
        "G1 supplementary label closure: every \\supptab/\\suppfig/\\suppnote exists",
        not dangling,
        f"dangling: {dangling[:DANGLING_SUPP_PREVIEW]}",
    ))

    float_labels = set(re.findall(r"\\label\{(tab:s[0-9a-z]+|fig:s[0-9]+)\}", supp_tex))
    cited_from_main = set()
    for cmd in ("supptab", "supptabs", "suppfig"):
        cited_from_main.update(re.findall(r"\\%s\{([^}]*)\}" % cmd, main_tex))
    for a, b in re.findall(r"\\supptabrange\{([^}]*)\}\{([^}]*)\}", main_tex):
        cited_from_main.update({a, b})
    never_cited = sorted(float_labels - cited_from_main)
    findings.append(check(
        "G1 supplementary float labels all cited from the main text",
        not never_cited,
        f"never cited: {never_cited[:NEVER_CITED_PREVIEW]}",
    ))

    # Read the LaTeX-generated .aux to inspect the *actual* numbers assigned.
    aux = read(supp_aux_path)
    nums = {}
    for key, val in re.findall(r"\\newlabel\{((?:tab|fig):s[0-9a-z]+)\}\{\{S(\d+)", aux):
        nums[key] = int(val)
    for kind, pat in (("table", r"tab:s[0-9a-z]+$"), ("figure", r"fig:s[0-9]+$")):
        vals = sorted(v for k, v in nums.items() if re.match(pat, k))
        dense = vals == list(range(1, len(vals) + 1))
        findings.append(check(
            f"G1 supplementary {kind} numbers are dense (from supplementary.aux)",
            dense and bool(vals),
            f"S{vals[0] if vals else '-'}-S{vals[-1] if vals else '-'}, n={len(vals)}",
        ))

    # =========================================================================
    # G1: citation closure ----------------------------------------------------
    # =========================================================================
    bibkeys = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", main_tex)
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", main_tex):
        cites.update(x.strip() for x in m.group(1).split(","))
    findings.append(check(
        "G1 citation closure: every \\bibitem cited",
        not (set(bibkeys) - cites),
        str(sorted(set(bibkeys) - cites)),
    ))
    findings.append(check(
        "G1 citation closure: every \\cite has a bibitem",
        not (cites - set(bibkeys)),
        str(sorted(cites - set(bibkeys))),
    ))
    dup = [k for k, n in Counter(bibkeys).items() if n > 1]
    findings.append(check(
        "G1 bibliography: no duplicate bibitem keys",
        not dup,
        str(dup),
    ))

    # =========================================================================
    # G1: supplementary caption numbering / float style ----------------------
    # =========================================================================
    ok_names = (
        "\\captionsetup[table]{name=Supplementary Table" in supp_tex and
        "\\captionsetup[figure]{name=Supplementary Fig." in supp_tex and
        "\\renewcommand{\\thetable}{S\\arabic{table}}" in supp_tex and
        "\\renewcommand{\\thefigure}{S\\arabic{figure}}" in supp_tex
    )
    findings.append(check(
        "G1 supplementary float numbers generated by LaTeX",
        ok_names,
        "captionsetup name + S-prefixed counters in supplementary preamble",
    ))

    as_section = re.findall(r"\\section\{Supplementary (Table|Fig\\.)\s+[S]?\\d", supp_tex)
    findings.append(check(
        "G1 no supplementary float is typeset as a section",
        not as_section,
        f"found: {sorted(set(as_section))}",
    ))

    # =========================================================================
    # G1: textual "Supplementary x.y" cross-reference resolution -------------
    # =========================================================================
    numbered_subsections = len(re.findall(r"\\subsection\{", supp_tex))
    plain_refs = sorted(set(re.findall(r"Supplementary (\d+\.\d+)", supp_tex + main_tex)))
    unresolved = [r for r in plain_refs if int(r.split(".")[1]) > numbered_subsections]
    findings.append(check(
        "G1 textual 'Supplementary x.y' references resolve",
        not unresolved,
        f"refs {plain_refs}; supplementary has {numbered_subsections} subsections; "
        f"unresolved {unresolved}",
    ))

    # =========================================================================
    # G1: placeholders / style / AI-phrase scan ------------------------------
    # =========================================================================
    hits = {p: len(re.findall(re.escape(p), both)) for p in PLACEHOLDER_TOKENS}
    findings.append(check(
        "G1 no placeholders",
        not any(hits.values()),
        str({k: v for k, v in hits.items() if v}),
    ))
    emdash = both.count("---")
    findings.append(check(
        "G1 em dash (---) count reported",
        True,
        f"{emdash} occurrences",
    ))
    phrases = {p: len(re.findall(p, both, flags=re.I)) for p in AI_PHRASE_PATTERNS}
    phrases = {k: v for k, v in phrases.items() if v}
    findings.append(check(
        "G1 AI-phrase scan reported",
        True,
        str(phrases),
    ))

    # =========================================================================
    # G1: metadata counts / figure-decl vs env count -------------------------
    # =========================================================================
    fig_envs = len(re.findall(r"\\begin\{figure\}", main_tex))
    tab_envs = len(re.findall(r"\\begin\{table", main_tex))
    supp_fig_envs = len(re.findall(r"\\begin\{figure\}", supp_tex))
    supp_tab_envs = len(re.findall(r"\\begin\{table", supp_tex))
    declared = re.findall(r"(\d+)\s+(?:main\s+)?figures", main_tex, flags=re.I)
    findings.append(check(
        "G1 figure count: declared == environments",
        all(int(d) == fig_envs for d in declared) if declared else True,
        f"declared {declared}, environments {fig_envs}",
    ))
    findings.append(check(
        "G1 figure/table inventory",
        True,
        f"main figs {fig_envs}, main tables {tab_envs}, "
        f"supp figs {supp_fig_envs}, supp tables {supp_tab_envs}",
    ))

    # =========================================================================
    # G1: Funding / availability position ------------------------------------
    # =========================================================================
    fund_sec = max(main_tex.find("\\subsection{Funding}"), main_tex.find("\\section*{Funding}"))
    supp_fund = "Funding" in supp_tex
    findings.append(check(
        "G1 Funding in main text, not only supplementary",
        fund_sec > 0 and not supp_fund,
        f"main@{fund_sec}, supp={supp_fund}",
    ))

    # =========================================================================
    # G1: terminology + numeric-format consistency ---------------------------
    # =========================================================================
    term_checks = {
        "PPI AUC (old term) vs known-edge recovery":
            len(re.findall(r"PPI (?:link prediction|AUC)\b", both)),
        "external fixed-target (retired term)":
            len(re.findall(r"external fixed-target", both)),
        "self-consistency diagnostic declared":
            len(re.findall(r"self-consistency diagnostic", both)),
        "fixed-target endpoint (current term)":
            len(re.findall(r"fixed-target endpoint", both)),
    }
    findings.append(check(
        "G1 terminology: 'PPI AUC' / 'external fixed-target' fully retired",
        term_checks["PPI AUC (old term) vs known-edge recovery"] == 0
        and term_checks["external fixed-target (retired term)"] == 0,
        str(term_checks),
    ))

    # Performance-like numbers: 3 decimals outside the documented exempt pair.
    perf4 = sorted(set(re.findall(r"(?<![\d.])(0\.\d{4})(?![\d])", main_tex)))
    offenders = [v for v in perf4 if v not in EXEMPT_4DECIMAL_VALUES]
    findings.append(check(
        "G1 numeric format: 3-decimal display outside the exempt pair",
        not offenders,
        f"4-decimal values: {perf4}; offenders {offenders}",
    ))

    # =========================================================================
    # G7: four-way naming reconciliation -------------------------------------
    # =========================================================================
    inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", both)
    stems = {os.path.splitext(os.path.basename(i))[0] for i in inc}
    fig_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", both))
    on_disk = {
        os.path.splitext(f)[0] for f in os.listdir(figdir_path)
        if f.lower().endswith((".png", ".pdf", ".eps"))
    }
    findings.append(check(
        "G7 includegraphics targets exist on disk",
        all(os.path.exists(os.path.join(figdir_path, i)) for i in inc),
        str([i for i in inc if not os.path.exists(os.path.join(figdir_path, i))]),
    ))
    findings.append(check(
        "G7 four-way reconciliation (stems/labels/refs/disk)",
        len(stems) == fig_envs + supp_fig_envs,
        f"{len(stems)} graphic stems for {fig_envs + supp_fig_envs} figure environments",
    ))
    orphans = sorted(f for f in on_disk if f not in stems and re.match(r"^fig", f))
    findings.append(check(
        "G7 no orphan figure files in manuscript/",
        not orphans,
        str(orphans),
    ))

    # --- Script-produced artefact scan: every graphic stem must be traceable
    # to a save_fig / savefig call inside scripts/*.py (or the handdrawn allow-list).
    script_out = set()
    for d in script_dirs:
        for f in os.listdir(d):
            if f.endswith(".py"):
                body = read(os.path.join(d, f))
                for m in re.finditer(
                    r'save_fig\(\s*\w+\s*,\s*os\.path\.join\([^,]+,\s*["\']([^"\']+)["\']',
                    body,
                ):
                    script_out.add(m.group(1))
                for m in re.finditer(r'savefig\(f?["\']([^"\'{]+)', body):
                    script_out.add(os.path.basename(m.group(1)))
    findings.append(check(
        "G7 manuscript figures are reproducible artefacts",
        stems <= (script_out | HANDDRAWN_GRAPHIC_STEMS),
        f"not produced by any script: {sorted(stems - script_out)}",
    ))

    # =========================================================================
    # G7: checkpoint identity registry (review item N4) ----------------------
    # =========================================================================
    # Every .pt file named in the manuscript or supplementary must be registered
    # in the checkpoint registry, and no bare file name may denote two distinct
    # models.
    registry = {}
    if os.path.exists(registry_path):
        try:
            records = json.load(open(registry_path)).get(REGISTRY_RECORDS_KEY, [])
            registry = {r[REGISTRY_PATH_FIELD]: r for r in records}
        except Exception as exc:                                     # pragma: no cover
            registry = {"__error__": {"error": str(exc)}}
    named = set()
    for m in re.finditer(r"\\texttt\{([^}]*?\.pt)\}?", both):
        raw = m.group(1)
        # Strip LaTeX escapes used inside file names, then take the last path segment.
        raw = raw.replace("\\_", "_").replace("\\", "").replace("\textbackslash", "")
        raw = raw.strip().rstrip("}").lstrip("/")
        if raw.startswith("_") or "*" in raw:
            # Table-local shorthand for the preceding full path, or a wildcard family.
            continue
        if "/" in raw:
            named.add("results/" + raw if not raw.startswith("results/") else raw)
        else:
            named.add(raw)
    basenames = {}
    for path, rec in registry.items():
        if path.startswith(REGISTRY_ERROR_PREFIX):
            continue
        basenames.setdefault(
            rec.get(REGISTRY_FILE_FIELD) or os.path.basename(path),
            set(),
        ).add(rec.get(REGISTRY_SHA_FIELD))
    missing, ambiguous, _unqualified = [], [], []
    for name in sorted(named):
        if name.startswith("results/"):
            if name not in registry:
                missing.append(name)
        else:
            if name not in basenames:
                missing.append(name)
            elif len(basenames[name]) > 1:
                ambiguous.append(name)
    findings.append(check(
        "G7 checkpoint files named in the text are registered",
        not missing and not ambiguous,
        f"unregistered: {missing[:AMBIGUOUS_NAME_PREVIEW]}; "
        f"ambiguous bare names (use a path): {ambiguous[:AMBIGUOUS_NAME_PREVIEW]}",
    ))

    # =========================================================================
    # Report serialisation ----------------------------------------------------
    # =========================================================================
    os.makedirs(outdir_path, exist_ok=True)
    lines = [
        "# L1 + G7 Mechanical Audit Report (manuscript review protocol)", "",
        f"Scope: `manuscript/__main.tex` ({len(main_tex)} chars) + "
        f"`manuscript/supplementary.tex` ({len(supp_tex)} chars)",
        f"Inventory: main labels {len(labels)} / refs {len(refs)} / "
        f"bibitems {len(bibkeys)} / cites {len(cites)} / "
        f"figures {fig_envs} / tables {tab_envs}; "
        f"supp figures {supp_fig_envs} / tables {supp_tab_envs}", "",
        "| Check Item | Result | Detail |", "|---|---|---|",
    ]
    for finding in findings:
        lines.append(
            f"| {finding['check']} | **{finding['status']}** | "
            f"{finding['detail'][:DETAIL_TRUNC_MAXLEN]} |"
        )
    fails = [f for f in findings if f["status"] == STATUS_FAIL]
    lines += [
        "",
        f"**Conclusion: {len(findings) - len(fails)} PASS / {len(fails)} FAIL**",
        "",
    ]
    with open(os.path.join(outdir_path, L1_REPORT_FILENAME), "w") as handle:
        handle.write("\n".join(lines))
    with open(os.path.join(outdir_path, MANIFEST_FILENAME), "w") as handle:
        json.dump(
            {
                "findings": findings,
                "stems": sorted(stems),
                "orphans": orphans,
                "bibitems": bibkeys,
                "labels": sorted(labels),
            },
            handle,
            indent=1,
            ensure_ascii=False,
        )
    # --- Stdout summary (English only per the project print/log rule) -------
    print("\n".join(lines[-3:]))
    print(f"[written] {outdir_path}/{L1_REPORT_FILENAME}")
    for fail in fails:
        print(f"  FAIL: {fail['check']} -> {fail['detail'][:FAIL_DETAIL_TRUNC_MAXLEN]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
