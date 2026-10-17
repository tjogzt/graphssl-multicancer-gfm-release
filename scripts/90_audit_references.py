#!/usr/bin/env python3
"""
@file    90_audit_references.py
@brief   Manuscript audit L4 (references): every bibliography entry that carries
         a DOI is verified against the Crossref API, and entries without a DOI
         fall back to Crossref bibliographic search then arXiv title lookup.
         Rate-limited (600 ms between requests) and cached under the audit
         output directory so repeated runs do not re-query the network.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  manuscript/__main.tex (thebibliography environment)
Output: reviews/manuscript_audit_20260918/L4_references.md + crossref_cache.json
Run:    GFM_PROJECT_ROOT=<repo> python scripts/90_audit_references.py
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# =============================================================================
# File-level constants
# =============================================================================
ENV_PROJECT_ROOT = "GFM_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_TEX_RELPATH = os.path.join("manuscript", "__main.tex")
OUTDIR_RELPATH = os.path.join("reviews", "manuscript_audit_20260918")
CACHE_FILENAME = "crossref_cache.json"
L4_REPORT_FILENAME = "L4_references.md"

# HTTP: Crossref API requires a descriptive User-Agent per their etiquette
# policy (see https://github.com/CrossRef/rest-api-doc#etiquette). The mailto
# address is a stable project-wide placeholder; replace with the corresponding
# author's contact before a first run against a clean cache.
HTTP_USER_AGENT = "manuscript-audit/1.0 (mailto:audit@example.org)"
HTTP_TIMEOUT_SECONDS = 30               # Crossref API call timeout
RATE_LIMIT_SLEEP_SECONDS = 0.6          # 600 ms between API hits (~1.66 QPS)

# DOI regex: RFC on DOI syntax is loose in practice; capture anything that
# starts with the 10. prefix and carries no whitespace/comma/closing-brace
# (trailing periods inside bibitem text are very common and are stripped later).
DOI_CAPTURE_RE = r"(10\.\d{4,9}/[^\s,}]+)"

# Similarity threshold below which a Crossref "best match" or arXiv hit is
# rejected (token-overlap Jaccard).
TITLE_SIMILARITY_THRESHOLD = 0.7

# arXiv API: query params. max_results=1 because only the top-ranked hit is
# compared; if the top hit is dissimilar the fallback search is aborted.
ARXIV_MAX_RESULTS = 1
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
ARXIV_SEARCH_FIELD_TI = "ti"          # title field token

# Crossref bibliographic search endpoint: rows=1; we only inspect the top
# candidate and fall back to arXiv if it is dissimilar.
CROSSREF_SEARCH_ROWS = 1
CROSSREF_SEARCH_ENDPOINT = "https://api.crossref.org/works"
CROSSREF_WORK_ENDPOINT = "https://api.crossref.org/works/"
CROSSREF_SELECT_FIELDS = "DOI,title,container-title,issued,volume"

# Crossref date fields consulted for year matching (journals often use a
# different online-first year than the print issue year; accept any).
CROSSREF_YEAR_FIELDS = ("published-print", "published-online", "issued", "created")

# Normalisation: keep only lowercase alphanumerics when comparing title / year
# / volume text against the manuscript printed form.
NORM_STRIP_RE = r"[^a-z0-9]+"
NORM_TITLE_PREFIX_LEN = 40              # first 40 chars after normalisation

# Report table: truncate displayed Crossref titles to this many characters.
TITLE_DISPLAY_TRUNC = 60

# Print formatting / verdict tokens (stable strings for downstream parsers).
VERDICT_OK = "OK"
VERDICT_MANUAL = "**manual check required**"
VERDICT_NO_DOI = "no DOI"
YEAR_TOLERANCE = 1                      # accept ±1 year for issue vs online

# JSON indent level used for both cache and report serialisation.
JSON_INDENT = 1


def entries(tex):
    """Parse ``\\begin{thebibliography}...\\end{thebibliography}`` into records.

    @brief  Extract every \\bibitem entry with its key, normalised body text,
            and the DOI (if present inside the body text).
    @param  tex   Full manuscript/__main.tex source.
    @return List of dicts: ``{"key", "text", "doi"}`` -- ``doi`` is ``None``
            if the entry does not contain a DOI-looking token.
    @note   Trailing punctuation (periods, commas, semicolons, closing parens)
            is stripped from the captured DOI because bibitem text commonly
            appends a period immediately after the DOI.
    """
    body = tex[tex.find("\\begin{thebibliography}"):]
    out = []
    for m in re.finditer(
        r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
        body,
        flags=re.S,
    ):
        text = re.sub(r"\s+", " ", m.group(2)).strip()
        doi_match = re.search(DOI_CAPTURE_RE, text)
        doi = doi_match.group(1).rstrip(".,;)") if doi_match else None
        out.append({"key": m.group(1), "text": text, "doi": doi})
    return out


def norm(s):
    """Aggressively normalise free-text strings for substring / overlap comparison.

    @brief  Lowercase the input, then strip every non-alphanumeric run. This
            makes ``Self-Supervised Graph Representation`` ==
            ``self supervised graph representation``.
    @param  s   Input string (may be None).
    @return Normalised string; None inputs produce the empty string.
    @note   Used by both the title-overlap check and the "year printed in entry"
            substring probe.
    """
    return re.sub(NORM_STRIP_RE, "", (s or "").lower())


def crossref(doi, cache):
    """Return the Crossref ``message`` payload for a DOI, cached on-disk.

    @brief  Look up a DOI in the in-memory cache; on miss query
            https://api.crossref.org/works/<doi> with the project UA, store the
            result, and rate-limit before returning.
    @param  doi     DOI string (already cleaned by entries()).
    @param  cache   Dict mutated in place; ``cache[doi]`` is always set by the
                    time the function returns (even on HTTP / network error).
    @return Crossref ``message`` dict on success; a dict of the form
            ``{"error": "..."}`` on any failure.
    @note   RATE_LIMIT_SLEEP_SECONDS is respected on *every* call that reaches
            the network; cache hits skip the sleep entirely so an audit rerun
            against a warm cache completes instantly.
    """
    if doi in cache:
        return cache[doi]
    url = CROSSREF_WORK_ENDPOINT + urllib.parse.quote(doi)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as handle:
            payload = json.load(handle)["message"]
    except urllib.error.HTTPError as exc:
        payload = {"error": f"HTTP {exc.code}"}
    except Exception as exc:                                   # network / timeout
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    cache[doi] = payload
    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
    return payload


def similar(a, b, threshold=TITLE_SIMILARITY_THRESHOLD):
    """Token-overlap (Jaccard) similarity used to reject bad search hits.

    @brief  Tokenise both strings on alphanumeric runs (case-insensitive) and
            return the Jaccard coefficient of the resulting token sets. If the
            returned value is >= ``threshold`` the two strings are considered
            to describe the same work.
    @param  a           First string.
    @param  b           Second string.
    @param  threshold   Default comparison threshold (kept as a named param so
                        callers can override).
    @return Float in [0, 1]; 0.0 if either input has no alphanumeric tokens.
    @note   Used by both Crossref bibliographic-search fallback and arXiv
            fallback; the same threshold is reused so behaviour is consistent.
    """
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def arxiv_search(entry_text, cache):
    """arXiv API lookup: authoritative for preprint / software references.

    @brief  Extract the ``back-tick quoted title`` convention used in bibitems
            that lack a DOI, query the arXiv export API for a single top-ranked
            hit, accept it only if the title is above the similarity threshold,
            and return a synthetic Crossref-style record.
    @param  entry_text  Normalised bibitem body text.
    @param  cache       In-memory rate-limit cache (keys are prefixed with
                        ``arxiv:`` so they never collide with raw DOIs).
    @return Dict with keys ``{doi, title, container, year}`` on a good hit;
            ``None`` if no title can be extracted, or if the top arXiv hit is
            dissimilar.
    """
    m = re.search(r"`([^`]+),?`", entry_text)
    if not m:
        return None
    title = m.group(1).strip().rstrip(",")
    cache_key = "arxiv:" + title
    if cache_key in cache:
        xml = cache[cache_key]
    else:
        url = (
            f"{ARXIV_ENDPOINT}?max_results={ARXIV_MAX_RESULTS}"
            f"&search_query={ARXIV_SEARCH_FIELD_TI}:%22"
            + urllib.parse.quote(title) + "%22"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as handle:
                xml = handle.read().decode("utf-8", "replace")
        except Exception:
            xml = ""
        cache[cache_key] = xml
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)
    hit = re.search(
        r"<entry>.*?<title>(.*?)</title>.*?<published>(\d{4})",
        xml,
        flags=re.S,
    )
    if not hit:
        return None
    found = re.sub(r"\s+", " ", hit.group(1)).strip()
    if similar(found, title) < TITLE_SIMILARITY_THRESHOLD:
        return None
    return {
        "doi": "arXiv",
        "title": found,
        "container": "arXiv preprint",
        "year": hit.group(2),
    }


def title_search(entry_text, cache):
    """Crossref bibliographic search fallback for entries without a DOI.

    @brief  Same back-tick extraction as ``arxiv_search``, but first try the
            Crossref /works query API with bibliographic title search. Only the
            top candidate is inspected; if it is dissimilar the function falls
            through to :func:`arxiv_search`.
    @param  entry_text  Normalised bibitem body text.
    @param  cache       In-memory cache (keys prefixed ``search:``).
    @return Crossref-style dict ``{doi, title, container, year}`` on a good
            hit; ``None`` otherwise.
    """
    m = re.search(r"`([^`]+),?`", entry_text)
    if not m:
        return None
    title = m.group(1).strip().rstrip(",")
    cache_key = "search:" + title
    if cache_key in cache:
        msg = cache[cache_key]
    else:
        url = (
            f"{CROSSREF_SEARCH_ENDPOINT}?rows={CROSSREF_SEARCH_ROWS}"
            f"&select={CROSSREF_SELECT_FIELDS}&query.bibliographic="
            + urllib.parse.quote(title)
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as handle:
                msg = json.load(handle)["message"]["items"]
        except Exception as exc:
            msg = [{"error": f"{type(exc).__name__}"}]
        cache[cache_key] = msg
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)
    if not msg or "error" in msg[0]:
        return None
    hit = msg[0]
    found = (hit.get("title") or [""])[0]
    if similar(found, title) < TITLE_SIMILARITY_THRESHOLD:
        return arxiv_search(entry_text, cache)
    parts = (hit.get("issued") or {}).get("date-parts") or [[None]]
    return {
        "doi": hit.get("DOI", ""),
        "title": found,
        "container": (hit.get("container-title") or [""])[0],
        "year": parts[0][0] if parts and parts[0] else "",
    }


def main():
    """Run the L4 reference audit and write L4_references.md plus the cache.

    @brief  (1) parse all bibitems; (2) for DOI-bearing entries query Crossref
            and compare title / year / volume with the printed form; (3) for
            entries without a DOI try Crossref title search then arXiv;
            (4) write a markdown report (per-entry table + per-problem list)
            and persist the HTTP cache for incremental reruns.
    @return Exit code 0 if every entry is clean (no problems list); 1 otherwise.
    """
    root = os.environ.get(ENV_PROJECT_ROOT, DEFAULT_PROJECT_ROOT)
    outdir = os.path.join(root, OUTDIR_RELPATH)
    cache_path = os.path.join(outdir, CACHE_FILENAME)

    with open(os.path.join(root, MAIN_TEX_RELPATH)) as handle:
        tex = handle.read()
    refs = entries(tex)
    os.makedirs(outdir, exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    rows, problems = [], []
    for ref in refs:
        # --- Branch A: no DOI -> try title search / arXiv fallback ----------
        if not ref["doi"]:
            guess = title_search(ref["text"], cache)
            if guess:
                rows.append((
                    ref["key"],
                    f"search\u2192 {guess['doi']}",
                    f"{guess['container']} {guess['year']}",
                    guess["title"][:TITLE_DISPLAY_TRUNC],
                    "resolved by title search",
                ))
            else:
                rows.append((ref["key"], VERDICT_NO_DOI, "", "", VERDICT_MANUAL))
                problems.append((ref["key"], "no DOI and title search found nothing"))
            continue

        # --- Branch B: DOI present -> authoritative Crossref lookup ---------
        msg = crossref(ref["doi"], cache)
        if "error" in msg:
            rows.append((
                ref["key"], ref["doi"], "", "",
                f"**{msg['error']}**",
            ))
            problems.append((ref["key"], f"DOI lookup failed: {msg['error']}"))
            continue

        title = (msg.get("title") or [""])[0]
        # Collect every year Crossref exposes (online-first year != print issue
        # year is common, so accept any match within tolerance).
        years = set()
        for field in CROSSREF_YEAR_FIELDS:
            parts = (msg.get(field) or {}).get("date-parts") or [[None]]
            if parts and parts[0] and parts[0][0]:
                years.add(str(parts[0][0]))
        year = sorted(years)[0] if years else None
        volume = msg.get("volume", "")
        container = (msg.get("container-title") or [""])[0]

        issues, notes = [], []
        if title and norm(title)[:NORM_TITLE_PREFIX_LEN] not in norm(ref["text"]):
            issues.append("title not found verbatim in entry")
        if years and not any(y in ref["text"] for y in years):
            printed_years = [int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", ref["text"])]
            near = any(
                abs(int(y) - p) <= YEAR_TOLERANCE for y in years for p in printed_years
            )
            if near and printed_years:
                notes.append(
                    f"year = issue year {printed_years[-1]} vs "
                    f"Crossref online {sorted(years)[0]}"
                )
            else:
                issues.append(f"year not printed (Crossref years {sorted(years)})")
        if volume and (
            f"vol.~{volume}" not in ref["text"]
            and f"vol.{volume}" not in ref["text"]
            and f"volume {volume}" not in ref["text"].lower()
        ):
            issues.append(f"volume {volume} not printed")

        if issues:
            problems.append((ref["key"], "; ".join(issues)))
        verdict = VERDICT_OK if not issues else "; ".join(issues)
        if notes and not issues:
            verdict = VERDICT_OK + " (" + "; ".join(notes) + ")"
        rows.append((
            ref["key"], ref["doi"],
            f"{container} {year}; vol {volume}",
            title[:TITLE_DISPLAY_TRUNC],
            verdict,
        ))

    # --- Serialise markdown report (English only) ------------------------------
    n_with_doi = sum(1 for r in refs if r["doi"])
    lines = [
        "# L4 Reference Audit Report (Crossref ground-truth)", "",
        f"Total bibitem entries: {len(refs)}; "
        f"{n_with_doi} entries carry a DOI; "
        f"{len(refs) - n_with_doi} resolved via title search / arXiv fallback.", "",
        "| key | DOI / lookup path | Crossref record | Title (truncated) | Verdict |",
        "|---|---|---|---|---|",
    ]
    for key, doi, rec, title_txt, verdict in rows:
        lines.append(f"| {key} | {doi or '\u2014'} | {rec} | {title_txt} | {verdict} |")
    lines += [
        "",
        f"**Conclusion: {len(refs) - len(problems)} entries clean / "
        f"{len(problems)} entries need attention**",
        "",
    ]
    for key, issue in problems:
        lines.append(f"- `{key}`: {issue}")
    with open(os.path.join(outdir, L4_REPORT_FILENAME), "w") as handle:
        handle.write("\n".join(lines))
    with open(cache_path, "w") as handle:
        json.dump(cache, handle, indent=JSON_INDENT)
    # --- English-only stdout ---------------------------------------------------
    print(f"[L4 summary] refs {len(refs)}; problems {len(problems)}")
    for key, issue in problems:
        print(f"  {key}: {issue}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
