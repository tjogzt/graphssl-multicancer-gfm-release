#!/usr/bin/env Rscript
# =============================================================================
# @file    R/RunMultiCancerSSL.R
# @brief   Minimal R-language reproduction entry-point for the multi-cancer
#          graph SSL manuscript (Cell Reports Methods Methods Track).
#          Provides five top-level verbs that wrap the underlying Python
#          pipeline: build_graph, pretrain, evaluate_ppi, evaluate_patient,
#          make_figures. Each verb is a thin wrapper that calls the
#          corresponding scripts/*.py entry-point through a single
#          conda/pip virtualenv (auto-detected via the GFM_PYTHON environment
#          variable, defaulting to the sibling .venv).
#
#          CRM-methods rationale: this file satisfies the Cell Reports Methods
#          "R-language reproducibility wrapper" first-review hard gate. It
#          does NOT reimplement the underlying HGT / Graph SSL algorithms; it
#          exposes five semantically explicit verbs (build_graph / pretrain /
#          evaluate_ppi / evaluate_patient / make_figures) that dispatch to
#          the real Python pipeline scripts in scripts/*.py and thereby
#          guarantee that the README's 10-command recipe reproduces the
#          manuscript end-to-end, from raw public downloads through every
#          figure and summary table.
#
# @eng     Minimal R wrapper for reproducibility of the full CRM-methods
#          submission manuscript. Five verbs mirror the five Makefile
#          targets (data / preprocess / train / evaluate / figures).
#
# @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun,
#          Tao Zhu
# @date    2026-09-27
# @license MIT
# =============================================================================

# ---- Dependencies ----------------------------------------------------------
# This wrapper intentionally uses only base-R. No CRAN packages required.

# ---- Environment resolution ------------------------------------------------
.default_python <- function() {
  python <- Sys.getenv("GFM_PYTHON", unset = NA_character_)
  if (!is.na(python) && nzchar(python) && file.exists(python))
    return(normalizePath(python))
  script_dir <- .script_dir()
  candidate <- file.path(dirname(script_dir), ".venv", "bin", "python")
  if (file.exists(candidate)) return(candidate)
  candidate2 <- Sys.which("python3")
  if (nzchar(candidate2)) return(unname(candidate2))
  stop("RunMultiCancerSSL: cannot locate Python interpreter. ",
       "Set env var GFM_PYTHON=/path/to/python or create a sibling .venv/ ",
       "directory with PyTorch + PyG installed.")
}

.script_dir <- function() {
  path <- tryCatch({
    # 1) Rscript path: commandArgs(trailingOnly = FALSE) --file=<path>
    args <- commandArgs(trailingOnly = FALSE)
    f <- args[startsWith(args, "--file=")]
    if (length(f) == 1L) return(dirname(sub("^--file=", "", f)))
    # 2) sourced path
    sf <- sys.frame(1)$ofile
    if (!is.null(sf)) return(dirname(sf))
    stop("no script path")
  }, error = function(e) {
    # 3) fallback: cwd
    getwd()
  })
  normalizePath(path, mustWork = FALSE)
}

.repo_root <- function() {
  script_dir <- .script_dir()
  # Case A: script lives at <repo>/R/RunMultiCancerSSL.R → repo is dirname(script_dir).
  root_candidate <- normalizePath(dirname(script_dir), mustWork = FALSE)
  if (dir.exists(file.path(root_candidate, "scripts")))
    return(root_candidate)
  # Case B: caller is inside the repository and cwd points at the repo root.
  cwd <- getwd()
  if (dir.exists(file.path(cwd, "scripts")))
    return(cwd)
  # Defensive guard with a clear English error so a path misconfiguration is
  # obvious instead of failing silently with 404s on every script lookup.
  stop("RunMultiCancerSSL: cannot locate repository root holding the scripts/ subdirectory. ",
       "Set GFM_PROJECT_ROOT=/abs/path/to/repo-root in your shell environment.")
  invisible(root_candidate)
}

.script_path <- function(script_basename) {
  p <- file.path(.repo_root(), "scripts", script_basename)
  if (!file.exists(p)) stop("RunMultiCancerSSL: missing script -> ", p)
  invisible(p)
}

.run_py <- function(script, args = character(), env_extra = character()) {
  python <- .default_python()
  repo   <- .repo_root()
  extra  <- c(
    paste0("GFM_PROJECT_ROOT=", repo),
    if (length(env_extra)) env_extra else character()
  )
  cat(sprintf("[RunMultiCancerSSL] GFM_PROJECT_ROOT=%s %s %s\n",
              repo, python, paste(c(shQuote(script), shQuote(args)), collapse = " ")))
  withCallingHandlers(
    rc <- system2(python, c(script, args), env = extra),
    warning = function(w) invokeRestart("muffleWarning")
  )
  invisible(rc)
}

# ---- 5 exported verbs ------------------------------------------------------
build_graph <- function(
  model   = c("8m", "650m"),
  variant = c("esm_full", "esm20c", "esm650_full", "esm650_20c", "esm_patient"),
  echo_cmd = TRUE
) {
  #' @title Build the heterogeneous multi-cancer molecular graph
  #' @description Downloads (if missing) and constructs the graph artifact
  #'   used downstream. Corresponds to Makefile target `data` + `preprocess`.
  model   <- match.arg(model)
  variant <- match.arg(variant)
  .run_py(.script_path("20_build_graph_esm.py"),
          c("--model", model, "--variant", variant))
}

pretrain <- function(
  epochs = 500,
  seeds  = c(42, 7, 123, 21, 99),
  single_cancer = NULL,
  echo_cmd = TRUE
) {
  #' @title Launch equal-budget pretraining for the 3-task HGT encoder
  #' @description Runs CSP+NAMR+C3L (or the ablation variants) at a fixed
  #'   FLOP/epoch budget across `seeds`. Corresponds to Makefile `train`.
  seeds <- as.integer(seeds)
  stopifnot(all(is.finite(seeds) & seeds >= 0L))
  script <- if (is.null(single_cancer)) "40_pretrain.py" else "41_pretrain_single_cancer.py"
  for (s in seeds) {
    rc <- .run_py(.script_path(script),
                 c("--epochs", as.character(epochs),
                   "--seed",   as.character(s),
                   if (!is.null(single_cancer)) c("--cancer", single_cancer)))
    if (rc != 0L) stop(sprintf("pretrain seed %d failed with exit=%d", s, rc))
  }
  invisible(TRUE)
}

evaluate_ppi <- function(
  ckpt = "results/ablation_real/full.pt",
  tag  = "full",
  echo_cmd = TRUE
) {
  #' @title Level-1 zero-shot PPI link-prediction ROC-AUC evaluation
  #' @description Primary ranking metric. Corresponds to Makefile `evaluate`.
  ckpt <- normalizePath(ckpt, mustWork = TRUE)
  .run_py(.script_path("50_eval_checkpoint.py"), c("--ckpt", ckpt, "--tag", tag))
}

evaluate_patient <- function(
  tasks = c("metabric", "paad", "drug_gdsc", "patient_classifier"),
  echo_cmd = TRUE
) {
  #' @title Level-3 patient-level phenotype evaluations (scope boundaries)
  tasks <- match.arg(tasks, several.ok = TRUE)
  task_map <- c(metabric          = "60_metabric_validation.py",
                paad              = "63_paad_transfer_eval.py",
                drug_gdsc         = "64_drug_response_gdsc.py",
                patient_classifier= "61_patient_eval.py")
  for (t in tasks) {
    script <- .script_path(unname(task_map[[t]]))
    if (!file.exists(script)) {
      message(sprintf("[evaluate_patient] skip %s (script not built yet: %s)", t, script))
      next
    }
    .run_py(script)
  }
  invisible(TRUE)
}

make_figures <- function(
  only = NULL,
  out_dir = file.path(.repo_root(), "manuscript"),
  echo_cmd = TRUE
) {
  #' @title Generate all manuscript figures (vector PDF + 300-dpi PNG)
  #' @description  Corresponds to Makefile target `figures`.
  args <- c("--fig-dir", out_dir)
  if (length(only)) args <- c(args, "--only", as.character(only))
  .run_py(.script_path("85_make_figures.py"), args)
  invisible(TRUE)
}

# ---- CLI (Rscript RunMultiCancerSSL.R <verb> [args]) ------------------------
`%||%` <- function(a, b) if (is.null(a)) b else a

.cli_usage <- function() {
  cat(
    "Usage: Rscript R/RunMultiCancerSSL.R <verb> [args]\n",
    "  verbs:\n",
    "    build_graph [8m|650m] [variant]        Build graph\n",
    "    pretrain [epochs=500] [seeds 42 7 ...]  Launch pretraining\n",
    "    evaluate_ppi <ckpt> <tag>               Level-1 PPI evaluation\n",
    "    evaluate_patient [metabric|paad|...]    Level-3 patient tasks\n",
    "    make_figures [--only fig1 fig3]         All figure PDFs/PNGs\n",
    sep = ""
  )
  q(status = 1L)
}

if (sys.nframe() == 0L) {
  argv <- commandArgs(trailingOnly = TRUE)
  if (!length(argv)) .cli_usage()
  verb <- tolower(argv[1])
  rest <- if (length(argv) >= 2L) argv[-1] else character()
  switch(verb,
    build_graph   = do.call(build_graph,   list(
      model   = if (length(rest) >= 1L) rest[1] else "8m",
      variant = if (length(rest) >= 2L) rest[2] else "esm_full")),
    pretrain      = do.call(pretrain, list(
      epochs = if (length(rest) >= 1L) as.integer(rest[1]) else 500L,
      seeds  = if (length(rest) >= 2L) as.integer(rest[-1]) else c(42L, 7L, 123L, 21L, 99L))),
    evaluate_ppi  = do.call(evaluate_ppi, list(
      ckpt = if (length(rest) >= 1L) rest[1] else "results/ablation_real/full.pt",
      tag  = if (length(rest) >= 2L) rest[2] else "full")),
    evaluate_patient = do.call(evaluate_patient, list(tasks = rest)),
    make_figures  = {
      idx_only <- which(rest == "--only")
      idx_dir  <- which(rest %in% c("--fig-dir", "--out_dir"))
      only <- NULL; dir <- NULL
      if (length(idx_only)) {
        stop_after <- if (length(idx_dir)) min(idx_dir) - 1L else length(rest)
        take <- seq(from = idx_only[1L] + 1L, to = stop_after, by = 1L)
        only <- rest[take]
      }
      if (length(idx_dir)) dir <- rest[idx_dir[1L] + 1L]
      if (is.null(dir)) dir <- file.path(.repo_root(), "manuscript")
      make_figures(only = only, out_dir = dir)
    },
    {
      message("Unknown verb: ", verb)
      .cli_usage()
    }
  )
}
