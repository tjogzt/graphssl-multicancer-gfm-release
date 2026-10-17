#!/usr/bin/env Rscript
#' @file    11_download_tcga_xena.R
#' @brief   Download TCGA multi-omics data (mRNA expression and clinical tables)
#'          for cancer cohorts via TCGAbiolinks and cache as per-cohort .rds files.
#' @author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
#' @date    2026-08 (normalized 2026-09)
#' @ingroup data_download
#'
#' Phase 1.1: download TCGA multi-omics data (mRNA expression and clinical tables)
#' for a list of cancer cohorts via TCGAbiolinks. The downloaded objects are cached
#' as per-cohort .rds files so later graph-building phases can load them offline.
#'
#' Input:  optional CLI argument = comma-separated TCGA cohort abbreviations
#'         (defaults to a 20-cohort list); the destination directory honours the
#'         GFM_PROJECT_ROOT / GFM_DATA_RAW environment overrides (mirror of
#'         scripts/config.py PROJECT_ROOT / DATA_RAW).
#' Output: per-cohort <DATA_RAW>/tcga/expression/<cohort>_expression.rds and
#'         <DATA_RAW>/tcga/clinical/<cohort>_clinical.rds, plus progress on stdout.
#' Run:    Rscript scripts/11_download_tcga_xena.R [BRCA,LUAD,...]
#' @note   2026-09-26 R 4.3.0+ deprecated-parameter migration audit.
#'         Static inspection via `formalArgs(TCGAbiolinks::GDCquery)` under
#'         R 4.5.0 + TCGAbiolinks 2.36.0 confirms that the four named
#'         arguments used in this script — `data.category`, `data.type`,
#'         `workflow.type`, and `sample.type` — remain valid and are NOT
#'         flagged as deprecated.  The lintr 3.3.0 CRAN release no longer
#'         exports `deprecated_linter()`, so equivalence was verified via
#'         `formalArgs()` + `parse()` instead; no parameter substitution is
#'         required.
#'

# --- Load required packages ---
# TCGAbiolinks: official GDC query/download interface
# SummarizedExperiment: S4 container returned by GDCprepare for expression data
suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
})

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================

#' Default TCGA cohort list used when no CLI argument is supplied.
#' These 20 cohorts represent the standard protocol default set.
DEFAULT_CANCER_TYPES <- c(
  "BRCA", "LUAD", "LUSC", "COAD", "PRAD", "HNSC", "KIRC", "LGG", "STAD", "BLCA",
  "LIHC", "UCEC", "CESC", "OV", "GBM", "THCA", "SKCM", "SARC", "LAML", "PAAD"
)

# Minimum sleep duration (seconds) between cohorts to rate-limit GDC API calls
# and avoid triggering server-side throttling or temporary bans.
API_RATE_LIMIT_SECONDS <- 1

# Relative subdirectory names under the data root (mirrors config.py layout).
SUBDIR_TCGA_ROOT <- "tcga"
SUBDIR_EXPRESSION <- "expression"
SUBDIR_MUTATION   <- "mutation"
SUBDIR_METHYLATION<- "methylation"
SUBDIR_CLINICAL   <- "clinical"

# Output file suffix pattern for saved RDS objects.
SUFFIX_EXPRESSION_RDS <- "_expression.rds"
SUFFIX_CLINICAL_RDS   <- "_clinical.rds"

# TCGA GDC query parameters (standard protocol values).
QUERY_DATA_CATEGORY <- "Transcriptome Profiling"
QUERY_DATA_TYPE     <- "Gene Expression Quantification"
QUERY_WORKFLOW_TYPE <- "STAR - Counts"
QUERY_SAMPLE_TYPES  <- c("Primary Tumor", "Solid Tissue Normal")

# =========================================================================
# Helper functions
# =========================================================================

#' @brief   Resolve the absolute project root directory from env vars or script path.
#'
#' @param   script_self  Path to this R script (as detected from commandArgs).
#' @return  Absolute path to the project root directory (character string).
#' @note    Mirrors the behaviour of scripts/config.py PROJECT_ROOT:
#'          prefer GFM_PROJECT_ROOT env var, otherwise derive from script location.
resolve_project_root <- function(script_self) {
  # Environment variable override takes highest precedence.
  root_from_env <- Sys.getenv("GFM_PROJECT_ROOT", unset = NA)
  if (!is.na(root_from_env) && root_from_env != "") {
    return(normalizePath(root_from_env, mustWork = FALSE))
  }
  # Derive root from script location: script is at <root>/scripts/<self>.
  return(normalizePath(file.path(dirname(script_self), "..")))
}

#' @brief   Resolve the raw-data root directory from env vars or project-root convention.
#'
#' @param   project_root  Absolute path to the project root.
#' @return  Absolute path to the raw data directory (character string).
#' @note    Mirrors config.DATA_RAW: prefer GFM_DATA_RAW env var, else <root>/data/raw.
resolve_data_raw_root <- function(project_root) {
  data_raw_from_env <- Sys.getenv("GFM_DATA_RAW", unset = NA)
  if (!is.na(data_raw_from_env) && data_raw_from_env != "") {
    return(normalizePath(data_raw_from_env, mustWork = FALSE))
  }
  return(file.path(project_root, "data", "raw"))
}

#' @brief   Detect this script's own filesystem path from commandArgs.
#'
#' @return  Path string of the currently running R script; falls back to the
#'          hard-coded relative path "scripts/11_download_tcga_xena.R" if the
#'          --file= flag cannot be parsed (e.g. when sourced interactively).
detect_script_path <- function() {
  args_all <- commandArgs(FALSE)
  # Rscript passes script path via --file=<path> when invoked non-interactively.
  file_match <- grep("^--file=", args_all, value = TRUE)
  if (length(file_match) >= 1) {
    return(sub("^--file=", "", file_match[1]))
  }
  return("scripts/11_download_tcga_xena.R")
}

#' @brief   Create (recursively) all required output subdirectories.
#'
#' @param   base_dir  Absolute path to the TCGA root directory.
#' @return  Named logical vector indicating creation success for each subdir.
#' @note    Existing directories are silently left untouched (showWarnings=FALSE).
ensure_output_dirs <- function(base_dir) {
  subdirs <- c(
    expression  = file.path(base_dir, SUBDIR_EXPRESSION),
    mutation    = file.path(base_dir, SUBDIR_MUTATION),
    methylation = file.path(base_dir, SUBDIR_METHYLATION),
    clinical    = file.path(base_dir, SUBDIR_CLINICAL)
  )
  return(vapply(subdirs, dir.create,
                logical(1), recursive = TRUE, showWarnings = FALSE))
}

#' @brief   Download and cache the mRNA expression SummarizedExperiment for one cohort.
#'
#' @param   cohort      TCGA cohort abbreviation (e.g. "BRCA").
#' @param   output_dir  TCGA root directory (contains expression/ subdir).
#' @return  Invisibly returns the number of samples on success; NA on error.
#' @note    Errors are caught and printed so one failing cohort does not abort
#'          the entire download loop.
download_expression <- function(cohort, output_dir) {
  tryCatch({
    # Build a GDC query targeting STAR-Counts expression for Tumor + Normal samples.
    query <- GDCquery(
      project      = paste0("TCGA-", cohort),
      data.category = QUERY_DATA_CATEGORY,
      data.type    = QUERY_DATA_TYPE,
      workflow.type = QUERY_WORKFLOW_TYPE,
      sample.type  = QUERY_SAMPLE_TYPES
    )
    # Download raw files into the per-cohort expression directory.
    GDCdownload(query, directory = file.path(output_dir, SUBDIR_EXPRESSION))
    # Prepare (assemble) the SummarizedExperiment from downloaded raw files.
    exp <- GDCprepare(query, directory = file.path(output_dir, SUBDIR_EXPRESSION))
    # Persist the prepared object so downstream phases can load it offline.
    saveRDS(exp, file.path(output_dir, SUBDIR_EXPRESSION,
                           paste0(cohort, SUFFIX_EXPRESSION_RDS)))
    cat(sprintf("  Expression: %d samples saved\n", ncol(exp)))
    return(invisible(ncol(exp)))
  }, error = function(e) {
    cat(sprintf("  Expression ERROR: %s\n", e$message))
    return(invisible(NA_integer_))
  })
}

#' @brief   Download and cache the clinical data.frame for one cohort.
#'
#' @param   cohort      TCGA cohort abbreviation (e.g. "BRCA").
#' @param   output_dir  TCGA root directory (contains clinical/ subdir).
#' @return  Invisibly returns the number of patients on success; NA on error.
download_clinical <- function(cohort, output_dir) {
  tryCatch({
    # GDCquery_clinic returns a plain data.frame of patient-level clinical fields.
    clin_query <- GDCquery_clinic(
      project = paste0("TCGA-", cohort),
      type    = "clinical"
    )
    saveRDS(clin_query, file.path(output_dir, SUBDIR_CLINICAL,
                                  paste0(cohort, SUFFIX_CLINICAL_RDS)))
    cat(sprintf("  Clinical: %d patients saved\n", nrow(clin_query)))
    return(invisible(nrow(clin_query)))
  }, error = function(e) {
    cat(sprintf("  Clinical ERROR: %s\n", e$message))
    return(invisible(NA_integer_))
  })
}

#' @brief   Parse CLI arguments and return the list of cohorts to process.
#'
#' @return  Character vector of cohort abbreviations.
#' @note    If no argument is provided, returns DEFAULT_CANCER_TYPES.
#'          A single comma-separated string is split on ",".
parse_cohort_args <- function() {
  trailing_args <- commandArgs(trailingOnly = TRUE)
  if (length(trailing_args) > 0) {
    return(strsplit(trailing_args[1], ",")[[1]])
  }
  return(DEFAULT_CANCER_TYPES)
}

# =========================================================================
# Main entry
# =========================================================================

#' @brief   Main driver: resolve paths, create dirs, iterate cohorts, sleep between.
#'
#' @return  Invisibly returns NULL on completion.
main <- function() {
  # --- Stage 1: resolve configuration and prepare filesystem ---
  script_self <- detect_script_path()
  project_root <- resolve_project_root(script_self)
  data_raw_root <- resolve_data_raw_root(project_root)
  tcga_data_dir <- file.path(data_raw_root, SUBDIR_TCGA_ROOT)

  # Ensure the full output directory tree exists before downloads start.
  dir.create(tcga_data_dir, recursive = TRUE, showWarnings = FALSE)
  ensure_output_dirs(tcga_data_dir)

  # --- Stage 2: determine cohort list from CLI or defaults ---
  cancer_types <- parse_cohort_args()

  cat(sprintf("Phase 1.1 TCGA download\n"))
  cat(sprintf("  Project root : %s\n", project_root))
  cat(sprintf("  Output dir   : %s\n", tcga_data_dir))
  cat(sprintf("  Cohorts (%d) : %s\n", length(cancer_types),
              paste(cancer_types, collapse = ", ")))

  # --- Stage 3: per-cohort download loop (expression + clinical) ---
  for (ca in cancer_types) {
    cat(sprintf("\n=== Processing cohort: %s ===\n", ca))
    download_expression(ca, tcga_data_dir)
    download_clinical(ca, tcga_data_dir)
    # Rate-limit between cohorts to avoid hammering the GDC API.
    Sys.sleep(API_RATE_LIMIT_SECONDS)
  }

  cat("\nPhase 1.1 TCGA download complete.\n")
  return(invisible(NULL))
}

# Guard so main() only runs when this file is executed as a script,
# not when it is sourced from another R session.
if (!interactive()) {
  main()
}
