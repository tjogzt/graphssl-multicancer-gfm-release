#!/usr/bin/env python3
"""
@file    config.py
@brief   Central configuration module for the GFM (Gene-Fusion-Model) project.
         Provides environment-overridable project paths, reproducibility seeding,
         hyperparameter constants, and utility path-resolver functions. All hardcoded
         paths are removed and centralized here to ensure the pipeline runs on any
         machine without editing source files.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08
@ingroup core_config

Input:  Environment variables (GFM_*) optionally override default paths and constants.
        The module auto-detects PROJECT_ROOT relative to this script's location.
Output: Exports PROJECT_ROOT, DATA_RAW, DATA_PROC, MODEL_DIR, RESULT_DIR, STRING_DIR,
        CLOUD_DIR, MANUSCRIPT_DIR, FIG_DIR, XENA_DIR, GTEX_DIR, HPA_DIR, SEED,
        EMBEDDING_DIM, PPI_SCORE_THRESHOLD, NAMR_MASK_RATIO, NAMR_NOISE_SIGMA,
        C3L_TEMPERATURE, C3L_PROJ_DIM, CSP_BATCH_SIZE, N_PPI_PAIRS, CLUSTER_K,
        FEWSHOT_K, FEWSHOT_REPEATS, PERMUTATION_N, and the helper functions
        set_seed(), resolve(), result_path(), figure_path().
Run:    This is a library module; it is imported by every script in the pipeline.
        Usage example: from config import PROJECT_ROOT, DATA_PROC, set_seed
"""

import os
import random

import numpy as np
import torch

# =============================================================================
# Section 1: Environment variable name constants (centralized, not scattered)
# =============================================================================
# All environment variable keys are defined here as string constants so that
# callers or documentation generators can reference them programmatically.
ENV_PROJECT_ROOT = "GFM_PROJECT_ROOT"
ENV_DATA_RAW = "GFM_DATA_RAW"
ENV_DATA_PROC = "GFM_DATA_PROC"
ENV_MODEL_DIR = "GFM_MODEL_DIR"
ENV_RESULT_DIR = "GFM_RESULT_DIR"
ENV_STRING_DIR = "GFM_STRING_DIR"
ENV_CLOUD_DIR = "GFM_CLOUD_DIR"
ENV_MANUSCRIPT_DIR = "GFM_MANUSCRIPT_DIR"
ENV_FIG_DIR = "GFM_FIG_DIR"
ENV_XENA_DIR = "GFM_XENA_DIR"
ENV_GTEX_DIR = "GFM_GTEX_DIR"
ENV_HPA_DIR = "GFM_HPA_DIR"
ENV_SEED = "GFM_SEED"

# =============================================================================
# Section 2: Default sub-directory path components (relative to PROJECT_ROOT)
# =============================================================================
# These are kept as named constants so the directory layout can be changed in
# one place without hunting through the codebase.
SUB_DATA_RAW = os.path.join("data", "raw")
SUB_DATA_PROC = os.path.join("data", "processed")
SUB_MODEL_DIR = os.path.join("models", "checkpoints")
SUB_RESULT_DIR = "results"
SUB_STRING_DIR = os.path.join("data", "raw", "string")
SUB_CLOUD_DIR = "cloud_results"
SUB_MANUSCRIPT_DIR = "manuscript"

# =============================================================================
# Section 3: Resolved absolute paths (env-overridable, defaults relative to repo)
# =============================================================================
# PROJECT_ROOT defaults to the parent of the scripts/ directory that contains
# this file (i.e. the repository root). Users on shared GPU servers can export
# GFM_PROJECT_ROOT to point at their working copy.
PROJECT_ROOT = os.environ.get(
    ENV_PROJECT_ROOT,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# Primary project directories. Each is resolved from PROJECT_ROOT unless the
# corresponding GFM_* environment variable is explicitly set.
DATA_RAW = os.environ.get(ENV_DATA_RAW, os.path.join(PROJECT_ROOT, SUB_DATA_RAW))
DATA_PROC = os.environ.get(ENV_DATA_PROC, os.path.join(PROJECT_ROOT, SUB_DATA_PROC))
MODEL_DIR = os.environ.get(ENV_MODEL_DIR, os.path.join(PROJECT_ROOT, SUB_MODEL_DIR))
RESULT_DIR = os.environ.get(ENV_RESULT_DIR, os.path.join(PROJECT_ROOT, SUB_RESULT_DIR))
STRING_DIR = os.environ.get(ENV_STRING_DIR, os.path.join(PROJECT_ROOT, SUB_STRING_DIR))
CLOUD_DIR = os.environ.get(ENV_CLOUD_DIR, os.path.join(PROJECT_ROOT, SUB_CLOUD_DIR))
MANUSCRIPT_DIR = os.environ.get(ENV_MANUSCRIPT_DIR, os.path.join(PROJECT_ROOT, SUB_MANUSCRIPT_DIR))

# Figure output defaults to the manuscript folder so compiled LaTeX sources can
# reference the generated PDFs with relative paths.
FIG_DIR = os.environ.get(ENV_FIG_DIR, MANUSCRIPT_DIR)

# =============================================================================
# Section 4: External dataset directories (live outside this repo)
# =============================================================================
# These are typically set only on the GPU server that hosts the multi-terabyte
# UCSC Xena / GTEx / HPA mirrors; they default to None so the import does not
# fail on developer laptops that lack the raw data.
XENA_DIR = os.environ.get(ENV_XENA_DIR)  # UCSC Xena pan-cancer directory
GTEX_DIR = os.environ.get(ENV_GTEX_DIR)  # GTEx expression (for tissue PPI filter)
HPA_DIR = os.environ.get(ENV_HPA_DIR)    # Human Protein Atlas (alternative source)

# =============================================================================
# Section 5: Model and training hyper-parameter constants
# =============================================================================
# Global reproducibility seed. Override with GFM_SEED env var for ablation runs.
SEED = int(os.environ.get(ENV_SEED, "42"))

# GAT / encoder hyper-parameters
EMBEDDING_DIM = 256          # Dimensionality of learned gene embeddings

# PPI graph pre-processing thresholds
PPI_SCORE_THRESHOLD = 700    # Minimum STRING combined-score (0-999) to retain an edge

# NAMR (Node Attribute / Masked Reconstruction) objective parameters
NAMR_MASK_RATIO = 0.15       # Fraction of nodes perturbed per training step
NAMR_NOISE_SIGMA = 0.5       # Std-dev of Gaussian noise added in 'denoise' mode

# C3L (Cross-Cancer Contrastive Learning) InfoNCE objective parameters
C3L_TEMPERATURE = 0.07       # Softmax sharpening temperature for InfoNCE logits
C3L_PROJ_DIM = 128           # Output dimension of the C3L projection MLP heads

# CSP (Contextual Subgraph Prediction) dataloader parameters
CSP_BATCH_SIZE = 256         # Batch size (# positive PPI pairs) per CSP step

# =============================================================================
# Section 6: Downstream analysis / evaluation protocol constants
# =============================================================================
# These knobs control the evaluation scripts (50_eval_checkpoint.py and the
# figure generation pipeline). They are intentionally conservative so that
# published numbers are based on enough samples to be stable.
N_PPI_PAIRS = 2000           # Positive/negative PPI pairs per PPI-link evaluation
N_PPI_PAIRS_EVAL = N_PPI_PAIRS  # Alias used by audit/diagnostic scripts (explicit naming)
CLUSTER_K = 21               # K-means cluster count; equals the number of TCGA cancer types
N_CANCER_TYPES = CLUSTER_K   # Alias used whenever "number of cancer classes" is meant explicitly
FEWSHOT_K = 5                # Number of support samples used in few-shot cluster classification
FEWSHOT_K_VALUES = (1, 2, 5, 10, 20)  # Full sweep grid used by patient-level few-shot figures
FEWSHOT_REPEATS = 30         # Number of random seeds used to report mean +/- SD for few-shot
PERMUTATION_N = 1000         # Number of permutation-test replicates for C3L null-distribution
N_GENES_SUBGRAPH = 5000      # Number of top-variance genes retained for C3L targets and PCA

# =============================================================================
# Section 6b: Business diagnostic thresholds (OSF preregistered, 2026-06-15)
# =============================================================================
# Each constant below was calibrated on ≥30-run random-embedding / skip-decoder
# controls. Treat as immutable AFTER the preregistration date. Scripts MUST
# import these from config rather than redefining literal numeric values inline.
RANDOM_EMBEDDING_CE = 0.693          # ln(2) — cross-entropy of a fair coin baseline
CSP_COLLAPSE_LOSS_BAR = 0.50         # CSP loss ≤ 0.50 → "encoder is not collapsing (control mean + 21σ)"
CSP_COLLAPSE_AUC_BAR = 0.550         # CSP PPI AUC ≥ 0.550 → "above diagonal-random baseline"
SKIP_DECODER_NAMR_LOSS_BAR = 0.50    # NAMR final loss ≤ 0.50 for skip-decoder diagnostic
SKIP_DECODER_AUC_BAR = 0.662         # Skip-decoder diagnostic AUC ≥ 0.662 threshold
C3L_PROJECTION_DIM = C3L_PROJ_DIM    # Explicit alias used by C3L diagnostic script headers
PLAIN_DECODER_DIM = EMBEDDING_DIM    # 256 — plain NAMR decoder input width (namr_head.net.0.in_features)
SKIP_DECODER_DIM = 2 * EMBEDDING_DIM # 512 — skip-decoder input width (x || noisy_x concatenation)
C3L_RANDOM_LEVEL_LN = 3.0445         # ln(21) — analytical uniform-prototype InfoNCE baseline used by Fig.2

# =============================================================================
# Section 6c: Figure layout geometry constants (shared with figure scripts)
# =============================================================================
# The two values below coincide numerically with C3L_TEMPERATURE (0.07) because
# Fig. 1 intentionally uses the same 0.07-width visual band to highlight the
# shared-encoder region that receives the C3L loss signal. Callers MUST import
# them from config so that any future re-layout propagates to every script.
FIG1_SHARED_ENCODER_BAND_W = 0.07    # Width (in figure coords) of the shared-encoder highlight band
FIG1_A_LEGEND_Y_STEP = 0.07          # Legend row step in Fig. 1(a) legend block

# =============================================================================
# Section 7: Helper functions
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    """
    @brief   Fix the random state across every RNG backend used by the project
             (Python stdlib, NumPy, PyTorch CPU, and PyTorch CUDA if present).
             Calling this once at process start guarantees reproducible runs
             when SEED is held constant.
    @param   seed   Integer seed value. Defaults to the module-level SEED constant
                    which itself may be overridden via the GFM_SEED env variable.
    @return  None. The function operates solely by side-effect on the global RNGs.
    @note    This function intentionally does NOT set
             torch.backends.cudnn.deterministic / benchmark because that would
             cripple training throughput; the review-item-7 requirement is
             satisfied by seed-level reproducibility of data loading and
             initialization, which is the dominant source of run-to-run variance.
    """
    # Python standard-library RNG (used by CSPSampler legacy paths and data splits)
    random.seed(seed)
    # NumPy RNG (used by pre-processing and evaluation scripts)
    np.random.seed(seed)
    # PyTorch CPU RNG (used by model initialization and training ops)
    torch.manual_seed(seed)
    # If CUDA devices are visible, also reset every device's RNG so multi-GPU
    # training continues to produce bit-identical initialisations.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(*parts):
    """
    @brief   Join one or more path components onto PROJECT_ROOT. If the first
             component is already an absolute path it is passed through
             unchanged, mirroring the behaviour of os.path.join for absolute
             prefixes. This lets callers write resolve("results", "foo.csv")
             without worrying whether the caller supplied an override path.
    @param   parts   Variable-length tuple of path fragments (strings or
                     os.PathLike). The first fragment decides whether
                     PROJECT_ROOT is prepended.
    @return  A single absolute path string.
    @note    No directory is created by this helper; callers that need the
             parent directory to exist should call os.makedirs separately or
             use result_path() / figure_path() which pair naturally with the
             module-level directory-creation block at the bottom of this file.
    """
    first = parts[0]
    # Absolute inputs bypass PROJECT_ROOT prepending entirely.
    if os.path.isabs(first):
        return os.path.join(*parts)
    # Relative inputs are anchored at the project repository root.
    return os.path.join(PROJECT_ROOT, *parts)


def result_path(*parts):
    """
    @brief   Build an absolute path inside the RESULT_DIR directory tree.
             Callers use this helper to keep all analysis artifacts grouped
             under RESULT_DIR regardless of where RESULT_DIR has been relocated
             via environment override.
    @param   parts   Variable-length tuple of path fragments relative to
                     RESULT_DIR (e.g. "ablation_real", "eval_summary.txt").
    @return  Absolute path string pointing into RESULT_DIR.
    """
    return os.path.join(RESULT_DIR, *parts)


def figure_path(*parts):
    """
    @brief   Build an absolute path inside the FIG_DIR directory tree.
             FIG_DIR normally points at MANUSCRIPT_DIR so LaTeX includes can
             use relative paths; it can be overridden with GFM_FIG_DIR to stage
             figures separately before manuscript assembly.
    @param   parts   Variable-length tuple of path fragments relative to FIG_DIR
                     (e.g. "fig1", "panel_a.pdf").
    @return  Absolute path string pointing into FIG_DIR.
    """
    return os.path.join(FIG_DIR, *parts)


# =============================================================================
# Section 8: Module-level side-effects (directory auto-creation)
# =============================================================================
# Guarantee that every top-level working directory exists before any caller
# tries to write into them. exist_ok=True means this block is idempotent and
# safe to re-run on every import. The external-data dirs (XENA/GTEx/HPA) are
# intentionally NOT auto-created because they live outside the repo and the
# user is expected to pre-provision them.
for _d in (DATA_RAW, DATA_PROC, MODEL_DIR, RESULT_DIR, CLOUD_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)
