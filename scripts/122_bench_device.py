#!/usr/bin/env python3
"""
@file    122_bench_device.py
@brief   Device micro-benchmark documenting the HGT encoder's forward-pass
         latency on CPU vs Apple MPS, plus the numpy/BLAS thread-throughput
         sweep under 1 vs 16 OpenBLAS threads (mirrors the cosine-known-edge
         probe workload used downstream).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalised 2026-09)
@ingroup audit_and_diagnostics

Input:  config.DATA_PROC/<GRAPH_FILE> and config.result_path("ablation_real", <CKPT_FILE>)
Output: stdout only (tabular latency numbers; nothing is serialised)
Run:    GFM_PROJECT_ROOT=<repo> python scripts/122_bench_device.py
"""
import importlib.util
import os
import time

import numpy as np
import torch

import config

# =============================================================================
# File-level constants
# =============================================================================
# Artifact basenames resolved against the config module paths.
GRAPH_FILE_BASENAME = "xena_graph_esm.pt"       # encoder input (processed graph)
FULL_CKPT_BASENAME = "full.pt"                   # full ablation checkpoint

# Helper script (loaded via importlib because 40_pretrain is not installed as
# a package; the scripts/ directory is self-contained).
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# --- HGT-encoder architecture hyper-parameters (must match the checkpoint) ---
HGT_HIDDEN_DIM = 256
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# --- Benchmark workload sizes ------------------------------------------------
HGT_N_TIMED_ITERATIONS = 20             # forward passes averaged per device
HGT_N_WARMUP_ITERATIONS = 3             # warmup (discarded) before timing
SYNTH_BENCH_N_ROWS = 5000               # gene-count proxy for the BLAS sweep
SYNTH_BENCH_N_PAIRS = config.N_PPI_PAIRS  # probe pair count for the BLAS sweep
BLAS_N_REPEATS = 30                     # averages per OpenBLAS thread setting
BLAS_THREAD_SETTINGS = ("1", "16")      # thread counts to sweep

# Numeric constants
NORM_EPSILON = 1e-9                     # L2-normalisation floor for cosine
MILLIS_PER_SECOND = 1000.0              # seconds -> milliseconds conversion
DEFAULT_MAP_DEVICE = "cpu"              # graph weights loaded on CPU first


def _load_pretrain_module():
    """Dynamically import 40_pretrain.py from the sibling scripts/ directory.

    @brief  Use ``importlib.util`` to treat the sibling script as a module
            without requiring scripts/ to be on ``sys.path`` or installed.
    @return The loaded module object (exposes ``build_data_dict`` and
            ``HGTEncoder`` used by the benchmark).
    @note   Internal helper; intentionally not exported.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(script_dir, PRETRAIN_SCRIPT_BASENAME),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(device):
    """Build the HGT encoder graph and load pretrained weights onto ``device``.

    @brief  (1) load the helper module; (2) load the heterogeneous graph;
            (3) convert to the PyG-style data dict / node_types / edge_types
            representation; (4) instantiate HGTEncoder with the matching
            architecture; (5) restore the checkpoint's ``model`` state_dict;
            (6) switch to eval() mode and return the callable plus its inputs.
    @param  device   torch.device (CPU or MPS in practice).
    @return Tuple ``(model, x_dict, ei_dict)`` ready for forward calls.
    """
    pf = _load_pretrain_module()
    graph = torch.load(
        os.path.join(config.DATA_PROC, GRAPH_FILE_BASENAME),
        map_location=DEFAULT_MAP_DEVICE,
        weights_only=False,
    )
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, device)
    x_dict = {nt: data[nt]["x"] for nt in node_types}
    ei_dict = {et: data[et]["edge_index"] for et in edge_types}
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(device)
    ckpt = torch.load(
        config.result_path("ablation_real", FULL_CKPT_BASENAME),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, x_dict, ei_dict


def bench(device, n_iter=HGT_N_TIMED_ITERATIONS):
    """Measure HGT forward-pass latency on the given device.

    @brief  Run ``HGT_N_WARMUP_ITERATIONS`` of warm up (discarded) to pay JIT
            and allocation costs, then time ``n_iter`` forward passes and
            return the per-iteration wall-clock time in seconds.
    @param  device   torch.device to benchmark on.
    @param  n_iter   Number of timed iterations (default: file-level constant).
    @return Per-iteration seconds (float).
    @note   For the MPS backend the function calls ``torch.mps.synchronize()``
            immediately after the loop; without it the time measurement would
            capture only the enqueue cost, not the actual kernel execution.
    """
    model, x_dict, ei_dict = load_model(device)
    with torch.no_grad():
        for _ in range(HGT_N_WARMUP_ITERATIONS):
            # Warm-up: JIT / memory allocator / MPS command buffer priming.
            model(x_dict, ei_dict)
        t0 = time.time()
        for _ in range(n_iter):
            model(x_dict, ei_dict)
        if device.type == "mps":
            # MPS kernels are asynchronous: flush the stream before the clock.
            torch.mps.synchronize()
        dt = (time.time() - t0) / n_iter
    return dt


def _run_blas_sweep():
    """Run the OpenBLAS thread sweep over SYNTH_BENCH_N_ROWS cosine pairs.

    @brief  Internal helper extracted from the __main__ block for readability;
            sets OPENBLAS_NUM_THREADS / OMP_NUM_THREADS via ``os.environ``
            (BLAS implementations typically read these at import time, but for
            OpenBLAS built with the runtime tunable the env read happens inside
            each BLAS call, so resetting it here works as expected).
    @return None (results are printed to stdout).
    @note   Internal helper; not exported.
    """
    rng = np.random.RandomState(config.SEED)
    X = rng.randn(SYNTH_BENCH_N_ROWS, config.EMBEDDING_DIM).astype(np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + NORM_EPSILON)
    pairs = rng.randint(0, SYNTH_BENCH_N_ROWS, size=(SYNTH_BENCH_N_PAIRS, 2))
    header = (
        f"=== numpy/BLAS thread comparison "
        f"({SYNTH_BENCH_N_ROWS}x{config.EMBEDDING_DIM} cosine, "
        f"{SYNTH_BENCH_N_PAIRS} pairs) ==="
    )
    print(header)
    for num_threads in BLAS_THREAD_SETTINGS:
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["OMP_NUM_THREADS"] = num_threads
        t0 = time.time()
        for _ in range(BLAS_N_REPEATS):
            # Pure list-comprehension cosine: mirrors 50_eval_checkpoint.py's
            # known-edge probe exactly (no BLAS-level vectorisation shortcut).
            _ = np.array([float(Xn[a] @ Xn[b]) for a, b in pairs])
        dt_ms = (time.time() - t0) / BLAS_N_REPEATS * MILLIS_PER_SECOND
        print(f"OPENBLAS_NUM_THREADS={num_threads}: {dt_ms:.1f} ms/run")


if __name__ == "__main__":
    # --- HGT encoder section (English-only stdout) ----------------------------
    print("=== HGT forward inference (seconds per pass) ===")
    t_cpu = bench(torch.device("cpu"))
    print(f"CPU : {t_cpu * MILLIS_PER_SECOND:.1f} ms/iter")
    if torch.backends.mps.is_available():
        t_mps = bench(torch.device("mps"))
        speedup = t_cpu / t_mps
        print(
            f"MPS : {t_mps * MILLIS_PER_SECOND:.1f} ms/iter  "
            f"(speedup {speedup:.1f}x; note: shares GPU with concurrent training)"
        )
    print()
    # --- Numpy / BLAS sweep ----------------------------------------------------
    _run_blas_sweep()
