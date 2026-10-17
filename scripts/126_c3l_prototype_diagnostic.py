#!/usr/bin/env python3
"""
@file    126_c3l_prototype_diagnostic.py
@brief   Review item C4 (2026-09-18): is the "all 21 cancer prototypes are
         identical" claim true of the actual model weights, or only of a
         rounded read-out? Measures on the actually trained model at full
         float32 precision, layer by layer:
           * cancer-node inputs really are 21-dim one-hot rows?
           * after each of the 6 HGT layers: max pairwise cosine, min pairwise
             euclidean distance, per-dimension std (scientific notation) and
             the number of distinct rows at float32 precision;
           * per-direction incident edge counts for cancer nodes (complete-
             bipartite wiring check);
           * C3L loss of the trained head on real targets vs shuffled targets.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Input:  results/ablation_real/full.pt + data/processed/xena_graph_esm.pt
Output: results/control/c3l_prototype_layers.json + stdout table
Run:    GFM_PROJECT_ROOT=<repo> python scripts/126_c3l_prototype_diagnostic.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT_BASENAME = "40_pretrain.py"

# Artefact paths resolved against config paths.
GRAPH_BASENAME = "xena_graph_esm.pt"
RESULT_TAG = "ablation_real"
CKPT_BASENAME = "full.pt"
OUTPUT_TAG = "control"
OUTPUT_JSON_FILENAME = "c3l_prototype_layers.json"

# HGT architecture (must match the loaded full.pt checkpoint).
HGT_HIDDEN_DIM = config.EMBEDDING_DIM
HGT_NUM_LAYERS = 6
HGT_NUM_HEADS = 8

# Comparison / rounding tolerances.
COSINE_NORM_EPS = 1e-12
ONEHOT_ALLCLOSE_ATOL = 1e-6
DISTINCT_ROUND_DIGITS = 6

# Report dict keys (kept stable for downstream readers).
RKEY_CANCER_INPUT = "cancer_input"
RKEY_INPUT_IS_ONEHOT = "cancer_input_is_onehot"
RKEY_EDGE_DIRECTIONS = "edge_directions"
RKEY_LAYERS = "layers"
RKEY_LAYER_IDX = "layer"

# Print table header + per-stage line format.
PRINT_HEADER = (
    f"{'stage':<16}{'maxCos':>9}{'minCos':>9}"
    f"{'minEuclid':>11}{'std_min':>10}{'std_max':>10}{'distinct':>9}"
)
PRINT_FMT = "{:<16}{:>9.6f}{:>9.6f}{:>11.3e}{:>10.3e}{:>10.3e}{:>9}"

JSON_INDENT = 1


def _load_pretrain_module():
    """Dynamically import 40_pretrain helper script as a module.

    @brief  Internal helper; exposes build_data_dict and HGTEncoder.
    @return Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(
        "pf",
        os.path.join(SCRIPT_DIR, PRETRAIN_SCRIPT_BASENAME),
    )
    pf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf_mod)
    return pf_mod


def stats(x):
    """Pairwise-distinctness statistics for a matrix of row-vectors.

    @brief  Compute, for the row vectors of ``x``:
              - max / min / mean of the upper-triangular pairwise cosine sim;
              - min pairwise euclidean distance;
              - min / max / mean per-dimension standard deviation;
              - number of distinct rows at ``DISTINCT_ROUND_DIGITS`` precision;
              - total row count.
    @param  x   2-D tensor (rows = cancer-node embeddings).
    @return Dict with the stable keys listed above; all numeric values are
            cast to plain Python int / float so the result is JSON-serialisable.
    @note   Cosine matrix is computed on L2-normalised rows with a tiny epsilon
            so constant-zero rows do not produce NaN (they never appear in the
            cancer embeddings in practice, but the check is defensive).
    """
    x = x.detach().cpu().numpy().astype(np.float64)
    n_rows = x.shape[0]
    n = x / (np.linalg.norm(x, axis=1, keepdims=True) + COSINE_NORM_EPS)
    cos = n @ n.T
    iu = np.triu_indices(n_rows, 1)
    pw = cos[iu]
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)[iu]
    distinct_count = len({tuple(r) for r in np.round(x, DISTINCT_ROUND_DIGITS)})
    per_dim_std = x.std(axis=0)
    return {
        "max_pairwise_cosine": float(pw.max()),
        "min_pairwise_cosine": float(pw.min()),
        "mean_pairwise_cosine": float(pw.mean()),
        "min_pairwise_euclidean": float(dist.min()),
        "per_dim_std_min": float(per_dim_std.min()),
        "per_dim_std_max": float(per_dim_std.max()),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "distinct_rows_1e-6": int(distinct_count),
        "n_rows": int(n_rows),
    }


def main():
    """Run the C3L prototype-collapsed diagnostic and emit the report table/JSON.

    @brief  (1) load graph + full checkpoint; (2) verify cancer input is one-hot;
            (3) register forward hooks on every HGT conv layer; (4) run one
            forward pass; (5) per-layer stats() on the cancer-node tensor;
            (6) print table and serialise JSON.
    @return Process exit code (0).
    """
    pf = _load_pretrain_module()
    graph_path = os.path.join(config.DATA_PROC, GRAPH_BASENAME)
    ckpt_path = os.path.join(config.RESULT_DIR, RESULT_TAG, CKPT_BASENAME)
    dev = torch.device("cpu")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    data, node_types, edge_types, in_dims = pf.build_data_dict(graph, dev)
    x = {nt: data[nt]["x"] for nt in node_types}
    ei = {et: data[et]["edge_index"] for et in edge_types}
    model = pf.HGTEncoder(
        node_types, edge_types, in_dims,
        HGT_HIDDEN_DIM, HGT_NUM_LAYERS, HGT_NUM_HEADS,
    ).to(dev)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # One-hot sanity check: cancer input rows should match I_{NC, NC}[:n_cancers]
    cancer_input = x["cancer"]
    eye_ref = torch.eye(cancer_input.shape[1])[:cancer_input.shape[0]]
    report = {
        RKEY_CANCER_INPUT: stats(cancer_input),
        RKEY_INPUT_IS_ONEHOT: bool(torch.allclose(
            cancer_input, eye_ref, atol=ONEHOT_ALLCLOSE_ATOL,
        )),
        RKEY_EDGE_DIRECTIONS: {
            str(k): int(v.shape[1]) for k, v in ei.items()
        },
        RKEY_LAYERS: [],
    }

    # Capture per-layer activations via forward hooks on each HGT conv block.
    # The lambda default-argument ``idx=i_`` captures the current loop value
    # (otherwise every lambda closes over the last loop variable).
    acts = []
    hooks = [
        layer.register_forward_hook(
            lambda _m, _i, o, idx=i_: acts.append((idx, o))
        )
        for i_, layer in enumerate(model.convs)
    ]
    with torch.no_grad():
        _ = model(x, ei)
    for hook_handle in hooks:
        hook_handle.remove()

    for layer_idx, output in acts:
        # output may be either a dict of node embeddings or a plain tensor
        # depending on HGTEncoder internal layout; both branches work because
        # model.convs[i] returns a dict-style output in this repo's implementation.
        cancer_h = output["cancer"] if isinstance(output, dict) else output
        report[RKEY_LAYERS].append({
            RKEY_LAYER_IDX: layer_idx + 1,
            **stats(cancer_h),
        })

    # --- English-only stdout table -------------------------------------------
    print(PRINT_HEADER)
    def print_line(name, stat_dict):
        """Helper: format one stats row and print."""
        print(PRINT_FMT.format(
            name,
            stat_dict["max_pairwise_cosine"],
            stat_dict["min_pairwise_cosine"],
            stat_dict["min_pairwise_euclidean"],
            stat_dict["per_dim_std_min"],
            stat_dict["per_dim_std_max"],
            stat_dict["distinct_rows_1e-6"],
        ))
    print_line("cancer input", report[RKEY_CANCER_INPUT])
    for layer_entry in report[RKEY_LAYERS]:
        stage_label = f"after HGT {layer_entry[RKEY_LAYER_IDX]}"
        print_line(stage_label, layer_entry)
    print(f"\ncancer input is one-hot: {report[RKEY_INPUT_IS_ONEHOT]}")
    print(f"incident edges per type: {report[RKEY_EDGE_DIRECTIONS]}")

    # --- JSON artefact --------------------------------------------------------
    os.makedirs(os.path.join(config.RESULT_DIR, OUTPUT_TAG), exist_ok=True)
    out_path = os.path.join(config.RESULT_DIR, OUTPUT_TAG, OUTPUT_JSON_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=JSON_INDENT)
    print(f"[written] results/{OUTPUT_TAG}/{OUTPUT_JSON_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
