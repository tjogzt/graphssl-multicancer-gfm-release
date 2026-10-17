#!/usr/bin/env python3
"""
@file    132_audit_checkpoints.py
@brief   Checkpoint identity audit, closing review issue N4 ("no two different
         models may share one file name").  Every .pt file under results/ is
         SHA-256 hashed, the decoder that each checkpoint ACTUALLY contains is
         MEASURED from the checkpoint's own NAMR head (NAMR first linear
         in_features == 256 plain / 512 skip-connected) instead of being
         inferred from the file name, and each run's unique identifier is read
         from the checkpoint's embedded ``args.out`` (which file names lack).
         Also reports basename collisions (same name, different SHA-256) --
         this is the specific failure mode that made ``namr_only_s7.pt`` (a
         skip-decoder run) look like a NAMR-only plain run in the first place.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-09
@ingroup audit_and_diagnostics

Outputs:
  results/checkpoint_registry.json      full per-checkpoint record (path/hash/run-id/decoder/args)
  results/checkpoint_hashes.json        {relative_path: sha256} (shape preserved for old tooling)
  results/checkpoint_decoder_map.json   list records (shape preserved)
  results/checkpoint_collisions.json    basename -> [paths] where hashes differ
Run: GFM_PROJECT_ROOT=$PWD .venv/bin/python scripts/132_audit_checkpoints.py
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =============================================================================
# File-level constants
# =============================================================================
ROOT_DIR = config.PROJECT_ROOT
WALK_ROOT_REL = "results"

# Directory names pruned from the results walk (e.g. .git should never appear
# under results/, but the filter is defensive so the audit never traverses
# unrelated repository state).
SKIP_DIR_PARTS = {".git"}

# .pt file filter.
PT_EXT = ".pt"
DOTFILE_PREFIX = "._"               # macOS resource-fork prefix; never hashed

# SHA-256 streaming hash chunk size (1 MiB by convention; tuned so the audit
# completes in seconds on a 128 GB machine without thrashing I/O cache).
HASH_CHUNK_BYTES = 1 << 20

# Run-configuration fields extracted from each checkpoint's embedded args dict
# (stored at checkpoint creation by 40_pretrain.py / gfm_train CLI). Order is
# semantic (seed/epochs/weights first, then data paths and HGT size) so the
# human-readable tables read naturally.
ARGS_OF_INTEREST = (
    "seed", "epochs", "standardize_features", "namr_mode",
    "namr_sigma", "csp_weight", "c3l_weight", "namr_weight",
    "data", "targets", "out", "hidden", "layers", "heads",
)

# NAMR-head measurement: the first weight matrix of the NAMR MLP is named
# ``net.0.weight`` (see 40_pretrain.MLPHead). Its second dimension is the
# input width we need.  Plain decoders have D_in == PLAIN_DECODER_DIM (256);
# skip-connected decoders concatenate the noisy input so D_in == SKIP_DECODER_DIM (512).
# Both constants are imported from the central config module (config §6b) so the
# audit script cannot silently diverge from the training script's dimensions.
NAMR_HEAD_FIRST_LIN = "net.0.weight"
PLAIN_DECODER_DIM = config.PLAIN_DECODER_DIM  # 256
SKIP_DECODER_DIM  = config.SKIP_DECODER_DIM   # 512

# --- Output file basenames (resolved via config.result_path). -------------
OUT_REGISTRY = "checkpoint_registry.json"
OUT_HASHES = "checkpoint_hashes.json"
OUT_DECODER_MAP = "checkpoint_decoder_map.json"
OUT_COLLISIONS = "checkpoint_collisions.json"
JSON_INDENT = 1

# --- Per-record canonical keys (used both in the registry and decoder map).
REC_FILE = "file"
REC_PATH = "path"
REC_SHA12 = "sha256_12"
REC_SIZE = "size_bytes"
REC_RUN_ID = "run_id"
REC_SEED = "seed"
REC_EPOCHS = "epochs"
REC_STD_FEAT = "standardize_features"
REC_NAMR_MODE = "namr_mode"
REC_DATA = "data"
REC_TARGETS = "targets"
REC_NAMR_IN = "namr_first_layer_in"
REC_DECODER = "decoder"
REC_LOAD_ERROR = "load_error"

# Decoder string representations.
DECODER_PLAIN = "plain"
DECODER_SKIP = "skip"
DECODER_NONE = "no namr head"
DECODER_FMT = "{kind} (in={width})"

# Short-hash length displayed in the collisions report (12 hex chars ~= 48 bits;
# more than sufficient to disambiguate the <1000 checkpoints audited here).
SHORT_SHA_HEX = 12

# Print summary limits.
MISSING_RUNID_PREVIEW = 10

# Full-registry payload meta keys.
REGISTRY_COUNT_KEY = "n_checkpoints"
REGISTRY_NOTE_KEY = "note"
REGISTRY_NOTE_TEXT = (
    "decoder measured from the checkpoint's own NAMR head input width; "
    "run_id read from the checkpoint's args.out, which file names lack"
)
REGISTRY_RECORDS_KEY = "records"


def write_json(rel_name, payload):
    """Serialise a payload to ``config.result_path(rel_name)`` using standard indent.

    @brief  Thin wrapper around json.dump with a stable indent level, so every
            output produced by the audit uses the same formatting convention.
    @param  rel_name   Relative basename under results/ (see OUT_* constants).
    @param  payload    JSON-serialisable dict / list.
    @return Absolute written path (for logging).
    """
    out_path = config.result_path(rel_name)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=JSON_INDENT)
    return out_path


def sha256(path, chunk=HASH_CHUNK_BYTES):
    """Compute the full SHA-256 hex digest of a file via streaming read.

    @brief  Streams the file in ``chunk``-byte blocks so arbitrarily large
            checkpoints (GB-scale) do not exhaust the Python process heap.
    @param  path    Absolute filesystem path.
    @param  chunk   Read buffer size in bytes.
    @return Lowercase 64-character hex string.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()


def decoder_of(checkpoint_dict):
    """Measure the actual NAMR decoder width stored inside a checkpoint dict.

    @brief  Inspects the checkpoint's ``namr`` state_dict (if present). Tries
            the well-known key ``NAMR_HEAD_FIRST_LIN`` first; if that weight is
            missing (e.g. older MLPHead implementations renamed layers) it
            falls back to the first 2-D tensor found in the NAMR state_dict,
            which is structurally guaranteed to be the first linear layer's
            weight.  Returns ``(input_width, decoder_kind_string)`` or
            ``(None, None)`` if no NAMR head is present.
    @param  checkpoint_dict   ``torch.load`` result (expected to be a dict).
    @return Tuple ``(input_width_int_or_None, kind_str_or_None)``.
    """
    if not isinstance(checkpoint_dict, dict):
        return None, None
    head = checkpoint_dict.get("namr")
    if head is None:
        return None, None
    if NAMR_HEAD_FIRST_LIN in head and head[NAMR_HEAD_FIRST_LIN].ndim == 2:
        width = int(head[NAMR_HEAD_FIRST_LIN].shape[1])
    else:
        width = None
        for _k, v in head.items():
            if v.ndim == 2:
                width = int(v.shape[1])
                break
        if width is None:
            return None, None
    kind = DECODER_PLAIN if width == PLAIN_DECODER_DIM else DECODER_SKIP
    return width, kind


def _normalise_args(args):
    """Return ``args`` as a plain dict (it may be an argparse.Namespace or None).

    @brief  Internal helper used inside the record-building loop so the same
            .get() pattern works regardless of how the original training script
            persisted ``args``.
    @param  args   Object potentially stored under the "args" checkpoint key.
    @return dict (may be empty).
    """
    if args is None:
        return {}
    if not isinstance(args, dict):
        try:
            return vars(args)
        except TypeError:
            return {}
    return args


def main():
    """Walk results/ for every .pt, build records, emit 4 JSON artifacts + summary.

    @brief  (1) recursively enumerate all .pt files under results/ skipping
            .git and dotfile prefixes; (2) hash each, load checkpoint, extract
            args + measure decoder; (3) group by basename to flag collisions;
            (4) write the 4 JSONs; (5) print a concise English summary with
            decoder breakdown, collision list, and the first few checkpoints
            lacking an args.out run id.
    @return None.
    """
    walk_root = os.path.join(ROOT_DIR, WALK_ROOT_REL)
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(walk_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
        for fn in filenames:
            if fn.endswith(PT_EXT) and not fn.startswith(DOTFILE_PREFIX):
                all_paths.append(os.path.join(dirpath, fn))
    all_paths.sort()

    records = []
    hashes = {}
    collisions = defaultdict(list)
    for p in all_paths:
        rel = os.path.relpath(p, ROOT_DIR)
        digest = sha256(p)
        hashes[rel] = digest
        short_digest = digest[:SHORT_SHA_HEX]
        collisions[os.path.basename(p)].append((rel, short_digest))
        rec = {
            REC_FILE: os.path.basename(p),
            REC_PATH: rel,
            REC_SHA12: short_digest,
            REC_SIZE: os.path.getsize(p),
        }
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            args = _normalise_args(ckpt.get("args") if isinstance(ckpt, dict) else None)
            rec[REC_RUN_ID]     = args.get("out")
            rec[REC_SEED]       = args.get("seed")
            rec[REC_EPOCHS]     = args.get("epochs")
            rec[REC_STD_FEAT]   = args.get("standardize_features")
            rec[REC_NAMR_MODE]  = args.get("namr_mode")
            rec[REC_DATA]       = args.get("data")
            rec[REC_TARGETS]    = args.get("targets")
            width, kind = decoder_of(ckpt)
            rec[REC_NAMR_IN] = width
            rec[REC_DECODER] = (
                DECODER_FMT.format(kind=kind, width=width) if kind else DECODER_NONE
            )
        except Exception as exc:                                   # pragma: no cover
            rec[REC_LOAD_ERROR] = f"{type(exc).__name__}: {exc}"
        records.append(rec)

    # Collision = basename shared by 2+ distinct byte-vectors.
    coll = {k: v for k, v in collisions.items() if len({sha for _rel, sha in v}) > 1}

    # --- 4 JSON artefacts -----------------------------------------------------
    payload_registry = {
        REGISTRY_COUNT_KEY: len(records),
        REGISTRY_NOTE_KEY: REGISTRY_NOTE_TEXT,
        REGISTRY_RECORDS_KEY: records,
    }
    write_json(OUT_REGISTRY, payload_registry)
    write_json(OUT_HASHES, hashes)
    # Decoder-map records: subset of keys + basename-normalised data/targets.
    decoder_records = []
    for r in records:
        data_basename = os.path.basename(r[REC_DATA]) if r.get(REC_DATA) else None
        tgt_basename = os.path.basename(r[REC_TARGETS]) if r.get(REC_TARGETS) else None
        decoder_records.append({
            REC_FILE:    r[REC_FILE],
            REC_PATH:    r[REC_PATH],
            REC_SHA12:   r[REC_SHA12],
            REC_NAMR_IN: r.get(REC_NAMR_IN),
            REC_DECODER: r.get(REC_DECODER),
            REC_SEED:    r.get(REC_SEED),
            REC_EPOCHS:  r.get(REC_EPOCHS),
            REC_NAMR_MODE: r.get(REC_NAMR_MODE),
            "std":       r.get(REC_STD_FEAT),
            REC_DATA:    data_basename,
            REC_TARGETS: tgt_basename,
            REC_RUN_ID:  r.get(REC_RUN_ID),
        })
    write_json(OUT_DECODER_MAP, decoder_records)
    write_json(OUT_COLLISIONS, coll)

    # --- English-only stdout summary -----------------------------------------
    print(f"checkpoints audited: {len(records)}")
    decoder_breakdown = {
        d: sum(1 for r in records if r.get(REC_DECODER) == d)
        for d in sorted({r.get(REC_DECODER) for r in records})
    }
    print(f"decoder breakdown: {decoder_breakdown}")
    print(f"\nbasename collisions (same name, different bytes): {len(coll)}")
    for base, entries in sorted(coll.items()):
        print(f"  {base}:")
        for rel, short in sorted(entries):
            print(f"      {rel}  [{short}]")
    missing_runid = [r[REC_PATH] for r in records if not r.get(REC_RUN_ID)]
    print(f"\ncheckpoints without args.out run id: {len(missing_runid)}")
    for m in missing_runid[:MISSING_RUNID_PREVIEW]:
        print(f"  {m}")


if __name__ == "__main__":
    main()
