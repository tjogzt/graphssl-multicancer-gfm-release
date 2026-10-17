#!/usr/bin/env python3
"""
@file    30_embed_esm2.py
@brief   Extract per-gene ESM-2 protein language model embeddings from the STRING
         v12 human proteome; mean-pool residues, average over isoforms, save as a
         gene-keyed PyTorch checkpoint for downstream graph builders.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup feature_engineering

Extract per-gene ESM-2 protein-language-model embeddings from the STRING v12
human proteome and save them as a single gene-keyed tensor. Sequences are
streamed from the gzipped FASTA, ENSP accessions are mapped to HGNC-style gene
symbols via the STRING protein-info table, each protein is mean-pooled over
residues 1..L (BOS/EOS excluded), and embeddings of isoforms that share a gene
symbol are averaged into one vector. Two model scales share this parameterised
entry point: the 8M backbone (320-dim) for the main experiments and the 650M
backbone (1280-dim) for the scale-up experiments reported in the manuscript.

Input:  STRING v12 files under config.STRING_DIR (env GFM_STRING_DIR):
        9606.protein.sequences.v12.0.fa.gz and 9606.protein.info.v12.0.txt
Output: <DATA_PROC>/esm2_gene_embeddings.pt        (--model 8m,   320-dim)
        <DATA_PROC>/esm2_650m_gene_embeddings.pt   (--model 650m, 1280-dim)
        checkpoint dict: {embeddings, model, dim, n_genes}
Run:    python scripts/30_embed_esm2.py --model 8m
        python scripts/30_embed_esm2.py --model 650m --batch-size 8
"""
import argparse
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Add scripts dir to path so config.py is discoverable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================

# --- STRING v12 input artifact names (resolved under config.STRING_DIR) ---
STRING_SEQ_FILENAME  = "9606.protein.sequences.v12.0.fa.gz"   # gzipped FASTA
STRING_INFO_FILENAME = "9606.protein.info.v12.0.txt"          # ENSP -> gene table

# --- Default output tensor names per model scale (resolved under config.DATA_PROC) ---
OUT_FILENAME_8M   = "esm2_gene_embeddings.pt"
OUT_FILENAME_650M = "esm2_650m_gene_embeddings.pt"

# Minimum residues a sequence must have before we consider it a valid protein
# (proteins shorter than this are likely fragments / annotations artifacts).
MIN_SEQUENCE_LENGTH_RESIDUES = 5

# --- Model registry: scale key -> (ESM-2 constructor name, embedding dim,
# representation layer index, default output filename, default batch size) ---
MODEL_REGISTRY = {
    "8m":   ("esm2_t6_8M_UR50D",     320,   6, OUT_FILENAME_8M,   32),
    "650m": ("esm2_t33_650M_UR50D", 1280, 33, OUT_FILENAME_650M, 8),
}

# Progress log cadence: emit a progress line every N completed batches.
PROGRESS_LOG_EVERY_N_BATCHES = 20


# =========================================================================
# Utility functions
# =========================================================================

def select_device():
    """
    @brief   Choose the best available accelerator, falling back through CUDA -> Apple MPS -> CPU.

    @return  torch.device object representing the chosen compute device.
    @note    CUDA is preferred when available (NVIDIA GPUs). Apple Silicon uses
             MPS. CPU (PyTorch native) is the universal fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # Apple Silicon acceleration (Metal Performance Shaders) if PyTorch >= 1.12.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sequences(fa_gz_path):
    """
    @brief   Stream a gzipped STRING FASTA file into a dict of {ENSP accession: AA sequence}.

    @param   fa_gz_path  Absolute path to the gzipped proteome FASTA.
    @return  Dict mapping ENSP accession (e.g. "ENSP00000253382") -> amino-acid string.
    @note    STRING FASTA headers look like ">9606.ENSP00000253382 ...". We keep
             only the second dot-separated token (the ENSP id).
    """
    sequences = {}
    current_id, current_seq = None, []
    with gzip.open(fa_gz_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                # Flush the previous record before switching to the new header.
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                # Parse the ENSP token from the header.
                parts = line[1:].split(".")
                current_id = parts[1] if len(parts) >= 2 else None
                current_seq = []
            elif current_id:
                current_seq.append(line)
        # Flush the final record (no trailing header after the last sequence).
        if current_id:
            sequences[current_id] = "".join(current_seq)
    return sequences


def load_ensp_to_gene(info_path):
    """
    @brief   Build ENSP accession -> uppercase gene-symbol mapping from STRING info TSV.

    @param   info_path  Absolute path to the STRING protein-info .txt table.
    @return  Dict mapping ENSP accession (string) -> HGNC uppercase gene symbol (string).
    @note    The STRING info table stores combined ids like "9606.ENSP00000253382"
             in column '#string_protein_id'; we split on '.' and keep the ENSP part.
    """
    info = pd.read_csv(info_path, sep="\t")
    info["gene_symbol"] = info["preferred_name"].str.upper()
    mapping = {}
    for _, row in info.iterrows():
        string_id = row["#string_protein_id"]
        if "." in string_id:
            # Keep only the ENSP token after the species prefix.
            mapping[string_id.split(".")[1]] = row["gene_symbol"]
    return mapping


def build_model(model_scale):
    """
    @brief   Instantiate the requested ESM-2 backbone (deferred heavy import of esm package).

    @param   model_scale  String key into MODEL_REGISTRY ("8m" or "650m").
    @return  Tuple (pretrained_model, alphabet) as returned by the esm.pretrained ctor.
    @note    The `esm` import is deferred inside this function so --help and
             basic argument parsing work on machines that don't yet have fairseq/esm.
    """
    ctor_name, _, _, _, _ = MODEL_REGISTRY[model_scale]
    # Deferred, per-scale import of the ESM pretrained model constructor.
    if model_scale == "8m":
        from esm.pretrained import esm2_t6_8M_UR50D as model_ctor
    elif model_scale == "650m":
        from esm.pretrained import esm2_t33_650M_UR50D as model_ctor
    else:  # pragma: no cover - argparse choices already guard this path
        raise ValueError(f"Unsupported model scale: {model_scale}")
    return model_ctor()


def embed_sequences(valid_proteins, model, alphabet, device,
                    repr_layer, batch_size):
    """
    @brief   Mean-pool per-residue representations over the requested layer for
             every valid (ENSP, gene, sequence) triple; accumulate per gene.

    @param   valid_proteins  Dict: ENSP -> (gene_symbol, aa_sequence_truncated).
    @param   model           Loaded ESM-2 PyTorch module (already on device, in eval).
    @param   alphabet        ESM alphabet object used to build batch converters.
    @param   device          torch.device for forward passes.
    @param   repr_layer      1-based layer index to extract representations from.
    @param   batch_size      Number of proteins per forward pass.
    @return  Dict: gene_symbol -> list of per-protein embedding arrays (isoforms list
             later averaged outside this function).
    @note    BOS/EOS token positions (index 0 and L+1 in each padded row) are
             EXCLUDED from the mean pool.
    """
    # ESM's batch converter: takes list[(label, seq)] -> (labels, strs, tokens_tensor).
    converter = alphabet.get_batch_converter()
    ensp_ids = list(valid_proteins.keys())
    emb_by_gene = {}
    started = time.time()

    # Iterate proteins in fixed-order batches to facilitate reproducibility.
    for start_idx in range(0, len(ensp_ids), batch_size):
        batch_keys = ensp_ids[start_idx:start_idx + batch_size]
        # Build per-batch label/sequence tuples expected by the ESM converter.
        batch_tuples = [(ensp, valid_proteins[ensp][1]) for ensp in batch_keys]
        _, _, tokens = converter(batch_tuples)
        tokens = tokens.to(device)

        # Disable gradient computation: pure inference on frozen pretrained weights.
        with torch.no_grad():
            layer_results = model(tokens, repr_layers=[repr_layer])
        representations = layer_results["representations"][repr_layer]  # [B, L, dim]

        # Per-row: mean-pool the residue-level vectors, drop BOS (pos 0) and EOS (pos L+1).
        for row_idx, ensp in enumerate(batch_keys):
            gene_symbol, aa_seq = valid_proteins[ensp]
            seq_len = len(aa_seq)
            # Slice: skip BOS, take seq_len residues, then average across positions.
            pooled = representations[row_idx, 1:seq_len + 1, :].mean(dim=0).cpu().numpy()
            # Accumulate all isoforms per gene (averaged at the end outside).
            emb_by_gene.setdefault(gene_symbol, []).append(pooled)

        # Periodic progress heartbeat so users can see forward progress on long runs.
        batch_num = start_idx // batch_size
        if batch_num % PROGRESS_LOG_EVERY_N_BATCHES == 0:
            elapsed = time.time() - started
            print(f"  Progress: {start_idx:,}/{len(ensp_ids):,} proteins "
                  f"processed | elapsed {elapsed:.0f}s")

    return emb_by_gene


# =========================================================================
# Main entry
# =========================================================================

def main():
    """
    @brief   Top-level driver: parse args, build model, stream sequences, embed in
             batches, average isoforms, save gene-keyed checkpoint.

    @return  None. Writes output checkpoint and prints final summary.
    """
    parser = argparse.ArgumentParser(
        description="Extract per-gene mean-pooled ESM-2 embeddings from the STRING v12 human proteome."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="8m",
        help="ESM-2 model scale: 8m (320-dim) or 650m (1280-dim)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Custom output .pt path (default: config.DATA_PROC/<registry default name>)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Proteins per forward pass (default: per-scale MODEL_REGISTRY value)."
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=None,
        help="Representation layer index (default: last layer per MODEL_REGISTRY)."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Truncate long proteins (e.g. titin) to this many residues to bound memory."
    )
    args = parser.parse_args()

    # Deterministic randomness (for hash / RNG ops; model itself is frozen eval).
    config.set_seed()

    # --- Resolve per-scale defaults from MODEL_REGISTRY ---
    ctor_name, dim, default_layer, default_out, default_batch = MODEL_REGISTRY[args.model]
    repr_layer = args.repr_layer if args.repr_layer is not None else default_layer
    batch_size = args.batch_size if args.batch_size is not None else default_batch
    out_path   = args.out       if args.out       is not None else os.path.join(config.DATA_PROC, default_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seq_path  = os.path.join(config.STRING_DIR, STRING_SEQ_FILENAME)
    info_path = os.path.join(config.STRING_DIR, STRING_INFO_FILENAME)

    # --- Accelerator selection + model instantiation ---
    device = select_device()
    print(f"[Init] Loading ESM-2 backbone '{ctor_name}' (dim={dim}) on device: {device} ...")
    model, alphabet = build_model(args.model)
    model = model.to(device).eval()

    # --- Load STRING input tables ---
    print("[Init] Loading STRING v12 input files ...")
    sequences    = load_sequences(seq_path)
    ensp_to_gene = load_ensp_to_gene(info_path)
    print(f"  Raw sequences loaded : {len(sequences):,}")
    print(f"  ENSP -> gene mappings: {len(ensp_to_gene):,}")

    # --- Filter to valid proteins: must have gene symbol + length >= MIN_SEQUENCE_LENGTH_RESIDUES ---
    valid_proteins = {}
    for ensp, seq in sequences.items():
        if ensp in ensp_to_gene and len(seq) >= MIN_SEQUENCE_LENGTH_RESIDUES:
            # Truncate ultra-long sequences to --max_seq_len to avoid OOM on titin-like proteins.
            truncated_seq = seq[:args.max_seq_len]
            valid_proteins[ensp] = (ensp_to_gene[ensp], truncated_seq)
    print(f"  Valid proteins for embedding: {len(valid_proteins):,}")

    # --- Batched ESM-2 embedding forward passes ---
    print("[Embed] Running batched ESM-2 forward passes ...")
    emb_by_gene = embed_sequences(
        valid_proteins, model, alphabet, device, repr_layer, batch_size
    )

    # --- Average isoforms: each gene's vector = mean of its isoform vectors ---
    final_embeddings = {
        gene: np.mean(embs, axis=0)
        for gene, embs in emb_by_gene.items()
    }
    if not final_embeddings:
        raise RuntimeError(
            "FATAL: No embeddings were produced. Verify STRING input files exist "
            "and contain valid proteins."
        )

    # --- Persist the final gene-keyed checkpoint ---
    checkpoint = {
        "embeddings": final_embeddings,
        "model":      ctor_name,
        "dim":        dim,
        "n_genes":    len(final_embeddings),
    }
    torch.save(checkpoint, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n[Done] Checkpoint saved: {out_path}")
    print(f"  Genes with embeddings : {len(final_embeddings):,}")
    print(f"  Embedding dimension   : {dim}")
    print(f"  File size             : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
