#!/usr/bin/env python3
"""
@file    41_pretrain_single_cancer.py
@brief   Equal-data single-cancer control pretraining (review item 2). This
         distinguishes "more data" from "transfer learning": it pretrains a GNN on
         a SINGLE cancer's subgraph while running the SAME number of total optimizer
         steps (equal data volume via batch oversampling) as the multi-cancer
         GFM. If the multi-cancer GFM still beats this control on a held-out
         cancer, the gain is attributable to cross-cancer transfer rather than to
         seeing more data. Only NAMR + CSP are used (C3L needs >=2 cancer
         contexts); patient nodes are dropped so node/edge types stay consistent
         (patient-level pretraining is handled separately by 40_pretrain.py).
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_pretraining

Input:  --data graph .pt (default <DATA_PROC>/xena_graph_patient.pt); --cancer
        cancer type (required); --epochs/--lr/--hidden/--layers/--heads;
        --namr_mode; --standardize_features; --save_every/--resume/--seed/--log_every.
Output: checkpoint .pt in config.MODEL_DIR (single_<cancer>_e<epochs>_s<seed>.pt)
        plus per-epoch loss lines on stdout.
Run:    python scripts/41_pretrain_single_cancer.py --cancer BRCA --epochs 500
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
import gfm_losses
from gfm_losses import CSPSampler, NAMRMask, csp_loss, namr_loss

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Default artifact filenames (resolved under config.DATA_PROC / config.MODEL_DIR).
DEFAULT_GRAPH = "xena_graph_patient.pt"               # Default input graph
CKPT_NAME = "single_{cancer}_e{epochs}_s{seed}.pt" # Output checkpoint pattern

# Default HGT encoder architecture hyperparameters.
DEFAULT_HIDDEN_DIM = 256                              # Default hidden dimension size
DEFAULT_HGT_LAYERS = 6                                  # Default number of HGT layers
DEFAULT_HGT_HEADS = 8                                   # Default number of attention heads

# Default optimization hyperparameters.
DEFAULT_LEARNING_RATE = 1e-3                            # Default AdamW learning rate
DEFAULT_WEIGHT_DECAY = 1e-5                              # Default AdamW weight decay
DEFAULT_GRAD_CLIP_NORM = 1.0                         # Default gradient clipping norm

# Default CLI parameter defaults.
DEFAULT_EPOCHS = 500                                    # Default training epochs
DEFAULT_SAVE_EVERY = 25                               # Default periodic checkpoint interval
DEFAULT_LOG_EVERY = 10                                   # Default logging interval
DEFAULT_NAMR_MODE = 'denoise'                         # Default NAMR masking mode

# Feature preprocessing constants.
FEATURE_STD_EPSILON = 1e-8                              # Numerical stability epsilon

# ---------------------------------------------------------------------------
# Reuse model classes from the fixed pretraining script (avoids code duplication)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "pretrain_fixed", os.path.join(os.path.dirname(__file__), "40_pretrain.py"))
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)
HGTEncoder, CSPHead, MLPHead = _pf.HGTEncoder, _pf.CSPHead, _pf.MLPHead


def filter_single_cancer(graph, cancer):
    """
    @brief   Extract a single-cancer subgraph by dropping all cancer/patient nodes and
             gene-cancer edges that do not belong to the specified cancer type.
    @param   graph : dict
             Full heterogeneous graph dictionary containing at least 'gene', 'cancer',
             PPI edges, and meta information.
    @param   cancer : str
             Cancer type identifier string (e.g. "BRCA") to retain in the subgraph.
    @return  dict
             Filtered graph containing: all genes, the single selected cancer node (reindexed
             to position 0), PPI edges, and bidirectional gene-cancer edges for the
             selected cancer type. Patient nodes are explicitly excluded.
    @note    Patient nodes are intentionally dropped to keep node/edge types consistent
             between single-cancer and multi-cancer pretraining (patient-level pretraining
             is handled separately by 40_pretrain.py).
    """
    meta = graph['meta']
    c_list = meta['cancer_list']
    # Validate that the requested cancer type exists in the graph
    if cancer not in c_list:
        raise SystemExit(f"cancer {cancer} not in graph ({len(c_list)} types)")
    ci = c_list.index(cancer)
    num_genes = graph['gene']['num_nodes']
    # Build the single-cancer subgraph: keep all genes + 1 cancer node + PPI edges
    new = {
        'gene': graph['gene'],
        'cancer': {
            'x': graph['cancer']['x'][ci:ci + 1].clone(),
            'num_nodes': 1
        },
        ('gene', 'ppi', 'gene'): graph[('gene', 'ppi', 'gene')],
        # Single cancer type -> every gene connects to cancer node 0 in both directions
        ('gene', 'expressed_in', 'cancer'): {
            'edge_index': torch.tensor([
                [i for i in range(num_genes)],
                [0] * num_genes
            ], dtype=torch.long)
        },
        ('cancer', 'expresses', 'gene'): {
            'edge_index': torch.tensor([
                [0] * num_genes,
                [i for i in range(num_genes)]
            ], dtype=torch.long)
        },
        'meta': {
            'gene_list': meta['gene_list'],
            'cancer_list': [cancer],
            'n_genes': meta['n_genes'],
            'n_cancers': 1,
            'embedding_dim': meta.get('embedding_dim', config.EMBEDDING_DIM)
        },
    }
    # NOTE: the equal-data single-cancer control operates at the GENE level
    # (NAMR + CSP only). Patient nodes are dropped here to keep node/edge types
    # consistent; patient-level pretraining is handled separately.
    return new


def pretrain(graph, args):
    """
    @brief   Single-cancer equal-data control pretraining loop. Builds a filtered
             single-cancer subgraph and runs the same number of optimizer steps as
             the multi-cancer GFM to enable an apples-to-apples transfer-learning
             comparison.
    @param   graph : dict
             Full heterogeneous graph dictionary (will be filtered internally).
    @param   args : argparse.Namespace
             Parsed CLI arguments with hyperparameters and paths.
    @return  None (side effects: prints loss logs, saves final checkpoint to disk)
    @note    Uses only NAMR + CSP objectives (C3L is disabled because a single
             cancer context cannot support cross-cancer contrastive learning).
    """
    # Fix random seed for reproducibility of the single-cancer control run
    config.set_seed(args.seed)
    # Auto-select best available compute device: CUDA > MPS > CPU
    device = torch.device(
        'cuda:0' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    # Extract the single-cancer subgraph (drops other cancers and all patients
    graph = filter_single_cancer(graph, args.cancer)

    # Build data dictionaries and extract graph metadata from filtered graph
    data, node_types, edge_types, in_dims = {}, [], [], {}
    for k, v in graph.items():
        if isinstance(k, str) and 'x' in v:
            node_types.append(k)
            in_dims[k] = v['x'].size(-1)
            data[k] = {'x': v['x'].to(device), 'num_nodes': v['num_nodes']}
        elif isinstance(k, tuple) and len(k) == 3 and 'edge_index' in v:
            edge_types.append(k)
            data[k] = {'edge_index': v['edge_index'].to(device)}

    # Build convenience references for node features and edge indices
    x_dict = {nt: data[nt]['x'] for nt in node_types}
    ei_dict = {et: data[et]['edge_index'] for et in edge_types}
    n_genes = data['gene']['num_nodes']

    # Optional per-dimension gene feature standardization (protocol H1)
    if args.standardize_features and 'gene' in x_dict:
        g = x_dict['gene']
        x_dict['gene'] = (g - g.mean(0)) / (g.std(0) + FEATURE_STD_EPSILON)

    # Instantiate model components (matching multi-cancer GFM architecture)
    model = HGTEncoder(node_types, edge_types, in_dims,
                      args.hidden, args.layers, args.heads).to(device)
    csp_head = CSPHead(args.hidden).to(device)
    namr_head = MLPHead(args.hidden, x_dict['gene'].size(-1)).to(device)

    # Initialize CSP pair sampler and NAMR masking strategy
    csp_sampler = CSPSampler(data[('gene', 'ppi', 'gene')]['edge_index'], n_genes)
    namr_masker = NAMRMask(
        config.NAMR_MASK_RATIO,
        config.NAMR_NOISE_SIGMA,
        args.namr_mode
    )

    # Collect trainable parameters and set up optimizer + LR schedule
    params = (list(model.parameters())
              + list(csp_head.parameters())
              + list(namr_head.parameters()))
    opt = AdamW(params, lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, args.epochs)

    # Compute output checkpoint path and print run configuration summary
    out = os.path.join(
        config.MODEL_DIR,
        CKPT_NAME.format(cancer=args.cancer, epochs=args.epochs, seed=args.seed)
    )
    print(f"Single-cancer ({args.cancer}) equal-data control | {n_genes} genes | "
          f"{args.epochs} epochs (= equal optimizer steps)")

    t0 = time.time()
    log = {'csp': [], 'namr': [], 'total': []}
    start_epoch = 1
    # Optionally resume from a previous single-cancer checkpoint
    if args.resume and os.path.exists(args.resume):
        old = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(old['model'])
        opt.load_state_dict(old.get('optimizer', opt.state_dict()))
        log = {k: list(v) for k, v in old.get('loss_history', {}).items()}
        start_epoch = int(old.get('epoch', 0)) + 1
        print(f'[resume] from epoch {start_epoch - 1}')

    # --- Main training epoch loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Forward pass through HGT encoder
        z = model(x_dict, ei_dict)
        gene_z = z['gene']

        # --- CSP objective: link-prediction on sampled PPI pairs ---
        pa, pb, na, nb = csp_sampler.sample(
            min(config.CSP_BATCH_SIZE, n_genes), device)
        l_csp = csp_loss(
            csp_head(gene_z[pa], gene_z[pb]),
            csp_head(gene_z[na], gene_z[nb])
        )

        # --- NAMR objective: denoise gene features and reconstruct ---
        x_noisy, midx = namr_masker.apply(x_dict['gene'], device)
        xnd = {k: (x_noisy if k == 'gene' else v) for k, v in x_dict.items()}
        zn = model(xnd, ei_dict)
        l_namr = namr_loss(namr_head(zn['gene'][midx]), x_dict['gene'][midx])

        # Simple unweighted sum of the two objectives (C3L not applicable here)
        loss = l_csp + l_namr

        # Backpropagation with gradient clipping
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP_NORM)
        opt.step()
        sch.step()

        # Log per-epoch loss values
        log['csp'].append(l_csp.item())
        log['namr'].append(l_namr.item())
        log['total'].append(loss.item())

        # Periodically save intermediate checkpoints
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'cancer': args.cancer,
                'args': vars(args),
                'loss_history': log,
                'optimizer': opt.state_dict(),
                'epoch': epoch
            }, out)
            print(f'[ckpt] epoch {epoch}')

        # Periodic progress logging
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"E {epoch:4d} | CSP {l_csp.item():.4f} | NAMR {l_namr.item():.4f} "
                  f"| Tot {loss.item():.4f}")

    # Save the final single-cancer control checkpoint
    out = os.path.join(
        config.MODEL_DIR,
        CKPT_NAME.format(cancer=args.cancer, epochs=args.epochs, seed=args.seed)
    )
    torch.save({
        'model': model.state_dict(),
        'cancer': args.cancer,
        'args': vars(args),
        'loss_history': log
    }, out)
    print(f"\nSaved {out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(config.DATA_PROC, DEFAULT_GRAPH))
    ap.add_argument('--cancer', required=True)
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument('--hidden', type=int, default=DEFAULT_HIDDEN_DIM)
    ap.add_argument('--layers', type=int, default=DEFAULT_HGT_LAYERS)
    ap.add_argument('--heads', type=int, default=DEFAULT_HGT_HEADS)
    ap.add_argument('--namr_mode', choices=['denoise', 'mask'], default=DEFAULT_NAMR_MODE)
    ap.add_argument('--standardize_features', action='store_true')
    ap.add_argument('--save_every', type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--log_every', type=int, default=DEFAULT_LOG_EVERY)
    args = ap.parse_args()
    pretrain(torch.load(args.data, map_location='cpu'), args)
