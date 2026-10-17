#!/usr/bin/env python3
"""
@file    40_pretrain.py
@brief   Pretrain the heterogeneous graph transformer (HGT) foundation model with
         three self-supervised objectives: CSP (contextual PPI link prediction),
         NAMR (node attribute denoising), and C3L (cross-cancer contrastive learning
         against REAL gene-cancer specificity targets precomputed by
         32_precompute_c3l_targets.py). Two NAMR decoders are supported (--decoder
         plain, the default; --decoder skip, a diagnostic concatenation of the noisy
         input with the hidden state). Patient nodes are used only when the graph
         carries patient-gene edges (built by 33_build_patient_gene_edges.py);
         otherwise patient tasks are skipped with an explicit warning rather than
         silently dropping nodes.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup model_pretraining

Input:  --data graph .pt (default <DATA_PROC>/xena_graph_esm.pt); --targets C3L
        targets .pt (default <DATA_PROC>/gene_cancer_targets.pt); training flags
        --epochs/--lr/--hidden/--layers/--heads; loss weights --csp_weight/
        --namr_weight/--c3l_weight; --namr_mode/--namr_sigma; --decoder;
        --standardize_features; --out/--resume/--seed/--save_every/--log_every.
Output: checkpoint .pt in config.MODEL_DIR (default gfm_fixed_e<epochs>.pt)
        plus per-epoch loss lines on stdout.
Run:    python scripts/40_pretrain.py --epochs 500
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import HGTConv, Linear

import config
import gfm_losses
from gfm_losses import CSPSampler, NAMRMask, c3l_info_nce_loss, csp_loss, namr_loss, namr_loss_sce

# ---------------------------------------------------------------------------
# Constant definitions - centralized hardcoded parameters
# ---------------------------------------------------------------------------

# Default artifact filenames (resolved under config.DATA_PROC / config.MODEL_DIR).
DEFAULT_GRAPH = "xena_graph_esm.pt"          # Default input heterogeneous graph
DEFAULT_TARGETS = "gene_cancer_targets.pt"   # Default real C3L specificity targets
CKPT_NAME = "gfm_fixed_e{epochs}.pt"         # Default checkpoint filename (epochs filled)

# Default HGT encoder architecture hyperparameters.
DEFAULT_HIDDEN_DIM = 256                     # Default hidden dimension size
DEFAULT_HGT_LAYERS = 6                       # Default number of HGT convolution layers
DEFAULT_HGT_HEADS = 8                        # Default number of attention heads
DEFAULT_DROPOUT_RATE = 0.2                   # Default dropout probability

# Default optimization hyperparameters.
DEFAULT_LEARNING_RATE = 1e-3                 # Default AdamW learning rate
DEFAULT_WEIGHT_DECAY = 1e-5                  # Default AdamW weight decay
DEFAULT_GRAD_CLIP_NORM = 1.0                 # Default gradient clipping norm threshold

# Default C3L contrastive learning hyperparameters.
DEFAULT_C3L_BATCH_SIZE = 256                 # Default number of genes subsampled per C3L step

# Default feature preprocessing constants.
FEATURE_STD_EPSILON = 1e-8                   # Epsilon for numerical stability in standardization

# Convergence diagnostic window sizes (number of epochs).
CONV_MID_WINDOW = 10                         # Window around mid-training for convergence check
CONV_FINAL_WINDOW = 10                       # Final window for convergence check

# ---------------------------------------------------------------------------
# Model architecture definitions
# ---------------------------------------------------------------------------

class HGTEncoder(nn.Module):
    """
    @brief   Heterogeneous Graph Transformer (HGT) encoder module.
             Projects per-node-type features to a shared hidden dimension, then
             applies a stack of HGT convolution layers with GELU activation and
             dropout.
    @param   node_types : list
             List of node type strings present in the heterogeneous graph.
    @param   edge_types : list
             List of (src_type, relation, dst_type) edge type tuples.
    @param   in_dims : dict
             Mapping from node type string to input feature dimension for that type.
    @param   hidden : int, optional
             Shared hidden dimension size (default: DEFAULT_HIDDEN_DIM).
    @param   layers : int, optional
             Number of stacked HGT convolution layers (default: DEFAULT_HGT_LAYERS).
    @param   heads : int, optional
             Number of attention heads per HGT layer (default: DEFAULT_HGT_HEADS).
    @param   dropout : float, optional
             Dropout probability applied after each HGT layer (default: DEFAULT_DROPOUT_RATE).
    @return  HGTEncoder instance (nn.Module subclass).
    @note    Each node type's raw features are first projected via a dedicated linear
             layer before entering the HGT convolution stack.
    """
    def __init__(self, node_types, edge_types, in_dims,
                 hidden=DEFAULT_HIDDEN_DIM, layers=DEFAULT_HGT_LAYERS,
                 heads=DEFAULT_HGT_HEADS, dropout=DEFAULT_DROPOUT_RATE):
        super().__init__()
        self.node_types = list(node_types)
        self.lin_dict = nn.ModuleDict()
        # Per-type linear projection: map each node type's raw features to shared hidden dimension
        for nt in self.node_types:
            self.lin_dict[nt] = Linear(in_dims[nt], hidden)
        # Stack of HGT convolution layers with heterogeneous attention
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, (self.node_types, list(edge_types)), heads=heads)
             for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, ei_dict):
        """
        @brief   Forward pass through the HGT encoder.
        @param   x_dict : dict
             Mapping from node type string to node feature tensor [num_nodes, in_dim].
        @param   ei_dict : dict
             Mapping from edge type tuple to edge_index tensor [2, num_edges].
        @return  dict
             Updated node feature dictionary with the same keys as x_dict, each value
             has shape [num_nodes, hidden].
        """
        # Step 1: project all node types to shared hidden dimension
        x_dict = {nt: self.lin_dict[nt](x_dict[nt]) for nt in self.node_types}
        # Step 2: apply each HGT conv layer with GELU activation and dropout
        for conv in self.convs:
            x_dict = conv(x_dict, ei_dict)
            x_dict = {k: self.dropout(F.gelu(v)) for k, v in x_dict.items()}
        return x_dict


class MLPHead(nn.Module):
    """
    @brief   Two-layer MLP projection head with GELU non-linearity.
    @param   in_dim : int
             Input feature dimension.
    @param   out_dim : int
             Output feature dimension.
    @param   hidden : int or None, optional
             Hidden layer dimension; if None, defaults to in_dim (default: None).
    @return  MLPHead instance (nn.Module subclass).
    @note    Architecture: Linear -> GELU -> Linear.
    """
    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            Linear(in_dim, hidden),
            nn.GELU(),
            Linear(hidden, out_dim)
        )

    def forward(self, x):
        """
        @brief   Forward pass through the MLP head.
        @param   x : torch.Tensor
             Input tensor of shape [..., in_dim].
        @return  torch.Tensor
             Output tensor of shape [..., out_dim].
        """
        return self.net(x)


class CSPHead(nn.Module):
    """
    @brief   Contextual PPI (CSP) link-prediction scoring head.
             Takes concatenated embeddings of two genes and produces a scalar
             logit indicating whether a PPI edge exists between them.
    @param   hidden : int
             Dimension of the input gene embeddings.
    @return  CSPHead instance (nn.Module subclass).
    @note    Uses concatenation of the two embeddings as pair representation,
             then maps through a 2-layer MLP to a scalar score.
    """
    def __init__(self, hidden):
        super().__init__()
        # Pair representation: [z1 || z2] (2*hidden) -> hidden -> scalar logit
        self.mlp = nn.Sequential(
            Linear(2 * hidden, hidden),
            nn.GELU(),
            Linear(hidden, 1)
        )

    def forward(self, z1, z2):
        """
        @brief   Compute PPI link logits for gene pairs (z1_i, z2_i).
        @param   z1 : torch.Tensor
             First gene embeddings, shape [num_pairs, hidden].
        @param   z2 : torch.Tensor
             Second gene embeddings, shape [num_pairs, hidden].
        @return  torch.Tensor
             Scalar logits per pair, shape [num_pairs].
        @note    The two inputs are concatenated along the feature dimension before
             being fed through the MLP.
        """
        # Concatenated pair embedding -> scalar PPI logit
        return self.mlp(torch.cat([z1, z2], dim=-1)).squeeze(-1)


class C3LHead(nn.Module):
    """
    @brief   Cross-Cancer Contrastive Learning (C3L) projection head.
             Projects gene/cancer embeddings to a normalized representation space
             where the dot product equals cosine similarity, suitable for InfoNCE.
    @param   hidden : int
             Dimension of the input embeddings (gene or cancer node hidden states).
    @param   proj : int, optional
             Output projection dimension; defaults to config.C3L_PROJ_DIM.
    @return  C3LHead instance (nn.Module subclass).
    @note    Output embeddings are L2-normalized so the InfoNCE dot product is
             equivalent to a cosine similarity score.
    """
    def __init__(self, hidden, proj=config.C3L_PROJ_DIM):
        super().__init__()
        # Projection MLP: hidden -> hidden -> proj_dim (with GELU non-linearity)
        self.proj = nn.Sequential(
            Linear(hidden, hidden),
            nn.GELU(),
            Linear(hidden, proj)
        )

    def forward(self, z):
        """
        @brief   Project and L2-normalize input embeddings.
        @param   z : torch.Tensor
             Input embeddings of shape [..., hidden].
        @return  torch.Tensor
             L2-normalized projections of shape [..., proj].
        """
        # L2-normalize projections so the InfoNCE dot product is a cosine score
        return F.normalize(self.proj(z), dim=-1)

# ---------------------------------------------------------------------------
# Data preparation and training utilities
# ---------------------------------------------------------------------------

def build_data_dict(graph, device):
    """
    @brief   Convert a heterogeneous graph dictionary to device-placed tensors and
             extract metadata (node types, edge types, input dimensions).
    @param   graph : dict
             Raw heterogeneous graph dictionary with node-type keys (str) mapping to
             {'x', 'num_nodes'} and edge-type keys (3-tuple) mapping to {'edge_index'}.
    @param   device : torch.device
             Target device to place tensors on (CPU / CUDA / MPS).
    @return  tuple of (data, node_types, edge_types, in_dims)
             - data : dict with same structure as graph but tensors moved to `device`
             - node_types : list of node type strings (sorted by graph order)
             - edge_types : list of edge type 3-tuples (sorted by graph order)
             - in_dims : dict mapping node_type -> input feature dimension
    @note    Graph items are partitioned by key type: string keys = node types,
             3-tuple keys = edge types.
    """
    data = {}
    node_types, edge_types, in_dims = [], [], {}
    for k, v in graph.items():
        # String keys with 'x' attribute represent node types
        if isinstance(k, str) and 'x' in v:
            node_types.append(k)
            in_dims[k] = v['x'].size(-1)
            data[k] = {'x': v['x'].to(device), 'num_nodes': v['num_nodes']}
        # 3-tuple keys with 'edge_index' attribute represent edge types
        elif isinstance(k, tuple) and len(k) == 3 and 'edge_index' in v:
            edge_types.append(k)
            data[k] = {'edge_index': v['edge_index'].to(device)}
    return data, node_types, edge_types, in_dims


def pretrain(graph, targets_path, args):
    """
    @brief   Main pretraining loop: build model, run CSP + NAMR + C3L multi-task
             optimization, and save the final checkpoint.
    @param   graph : dict
             Raw heterogeneous graph dictionary (see build_data_dict for structure).
    @param   targets_path : str
             File path to the precomputed C3L gene-cancer specificity targets .pt file.
    @param   args : argparse.Namespace
             Parsed CLI arguments with hyperparameters and paths.
    @return  None (side effects: prints loss logs, saves checkpoint file to disk)
    @note    All three objectives are weighted sum combined; patient-level NAMR is
             conditionally enabled only when patient-gene edges exist in the graph.
    """
    # Fix random seed for full reproducibility of the training run
    config.set_seed(args.seed)
    # Auto-select the best available compute device: CUDA > MPS > CPU
    device = torch.device(
        'cuda:0' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    # Build data dictionaries and extract graph metadata
    data, node_types, edge_types, in_dims = build_data_dict(graph, device)
    has_patient = 'patient' in data
    has_patient_gene = any(
        e[0] == 'patient' or e[2] == 'patient'
        for e in edge_types if 'gene' in (e[0], e[2])
    )

    n_genes = data['gene']['num_nodes']
    n_cancers = data['cancer']['num_nodes']

    # Build convenience references for node features and edge indices
    x_dict = {nt: data[nt]['x'] for nt in node_types}
    ei_dict = {et: data[et]['edge_index'] for et in edge_types}

    # Optional per-dimension gene feature standardization (protocol H1)
    if args.standardize_features and 'gene' in x_dict:
        g = x_dict['gene']
        x_dict['gene'] = (g - g.mean(0)) / (g.std(0) + FEATURE_STD_EPSILON)

    # Instantiate all model components and move to target device
    model = HGTEncoder(node_types, edge_types, in_dims,
                       args.hidden, args.layers, args.heads).to(device)
    csp_head = CSPHead(args.hidden).to(device)
    if args.decoder == 'skip':
        # Skip-connected NAMR decoder: concatenate noisy input with hidden state
        namr_head = MLPHead(args.hidden + x_dict['gene'].size(-1),
                            x_dict['gene'].size(-1)).to(device)
    else:
        namr_head = MLPHead(args.hidden, x_dict['gene'].size(-1)).to(device)
    c3l_head = C3LHead(args.hidden).to(device)

    # Patient-level denoising head (skip decoder mirrors the gene NAMR decoder)
    p_head = None
    if has_patient and has_patient_gene:
        if args.decoder == 'skip':
            p_head = MLPHead(args.hidden + x_dict['patient'].size(-1),
                             x_dict['patient'].size(-1)).to(device)
        else:
            p_head = MLPHead(args.hidden, x_dict['patient'].size(-1)).to(device)

    # Initialize CSP pair sampler on the full PPI edge index
    ppi_ei = data[('gene', 'ppi', 'gene')]['edge_index']
    csp_sampler = CSPSampler(ppi_ei, n_genes)

    # Load real C3L targets: N-way InfoNCE against precomputed specificity labels
    targets = None
    if os.path.exists(targets_path):
        targets = torch.load(targets_path, map_location='cpu')['targets'].to(device)
        assert targets.numel() == n_genes, \
            f"targets({targets.numel()}) != genes({n_genes})"
    else:
        print("[WARN] No C3L targets file found; C3L disabled. "
              "Run scripts/32_precompute_c3l_targets.py first.")

    # Configure NAMR masking strategy and reconstruction loss function
    namr_masker = NAMRMask(
        mask_ratio=config.NAMR_MASK_RATIO,
        sigma=args.namr_sigma,
        mode=args.namr_mode
    )
    namr_loss_fn = namr_loss_sce if args.namr_loss == 'sce' else namr_loss

    # Collect trainable parameters across all model heads
    params = (list(model.parameters()) + list(csp_head.parameters())
              + list(namr_head.parameters()) + list(c3l_head.parameters()))
    if p_head is not None:
        params += list(p_head.parameters())

    # Set up AdamW optimizer with cosine annealing LR schedule
    opt = AdamW(params, lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, args.epochs)
    n_params = sum(p.numel() for p in params)
    total_edges = sum(v.size(1) for v in ei_dict.values())
    print(f"Model {n_params:,} params | {n_genes} genes | {n_cancers} cancers | "
          f"{'patient+' if has_patient else 'no patient'} | {total_edges:,} edges | "
          f"C3L={'real targets' if targets is not None else 'DISABLED'}")

    # Initialize per-loss history tracking dict
    log = {'csp': [], 'namr': [], 'c3l': [], 'total': []}
    t0 = time.time()
    start_epoch = 1
    # Optionally resume from a previous checkpoint
    if args.resume and os.path.exists(args.resume):
        old = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(old['model'])
        csp_head.load_state_dict(old['csp'])
        namr_head.load_state_dict(old['namr'])
        c3l_head.load_state_dict(old['c3l'])
        opt.load_state_dict(old.get('optimizer', opt.state_dict()))
        log = {k: list(v) for k, v in old.get('loss_history', {}).items()}
        start_epoch = int(old.get('epoch', 0)) + 1
        print(f'[resume] from epoch {start_epoch - 1}')

    # --- Main training epoch loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Forward pass through HGT encoder to get node embeddings
        z = model(x_dict, ei_dict)
        gene_z = z['gene']

        # --- CSP objective: link-prediction on sampled positive/negative PPI pairs ---
        pa, pb, na, nb = csp_sampler.sample(
            min(config.CSP_BATCH_SIZE, n_genes), device)
        loss_csp = csp_loss(
            csp_head(gene_z[pa], gene_z[pb]),
            csp_head(gene_z[na], gene_z[nb])
        )

        # --- NAMR objective: denoise/mask a subset of gene features and reconstruct ---
        if args.namr_weight > 0:
            x_noisy, midx = namr_masker.apply(x_dict['gene'], device)
            xnd = {k: (x_noisy if k == 'gene' else v) for k, v in x_dict.items()}
            zn = model(xnd, ei_dict)
            if args.decoder == 'skip':
                # Skip decoder variant: concatenate hidden state with noisy input
                namr_in = torch.cat([zn['gene'][midx], x_noisy[midx]], dim=-1)
            else:
                namr_in = zn['gene'][midx]
            loss_namr = namr_loss_fn(namr_head(namr_in), x_dict['gene'][midx])
        else:
            loss_namr = torch.tensor(0.0, device=device)

        # --- C3L objective (real targets): contrastive gene<->cancer InfoNCE loss ---
        if targets is not None and n_cancers >= 2:
            gene_proj = c3l_head(gene_z)
            cancer_proj = c3l_head(z['cancer'])
            # Subsample genes per step to bound the contrastive batch size
            n_c3l = min(DEFAULT_C3L_BATCH_SIZE, n_genes)
            gidx = torch.randperm(n_genes, device=device)[:n_c3l]
            loss_c3l = c3l_info_nce_loss(
                gene_proj[gidx], cancer_proj, targets[gidx],
                temperature=config.C3L_TEMPERATURE
            )
        else:
            loss_c3l = torch.tensor(0.0, device=device)

        # Weighted sum combination of all three training objectives
        loss = (args.csp_weight * loss_csp
                + args.namr_weight * loss_namr
                + args.c3l_weight * loss_c3l)

        # --- Patient-level NAMR (only active when patient-gene edges exist) ---
        if has_patient and has_patient_gene and 'patient' in z:
            p_noisy, pmid = namr_masker.apply(x_dict['patient'], device)
            xndp = {k: (p_noisy if k == 'patient' else v) for k, v in x_dict.items()}
            znp = model(xndp, ei_dict)
            if args.decoder == 'skip':
                p_in = torch.cat([znp['patient'][pmid], p_noisy[pmid]], dim=-1)
            else:
                p_in = znp['patient'][pmid]
            loss = loss + args.namr_weight * namr_loss_fn(
                p_head(p_in), x_dict['patient'][pmid])
        elif has_patient and not has_patient_gene:
            print("[WARN] patient nodes present but no patient-gene edges; "
                  "patient tasks skipped (HGTConv would drop them). "
                  "Run 33_build_patient_gene_edges.py.")

        # Backpropagation with gradient clipping to prevent exploding gradients
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, DEFAULT_GRAD_CLIP_NORM)
        opt.step()
        sch.step()

        # Log per-epoch loss values for post-hoc analysis
        log['csp'].append(loss_csp.item())
        log['namr'].append(loss_namr.item())
        log['c3l'].append(loss_c3l.item() if isinstance(loss_c3l, torch.Tensor) else 0.0)
        log['total'].append(loss.item())

        # Periodically save intermediate checkpoints
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_out = args.out or os.path.join(
                config.MODEL_DIR, CKPT_NAME.format(epochs=args.epochs))
            torch.save({
                'model': model.state_dict(),
                'csp': csp_head.state_dict(),
                'namr': namr_head.state_dict(),
                'c3l': c3l_head.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch,
                'loss_history': {k: v for k, v in log.items()},
                'args': vars(args)
            }, ckpt_out)
            print(f'[ckpt] saved epoch {epoch} -> {ckpt_out}')

        # Periodic progress logging to stdout
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"E {epoch:4d} | CSP {loss_csp.item():.4f} | NAMR {loss_namr.item():.4f} "
                  f"| C3L {loss_c3l.item() if isinstance(loss_c3l, torch.Tensor) else 0:.4f} "
                  f"| Tot {loss.item():.4f}")

    # --- Convergence heuristic diagnostic ---
    # Compare mean total loss around the mid-training window vs the final window
    mid_idx = len(log['total']) // 2
    mid = np.mean(log['total'][mid_idx - CONV_MID_WINDOW:mid_idx + CONV_MID_WINDOW])
    end = np.mean(log['total'][-CONV_FINAL_WINDOW:])
    print(f"\nDone in {(time.time()-t0)/60:.1f} min | conv={'PASS' if end < mid else 'MORE'} "
          f"(mid {mid:.4f} -> end {end:.4f})")

    # Save the final model checkpoint (includes all heads + loss history + args)
    out = args.out or os.path.join(
        config.MODEL_DIR, CKPT_NAME.format(epochs=args.epochs))
    torch.save({
        'model': model.state_dict(),
        'csp': csp_head.state_dict(),
        'namr': namr_head.state_dict(),
        'c3l': c3l_head.state_dict(),
        'loss_history': {k: v for k, v in log.items()},
        'args': vars(args)
    }, out)
    print(f"Saved {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(config.DATA_PROC, DEFAULT_GRAPH))
    ap.add_argument('--targets', default=os.path.join(config.DATA_PROC, DEFAULT_TARGETS))
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument('--hidden', type=int, default=DEFAULT_HIDDEN_DIM)
    ap.add_argument('--layers', type=int, default=DEFAULT_HGT_LAYERS)
    ap.add_argument('--heads', type=int, default=DEFAULT_HGT_HEADS)
    ap.add_argument('--namr_mode', choices=['denoise', 'mask'], default='denoise')
    ap.add_argument('--namr_loss', choices=['mse_cos', 'sce'], default='mse_cos',
                    help='NAMR reconstruction criterion (mse_cos = default; sce = GraphMAE scaled cosine error)')
    ap.add_argument('--namr_sigma', type=float, default=config.NAMR_NOISE_SIGMA,
                    help='NAMR noise std (H2)')
    ap.add_argument('--namr_weight', type=float, default=1.0,
                    help='NAMR loss weight (H4)')
    ap.add_argument('--csp_weight', type=float, default=1.0,
                    help='CSP loss weight (0 disables CSP)')
    ap.add_argument('--c3l_weight', type=float, default=0.5,
                    help='C3L loss weight (0 disables C3L)')
    ap.add_argument('--standardize_features', action='store_true',
                    help='re-standardize gene features per-dim in memory (H1)')
    ap.add_argument('--out', default=None, help='checkpoint output path')
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--save_every', type=int, default=100,
                    help='periodic checkpoint interval (0 = off)')
    ap.add_argument('--resume', default=None,
                    help='resume from checkpoint path')
    ap.add_argument('--decoder', choices=['plain', 'skip'], default='plain',
                    help='NAMR decoder: plain (default) or skip-connected (diagnostic variant)')
    args = ap.parse_args()
    pretrain(torch.load(args.data, map_location='cpu'), args.targets, args)
