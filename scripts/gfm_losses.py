#!/usr/bin/env python3
"""
@file    gfm_losses.py
@brief   Self-supervised pre-training loss functions and samplers for the GFM
         project. Implements the three pre-training objectives used by the
         Gene-Fusion-Model: (1) NAMR — Node Attribute (Masked) Reconstruction,
         (2) CSP — Contextual Subgraph Prediction (binary PPI link prediction),
         and (3) C3L — Cross-Cancer Contrastive Learning (InfoNCE over real
         cancer-type specificity targets). All code is pure PyTorch and does
         not depend on PyTorch Geometric, enabling lightweight CPU unit tests.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08
@ingroup core_losses

Input:  PyTorch tensors produced by the GFM encoder heads and, for the CSP
        sampler, the PPI edge_index tensor. Default hyper-parameters are drawn
        from config.py when this module is used inside the training loop.
Output: Scalar loss tensors (NAMR, CSP, C3L) and, for the samplers, structured
        tuples of indices ready to be fed into the encoder.
Run:    This is primarily a library module. Standalone usage for unit tests:
            python scripts/gfm_losses.py --help
        (No CLI is implemented yet; import the symbols below directly.)
"""

import torch
import torch.nn.functional as F

# Public API exported by this module. Import with:
#   from gfm_losses import namr_loss, csp_loss, c3l_info_nce_loss, CSPSampler, NAMRMask
__all__ = [
    "namr_loss",
    "namr_loss_sce",
    "csp_loss",
    "c3l_info_nce_loss",
    "CSPSampler",
    "NAMRMask",
]

# =============================================================================
# Section 1: NAMR — Node Attribute (Masked) Reconstruction
# =============================================================================
# NAMR reconstructs node-level features after a perturbation. The original
# implementation adds Gaussian noise rather than zeroing tokens, so the
# objective is formally a *denoising* criterion. We keep the behaviour for
# backward compatibility but expose a true masking mode via NAMRMask.mode.
#
# Default hyper-parameters for NAMR (shared by NAMRMask and the CLI). These
# mirror the constants in config.py but are re-declared here so the loss
# module remains self-contained for unit testing.
NAMR_DEFAULT_MASK_RATIO = 0.15
NAMR_DEFAULT_NOISE_SIGMA = 0.5
NAMR_COSINE_WEIGHT = 0.1      # Lambda weighting for the cosine-similarity penalty
NAMR_SCE_GAMMA_DEFAULT = 2.0  # Scaling exponent for GraphMAE-style SCE loss


def namr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    @brief   Compute the composite NAMR reconstruction loss: pixel-level MSE
             plus a weighted (1 - cosine-similarity) term that penalises
             angular error between the predicted and clean feature vectors.
    @param   pred     Tensor of shape [N, D] containing reconstructed features.
    @param   target   Tensor of shape [N, D] holding the original clean features
                      (the denoising target). Must share dtype and device with pred.
    @return  Scalar loss tensor (mean-reduced over all N nodes and D dimensions),
             differentiable w.r.t. pred.
    @note    The cosine-similarity weight is fixed at NAMR_COSINE_WEIGHT = 0.1
             which was tuned on the TCGA pan-cancer pilot. For a pure cosine
             criterion without MSE see namr_loss_sce() (the GraphMAE baseline).
    """
    # Element-wise L2 reconstruction error — penalises magnitude mismatch
    mse = F.mse_loss(pred, target)
    # Angular penalty (1 - mean cosine sim) — penalises direction mismatch,
    # averaged across the N nodes before weighting.
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    # Combine the two objectives with the tuned weighting factor.
    return mse + NAMR_COSINE_WEIGHT * cos


def namr_loss_sce(pred: torch.Tensor, target: torch.Tensor,
                  gamma: float = NAMR_SCE_GAMMA_DEFAULT) -> torch.Tensor:
    """
    @brief   GraphMAE-style Scaled Cosine Error (SCE) reconstruction loss.
             Computes mean((1 - cos(pred, target))^gamma). Used as a baseline
             objective for the existing-methods ablation (review item 6).
    @param   pred     Reconstructed feature matrix, shape [N, D].
    @param   target   Clean target feature matrix, shape [N, D].
    @param   gamma    Scaling exponent for the (1 - cos) term. gamma == 1
                      recovers plain cosine loss; gamma == 2 penalises large
                      angular errors more heavily (GraphMAE default).
    @return  Scalar loss tensor, mean-reduced over nodes.
    @note    Reference: Hou et al. "GraphMAE: Self-Supervised Masked Graph
             Autoencoders", KDD 2022. The SCE criterion is more robust to
             feature-magnitude variations than raw MSE on expression profiles.
    """
    # Pairwise cosine similarity along the feature dimension -> shape [N]
    cos = F.cosine_similarity(pred, target, dim=-1)
    # Scale by raising (1 - cos) to the gamma-th power, then average over nodes.
    return (1.0 - cos).pow(gamma).mean()


class NAMRMask:
    """
    @brief   Configurable perturbation strategy for the NAMR denoising /
             masked-autoencoder objective. Two modes are supported:
             - "denoise" (default): adds N(0, sigma) Gaussian noise to the
               feature rows of the selected nodes (matches the published runs).
             - "mask":            replaces the selected rows with zeros,
               producing a true masked-autoencoder setup as in GraphMAE.
    """

    def __init__(self, mask_ratio: float = NAMR_DEFAULT_MASK_RATIO,
                 sigma: float = NAMR_DEFAULT_NOISE_SIGMA,
                 mode: str = "denoise"):
        """
        @brief   Construct a NAMRMask generator with fixed hyper-parameters.
        @param   mask_ratio   Fraction (0, 1) of nodes to perturb per call to apply().
                              Must be strictly between 0 and 1.
        @param   sigma        Standard deviation of the Gaussian noise applied in
                              "denoise" mode. Ignored when mode == "mask".
        @param   mode         Perturbation strategy selector; one of {"denoise", "mask"}.
        @return  NAMRMask instance configured for the requested perturbation style.
        @note    A minimum of one node is always perturbed regardless of how
                 small mask_ratio * N becomes; this avoids degenerate zero-loss
                 batches on tiny sub-graphs.
        """
        # Guard against invalid configuration up-front so errors surface at
        # construction time rather than mid-training.
        assert 0 < mask_ratio < 1, "mask_ratio must lie strictly inside (0, 1)"
        assert mode in ("denoise", "mask"), f"unknown mode '{mode}'"
        self.mask_ratio = mask_ratio
        self.sigma = sigma
        self.mode = mode

    def apply(self, x: torch.Tensor, device) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        @brief   Apply the configured perturbation to a copy of the input
                 feature matrix and return both the noisy matrix and the list
                 of perturbed row indices.
        @param   x        Full feature matrix of shape [N, D]. This tensor is
                          NOT modified in-place; the returned x_noisy is a clone.
        @param   device   torch.device on which to allocate the random index
                          tensor and the Gaussian noise sample. Typically equal
                          to x.device but passed explicitly for symmetry with
                          CSPSampler.sample().
        @return  Tuple of (x_noisy, mask_indices) where:
                 - x_noisy: Tensor of shape [N, D], a clone of x with mask_ratio
                   fraction of the rows perturbed according to self.mode.
                 - mask_indices: LongTensor of shape [n_mask] listing the row
                   indices that were modified; useful for computing loss only
                   over the perturbed nodes (not currently done by namr_loss).
        """
        n = x.size(0)                              # Total number of nodes in this batch
        # Ensure at least one node is selected even on tiny graphs where
        # floor(mask_ratio * n) would otherwise round down to 0.
        n_mask = max(1, int(n * self.mask_ratio))
        # Sample n_mask unique row indices without replacement via randperm.
        idx = torch.randperm(n, device=device)[:n_mask]
        # Operate on a clone so the caller's original feature matrix stays intact.
        x_noisy = x.clone()
        if self.mode == "denoise":
            # Denoising objective: add zero-mean Gaussian noise with std = sigma.
            x_noisy[idx] += torch.randn_like(x_noisy[idx]) * self.sigma
        else:
            # Masking objective: replace entire rows with the zero token.
            x_noisy[idx] = 0.0
        return x_noisy, idx


# =============================================================================
# Section 2: CSP — Contextual Subgraph Prediction (binary PPI link prediction)
# =============================================================================
# CSP frames PPI link prediction as a binary classification task: given two
# gene embeddings (a, b) predict whether (a, b) is a real edge in the STRING
# graph. The loss is simply symmetric BCE with logits averaged over the
# positive and negative halves of the mini-batch.

def csp_loss(pred_pos: torch.Tensor, pred_neg: torch.Tensor) -> torch.Tensor:
    """
    @brief   Compute the symmetric binary cross-entropy CSP loss over positive
             (real PPI) and negative (synthetic non-PPI) logit predictions.
    @param   pred_pos   Logit tensor of shape [B] for positive PPI pairs.
                        Higher values encode the model's belief the pair interacts.
    @param   pred_neg   Logit tensor of shape [B] for negative non-interacting pairs.
    @return  Scalar loss equal to 0.5 * (BCE(pred_pos, 1) + BCE(pred_neg, 0)).
    @note    Using binary_cross_entropy_with_logits (rather than sigmoid + BCE)
             is numerically more stable and preserves automatic mixed-precision
             compatibility under torch.cuda.amp.
    """
    # Positive pairs: encourage high logits -> target = 1
    pos = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos))
    # Negative pairs: encourage low logits -> target = 0
    neg = F.binary_cross_entropy_with_logits(pred_neg, torch.zeros_like(pred_neg))
    # Equal weighting per half-batch keeps the loss scale independent of B.
    return (pos + neg) * 0.5


class CSPSampler:
    """
    @brief   Vectorised mini-batch sampler for the CSP pre-training objective.
             Produces balanced batches of (a) positive gene pairs drawn
             uniformly from the known PPI edge set and (b) negative gene pairs
             drawn uniformly from the complement of the PPI set.

             Negative sampling uses canonical integer edge keys combined with
             torch.searchsorted on a sorted key tensor to reject real edges in
             bulk, avoiding the O(batch_size) Python loop + .item() calls that
             existed in the original implementation (review item 7).
    """

    def __init__(self, edge_index: torch.Tensor, n_genes: int):
        """
        @brief   Build a sampler from the full undirected PPI adjacency list.
                 Computes a sorted, deduplicated set of canonical edge keys
                 used later for fast negative-sample rejection.
        @param   edge_index   PPI edge tensor of shape [2, E]. Both (u->v) and
                              (v->u) may be present; the constructor canonicalises
                              them automatically.
        @param   n_genes      Total number of distinct gene nodes |V| in the
                              graph. Required to map unordered (u, v) pairs to
                              unique integer keys via base-|V| encoding.
        @return  CSPSampler instance with pre-computed PPI lookup structures.
        @note    edge_index is assumed to hold indices in [0, n_genes); passing
                 out-of-range indices will silently corrupt the canonical key
                 computation without raising an error.
        """
        # Canonicalise each undirected edge so (u, v) and (v, u) share one key.
        # lo holds the smaller index per edge; hi holds the larger.
        lo = torch.minimum(edge_index[0], edge_index[1])
        hi = torch.maximum(edge_index[0], edge_index[1])
        # Base-n_genes encoding produces a unique integer key per unordered pair.
        keys = lo * n_genes + hi
        # Deduplicate and sort the keys — torch.searchsorted requires sorted input.
        self.ppi_keys = torch.unique(keys).to(torch.long)
        # Also retain the canonicalised positive pairs (shape [2, P]) for fast
        # positive sampling via randint indexing.
        self.ppi_tensor = torch.stack([lo, hi], dim=0)
        self.n_genes = n_genes

    def sample(self, batch_size: int, device) -> "tuple":
        """
        @brief   Draw one balanced CSP mini-batch of positive and negative pairs.
        @param   batch_size   Number of POSITIVE pairs to sample; the sampler
                              also produces exactly batch_size NEGATIVE pairs,
                              giving a total batch cardinality of 2 * batch_size.
        @param   device       torch.device on which the returned index tensors
                              are allocated. Matrices are copied off the CPU as
                              needed so the main training loop stays on GPU.
        @return  4-tuple of LongTensors (pos_a, pos_b, neg_a, neg_b), each of
                 shape [batch_size], where:
                 - (pos_a[i], pos_b[i]) is a real PPI edge,
                 - (neg_a[i], neg_b[i]) is a guaranteed non-PPI pair with a_i != b_i.
        @note    Negative pairs are generated via vectorised rejection sampling.
                 Because the expected number of rejections per round is
                 batch_size * (2E / (n_genes choose 2)) and the graph is sparse,
                 the loop almost always exits after a single iteration.
        """
        # ---------- Positive sampling: pick batch_size edges from the PPI set --
        # Choose edge indices uniformly at random, then dereference into the
        # canonicalised pair list and move tensors to the requested device.
        pos_idx = torch.randint(0, self.ppi_tensor.size(1), (batch_size,), device=device)
        pos_a = self.ppi_tensor[0, pos_idx].to(device)
        pos_b = self.ppi_tensor[1, pos_idx].to(device)

        # ---------- Negative sampling: vectorised rejection loop ---------------
        # Move the sorted PPI key set to GPU once so each round avoids a copy.
        keys_dev = self.ppi_keys.to(device)
        # Pre-allocate the negative-pair output buffers.
        neg_a = torch.empty(batch_size, dtype=torch.long, device=device)
        neg_b = torch.empty(batch_size, dtype=torch.long, device=device)
        # Boolean mask tracking which output slots still need a valid sample.
        remaining = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Keep sampling until every output slot has been filled with a pair
        # that is neither a self-loop nor a real PPI edge.
        while remaining.any():
            n = int(remaining.sum())   # Number of slots still vacant this round
            # Draw two independent integer samples in [0, n_genes) for each slot.
            a = torch.randint(0, self.n_genes, (n,), device=device)
            b = torch.randint(0, self.n_genes, (n,), device=device)
            # Canonicalise the candidate pairs for the PPI-membership test.
            lo = torch.minimum(a, b)
            hi = torch.maximum(a, b)
            keys = lo * self.n_genes + hi
            # searchsorted returns the insertion position of each candidate key
            # in the sorted PPI key set; this lets us check membership in O(log P).
            pos = torch.searchsorted(keys_dev, keys)
            # Clamp to the valid index range to avoid out-of-bounds on the next line
            # when a key is larger than every PPI key (searchsorted returns len(keys_dev)).
            pos = torch.clamp(pos, 0, keys_dev.numel() - 1)
            # Candidate is a real PPI edge iff the key at the insertion position
            # exactly matches the candidate key.
            is_ppi = keys_dev[pos] == keys
            # Reject trivial self-loops as well — they are not valid negatives.
            is_self = a == b
            # A candidate is acceptable when both failure flags are False.
            good = ~(is_ppi | is_self)
            # Map back from the dense "n"-sized candidate tensor to the correct
            # positions in the original batch_size-sized output buffers.
            idx_rem = remaining.nonzero(as_tuple=False).squeeze(-1)
            take = idx_rem[good]     # Slice of output slots we can fill this round
            neg_a[take] = a[good]
            neg_b[take] = b[good]
            # Mark the filled slots as completed before the next iteration.
            remaining[take] = False

        return pos_a, pos_b, neg_a, neg_b


# =============================================================================
# Section 3: C3L — Cross-Cancer Contrastive Learning with REAL specificity targets
# =============================================================================
# C3L casts cross-cancer representation learning as an N-way InfoNCE / softmax
# classification task. Each gene embedding is pushed toward the embedding of
# its TRUE most-specific cancer type (pre-computed from TCGA expression
# differential analysis) while being pulled away from the embeddings of the
# other C-1 cancer types. This fixes the earlier simplified C3L variant which
# used random targets and consequently reported ln(C) ~= 3.04 as its loss floor.

# Default InfoNCE temperature — shared with config.C3L_TEMPERATURE but
# re-declared here so the loss module is self-contained for unit tests.
C3L_DEFAULT_TEMPERATURE = 0.07


def c3l_info_nce_loss(
    gene_proj: torch.Tensor,
    cancer_proj: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = C3L_DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """
    @brief   Compute the C3L cross-cancer InfoNCE classification loss.

             For each gene embedding g_i the model picks a positive cancer
             target y_i = targets[i] (the cancer type where g_i is most
             specifically expressed according to real TCGA data) and treats
             the remaining C-1 cancer embeddings as negatives. The loss is
             standard cross-entropy over (g_i · c_j / T)_{j=1..C}.
    @param   gene_proj     Projected L2-normalised gene embeddings of shape [N, D].
                           Must already be projected through the C3L gene head and
                           normalised; the dot products then equal cosine similarities.
    @param   cancer_proj   Projected L2-normalised cancer-type embeddings of
                           shape [C, D]. Same D as gene_proj.
    @param   targets       LongTensor of shape [N] indexing into cancer_proj.
                           targets[i] = the positive cancer-type label for gene i.
    @param   temperature   InfoNCE sharpening temperature (default 0.07). Smaller
                           values sharpen the softmax and push the model toward
                           harder contrast; 0.07 is the standard SimCLR setting.
    @return  Scalar cross-entropy loss averaged over the N genes in the batch.
    @note    Expected random baseline is ln(C) (cross-entropy of a uniform
             distribution). For C = 21 TCGA cancer types this is ~ 3.04 nats,
             which was exactly the loss value reported by the earlier
             random-target C3L implementation — confirming the fix is correct.
    """
    # Similarity matrix S[i, j] = cos(gene_proj[i], cancer_proj[j]) == dot product
    # because both inputs are L2-normalised. Shape: [N, C].
    logits = gene_proj @ cancer_proj.t()
    # Apply the sharpening temperature before the softmax inside cross_entropy.
    logits = logits / temperature
    # F.cross_entropy implicitly does log_softmax + NLLLoss, mean-reduced over N.
    return F.cross_entropy(logits, targets)
