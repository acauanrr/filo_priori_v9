"""
LambdaRank Loss Implementation for Filo-Priori V10.

LambdaRank is a Learning-to-Rank algorithm that directly optimizes
ranking metrics like NDCG by computing gradients proportional to
the change in metric if two items are swapped.

Key Insight:
    Instead of optimizing a surrogate loss (like cross-entropy),
    LambdaRank computes:

    λ_ij = |ΔNDCG_ij| × sigmoid(-(s_i - s_j))

    This ensures the gradient is proportional to:
    1. How much NDCG would improve if we fix the ordering
    2. How wrong the current prediction is

Why LambdaRank for TCP?
    - Focal Loss optimizes classification, not ranking
    - APFD is a ranking metric
    - Failures are rare (0.2-5%), so we need to focus on getting them right
    - LambdaRank naturally handles class imbalance

References:
    - Burges et al., "Learning to Rank using Gradient Descent", ICML 2005
    - Burges et al., "From RankNet to LambdaRank to LambdaMART", 2010
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ndcg_utils import compute_ideal_dcg, compute_delta_ndcg


class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss for Learning-to-Rank.

    This loss function computes gradients that are proportional to
    the change in NDCG that would result from swapping two items.

    Args:
        sigma: Scaling factor for the sigmoid. Higher = sharper gradients.
        ndcg_at_k: Compute ΔNDCG at this cutoff (None = full list).
        weighting: How to weight pairs:
            - 'ndcg': Weight by |ΔNDCG| (standard LambdaRank)
            - 'uniform': All pairs equal weight
            - 'position': Weight by position difference

    Example:
        >>> loss_fn = LambdaRankLoss(sigma=1.0, ndcg_at_k=10)
        >>> scores = model(features)  # [batch, num_items]
        >>> loss = loss_fn(scores, relevances)
        >>> loss.backward()
    """

    def __init__(
        self,
        sigma: float = 1.0,
        ndcg_at_k: Optional[int] = None,
        weighting: str = 'ndcg',
        reduction: str = 'mean'
    ):
        super().__init__()
        self.sigma = sigma
        self.ndcg_at_k = ndcg_at_k
        self.weighting = weighting
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute LambdaRank loss.

        Args:
            scores: Predicted scores [batch, num_items].
                   Higher score = higher predicted relevance.
            relevances: Ground truth relevances [batch, num_items].
                       For TCP: 1 = fail, 0 = pass.
            mask: Optional mask for padding [batch, num_items].
                  1 = valid, 0 = padding.

        Returns:
            Loss value (scalar or [batch] depending on reduction).
        """
        batch_size, num_items = scores.shape
        device = scores.device

        # Handle masking
        if mask is not None:
            # Set masked items to very low score so they rank last
            scores = scores.masked_fill(mask == 0, float('-inf'))
            relevances = relevances.masked_fill(mask == 0, 0)

        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for b in range(batch_size):
            s = scores[b]  # [num_items]
            y = relevances[b]  # [num_items]

            # Find relevant and non-relevant items
            # For TCP: relevant = fails (1), non-relevant = passes (0)
            rel_indices = (y > 0).nonzero(as_tuple=True)[0]
            non_rel_indices = (y == 0).nonzero(as_tuple=True)[0]

            if len(rel_indices) == 0 or len(non_rel_indices) == 0:
                # No valid pairs in this sample
                continue

            # Compute IDCG for normalization
            idcg = compute_ideal_dcg(y.unsqueeze(0), k=self.ndcg_at_k)[0]

            if idcg == 0:
                continue

            # For each pair (i, j) where y_i > y_j
            for i in rel_indices:
                for j in non_rel_indices:
                    # Score difference
                    s_diff = s[i] - s[j]

                    # Compute weight (|ΔNDCG|)
                    if self.weighting == 'ndcg':
                        delta_weight = self._compute_delta_ndcg_fast(y, i, j, idcg)
                    elif self.weighting == 'position':
                        delta_weight = abs(i.item() - j.item()) / num_items
                    else:  # uniform
                        delta_weight = 1.0

                    # LambdaRank gradient magnitude
                    # λ = -σ × sigmoid(-σ(s_i - s_j)) × |ΔNDCG|
                    # Loss = -log(sigmoid(σ(s_i - s_j)))
                    # = log(1 + exp(-σ(s_i - s_j)))

                    # Weighted pairwise loss
                    pair_loss = delta_weight * F.softplus(-self.sigma * s_diff)
                    total_loss = total_loss + pair_loss
                    num_pairs += 1

        # Normalize
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        if self.reduction == 'none':
            return total_loss
        elif self.reduction == 'sum':
            return total_loss * batch_size
        else:  # mean
            return total_loss

    def _compute_delta_ndcg_fast(
        self,
        relevances: torch.Tensor,
        i: int,
        j: int,
        idcg: float
    ) -> float:
        """
        Fast computation of |ΔNDCG| for a pair swap.

        For binary relevances (0/1), this simplifies considerably.
        """
        # For binary relevance:
        # If i is relevant (1) and j is not (0):
        # Current DCG contribution: (2^1 - 1)/log(i+2) + (2^0 - 1)/log(j+2)
        #                         = 1/log(i+2) + 0
        # After swap:             = 1/log(j+2) + 0

        # Gains (for binary: 2^1 - 1 = 1, 2^0 - 1 = 0)
        gain_i = 1.0  # Assuming i is relevant
        gain_j = 0.0  # Assuming j is not relevant

        # Discounts
        disc_i = 1.0 / math.log2(i.item() + 2)
        disc_j = 1.0 / math.log2(j.item() + 2)

        # Delta DCG = (gain_i - gain_j) × (disc_j - disc_i)
        #           = 1 × (disc_j - disc_i)
        delta_dcg = abs(disc_j - disc_i)

        # Normalize
        delta_ndcg = delta_dcg / (idcg + 1e-8)

        return delta_ndcg


class LambdaLoss(nn.Module):
    """
    Generalized LambdaLoss framework.

    This is a more modern and flexible version of LambdaRank that
    supports different weighting schemes and loss functions.

    From: Wang et al., "The LambdaLoss Framework for Ranking Metric
    Optimization", CIKM 2018.

    Args:
        sigma: Scaling factor.
        weighting_scheme: How to weight pairs:
            - 'lambdarank': Standard |ΔNDCG| weighting
            - 'ndcg1': NDCG@1 focused
            - 'ndcg2': Squared NDCG weighting
            - 'dcg': DCG-based weighting
        pair_loss: Pairwise loss function:
            - 'logistic': log(1 + exp(-s_diff))
            - 'hinge': max(0, margin - s_diff)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        weighting_scheme: str = 'lambdarank',
        pair_loss: str = 'logistic',
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.sigma = sigma
        self.weighting_scheme = weighting_scheme
        self.pair_loss = pair_loss
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute LambdaLoss.

        Args:
            scores: Predicted scores [batch, num_items].
            relevances: Ground truth relevances [batch, num_items].
            mask: Optional padding mask.

        Returns:
            Loss value.
        """
        batch_size, num_items = scores.shape
        device = scores.device

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            relevances = relevances.masked_fill(mask == 0, 0)

        # Compute pairwise score differences
        # s_diff[i, j] = s[i] - s[j]
        s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [batch, n, n]

        # Compute pairwise relevance differences
        # We want pairs where y[i] > y[j]
        y_diff = relevances.unsqueeze(2) - relevances.unsqueeze(1)  # [batch, n, n]

        # Valid pairs: y[i] > y[j] (i.e., y_diff > 0)
        valid_pairs = (y_diff > 0).float()

        # Compute weights
        if self.weighting_scheme == 'lambdarank':
            weights = self._compute_ndcg_weights(relevances)  # [batch, n, n]
        elif self.weighting_scheme == 'uniform':
            weights = torch.ones_like(s_diff)
        else:
            weights = torch.ones_like(s_diff)

        # Compute pairwise loss
        if self.pair_loss == 'logistic':
            pair_losses = F.softplus(-self.sigma * s_diff)
        elif self.pair_loss == 'hinge':
            pair_losses = F.relu(self.margin - self.sigma * s_diff)
        else:
            pair_losses = F.softplus(-self.sigma * s_diff)

        # Apply weights and mask
        weighted_losses = weights * valid_pairs * pair_losses

        # Reduce
        if self.reduction == 'mean':
            num_valid = valid_pairs.sum()
            if num_valid > 0:
                return weighted_losses.sum() / num_valid
            else:
                return torch.tensor(0.0, device=device)
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        else:
            return weighted_losses.sum(dim=[1, 2])

    def _compute_ndcg_weights(
        self,
        relevances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute |ΔNDCG| weights for all pairs.

        Returns:
            Weight matrix [batch, num_items, num_items].
        """
        batch_size, num_items = relevances.shape
        device = relevances.device

        # Precompute discounts
        positions = torch.arange(1, num_items + 1, device=device).float()
        discounts = 1.0 / torch.log2(positions + 1)

        # Precompute gains
        gains = (2.0 ** relevances) - 1.0  # [batch, num_items]

        # IDCG
        sorted_gains, _ = torch.sort(gains, dim=1, descending=True)
        idcg = (sorted_gains * discounts).sum(dim=1, keepdim=True)  # [batch, 1]
        idcg = idcg.clamp(min=1e-8)

        # For each pair (i, j):
        # ΔDCG = |gain_i - gain_j| × |discount_i - discount_j|
        gain_diff = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()  # [batch, n, n]
        disc_diff = (discounts.unsqueeze(1) - discounts.unsqueeze(0)).abs()  # [n, n]

        delta_dcg = gain_diff * disc_diff.unsqueeze(0)

        # Normalize
        weights = delta_dcg / idcg.unsqueeze(2)

        return weights


class RankingHingeLoss(nn.Module):
    """
    Simple pairwise ranking loss with hinge margin.

    For pairs (i, j) where y_i > y_j, we want s_i > s_j + margin.

    Loss = Σ max(0, margin - (s_i - s_j))
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor
    ) -> torch.Tensor:
        """Compute hinge ranking loss."""
        batch_size, num_items = scores.shape

        total_loss = 0.0
        num_pairs = 0

        for b in range(batch_size):
            s = scores[b]
            y = relevances[b]

            rel_idx = (y > 0).nonzero(as_tuple=True)[0]
            non_rel_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(rel_idx) == 0 or len(non_rel_idx) == 0:
                continue

            for i in rel_idx:
                for j in non_rel_idx:
                    pair_loss = F.relu(self.margin - (s[i] - s[j]))
                    total_loss += pair_loss
                    num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss


class ListwiseSoftmaxLoss(nn.Module):
    """
    Listwise softmax cross-entropy loss (ListNet).

    Treats ranking as a probability distribution and minimizes
    KL divergence between predicted and true distributions.

    P_true(i) = exp(y_i) / Σ exp(y_j)
    P_pred(i) = exp(s_i) / Σ exp(s_j)

    Loss = -Σ P_true(i) × log(P_pred(i))
    """

    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute listwise softmax loss."""
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            relevances = relevances.masked_fill(mask == 0, float('-inf'))

        # Compute distributions
        p_true = F.softmax(relevances / self.temperature, dim=-1)
        log_p_pred = F.log_softmax(scores / self.temperature, dim=-1)

        # KL divergence (cross-entropy since we have true distribution)
        loss = -(p_true * log_p_pred).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
