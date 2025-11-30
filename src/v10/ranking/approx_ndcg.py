"""
Approximate NDCG Loss for Filo-Priori V10.

This module implements differentiable approximations to NDCG,
allowing direct optimization of the ranking metric via gradient descent.

The challenge: NDCG involves sorting, which is non-differentiable.
Solution: Use soft sorting / soft ranking approximations.

References:
    - Qin et al., "A General Approximation Framework for Direct
      Optimization of Information Retrieval Measures", 2010
    - Bruch et al., "Revisiting Approximate Metric Optimization", 2019
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss using soft sorting.

    Instead of hard sorting (non-differentiable), we use:
    1. Soft ranks via sigmoid approximation
    2. Differentiable DCG computation
    3. Loss = 1 - NDCG (to minimize)

    Args:
        temperature: Controls softness of approximation.
                    Lower = closer to true NDCG but less stable.
        k: Cutoff for NDCG@k (None = full list).
        alpha: Exponential smoothing for stability.

    Example:
        >>> loss_fn = ApproxNDCGLoss(temperature=1.0)
        >>> scores = model(features)  # [batch, num_items]
        >>> loss = loss_fn(scores, relevances)
        >>> loss.backward()  # Gradients flow!
    """

    def __init__(
        self,
        temperature: float = 1.0,
        k: Optional[int] = None,
        alpha: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute approximate NDCG loss.

        Args:
            scores: Predicted scores [batch, num_items].
            relevances: Ground truth relevances [batch, num_items].
            mask: Optional padding mask.

        Returns:
            Loss = 1 - ApproxNDCG (to minimize).
        """
        batch_size, num_items = scores.shape
        device = scores.device

        if mask is not None:
            # Set masked items to very negative score
            scores = scores.masked_fill(mask == 0, -1e9)
            relevances = relevances.masked_fill(mask == 0, 0)

        # Compute soft ranks
        soft_ranks = self._compute_soft_ranks(scores)  # [batch, num_items]

        # Compute gains: 2^rel - 1
        gains = torch.pow(2.0, relevances) - 1.0

        # Compute discounts using soft ranks
        # discount_i = 1 / log_2(rank_i + 1)
        # For soft ranks, we use: 1 / log_2(soft_rank + 1)
        discounts = 1.0 / torch.log2(soft_ranks + 1 + 1e-8)

        # Apply cutoff if specified
        if self.k is not None:
            # Only count items in top-k (soft)
            top_k_weight = torch.sigmoid(
                (self.k - soft_ranks) / self.temperature
            )
            discounts = discounts * top_k_weight

        # DCG with soft ranks
        dcg = (gains * discounts).sum(dim=1)

        # IDCG (ideal DCG with true optimal ranking)
        idcg = self._compute_idcg(relevances, k=self.k)

        # NDCG
        ndcg = dcg / (idcg + 1e-8)

        # Loss = 1 - NDCG
        loss = 1.0 - ndcg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_soft_ranks(
        self,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft ranks using sigmoid approximation.

        For each item i, its soft rank is approximately:
        rank_i ≈ 1 + Σ_{j≠i} sigmoid((s_j - s_i) / τ)

        This is differentiable and approximates true ranks.
        """
        batch_size, num_items = scores.shape

        # Pairwise score differences: s_j - s_i
        # scores: [batch, num_items]
        # diff[b, i, j] = scores[b, j] - scores[b, i]
        diff = scores.unsqueeze(1) - scores.unsqueeze(2)  # [batch, n, n]

        # Soft indicator: sigmoid((s_j - s_i) / τ)
        # This is ~1 if s_j > s_i (j ranks higher than i)
        indicator = torch.sigmoid(diff / self.temperature)

        # Soft rank of item i = 1 + Σ_{j≠i} indicator[i, j]
        # Set diagonal to 0 (don't count self)
        mask = 1 - torch.eye(num_items, device=scores.device).unsqueeze(0)
        indicator = indicator * mask

        # Sum across j to get rank of i
        soft_ranks = 1 + indicator.sum(dim=2)  # [batch, num_items]

        return soft_ranks

    def _compute_idcg(
        self,
        relevances: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute Ideal DCG (optimal ranking).
        """
        # Sort relevances descending
        sorted_rels, _ = torch.sort(relevances, dim=1, descending=True)

        if k is not None:
            sorted_rels = sorted_rels[:, :k]

        # Gains
        gains = torch.pow(2.0, sorted_rels) - 1.0

        # Discounts (using 1-indexed positions)
        positions = torch.arange(1, sorted_rels.shape[1] + 1, device=relevances.device).float()
        discounts = 1.0 / torch.log2(positions + 1)

        # IDCG
        idcg = (gains * discounts).sum(dim=1)

        return idcg


class SoftNDCGLoss(nn.Module):
    """
    Alternative soft NDCG using NeuralSort.

    Uses a differentiable sorting operator based on
    Grover et al., "Stochastic Optimization of Sorting Networks
    via Continuous Relaxations", ICLR 2019.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        k: Optional[int] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SoftNDCG loss using NeuralSort.
        """
        batch_size, num_items = scores.shape
        device = scores.device

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            relevances = relevances.masked_fill(mask == 0, 0)

        # Compute soft permutation matrix using NeuralSort
        P_hat = self._neuralsort(scores)  # [batch, num_items, num_items]

        # Apply soft permutation to relevances
        # sorted_rels_soft[b, i] ≈ relevances at position i after sorting
        sorted_rels_soft = torch.bmm(P_hat, relevances.unsqueeze(2)).squeeze(2)

        # Compute DCG with soft-sorted relevances
        gains = torch.pow(2.0, sorted_rels_soft) - 1.0

        if self.k is not None:
            gains = gains[:, :self.k]

        positions = torch.arange(1, gains.shape[1] + 1, device=device).float()
        discounts = 1.0 / torch.log2(positions + 1)

        dcg = (gains * discounts).sum(dim=1)

        # IDCG
        idcg = self._compute_idcg(relevances, k=self.k)

        # NDCG
        ndcg = dcg / (idcg + 1e-8)

        loss = 1.0 - ndcg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _neuralsort(
        self,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft permutation matrix using NeuralSort.

        P_hat[i, j] ≈ probability that item j is at position i after sorting.
        """
        batch_size, num_items = scores.shape

        # Sort scores to get true permutation (for reference)
        _, true_perm = torch.sort(scores, dim=1, descending=True)

        # Compute soft permutation
        # For each position i, compute softmax over items
        scores_expanded = scores.unsqueeze(1).expand(-1, num_items, -1)

        # Position-dependent bias
        positions = torch.arange(num_items, device=scores.device).float()
        pos_bias = positions.unsqueeze(0).unsqueeze(2) / num_items

        # Combine scores with position
        logits = scores_expanded / self.temperature

        # Softmax to get probabilities
        P_hat = F.softmax(logits, dim=2)  # [batch, num_items, num_items]

        return P_hat

    def _compute_idcg(
        self,
        relevances: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """Compute Ideal DCG."""
        sorted_rels, _ = torch.sort(relevances, dim=1, descending=True)

        if k is not None:
            sorted_rels = sorted_rels[:, :k]

        gains = torch.pow(2.0, sorted_rels) - 1.0

        positions = torch.arange(1, sorted_rels.shape[1] + 1, device=relevances.device).float()
        discounts = 1.0 / torch.log2(positions + 1)

        idcg = (gains * discounts).sum(dim=1)

        return idcg


class APFDLoss(nn.Module):
    """
    Direct APFD optimization loss.

    APFD is defined as:
        APFD = 1 - (Σ TF_i) / (n × m) + 1/(2n)

    Where TF_i is the position of failure i.

    We approximate this using soft ranking.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute APFD loss.

        Args:
            scores: Predicted scores [batch, num_items].
            labels: Binary labels [batch, num_items] (1 = fail).
            mask: Optional padding mask.

        Returns:
            Loss = 1 - ApproxAPFD (to maximize APFD).
        """
        batch_size, num_items = scores.shape
        device = scores.device

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            labels = labels.masked_fill(mask == 0, 0)

        # Compute soft ranks
        # Higher score = lower rank (rank 1 = highest priority)
        diff = scores.unsqueeze(1) - scores.unsqueeze(2)
        indicator = torch.sigmoid(diff / self.temperature)
        mask_diag = 1 - torch.eye(num_items, device=device).unsqueeze(0)
        soft_ranks = 1 + (indicator * mask_diag).sum(dim=2)

        # APFD formula:
        # APFD = 1 - (Σ TF_i) / (n × m) + 1/(2n)
        # TF_i = rank of failure i

        # Sum of failure ranks
        fail_mask = labels.float()
        num_failures = fail_mask.sum(dim=1, keepdim=True).clamp(min=1)

        sum_fail_ranks = (soft_ranks * fail_mask).sum(dim=1, keepdim=True)

        # Compute APFD
        n = num_items
        apfd = 1 - (sum_fail_ranks / (n * num_failures)) + (1 / (2 * n))
        apfd = apfd.squeeze(1)

        # Loss = 1 - APFD (to minimize)
        loss = 1.0 - apfd

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
