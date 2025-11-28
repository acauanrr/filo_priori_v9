"""
Learning-to-Rank Loss Functions for Test Case Prioritization.

This module implements several L2R loss functions suitable for TCP:

1. ListNet: Listwise approach using cross-entropy on top-1 probabilities
2. ListMLE: Maximum likelihood estimation for permutation probability
3. LambdaRank: Pairwise approach with NDCG-aware gradients
4. ApproxNDCG: Differentiable approximation of NDCG

Reference papers:
- ListNet: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach", ICML 2007
- ListMLE: Xia et al., "Listwise Approach to Learning to Rank", ICML 2008
- LambdaRank: Burges et al., "Learning to Rank using Gradient Descent", ICML 2005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ListNetLoss(nn.Module):
    """
    ListNet Loss: Cross-entropy on top-1 probabilities.

    The loss compares the probability distribution induced by predicted
    scores with the distribution induced by true relevance labels.

    For TCP: relevance = 1 for failing tests, 0 for passing tests.

    Loss = -sum(P_y(j) * log(P_s(j)))
    where P_y(j) = exp(y_j) / sum(exp(y_i)) is the "true" distribution
    and P_s(j) = exp(s_j) / sum(exp(s_i)) is the predicted distribution
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        """
        Args:
            temperature: Temperature for softmax (higher = softer distribution)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ListNet loss.

        Args:
            scores: Predicted scores [batch_size, list_size] or [list_size]
            relevance: True relevance labels [batch_size, list_size] or [list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        # Handle 1D input (single list)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            # Set masked positions to very negative value
            scores = scores.masked_fill(~mask, float('-inf'))
            relevance = relevance.masked_fill(~mask, float('-inf'))

        # Compute softmax distributions
        p_true = F.softmax(relevance / self.temperature, dim=-1)
        p_pred = F.softmax(scores / self.temperature, dim=-1)

        # Cross-entropy loss
        loss = -torch.sum(p_true * torch.log(p_pred + self.eps), dim=-1)

        return loss.mean()


class ListMLELoss(nn.Module):
    """
    ListMLE Loss: Maximum likelihood estimation for ranking.

    Models the probability of observing the true ranking as a product
    of conditional probabilities (Plackett-Luce model).

    Loss = -log P(π* | s) where π* is the optimal permutation
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ListMLE loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Sort by relevance to get optimal permutation
        _, perm = relevance.sort(dim=-1, descending=True)

        # Reorder scores according to optimal permutation
        scores_sorted = torch.gather(scores, dim=-1, index=perm)

        # Apply mask
        if mask is not None:
            mask_sorted = torch.gather(mask, dim=-1, index=perm)
            scores_sorted = scores_sorted.masked_fill(~mask_sorted, float('-inf'))

        # Compute log-likelihood using Plackett-Luce
        # log P(π) = sum_i [s_π(i) - log(sum_j>=i exp(s_π(j)))]

        # Compute cumulative logsumexp from the end
        # This gives log(sum_j>=i exp(s_j)) for each position i
        max_score = scores_sorted.max(dim=-1, keepdim=True)[0]
        scores_shifted = scores_sorted - max_score

        # Reverse cumsum of exp
        exp_scores = torch.exp(scores_shifted)
        cumsum_exp = torch.flip(
            torch.cumsum(torch.flip(exp_scores, dims=[-1]), dim=-1),
            dims=[-1]
        )

        log_likelihood = scores_sorted - (torch.log(cumsum_exp + self.eps) + max_score)

        # Sum over list positions
        if mask is not None:
            log_likelihood = log_likelihood.masked_fill(~mask_sorted, 0)

        loss = -log_likelihood.sum(dim=-1)

        return loss.mean()


class LambdaRankLoss(nn.Module):
    """
    LambdaRank Loss: Pairwise loss weighted by NDCG delta.

    For each pair (i, j) where relevance[i] > relevance[j]:
    Loss += |delta_NDCG| * log(1 + exp(s_j - s_i))

    The delta_NDCG term ensures the loss focuses on swaps that
    matter most for the ranking metric.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        k: Optional[int] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            sigma: Scaling factor for score differences
            k: Cutoff for NDCG@k (None = full list)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.eps = eps

    def _dcg_gain(self, relevance: torch.Tensor) -> torch.Tensor:
        """Compute DCG gain: 2^rel - 1"""
        return torch.pow(2.0, relevance) - 1.0

    def _dcg_discount(self, rank: torch.Tensor) -> torch.Tensor:
        """Compute DCG discount: 1 / log2(rank + 1)"""
        return 1.0 / torch.log2(rank.float() + 2.0)

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute LambdaRank loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Get current ranking (by predicted scores)
        _, pred_ranks = scores.sort(dim=-1, descending=True)
        ranks = torch.zeros_like(pred_ranks)
        ranks.scatter_(1, pred_ranks, torch.arange(list_size, device=device).expand(batch_size, -1))
        ranks = ranks + 1  # 1-indexed

        # Compute ideal DCG
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        if self.k is not None:
            k = min(self.k, list_size)
            sorted_rel = sorted_rel[:, :k]

        ideal_dcg = (self._dcg_gain(sorted_rel) *
                    self._dcg_discount(torch.arange(sorted_rel.size(1), device=device))).sum(dim=-1)
        ideal_dcg = ideal_dcg.clamp(min=self.eps)

        # Create pairwise comparisons
        # rel_diff[i,j] = relevance[i] - relevance[j]
        rel_diff = relevance.unsqueeze(-1) - relevance.unsqueeze(-2)
        score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)

        # Only consider pairs where i should be ranked higher than j
        valid_pairs = (rel_diff > 0).float()

        # Apply mask
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            valid_pairs = valid_pairs * pair_mask.float()

        # Compute delta NDCG for each swap
        gain_i = self._dcg_gain(relevance).unsqueeze(-1)
        gain_j = self._dcg_gain(relevance).unsqueeze(-2)

        disc_i = self._dcg_discount(ranks).unsqueeze(-1)
        disc_j = self._dcg_discount(ranks).unsqueeze(-2)

        # Delta if we swap i and j
        delta_ndcg = torch.abs(
            (gain_i - gain_j) * (disc_i - disc_j)
        ) / ideal_dcg.unsqueeze(-1).unsqueeze(-1)

        # Pairwise loss
        pairwise_loss = torch.log1p(torch.exp(-self.sigma * score_diff))

        # Weight by delta NDCG
        weighted_loss = delta_ndcg * pairwise_loss * valid_pairs

        # Sum over pairs
        loss = weighted_loss.sum(dim=(-1, -2)) / (valid_pairs.sum(dim=(-1, -2)) + self.eps)

        return loss.mean()


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG Loss: Differentiable approximation of 1-NDCG.

    Uses a soft ranking function to make NDCG differentiable.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        k: Optional[int] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            temperature: Temperature for soft ranking
            k: Cutoff for NDCG@k (None = full list)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.eps = eps

    def _soft_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute soft ranks using sigmoid approximation.

        Rank of item i ≈ 1 + sum_j sigmoid((s_j - s_i) / temperature)
        """
        batch_size, list_size = scores.shape

        # Pairwise differences
        diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)

        # Soft comparison
        comparison = torch.sigmoid(diff / self.temperature)

        # Sum to get soft rank (higher score = lower rank)
        soft_ranks = comparison.sum(dim=-1)

        return soft_ranks

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute approximate NDCG loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar): 1 - approx_NDCG
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Compute soft ranks
        soft_ranks = self._soft_rank(scores)

        # DCG computation
        gains = torch.pow(2.0, relevance) - 1.0
        discounts = 1.0 / torch.log2(soft_ranks + 2.0)

        if mask is not None:
            gains = gains * mask.float()
            discounts = discounts * mask.float()

        dcg = (gains * discounts).sum(dim=-1)

        # Ideal DCG (using true relevance order)
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        if self.k is not None:
            k = min(self.k, list_size)
            sorted_rel = sorted_rel[:, :k]

        ideal_gains = torch.pow(2.0, sorted_rel) - 1.0
        ideal_discounts = 1.0 / torch.log2(
            torch.arange(sorted_rel.size(1), device=device).float() + 2.0
        )

        ideal_dcg = (ideal_gains * ideal_discounts).sum(dim=-1)

        # NDCG
        ndcg = dcg / (ideal_dcg + self.eps)

        # Loss = 1 - NDCG
        loss = 1.0 - ndcg

        return loss.mean()


class RankingMSELoss(nn.Module):
    """
    Pointwise MSE loss for ranking.

    Simpler approach: directly regress to relevance scores.
    Can work well for TCP where we care about binary relevance (fail/pass).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between scores and relevance.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding

        Returns:
            MSE loss
        """
        if mask is not None:
            scores = scores[mask]
            relevance = relevance[mask]

        return self.mse(scores, relevance.float())


def create_ranking_loss(config: dict) -> nn.Module:
    """
    Create ranking loss function based on configuration.

    Args:
        config: Configuration dictionary with 'training.ranking_loss' section

    Returns:
        Loss function module
    """
    loss_config = config.get('training', {}).get('ranking_loss', {})
    loss_type = loss_config.get('type', 'listnet')

    if loss_type == 'listnet':
        return ListNetLoss(
            temperature=loss_config.get('temperature', 1.0)
        )
    elif loss_type == 'listmle':
        return ListMLELoss()
    elif loss_type == 'lambdarank':
        return LambdaRankLoss(
            sigma=loss_config.get('sigma', 1.0),
            k=loss_config.get('k', None)
        )
    elif loss_type == 'approx_ndcg':
        return ApproxNDCGLoss(
            temperature=loss_config.get('temperature', 1.0),
            k=loss_config.get('k', None)
        )
    elif loss_type == 'mse':
        return RankingMSELoss()
    else:
        raise ValueError(f"Unknown ranking loss type: {loss_type}")
