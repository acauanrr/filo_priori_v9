"""
NDCG Utility Functions for Filo-Priori V10.

This module provides utilities for computing NDCG (Normalized Discounted
Cumulative Gain) and related metrics.

NDCG is used:
1. As an evaluation metric
2. Within LambdaRank to compute gradient weights

Formulas:
    DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)
    IDCG@k = DCG@k with optimal ranking
    NDCG@k = DCG@k / IDCG@k
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_dcg(
    relevances: torch.Tensor,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Discounted Cumulative Gain.

    Args:
        relevances: Relevance scores in current ranking order [batch, num_items].
        k: Cutoff position (None = use all).

    Returns:
        DCG values [batch].

    Formula:
        DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)
    """
    if k is not None:
        relevances = relevances[:, :k]

    # Gains: 2^rel - 1
    gains = torch.pow(2.0, relevances) - 1.0

    # Discounts: 1 / log_2(position + 1)
    positions = torch.arange(1, relevances.shape[1] + 1, device=relevances.device).float()
    discounts = torch.log2(positions + 1.0)

    # DCG
    dcg = (gains / discounts).sum(dim=1)

    return dcg


def compute_ideal_dcg(
    relevances: torch.Tensor,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Ideal DCG (DCG with optimal ranking).

    Args:
        relevances: Relevance scores (any order) [batch, num_items].
        k: Cutoff position.

    Returns:
        IDCG values [batch].
    """
    # Sort by relevance (descending)
    sorted_relevances, _ = torch.sort(relevances, dim=1, descending=True)

    return compute_dcg(sorted_relevances, k=k)


def compute_ndcg(
    relevances: torch.Tensor,
    k: Optional[int] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Normalized DCG.

    Args:
        relevances: Relevance scores in current ranking [batch, num_items].
        k: Cutoff position.
        eps: Small constant to avoid division by zero.

    Returns:
        NDCG values [batch] in range [0, 1].
    """
    dcg = compute_dcg(relevances, k=k)
    idcg = compute_ideal_dcg(relevances, k=k)

    ndcg = dcg / (idcg + eps)

    return ndcg


def compute_delta_ndcg(
    relevances: torch.Tensor,
    pos_i: int,
    pos_j: int,
    k: Optional[int] = None
) -> torch.Tensor:
    """
    Compute the change in NDCG if positions i and j are swapped.

    This is used in LambdaRank to weight the gradient.

    Args:
        relevances: Relevance scores [num_items].
        pos_i: First position (0-indexed).
        pos_j: Second position (0-indexed).
        k: Cutoff position.

    Returns:
        Absolute change in NDCG (scalar).
    """
    n = len(relevances)

    # Skip if beyond cutoff
    if k is not None and min(pos_i, pos_j) >= k:
        return torch.tensor(0.0, device=relevances.device)

    # Gains
    gain_i = (2.0 ** relevances[pos_i]) - 1.0
    gain_j = (2.0 ** relevances[pos_j]) - 1.0

    # Discounts (1-indexed positions)
    disc_i = 1.0 / math.log2(pos_i + 2)  # +2 because 0-indexed
    disc_j = 1.0 / math.log2(pos_j + 2)

    # IDCG
    idcg = compute_ideal_dcg(relevances.unsqueeze(0), k=k)[0]

    # Delta DCG from swap
    # Current: gain_i * disc_i + gain_j * disc_j
    # After swap: gain_i * disc_j + gain_j * disc_i
    # Delta: (gain_i - gain_j) * (disc_j - disc_i)
    delta_dcg = (gain_i - gain_j) * (disc_j - disc_i)

    # Normalize by IDCG
    delta_ndcg = torch.abs(delta_dcg) / (idcg + 1e-8)

    return delta_ndcg


def compute_apfd(
    rankings: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute Average Percentage of Faults Detected.

    APFD = 1 - (Σ TF_i) / (n × m) + 1/(2n)

    Where:
        TF_i = position of failure i (1-indexed)
        n = total number of tests
        m = number of failing tests

    Args:
        rankings: Predicted rankings [batch, num_tests] (lower = higher priority).
        labels: Binary labels [batch, num_tests] (1 = fail).

    Returns:
        APFD values [batch] in range [0, 1].
    """
    batch_size, n = rankings.shape

    apfd_values = []

    for b in range(batch_size):
        rank = rankings[b]
        label = labels[b]

        # Get positions of failures (1-indexed)
        fail_mask = label == 1
        m = fail_mask.sum().item()

        if m == 0:
            # No failures - undefined, return 1.0 (best case)
            apfd_values.append(1.0)
            continue

        # Sum of failure positions
        # rank gives the order, but we need positions
        # If rank[i] = k, test i is at position k+1
        fail_positions = rank[fail_mask] + 1  # Convert to 1-indexed
        sum_tf = fail_positions.sum().item()

        # APFD formula
        apfd = 1 - (sum_tf / (n * m)) + (1 / (2 * n))
        apfd_values.append(apfd)

    return torch.tensor(apfd_values, device=rankings.device)


def ranks_from_scores(
    scores: torch.Tensor
) -> torch.Tensor:
    """
    Convert scores to rankings (lower rank = higher priority).

    Args:
        scores: Prediction scores [batch, num_items] (higher = more relevant).

    Returns:
        Rankings [batch, num_items] where 0 = highest priority.
    """
    # Sort descending to get indices
    _, indices = torch.sort(scores, dim=1, descending=True)

    # Create rankings
    rankings = torch.zeros_like(scores, dtype=torch.long)
    batch_size = scores.shape[0]

    for b in range(batch_size):
        rankings[b, indices[b]] = torch.arange(scores.shape[1], device=scores.device)

    return rankings


def compute_precision_at_k(
    scores: torch.Tensor,
    relevances: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute Precision@k.

    Args:
        scores: Prediction scores [batch, num_items].
        relevances: Binary relevances [batch, num_items].
        k: Cutoff position.

    Returns:
        Precision@k values [batch].
    """
    # Get top-k predictions
    _, top_k_indices = torch.topk(scores, k, dim=1)

    # Check relevance of top-k
    batch_size = scores.shape[0]
    precisions = []

    for b in range(batch_size):
        relevant_in_top_k = relevances[b, top_k_indices[b]].sum().item()
        precision = relevant_in_top_k / k
        precisions.append(precision)

    return torch.tensor(precisions, device=scores.device)


def compute_recall_at_k(
    scores: torch.Tensor,
    relevances: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute Recall@k.

    Args:
        scores: Prediction scores [batch, num_items].
        relevances: Binary relevances [batch, num_items].
        k: Cutoff position.

    Returns:
        Recall@k values [batch].
    """
    _, top_k_indices = torch.topk(scores, k, dim=1)

    batch_size = scores.shape[0]
    recalls = []

    for b in range(batch_size):
        total_relevant = relevances[b].sum().item()
        if total_relevant == 0:
            recalls.append(1.0)  # No relevant items = perfect recall
            continue

        relevant_in_top_k = relevances[b, top_k_indices[b]].sum().item()
        recall = relevant_in_top_k / total_relevant
        recalls.append(recall)

    return torch.tensor(recalls, device=scores.device)


def compute_mrr(
    scores: torch.Tensor,
    relevances: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mean Reciprocal Rank.

    MRR = 1/|Q| × Σ 1/rank_i

    Where rank_i is the position of the first relevant item.

    Args:
        scores: Prediction scores [batch, num_items].
        relevances: Binary relevances [batch, num_items].

    Returns:
        MRR value (scalar).
    """
    # Sort by scores descending
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)

    batch_size = scores.shape[0]
    reciprocal_ranks = []

    for b in range(batch_size):
        # Get relevances in sorted order
        sorted_relevances = relevances[b, sorted_indices[b]]

        # Find first relevant item
        first_relevant = (sorted_relevances == 1).nonzero(as_tuple=True)[0]

        if len(first_relevant) == 0:
            reciprocal_ranks.append(0.0)
        else:
            rank = first_relevant[0].item() + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)

    return torch.tensor(reciprocal_ranks, device=scores.device).mean()
