"""
GNN Uncertainty Feature Extractor

Extracts uncertainty-based features for test prioritization in GNN classification.
These features replace the "test execution history" features used in the original
Filo-Priori (failure_rate, recent_failure_rate, etc.).

Features extracted:
1. max_softmax: Maximum probability (confidence)
2. entropy: Prediction entropy (uncertainty)
3. margin: Difference between top-2 predictions
4. gini: Gini impurity of prediction distribution
5. degree_centrality: Node importance in graph
6. neighbor_entropy: Average entropy of neighbors

Reference:
- DeepGini uses Gini impurity for prioritization
- Entropy is a standard uncertainty measure
- Margin is used in VanillaSM method

Author: Filo-Priori Team
Date: December 2025
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.stats import entropy as scipy_entropy
import logging

logger = logging.getLogger(__name__)


class GNNUncertaintyExtractor:
    """
    Extract uncertainty features from GNN predictions.

    These features serve as the "structural features" for the Filo-Priori
    model adapted for GNN datasets.
    """

    # Feature names (10 features total, matching V2.5 extractor)
    FEATURE_NAMES = [
        'max_softmax',           # Confidence (higher = more certain)
        'entropy',               # Uncertainty (higher = less certain)
        'margin',                # Top-2 margin (higher = more certain)
        'gini',                  # Gini impurity (higher = less certain)
        'least_confidence',      # 1 - max_softmax
        'degree_centrality',     # Node importance in graph
        'neighbor_entropy',      # Average entropy of neighbors
        'neighbor_agreement',    # Agreement with neighbors
        'prediction_variance',   # Variance of prediction
        'top2_ratio',            # Ratio of top-2 probabilities
    ]

    def __init__(
        self,
        num_classes: int,
        normalize: bool = True,
        epsilon: float = 1e-8
    ):
        """
        Initialize the extractor.

        Args:
            num_classes: Number of classes in the classification task
            normalize: Whether to normalize features to [0, 1]
            epsilon: Small constant for numerical stability
        """
        self.num_classes = num_classes
        self.normalize = normalize
        self.epsilon = epsilon

    def extract_features(
        self,
        probs: np.ndarray,
        edge_index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Extract uncertainty features from prediction probabilities.

        Args:
            probs: Prediction probabilities [N, C]
            edge_index: Graph edges [2, E]
            num_nodes: Total number of nodes

        Returns:
            features: Uncertainty features [N, 10]
        """
        N = probs.shape[0]
        features = np.zeros((N, len(self.FEATURE_NAMES)), dtype=np.float32)

        # Ensure probs are valid probabilities
        probs = np.clip(probs, self.epsilon, 1.0 - self.epsilon)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # 1. Max Softmax (confidence)
        features[:, 0] = probs.max(axis=1)

        # 2. Entropy (uncertainty)
        features[:, 1] = scipy_entropy(probs, axis=1, base=2)
        # Normalize entropy to [0, 1] by dividing by max entropy (log2(num_classes))
        max_entropy = np.log2(self.num_classes)
        features[:, 1] = features[:, 1] / max_entropy

        # 3. Margin (top-2 difference)
        sorted_probs = np.sort(probs, axis=1)
        features[:, 2] = sorted_probs[:, -1] - sorted_probs[:, -2]

        # 4. Gini impurity: 1 - sum(p^2)
        features[:, 3] = 1.0 - (probs ** 2).sum(axis=1)

        # 5. Least confidence: 1 - max_softmax
        features[:, 4] = 1.0 - features[:, 0]

        # 6. Degree centrality
        degree = self._compute_degree_centrality(edge_index, num_nodes)
        features[:, 5] = degree

        # 7. Neighbor entropy (average entropy of neighbors)
        neighbor_entropy = self._compute_neighbor_entropy(
            features[:, 1],  # Use normalized entropy
            edge_index,
            num_nodes
        )
        features[:, 6] = neighbor_entropy

        # 8. Neighbor agreement (fraction of neighbors with same prediction)
        predictions = probs.argmax(axis=1)
        neighbor_agreement = self._compute_neighbor_agreement(
            predictions,
            edge_index,
            num_nodes
        )
        features[:, 7] = neighbor_agreement

        # 9. Prediction variance
        features[:, 8] = probs.var(axis=1)

        # 10. Top-2 ratio
        features[:, 9] = sorted_probs[:, -1] / (sorted_probs[:, -2] + self.epsilon)
        # Clip to reasonable range
        features[:, 9] = np.clip(features[:, 9], 0, 10) / 10

        # Normalize features if requested
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def _compute_degree_centrality(
        self,
        edge_index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """Compute normalized degree centrality."""
        degree = np.zeros(num_nodes, dtype=np.float32)

        if edge_index.size > 0:
            src = edge_index[0]
            np.add.at(degree, src, 1)

            # For undirected graphs, count target as well
            tgt = edge_index[1]
            np.add.at(degree, tgt, 1)

        # Normalize by max possible degree
        max_degree = max(degree.max(), 1)
        degree = degree / max_degree

        return degree

    def _compute_neighbor_entropy(
        self,
        entropy: np.ndarray,
        edge_index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """Compute average entropy of neighbors."""
        neighbor_entropy_sum = np.zeros(num_nodes, dtype=np.float32)
        neighbor_count = np.zeros(num_nodes, dtype=np.float32)

        if edge_index.size > 0:
            src, tgt = edge_index[0], edge_index[1]

            # For each edge (src, tgt), add target's entropy to source
            np.add.at(neighbor_entropy_sum, src, entropy[tgt])
            np.add.at(neighbor_count, src, 1)

            # For undirected graphs
            np.add.at(neighbor_entropy_sum, tgt, entropy[src])
            np.add.at(neighbor_count, tgt, 1)

        # Avoid division by zero
        neighbor_count = np.maximum(neighbor_count, 1)
        neighbor_entropy = neighbor_entropy_sum / neighbor_count

        return neighbor_entropy

    def _compute_neighbor_agreement(
        self,
        predictions: np.ndarray,
        edge_index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """Compute fraction of neighbors with same prediction."""
        agreement_sum = np.zeros(num_nodes, dtype=np.float32)
        neighbor_count = np.zeros(num_nodes, dtype=np.float32)

        if edge_index.size > 0:
            src, tgt = edge_index[0], edge_index[1]

            # Check if source and target have same prediction
            same_pred = (predictions[src] == predictions[tgt]).astype(np.float32)

            np.add.at(agreement_sum, src, same_pred)
            np.add.at(neighbor_count, src, 1)

            # For undirected graphs
            np.add.at(agreement_sum, tgt, same_pred)
            np.add.at(neighbor_count, tgt, 1)

        # Avoid division by zero
        neighbor_count = np.maximum(neighbor_count, 1)
        agreement = agreement_sum / neighbor_count

        return agreement

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + self.epsilon
        return (features - mean) / std

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.FEATURE_NAMES.copy()


def extract_uncertainty_features(
    probs: np.ndarray,
    edge_index: np.ndarray,
    num_classes: int,
    normalize: bool = True
) -> Tuple[np.ndarray, list]:
    """
    Convenience function to extract uncertainty features.

    Args:
        probs: Prediction probabilities [N, C]
        edge_index: Graph edges [2, E]
        num_classes: Number of classes

    Returns:
        features: [N, 10] uncertainty features
        feature_names: List of feature names
    """
    extractor = GNNUncertaintyExtractor(
        num_classes=num_classes,
        normalize=normalize
    )

    num_nodes = probs.shape[0]
    features = extractor.extract_features(probs, edge_index, num_nodes)

    return features, extractor.get_feature_names()


def compute_prioritization_score(
    probs: np.ndarray,
    edge_index: np.ndarray,
    num_classes: int,
    method: str = 'filo_priori',
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3
) -> np.ndarray:
    """
    Compute prioritization scores using different methods.

    Args:
        probs: Prediction probabilities [N, C]
        edge_index: Graph edges [2, E]
        num_classes: Number of classes
        method: Prioritization method
        alpha, beta, gamma: Weights for Filo-Priori combination

    Returns:
        scores: Prioritization scores [N] (higher = prioritize first)
    """
    N = probs.shape[0]

    if method == 'random':
        return np.random.rand(N)

    elif method == 'deepgini':
        # Gini impurity: 1 - sum(p^2)
        return 1.0 - (probs ** 2).sum(axis=1)

    elif method == 'entropy':
        # Normalized entropy
        ent = scipy_entropy(probs, axis=1, base=2)
        max_ent = np.log2(num_classes)
        return ent / max_ent

    elif method == 'margin':
        # Negative margin (smaller margin = higher priority)
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1.0 - margin

    elif method == 'least_confidence':
        return 1.0 - probs.max(axis=1)

    elif method == 'filo_priori':
        # Filo-Priori: Combine multiple signals
        # 1. Uncertainty (entropy)
        ent = scipy_entropy(probs, axis=1, base=2)
        max_ent = np.log2(num_classes)
        uncertainty = ent / max_ent

        # 2. Structural importance (degree centrality)
        extractor = GNNUncertaintyExtractor(num_classes, normalize=False)
        degree = extractor._compute_degree_centrality(edge_index, N)

        # 3. Neighbor disagreement
        predictions = probs.argmax(axis=1)
        agreement = extractor._compute_neighbor_agreement(predictions, edge_index, N)
        disagreement = 1.0 - agreement

        # Combine: higher score = prioritize first
        score = alpha * uncertainty + beta * degree + gamma * disagreement

        return score

    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = [
    'GNNUncertaintyExtractor',
    'extract_uncertainty_features',
    'compute_prioritization_score'
]
