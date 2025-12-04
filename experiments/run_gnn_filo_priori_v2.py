#!/usr/bin/env python3
"""
GNN Filo-Priori V2 - Complete Implementation

This version implements the FULL Filo-Priori approach for GNN datasets:

1. MULTI-EDGE GRAPH:
   - Original edges (citation/social network) - weight 1.0
   - Semantic similarity edges (node features) - weight 0.3
   - Prediction similarity edges (same predicted class) - weight 0.5

2. STRUCTURAL FEATURES (10 features):
   - degree_centrality, in_degree, out_degree
   - clustering_coefficient
   - pagerank
   - neighbor_entropy
   - prediction confidence features

3. RANKING MODEL:
   - XGBoost/LightGBM to learn which nodes are likely misclassified
   - Features: uncertainty + structural + neighborhood

4. ORPHAN HANDLING:
   - KNN-based scoring for isolated nodes
   - Temperature-scaled softmax weighting

Reference:
- "Test Input Prioritization for Graph Neural Networks" (IEEE TSE 2024)
- Filo-Priori Technical Report (APFD 0.7595)

Author: Filo-Priori Team
Date: December 2025
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import PyTorch Geometric
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("ERROR: torch_geometric not installed")
    sys.exit(1)

# Try to import ranking libraries
try:
    from lightgbm import LGBMRanker, LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not available. Using fallback ranker.")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-EDGE GRAPH BUILDER (Key Component from Filo-Priori)
# ============================================================================

class MultiEdgeGraphBuilder:
    """
    Build a dense multi-edge graph for GNN test prioritization.

    Edge Types (adapted for GNN datasets):
    1. Original edges (citation/social) - weight 1.0
    2. Semantic similarity (node features) - weight 0.3
    3. Prediction agreement (same predicted class) - weight 0.5

    This increases graph density and enables better message passing.
    """

    def __init__(
        self,
        semantic_top_k: int = 10,
        semantic_threshold: float = 0.65,
        edge_weights: Dict[str, float] = None
    ):
        self.semantic_top_k = semantic_top_k
        self.semantic_threshold = semantic_threshold
        self.edge_weights = edge_weights or {
            'original': 1.0,
            'semantic': 0.3,
            'prediction': 0.5
        }

    def build(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        predictions: np.ndarray = None,
        node_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build multi-edge graph.

        Args:
            x: Node features [N, F]
            edge_index: Original edges [2, E]
            predictions: Predicted classes [N]
            node_mask: Mask for nodes to consider [N]

        Returns:
            combined_edge_index: [2, E_combined]
            combined_edge_weights: [E_combined]
        """
        N = x.shape[0]

        # Initialize edge dictionary
        edge_dict = defaultdict(lambda: defaultdict(float))

        # 1. Original edges (weight 1.0)
        logger.info("  Adding original edges...")
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            edge_dict[src][tgt] += self.edge_weights['original']

        original_count = edge_index.shape[1]

        # 2. Semantic similarity edges (weight 0.3)
        logger.info("  Computing semantic similarity edges...")
        semantic_count = 0

        # Normalize features for cosine similarity
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        # For each node, find top-k most similar neighbors
        for i in range(N):
            similarities = np.dot(x_norm, x_norm[i])
            similarities[i] = -1  # Exclude self

            # Get top-k indices
            top_k_idx = np.argsort(similarities)[-self.semantic_top_k:]

            for j in top_k_idx:
                if similarities[j] >= self.semantic_threshold:
                    edge_dict[i][j] += self.edge_weights['semantic'] * similarities[j]
                    semantic_count += 1

        logger.info(f"    Added {semantic_count} semantic edges")

        # 3. Prediction agreement edges (weight 0.5)
        pred_count = 0
        if predictions is not None:
            logger.info("  Computing prediction agreement edges...")
            for i in range(N):
                # Find nodes with same prediction
                same_pred = np.where(predictions == predictions[i])[0]

                # Add edges to nearest neighbors with same prediction
                if len(same_pred) > 1:
                    # Get similarities among same-prediction nodes
                    sims = np.dot(x_norm[same_pred], x_norm[i])

                    # Top-5 most similar with same prediction
                    top_idx = np.argsort(sims)[-6:-1]  # Exclude self
                    for idx in top_idx:
                        j = same_pred[idx]
                        if i != j:
                            edge_dict[i][j] += self.edge_weights['prediction']
                            pred_count += 1

            logger.info(f"    Added {pred_count} prediction agreement edges")

        # Convert to arrays
        edges_src = []
        edges_tgt = []
        weights = []

        for src, tgt_dict in edge_dict.items():
            for tgt, weight in tgt_dict.items():
                edges_src.append(src)
                edges_tgt.append(tgt)
                weights.append(weight)

        combined_edge_index = np.array([edges_src, edges_tgt], dtype=np.int64)
        combined_edge_weights = np.array(weights, dtype=np.float32)

        # Normalize weights
        max_weight = combined_edge_weights.max() if len(combined_edge_weights) > 0 else 1.0
        combined_edge_weights = combined_edge_weights / max_weight

        logger.info(f"  Multi-edge graph: {len(weights)} total edges "
                   f"(original: {original_count}, semantic: {semantic_count}, pred: {pred_count})")

        return combined_edge_index, combined_edge_weights


# ============================================================================
# STRUCTURAL FEATURE EXTRACTOR (10 features like Filo-Priori)
# ============================================================================

class GNNStructuralFeatureExtractor:
    """
    Extract structural features for GNN nodes.

    Features (10 total):
    1. degree_centrality: Normalized degree
    2. in_degree: Incoming edges (for directed graphs)
    3. out_degree: Outgoing edges
    4. clustering_coeff: Local clustering coefficient
    5. pagerank: PageRank score
    6. neighbor_degree_mean: Average degree of neighbors
    7. neighbor_degree_std: Std of neighbor degrees
    8. edge_weight_sum: Sum of edge weights (for multi-edge graph)
    9. edge_weight_mean: Mean edge weight
    10. betweenness_approx: Approximate betweenness (based on degree)
    """

    def __init__(self):
        self.feature_names = [
            'degree_centrality',
            'in_degree',
            'out_degree',
            'clustering_coeff',
            'pagerank',
            'neighbor_degree_mean',
            'neighbor_degree_std',
            'edge_weight_sum',
            'edge_weight_mean',
            'betweenness_approx'
        ]

    def extract(
        self,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Extract structural features for all nodes.

        Args:
            edge_index: [2, E]
            edge_weights: [E]
            num_nodes: N

        Returns:
            features: [N, 10]
        """
        features = np.zeros((num_nodes, 10), dtype=np.float32)

        if edge_index.size == 0:
            return features

        src, tgt = edge_index[0], edge_index[1]

        # 1-3. Degree features
        out_degree = np.zeros(num_nodes)
        in_degree = np.zeros(num_nodes)
        np.add.at(out_degree, src, 1)
        np.add.at(in_degree, tgt, 1)

        total_degree = out_degree + in_degree
        max_degree = max(total_degree.max(), 1)

        features[:, 0] = total_degree / max_degree  # degree_centrality
        features[:, 1] = in_degree / max_degree      # in_degree
        features[:, 2] = out_degree / max_degree     # out_degree

        # 4. Clustering coefficient (approximation)
        # For each node, count triangles
        neighbors = defaultdict(set)
        for s, t in zip(src, tgt):
            neighbors[s].add(t)
            neighbors[t].add(s)

        for i in range(num_nodes):
            n_neighbors = len(neighbors[i])
            if n_neighbors < 2:
                features[i, 3] = 0
            else:
                # Count edges between neighbors
                triangles = 0
                neighbor_list = list(neighbors[i])
                for j in range(len(neighbor_list)):
                    for k in range(j + 1, len(neighbor_list)):
                        if neighbor_list[k] in neighbors[neighbor_list[j]]:
                            triangles += 1
                possible = n_neighbors * (n_neighbors - 1) / 2
                features[i, 3] = triangles / possible if possible > 0 else 0

        # 5. PageRank (simple power iteration)
        features[:, 4] = self._compute_pagerank(edge_index, num_nodes)

        # 6-7. Neighbor degree statistics
        for i in range(num_nodes):
            if len(neighbors[i]) > 0:
                neighbor_degrees = [total_degree[j] for j in neighbors[i]]
                features[i, 5] = np.mean(neighbor_degrees) / max_degree
                features[i, 6] = np.std(neighbor_degrees) / max_degree if len(neighbor_degrees) > 1 else 0

        # 8-9. Edge weight statistics
        weight_sum = np.zeros(num_nodes)
        weight_count = np.zeros(num_nodes)
        np.add.at(weight_sum, src, edge_weights)
        np.add.at(weight_count, src, 1)
        np.add.at(weight_sum, tgt, edge_weights)
        np.add.at(weight_count, tgt, 1)

        weight_count = np.maximum(weight_count, 1)
        features[:, 7] = weight_sum / weight_sum.max() if weight_sum.max() > 0 else 0
        features[:, 8] = (weight_sum / weight_count) / (weight_sum / weight_count).max() if (weight_sum / weight_count).max() > 0 else 0

        # 10. Betweenness approximation (based on degree * clustering)
        features[:, 9] = features[:, 0] * (1 - features[:, 3])

        return features

    def _compute_pagerank(
        self,
        edge_index: np.ndarray,
        num_nodes: int,
        damping: float = 0.85,
        iterations: int = 20
    ) -> np.ndarray:
        """Simple PageRank computation."""
        pr = np.ones(num_nodes) / num_nodes
        src, tgt = edge_index[0], edge_index[1]

        out_degree = np.zeros(num_nodes)
        np.add.at(out_degree, src, 1)
        out_degree = np.maximum(out_degree, 1)

        for _ in range(iterations):
            new_pr = np.ones(num_nodes) * (1 - damping) / num_nodes
            contributions = pr[src] / out_degree[src]
            np.add.at(new_pr, tgt, damping * contributions)
            pr = new_pr

        return pr / pr.max() if pr.max() > 0 else pr


# ============================================================================
# UNCERTAINTY FEATURE EXTRACTOR (for ranking)
# ============================================================================

class UncertaintyFeatureExtractor:
    """
    Extract uncertainty-based features for ranking.

    Features (10 total):
    1. max_softmax: Confidence
    2. entropy: Prediction entropy
    3. margin: Top-2 margin
    4. gini: Gini impurity
    5. least_confidence: 1 - max_softmax
    6. second_prob: Second highest probability
    7. top2_ratio: Ratio of top-2 probs
    8. variance: Prediction variance
    9. kl_uniform: KL divergence from uniform
    10. std_prob: Std of probabilities
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.feature_names = [
            'max_softmax', 'entropy', 'margin', 'gini', 'least_confidence',
            'second_prob', 'top2_ratio', 'variance', 'kl_uniform', 'std_prob'
        ]

    def extract(self, probs: np.ndarray) -> np.ndarray:
        """
        Extract uncertainty features.

        Args:
            probs: [N, C] prediction probabilities

        Returns:
            features: [N, 10]
        """
        N = probs.shape[0]
        features = np.zeros((N, 10), dtype=np.float32)

        # Ensure valid probabilities
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Sort probabilities
        sorted_probs = np.sort(probs, axis=1)

        # 1. Max softmax (confidence)
        features[:, 0] = probs.max(axis=1)

        # 2. Entropy (normalized)
        ent = scipy_entropy(probs, axis=1, base=2)
        max_ent = np.log2(self.num_classes)
        features[:, 1] = ent / max_ent

        # 3. Margin
        features[:, 2] = sorted_probs[:, -1] - sorted_probs[:, -2]

        # 4. Gini
        features[:, 3] = 1.0 - (probs ** 2).sum(axis=1)

        # 5. Least confidence
        features[:, 4] = 1.0 - features[:, 0]

        # 6. Second probability
        features[:, 5] = sorted_probs[:, -2]

        # 7. Top-2 ratio
        features[:, 6] = sorted_probs[:, -1] / (sorted_probs[:, -2] + 1e-8)
        features[:, 6] = np.clip(features[:, 6], 0, 10) / 10

        # 8. Variance
        features[:, 7] = probs.var(axis=1)

        # 9. KL from uniform
        uniform = np.ones(self.num_classes) / self.num_classes
        features[:, 8] = np.array([scipy_entropy(p, uniform) for p in probs])
        features[:, 8] = features[:, 8] / features[:, 8].max() if features[:, 8].max() > 0 else 0

        # 10. Std of probabilities
        features[:, 9] = probs.std(axis=1)

        return features


# ============================================================================
# NEIGHBORHOOD FEATURE EXTRACTOR
# ============================================================================

class NeighborhoodFeatureExtractor:
    """
    Extract neighborhood-based features.

    Features (5 total):
    1. neighbor_entropy_mean: Mean entropy of neighbors
    2. neighbor_confidence_mean: Mean confidence of neighbors
    3. neighbor_agreement: Fraction with same prediction
    4. neighbor_disagreement: Fraction with different prediction
    5. neighbor_homophily: Fraction with same ground truth (if available)
    """

    def __init__(self):
        self.feature_names = [
            'neighbor_entropy_mean', 'neighbor_confidence_mean',
            'neighbor_agreement', 'neighbor_disagreement', 'neighbor_homophily'
        ]

    def extract(
        self,
        probs: np.ndarray,
        edge_index: np.ndarray,
        num_nodes: int,
        labels: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract neighborhood features.

        Args:
            probs: [N, C]
            edge_index: [2, E]
            num_nodes: N
            labels: [N] ground truth (optional)

        Returns:
            features: [N, 5]
        """
        features = np.zeros((num_nodes, 5), dtype=np.float32)

        if edge_index.size == 0:
            return features

        predictions = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        entropy = scipy_entropy(probs, axis=1, base=2)
        max_ent = np.log2(probs.shape[1])
        entropy = entropy / max_ent

        # Build neighbor lists
        neighbors = defaultdict(list)
        src, tgt = edge_index[0], edge_index[1]
        for s, t in zip(src, tgt):
            neighbors[s].append(t)
            neighbors[t].append(s)

        for i in range(num_nodes):
            if len(neighbors[i]) == 0:
                continue

            neighbor_idx = neighbors[i]

            # 1. Mean entropy of neighbors
            features[i, 0] = np.mean(entropy[neighbor_idx])

            # 2. Mean confidence of neighbors
            features[i, 1] = np.mean(confidence[neighbor_idx])

            # 3. Agreement (same prediction)
            same_pred = np.sum(predictions[neighbor_idx] == predictions[i])
            features[i, 2] = same_pred / len(neighbor_idx)

            # 4. Disagreement
            features[i, 3] = 1.0 - features[i, 2]

            # 5. Homophily (if labels available)
            if labels is not None:
                same_label = np.sum(labels[neighbor_idx] == labels[i])
                features[i, 4] = same_label / len(neighbor_idx)

        return features


# ============================================================================
# RANKING MODEL (Learn to rank misclassified samples)
# ============================================================================

class MisclassificationRanker:
    """
    Learn to rank nodes by likelihood of misclassification.

    Uses gradient boosting to predict which nodes are misclassified
    based on uncertainty, structural, and neighborhood features.
    """

    def __init__(self, use_lightgbm: bool = True):
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.scaler = StandardScaler()
        self.model = None

    def fit(
        self,
        features: np.ndarray,
        is_misclassified: np.ndarray
    ):
        """
        Train the ranking model.

        Args:
            features: [N, F] combined features
            is_misclassified: [N] binary labels (1 = misclassified)
        """
        # Scale features
        X = self.scaler.fit_transform(features)
        y = is_misclassified.astype(int)

        if self.use_lightgbm:
            self.model = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        elif HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X, y)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Predict misclassification probability scores.

        Args:
            features: [N, F]

        Returns:
            scores: [N] probability of being misclassified
        """
        X = self.scaler.transform(features)
        return self.model.predict_proba(X)[:, 1]


# ============================================================================
# ORPHAN HANDLING (KNN-based scoring)
# ============================================================================

def compute_orphan_scores(
    orphan_features: np.ndarray,
    in_graph_features: np.ndarray,
    in_graph_scores: np.ndarray,
    k_neighbors: int = 20,
    temperature: float = 0.7,
    alpha_blend: float = 0.55
) -> np.ndarray:
    """
    Compute scores for orphan nodes using KNN.

    Args:
        orphan_features: [N_orphan, F]
        in_graph_features: [N_ingraph, F]
        in_graph_scores: [N_ingraph]
        k_neighbors: Number of neighbors
        temperature: Softmax temperature
        alpha_blend: Blend factor

    Returns:
        scores: [N_orphan]
    """
    if len(in_graph_features) == 0 or len(orphan_features) == 0:
        return np.ones(len(orphan_features)) * 0.5

    # Compute distances
    distances = cdist(orphan_features, in_graph_features, metric='euclidean')
    similarities = np.exp(-distances)

    scores = np.zeros(len(orphan_features))

    for i in range(len(orphan_features)):
        # Get top-k neighbors
        top_k_idx = np.argsort(similarities[i])[-k_neighbors:]
        top_k_sims = similarities[i, top_k_idx]

        # Temperature-scaled softmax
        scaled = top_k_sims / temperature
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        # KNN score
        knn_score = np.dot(weights, in_graph_scores[top_k_idx])

        # Blend with base score
        scores[i] = alpha_blend * knn_score + (1 - alpha_blend) * 0.5

    return scores


# ============================================================================
# GNN MODEL (Simple GCN for getting embeddings)
# ============================================================================

class SimpleGCN(nn.Module):
    """Simple GCN for getting node embeddings."""

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================================
# APFD CALCULATION
# ============================================================================

def calculate_apfd(ranking: np.ndarray, is_misclassified: np.ndarray) -> float:
    """Calculate APFD score."""
    n = len(ranking)
    m = is_misclassified.sum()

    if m == 0:
        return 1.0

    fault_positions = []
    for pos, idx in enumerate(ranking, start=1):
        if is_misclassified[idx]:
            fault_positions.append(pos)

    sum_positions = sum(fault_positions)
    return 1 - (sum_positions / (n * m)) + (1 / (2 * n))


def calculate_pfd_at_k(ranking: np.ndarray, is_misclassified: np.ndarray, k_percent: float) -> float:
    """Calculate PFD@k."""
    n = len(ranking)
    m = is_misclassified.sum()

    if m == 0:
        return 1.0

    k = max(1, int(n * k_percent / 100))
    top_k = ranking[:k]
    return is_misclassified[top_k].sum() / m


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def load_dataset(name: str) -> Tuple[Data, int]:
    """Load GNN benchmark dataset."""
    data_dir = project_root / 'datasets' / '03_gnn_benchmarks' / 'raw'
    name_map = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}
    dataset = Planetoid(root=str(data_dir / 'planetoid'), name=name_map[name.lower()])
    return dataset[0], dataset.num_classes


def run_single_experiment(
    dataset_name: str,
    config: Dict,
    device: torch.device,
    run_id: int = 0
) -> Dict:
    """Run a single experiment with full Filo-Priori pipeline."""

    seed = config.get('seed', 42) + run_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data, num_classes = load_dataset(dataset_name)
    logger.info(f"\nDataset: {dataset_name.upper()}")
    logger.info(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Classes: {num_classes}")

    # =========================================================================
    # STEP 1: Train base GNN model
    # =========================================================================
    logger.info("\nSTEP 1: Training base GNN model...")

    model = SimpleGCN(data.num_node_features, num_classes, hidden_dim=256).to(device)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    best_val_acc = 0
    best_state = None

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            val_acc = (pred[data.val_mask] == y[data.val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        embeddings = model.get_embeddings(x, edge_index).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()

    # Get test results
    test_mask = data.test_mask.numpy()
    test_indices = np.where(test_mask)[0]
    test_probs = probs[test_indices]
    test_labels = data.y.numpy()[test_indices]
    test_preds = predictions[test_indices]
    is_misclassified = (test_preds != test_labels)

    test_acc = (test_preds == test_labels).mean()
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Misclassified: {is_misclassified.sum()}/{len(test_labels)} ({100*is_misclassified.mean():.1f}%)")

    # =========================================================================
    # STEP 2: Build multi-edge graph
    # =========================================================================
    logger.info("\nSTEP 2: Building multi-edge graph...")

    graph_builder = MultiEdgeGraphBuilder(
        semantic_top_k=config.get('semantic_top_k', 10),
        semantic_threshold=config.get('semantic_threshold', 0.65)
    )

    multi_edge_index, multi_edge_weights = graph_builder.build(
        x=data.x.numpy(),
        edge_index=data.edge_index.numpy(),
        predictions=predictions
    )

    # =========================================================================
    # STEP 3: Extract features
    # =========================================================================
    logger.info("\nSTEP 3: Extracting features...")

    # Structural features
    struct_extractor = GNNStructuralFeatureExtractor()
    structural_features = struct_extractor.extract(
        multi_edge_index, multi_edge_weights, data.num_nodes
    )
    logger.info(f"  Structural features: {structural_features.shape}")

    # Uncertainty features
    uncertainty_extractor = UncertaintyFeatureExtractor(num_classes)
    uncertainty_features = uncertainty_extractor.extract(probs)
    logger.info(f"  Uncertainty features: {uncertainty_features.shape}")

    # Neighborhood features
    neighborhood_extractor = NeighborhoodFeatureExtractor()
    neighborhood_features = neighborhood_extractor.extract(
        probs, multi_edge_index, data.num_nodes, data.y.numpy()
    )
    logger.info(f"  Neighborhood features: {neighborhood_features.shape}")

    # Combine all features
    all_features = np.concatenate([
        structural_features,
        uncertainty_features,
        neighborhood_features
    ], axis=1)
    logger.info(f"  Combined features: {all_features.shape}")

    # =========================================================================
    # STEP 4: Train ranking model on TRAIN+VAL nodes
    # =========================================================================
    logger.info("\nSTEP 4: Training ranking model...")

    train_val_mask = data.train_mask.numpy() | data.val_mask.numpy()
    train_val_indices = np.where(train_val_mask)[0]

    # Get train+val labels for is_misclassified
    train_val_preds = predictions[train_val_indices]
    train_val_labels = data.y.numpy()[train_val_indices]
    train_val_misclassified = (train_val_preds != train_val_labels)

    ranker = MisclassificationRanker(use_lightgbm=HAS_LIGHTGBM)
    ranker.fit(
        features=all_features[train_val_indices],
        is_misclassified=train_val_misclassified
    )

    # =========================================================================
    # STEP 5: Score test nodes
    # =========================================================================
    logger.info("\nSTEP 5: Scoring test nodes...")

    test_features = all_features[test_indices]

    # Filo-Priori: Use learned ranker
    filo_scores = ranker.predict_scores(test_features)

    # Baseline methods
    results = {
        'dataset': dataset_name,
        'run_id': run_id,
        'test_accuracy': test_acc,
        'num_test': len(test_labels),
        'num_misclassified': int(is_misclassified.sum())
    }

    methods = {
        'Random': np.random.rand(len(test_indices)),
        'DeepGini': 1.0 - (test_probs ** 2).sum(axis=1),
        'Entropy': scipy_entropy(test_probs, axis=1, base=2),
        'VanillaSM': 1.0 - (np.sort(test_probs, axis=1)[:, -1] - np.sort(test_probs, axis=1)[:, -2]),
        'PCS': 1.0 - test_probs.max(axis=1),
        'Filo-Priori': filo_scores
    }

    for method_name, scores in methods.items():
        if method_name == 'Random':
            # Average over 10 runs
            apfd_list, pfd10_list, pfd20_list = [], [], []
            for _ in range(10):
                ranking = np.argsort(-np.random.rand(len(test_indices)))
                apfd_list.append(calculate_apfd(ranking, is_misclassified))
                pfd10_list.append(calculate_pfd_at_k(ranking, is_misclassified, 10))
                pfd20_list.append(calculate_pfd_at_k(ranking, is_misclassified, 20))
            apfd, pfd10, pfd20 = np.mean(apfd_list), np.mean(pfd10_list), np.mean(pfd20_list)
        else:
            ranking = np.argsort(-scores)
            apfd = calculate_apfd(ranking, is_misclassified)
            pfd10 = calculate_pfd_at_k(ranking, is_misclassified, 10)
            pfd20 = calculate_pfd_at_k(ranking, is_misclassified, 20)

        results[f'{method_name}_APFD'] = apfd
        results[f'{method_name}_PFD@10'] = pfd10
        results[f'{method_name}_PFD@20'] = pfd20

        logger.info(f"  {method_name}: APFD={apfd:.4f}, PFD@10={pfd10:.4f}, PFD@20={pfd20:.4f}")

    return results


def run_full_experiment(
    datasets: List[str],
    config: Dict,
    n_runs: int = 5,
    output_dir: str = None
) -> pd.DataFrame:
    """Run full experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if output_dir is None:
        output_dir = project_root / 'experiments' / 'results' / 'gnn_filo_priori_v2'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*70}")

        for run_id in range(n_runs):
            logger.info(f"\n--- Run {run_id + 1}/{n_runs} ---")
            results = run_single_experiment(dataset, config, device, run_id)
            all_results.append(results)

    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'full_results_{timestamp}.csv', index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY: GNN Filo-Priori V2 (Full Pipeline)")
    print("=" * 80)

    methods = ['Random', 'DeepGini', 'Entropy', 'VanillaSM', 'PCS', 'Filo-Priori']

    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}")
        print("-" * 60)
        print(f"{'Method':<15} {'APFD':>15} {'PFD@10':>15} {'PFD@20':>15}")
        print("-" * 60)

        dataset_df = df[df['dataset'] == dataset]
        for method in methods:
            col = f'{method}_APFD'
            if col in dataset_df.columns:
                mean = dataset_df[col].mean()
                std = dataset_df[col].std()
                pfd10_mean = dataset_df[f'{method}_PFD@10'].mean()
                pfd10_std = dataset_df[f'{method}_PFD@10'].std()
                pfd20_mean = dataset_df[f'{method}_PFD@20'].mean()
                pfd20_std = dataset_df[f'{method}_PFD@20'].std()
                print(f"{method:<15} {mean:.4f}±{std:.4f}  {pfd10_mean:.4f}±{pfd10_std:.4f}  {pfd20_mean:.4f}±{pfd20_std:.4f}")

    print("\n" + "=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")

    return df


def main():
    parser = argparse.ArgumentParser(description='GNN Filo-Priori V2 Experiment')
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = {
        'seed': 42,
        'semantic_top_k': 10,
        'semantic_threshold': 0.65
    }

    logger.info("=" * 70)
    logger.info("GNN FILO-PRIORI V2 - FULL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Runs: {args.n_runs}")
    logger.info("=" * 70)

    run_full_experiment(
        datasets=args.datasets,
        config=config,
        n_runs=args.n_runs,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
