#!/usr/bin/env python3
"""
GNN Filo-Priori Experiment - Full Training Pipeline

This script implements the complete Filo-Priori pipeline adapted for GNN benchmark
datasets (Cora, CiteSeer, PubMed), based on the architecture that achieved
APFD 0.7595 on the industry dataset.

Key Components:
1. GNN Filo-Priori Model: Dual-stream architecture with GAT + Cross-Attention
2. Focal Loss: Handles class imbalance
3. Balanced Sampling: Addresses rare failure cases
4. Uncertainty Features: For prioritization scoring
5. APFD Evaluation: Standard metric from NodeRank paper

Reference:
- "Test Input Prioritization for Graph Neural Networks" (IEEE TSE 2024)
- Filo-Priori V9 Technical Report (APFD 0.7595)

Usage:
    python experiments/run_gnn_filo_priori.py
    python experiments/run_gnn_filo_priori.py --datasets cora citeseer
    python experiments/run_gnn_filo_priori.py --n_runs 5

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
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import PyTorch Geometric
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("ERROR: torch_geometric not installed. Run: pip install torch-geometric")
    sys.exit(1)

# Import Filo-Priori components
from models.gnn_filo_priori import GNNFiloPriori, create_gnn_filo_priori
from evaluation.gnn_uncertainty_features import (
    GNNUncertaintyExtractor,
    extract_uncertainty_features,
    compute_prioritization_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FOCAL LOSS (from Filo-Priori V9)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor (0.5 = balanced)
        gamma: Focusing parameter (2.0 standard)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [N, C]
            targets: Ground truth labels [N]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(name: str, data_dir: str = None) -> Data:
    """
    Load a GNN benchmark dataset.

    Args:
        name: Dataset name ('cora', 'citeseer', 'pubmed')
        data_dir: Directory to store data

    Returns:
        PyG Data object
    """
    if data_dir is None:
        data_dir = project_root / 'datasets' / '03_gnn_benchmarks' / 'raw'

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        'cora': 'Cora',
        'citeseer': 'CiteSeer',
        'pubmed': 'PubMed'
    }

    if name.lower() not in name_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(name_map.keys())}")

    logger.info(f"Loading {name_map[name.lower()]} dataset...")
    dataset = Planetoid(root=str(data_dir / 'planetoid'), name=name_map[name.lower()])
    data = dataset[0]

    logger.info(f"  Nodes: {data.num_nodes}")
    logger.info(f"  Edges: {data.num_edges}")
    logger.info(f"  Features: {data.num_node_features}")
    logger.info(f"  Classes: {dataset.num_classes}")
    logger.info(f"  Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

    return data, dataset.num_classes


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()

    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    optimizer.zero_grad()

    # Forward pass on full graph, mask for loss
    logits = model(x, edge_index)
    loss = criterion(logits[train_mask], y[train_mask])

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    device: torch.device,
    mask: torch.Tensor
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a split.

    Returns:
        accuracy: Classification accuracy
        probs: Prediction probabilities [N_masked, C]
        labels: Ground truth labels [N_masked]
    """
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    mask = mask.to(device)

    logits = model(x, edge_index)
    probs = F.softmax(logits, dim=1)

    # Masked predictions
    preds = logits[mask].argmax(dim=1)
    labels = y[mask]
    correct = (preds == labels).sum().item()
    accuracy = correct / mask.sum().item()

    return accuracy, probs.cpu().numpy(), y.cpu().numpy()


# ============================================================================
# APFD CALCULATION
# ============================================================================

def calculate_apfd(ranking: np.ndarray, is_misclassified: np.ndarray) -> float:
    """
    Calculate Average Percentage of Fault Detection (APFD).

    APFD = 1 - (sum of positions of faults) / (n * m) + 1 / (2 * n)

    where:
        n = total number of test cases
        m = number of faults (misclassifications)

    Args:
        ranking: Indices in priority order (first = highest priority)
        is_misclassified: Boolean array [N] indicating misclassifications

    Returns:
        APFD score in [0, 1], higher is better
    """
    n = len(ranking)
    m = is_misclassified.sum()

    if m == 0:
        return 1.0  # No faults, perfect score

    # Get positions of faults in the ranking (1-indexed)
    fault_positions = []
    for pos, idx in enumerate(ranking, start=1):
        if is_misclassified[idx]:
            fault_positions.append(pos)

    sum_positions = sum(fault_positions)
    apfd = 1 - (sum_positions / (n * m)) + (1 / (2 * n))

    return apfd


def calculate_pfd_at_k(ranking: np.ndarray, is_misclassified: np.ndarray, k_percent: float) -> float:
    """
    Calculate Percentage of Faults Detected at k%.

    Args:
        ranking: Indices in priority order
        is_misclassified: Boolean array indicating misclassifications
        k_percent: Percentage of test cases to consider (e.g., 10, 20, 50)

    Returns:
        PFD@k score
    """
    n = len(ranking)
    m = is_misclassified.sum()

    if m == 0:
        return 1.0

    k = int(n * k_percent / 100)
    k = max(1, k)  # At least 1

    top_k_indices = ranking[:k]
    faults_found = is_misclassified[top_k_indices].sum()

    return faults_found / m


# ============================================================================
# PRIORITIZATION METHODS
# ============================================================================

def prioritize_random(probs: np.ndarray, **kwargs) -> np.ndarray:
    """Random prioritization (baseline)."""
    n = probs.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


def prioritize_deepgini(probs: np.ndarray, **kwargs) -> np.ndarray:
    """DeepGini: prioritize by Gini impurity."""
    gini = 1.0 - (probs ** 2).sum(axis=1)
    return np.argsort(-gini)  # Higher Gini first


def prioritize_entropy(probs: np.ndarray, **kwargs) -> np.ndarray:
    """Entropy: prioritize by prediction entropy."""
    from scipy.stats import entropy
    ent = entropy(probs, axis=1, base=2)
    return np.argsort(-ent)  # Higher entropy first


def prioritize_margin(probs: np.ndarray, **kwargs) -> np.ndarray:
    """VanillaSM: prioritize by negative margin (smaller margin first)."""
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argsort(margin)  # Smaller margin first


def prioritize_least_confidence(probs: np.ndarray, **kwargs) -> np.ndarray:
    """PCS: prioritize by least confidence."""
    confidence = probs.max(axis=1)
    return np.argsort(confidence)  # Lower confidence first


def prioritize_filo_priori(
    probs: np.ndarray,
    edge_index: np.ndarray,
    num_classes: int,
    test_indices: np.ndarray,
    num_nodes: int,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    **kwargs
) -> np.ndarray:
    """
    Filo-Priori: combine uncertainty with structural features.

    Score = alpha * uncertainty + beta * structural + gamma * neighbor_disagreement

    Args:
        probs: Test node probabilities [N_test, C]
        edge_index: Full graph edges [2, E]
        num_classes: Number of classes
        test_indices: Indices of test nodes in the full graph [N_test]
        num_nodes: Total number of nodes in full graph
        alpha, beta, gamma: Weights for combining signals
    """
    from scipy.stats import entropy as scipy_entropy

    N_test = probs.shape[0]

    # 1. Uncertainty (entropy) - computed on test probs
    ent = scipy_entropy(probs, axis=1, base=2)
    max_ent = np.log2(num_classes)
    uncertainty = ent / max_ent

    # 2. Structural importance (degree centrality from full graph)
    # Compute degree for all nodes, then select test nodes
    degree_full = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.size > 0:
        src, tgt = edge_index[0], edge_index[1]
        np.add.at(degree_full, src, 1)
        np.add.at(degree_full, tgt, 1)
    max_degree = max(degree_full.max(), 1)
    degree_full = degree_full / max_degree
    degree = degree_full[test_indices]

    # 3. Neighbor disagreement
    # For each test node, check what fraction of neighbors have different predicted class
    predictions = probs.argmax(axis=1)

    # Create mapping from global index to test index
    global_to_test = {g: t for t, g in enumerate(test_indices)}

    # Build neighbor info for test nodes
    disagreement = np.zeros(N_test, dtype=np.float32)
    neighbor_count = np.zeros(N_test, dtype=np.float32)

    if edge_index.size > 0:
        src, tgt = edge_index[0], edge_index[1]

        for s, t in zip(src, tgt):
            # If source is a test node
            if s in global_to_test:
                src_test_idx = global_to_test[s]
                # If target is also a test node, check if they disagree
                if t in global_to_test:
                    tgt_test_idx = global_to_test[t]
                    if predictions[src_test_idx] != predictions[tgt_test_idx]:
                        disagreement[src_test_idx] += 1
                neighbor_count[src_test_idx] += 1

            # For undirected edges
            if t in global_to_test:
                tgt_test_idx = global_to_test[t]
                if s in global_to_test:
                    src_test_idx = global_to_test[s]
                    if predictions[tgt_test_idx] != predictions[src_test_idx]:
                        disagreement[tgt_test_idx] += 1
                neighbor_count[tgt_test_idx] += 1

    # Avoid division by zero
    neighbor_count = np.maximum(neighbor_count, 1)
    disagreement = disagreement / neighbor_count

    # Combine: higher score = prioritize first
    score = alpha * uncertainty + beta * degree + gamma * disagreement

    return np.argsort(-score)  # Higher score first


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    dataset_name: str,
    config: Dict,
    device: torch.device,
    run_id: int = 0
) -> Dict:
    """
    Run a single experiment on a dataset.

    Args:
        dataset_name: Name of dataset
        config: Model and training configuration
        device: torch device
        run_id: Run identifier for reproducibility

    Returns:
        Dictionary with results
    """
    # Set seed
    seed = config.get('seed', 42) + run_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data, num_classes = load_dataset(dataset_name)

    # Create model
    model = create_gnn_filo_priori(
        num_features=data.num_node_features,
        num_classes=num_classes,
        config=config.get('model', {})
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    num_epochs = config.get('num_epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Loss function
    criterion = FocalLoss(
        alpha=config.get('focal_alpha', 0.5),
        gamma=config.get('focal_gamma', 2.0)
    )

    # Training
    best_val_acc = 0.0
    best_model_state = None
    patience = config.get('patience', 20)
    patience_counter = 0

    logger.info(f"\nTraining GNN Filo-Priori on {dataset_name}...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion, device)

        # Validate
        val_acc, _, _ = evaluate(model, data, device, data.val_mask)

        # Scheduler step
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            logger.info(f"  Epoch {epoch}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_acc, probs, labels = evaluate(model, data, device, data.test_mask)
    logger.info(f"  Test Accuracy: {test_acc:.4f}")

    # Get test indices and predictions
    test_mask = data.test_mask.numpy()
    test_indices = np.where(test_mask)[0]
    test_probs = probs[test_indices]
    test_labels = labels[test_indices]

    # Identify misclassifications
    test_preds = test_probs.argmax(axis=1)
    is_misclassified = (test_preds != test_labels)
    num_misclassified = is_misclassified.sum()
    logger.info(f"  Misclassified: {num_misclassified}/{len(test_labels)} ({100*num_misclassified/len(test_labels):.1f}%)")

    # Get edge_index for the test set (needed for structural features)
    edge_index = data.edge_index.numpy()

    # Total number of nodes in the graph
    num_nodes = data.num_nodes

    # Prioritization methods
    methods = {
        'Random': lambda: prioritize_random(test_probs),
        'DeepGini': lambda: prioritize_deepgini(test_probs),
        'Entropy': lambda: prioritize_entropy(test_probs),
        'VanillaSM': lambda: prioritize_margin(test_probs),
        'PCS': lambda: prioritize_least_confidence(test_probs),
        'Filo-Priori': lambda: prioritize_filo_priori(
            test_probs, edge_index, num_classes,
            test_indices=test_indices,
            num_nodes=num_nodes,
            alpha=config.get('filo_alpha', 0.4),
            beta=config.get('filo_beta', 0.3),
            gamma=config.get('filo_gamma', 0.3)
        )
    }

    results = {
        'dataset': dataset_name,
        'run_id': run_id,
        'test_accuracy': test_acc,
        'num_test': len(test_labels),
        'num_misclassified': int(num_misclassified),
        'misclassified_rate': num_misclassified / len(test_labels)
    }

    # Calculate metrics for each method
    for method_name, prioritize_fn in methods.items():
        # Get ranking (average over 10 random runs for Random method)
        if method_name == 'Random':
            apfd_scores = []
            pfd10_scores = []
            pfd20_scores = []

            for _ in range(10):
                ranking = prioritize_fn()
                apfd_scores.append(calculate_apfd(ranking, is_misclassified))
                pfd10_scores.append(calculate_pfd_at_k(ranking, is_misclassified, 10))
                pfd20_scores.append(calculate_pfd_at_k(ranking, is_misclassified, 20))

            apfd = np.mean(apfd_scores)
            pfd10 = np.mean(pfd10_scores)
            pfd20 = np.mean(pfd20_scores)
        else:
            ranking = prioritize_fn()
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
    """
    Run full experiment across all datasets.

    Args:
        datasets: List of dataset names
        config: Configuration dictionary
        n_runs: Number of runs per dataset
        output_dir: Directory to save results

    Returns:
        DataFrame with all results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if output_dir is None:
        output_dir = project_root / 'experiments' / 'results' / 'gnn_filo_priori'
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

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'full_results_{timestamp}.csv', index=False)

    # Generate summary statistics
    summary = generate_summary(df)
    summary.to_csv(output_dir / f'summary_{timestamp}.csv')

    # Print summary
    print_summary(summary)

    return df


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from results."""
    methods = ['Random', 'DeepGini', 'Entropy', 'VanillaSM', 'PCS', 'Filo-Priori']
    metrics = ['APFD', 'PFD@10', 'PFD@20']

    summary_data = []

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]

        for method in methods:
            row = {'Dataset': dataset, 'Method': method}

            for metric in metrics:
                col = f'{method}_{metric}'
                if col in dataset_df.columns:
                    row[f'{metric}_mean'] = dataset_df[col].mean()
                    row[f'{metric}_std'] = dataset_df[col].std()

            summary_data.append(row)

    return pd.DataFrame(summary_data)


def print_summary(summary: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY: GNN Filo-Priori vs Baselines")
    print("=" * 80)

    datasets = summary['Dataset'].unique()

    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-" * 60)

        dataset_df = summary[summary['Dataset'] == dataset]
        print(f"{'Method':<15} {'APFD':>12} {'PFD@10':>12} {'PFD@20':>12}")
        print("-" * 60)

        for _, row in dataset_df.iterrows():
            method = row['Method']
            apfd = f"{row['APFD_mean']:.4f}±{row['APFD_std']:.4f}"
            pfd10 = f"{row['PFD@10_mean']:.4f}±{row['PFD@10_std']:.4f}"
            pfd20 = f"{row['PFD@20_mean']:.4f}±{row['PFD@20_std']:.4f}"
            print(f"{method:<15} {apfd:>12} {pfd10:>12} {pfd20:>12}")

    print("\n" + "=" * 80)

    # Overall comparison
    print("\nOVERALL IMPROVEMENT (Filo-Priori vs Random):")
    print("-" * 60)

    for dataset in datasets:
        dataset_df = summary[summary['Dataset'] == dataset]
        random_apfd = dataset_df[dataset_df['Method'] == 'Random']['APFD_mean'].values[0]
        filo_apfd = dataset_df[dataset_df['Method'] == 'Filo-Priori']['APFD_mean'].values[0]
        improvement = (filo_apfd - random_apfd) / random_apfd * 100
        print(f"  {dataset}: +{improvement:.1f}% APFD improvement")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run GNN Filo-Priori Experiment')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['cora', 'citeseer', 'pubmed'],
        help='Datasets to run (default: cora citeseer pubmed)'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=5,
        help='Number of runs per dataset (default: 5)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension (default: 256)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Configuration
    config = {
        'seed': 42,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'patience': 20,
        'focal_alpha': 0.5,
        'focal_gamma': 2.0,

        # Filo-Priori weights
        'filo_alpha': 0.4,   # Uncertainty weight
        'filo_beta': 0.3,    # Structural weight
        'filo_gamma': 0.3,   # Neighbor disagreement weight

        # Model config
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_gat_heads': 4,
            'num_gat_layers': 2,
            'num_ffn_layers': 2,
            'fusion_type': 'cross_attention',
            'dropout': 0.3,
            'classifier_hidden_dims': [128, 64]
        }
    }

    # Save config
    output_dir = Path(args.output_dir) if args.output_dir else project_root / 'experiments' / 'results' / 'gnn_filo_priori'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Run experiment
    logger.info("=" * 70)
    logger.info("GNN FILO-PRIORI EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Runs per dataset: {args.n_runs}")
    logger.info(f"Config: {config}")
    logger.info("=" * 70)

    df = run_full_experiment(
        datasets=args.datasets,
        config=config,
        n_runs=args.n_runs,
        output_dir=output_dir
    )

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
