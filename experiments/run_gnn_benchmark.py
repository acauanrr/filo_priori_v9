#!/usr/bin/env python3
"""
GNN Benchmark Experiment Script

This script runs experiments to compare Filo-Priori with test prioritization
methods from the NodeRank paper (IEEE TSE 2024):
"Test Input Prioritization for Graph Neural Networks"

Datasets:
- Cora (2,708 nodes, 7 classes)
- CiteSeer (3,327 nodes, 6 classes)
- PubMed (19,717 nodes, 3 classes)

GNN Models:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Graph Sample and Aggregate)
- TAGCN (Topology Adaptive GCN)

Compared Methods:
- Random: Random ordering
- DeepGini: Based on Gini coefficient of predictions
- VanillaSM: 1 - max(softmax)
- PCS: Prediction Confidence Score (max - second_max)
- Entropy: Entropy of softmax predictions
- Filo-Priori: Our semantic+structural approach (adapted for GNN)

Metrics:
- APFD (Average Percentage of Fault Detection)
- PFD-n (Percentage of Faults Detected at n%)

Usage:
    python run_gnn_benchmark.py [--datasets cora citeseer pubmed] [--models gcn gat sage tagcn]

Author: Filo-Priori Team
Date: 2024
"""

import os
import sys
import argparse
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TAGConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# GNN MODEL IMPLEMENTATIONS
# =============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017)"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Get node embeddings before final classification"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network (Velickovic et al., 2018)"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        return x


class GraphSAGE(nn.Module):
    """Graph Sample and Aggregate (Hamilton et al., 2017)"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


class TAGCN(nn.Module):
    """Topology Adaptive Graph Convolutional Network (Du et al., 2017)"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 K: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels, K=K)
        self.conv2 = TAGConv(hidden_channels, out_channels, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_dataset(name: str, data_dir: str = None) -> Tuple[Data, dict]:
    """Load a preprocessed GNN benchmark dataset.

    Args:
        name: Dataset name ('cora', 'citeseer', 'pubmed')
        data_dir: Directory containing processed datasets

    Returns:
        PyG Data object and dataset info dict
    """
    if data_dir is None:
        data_dir = project_root / 'datasets' / '03_gnn_benchmarks' / 'processed'

    pkl_path = Path(data_dir) / name / f'{name}_processed.pkl'

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Dataset {name} not found at {pkl_path}. "
            f"Run 'python datasets/03_gnn_benchmarks/download_datasets.py' first."
        )

    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    # Convert to PyG Data object
    data = Data(
        x=torch.FloatTensor(dataset['x']),
        edge_index=torch.LongTensor(dataset['edge_index']),
        y=torch.LongTensor(dataset['y']),
        train_mask=torch.BoolTensor(dataset['train_mask']),
        val_mask=torch.BoolTensor(dataset['val_mask']) if dataset['val_mask'] is not None else None,
        test_mask=torch.BoolTensor(dataset['test_mask'])
    )

    info = {
        'name': name,
        'num_nodes': dataset['num_nodes'],
        'num_edges': dataset['num_edges'],
        'num_features': dataset['num_features'],
        'num_classes': dataset['num_classes']
    }

    return data, info


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_gnn(model: nn.Module, data: Data, epochs: int = 200, lr: float = 0.01,
              weight_decay: float = 5e-4, patience: int = 20) -> Tuple[nn.Module, dict]:
    """Train a GNN model.

    Args:
        model: GNN model
        data: PyG Data object
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            train_acc = float((pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum())

            if data.val_mask is not None:
                val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum())
            else:
                val_acc = train_acc

            history['train_loss'].append(float(loss))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_val_acc'] = best_val_acc

    return model, history


# =============================================================================
# TEST PRIORITIZATION METHODS
# =============================================================================

def get_predictions(model: nn.Module, data: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get model predictions and probabilities for test nodes.

    Returns:
        - predictions: Predicted class for each test node
        - probabilities: Softmax probabilities (n_test x n_classes)
        - true_labels: Ground truth labels
    """
    model.eval()
    data = data.to(DEVICE)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

    test_mask = data.test_mask.cpu().numpy()

    return (
        preds.cpu().numpy()[test_mask],
        probs.cpu().numpy()[test_mask],
        data.y.cpu().numpy()[test_mask]
    )


def prioritize_random(n: int, seed: int = 42) -> np.ndarray:
    """Random prioritization (baseline)."""
    np.random.seed(seed)
    return np.random.permutation(n)


def prioritize_deepgini(probs: np.ndarray) -> np.ndarray:
    """DeepGini: Prioritize by Gini impurity of predictions.

    Higher Gini = more uncertain = higher priority
    Gini = 1 - sum(p_i^2)
    """
    gini = 1 - np.sum(probs ** 2, axis=1)
    return np.argsort(-gini)  # Higher Gini first


def prioritize_vanilla_softmax(probs: np.ndarray) -> np.ndarray:
    """VanillaSM: Prioritize by 1 - max(softmax).

    Lower confidence = higher priority
    """
    max_probs = np.max(probs, axis=1)
    scores = 1 - max_probs
    return np.argsort(-scores)  # Higher score (lower confidence) first


def prioritize_pcs(probs: np.ndarray) -> np.ndarray:
    """PCS (Prediction Confidence Score): max - second_max.

    Lower margin = higher priority
    """
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argsort(margins)  # Lower margin first


def prioritize_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy: Prioritize by prediction entropy.

    Higher entropy = more uncertain = higher priority
    """
    # Add small epsilon to avoid log(0)
    probs_safe = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    return np.argsort(-entropy)  # Higher entropy first


def prioritize_filo_priori(model: nn.Module, data: Data, probs: np.ndarray) -> np.ndarray:
    """Filo-Priori: Combine structural features with semantic similarity.

    Adaptation for GNN:
    - Use node embeddings as semantic features
    - Use graph structure features (degree, neighbors)
    - Combine with prediction uncertainty

    Score = alpha * uncertainty + beta * structural_score + gamma * neighborhood_failure_prob
    """
    model.eval()
    data = data.to(DEVICE)

    test_mask = data.test_mask.cpu().numpy()
    test_indices = np.where(test_mask)[0]
    n_test = len(test_indices)

    with torch.no_grad():
        # Get node embeddings
        embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    # Component 1: Uncertainty (using entropy)
    probs_safe = np.clip(probs, 1e-10, 1.0)
    uncertainty = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-10)

    # Component 2: Structural features (node degree normalized)
    edge_index = data.edge_index.cpu().numpy()
    degrees = np.zeros(data.num_nodes)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[0, i]] += 1
    test_degrees = degrees[test_indices]
    test_degrees = (test_degrees - test_degrees.min()) / (test_degrees.max() - test_degrees.min() + 1e-10)

    # Component 3: Neighborhood prediction diversity
    # Nodes whose neighbors have diverse predictions are more likely to be misclassified
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        all_probs = F.softmax(out, dim=1).cpu().numpy()

    neighbor_diversity = np.zeros(n_test)
    for idx, node_idx in enumerate(test_indices):
        # Get neighbors
        neighbors = edge_index[1, edge_index[0] == node_idx]
        if len(neighbors) > 0:
            neighbor_probs = all_probs[neighbors]
            # Diversity = average entropy of neighbors
            neighbor_entropy = -np.sum(neighbor_probs * np.log(np.clip(neighbor_probs, 1e-10, 1.0)), axis=1)
            neighbor_diversity[idx] = np.mean(neighbor_entropy)

    if neighbor_diversity.max() > neighbor_diversity.min():
        neighbor_diversity = (neighbor_diversity - neighbor_diversity.min()) / (neighbor_diversity.max() - neighbor_diversity.min())

    # Combine scores (hyperparameters can be tuned)
    alpha = 0.5  # uncertainty weight
    beta = 0.3   # structural weight
    gamma = 0.2  # neighborhood weight

    scores = alpha * uncertainty + beta * test_degrees + gamma * neighbor_diversity

    return np.argsort(-scores)  # Higher score first


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_apfd(ranking: np.ndarray, is_misclassified: np.ndarray) -> float:
    """Calculate APFD (Average Percentage of Faults Detected).

    Args:
        ranking: Order of test indices (ranking[0] is tested first)
        is_misclassified: Boolean array indicating misclassified samples

    Returns:
        APFD score (0 to 1, higher is better)
    """
    n = len(ranking)
    n_faults = is_misclassified.sum()

    if n_faults == 0:
        return None
    if n == 1:
        return 1.0

    # Get positions (1-indexed) of misclassified samples in the ranking
    fault_positions = []
    for pos, idx in enumerate(ranking, start=1):
        if is_misclassified[idx]:
            fault_positions.append(pos)

    apfd = 1 - sum(fault_positions) / (n_faults * n) + 1 / (2 * n)
    return float(np.clip(apfd, 0, 1))


def calculate_pfd(ranking: np.ndarray, is_misclassified: np.ndarray,
                  percentages: List[int] = [10, 20, 30, 40, 50, 60]) -> Dict[str, float]:
    """Calculate PFD at various percentages.

    Args:
        ranking: Order of test indices
        is_misclassified: Boolean array indicating misclassified samples
        percentages: List of percentages to calculate PFD for

    Returns:
        Dictionary with PFD-n values
    """
    n = len(ranking)
    n_faults = is_misclassified.sum()

    if n_faults == 0:
        return {f'PFD-{p}': None for p in percentages}

    results = {}
    for p in percentages:
        k = int(n * p / 100)
        selected = ranking[:k]
        detected = sum(is_misclassified[selected])
        results[f'PFD-{p}'] = detected / n_faults

    return results


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(dataset_name: str, model_name: str, seed: int = 42) -> Dict:
    """Run a single experiment (one dataset + one model).

    Args:
        dataset_name: Name of dataset
        model_name: Name of GNN model
        seed: Random seed

    Returns:
        Dictionary with results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}, Model: {model_name.upper()}")
    print(f"{'='*60}")

    # Load dataset
    data, info = load_dataset(dataset_name)
    print(f"Loaded {info['name']}: {info['num_nodes']} nodes, {info['num_edges']} edges, "
          f"{info['num_features']} features, {info['num_classes']} classes")

    # Create model
    hidden_channels = 64
    if model_name == 'gcn':
        model = GCN(info['num_features'], hidden_channels, info['num_classes'])
    elif model_name == 'gat':
        model = GAT(info['num_features'], hidden_channels, info['num_classes'])
    elif model_name == 'sage':
        model = GraphSAGE(info['num_features'], hidden_channels, info['num_classes'])
    elif model_name == 'tagcn':
        model = TAGCN(info['num_features'], hidden_channels, info['num_classes'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train model
    print(f"\nTraining {model_name.upper()}...")
    start_time = time.time()
    model, history = train_gnn(model, data)
    train_time = time.time() - start_time
    print(f"  Best validation accuracy: {history['best_val_acc']:.4f}")
    print(f"  Training time: {train_time:.1f}s")

    # Get predictions for test set
    preds, probs, true_labels = get_predictions(model, data)
    is_misclassified = preds != true_labels
    n_test = len(preds)
    n_misclassified = is_misclassified.sum()
    test_acc = 1 - n_misclassified / n_test

    print(f"\nTest set: {n_test} nodes")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Misclassified: {n_misclassified} ({100*n_misclassified/n_test:.1f}%)")

    if n_misclassified == 0:
        print("  WARNING: No misclassified samples. Skipping prioritization.")
        return {
            'dataset': dataset_name,
            'model': model_name,
            'test_accuracy': test_acc,
            'n_test': n_test,
            'n_misclassified': 0,
            'error': 'No misclassified samples'
        }

    # Run all prioritization methods
    methods = {
        'Random': prioritize_random(n_test, seed),
        'DeepGini': prioritize_deepgini(probs),
        'VanillaSM': prioritize_vanilla_softmax(probs),
        'PCS': prioritize_pcs(probs),
        'Entropy': prioritize_entropy(probs),
        'Filo-Priori': prioritize_filo_priori(model, data, probs)
    }

    # Calculate metrics
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'test_accuracy': test_acc,
        'n_test': n_test,
        'n_misclassified': int(n_misclassified),
        'train_time': train_time,
        'methods': {}
    }

    print(f"\nResults:")
    print(f"{'Method':<15} {'APFD':>8} {'PFD-10':>8} {'PFD-20':>8} {'PFD-30':>8} {'PFD-50':>8}")
    print("-" * 65)

    for method_name, ranking in methods.items():
        apfd = calculate_apfd(ranking, is_misclassified)
        pfd = calculate_pfd(ranking, is_misclassified)

        results['methods'][method_name] = {
            'APFD': apfd,
            **pfd
        }

        print(f"{method_name:<15} {apfd:>8.4f} {pfd['PFD-10']:>8.4f} {pfd['PFD-20']:>8.4f} "
              f"{pfd['PFD-30']:>8.4f} {pfd['PFD-50']:>8.4f}")

    return results


def run_full_experiment(datasets: List[str], models: List[str],
                        n_runs: int = 1, output_dir: str = None) -> pd.DataFrame:
    """Run full experiment across all datasets and models.

    Args:
        datasets: List of dataset names
        models: List of model names
        n_runs: Number of runs (for statistical analysis)
        output_dir: Directory to save results

    Returns:
        DataFrame with all results
    """
    if output_dir is None:
        output_dir = project_root / 'results' / 'gnn_benchmark'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in datasets:
        for model in models:
            for run in range(n_runs):
                try:
                    result = run_single_experiment(dataset, model, seed=42 + run)
                    result['run'] = run
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {dataset}/{model}/run{run}: {e}")
                    import traceback
                    traceback.print_exc()

    # Convert to flat DataFrame
    rows = []
    for r in all_results:
        if 'methods' not in r:
            continue
        for method, metrics in r['methods'].items():
            row = {
                'Dataset': r['dataset'],
                'Model': r['model'],
                'Run': r.get('run', 0),
                'Test_Accuracy': r['test_accuracy'],
                'N_Test': r['n_test'],
                'N_Misclassified': r['n_misclassified'],
                'Method': method,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(output_dir / f'gnn_benchmark_results_{timestamp}.csv', index=False)

    # Save raw results as JSON
    with open(output_dir / f'gnn_benchmark_raw_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    summary = df.groupby(['Dataset', 'Model', 'Method']).agg({
        'APFD': ['mean', 'std'],
        'PFD-10': 'mean',
        'PFD-20': 'mean',
        'PFD-30': 'mean',
        'PFD-50': 'mean'
    }).round(4)

    print(summary)

    # Save summary
    summary.to_csv(output_dir / f'gnn_benchmark_summary_{timestamp}.csv')

    # Calculate improvements
    print("\n" + "="*80)
    print("FILO-PRIORI IMPROVEMENT OVER BASELINES")
    print("="*80)

    for dataset in datasets:
        for model in models:
            subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if len(subset) == 0:
                continue

            filo_apfd = subset[subset['Method'] == 'Filo-Priori']['APFD'].mean()
            print(f"\n{dataset.upper()} + {model.upper()}:")
            print(f"  Filo-Priori APFD: {filo_apfd:.4f}")

            for method in ['Random', 'DeepGini', 'VanillaSM', 'PCS', 'Entropy']:
                baseline_apfd = subset[subset['Method'] == method]['APFD'].mean()
                improvement = (filo_apfd - baseline_apfd) / baseline_apfd * 100
                print(f"  vs {method}: {improvement:+.2f}%")

    print(f"\nResults saved to: {output_dir}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run GNN Benchmark Experiments')
    parser.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'],
                        help='Datasets to use')
    parser.add_argument('--models', nargs='+', default=['gcn', 'gat', 'sage', 'tagcn'],
                        help='GNN models to use')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for statistical analysis')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    print("="*80)
    print("GNN BENCHMARK EXPERIMENT")
    print("Comparing Filo-Priori with NodeRank baselines")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Runs: {args.runs}")

    # Run experiment
    results_df = run_full_experiment(
        datasets=args.datasets,
        models=args.models,
        n_runs=args.runs,
        output_dir=args.output
    )

    print("\nExperiment completed!")


if __name__ == '__main__':
    main()
