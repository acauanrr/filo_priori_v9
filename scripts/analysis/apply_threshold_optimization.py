#!/usr/bin/env python3
"""
Script to apply threshold optimization to a trained model.

This script:
1. Loads a trained model from experiment
2. Gets predictions on validation set
3. Finds optimal threshold using threshold_optimizer
4. Re-evaluates on test set with optimized threshold
5. Compares results with default threshold (0.5)

Usage:
    python apply_threshold_optimization.py --config configs/experiment_04a_weighted_ce_only.yaml \
                                          --model-path best_model_v8.pt \
                                          --strategy f1_macro
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.threshold_optimizer import find_optimal_threshold, optimize_threshold_for_minority
from src.preprocessing.data_loader import load_and_prepare_data
from src.embeddings.embedding_manager import EmbeddingManager
from src.phylogenetic.graph_builder import build_phylogenetic_graph
from src.model.dual_stream_model import DualStreamGNN
from src.preprocessing.structural_feature_extractor import StructuralFeatureExtractor
from src.preprocessing.structural_feature_imputation import StructuralFeatureImputer
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluate predictions with a specific threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        threshold: Classification threshold

    Returns:
        Dictionary with metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, recall, F1 per class
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUROC and AUPRC
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except:
        auroc = 0.0
        auprc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Prediction diversity (percentage of samples in each class)
    pred_class_0_pct = (y_pred == 0).sum() / len(y_pred) * 100
    pred_class_1_pct = (y_pred == 1).sum() / len(y_pred) * 100

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': cm,
        'pred_class_0_pct': pred_class_0_pct,
        'pred_class_1_pct': pred_class_1_pct,
        'y_pred': y_pred
    }


def print_comparison(default_metrics: dict, optimized_metrics: dict):
    """Print side-by-side comparison of default vs optimized thresholds."""

    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)

    print(f"\n{'Metric':<30} {'Default (0.5)':<20} {'Optimized':<20} {'Change':<15}")
    print("-" * 85)

    # Threshold
    print(f"{'Threshold':<30} {default_metrics['threshold']:<20.4f} "
          f"{optimized_metrics['threshold']:<20.4f} "
          f"{optimized_metrics['threshold'] - default_metrics['threshold']:+.4f}")

    # Accuracy
    acc_change = optimized_metrics['accuracy'] - default_metrics['accuracy']
    acc_change_pct = acc_change * 100
    print(f"{'Accuracy':<30} {default_metrics['accuracy']:<20.4f} "
          f"{optimized_metrics['accuracy']:<20.4f} "
          f"{acc_change:+.4f} ({acc_change_pct:+.1f}%)")

    # F1 Macro
    f1_change = optimized_metrics['f1_macro'] - default_metrics['f1_macro']
    f1_change_pct = (f1_change / default_metrics['f1_macro'] * 100) if default_metrics['f1_macro'] > 0 else 0
    print(f"{'F1 Macro':<30} {default_metrics['f1_macro']:<20.4f} "
          f"{optimized_metrics['f1_macro']:<20.4f} "
          f"{f1_change:+.4f} ({f1_change_pct:+.1f}%)")

    # Precision Macro
    prec_change = optimized_metrics['precision_macro'] - default_metrics['precision_macro']
    print(f"{'Precision Macro':<30} {default_metrics['precision_macro']:<20.4f} "
          f"{optimized_metrics['precision_macro']:<20.4f} "
          f"{prec_change:+.4f}")

    # Recall Macro
    rec_change = optimized_metrics['recall_macro'] - default_metrics['recall_macro']
    print(f"{'Recall Macro':<30} {default_metrics['recall_macro']:<20.4f} "
          f"{optimized_metrics['recall_macro']:<20.4f} "
          f"{rec_change:+.4f}")

    # AUROC
    print(f"{'AUROC':<30} {default_metrics['auroc']:<20.4f} "
          f"{optimized_metrics['auroc']:<20.4f} "
          f"{'(unchanged)':<15}")

    # AUPRC
    print(f"{'AUPRC':<30} {default_metrics['auprc']:<20.4f} "
          f"{optimized_metrics['auprc']:<20.4f} "
          f"{'(unchanged)':<15}")

    print("\n" + "-" * 85)
    print("PER-CLASS METRICS")
    print("-" * 85)

    # Not-Pass class (minority - class 0)
    print(f"\n{'Not-Pass (Minority Class)':}")
    print(f"  {'Precision':<28} {default_metrics['precision_per_class'][0]:<20.4f} "
          f"{optimized_metrics['precision_per_class'][0]:<20.4f} "
          f"{optimized_metrics['precision_per_class'][0] - default_metrics['precision_per_class'][0]:+.4f}")

    recall_notpass_change = optimized_metrics['recall_per_class'][0] - default_metrics['recall_per_class'][0]
    recall_notpass_change_pct = (recall_notpass_change / default_metrics['recall_per_class'][0] * 100) if default_metrics['recall_per_class'][0] > 0 else float('inf')
    print(f"  {'Recall':<28} {default_metrics['recall_per_class'][0]:<20.4f} "
          f"{optimized_metrics['recall_per_class'][0]:<20.4f} "
          f"{recall_notpass_change:+.4f} ({recall_notpass_change_pct:+.1f}%)")

    print(f"  {'F1-Score':<28} {default_metrics['f1_per_class'][0]:<20.4f} "
          f"{optimized_metrics['f1_per_class'][0]:<20.4f} "
          f"{optimized_metrics['f1_per_class'][0] - default_metrics['f1_per_class'][0]:+.4f}")

    # Pass class (majority - class 1)
    print(f"\n{'Pass (Majority Class)':}")
    print(f"  {'Precision':<28} {default_metrics['precision_per_class'][1]:<20.4f} "
          f"{optimized_metrics['precision_per_class'][1]:<20.4f} "
          f"{optimized_metrics['precision_per_class'][1] - default_metrics['precision_per_class'][1]:+.4f}")

    print(f"  {'Recall':<28} {default_metrics['recall_per_class'][1]:<20.4f} "
          f"{optimized_metrics['recall_per_class'][1]:<20.4f} "
          f"{optimized_metrics['recall_per_class'][1] - default_metrics['recall_per_class'][1]:+.4f}")

    print(f"  {'F1-Score':<28} {default_metrics['f1_per_class'][1]:<20.4f} "
          f"{optimized_metrics['f1_per_class'][1]:<20.4f} "
          f"{optimized_metrics['f1_per_class'][1] - default_metrics['f1_per_class'][1]:+.4f}")

    print("\n" + "-" * 85)
    print("PREDICTION DISTRIBUTION")
    print("-" * 85)

    print(f"\n{'Class':<30} {'Default (0.5)':<20} {'Optimized':<20}")
    print(f"{'Not-Pass predictions':<30} {default_metrics['pred_class_0_pct']:<20.2f}% "
          f"{optimized_metrics['pred_class_0_pct']:<20.2f}%")
    print(f"{'Pass predictions':<30} {default_metrics['pred_class_1_pct']:<20.2f}% "
          f"{optimized_metrics['pred_class_1_pct']:<20.2f}%")

    print("\n" + "="*80)


def plot_threshold_curves(y_true: np.ndarray, y_prob: np.ndarray,
                          optimal_threshold: float, output_dir: str):
    """
    Plot threshold analysis curves.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        optimal_threshold: The optimal threshold found
        output_dir: Directory to save plots
    """
    thresholds = np.linspace(0.01, 0.99, 99)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores_list = []
    recall_minority = []
    recall_majority = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1_scores_list.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        recall_minority.append(recall_per_class[0])
        recall_majority.append(recall_per_class[1])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Overall metrics
    ax = axes[0, 0]
    ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    ax.plot(thresholds, precisions, label='Precision (Macro)', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall (Macro)', linewidth=2)
    ax.plot(thresholds, f1_scores_list, label='F1 (Macro)', linewidth=2, color='red')
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default=0.5')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: F1 Score (zoomed)
    ax = axes[0, 1]
    ax.plot(thresholds, f1_scores_list, linewidth=2, color='red')
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default=0.5')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('F1 Macro Score vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-class recall
    ax = axes[1, 0]
    ax.plot(thresholds, recall_minority, label='Recall Not-Pass (Minority)', linewidth=2, color='blue')
    ax.plot(thresholds, recall_majority, label='Recall Pass (Majority)', linewidth=2, color='green')
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default=0.5')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Per-Class Recall vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 4: Prediction distribution
    ax = axes[1, 1]
    pred_minority_pct = [(y_prob < t).sum() / len(y_prob) * 100 for t in thresholds]
    pred_majority_pct = [(y_prob >= t).sum() / len(y_prob) * 100 for t in thresholds]
    ax.plot(thresholds, pred_minority_pct, label='% Predicted Not-Pass', linewidth=2, color='blue')
    ax.plot(thresholds, pred_majority_pct, label='% Predicted Pass', linewidth=2, color='green')
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Default=0.5')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Prediction Distribution vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'threshold_optimization_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Threshold analysis curves saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Apply threshold optimization to trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config file')
    parser.add_argument('--model-path', type=str, default='best_model_v8.pt',
                       help='Path to saved model checkpoint')
    parser.add_argument('--strategy', type=str, default='f1_macro',
                       choices=['f1_macro', 'recall_minority', 'youden', 'custom'],
                       help='Threshold optimization strategy')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (defaults to experiment results dir)')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = config.get('output', {}).get('results_dir', 'results/threshold_optimization')

    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)

    data_config = config['data']
    df_train, df_val, df_test = load_and_prepare_data(
        train_path=data_config['train_path'],
        test_path=data_config.get('test_path'),
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        test_split=data_config.get('test_split', 0.1),
        random_seed=data_config.get('random_seed', 42),
        binary_classification=data_config.get('binary_classification', True),
        binary_strategy=data_config.get('binary_strategy', 'pass_vs_fail'),
        binary_positive_class=data_config.get('binary_positive_class', 'Pass'),
        binary_negative_class=data_config.get('binary_negative_class', 'Fail'),
        smote_config=data_config.get('smote', {})
    )

    logger.info(f"  Train: {len(df_train)} samples")
    logger.info(f"  Val: {len(df_val)} samples")
    logger.info(f"  Test: {len(df_test)} samples")

    # Load embeddings
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING EMBEDDINGS")
    logger.info("="*80)

    embedding_manager = EmbeddingManager(config['embedding'])
    train_embeddings, val_embeddings, test_embeddings = embedding_manager.get_or_create_embeddings(
        df_train, df_val, df_test
    )

    logger.info(f"  Train embeddings: {train_embeddings.shape}")
    logger.info(f"  Val embeddings: {val_embeddings.shape}")
    logger.info(f"  Test embeddings: {test_embeddings.shape}")

    # Extract structural features
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EXTRACTING STRUCTURAL FEATURES")
    logger.info("="*80)

    structural_config = config.get('structural', {})
    extractor = StructuralFeatureExtractor(
        recent_window=structural_config.get('extractor', {}).get('recent_window', 5),
        min_history=structural_config.get('extractor', {}).get('min_history', 2),
        cache_path=structural_config.get('extractor', {}).get('cache_path')
    )

    extractor.fit(df_train)

    train_structural = extractor.transform(df_train)
    val_structural = extractor.transform(pd.concat([df_train, df_val]))
    test_structural = extractor.transform(pd.concat([df_train, df_val, df_test]))

    # Imputation
    imputer = StructuralFeatureImputer()
    imputer.fit(df_train, train_structural, train_embeddings)

    train_structural_imputed = imputer.transform(df_train, train_structural, train_embeddings)
    val_structural_imputed = imputer.transform(
        pd.concat([df_train, df_val]), val_structural, val_embeddings
    )
    test_structural_imputed = imputer.transform(
        pd.concat([df_train, df_val, df_test]), test_structural, test_embeddings
    )

    logger.info(f"  Train structural features: {train_structural_imputed.shape}")
    logger.info(f"  Val structural features: {val_structural_imputed.shape}")
    logger.info(f"  Test structural features: {test_structural_imputed.shape}")

    # Build graph
    logger.info("\n" + "="*80)
    logger.info("STEP 4: BUILDING GRAPH")
    logger.info("="*80)

    graph_config = config['graph']
    use_multi_edge = graph_config.get('use_multi_edge', False)

    if use_multi_edge:
        from src.phylogenetic.multi_edge_graph_builder import MultiEdgeGraphBuilder

        graph_builder = MultiEdgeGraphBuilder(
            edge_types=graph_config.get('edge_types', ['co_failure', 'co_success', 'semantic']),
            edge_weights=graph_config.get('edge_weights'),
            min_co_occurrences=graph_config.get('min_co_occurrences', 1),
            weight_threshold=graph_config.get('weight_threshold', 0.05),
            semantic_top_k=graph_config.get('semantic_top_k', 10),
            semantic_threshold=graph_config.get('semantic_threshold', 0.7)
        )
        graph_builder.fit(df_train, train_embeddings)
    else:
        graph_builder = build_phylogenetic_graph(df_train, use_multi_edge=False)

    # Create PyG Data objects
    val_labels = df_val['label'].values
    test_labels = df_test['label'].values

    val_tc_keys = df_val['TC_Key'].values
    test_tc_keys = df_test['TC_Key'].values

    # Map TC_Keys to indices
    tc_to_idx = {tc: idx for idx, tc in enumerate(df_train['TC_Key'].unique())}

    val_indices = [tc_to_idx[tc] for tc in val_tc_keys if tc in tc_to_idx]
    test_indices = [tc_to_idx[tc] for tc in test_tc_keys if tc in tc_to_idx]

    # Load model
    logger.info("\n" + "="*80)
    logger.info("STEP 5: LOADING MODEL")
    logger.info("="*80)

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    model_config = config['model']
    model = DualStreamGNN(
        semantic_input_dim=model_config['semantic']['input_dim'],
        structural_input_dim=model_config['structural']['input_dim'],
        semantic_hidden_dim=model_config['semantic']['hidden_dim'],
        structural_hidden_dim=model_config['structural']['hidden_dim'],
        gnn_hidden_dim=model_config['gnn']['hidden_dim'],
        num_classes=model_config['num_classes'],
        gnn_type=model_config['gnn']['type'],
        num_gnn_layers=model_config['gnn']['num_layers'],
        num_heads=model_config['gnn'].get('num_heads', 4),
        fusion_hidden_dim=model_config['fusion']['hidden_dim'],
        classifier_hidden_dim=model_config['classifier']['hidden_dim'],
        dropout=model_config['gnn'].get('dropout', 0.2)
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    logger.info(f"Model loaded from: {args.model_path}")

    # Get predictions on validation set
    logger.info("\n" + "="*80)
    logger.info("STEP 6: GETTING PREDICTIONS ON VALIDATION SET")
    logger.info("="*80)

    with torch.no_grad():
        # Prepare validation data
        edge_index, edge_weight = graph_builder.get_edge_index()
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)

        x_semantic = torch.tensor(val_embeddings, dtype=torch.float).to(device)
        x_structural = torch.tensor(val_structural_imputed, dtype=torch.float).to(device)

        # Forward pass
        val_logits = model(x_semantic, x_structural, edge_index, edge_weight)
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_probs_positive = val_probs[:, 1]  # Probability of Pass class

    logger.info(f"  Validation predictions shape: {val_probs.shape}")

    # Find optimal threshold
    logger.info("\n" + "="*80)
    logger.info("STEP 7: FINDING OPTIMAL THRESHOLD")
    logger.info("="*80)

    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_true=val_labels,
        y_prob=val_probs_positive,
        strategy=args.strategy,
        min_threshold=0.01,
        max_threshold=0.99,
        num_thresholds=99
    )

    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"  Expected F1 Macro: {threshold_metrics['f1_macro']:.4f}")

    # Evaluate on test set with both thresholds
    logger.info("\n" + "="*80)
    logger.info("STEP 8: EVALUATING ON TEST SET")
    logger.info("="*80)

    with torch.no_grad():
        x_semantic_test = torch.tensor(test_embeddings, dtype=torch.float).to(device)
        x_structural_test = torch.tensor(test_structural_imputed, dtype=torch.float).to(device)

        test_logits = model(x_semantic_test, x_structural_test, edge_index, edge_weight)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_probs_positive = test_probs[:, 1]

    logger.info(f"  Test predictions shape: {test_probs.shape}")

    # Evaluate with default threshold (0.5)
    logger.info("\nEvaluating with default threshold (0.5)...")
    default_metrics = evaluate_with_threshold(test_labels, test_probs_positive, threshold=0.5)

    # Evaluate with optimized threshold
    logger.info(f"Evaluating with optimized threshold ({optimal_threshold:.4f})...")
    optimized_metrics = evaluate_with_threshold(test_labels, test_probs_positive, threshold=optimal_threshold)

    # Print comparison
    print_comparison(default_metrics, optimized_metrics)

    # Plot threshold curves
    logger.info("\n" + "="*80)
    logger.info("STEP 9: GENERATING THRESHOLD ANALYSIS PLOTS")
    logger.info("="*80)

    plot_threshold_curves(test_labels, test_probs_positive, optimal_threshold, args.output_dir)

    # Save results to file
    results_file = os.path.join(args.output_dir, 'threshold_optimization_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiment Config: {args.config}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Optimization Strategy: {args.strategy}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("COMPARISON: DEFAULT (0.5) vs OPTIMIZED\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Metric':<30} {'Default':<15} {'Optimized':<15} {'Change':<15}\n")
        f.write("-" * 75 + "\n")

        f.write(f"{'Threshold':<30} {0.5:<15.4f} {optimal_threshold:<15.4f} {optimal_threshold-0.5:+.4f}\n")
        f.write(f"{'Accuracy':<30} {default_metrics['accuracy']:<15.4f} "
                f"{optimized_metrics['accuracy']:<15.4f} "
                f"{optimized_metrics['accuracy']-default_metrics['accuracy']:+.4f}\n")
        f.write(f"{'F1 Macro':<30} {default_metrics['f1_macro']:<15.4f} "
                f"{optimized_metrics['f1_macro']:<15.4f} "
                f"{optimized_metrics['f1_macro']-default_metrics['f1_macro']:+.4f}\n")
        f.write(f"{'Precision Macro':<30} {default_metrics['precision_macro']:<15.4f} "
                f"{optimized_metrics['precision_macro']:<15.4f} "
                f"{optimized_metrics['precision_macro']-default_metrics['precision_macro']:+.4f}\n")
        f.write(f"{'Recall Macro':<30} {default_metrics['recall_macro']:<15.4f} "
                f"{optimized_metrics['recall_macro']:<15.4f} "
                f"{optimized_metrics['recall_macro']-default_metrics['recall_macro']:+.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("PER-CLASS RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("Not-Pass (Minority) - Default:\n")
        f.write(f"  Precision: {default_metrics['precision_per_class'][0]:.4f}\n")
        f.write(f"  Recall: {default_metrics['recall_per_class'][0]:.4f}\n")
        f.write(f"  F1: {default_metrics['f1_per_class'][0]:.4f}\n\n")

        f.write("Not-Pass (Minority) - Optimized:\n")
        f.write(f"  Precision: {optimized_metrics['precision_per_class'][0]:.4f}\n")
        f.write(f"  Recall: {optimized_metrics['recall_per_class'][0]:.4f}\n")
        f.write(f"  F1: {optimized_metrics['f1_per_class'][0]:.4f}\n\n")

        f.write("Pass (Majority) - Default:\n")
        f.write(f"  Precision: {default_metrics['precision_per_class'][1]:.4f}\n")
        f.write(f"  Recall: {default_metrics['recall_per_class'][1]:.4f}\n")
        f.write(f"  F1: {default_metrics['f1_per_class'][1]:.4f}\n\n")

        f.write("Pass (Majority) - Optimized:\n")
        f.write(f"  Precision: {optimized_metrics['precision_per_class'][1]:.4f}\n")
        f.write(f"  Recall: {optimized_metrics['recall_per_class'][1]:.4f}\n")
        f.write(f"  F1: {optimized_metrics['f1_per_class'][1]:.4f}\n")

    logger.info(f"\nResults saved to: {results_file}")

    logger.info("\n" + "="*80)
    logger.info("THRESHOLD OPTIMIZATION COMPLETE!")
    logger.info("="*80)

    logger.info(f"\nKey Improvements:")
    logger.info(f"  Recall Not-Pass: {default_metrics['recall_per_class'][0]:.4f} → "
                f"{optimized_metrics['recall_per_class'][0]:.4f} "
                f"({(optimized_metrics['recall_per_class'][0] - default_metrics['recall_per_class'][0]) / default_metrics['recall_per_class'][0] * 100:+.1f}%)")
    logger.info(f"  F1 Macro: {default_metrics['f1_macro']:.4f} → "
                f"{optimized_metrics['f1_macro']:.4f} "
                f"({(optimized_metrics['f1_macro'] - default_metrics['f1_macro']) / default_metrics['f1_macro'] * 100:+.1f}%)")


if __name__ == '__main__':
    main()
