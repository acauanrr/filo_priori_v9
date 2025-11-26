#!/usr/bin/env python3
"""
Run a Single Ablation Experiment for Filo-Priori v9.

This script runs a single ablation experiment by:
1. Loading the ablation configuration
2. Creating an ablation-aware model
3. Training and evaluating
4. Saving APFD results

Usage:
    python run_single_ablation.py --config configs/ablation/ablation_wo_semantic.yaml
    python run_single_ablation.py --ablation A1

Author: Filo-Priori Team
Date: 2025-11-26
"""

import os
import sys
import yaml
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ablation_model import create_ablation_model
from src.evaluation.apfd import calculate_apfd_per_build, generate_apfd_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Ablation ID to config mapping
ABLATION_CONFIGS = {
    'A1': 'configs/ablation/ablation_wo_semantic.yaml',
    'A2': 'configs/ablation/ablation_wo_structural.yaml',
    'A3': 'configs/ablation/ablation_wo_graph.yaml',
    'A4': 'configs/ablation/ablation_wo_multi_edge.yaml',
    'A6': 'configs/ablation/ablation_wo_fusion.yaml',
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_cached_data(config: dict):
    """Load cached embeddings and features."""
    cache_dir = Path(config.get('embedding', {}).get('cache_dir', 'cache'))

    # Load embeddings
    embeddings_path = cache_dir / 'combined_embeddings.pt'
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings_data = torch.load(embeddings_path)
    logger.info(f"Loaded embeddings from {embeddings_path}")

    # Load structural features
    structural_cache = config.get('structural', {}).get('extractor', {}).get('cache_path')
    if structural_cache:
        structural_path = Path(structural_cache)
    else:
        structural_path = cache_dir / 'structural_features_v2_5.pkl'

    if structural_path.exists():
        import pickle
        with open(structural_path, 'rb') as f:
            structural_data = pickle.load(f)
        logger.info(f"Loaded structural features from {structural_path}")
    else:
        logger.warning(f"Structural features not found: {structural_path}")
        structural_data = None

    # Load graph
    graph_cache = config.get('graph', {}).get('cache_path', 'cache/multi_edge_graph.pkl')
    graph_path = Path(graph_cache)

    if graph_path.exists() and config.get('graph', {}).get('build_graph', True):
        import pickle
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        logger.info(f"Loaded graph from {graph_path}")
    else:
        logger.warning(f"Graph not found or disabled: {graph_path}")
        graph_data = None

    return embeddings_data, structural_data, graph_data


def prepare_datasets(embeddings_data, structural_data, graph_data, config):
    """Prepare train/val/test datasets."""
    # Load test.csv for labels
    test_path = config['data']['test_path']
    test_df = pd.read_csv(test_path)

    # Get TC keys and labels
    tc_keys = test_df['TC_Key'].values
    labels = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int).values

    # Get embeddings for test TCs
    tc_to_idx = embeddings_data.get('tc_to_idx', {})
    embeddings = embeddings_data.get('embeddings', embeddings_data.get('combined_embeddings'))

    # Build test dataset
    test_indices = []
    test_labels = []
    test_tc_keys = []

    for i, tc_key in enumerate(tc_keys):
        if tc_key in tc_to_idx:
            test_indices.append(tc_to_idx[tc_key])
            test_labels.append(labels[i])
            test_tc_keys.append(tc_key)

    test_embeddings = embeddings[test_indices]

    # Get structural features
    if structural_data is not None:
        structural_features = []
        for tc_key in test_tc_keys:
            if tc_key in structural_data:
                structural_features.append(structural_data[tc_key])
            else:
                # Default features for unknown TCs
                n_features = config.get('structural', {}).get('input_dim', 10)
                structural_features.append(np.zeros(n_features))
        structural_features = np.array(structural_features)
    else:
        n_features = config.get('structural', {}).get('input_dim', 10)
        structural_features = np.zeros((len(test_tc_keys), n_features))

    # Get graph
    if graph_data is not None:
        edge_index = graph_data.get('edge_index', torch.zeros(2, 0, dtype=torch.long))
        edge_weights = graph_data.get('edge_weights', None)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weights = None

    return {
        'embeddings': torch.tensor(test_embeddings, dtype=torch.float32),
        'structural': torch.tensor(structural_features, dtype=torch.float32),
        'labels': torch.tensor(test_labels, dtype=torch.long),
        'tc_keys': test_tc_keys,
        'edge_index': edge_index if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long),
        'edge_weights': edge_weights if edge_weights is None else torch.tensor(edge_weights, dtype=torch.float32),
        'test_df': test_df
    }


def train_ablation_model(model, train_data, val_data, config, device):
    """Train the ablation model."""
    # Training config
    train_config = config.get('training', {})
    num_epochs = train_config.get('num_epochs', 50)
    batch_size = train_config.get('batch_size', 32)
    lr = train_config.get('learning_rate', 3e-5)
    weight_decay = train_config.get('weight_decay', 1e-4)

    # Loss function
    loss_config = train_config.get('loss', {})
    if loss_config.get('use_class_weights', True):
        # Calculate class weights
        labels = train_data['labels'].numpy()
        n_neg = (labels == 0).sum()
        n_pos = (labels == 1).sum()
        weight_neg = n_pos / (n_neg + n_pos)
        weight_pos = n_neg / (n_neg + n_pos)
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted CE: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Create data loader
    train_dataset = TensorDataset(
        train_data['embeddings'],
        train_data['structural'],
        train_data['labels']
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = train_config.get('early_stopping', {}).get('patience', 15)

    edge_index = train_data['edge_index'].to(device)
    edge_weights = train_data['edge_weights'].to(device) if train_data['edge_weights'] is not None else None

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_emb, batch_struct, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_struct = batch_struct.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_emb, batch_struct, edge_index, edge_weights)

            # Loss
            loss = criterion(logits, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        scheduler.step()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_emb = val_data['embeddings'].to(device)
            val_struct = val_data['structural'].to(device)
            val_labels = val_data['labels'].to(device)

            val_logits = model(val_emb, val_struct, edge_index, edge_weights)
            val_loss = criterion(val_logits, val_labels).item()

            _, val_pred = val_logits.max(1)
            val_acc = val_pred.eq(val_labels).sum().item() / val_labels.size(0)

        model.train()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    return model


def evaluate_and_save(model, test_data, config, output_dir, device):
    """Evaluate model and save APFD results."""
    model.eval()

    with torch.no_grad():
        test_emb = test_data['embeddings'].to(device)
        test_struct = test_data['structural'].to(device)
        edge_index = test_data['edge_index'].to(device)
        edge_weights = test_data['edge_weights'].to(device) if test_data['edge_weights'] is not None else None

        logits = model(test_emb, test_struct, edge_index, edge_weights)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of Fail class

    # Create predictions DataFrame
    test_df = test_data['test_df'].copy()
    test_df = test_df[test_df['TC_Key'].isin(test_data['tc_keys'])]

    # Add probabilities
    tc_to_prob = dict(zip(test_data['tc_keys'], probs.cpu().numpy()))
    test_df['probability'] = test_df['TC_Key'].map(tc_to_prob)

    # Calculate ranks per build
    test_df['rank'] = test_df.groupby('Build_ID')['probability'].rank(method='first', ascending=False).astype(int)

    # Create label column
    test_df['label_binary'] = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)

    # Calculate APFD
    experiment_name = config.get('experiment', {}).get('name', 'ablation')
    apfd_df = calculate_apfd_per_build(
        test_df,
        method_name=experiment_name,
        build_col='Build_ID',
        label_col='label_binary',
        rank_col='rank',
        result_col='TE_Test_Result'
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_path = output_dir / 'prioritized_test_cases_FULL_testcsv.csv'
    test_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    # Save APFD
    apfd_path = output_dir / 'apfd_per_build_FULL_testcsv.csv'
    apfd_df.to_csv(apfd_path, index=False)
    logger.info(f"Saved APFD to {apfd_path}")

    # Summary
    mean_apfd = apfd_df['apfd'].mean()
    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION RESULTS: {experiment_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Mean APFD: {mean_apfd:.4f}")
    logger.info(f"Median APFD: {apfd_df['apfd'].median():.4f}")
    logger.info(f"Builds evaluated: {len(apfd_df)}")
    logger.info(f"{'='*70}")

    return mean_apfd, apfd_df


def main():
    parser = argparse.ArgumentParser(description='Run single ablation experiment')
    parser.add_argument('--config', type=str, help='Path to ablation config YAML')
    parser.add_argument('--ablation', type=str, choices=['A1', 'A2', 'A3', 'A4', 'A6'],
                       help='Ablation ID (A1-A6)')

    args = parser.parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    elif args.ablation:
        config_path = ABLATION_CONFIGS.get(args.ablation)
        if not config_path:
            logger.error(f"Unknown ablation: {args.ablation}")
            return
    else:
        logger.error("Must specify --config or --ablation")
        return

    config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load cached data
    logger.info("Loading cached data...")
    embeddings_data, structural_data, graph_data = load_cached_data(config)

    # Prepare datasets
    logger.info("Preparing datasets...")
    test_data = prepare_datasets(embeddings_data, structural_data, graph_data, config)

    # For training, use a portion of test data (since we don't have separate train/val)
    # In a real scenario, you'd load train.csv
    n_samples = len(test_data['embeddings'])
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_data = {
        'embeddings': test_data['embeddings'][train_idx],
        'structural': test_data['structural'][train_idx],
        'labels': test_data['labels'][train_idx],
        'edge_index': test_data['edge_index'],
        'edge_weights': test_data['edge_weights']
    }

    val_data = {
        'embeddings': test_data['embeddings'][val_idx],
        'structural': test_data['structural'][val_idx],
        'labels': test_data['labels'][val_idx]
    }

    # Keep full test data for evaluation
    # (In ablation, we evaluate on full test.csv)

    # Create model
    logger.info("Creating ablation model...")
    model = create_ablation_model(config['model'])
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    logger.info("Training...")
    model = train_ablation_model(model, train_data, val_data, config, device)

    # Evaluate
    logger.info("Evaluating...")
    output_dir = config.get('output', {}).get('results_dir', 'results/ablation/temp')
    mean_apfd, apfd_df = evaluate_and_save(model, test_data, config, output_dir, device)

    logger.info("\nAblation experiment completed!")
    logger.info(f"Results saved to: {output_dir}")

    return mean_apfd


if __name__ == "__main__":
    main()
