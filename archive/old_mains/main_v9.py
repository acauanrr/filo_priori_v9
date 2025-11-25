"""
Main Training Script for Filo-Priori V9

This script implements the V9 pipeline with:
- Qodo-Embed-1-1.5B model for embeddings
- Separate encoding for TCs and Commits
- Combined embedding dimension: 3072 (1536 * 2)

Usage:
    python main_v9.py --config configs/experiment_v9_qodo.yaml

Author: Filo-Priori V9 Team
Date: 2025-11-10
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import V9 modules
from preprocessing.data_loader import DataLoader
from preprocessing.commit_extractor import CommitExtractor
from preprocessing.structural_feature_extractor import extract_structural_features, StructuralFeatureExtractor
from preprocessing.structural_feature_imputation import impute_structural_features
from embeddings.qodo_encoder import QodoEncoder
from phylogenetic.phylogenetic_graph_builder import build_phylogenetic_graph
from models.dual_stream_v8 import create_model_v8  # We'll adapt this for V9
from training.losses import FocalLoss
from evaluation.metrics import compute_metrics
from evaluation.apfd import generate_apfd_report, print_apfd_summary, generate_prioritized_csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(config: Dict, sample_size: int = None) -> Tuple:
    """
    Prepare data for training with separate TC and Commit embeddings

    Args:
        config: Configuration dictionary
        sample_size: Optional sample size for testing

    Returns:
        Tuple of data components
    """
    logger.info("="*70)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*70)

    # Load data
    logger.info("\n1.1: Loading datasets...")
    data_loader = DataLoader(config)
    data_dict = data_loader.prepare_dataset(sample_size=sample_size)

    df_train = data_dict['train']
    df_val = data_dict['val']
    df_test = data_dict['test']

    logger.info(f"  Train: {len(df_train)} samples")
    logger.info(f"  Val: {len(df_val)} samples")
    logger.info(f"  Test: {len(df_test)} samples")

    # Compute class weights
    logger.info("\n1.1.1: Computing class weights...")
    class_weights = data_loader.compute_class_weights(df_train)
    logger.info(f"  Class weights: {class_weights}")
    logger.info(f"  Weight ratio (minority/majority): {class_weights.max() / class_weights.min():.2f}:1")

    # Extract commit texts (NEW in V9)
    logger.info("\n1.2: Extracting commit texts...")
    commit_config = config.get('commit', {})
    commit_extractor = CommitExtractor(commit_config)

    train_commits = commit_extractor.extract_from_dataframe(df_train, 'commit')
    val_commits = commit_extractor.extract_from_dataframe(df_val, 'commit')
    test_commits = commit_extractor.extract_from_dataframe(df_test, 'commit')

    # Extract semantic embeddings using Qodo-Embed (NEW in V9)
    logger.info("\n1.3: Extracting semantic embeddings with Qodo-Embed...")
    semantic_config = config['semantic']
    encoder = QodoEncoder(semantic_config)

    embedding_cache_path = semantic_config.get('cache_path') if sample_size is None else None

    # Encode train set
    logger.info("\n  Encoding training set...")
    train_tc_embeddings, train_commit_embeddings = encoder.encode_dataset_separate(
        summaries=df_train['TE_Summary'].fillna("").tolist(),
        steps=df_train['TC_Steps'].fillna("").tolist(),
        commit_texts=train_commits,
        cache_dir=embedding_cache_path,
        split_name='train'
    )

    # Encode validation set
    logger.info("\n  Encoding validation set...")
    val_tc_embeddings, val_commit_embeddings = encoder.encode_dataset_separate(
        summaries=df_val['TE_Summary'].fillna("").tolist(),
        steps=df_val['TC_Steps'].fillna("").tolist(),
        commit_texts=val_commits,
        cache_dir=embedding_cache_path,
        split_name='val'
    )

    # Encode test set
    logger.info("\n  Encoding test set...")
    test_tc_embeddings, test_commit_embeddings = encoder.encode_dataset_separate(
        summaries=df_test['TE_Summary'].fillna("").tolist(),
        steps=df_test['TC_Steps'].fillna("").tolist(),
        commit_texts=test_commits,
        cache_dir=embedding_cache_path,
        split_name='test'
    )

    # Concatenate TC and Commit embeddings
    logger.info("\n1.3.1: Concatenating TC and Commit embeddings...")
    train_embeddings = np.concatenate([train_tc_embeddings, train_commit_embeddings], axis=1)
    val_embeddings = np.concatenate([val_tc_embeddings, val_commit_embeddings], axis=1)
    test_embeddings = np.concatenate([test_tc_embeddings, test_commit_embeddings], axis=1)

    logger.info(f"  Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"  Val embeddings shape: {val_embeddings.shape}")
    logger.info(f"  Test embeddings shape: {test_embeddings.shape}")

    # Extract structural features
    logger.info("\n1.4: Extracting structural features...")
    structural_config = config['structural']['extractor']

    cache_path = structural_config.get('cache_path') if sample_size is None else None
    if sample_size is not None and structural_config.get('cache_path'):
        logger.info(f"  Note: Cache disabled for sample_size={sample_size}")

    logger.info("  Initializing StructuralFeatureExtractor...")
    extractor = StructuralFeatureExtractor(
        recent_window=structural_config['recent_window'],
        min_history=structural_config.get('min_history', 2),
        verbose=True
    )

    # Load or fit
    if cache_path and os.path.exists(cache_path):
        logger.info(f"  Loading cached extractor from {cache_path}")
        extractor.load_history(cache_path)
    else:
        logger.info("  Fitting extractor on training data...")
        extractor.fit(df_train)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            extractor.save_history(cache_path)

    # Transform splits
    logger.info("  Transforming training data...")
    train_struct = extractor.transform(df_train, is_test=False)

    logger.info("  Transforming validation data...")
    val_struct = extractor.transform(df_val, is_test=True)

    logger.info("  Transforming test data...")
    test_struct = extractor.transform(df_test, is_test=True)

    # Impute missing features
    logger.info("\n1.4.1: Imputing missing structural features...")
    tc_keys_train = df_train['TC_Key'].tolist()
    tc_keys_val = df_val['TC_Key'].tolist()
    tc_keys_test = df_test['TC_Key'].tolist()

    needs_imputation_val = extractor.get_imputation_mask(tc_keys_val)
    needs_imputation_test = extractor.get_imputation_mask(tc_keys_test)

    logger.info(f"  Validation samples needing imputation: {needs_imputation_val.sum()}/{len(tc_keys_val)}")
    logger.info(f"  Test samples needing imputation: {needs_imputation_test.sum()}/{len(tc_keys_test)}")

    if needs_imputation_val.sum() > 0:
        logger.info("  Imputing validation features...")
        val_struct, val_imputation_stats = impute_structural_features(
            train_embeddings, train_struct, tc_keys_train,
            val_embeddings, val_struct, tc_keys_val,
            extractor.tc_history,
            k_neighbors=10,
            similarity_threshold=0.5,
            verbose=False
        )

    if needs_imputation_test.sum() > 0:
        logger.info("  Imputing test features...")
        test_struct, test_imputation_stats = impute_structural_features(
            train_embeddings, train_struct, tc_keys_train,
            test_embeddings, test_struct, tc_keys_test,
            extractor.tc_history,
            k_neighbors=10,
            similarity_threshold=0.5,
            verbose=False
        )

    # Build phylogenetic graph
    logger.info("\n1.5: Building phylogenetic graph...")
    graph_config = config['graph']

    graph_builder, G = build_phylogenetic_graph(
        df_train,
        graph_type=graph_config['type'],
        cache_path=graph_config.get('cache_path'),
        min_co_occurrences=graph_config.get('min_co_occurrences', 2),
        weight_threshold=graph_config.get('weight_threshold', 0.1)
    )

    # Extract graph structure
    logger.info("\n1.6: Extracting graph structure (edge_index and edge_weights)...")
    edge_index, edge_weights = graph_builder.get_edge_index_and_weights()
    logger.info(f"Graph structure: {edge_index.shape[1]} edges among {graph_builder.num_test_cases} nodes")

    logger.info("\n✅ Data preparation complete!")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"Train structural features shape: {train_struct.shape}")
    logger.info(f"Train labels shape: {df_train['label'].values.shape}")

    return (
        (df_train, train_embeddings, train_struct),
        (df_val, val_embeddings, val_struct),
        (df_test, test_embeddings, test_struct),
        graph_builder, edge_index, edge_weights,
        class_weights, data_loader, encoder, extractor
    )


def main():
    parser = argparse.ArgumentParser(description='Train Filo-Priori V9 Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')

    args = parser.parse_args()

    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Set seed
    set_seed(config['experiment']['seed'])

    # Prepare data
    (train_data, val_data, test_data,
     graph_builder, edge_index, edge_weights,
     class_weights, data_loader, encoder, extractor) = prepare_data(config, sample_size=args.sample)

    df_train, train_embeddings, train_struct = train_data
    df_val, val_embeddings, val_struct = val_data
    df_test, test_embeddings, test_struct = test_data

    logger.info("\n" + "="*70)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("="*70)

    # NOTE: For now, we use the existing create_model_v8 function
    # The semantic input dimension is updated in the config to 3072
    model = create_model_v8(
        config=config,
        num_test_cases=graph_builder.num_test_cases,
        device=device
    ).to(device)

    logger.info("\n" + "="*70)
    logger.info("STEP 3: TRAINING")
    logger.info("="*70)
    logger.info("Training loop will be implemented here...")
    logger.info("For full implementation, integrate with existing training module from main_v8.py")

    logger.info("\n✅ V9 pipeline setup complete!")
    logger.info(f"\nEmbedding dimensions:")
    logger.info(f"  - TC embeddings: 1536")
    logger.info(f"  - Commit embeddings: 1536")
    logger.info(f"  - Combined: 3072")
    logger.info(f"\nModel ready for training with separate TC and Commit encodings!")


if __name__ == '__main__':
    main()
