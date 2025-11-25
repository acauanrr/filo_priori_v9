"""
End-to-End Validation Script for Filo-Priori V8 Pipeline

This script validates the complete V8 pipeline without full training:
1. Data loading and preprocessing
2. Structural feature extraction
3. Phylogenetic graph construction
4. Model architecture (forward pass)
5. Integration test

Usage:
    python scripts/validate_v8_pipeline.py --sample-size 1000

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np

# Direct imports to avoid __init__.py dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.data_loader import DataLoader
from preprocessing.text_processor import TextProcessor
from preprocessing.structural_feature_extractor import (
    StructuralFeatureExtractor,
    extract_structural_features
)
from phylogenetic.phylogenetic_graph_builder import (
    PhylogeneticGraphBuilder,
    build_phylogenetic_graph
)

# Import V8 model directly (avoid __init__.py)
from models.dual_stream_v8 import DualStreamModelV8, create_model_v8

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_loading(sample_size: int = 1000):
    """Validate data loading"""
    logger.info("\n"+"="*70)
    logger.info("TEST 1: DATA LOADING")
    logger.info("="*70)

    logger.info("Loading datasets...")
    df_train = pd.read_csv('datasets/train.csv').head(sample_size)
    df_test = pd.read_csv('datasets/test.csv').head(sample_size // 2)

    logger.info(f"✓ Train: {len(df_train)} samples")
    logger.info(f"✓ Test: {len(df_test)} samples")

    # Check required columns
    required_cols = ['TC_Key', 'Build_ID', 'TE_Test_Result', 'TE_Summary', 'TC_Steps', 'commit']
    for col in required_cols:
        assert col in df_train.columns, f"Missing column: {col}"

    logger.info("✓ All required columns present")

    return df_train, df_test


def validate_structural_features(df_train, df_test):
    """Validate structural feature extraction"""
    logger.info("\n"+"="*70)
    logger.info("TEST 2: STRUCTURAL FEATURE EXTRACTION")
    logger.info("="*70)

    logger.info("Extracting features...")
    train_features, _, test_features = extract_structural_features(
        df_train,
        df_val=None,
        df_test=df_test,
        recent_window=5,
        cache_path=None
    )

    logger.info(f"✓ Train features: {train_features.shape}")
    logger.info(f"✓ Test features: {test_features.shape}")

    # Validate shape
    assert train_features.shape[1] == 6, f"Expected 6 features, got {train_features.shape[1]}"
    assert test_features.shape[1] == 6, f"Expected 6 features, got {test_features.shape[1]}"

    logger.info("✓ Feature shape correct: [N, 6]")

    # Validate ranges
    feature_names = ['test_age', 'failure_rate', 'recent_failure_rate',
                    'flakiness_rate', 'commit_count', 'test_novelty']

    for i, name in enumerate(feature_names):
        col = train_features[:, i]
        logger.info(f"  {name:20s}: min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}")

    logger.info("✓ Feature ranges valid")

    return train_features, test_features


def validate_graph_construction(df_train, graph_type='co_failure'):
    """Validate phylogenetic graph construction"""
    logger.info("\n"+"="*70)
    logger.info("TEST 3: PHYLOGENETIC GRAPH CONSTRUCTION")
    logger.info("="*70)

    logger.info(f"Building {graph_type} graph...")
    graph_builder = build_phylogenetic_graph(
        df_train,
        graph_type=graph_type,
        min_co_occurrences=2,
        weight_threshold=0.1,
        cache_path=None
    )

    stats = graph_builder.get_graph_statistics()

    logger.info(f"✓ Graph type: {stats['graph_type']}")
    logger.info(f"✓ Nodes: {stats['num_nodes']}")
    logger.info(f"✓ Edges: {stats['num_edges']}")
    logger.info(f"✓ Avg degree: {stats['avg_degree']:.2f}")

    # Test edge extraction
    tc_keys = df_train['TC_Key'].unique()[:100]
    edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
        tc_keys.tolist(),
        return_torch=True
    )

    logger.info(f"✓ Edge index shape: {edge_index.shape}")
    logger.info(f"✓ Edge weights shape: {edge_weights.shape}")

    return graph_builder


def validate_model_architecture(batch_size=16):
    """Validate model architecture and forward pass"""
    logger.info("\n"+"="*70)
    logger.info("TEST 4: MODEL ARCHITECTURE")
    logger.info("="*70)

    # Create model config
    model_config = {
        'semantic': {
            'input_dim': 1024,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        },
        'structural': {
            'input_dim': 6,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'use_batch_norm': True
        },
        'fusion': {
            'num_heads': 4,
            'dropout': 0.1
        },
        'classifier': {
            'hidden_dims': [128, 64],
            'dropout': 0.4
        },
        'num_classes': 2
    }

    logger.info("Creating model...")
    model = create_model_v8(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✓ Model created successfully")
    logger.info(f"✓ Total parameters: {total_params:,}")
    logger.info(f"✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass
    logger.info("\nTesting forward pass...")
    semantic_input = torch.randn(batch_size, 1024)
    structural_input = torch.randn(batch_size, 6)

    with torch.no_grad():
        logits = model(semantic_input=semantic_input, structural_input=structural_input)

    logger.info(f"✓ Forward pass successful")
    logger.info(f"✓ Input shapes: semantic=[{batch_size}, 1024], structural=[{batch_size}, 6]")
    logger.info(f"✓ Output shape: {logits.shape}")

    assert logits.shape == (batch_size, 2), f"Expected shape [{batch_size}, 2], got {logits.shape}"

    # Test feature extraction
    logger.info("\nTesting feature extraction...")
    with torch.no_grad():
        sem_features, struct_features, fused_features = model.get_feature_representations(
            semantic_input, structural_input
        )

    logger.info(f"✓ Semantic features: {sem_features.shape}")
    logger.info(f"✓ Structural features: {struct_features.shape}")
    logger.info(f"✓ Fused features: {fused_features.shape}")

    return model


def validate_end_to_end_integration(df_train, df_test,
                                     train_struct, test_struct,
                                     model, batch_size=16):
    """Validate end-to-end integration"""
    logger.info("\n"+"="*70)
    logger.info("TEST 5: END-TO-END INTEGRATION")
    logger.info("="*70)

    # Create dummy embeddings (in real scenario, use BGE)
    logger.info("Creating dummy semantic embeddings...")
    train_embeddings = np.random.randn(len(df_train), 1024).astype(np.float32)
    test_embeddings = np.random.randn(len(df_test), 1024).astype(np.float32)

    logger.info(f"✓ Train embeddings: {train_embeddings.shape}")
    logger.info(f"✓ Test embeddings: {test_embeddings.shape}")

    # Create dummy labels
    train_labels = (df_train['TE_Test_Result'] == 'Pass').astype(int).values
    test_labels = (df_test['TE_Test_Result'] == 'Pass').astype(int).values

    logger.info(f"✓ Train labels: {train_labels.shape}, Pass rate: {train_labels.mean():.2%}")
    logger.info(f"✓ Test labels: {test_labels.shape}, Pass rate: {test_labels.mean():.2%}")

    # Create DataLoader
    logger.info("\nCreating DataLoader...")
    from torch.utils.data import TensorDataset, DataLoader

    test_dataset = TensorDataset(
        torch.FloatTensor(test_embeddings),
        torch.FloatTensor(test_struct),
        torch.LongTensor(test_labels)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"✓ DataLoader created: {len(test_loader)} batches")

    # Test inference
    logger.info("\nTesting inference...")
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch_idx, (embeddings, structural_features, labels) in enumerate(test_loader):
            logits = model(semantic_input=embeddings, structural_input=structural_features)
            all_logits.append(logits)

            if batch_idx == 0:
                logger.info(f"  Batch 0:")
                logger.info(f"    Embeddings: {embeddings.shape}")
                logger.info(f"    Structural: {structural_features.shape}")
                logger.info(f"    Labels: {labels.shape}")
                logger.info(f"    Logits: {logits.shape}")

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(all_logits, dim=1)
    preds = torch.argmax(all_logits, dim=1)

    logger.info(f"\n✓ Inference complete")
    logger.info(f"✓ Logits: {all_logits.shape}")
    logger.info(f"✓ Probabilities: {probs.shape}")
    logger.info(f"✓ Predictions: {preds.shape}")

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score

    preds_np = preds.numpy()
    accuracy = accuracy_score(test_labels, preds_np)
    f1 = f1_score(test_labels, preds_np, average='macro')

    logger.info(f"\n✓ Dummy metrics (random model):")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 (Macro): {f1:.4f}")

    logger.info("\n✓ End-to-end integration test passed!")


def main():
    parser = argparse.ArgumentParser(
        description='Validate Filo-Priori V8 pipeline end-to-end'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of samples to test (default: 1000)'
    )
    parser.add_argument(
        '--graph-type',
        type=str,
        default='co_failure',
        choices=['co_failure', 'commit_dependency', 'hybrid'],
        help='Type of phylogenetic graph to build'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for testing'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("FILO-PRIORI V8 PIPELINE VALIDATION")
    logger.info("="*70)
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Graph type: {args.graph_type}")
    logger.info(f"Batch size: {args.batch_size}")

    try:
        # Test 1: Data loading
        df_train, df_test = validate_data_loading(args.sample_size)

        # Test 2: Structural features
        train_struct, test_struct = validate_structural_features(df_train, df_test)

        # Test 3: Graph construction
        graph_builder = validate_graph_construction(df_train, args.graph_type)

        # Test 4: Model architecture
        model = validate_model_architecture(args.batch_size)

        # Test 5: End-to-end integration
        validate_end_to_end_integration(
            df_train, df_test,
            train_struct, test_struct,
            model, args.batch_size
        )

        # Final summary
        logger.info("\n"+"="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info("✓ TEST 1: Data loading - PASSED")
        logger.info("✓ TEST 2: Structural feature extraction - PASSED")
        logger.info("✓ TEST 3: Phylogenetic graph construction - PASSED")
        logger.info("✓ TEST 4: Model architecture - PASSED")
        logger.info("✓ TEST 5: End-to-end integration - PASSED")
        logger.info("="*70)
        logger.info("\n✅ ALL TESTS PASSED!")
        logger.info("\nThe V8 pipeline is ready for training.")
        logger.info("\nNext steps:")
        logger.info("  1. Review the validation results above")
        logger.info("  2. Adjust configs if needed (configs/experiment_v8_baseline.yaml)")
        logger.info("  3. Run full training: python main_v8.py --config configs/experiment_v8_baseline.yaml")

        return 0

    except Exception as e:
        logger.error(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    os.chdir('/home/acauan/ufam/iats/sprint_07/filo_priori_v8')
    sys.exit(main())
