"""
Main Training Script for Filo-Priori V8

This script implements the V8 pipeline with true structural features,
breaking the "Semantic Echo Chamber" of V7.

Usage:
    python main_v8.py --config configs/experiment_v8_baseline.yaml

Author: Filo-Priori V8 Team
Date: 2025-11-06
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

# Import V8 modules directly to avoid __init__.py issues
from preprocessing.data_loader import DataLoader
from preprocessing.text_processor import TextProcessor
from preprocessing.structural_feature_extractor import extract_structural_features, StructuralFeatureExtractor
from preprocessing.structural_feature_imputation import impute_structural_features
from embeddings.semantic_encoder import SemanticEncoder
from phylogenetic.phylogenetic_graph_builder import build_phylogenetic_graph
from models.dual_stream_v8 import create_model_v8
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
    Prepare data for training

    Args:
        config: Configuration dictionary
        sample_size: Optional sample size for testing

    Returns:
        Tuple of (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
                  class_weights, data_loader, encoder, text_processor, extractor)
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

    # Compute class weights for weighted cross-entropy loss
    logger.info("\n1.1.1: Computing class weights...")
    class_weights = data_loader.compute_class_weights(df_train)
    logger.info(f"  Class weights: {class_weights}")
    logger.info(f"  Weight ratio (minority/majority): {class_weights.max() / class_weights.min():.2f}:1")

    # Extract semantic embeddings (BGE)
    logger.info("\n1.2: Extracting semantic embeddings...")
    semantic_config = config['semantic']
    encoder = SemanticEncoder(semantic_config)

    # Process text
    text_processor = TextProcessor()

    train_texts = text_processor.prepare_batch_texts(
        df_train['TE_Summary'].tolist(),
        df_train['TC_Steps'].tolist(),
        df_train['commit'].tolist()
    )

    val_texts = text_processor.prepare_batch_texts(
        df_val['TE_Summary'].tolist(),
        df_val['TC_Steps'].tolist(),
        df_val['commit'].tolist()
    )

    test_texts = text_processor.prepare_batch_texts(
        df_test['TE_Summary'].tolist(),
        df_test['TC_Steps'].tolist(),
        df_test['commit'].tolist()
    )

    # Encode (disable cache if using sample_size)
    embedding_cache_path = semantic_config['cache_path'] if sample_size is None else None

    train_embeddings = encoder.encode_dataset(
        train_texts,
        cache_path=f"{embedding_cache_path}/train_embeddings.npy" if embedding_cache_path else None
    )

    val_embeddings = encoder.encode_dataset(
        val_texts,
        cache_path=f"{embedding_cache_path}/val_embeddings.npy" if embedding_cache_path else None
    )

    test_embeddings = encoder.encode_dataset(
        test_texts,
        cache_path=f"{embedding_cache_path}/test_embeddings.npy" if embedding_cache_path else None
    )

    # Extract structural features (NEW in V8!)
    logger.info("\n1.3: Extracting structural features...")
    structural_config = config['structural']['extractor']

    # Disable cache if using sample_size to avoid size mismatches
    cache_path = structural_config.get('cache_path') if sample_size is None else None
    if sample_size is not None and structural_config.get('cache_path'):
        logger.info(f"  Note: Cache disabled for sample_size={sample_size} to ensure correct shapes")

    # Create extractor manually to get access to tc_history for imputation
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

    # Impute missing features using semantic similarity
    logger.info("\n1.3b: Imputing missing structural features...")
    logger.info("  (Uses semantic similarity to estimate features for tests without history)")

    # Get TC_Keys for each split
    tc_keys_train = df_train['TC_Key'].tolist()
    tc_keys_val = df_val['TC_Key'].tolist()
    tc_keys_test = df_test['TC_Key'].tolist()

    # Check how many samples need imputation
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

    # Apply SMOTE if enabled
    if config['data'].get('smote', {}).get('enabled', False):
        logger.info("\n1.4: Applying SMOTE to balance training data...")
        try:
            from imblearn.over_sampling import SMOTE

            smote_config = config['data']['smote']
            sampling_strategy = smote_config.get('sampling_strategy', 'auto')
            k_neighbors = smote_config.get('k_neighbors', 5)

            logger.info(f"  SMOTE configuration:")
            logger.info(f"    Sampling strategy: {sampling_strategy}")
            logger.info(f"    K neighbors: {k_neighbors}")

            # Combine embeddings and structural features
            X_train = np.concatenate([train_embeddings, train_struct], axis=1)
            y_train = df_train['label'].values

            logger.info(f"  Before SMOTE: {len(y_train)} samples")
            logger.info(f"    Class distribution: {np.bincount(y_train)}")

            # Apply SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=config['experiment']['seed']
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            logger.info(f"  After SMOTE: {len(y_train_resampled)} samples")
            logger.info(f"    Class distribution: {np.bincount(y_train_resampled)}")

            # Split back into embeddings and structural features
            train_embeddings = X_train_resampled[:, :train_embeddings.shape[1]]
            train_struct = X_train_resampled[:, train_embeddings.shape[1]:]

            # Update df_train labels (note: TC_Keys will be duplicated)
            # We'll create a synthetic df by repeating rows
            original_len = len(df_train)
            n_synthetic = len(y_train_resampled) - original_len

            if n_synthetic > 0:
                logger.info(f"  Created {n_synthetic} synthetic samples")

                # Get SMOTE sample indices (original + synthetic)
                # Synthetic samples are appended after originals in SMOTE
                df_train_original = df_train.copy()

                # For synthetic samples, duplicate random original samples for metadata
                # This preserves TC_Key, Build_ID etc. (needed for graph building)
                np.random.seed(config['experiment']['seed'])
                synthetic_indices = np.random.choice(len(df_train_original), n_synthetic, replace=True)
                df_synthetic = df_train_original.iloc[synthetic_indices].copy()

                df_train = pd.concat([df_train_original, df_synthetic], ignore_index=True)
                df_train['label'] = y_train_resampled

            logger.info("✅ SMOTE applied successfully!")

        except ImportError:
            logger.error("❌ ERROR: imblearn not installed. Cannot apply SMOTE.")
            logger.error("   Install with: pip install imbalanced-learn")
            logger.error("   Continuing without SMOTE...")
        except Exception as e:
            logger.error(f"❌ ERROR applying SMOTE: {e}")
            logger.error("   Continuing without SMOTE...")

    # Build phylogenetic graph (optional)
    graph_builder = None
    if config['graph'].get('build_graph', True):
        logger.info("\n1.5: Building phylogenetic graph...")
        graph_config = config['graph']

        # Disable cache if using sample_size
        graph_cache_path = graph_config.get('cache_path') if sample_size is None else None

        graph_builder = build_phylogenetic_graph(
            df_train,
            graph_type=graph_config['type'],
            min_co_occurrences=graph_config['min_co_occurrences'],
            weight_threshold=graph_config['weight_threshold'],
            cache_path=graph_cache_path
        )

    logger.info("\n✓ Data preparation complete!")

    # Package data
    train_data = {
        'embeddings': train_embeddings,
        'structural_features': train_struct,
        'labels': df_train['label'].values,
        'df': df_train
    }

    val_data = {
        'embeddings': val_embeddings,
        'structural_features': val_struct,
        'labels': df_val['label'].values,
        'df': df_val
    }

    test_data = {
        'embeddings': test_embeddings,
        'structural_features': test_struct,
        'labels': df_test['label'].values,
        'df': df_test
    }

    # Extract edge_index and edge_weights for the ENTIRE training set
    # This graph structure is used for all batches during training
    logger.info("\n1.6: Extracting graph structure (edge_index and edge_weights)...")
    all_tc_keys = df_train['TC_Key'].unique().tolist()
    edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
        tc_keys=all_tc_keys,
        return_torch=True
    )
    logger.info(f"Graph structure: {edge_index.shape[1]} edges among {len(all_tc_keys)} nodes")

    return train_data, val_data, test_data, graph_builder, edge_index, edge_weights, class_weights, data_loader, encoder, text_processor, extractor


def create_dataloaders(train_data: Dict, val_data: Dict, test_data: Dict, batch_size: int):
    """Create PyTorch DataLoaders"""
    from torch.utils.data import TensorDataset, DataLoader

    # Debug: Print shapes
    logger.info(f"Train embeddings shape: {train_data['embeddings'].shape}")
    logger.info(f"Train structural features shape: {train_data['structural_features'].shape}")
    logger.info(f"Train labels shape: {train_data['labels'].shape}")

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['embeddings']),
        torch.FloatTensor(train_data['structural_features']),
        torch.LongTensor(train_data['labels'])
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(val_data['embeddings']),
        torch.FloatTensor(val_data['structural_features']),
        torch.LongTensor(val_data['labels'])
    )

    test_dataset = TensorDataset(
        torch.FloatTensor(test_data['embeddings']),
        torch.FloatTensor(test_data['structural_features']),
        torch.LongTensor(test_data['labels'])
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, edge_index, edge_weights, all_structural_features):
    """
    Train for one epoch

    Args:
        all_structural_features: All structural features [N_total, 6] for full-graph GAT processing
    """
    model.train()
    total_loss = 0.0

    all_structural_features_device = all_structural_features.to(device)

    batch_start_idx = 0
    for embeddings, structural_features, labels in loader:
        batch_size = embeddings.size(0)
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Process structural stream with GAT (full-graph) for THIS batch
        # We need to recompute for each batch to maintain gradient flow
        structural_embeddings = model.structural_stream(
            all_structural_features_device,
            edge_index,
            edge_weights
        )  # [N_total, 256]

        # Select structural embeddings for current batch
        batch_structural = structural_embeddings[batch_start_idx:batch_start_idx + batch_size]

        # Process semantic stream
        semantic_features = model.semantic_stream(embeddings)

        # Fuse and classify
        fused_features = model.fusion(semantic_features, batch_structural)
        logits = model.classifier(fused_features)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_start_idx += batch_size

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, edge_index, edge_weights, all_structural_features):
    """
    Evaluate model

    Args:
        all_structural_features: All structural features [N_total, 6] for full-graph GAT processing
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    # Process ALL structural features once with GAT (full-graph)
    all_structural_features_device = all_structural_features.to(device)

    # Get structural embeddings for ALL nodes
    structural_embeddings = model.structural_stream(
        all_structural_features_device,
        edge_index,
        edge_weights
    )  # [N_total, 256]

    batch_start_idx = 0
    for embeddings, structural_features, labels in loader:
        batch_size = embeddings.size(0)
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Select structural embeddings for current batch
        batch_structural = structural_embeddings[batch_start_idx:batch_start_idx + batch_size]

        # Process semantic stream
        semantic_features = model.semantic_stream(embeddings)

        # Fuse and classify
        fused_features = model.fusion(semantic_features, batch_structural)
        logits = model.classifier(fused_features)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        batch_start_idx += batch_size

    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(
        predictions=all_preds,
        labels=all_labels,
        num_classes=2,
        label_names=['Not-Pass', 'Pass'],
        probabilities=all_probs
    )

    return avg_loss, metrics, all_probs


def main():
    parser = argparse.ArgumentParser(description='Train Filo-Priori V8 Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size for quick testing')
    args = parser.parse_args()

    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Set seed
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare data
    (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
     class_weights, data_loader, encoder, text_processor, extractor) = prepare_data(config, args.sample_size)

    # Extract train data for STEP 6 imputation
    train_embeddings = train_data['embeddings']
    train_struct = train_data['structural_features']
    tc_keys_train = train_data['df']['TC_Key'].tolist()

    # Move graph structure to device
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    # Extract structural features for GAT processing
    train_structural_features = torch.FloatTensor(train_data['structural_features'])
    val_structural_features = torch.FloatTensor(val_data['structural_features'])
    test_structural_features = torch.FloatTensor(test_data['structural_features'])

    # Create data loaders
    logger.info("\nCreating data loaders...")
    batch_size = config['training']['batch_size']
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size
    )

    # Create model
    logger.info("\n"+"="*70)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("="*70)

    model = create_model_v8(config['model']).to(device)

    # Loss function
    logger.info("\nInitializing loss function...")
    if config['loss']['type'] == 'focal':
        criterion = FocalLoss(
            alpha=config['loss']['focal']['alpha'],
            gamma=config['loss']['focal']['gamma']
        ).to(device)
        logger.info(f"  Using Focal Loss with alpha={config['loss']['focal']['alpha']}, gamma={config['loss']['focal']['gamma']}")
    elif config['loss']['type'] == 'weighted_ce':
        # Use class weights: custom or computed from training data
        wce_config = config['loss']['weighted_ce']

        if wce_config.get('use_class_weights', True):
            # Use automatically computed class weights
            weights_to_use = class_weights
            logger.info(f"  Using AUTO-COMPUTED class weights: {weights_to_use}")
        else:
            # Use custom class weights from config
            weights_to_use = np.array(wce_config['class_weights'])
            logger.info(f"  Using CUSTOM class weights: {weights_to_use}")

        class_weights_tensor = torch.FloatTensor(weights_to_use).to(device)

        # Get label smoothing parameter (default 0.0)
        label_smoothing = float(wce_config.get('label_smoothing', 0.0))

        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=label_smoothing
        ).to(device)

        logger.info(f"  Using Weighted Cross-Entropy Loss:")
        logger.info(f"    Class weights: {weights_to_use}")
        logger.info(f"    Weight ratio (minority/majority): {weights_to_use.max() / weights_to_use.min():.2f}:1")
        logger.info(f"    Label smoothing: {label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        logger.info("  Using standard Cross-Entropy Loss")

    # Optimizer
    logger.info("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=float(config['training']['scheduler']['eta_min'])
    )

    # Training loop
    logger.info("\n"+"="*70)
    logger.info("STEP 3: TRAINING")
    logger.info("="*70)

    best_val_f1 = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']

    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, edge_index, edge_weights, train_structural_features)

        # Validate
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, edge_index, edge_weights, val_structural_features)

        # Update scheduler
        scheduler.step()

        # Log
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Val F1={val_metrics['f1_macro']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}"
        )

        # Early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), 'best_model_v8.pt')
            logger.info(f"  → New best model saved! (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    logger.info("\nLoading best model...")
    model.load_state_dict(torch.load('best_model_v8.pt'))

    # Test evaluation
    logger.info("\n"+"="*70)
    logger.info("STEP 4: TEST EVALUATION")
    logger.info("="*70)

    test_loss, test_metrics, test_probs = evaluate(model, test_loader, criterion, device, edge_index, edge_weights, test_structural_features)

    logger.info("\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  AUPRC (Macro): {test_metrics.get('auprc_macro', 0.0):.4f}")

    # APFD calculation
    logger.info("\n"+"="*70)
    logger.info("STEP 5: APFD CALCULATION")
    logger.info("="*70)

    # Add probabilities to test DataFrame
    test_df = test_data['df'].copy()
    test_df['probability'] = test_probs[:, 0]  # P(Fail) - class 0 with pass_vs_fail

    # CRITICAL: Use TE_Test_Result from original CSV for correct APFD
    if 'TE_Test_Result' not in test_df.columns:
        logger.error("❌ CRITICAL: TE_Test_Result column not found in test DataFrame!")
        logger.error("   This column is required for correct APFD calculation.")
        logger.error("   APFD should only count builds with TE_Test_Result == 'Fail'")
        logger.error("   Check if data_loader is preserving this column from test.csv")
        # Fallback: create from pass_vs_fail labels (not ideal but better than nothing)
        logger.warning("   Using fallback: mapping labels to TE_Test_Result")
        test_df['TE_Test_Result'] = test_data['labels'].map({0: 'Fail', 1: 'Pass'})
    else:
        logger.info(f"✅ TE_Test_Result column found with {len(test_df['TE_Test_Result'].unique())} unique values")
        logger.info(f"   Values: {test_df['TE_Test_Result'].value_counts().to_dict()}")

    # Create label_binary from TE_Test_Result (not from processed labels)
    test_df['label_binary'] = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
    logger.info(f"   label_binary distribution: {test_df['label_binary'].value_counts().to_dict()}")

    # Verify Build_ID exists
    if 'Build_ID' not in test_df.columns:
        logger.error("❌ CRITICAL: Build_ID column not found!")
        logger.error("   Cannot calculate APFD per build.")
    else:
        logger.info(f"✅ Build_ID column found: {test_df['Build_ID'].nunique()} unique builds")

    # Get results directory from config
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Generate prioritized CSV with ranks per build
    prioritized_path = os.path.join(results_dir, 'prioritized_test_cases.csv')
    test_df_with_ranks = generate_prioritized_csv(
        test_df,
        output_path=prioritized_path,
        probability_col='probability',
        label_col='label_binary',
        build_col='Build_ID'
    )
    logger.info(f"✅ Prioritized test cases saved to: {prioritized_path}")

    # Calculate APFD per build
    apfd_path = os.path.join(results_dir, 'apfd_per_build.csv')
    apfd_results_df, apfd_summary = generate_apfd_report(
        test_df_with_ranks,
        method_name=config['experiment']['name'],
        test_scenario="v8_full_test",
        output_path=apfd_path
    )

    # Print summary
    print_apfd_summary(apfd_summary)

    # Log results
    if apfd_summary:
        logger.info(f"\n✅ APFD per-build report saved to: {apfd_path}")
        logger.info(f"📊 Mean APFD: {apfd_summary['mean_apfd']:.4f} (across {apfd_summary['total_builds']} builds)")

        # Verify expected 277 builds
        if apfd_summary['total_builds'] != 277:
            logger.warning(f"⚠️  WARNING: Expected 277 builds but got {apfd_summary['total_builds']}")
            logger.warning(f"   This may indicate incorrect filtering or data issues")
    else:
        logger.warning("⚠️  No builds with failures found - APFD cannot be calculated")

    # ==============================================================================
    # STEP 6: PROCESS FULL TEST.CSV (277 BUILDS) FOR FINAL APFD CALCULATION
    # ==============================================================================

    logger.info("\n"+"="*70)
    logger.info("STEP 6: PROCESSING FULL TEST.CSV FOR FINAL APFD")
    logger.info("="*70)

    try:
        # Load FULL test dataset (test.csv)
        logger.info("\n6.1: Loading FULL test.csv...")
        test_df_full = data_loader.load_full_test_dataset()

        logger.info(f"✅ Loaded full test.csv:")
        logger.info(f"   Total samples: {len(test_df_full)}")
        logger.info(f"   Total builds: {test_df_full['Build_ID'].nunique()}")

        builds_with_fail = test_df_full[test_df_full['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
        logger.info(f"   Builds with 'Fail': {builds_with_fail}")

        if builds_with_fail != 277:
            logger.warning(f"⚠️  WARNING: Expected 277 builds but found {builds_with_fail}")

        # Generate embeddings for full test set
        logger.info("\n6.2: Generating semantic embeddings for full test set...")

        # Prepare texts
        test_texts_full = text_processor.prepare_batch_texts(
            summaries=test_df_full['TE_Summary'].tolist(),
            steps=test_df_full['TC_Steps'].tolist(),
            commits=test_df_full['commit_processed'].tolist()
        )

        # Generate embeddings
        test_embeddings_full = encoder.encode_texts(test_texts_full)
        logger.info(f"✅ Generated embeddings: {test_embeddings_full.shape}")

        # Extract structural features for full test set
        logger.info("\n6.3: Extracting structural features for full test set...")

        # Use the already fitted extractor
        test_struct_full = extractor.transform(test_df_full, is_test=True)
        logger.info(f"✅ Extracted structural features: {test_struct_full.shape}")

        # Impute if needed
        tc_keys_test_full = test_df_full['TC_Key'].tolist()
        needs_imputation_full = extractor.get_imputation_mask(tc_keys_test_full)

        logger.info(f"   Samples needing imputation: {needs_imputation_full.sum()}/{len(tc_keys_test_full)}")

        if needs_imputation_full.sum() > 0:
            logger.info("   Imputing features...")
            test_struct_full, _ = impute_structural_features(
                train_embeddings, train_struct, tc_keys_train,
                test_embeddings_full, test_struct_full, tc_keys_test_full,
                extractor.tc_history,
                k_neighbors=10,
                similarity_threshold=0.5,
                verbose=False
            )

        # Generate predictions on full test set
        logger.info("\n6.4: Generating predictions on full test set...")

        # Convert to tensors
        test_embeddings_full_tensor = torch.from_numpy(test_embeddings_full).to(device)
        test_struct_full_tensor = torch.from_numpy(test_struct_full).to(device)

        # Use DataLoader for batching
        test_dataset_full = torch.utils.data.TensorDataset(
            test_embeddings_full_tensor,
            test_struct_full_tensor,
            torch.zeros(len(test_embeddings_full_tensor))  # Dummy labels
        )

        test_loader_full = torch.utils.data.DataLoader(
            test_dataset_full,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

        # Predict
        model.eval()
        all_probs_full = []

        # Get structural embeddings for ALL nodes once
        all_structural_features_full = test_struct_full_tensor
        structural_embeddings_full = model.structural_stream(
            all_structural_features_full,
            edge_index,
            edge_weights
        )

        batch_start_idx = 0
        with torch.no_grad():
            for embeddings_batch, _, _ in test_loader_full:
                batch_size = embeddings_batch.size(0)

                # Select structural embeddings for current batch
                batch_structural = structural_embeddings_full[batch_start_idx:batch_start_idx + batch_size]

                # Process semantic stream
                semantic_features = model.semantic_stream(embeddings_batch)

                # Fuse and classify
                fused_features = model.fusion(semantic_features, batch_structural)
                logits = model.classifier(fused_features)

                # Get probabilities
                probs = torch.softmax(logits, dim=1)
                all_probs_full.extend(probs.cpu().numpy())

                batch_start_idx += batch_size

        all_probs_full = np.array(all_probs_full)
        logger.info(f"✅ Predictions generated: {all_probs_full.shape}")

        # Prepare DataFrame for APFD
        logger.info("\n6.5: Preparing data for APFD calculation...")

        # P(Fail) = probabilities[:, 0] (class 0 with pass_vs_fail)
        failure_probs_full = all_probs_full[:, 0]
        test_df_full['probability'] = failure_probs_full

        # CRITICAL: Use TE_Test_Result for APFD
        test_df_full['label_binary'] = (test_df_full['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)

        logger.info(f"   Failures (TE_Test_Result=='Fail'): {test_df_full['label_binary'].sum()}")
        logger.info(f"   Passes: {(test_df_full['label_binary'] == 0).sum()}")

        # Generate prioritized CSV with ranks per build
        logger.info("\n6.6: Generating prioritized test cases CSV...")

        prioritized_path_full = os.path.join(results_dir, 'prioritized_test_cases_FULL_testcsv.csv')
        test_df_full_with_ranks = generate_prioritized_csv(
            test_df_full,
            output_path=prioritized_path_full,
            probability_col='probability',
            label_col='label_binary',
            build_col='Build_ID'
        )
        logger.info(f"✅ Prioritized test cases (FULL) saved to: {prioritized_path_full}")

        # Calculate APFD per build
        logger.info("\n6.7: Calculating APFD per build on FULL test.csv...")

        apfd_path_full = os.path.join(results_dir, 'apfd_per_build_FULL_testcsv.csv')
        method_name_full = f"{config['experiment']['name']}_FULL_testcsv"

        apfd_results_df_full, apfd_summary_full = generate_apfd_report(
            test_df_full_with_ranks,
            method_name=method_name_full,
            test_scenario="full_test_csv_277_builds",
            output_path=apfd_path_full
        )

        # Print APFD summary
        logger.info("\n" + "="*70)
        logger.info("FINAL APFD RESULTS - FULL TEST.CSV (277 BUILDS)")
        logger.info("="*70)
        print_apfd_summary(apfd_summary_full)

        # Validation
        logger.info("\n" + "="*70)
        logger.info("VALIDATION")
        logger.info("="*70)

        if apfd_summary_full and apfd_summary_full['total_builds'] == 277:
            logger.info("✅ SUCCESS: Found exactly 277 builds with failures!")
            logger.info(f"✅ Mean APFD: {apfd_summary_full['mean_apfd']:.4f}")
        else:
            builds_found = apfd_summary_full['total_builds'] if apfd_summary_full else 0
            logger.warning(f"⚠️  WARNING: Expected 277 builds but found {builds_found}")

        logger.info(f"\n✅ All results saved to: {results_dir}/")
        logger.info(f"   - prioritized_test_cases.csv (test split)")
        logger.info(f"   - apfd_per_build.csv (test split)")
        logger.info(f"   - prioritized_test_cases_FULL_testcsv.csv (all 277 builds)")
        logger.info(f"   - apfd_per_build_FULL_testcsv.csv (all 277 builds)")

    except Exception as e:
        logger.error(f"\n❌ ERROR processing full test.csv: {e}")
        logger.error("   Continuing with split test results only...")
        import traceback
        traceback.print_exc()

    # ==============================================================================
    # TRAINING COMPLETE
    # ==============================================================================

    logger.info("\n"+"="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Test F1: {test_metrics['f1_macro']:.4f}")

    if apfd_summary:
        logger.info(f"Mean APFD (test split): {apfd_summary.get('mean_apfd', 0.0):.4f}")

    try:
        if apfd_summary_full:
            logger.info(f"Mean APFD (FULL test.csv, 277 builds): {apfd_summary_full.get('mean_apfd', 0.0):.4f}")
    except:
        pass


if __name__ == '__main__':
    main()
