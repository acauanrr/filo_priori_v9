#!/usr/bin/env python3
"""
Filo-Priori Learning-to-Rank Pipeline for RTPTorrent.

This script implements the complete pipeline for test case prioritization
using learning-to-rank on the RTPTorrent benchmark dataset.

Key differences from main.py (classification):
1. Uses listwise/pairwise ranking losses instead of classification
2. Evaluates using APFD instead of F1/Accuracy
3. Compares against 7 baseline strategies from RTPTorrent
4. Processes data by build for ranking evaluation

Usage:
    python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml

    # With preprocessing:
    python scripts/preprocessing/preprocess_rtptorrent_ranking.py --preset small
    python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.sbert_encoder import SBERTEncoder
from src.training.ranking_losses import create_ranking_loss
from src.evaluation.rtptorrent_evaluator import RTPTorrentEvaluator, evaluate_rtptorrent_experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class RTPTorrentRankingDataset(Dataset):
    """
    Dataset for learning-to-rank on RTPTorrent.

    Each sample is a build containing multiple test cases.
    The model learns to rank tests within each build.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        feature_columns: List[str],
        max_tests_per_build: int = 100
    ):
        """
        Args:
            df: DataFrame with columns including Build_ID, TC_Key, is_failure, features
            embeddings: Semantic embeddings array [n_samples, embed_dim]
            feature_columns: List of structural feature column names
            max_tests_per_build: Maximum tests to include per build
        """
        self.df = df
        self.embeddings = embeddings
        self.feature_columns = feature_columns
        self.max_tests = max_tests_per_build

        # Group by build
        self.builds = df['Build_ID'].unique().tolist()
        self.build_indices = {
            build: df[df['Build_ID'] == build].index.tolist()
            for build in self.builds
        }

        logger.info(f"Dataset: {len(self.builds)} builds, {len(df)} total samples")

    def __len__(self):
        return len(self.builds)

    def __getitem__(self, idx):
        build_id = self.builds[idx]
        indices = self.build_indices[build_id]

        # Truncate if too many tests
        if len(indices) > self.max_tests:
            indices = indices[:self.max_tests]

        # Get data for this build
        build_df = self.df.loc[indices]

        # Semantic embeddings
        semantic = self.embeddings[indices]

        # Structural features
        structural = build_df[self.feature_columns].values

        # Relevance labels (1 for failure, 0 for pass)
        relevance = build_df['is_failure'].values

        # Create mask for padding
        n_tests = len(indices)
        mask = np.ones(n_tests, dtype=bool)

        return {
            'build_id': build_id,
            'semantic': torch.tensor(semantic, dtype=torch.float32),
            'structural': torch.tensor(structural, dtype=torch.float32),
            'relevance': torch.tensor(relevance, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'n_tests': n_tests,
            'indices': indices
        }


def collate_ranking_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length build lists.

    Pads all builds to the maximum size in the batch.
    """
    max_tests = max(item['n_tests'] for item in batch)
    batch_size = len(batch)

    # Get dimensions
    semantic_dim = batch[0]['semantic'].shape[1]
    structural_dim = batch[0]['structural'].shape[1]

    # Initialize padded tensors
    semantic = torch.zeros(batch_size, max_tests, semantic_dim)
    structural = torch.zeros(batch_size, max_tests, structural_dim)
    relevance = torch.zeros(batch_size, max_tests)
    mask = torch.zeros(batch_size, max_tests, dtype=torch.bool)

    build_ids = []
    all_indices = []

    for i, item in enumerate(batch):
        n = item['n_tests']
        semantic[i, :n] = item['semantic']
        structural[i, :n] = item['structural']
        relevance[i, :n] = item['relevance']
        mask[i, :n] = item['mask']
        build_ids.append(item['build_id'])
        all_indices.append(item['indices'])

    return {
        'semantic': semantic,
        'structural': structural,
        'relevance': relevance,
        'mask': mask,
        'build_ids': build_ids,
        'indices': all_indices
    }


# =============================================================================
# MODEL
# =============================================================================

class RankingModel(nn.Module):
    """
    Learning-to-Rank model for TCP.

    Architecture:
    - Semantic stream: MLP on SBERT embeddings
    - Structural stream: MLP on historical features
    - Fusion: Concatenation + MLP
    - Output: Single score per test case
    """

    def __init__(
        self,
        semantic_dim: int = 768,
        structural_dim: int = 9,
        semantic_hidden: int = 128,
        structural_hidden: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.3,
        structural_weight: float = 1.0
    ):
        super().__init__()

        self.structural_weight = structural_weight

        # Semantic stream
        self.semantic_stream = nn.Sequential(
            nn.Linear(semantic_dim, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden, semantic_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Structural stream (more important for RTPTorrent)
        self.structural_stream = nn.Sequential(
            nn.Linear(structural_dim, structural_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for structural
            nn.Linear(structural_hidden, structural_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Fusion
        fusion_input = semantic_hidden + structural_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output score
        self.scorer = nn.Sequential(
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        semantic: torch.Tensor,
        structural: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            semantic: [batch, n_tests, semantic_dim]
            structural: [batch, n_tests, structural_dim]
            mask: [batch, n_tests] - True for valid positions

        Returns:
            scores: [batch, n_tests] - Ranking scores
        """
        batch_size, n_tests, _ = semantic.shape

        # Reshape for MLP
        semantic_flat = semantic.view(-1, semantic.shape[-1])
        structural_flat = structural.view(-1, structural.shape[-1])

        # Process streams
        sem_out = self.semantic_stream(semantic_flat)
        struct_out = self.structural_stream(structural_flat)

        # Apply structural weight
        struct_out = struct_out * self.structural_weight

        # Concatenate and fuse
        combined = torch.cat([sem_out, struct_out], dim=-1)
        fused = self.fusion(combined)

        # Get scores
        scores = self.scorer(fused).squeeze(-1)

        # Reshape back
        scores = scores.view(batch_size, n_tests)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        return scores


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        semantic = batch['semantic'].to(device)
        structural = batch['structural'].to(device)
        relevance = batch['relevance'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        scores = model(semantic, structural, mask)

        # Compute loss
        loss = loss_fn(scores, relevance, mask)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_apfd(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, List[Dict]]:
    """
    Evaluate model using APFD metric.

    Returns:
        mean_apfd: Mean APFD across all builds
        per_build_results: List of per-build results
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            semantic = batch['semantic'].to(device)
            structural = batch['structural'].to(device)
            relevance = batch['relevance'].to(device)
            mask = batch['mask'].to(device)

            # Get scores
            scores = model(semantic, structural, mask)

            # Process each build in batch
            batch_size = scores.shape[0]
            for i in range(batch_size):
                build_scores = scores[i][mask[i]].cpu().numpy()
                build_relevance = relevance[i][mask[i]].cpu().numpy()

                # Sort by scores (descending)
                ranking = np.argsort(-build_scores)
                sorted_relevance = build_relevance[ranking]

                # Find failure positions
                failure_positions = np.where(sorted_relevance == 1)[0] + 1  # 1-indexed

                n_tests = len(build_scores)
                n_failures = len(failure_positions)

                if n_failures > 0:
                    apfd = 1 - (failure_positions.sum() / (n_tests * n_failures)) + 1 / (2 * n_tests)
                    first_failure = failure_positions.min()
                else:
                    apfd = None
                    first_failure = None

                results.append({
                    'build_id': batch['build_ids'][i],
                    'n_tests': n_tests,
                    'n_failures': n_failures,
                    'apfd': apfd,
                    'first_failure_position': first_failure
                })

    # Calculate mean APFD (only for builds with failures)
    valid_apfds = [r['apfd'] for r in results if r['apfd'] is not None]
    mean_apfd = np.mean(valid_apfds) if valid_apfds else 0.0

    return mean_apfd, results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Filo-Priori L2R for RTPTorrent")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--preprocess-only', action='store_true', help='Only run preprocessing')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths
    processed_path = Path(config['dataset']['processed_path'])
    raw_path = Path(config['dataset']['raw_path'])
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if preprocessing is needed
    train_file = processed_path / config['dataset']['train_file']
    if not train_file.exists():
        logger.error(f"Processed data not found: {train_file}")
        logger.error("Please run preprocessing first:")
        logger.error("  python scripts/preprocessing/preprocess_rtptorrent_ranking.py --preset small")
        sys.exit(1)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 60)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(processed_path / config['dataset']['test_file'])

    logger.info(f"Train: {len(train_df)} samples, {train_df['Build_ID'].nunique()} builds")
    logger.info(f"Test: {len(test_df)} samples, {test_df['Build_ID'].nunique()} builds")

    # Feature columns
    feature_cols = config['dataset']['columns']['historical_features']
    logger.info(f"Historical features: {feature_cols}")

    # =========================================================================
    # STEP 2: GENERATE EMBEDDINGS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: GENERATING EMBEDDINGS")
    logger.info("=" * 60)

    # Create embedding config compatible with SBERTEncoder
    embedding_config = {
        'embedding': {
            'model': config['embeddings']['model'],
            'batch_size': config['embeddings']['batch_size'],
            'max_length': config['embeddings'].get('max_length', 64)
        }
    }
    encoder = SBERTEncoder(embedding_config, device=str(device))

    # Generate embeddings
    train_texts = train_df['semantic_text'].fillna('').tolist()
    test_texts = test_df['semantic_text'].fillna('').tolist()

    logger.info(f"Encoding {len(train_texts)} train texts...")
    train_embeddings = encoder.encode_texts_chunked(train_texts, desc="Train")

    logger.info(f"Encoding {len(test_texts)} test texts...")
    test_embeddings = encoder.encode_texts_chunked(test_texts, desc="Test")

    logger.info(f"Embedding shape: {train_embeddings.shape}")

    # =========================================================================
    # STEP 3: CREATE DATASETS AND DATALOADERS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: CREATING DATASETS")
    logger.info("=" * 60)

    max_tests = config['training'].get('max_tests_per_build', 100)

    train_dataset = RTPTorrentRankingDataset(
        train_df.reset_index(drop=True),
        train_embeddings,
        feature_cols,
        max_tests
    )

    test_dataset = RTPTorrentRankingDataset(
        test_df.reset_index(drop=True),
        test_embeddings,
        feature_cols,
        max_tests
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=config['hardware'].get('num_workers', 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=config['hardware'].get('num_workers', 0)
    )

    # =========================================================================
    # STEP 4: INITIALIZE MODEL
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: INITIALIZING MODEL")
    logger.info("=" * 60)

    model_config = config['model']
    model = RankingModel(
        semantic_dim=model_config.get('semantic_dim', 768),
        structural_dim=len(feature_cols),
        semantic_hidden=model_config['semantic_stream']['hidden_dim'],
        structural_hidden=model_config['structural_stream']['hidden_dim'],
        fusion_hidden=model_config['fusion']['hidden_dim'],
        dropout=model_config['semantic_stream'].get('dropout', 0.3),
        structural_weight=model_config['structural_stream'].get('weight', 1.0)
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    loss_fn = create_ranking_loss(config)
    logger.info(f"Loss function: {config['training']['ranking_loss']['type']}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # =========================================================================
    # STEP 5: TRAINING
    # =========================================================================
    if not args.eval_only:
        logger.info("=" * 60)
        logger.info("STEP 5: TRAINING")
        logger.info("=" * 60)

        best_apfd = 0
        patience_counter = 0
        patience = config['training']['early_stopping']['patience']

        for epoch in range(config['training']['epochs']):
            # Train
            train_loss = train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                config['training'].get('gradient_clip', 1.0)
            )

            # Evaluate
            val_apfd, _ = evaluate_apfd(model, test_loader, device)

            logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}: "
                       f"Loss={train_loss:.4f}, Val APFD={val_apfd:.4f}")

            # Save best model
            if val_apfd > best_apfd:
                best_apfd = val_apfd
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info(f"  → New best model saved! (APFD={val_apfd:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))

    # =========================================================================
    # STEP 6: FINAL EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: FINAL EVALUATION")
    logger.info("=" * 60)

    # Get predictions
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            semantic = batch['semantic'].to(device)
            structural = batch['structural'].to(device)
            mask = batch['mask'].to(device)

            scores = model(semantic, structural, mask)

            for i in range(len(batch['build_ids'])):
                build_mask = mask[i].cpu().numpy()
                build_scores = scores[i].cpu().numpy()[build_mask]
                indices = batch['indices'][i]

                for j, idx in enumerate(indices):
                    if j < len(build_scores):
                        all_predictions.append({
                            'idx': idx,
                            'score': build_scores[j]
                        })

    # Merge predictions with test data
    pred_df = pd.DataFrame(all_predictions)
    test_with_scores = test_df.copy()
    test_with_scores['score'] = 0.0

    for _, row in pred_df.iterrows():
        test_with_scores.loc[row['idx'], 'score'] = row['score']

    # Add travisJobId for evaluator
    test_with_scores['travisJobId'] = test_with_scores['Build_ID'].apply(
        lambda x: int(x.split('_')[-1]) if '_' in x else 0
    )

    # Evaluate against baselines
    logger.info("\nEvaluating against RTPTorrent baselines...")

    # Get unique projects in test set
    projects = test_df['project'].unique() if 'project' in test_df.columns else ['unknown']

    for project in projects:
        logger.info(f"\n{'='*60}")
        logger.info(f"PROJECT: {project}")
        logger.info("=" * 60)

        project_df = test_with_scores[test_with_scores['project'] == project] if 'project' in test_with_scores.columns else test_with_scores

        try:
            evaluator = RTPTorrentEvaluator(raw_path, project)
            test_build_ids = project_df['travisJobId'].unique().tolist()

            results = evaluator.evaluate_model_vs_baselines(project_df, test_build_ids)
            report = evaluator.generate_report(results, output_dir / f"report_{project.replace('@', '_')}.txt")
            print(report)

            # Save results
            with open(output_dir / f"results_{project.replace('@', '_')}.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Could not evaluate {project}: {e}")

    # Save predictions
    test_with_scores.to_csv(output_dir / "predictions.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
