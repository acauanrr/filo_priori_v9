#!/usr/bin/env python3
"""
Filo-Priori V10 Experiment Runner - Simplified Version.

This script implements a simplified V10 pipeline that works with
the already processed RTPTorrent data.

Usage:
    python run_v10_experiment.py

Author: Filo-Priori Team
Version: 10.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLIFIED V10 MODEL (without CodeBERT to avoid heavy dependencies)
# ============================================================================

class SimplifiedV10Model(nn.Module):
    """
    Simplified V10 model that uses pre-computed features from the dataset.

    This version:
    1. Uses structural features directly (no CodeBERT)
    2. Applies time-decay weighting to historical features
    3. Uses LambdaRank-style loss
    4. Implements residual learning with heuristic bias
    """

    def __init__(
        self,
        input_dim: int = 9,  # Matches RTPTorrent features
        hidden_dim: int = 64,
        heuristic_weight: float = 0.5,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Heuristic scorer (directly from recency + failure_rate)
        self.heuristic_scorer = nn.Linear(3, 1)  # recency, failure_rate, recent_failures

        # Neural scorer
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # Learnable alpha for residual fusion
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Initialize heuristic scorer to approximate recently-failed baseline
        with torch.no_grad():
            self.heuristic_scorer.weight.data = torch.tensor([[1.0, 0.5, 0.3]])
            self.heuristic_scorer.bias.data = torch.tensor([0.0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [batch, num_tests, input_dim]

        Returns:
            scores: [batch, num_tests]
        """
        batch_size, num_tests, _ = features.shape

        # Flatten for encoding
        flat_features = features.view(-1, self.input_dim)

        # Encode features
        encoded = self.encoder(flat_features)  # [batch*tests, hidden]

        # Neural score
        neural_score = self.neural_scorer(encoded)  # [batch*tests, 1]
        neural_score = neural_score.view(batch_size, num_tests)

        # Heuristic score (from key features: recency proxy, failure_rate, recent_failures)
        # Indices: failure_rate=4, recent_failures=5, last_failure_recency=8
        heuristic_features = torch.stack([
            1.0 / (features[:, :, 8] + 1),  # Recency transform: 1/(last_failure_recency+1)
            features[:, :, 4],  # failure_rate
            features[:, :, 5] / (features[:, :, 6] + 1),  # recent_failures / recent_executions
        ], dim=-1)

        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)  # [batch, tests]

        # Residual fusion: α * heuristic + (1-α) * neural
        alpha = torch.sigmoid(self.alpha_logit)
        scores = alpha * heuristic_score + (1 - alpha) * neural_score

        return scores

    def get_alpha(self) -> float:
        """Get current heuristic weight."""
        return torch.sigmoid(self.alpha_logit).item()


class LambdaRankLoss(nn.Module):
    """
    Simplified LambdaRank loss for pairwise ranking.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        For each pair (i, j) where relevance[i] > relevance[j],
        we want score[i] > score[j].
        """
        batch_size, num_items = scores.shape
        device = scores.device

        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for b in range(batch_size):
            s = scores[b]
            y = relevances[b]

            # Find positive (failed) and negative (passed) indices
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # Pairwise loss
            for i in pos_idx:
                for j in neg_idx:
                    s_diff = s[i] - s[j]
                    # We want s[i] > s[j], so loss = log(1 + exp(-s_diff))
                    pair_loss = F.softplus(-self.sigma * s_diff)
                    total_loss = total_loss + pair_loss
                    num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss


class APFDLoss(nn.Module):
    """
    Direct APFD optimization loss using soft ranking.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        relevances: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute soft APFD loss."""
        batch_size, num_items = scores.shape
        device = scores.device

        losses = []

        for b in range(batch_size):
            s = scores[b]
            y = relevances[b]

            num_failures = y.sum()
            if num_failures == 0:
                continue

            # Soft ranking via pairwise comparisons
            diff = s.unsqueeze(0) - s.unsqueeze(1)  # [n, n]
            soft_rank = torch.sigmoid(diff / self.temperature).sum(dim=1)  # [n]

            # Sum of failure ranks
            fail_rank_sum = (soft_rank * y).sum()

            # APFD = 1 - sum_ranks / (n * m) + 1/(2n)
            n = num_items
            m = num_failures
            apfd = 1 - fail_rank_sum / (n * m) + 1 / (2 * n)

            losses.append(1 - apfd)  # Minimize 1-APFD

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_rtptorrent_data(data_dir: str = "datasets/02_rtptorrent/processed_ranking"):
    """
    Load the processed RTPTorrent data.
    """
    data_dir = Path(data_dir)

    # Load train and test
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
    logger.info(f"Projects: {list(metadata['projects'].keys())}")

    return train_df, test_df, metadata


def prepare_features(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Extract feature names and values from dataframe.
    """
    # Feature columns
    feature_cols = [
        'duration', 'count', 'failures', 'errors',
        'failure_rate', 'recent_failures', 'recent_executions',
        'avg_duration', 'last_failure_recency'
    ]

    # Ensure columns exist
    available_cols = [c for c in feature_cols if c in df.columns]

    return available_cols, df[available_cols].values


def create_build_batches(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """
    Group data by build and create batches.
    """
    builds = []

    for build_id, group in df.groupby('Build_ID'):
        # Get features
        features = group[feature_cols].values.astype(np.float32)

        # Fill NaN with 0
        features = np.nan_to_num(features, nan=0.0)

        # Get labels (1 = failure)
        labels = group['is_failure'].values.astype(np.float32)

        # Get test IDs
        test_ids = group['TC_Key'].tolist()

        builds.append({
            'build_id': build_id,
            'features': torch.tensor(features),
            'labels': torch.tensor(labels),
            'test_ids': test_ids,
            'num_tests': len(group),
            'num_failures': int(labels.sum())
        })

    return builds


# ============================================================================
# METRICS
# ============================================================================

def compute_apfd(rankings: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute APFD for a single build.

    Args:
        rankings: [num_tests] - rank of each test (0 = highest priority)
        labels: [num_tests] - 1 if failed, 0 if passed

    Returns:
        APFD value in [0, 1]
    """
    n = len(labels)
    m = labels.sum().item()

    if m == 0:
        return 1.0  # No failures, perfect by default

    # Get positions of failures (1-indexed for APFD formula)
    # rankings gives rank (0=first), so position = rank + 1
    fail_positions = (rankings[labels == 1] + 1).float()

    # APFD = 1 - (sum of positions) / (n * m) + 1/(2n)
    apfd = 1 - fail_positions.sum().item() / (n * m) + 1 / (2 * n)

    return min(max(apfd, 0.0), 1.0)


def compute_metrics(model: nn.Module, builds: List[Dict], device: torch.device) -> Dict:
    """
    Compute evaluation metrics on a set of builds.
    """
    model.eval()

    all_apfd = []
    builds_with_failures = 0

    with torch.no_grad():
        for build in tqdm(builds, desc="Evaluating"):
            features = build['features'].unsqueeze(0).to(device)  # [1, n, d]
            labels = build['labels'].to(device)

            if build['num_failures'] == 0:
                continue

            builds_with_failures += 1

            # Get scores
            scores = model(features).squeeze(0)  # [n]

            # Convert to rankings (higher score = lower rank = higher priority)
            _, indices = scores.sort(descending=True)
            rankings = torch.zeros_like(scores, dtype=torch.long)
            rankings[indices] = torch.arange(len(scores), device=device)

            # Compute APFD
            apfd = compute_apfd(rankings, labels)
            all_apfd.append(apfd)

    if not all_apfd:
        return {'apfd': 0.0, 'apfd_std': 0.0, 'num_builds': 0}

    return {
        'apfd': np.mean(all_apfd),
        'apfd_std': np.std(all_apfd),
        'apfd_median': np.median(all_apfd),
        'apfd_min': np.min(all_apfd),
        'apfd_max': np.max(all_apfd),
        'num_builds': builds_with_failures
    }


# ============================================================================
# BASELINES
# ============================================================================

def baseline_random(builds: List[Dict]) -> List[float]:
    """Random baseline."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        n = build['num_tests']
        rankings = torch.randperm(n)
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


def baseline_recently_failed(builds: List[Dict], history: Dict) -> List[float]:
    """
    Recently-failed baseline using last_failure_recency.
    Lower recency = higher priority.
    """
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue

        # Get last_failure_recency (index 8)
        recency = build['features'][:, 8]  # Lower = more recent

        # Rank by recency (ascending - lower recency = higher priority)
        _, indices = recency.sort()
        rankings = torch.zeros_like(recency, dtype=torch.long)
        rankings[indices] = torch.arange(len(recency))

        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)

    return apfds


def baseline_failure_rate(builds: List[Dict]) -> List[float]:
    """
    Failure rate baseline - prioritize tests with high historical failure rate.
    """
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue

        # Get failure_rate (index 4)
        fail_rate = build['features'][:, 4]

        # Rank by failure rate (descending - higher rate = higher priority)
        _, indices = fail_rate.sort(descending=True)
        rankings = torch.zeros_like(fail_rate, dtype=torch.long)
        rankings[indices] = torch.arange(len(fail_rate))

        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)

    return apfds


# ============================================================================
# TRAINING
# ============================================================================

def train_v10(
    model: nn.Module,
    train_builds: List[Dict],
    val_builds: List[Dict],
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    loss_type: str = 'apfd'
) -> Dict:
    """
    Train the V10 model.
    """
    model = model.to(device)

    # Loss function
    if loss_type == 'lambda':
        criterion = LambdaRankLoss(sigma=1.0)
    else:
        criterion = APFDLoss(temperature=1.0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Filter builds with failures for training
    train_builds_with_failures = [b for b in train_builds if b['num_failures'] > 0]
    logger.info(f"Training on {len(train_builds_with_failures)} builds with failures")

    best_apfd = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_apfd': [], 'alpha': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle training data
        np.random.shuffle(train_builds_with_failures)

        for build in tqdm(train_builds_with_failures, desc=f"Epoch {epoch+1}", leave=False):
            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].unsqueeze(0).to(device)

            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        # Validation
        val_metrics = compute_metrics(model, val_builds, device)
        val_apfd = val_metrics['apfd']

        # Track history
        history['train_loss'].append(avg_loss)
        history['val_apfd'].append(val_apfd)
        history['alpha'].append(model.get_alpha())

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val APFD: {val_apfd:.4f} (+/- {val_metrics['apfd_std']:.4f}) - "
            f"Alpha: {model.get_alpha():.3f}"
        )

        # Early stopping
        if val_apfd > best_apfd:
            best_apfd = val_apfd
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_state)

    return {
        'best_apfd': best_apfd,
        'history': history,
        'final_alpha': model.get_alpha()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment runner."""
    logger.info("="*60)
    logger.info("Filo-Priori V10 Experiment")
    logger.info("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    logger.info("\n[1/5] Loading data...")
    train_df, test_df, metadata = load_rtptorrent_data()

    # Prepare features
    feature_cols, _ = prepare_features(train_df)
    logger.info(f"Features: {feature_cols}")

    # Create batches
    logger.info("\n[2/5] Creating build batches...")
    train_builds = create_build_batches(train_df, feature_cols)
    test_builds = create_build_batches(test_df, feature_cols)

    train_with_failures = len([b for b in train_builds if b['num_failures'] > 0])
    test_with_failures = len([b for b in test_builds if b['num_failures'] > 0])

    logger.info(f"Train builds: {len(train_builds)} ({train_with_failures} with failures)")
    logger.info(f"Test builds: {len(test_builds)} ({test_with_failures} with failures)")

    # Split train into train/val (80/20)
    np.random.shuffle(train_builds)
    split_idx = int(len(train_builds) * 0.8)
    actual_train = train_builds[:split_idx]
    val_builds = train_builds[split_idx:]

    logger.info(f"Train/Val split: {len(actual_train)}/{len(val_builds)}")

    # Compute baselines
    logger.info("\n[3/5] Computing baselines on TEST set...")

    random_apfds = baseline_random(test_builds)
    recently_failed_apfds = baseline_recently_failed(test_builds, {})
    failure_rate_apfds = baseline_failure_rate(test_builds)

    logger.info(f"  Random:          APFD = {np.mean(random_apfds):.4f} (+/- {np.std(random_apfds):.4f})")
    logger.info(f"  Recently-Failed: APFD = {np.mean(recently_failed_apfds):.4f} (+/- {np.std(recently_failed_apfds):.4f})")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(failure_rate_apfds):.4f} (+/- {np.std(failure_rate_apfds):.4f})")

    # Create model
    logger.info("\n[4/5] Training V10 model...")
    model = SimplifiedV10Model(
        input_dim=len(feature_cols),
        hidden_dim=64,
        heuristic_weight=0.5,
        dropout=0.3
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with APFD loss
    train_result = train_v10(
        model=model,
        train_builds=actual_train,
        val_builds=val_builds,
        device=device,
        num_epochs=30,
        lr=1e-3,
        patience=7,
        loss_type='apfd'
    )

    logger.info(f"Training complete! Best Val APFD: {train_result['best_apfd']:.4f}")
    logger.info(f"Final Alpha (heuristic weight): {train_result['final_alpha']:.3f}")

    # Evaluate on test
    logger.info("\n[5/5] Evaluating on TEST set...")
    test_metrics = compute_metrics(model, test_builds, device)

    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS ON TEST SET")
    logger.info(f"{'='*60}")
    logger.info(f"V10 Model:         APFD = {test_metrics['apfd']:.4f} (+/- {test_metrics['apfd_std']:.4f})")
    logger.info(f"  Median: {test_metrics['apfd_median']:.4f}")
    logger.info(f"  Min: {test_metrics['apfd_min']:.4f}, Max: {test_metrics['apfd_max']:.4f}")
    logger.info(f"  Builds evaluated: {test_metrics['num_builds']}")

    logger.info(f"\nBaselines:")
    logger.info(f"  Random:          APFD = {np.mean(random_apfds):.4f}")
    logger.info(f"  Recently-Failed: APFD = {np.mean(recently_failed_apfds):.4f}")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(failure_rate_apfds):.4f}")

    # Compute improvements
    v10_apfd = test_metrics['apfd']
    rf_apfd = np.mean(recently_failed_apfds)
    improvement = (v10_apfd - rf_apfd) / rf_apfd * 100 if rf_apfd > 0 else 0

    logger.info(f"\nImprovement over Recently-Failed: {improvement:+.2f}%")

    # Save results
    results_dir = Path("results/experiment_v10_rtptorrent")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'v10': {
            'apfd': test_metrics['apfd'],
            'apfd_std': test_metrics['apfd_std'],
            'apfd_median': test_metrics['apfd_median'],
            'num_builds': test_metrics['num_builds'],
            'alpha': train_result['final_alpha']
        },
        'baselines': {
            'random': {'apfd': np.mean(random_apfds), 'std': np.std(random_apfds)},
            'recently_failed': {'apfd': np.mean(recently_failed_apfds), 'std': np.std(recently_failed_apfds)},
            'failure_rate': {'apfd': np.mean(failure_rate_apfds), 'std': np.std(failure_rate_apfds)}
        },
        'improvement_vs_recently_failed': improvement,
        'training_history': train_result['history']
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': len(feature_cols),
            'hidden_dim': 64,
            'heuristic_weight': 0.5
        },
        'results': results
    }, results_dir / 'best_model.pt')

    logger.info(f"\nResults saved to {results_dir}")
    logger.info("="*60)

    return results


if __name__ == '__main__':
    main()
