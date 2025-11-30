#!/usr/bin/env python3
"""
Filo-Priori V10 Unified Experiment Runner.

This script runs V10 experiments on both datasets:
- 01_industry (Industrial QTA)
- 02_rtptorrent (Open Source)

It handles the different data formats and computes consistent features.

Usage:
    python run_v10_unified.py --dataset industry
    python run_v10_unified.py --dataset rtptorrent
    python run_v10_unified.py --dataset all

Author: Filo-Priori Team
Version: 10.0.0
"""

import os
import sys
import json
import argparse
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
from tqdm import tqdm
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# V10 MODEL (Same as before, but cleaner)
# =============================================================================

class FiloPrioriV10(nn.Module):
    """
    Filo-Priori V10 Model with Residual Learning.

    Architecture:
    - Neural encoder for learned features
    - Heuristic scorer for domain knowledge
    - Learnable alpha for residual fusion
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim

        # Neural feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Heuristic scorer (from key features)
        self.heuristic_scorer = nn.Linear(3, 1)

        # Neural scorer
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # Learnable alpha for residual fusion
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # Initialize heuristic scorer weights
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

        # Encode features
        flat_features = features.view(-1, self.input_dim)
        encoded = self.encoder(flat_features)
        neural_score = self.neural_scorer(encoded).view(batch_size, num_tests)

        # Heuristic features: recency_score, failure_rate, recent_fail_rate
        # Indices depend on feature order (see prepare_features)
        heuristic_features = self._extract_heuristic_features(features)
        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)

        # Residual fusion
        alpha = torch.sigmoid(self.alpha_logit)
        scores = alpha * heuristic_score + (1 - alpha) * neural_score

        return scores

    def _extract_heuristic_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract key heuristic features for scoring."""
        # Feature indices (from prepare_features):
        # 0: duration, 1: count, 2: failures, 3: errors
        # 4: failure_rate, 5: recent_failures, 6: recent_executions
        # 7: avg_duration, 8: last_failure_recency

        recency_score = 1.0 / (features[:, :, 8] + 1)  # 1/(last_failure_recency+1)
        failure_rate = features[:, :, 4]  # failure_rate
        recent_ratio = features[:, :, 5] / (features[:, :, 6] + 1)  # recent_failures/recent_executions

        return torch.stack([recency_score, failure_rate, recent_ratio], dim=-1)

    def get_alpha(self) -> float:
        """Get current heuristic weight."""
        return torch.sigmoid(self.alpha_logit).item()


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class APFDLoss(nn.Module):
    """Direct APFD optimization loss."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, relevances, mask=None):
        batch_size, num_items = scores.shape
        device = scores.device
        losses = []

        for b in range(batch_size):
            s, y = scores[b], relevances[b]
            num_failures = y.sum()
            if num_failures == 0:
                continue

            diff = s.unsqueeze(0) - s.unsqueeze(1)
            soft_rank = torch.sigmoid(diff / self.temperature).sum(dim=1)
            fail_rank_sum = (soft_rank * y).sum()

            n, m = num_items, num_failures
            apfd = 1 - fail_rank_sum / (n * m) + 1 / (2 * n)
            losses.append(1 - apfd)

        if losses:
            return torch.stack(losses).mean()
        else:
            # Return zero with gradient
            return scores.sum() * 0.0


class LambdaRankLoss(nn.Module):
    """LambdaRank pairwise ranking loss."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, scores, relevances, mask=None):
        batch_size, num_items = scores.shape
        device = scores.device

        # Initialize with a dummy that has gradients
        total_loss = scores.sum() * 0.0
        num_pairs = 0

        for b in range(batch_size):
            s, y = scores[b], relevances[b]
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # Vectorized pairwise computation for efficiency
            pos_scores = s[pos_idx]  # [num_pos]
            neg_scores = s[neg_idx]  # [num_neg]

            # Compute all pairwise differences
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [num_pos, num_neg]
            pair_losses = F.softplus(-self.sigma * diff)

            total_loss = total_loss + pair_losses.sum()
            num_pairs += len(pos_idx) * len(neg_idx)

        return total_loss / max(num_pairs, 1)


# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Unified data processor for both datasets."""

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.feature_cols = [
            'duration', 'count', 'failures', 'errors',
            'failure_rate', 'recent_failures', 'recent_executions',
            'avg_duration', 'last_failure_recency'
        ]

    def load_and_process(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process data based on dataset type."""
        if self.dataset_type == 'industry':
            return self._process_industry(data_dir)
        elif self.dataset_type == 'rtptorrent':
            return self._process_rtptorrent(data_dir)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _process_industry(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process industrial dataset from raw format."""
        logger.info("Processing industrial dataset...")

        # Load raw data
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"

        logger.info(f"Loading train data from {train_path}...")
        train_df = pd.read_csv(train_path, low_memory=False)
        logger.info(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path, low_memory=False)

        logger.info(f"Raw train size: {len(train_df)}, test size: {len(test_df)}")

        # Convert result to binary
        train_df['is_failure'] = (train_df['TE_Test_Result'] == 'Fail').astype(int)
        test_df['is_failure'] = (test_df['TE_Test_Result'] == 'Fail').astype(int)

        # Compute features
        train_df = self._compute_features(train_df, is_train=True)
        test_df = self._compute_features(test_df, is_train=False, reference_df=train_df)

        return train_df, test_df

    def _process_rtptorrent(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load already processed RTPTorrent data."""
        logger.info("Loading processed RTPTorrent dataset...")

        processed_dir = data_dir / "processed_ranking"

        train_df = pd.read_csv(processed_dir / "train.csv")
        test_df = pd.read_csv(processed_dir / "test.csv")

        logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

        return train_df, test_df

    def _compute_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        reference_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Compute features for ranking."""
        logger.info("Computing features...")

        # Initialize feature columns with defaults
        df['duration'] = 0.0
        df['count'] = 0
        df['failures'] = 0
        df['errors'] = 0
        df['failure_rate'] = 0.0
        df['recent_failures'] = 0
        df['recent_executions'] = 0
        df['avg_duration'] = 0.0
        df['last_failure_recency'] = 999

        # Sort by build date for temporal features
        if 'Build_Test_Start_Date' in df.columns:
            df['Build_Test_Start_Date'] = pd.to_datetime(df['Build_Test_Start_Date'], errors='coerce')
            df = df.sort_values('Build_Test_Start_Date')

        # Group by test case to compute historical features
        test_history = defaultdict(lambda: {'count': 0, 'failures': 0, 'recent': []})

        # If test data, use train history as base
        if not is_train and reference_df is not None:
            for _, row in reference_df.iterrows():
                tc_key = row['TC_Key']
                test_history[tc_key]['count'] += 1
                if row['is_failure'] == 1:
                    test_history[tc_key]['failures'] += 1
                test_history[tc_key]['recent'].append(row['is_failure'])
                if len(test_history[tc_key]['recent']) > 10:
                    test_history[tc_key]['recent'].pop(0)

        # Build features row by row
        features_list = []

        # Get unique builds with order
        build_order = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in build_order.columns:
            build_order = build_order.sort_values('Build_Test_Start_Date')
        build_to_idx = {b: i for i, b in enumerate(build_order['Build_ID'])}

        for build_id, build_group in tqdm(df.groupby('Build_ID', sort=False),
                                          desc="Computing features"):
            build_idx = build_to_idx.get(build_id, 0)

            for idx, row in build_group.iterrows():
                tc_key = row['TC_Key']
                history = test_history[tc_key]

                # Current features based on history
                count = history['count']
                failures = history['failures']
                failure_rate = failures / count if count > 0 else 0.0

                recent = history['recent']
                recent_failures = sum(recent[-5:]) if recent else 0
                recent_executions = min(len(recent), 5)

                # Last failure recency (builds since last failure)
                last_failure_recency = 999
                for i, r in enumerate(reversed(recent)):
                    if r == 1:
                        last_failure_recency = i
                        break

                features_list.append({
                    'Build_ID': build_id,
                    'TC_Key': tc_key,
                    'is_failure': row['is_failure'],
                    'duration': 0.0,  # Not available in industry dataset
                    'count': count,
                    'failures': failures,
                    'errors': 0,
                    'failure_rate': failure_rate,
                    'recent_failures': recent_failures,
                    'recent_executions': recent_executions,
                    'avg_duration': 0.0,
                    'last_failure_recency': last_failure_recency
                })

                # Update history for next iteration (only in train)
                if is_train:
                    history['count'] += 1
                    if row['is_failure'] == 1:
                        history['failures'] += 1
                    history['recent'].append(row['is_failure'])
                    if len(history['recent']) > 10:
                        history['recent'].pop(0)

        result_df = pd.DataFrame(features_list)
        logger.info(f"Computed features for {len(result_df)} rows")

        return result_df


def create_builds(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """Group data by build and create batches."""
    builds = []

    for build_id, group in df.groupby('Build_ID'):
        features = group[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        labels = group['is_failure'].values.astype(np.float32)

        builds.append({
            'build_id': build_id,
            'features': torch.tensor(features),
            'labels': torch.tensor(labels),
            'num_tests': len(group),
            'num_failures': int(labels.sum())
        })

    return builds


# =============================================================================
# METRICS
# =============================================================================

def compute_apfd(rankings: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute APFD metric."""
    n = len(labels)
    m = labels.sum().item()
    if m == 0:
        return 1.0
    fail_positions = (rankings[labels == 1] + 1).float()
    apfd = 1 - fail_positions.sum().item() / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


def evaluate_model(model: nn.Module, builds: List[Dict], device: torch.device) -> Dict:
    """Evaluate model on builds."""
    model.eval()
    apfds = []

    with torch.no_grad():
        for build in builds:
            if build['num_failures'] == 0:
                continue

            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].to(device)

            scores = model(features).squeeze(0)
            _, indices = scores.sort(descending=True)
            rankings = torch.zeros_like(scores, dtype=torch.long)
            rankings[indices] = torch.arange(len(scores), device=device)

            apfd = compute_apfd(rankings, labels)
            apfds.append(apfd)

    return {
        'apfd': np.mean(apfds) if apfds else 0.0,
        'apfd_std': np.std(apfds) if apfds else 0.0,
        'apfd_values': apfds,
        'num_builds': len(apfds)
    }


# =============================================================================
# BASELINES
# =============================================================================

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


def baseline_recently_failed(builds: List[Dict]) -> List[float]:
    """Recently-failed baseline using last_failure_recency."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        recency = build['features'][:, 8]  # last_failure_recency
        _, indices = recency.sort()
        rankings = torch.zeros_like(recency, dtype=torch.long)
        rankings[indices] = torch.arange(len(recency))
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


def baseline_failure_rate(builds: List[Dict]) -> List[float]:
    """Failure rate baseline."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        fail_rate = build['features'][:, 4]  # failure_rate
        _, indices = fail_rate.sort(descending=True)
        rankings = torch.zeros_like(fail_rate, dtype=torch.long)
        rankings[indices] = torch.arange(len(fail_rate))
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_builds: List[Dict],
    val_builds: List[Dict],
    device: torch.device,
    num_epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 7,
    loss_type: str = 'lambda'
) -> Dict:
    """Train the V10 model."""
    model = model.to(device)

    criterion = LambdaRankLoss() if loss_type == 'lambda' else APFDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_with_failures = [b for b in train_builds if b['num_failures'] > 0]
    logger.info(f"Training on {len(train_with_failures)} builds with failures")

    best_apfd = 0.0
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_apfd': [], 'alpha': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        np.random.shuffle(train_with_failures)

        for build in tqdm(train_with_failures, desc=f"Epoch {epoch+1}", leave=False):
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
        val_metrics = evaluate_model(model, val_builds, device)

        history['train_loss'].append(avg_loss)
        history['val_apfd'].append(val_metrics['apfd'])
        history['alpha'].append(model.get_alpha())

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val APFD: {val_metrics['apfd']:.4f} - "
            f"Alpha: {model.get_alpha():.3f}"
        )

        if val_metrics['apfd'] > best_apfd:
            best_apfd = val_metrics['apfd']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        'best_val_apfd': best_apfd,
        'final_alpha': model.get_alpha(),
        'history': history
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(dataset_type: str, base_dir: Path) -> Dict:
    """Run V10 experiment on a specific dataset."""
    logger.info("=" * 70)
    logger.info(f"FILO-PRIORI V10 EXPERIMENT: {dataset_type.upper()}")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Set data directory
    if dataset_type == 'industry':
        data_dir = base_dir / "datasets/01_industry"
    else:
        data_dir = base_dir / "datasets/02_rtptorrent"

    # Process data
    processor = DataProcessor(dataset_type)
    train_df, test_df = processor.load_and_process(data_dir)

    # Create builds
    feature_cols = processor.feature_cols
    train_builds = create_builds(train_df, feature_cols)
    test_builds = create_builds(test_df, feature_cols)

    train_with_failures = len([b for b in train_builds if b['num_failures'] > 0])
    test_with_failures = len([b for b in test_builds if b['num_failures'] > 0])

    logger.info(f"Train builds: {len(train_builds)} ({train_with_failures} with failures)")
    logger.info(f"Test builds: {len(test_builds)} ({test_with_failures} with failures)")

    # Split train/val
    np.random.shuffle(train_builds)
    split_idx = int(len(train_builds) * 0.8)
    actual_train = train_builds[:split_idx]
    val_builds = train_builds[split_idx:]

    logger.info(f"Train/Val split: {len(actual_train)}/{len(val_builds)}")

    # Compute baselines
    logger.info("\nComputing baselines on TEST set...")
    random_apfds = baseline_random(test_builds)
    rf_apfds = baseline_recently_failed(test_builds)
    fr_apfds = baseline_failure_rate(test_builds)

    logger.info(f"  Random:          APFD = {np.mean(random_apfds):.4f} (+/- {np.std(random_apfds):.4f})")
    logger.info(f"  Recently-Failed: APFD = {np.mean(rf_apfds):.4f} (+/- {np.std(rf_apfds):.4f})")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(fr_apfds):.4f} (+/- {np.std(fr_apfds):.4f})")

    # Create and train model
    logger.info("\nTraining V10 model...")
    model = FiloPrioriV10(input_dim=len(feature_cols), hidden_dim=64, dropout=0.3)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_result = train_model(
        model=model,
        train_builds=actual_train,
        val_builds=val_builds,
        device=device,
        num_epochs=30,
        lr=1e-3,
        patience=7,
        loss_type='lambda'
    )

    # Evaluate on test
    logger.info("\nEvaluating on TEST set...")
    test_metrics = evaluate_model(model, test_builds, device)

    # Compute improvement
    rf_mean = np.mean(rf_apfds)
    v10_apfd = test_metrics['apfd']
    improvement = (v10_apfd - rf_mean) / rf_mean * 100 if rf_mean > 0 else 0

    # Statistical significance
    min_len = min(len(test_metrics['apfd_values']), len(rf_apfds))
    if min_len >= 5:
        stat, p_value = stats.wilcoxon(
            test_metrics['apfd_values'][:min_len],
            rf_apfds[:min_len],
            alternative='greater'
        )
    else:
        p_value = 1.0

    # Results
    logger.info("\n" + "=" * 70)
    logger.info(f"FINAL RESULTS - {dataset_type.upper()}")
    logger.info("=" * 70)
    logger.info(f"V10 Model:         APFD = {v10_apfd:.4f} (+/- {test_metrics['apfd_std']:.4f})")
    logger.info(f"Recently-Failed:   APFD = {rf_mean:.4f}")
    logger.info(f"Improvement:       {improvement:+.2f}%")
    logger.info(f"p-value:           {p_value:.4f}")
    logger.info(f"Alpha (learned):   {train_result['final_alpha']:.3f}")
    logger.info("=" * 70)

    # Save results
    results_dir = base_dir / f"results/experiment_v10_{dataset_type}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': dataset_type,
        'v10': {
            'apfd': v10_apfd,
            'apfd_std': test_metrics['apfd_std'],
            'num_builds': test_metrics['num_builds'],
            'alpha': train_result['final_alpha']
        },
        'baselines': {
            'random': {'apfd': float(np.mean(random_apfds)), 'std': float(np.std(random_apfds))},
            'recently_failed': {'apfd': float(np.mean(rf_apfds)), 'std': float(np.std(rf_apfds))},
            'failure_rate': {'apfd': float(np.mean(fr_apfds)), 'std': float(np.std(fr_apfds))}
        },
        'improvement_vs_rf': improvement,
        'p_value': p_value,
        'history': train_result['history']
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'input_dim': len(feature_cols), 'hidden_dim': 64},
        'results': results
    }, results_dir / 'best_model.pt')

    logger.info(f"Results saved to {results_dir}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Filo-Priori V10 Unified Experiment')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['industry', 'rtptorrent', 'all'],
        default='industry',
        help='Dataset to run experiment on'
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.dataset == 'all':
        datasets = ['industry', 'rtptorrent']
    else:
        datasets = [args.dataset]

    all_results = {}

    for dataset in datasets:
        results = run_experiment(dataset, base_dir)
        all_results[dataset] = results

    # If running both, show comparison
    if len(all_results) == 2:
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-DATASET COMPARISON")
        logger.info("=" * 70)
        logger.info(f"{'Dataset':<15} {'V10 APFD':<12} {'RF Baseline':<12} {'Improvement':<12}")
        logger.info("-" * 51)

        for dataset, res in all_results.items():
            v10 = res['v10']['apfd']
            rf = res['baselines']['recently_failed']['apfd']
            imp = res['improvement_vs_rf']
            logger.info(f"{dataset:<15} {v10:<12.4f} {rf:<12.4f} {imp:+.2f}%")

        logger.info("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
