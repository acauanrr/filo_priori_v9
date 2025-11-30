#!/usr/bin/env python3
"""
Filo-Priori V10 Ablation Study.

This script runs ablation experiments to understand the contribution
of each component in the V10 architecture.

Ablation Variants:
1. V10-Full: Complete model with residual learning (from main experiment)
2. V10-NeuralOnly: Only neural component, no heuristics
3. V10-HeuristicOnly: Only heuristic scoring, no neural learning
4. V10-HighAlpha: α=0.8 (heavy heuristic bias)
5. V10-LowAlpha: α=0.2 (heavy neural bias)
6. V10-LambdaRank: Using LambdaRank loss instead of APFD

Author: Filo-Priori Team
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# MODEL VARIANTS
# =============================================================================

class NeuralOnlyModel(nn.Module):
    """Neural-only model without heuristic residual."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tests, input_dim = features.shape
        flat = features.view(-1, input_dim)
        encoded = self.encoder(flat)
        scores = self.scorer(encoded).view(batch_size, num_tests)
        return scores

    def get_alpha(self):
        return 0.0  # No heuristic component


class HeuristicOnlyModel(nn.Module):
    """Heuristic-only model - no learning, just feature combination."""

    def __init__(self):
        super().__init__()
        # Learnable weights for heuristic features
        self.weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Features: recency (idx 8), failure_rate (idx 4), recent_failures (idx 5)
        recency_score = 1.0 / (features[:, :, 8] + 1)
        failure_rate = features[:, :, 4]
        recent_ratio = features[:, :, 5] / (features[:, :, 6] + 1)

        heuristic_features = torch.stack([recency_score, failure_rate, recent_ratio], dim=-1)
        scores = (heuristic_features * self.weights.abs()).sum(dim=-1)
        return scores

    def get_alpha(self):
        return 1.0  # Pure heuristic


class FixedAlphaModel(nn.Module):
    """V10 model with fixed alpha (no learning of alpha)."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64,
                 fixed_alpha: float = 0.5, dropout: float = 0.3):
        super().__init__()

        self.fixed_alpha = fixed_alpha
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heuristic_scorer = nn.Linear(3, 1)
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        with torch.no_grad():
            self.heuristic_scorer.weight.data = torch.tensor([[1.0, 0.5, 0.3]])
            self.heuristic_scorer.bias.data = torch.tensor([0.0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tests, _ = features.shape
        flat_features = features.view(-1, self.input_dim)

        encoded = self.encoder(flat_features)
        neural_score = self.neural_scorer(encoded).view(batch_size, num_tests)

        heuristic_features = torch.stack([
            1.0 / (features[:, :, 8] + 1),
            features[:, :, 4],
            features[:, :, 5] / (features[:, :, 6] + 1),
        ], dim=-1)
        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)

        scores = self.fixed_alpha * heuristic_score + (1 - self.fixed_alpha) * neural_score
        return scores

    def get_alpha(self):
        return self.fixed_alpha


class LearnableAlphaModel(nn.Module):
    """V10 model with learnable alpha (same as main experiment)."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heuristic_scorer = nn.Linear(3, 1)
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.heuristic_scorer.weight.data = torch.tensor([[1.0, 0.5, 0.3]])
            self.heuristic_scorer.bias.data = torch.tensor([0.0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tests, _ = features.shape
        flat_features = features.view(-1, self.input_dim)

        encoded = self.encoder(flat_features)
        neural_score = self.neural_scorer(encoded).view(batch_size, num_tests)

        heuristic_features = torch.stack([
            1.0 / (features[:, :, 8] + 1),
            features[:, :, 4],
            features[:, :, 5] / (features[:, :, 6] + 1),
        ], dim=-1)
        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)

        alpha = torch.sigmoid(self.alpha_logit)
        scores = alpha * heuristic_score + (1 - alpha) * neural_score
        return scores

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit).item()


# =============================================================================
# LOSSES
# =============================================================================

class APFDLoss(nn.Module):
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

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


class LambdaRankLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, scores, relevances, mask=None):
        batch_size, num_items = scores.shape
        device = scores.device
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for b in range(batch_size):
            s, y = scores[b], relevances[b]
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            for i in pos_idx:
                for j in neg_idx:
                    s_diff = s[i] - s[j]
                    pair_loss = F.softplus(-self.sigma * s_diff)
                    total_loss = total_loss + pair_loss
                    num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else total_loss


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir: str = "datasets/02_rtptorrent/processed_ranking"):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df


def create_builds(df, feature_cols):
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

def compute_apfd(rankings, labels):
    n = len(labels)
    m = labels.sum().item()
    if m == 0:
        return 1.0
    fail_positions = (rankings[labels == 1] + 1).float()
    apfd = 1 - fail_positions.sum().item() / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


def evaluate_model(model, builds, device):
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


def train_model(model, train_builds, val_builds, device,
                num_epochs=30, lr=1e-3, patience=7, loss_type='apfd'):
    model = model.to(device)

    criterion = LambdaRankLoss() if loss_type == 'lambda' else APFDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_with_failures = [b for b in train_builds if b['num_failures'] > 0]

    best_apfd = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_with_failures)

        for build in train_with_failures:
            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].unsqueeze(0).to(device)

            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_metrics = evaluate_model(model, val_builds, device)

        if val_metrics['apfd'] > best_apfd:
            best_apfd = val_metrics['apfd']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return {'best_val_apfd': best_apfd, 'final_alpha': model.get_alpha()}


# =============================================================================
# BASELINES
# =============================================================================

def baseline_recently_failed(builds):
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        recency = build['features'][:, 8]
        _, indices = recency.sort()
        rankings = torch.zeros_like(recency, dtype=torch.long)
        rankings[indices] = torch.arange(len(recency))
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


# =============================================================================
# MAIN ABLATION
# =============================================================================

def run_ablation():
    logger.info("=" * 70)
    logger.info("FILO-PRIORI V10 ABLATION STUDY")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    logger.info("\nLoading data...")
    train_df, test_df = load_data()

    feature_cols = ['duration', 'count', 'failures', 'errors', 'failure_rate',
                    'recent_failures', 'recent_executions', 'avg_duration',
                    'last_failure_recency']

    train_builds = create_builds(train_df, feature_cols)
    test_builds = create_builds(test_df, feature_cols)

    # Split train/val
    np.random.shuffle(train_builds)
    split_idx = int(len(train_builds) * 0.8)
    actual_train = train_builds[:split_idx]
    val_builds = train_builds[split_idx:]

    logger.info(f"Train: {len(actual_train)}, Val: {len(val_builds)}, Test: {len(test_builds)}")

    # Baselines
    rf_apfds = baseline_recently_failed(test_builds)
    rf_mean = np.mean(rf_apfds)
    logger.info(f"\nRecently-Failed baseline: APFD = {rf_mean:.4f}")

    # Define ablation experiments
    experiments = [
        {
            'name': 'V10-Full-APFD',
            'model': LearnableAlphaModel(input_dim=9),
            'loss': 'apfd',
            'description': 'Full V10 with learnable alpha + APFD loss'
        },
        {
            'name': 'V10-Full-Lambda',
            'model': LearnableAlphaModel(input_dim=9),
            'loss': 'lambda',
            'description': 'Full V10 with learnable alpha + LambdaRank loss'
        },
        {
            'name': 'V10-NeuralOnly',
            'model': NeuralOnlyModel(input_dim=9),
            'loss': 'apfd',
            'description': 'Neural component only, no heuristics'
        },
        {
            'name': 'V10-HeuristicOnly',
            'model': HeuristicOnlyModel(),
            'loss': 'apfd',
            'description': 'Heuristic scoring only'
        },
        {
            'name': 'V10-HighAlpha',
            'model': FixedAlphaModel(input_dim=9, fixed_alpha=0.8),
            'loss': 'apfd',
            'description': 'Fixed alpha=0.8 (heavy heuristic bias)'
        },
        {
            'name': 'V10-LowAlpha',
            'model': FixedAlphaModel(input_dim=9, fixed_alpha=0.2),
            'loss': 'apfd',
            'description': 'Fixed alpha=0.2 (heavy neural bias)'
        },
    ]

    results = {}

    logger.info("\n" + "=" * 70)
    logger.info("RUNNING ABLATION EXPERIMENTS")
    logger.info("=" * 70)

    for exp in experiments:
        logger.info(f"\n>>> {exp['name']}: {exp['description']}")

        # Reset seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Train
        train_result = train_model(
            model=exp['model'],
            train_builds=actual_train,
            val_builds=val_builds,
            device=device,
            num_epochs=30,
            lr=1e-3,
            patience=7,
            loss_type=exp['loss']
        )

        # Evaluate on test
        test_metrics = evaluate_model(exp['model'], test_builds, device)

        # Compute improvement
        improvement = (test_metrics['apfd'] - rf_mean) / rf_mean * 100

        results[exp['name']] = {
            'description': exp['description'],
            'test_apfd': test_metrics['apfd'],
            'test_apfd_std': test_metrics['apfd_std'],
            'val_apfd': train_result['best_val_apfd'],
            'alpha': train_result['final_alpha'],
            'improvement_vs_rf': improvement,
            'apfd_values': test_metrics['apfd_values']
        }

        logger.info(f"    Test APFD: {test_metrics['apfd']:.4f} (+/- {test_metrics['apfd_std']:.4f})")
        logger.info(f"    Improvement vs RF: {improvement:+.2f}%")
        logger.info(f"    Alpha: {train_result['final_alpha']:.3f}")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n{'Model':<25} {'APFD':<12} {'vs RF':<10} {'Alpha':<8}")
    logger.info("-" * 55)
    logger.info(f"{'Recently-Failed (base)':<25} {rf_mean:<12.4f} {'---':<10} {'---':<8}")

    for name, res in results.items():
        logger.info(f"{name:<25} {res['test_apfd']:<12.4f} {res['improvement_vs_rf']:+.2f}%{'':<4} {res['alpha']:.3f}")

    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank test vs Recently-Failed)")
    logger.info("=" * 70)

    for name, res in results.items():
        if len(res['apfd_values']) >= 5 and len(rf_apfds) >= 5:
            # Use min length for paired test
            min_len = min(len(res['apfd_values']), len(rf_apfds))
            stat, p_value = stats.wilcoxon(
                res['apfd_values'][:min_len],
                rf_apfds[:min_len],
                alternative='greater'
            )
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            logger.info(f"{name:<25} p={p_value:.4f} {significance}")

    # Key findings
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)

    best_model = max(results.items(), key=lambda x: x[1]['test_apfd'])
    logger.info(f"\n1. Best model: {best_model[0]} (APFD={best_model[1]['test_apfd']:.4f})")

    # Compare neural-only vs full
    if 'V10-NeuralOnly' in results and 'V10-Full-APFD' in results:
        neural_apfd = results['V10-NeuralOnly']['test_apfd']
        full_apfd = results['V10-Full-APFD']['test_apfd']
        contrib = (full_apfd - neural_apfd) / neural_apfd * 100
        logger.info(f"\n2. Heuristic residual contribution: {contrib:+.2f}%")
        logger.info(f"   (Neural-only: {neural_apfd:.4f} vs Full: {full_apfd:.4f})")

    # Compare loss functions
    if 'V10-Full-APFD' in results and 'V10-Full-Lambda' in results:
        apfd_loss_result = results['V10-Full-APFD']['test_apfd']
        lambda_loss_result = results['V10-Full-Lambda']['test_apfd']
        loss_diff = (apfd_loss_result - lambda_loss_result) / lambda_loss_result * 100
        better_loss = 'APFD' if apfd_loss_result > lambda_loss_result else 'LambdaRank'
        logger.info(f"\n3. Loss function comparison:")
        logger.info(f"   APFD Loss: {apfd_loss_result:.4f}")
        logger.info(f"   LambdaRank Loss: {lambda_loss_result:.4f}")
        logger.info(f"   Better: {better_loss}")

    # Alpha analysis
    logger.info(f"\n4. Learned alpha values:")
    for name, res in results.items():
        if 'Learnable' in name or 'Full' in name:
            logger.info(f"   {name}: α = {res['alpha']:.3f}")

    # Save results
    results_dir = Path("results/ablation_study")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {k: v if not isinstance(v, (np.ndarray, list)) or k != 'apfd_values'
                                       else [float(x) for x in v]
                                       for k, v in res.items()}

    with open(results_dir / 'ablation_results.json', 'w') as f:
        json.dump({
            'baseline_rf': rf_mean,
            'experiments': serializable_results
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_dir}")
    logger.info("=" * 70)

    return results


if __name__ == '__main__':
    run_ablation()
