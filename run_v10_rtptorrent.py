#!/usr/bin/env python3
"""
Filo-Priori V10 - RTPTorrent Dataset Pipeline.

This script implements the V10 architecture optimized for the RTPTorrent dataset:
1. Advanced ranking features with time-decay weighting
2. LambdaRank + ListMLE combined loss
3. Residual learning with heuristic bias
4. Deep feature encoder with attention

Target: Surpass Recently-Failed baseline by 15%+

Usage:
    python run_v10_rtptorrent.py

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
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


# =============================================================================
# ADVANCED RANKING FEATURES
# =============================================================================

class RankingFeatureExtractor:
    """
    Extract ranking-optimized features for test case prioritization.

    Features designed for ranking (not classification):
    1. Historical failure metrics
    2. Recency-weighted scores
    3. Trend indicators
    4. Volatility measures
    """

    def __init__(self, decay_lambda: float = 0.1, recent_window: int = 5):
        self.decay_lambda = decay_lambda
        self.recent_window = recent_window
        self.tc_history = defaultdict(lambda: {
            'results': [],
            'timestamps': [],
            'build_indices': []
        })
        self.max_build_idx = 0

    def fit(self, df: pd.DataFrame):
        """Build history from training data."""
        logger.info("Building test case history for ranking features...")

        # Sort by time if available
        df = df.copy()
        if 'Build_Test_Start_Date' in df.columns:
            df['Build_Test_Start_Date'] = pd.to_datetime(df['Build_Test_Start_Date'], errors='coerce')
            df = df.sort_values('Build_Test_Start_Date')

        # Get build order
        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i for i, b in enumerate(builds['Build_ID'])}
        self.max_build_idx = max(build_order.values()) if build_order else 0

        # Build history
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building history"):
            tc_key = row['TC_Key']
            result = row['is_failure']
            build_idx = build_order.get(row['Build_ID'], 0)

            self.tc_history[tc_key]['results'].append(result)
            self.tc_history[tc_key]['build_indices'].append(build_idx)

        logger.info(f"History built: {len(self.tc_history)} test cases")

    def extract_features(self, tc_key: str, current_build: int) -> np.ndarray:
        """Extract ranking features for a test case."""
        history = self.tc_history.get(tc_key)

        if history is None or len(history['results']) == 0:
            # New test - high priority (unknown risk)
            return np.array([
                1.0,    # novelty_score (new test)
                0.5,    # base_risk (unknown)
                0.0,    # historical_failure_rate
                0.0,    # recent_failure_rate
                0.0,    # very_recent_failure_rate
                0.0,    # time_decay_score
                0.0,    # consecutive_failures
                0.0,    # failure_trend
                1.0,    # recency_score (recent = new)
                0.0,    # volatility
                0.0,    # max_consecutive_failures
                0.5,    # ranking_prior (neutral)
            ], dtype=np.float32)

        results = history['results']
        build_indices = history['build_indices']

        # Basic stats
        total = len(results)
        failures = sum(results)

        # 1. Novelty score (inverse of age)
        age = current_build - min(build_indices) if build_indices else 0
        novelty_score = 1.0 / (1.0 + age)

        # 2. Base risk (prior probability)
        base_risk = (failures + 1) / (total + 2)  # Laplace smoothing

        # 3. Historical failure rate
        historical_rate = failures / total if total > 0 else 0

        # 4. Recent failure rate
        recent = results[-self.recent_window:]
        recent_rate = sum(recent) / len(recent) if recent else 0

        # 5. Very recent failure rate (last 2)
        very_recent = results[-2:]
        very_recent_rate = sum(very_recent) / len(very_recent) if very_recent else 0

        # 6. Time-decay weighted score
        time_decay_score = 0.0
        for i, (r, b_idx) in enumerate(zip(results, build_indices)):
            if r == 1:  # Failure
                delta = current_build - b_idx
                weight = np.exp(-self.decay_lambda * delta)
                time_decay_score += weight
        time_decay_score = min(time_decay_score, 10.0)  # Clip

        # 7. Consecutive failures (current streak)
        consecutive = 0
        for r in reversed(results):
            if r == 1:
                consecutive += 1
            else:
                break

        # 8. Failure trend (recent vs old)
        if total >= 6:
            old_rate = sum(results[:total//2]) / (total//2)
            new_rate = sum(results[total//2:]) / (total - total//2)
            trend = new_rate - old_rate
        else:
            trend = 0.0

        # 9. Recency score (time since last failure)
        recency = 999
        for i, r in enumerate(reversed(results)):
            if r == 1:
                recency = i
                break
        recency_score = 1.0 / (1.0 + recency)

        # 10. Volatility (rate of status changes)
        if len(results) > 1:
            changes = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
            volatility = changes / (len(results) - 1)
        else:
            volatility = 0

        # 11. Max consecutive failures
        max_consec = 0
        current_streak = 0
        for r in results:
            if r == 1:
                current_streak += 1
                max_consec = max(max_consec, current_streak)
            else:
                current_streak = 0

        # 12. Ranking prior (combined heuristic score)
        ranking_prior = 0.4 * recent_rate + 0.3 * time_decay_score / 10 + 0.3 * recency_score

        return np.array([
            novelty_score,
            base_risk,
            historical_rate,
            recent_rate,
            very_recent_rate,
            time_decay_score / 10,  # Normalized
            min(consecutive / 5, 1.0),  # Normalized
            trend,
            recency_score,
            volatility,
            min(max_consec / 5, 1.0),  # Normalized
            ranking_prior,
        ], dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with ranking features."""
        logger.info("Extracting ranking features...")

        # Get build order
        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds['Build_Test_Start_Date'] = pd.to_datetime(builds['Build_Test_Start_Date'], errors='coerce')
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i + self.max_build_idx for i, b in enumerate(builds['Build_ID'])}

        feature_names = [
            'novelty_score', 'base_risk', 'historical_rate', 'recent_rate',
            'very_recent_rate', 'time_decay_score', 'consecutive_failures',
            'failure_trend', 'recency_score', 'volatility',
            'max_consecutive_failures', 'ranking_prior'
        ]

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            build_idx = build_order.get(row['Build_ID'], self.max_build_idx)
            features = self.extract_features(row['TC_Key'], build_idx)

            row_dict = {
                'Build_ID': row['Build_ID'],
                'TC_Key': row['TC_Key'],
                'is_failure': row['is_failure'],
            }
            for name, value in zip(feature_names, features):
                row_dict[name] = value

            rows.append(row_dict)

        return pd.DataFrame(rows)


# =============================================================================
# V10 RANKING MODEL
# =============================================================================

class V10RankingModel(nn.Module):
    """
    V10 Ranking Model with Residual Learning.

    Architecture:
    - Deep feature encoder with attention
    - Heuristic scorer (domain knowledge)
    - Learnable residual fusion
    - Ranking-optimized output
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Feature attention (learn which features matter)
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Deep encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention over tests
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Neural ranking scorer
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Heuristic scorer (uses ranking_prior and key features)
        self.heuristic_scorer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # Learnable alpha for residual fusion
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch, num_tests, input_dim]

        Returns:
            scores: [batch, num_tests] - ranking scores
            alpha: scalar - fusion weight
        """
        batch_size, num_tests, _ = features.shape

        # Normalize input
        x = self.input_norm(features)

        # Feature attention
        attn_weights = self.feature_attention(x)
        x_weighted = x * attn_weights

        # Encode
        x_flat = x_weighted.view(-1, self.input_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, num_tests, -1)

        # Self-attention over tests
        if num_tests > 1:
            attended, _ = self.self_attention(encoded, encoded, encoded)
            encoded = encoded + attended  # Residual

        # Neural score
        neural_score = self.neural_scorer(encoded).squeeze(-1)

        # Heuristic score (from key features)
        # Indices: ranking_prior=11, recency_score=8, recent_rate=3, time_decay=5
        heuristic_features = torch.stack([
            features[:, :, 11],  # ranking_prior
            features[:, :, 8],   # recency_score
            features[:, :, 3],   # recent_rate
            features[:, :, 5],   # time_decay_score
        ], dim=-1)

        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)

        # Residual fusion
        alpha = torch.sigmoid(self.alpha_logit)
        scores = alpha * heuristic_score + (1 - alpha) * neural_score

        return scores, alpha

    def get_alpha(self) -> float:
        return torch.sigmoid(self.alpha_logit).item()


# =============================================================================
# COMBINED RANKING LOSS
# =============================================================================

class CombinedRankingLoss(nn.Module):
    """
    Combined loss for ranking optimization:
    - LambdaRank (pairwise)
    - ListMLE (listwise)
    - ApproxNDCG (metric-focused)
    """

    def __init__(
        self,
        lambda_weight: float = 0.4,
        listmle_weight: float = 0.3,
        approx_ndcg_weight: float = 0.3,
        sigma: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.listmle_weight = listmle_weight
        self.approx_ndcg_weight = approx_ndcg_weight
        self.sigma = sigma
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        batch_size, num_items = scores.shape
        device = scores.device

        total_loss = scores.sum() * 0.0

        for b in range(batch_size):
            s, y = scores[b], relevances[b]
            num_failures = y.sum().item()

            if num_failures == 0:
                continue

            # 1. LambdaRank loss
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_idx) > 0 and len(neg_idx) > 0:
                pos_scores = s[pos_idx]
                neg_scores = s[neg_idx]
                diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                lambda_loss = F.softplus(-self.sigma * diff).mean()
                total_loss = total_loss + self.lambda_weight * lambda_loss

            # 2. ListMLE loss
            sorted_idx = y.argsort(descending=True)
            sorted_scores = s[sorted_idx]

            listmle_loss = 0.0
            for i in range(min(len(sorted_scores) - 1, 20)):  # Top-20
                remaining = sorted_scores[i:]
                listmle_loss = listmle_loss - sorted_scores[i] + torch.logsumexp(remaining, dim=0)
            listmle_loss = listmle_loss / max(min(len(sorted_scores) - 1, 20), 1)
            total_loss = total_loss + self.listmle_weight * listmle_loss

            # 3. ApproxNDCG loss
            soft_rank = torch.sigmoid((s.unsqueeze(0) - s.unsqueeze(1)) / self.temperature).sum(dim=1)
            dcg = (y / torch.log2(soft_rank + 2)).sum()
            ideal_dcg = (y.sort(descending=True)[0] / torch.log2(torch.arange(1, len(y) + 1, device=device).float() + 1)).sum()
            ndcg = dcg / (ideal_dcg + 1e-8)
            approx_ndcg_loss = 1 - ndcg
            total_loss = total_loss + self.approx_ndcg_weight * approx_ndcg_loss

        return total_loss / max(batch_size, 1)


# =============================================================================
# METRICS
# =============================================================================

def compute_apfd(rankings: torch.Tensor, labels: torch.Tensor) -> float:
    n = len(labels)
    m = labels.sum().item()
    if m == 0:
        return 1.0
    fail_positions = (rankings[labels == 1] + 1).float()
    apfd = 1 - fail_positions.sum().item() / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


def compute_ndcg(scores: torch.Tensor, relevances: torch.Tensor, k: int = 10) -> float:
    """Compute NDCG@k."""
    device = scores.device
    n = len(scores)
    k = min(k, n)

    # Get top-k by score
    _, indices = scores.topk(k)
    top_relevances = relevances[indices]

    # DCG
    discounts = torch.log2(torch.arange(2, k + 2, device=device).float())
    dcg = (top_relevances / discounts).sum().item()

    # Ideal DCG
    sorted_rel = relevances.sort(descending=True)[0][:k]
    idcg = (sorted_rel / discounts[:len(sorted_rel)]).sum().item()

    return dcg / (idcg + 1e-8)


def evaluate_model(model: nn.Module, builds: List[Dict], device: torch.device) -> Dict:
    """Evaluate model on builds."""
    model.eval()
    apfds = []
    ndcgs = []

    with torch.no_grad():
        for build in builds:
            if build['num_failures'] == 0:
                continue

            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].to(device)

            scores, _ = model(features)
            scores = scores.squeeze(0)

            # Rankings
            _, indices = scores.sort(descending=True)
            rankings = torch.zeros_like(scores, dtype=torch.long)
            rankings[indices] = torch.arange(len(scores), device=device)

            apfd = compute_apfd(rankings, labels)
            ndcg = compute_ndcg(scores, labels, k=10)

            apfds.append(apfd)
            ndcgs.append(ndcg)

    return {
        'apfd': np.mean(apfds) if apfds else 0.0,
        'apfd_std': np.std(apfds) if apfds else 0.0,
        'ndcg_at_10': np.mean(ndcgs) if ndcgs else 0.0,
        'apfd_values': apfds,
        'num_builds': len(apfds)
    }


# =============================================================================
# BASELINES
# =============================================================================

def baseline_recently_failed(builds: List[Dict]) -> List[float]:
    """Recently-failed baseline using recency_score."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        # Use recency_score (index 8)
        recency = build['features'][:, 8]
        _, indices = recency.sort(descending=True)  # Higher recency = higher priority
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
        fail_rate = build['features'][:, 2]  # historical_rate
        _, indices = fail_rate.sort(descending=True)
        rankings = torch.zeros_like(fail_rate, dtype=torch.long)
        rankings[indices] = torch.arange(len(fail_rate))
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rtptorrent_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load RTPTorrent processed data."""
    data_dir = BASE_DIR / "datasets" / "02_rtptorrent" / "processed_ranking"

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

    return train_df, test_df


def create_builds(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """Create build batches."""
    builds = []

    for build_id, group in df.groupby('Build_ID'):
        features = group[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
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
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_builds: List[Dict],
    val_builds: List[Dict],
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10
) -> Dict:
    """Train the V10 model."""
    model = model.to(device)

    criterion = CombinedRankingLoss(
        lambda_weight=0.4,
        listmle_weight=0.3,
        approx_ndcg_weight=0.3
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    train_with_failures = [b for b in train_builds if b['num_failures'] > 0]
    logger.info(f"Training on {len(train_with_failures)} builds with failures")

    best_apfd = 0.0
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_apfd': [], 'val_ndcg': [], 'alpha': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        np.random.shuffle(train_with_failures)

        pbar = tqdm(train_with_failures, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for build in pbar:
            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].unsqueeze(0).to(device)

            optimizer.zero_grad()
            scores, alpha = model(features)
            loss = criterion(scores, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        # Validation
        val_metrics = evaluate_model(model, val_builds, device)

        history['train_loss'].append(avg_loss)
        history['val_apfd'].append(val_metrics['apfd'])
        history['val_ndcg'].append(val_metrics['ndcg_at_10'])
        history['alpha'].append(model.get_alpha())

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val APFD: {val_metrics['apfd']:.4f} - "
            f"Val NDCG@10: {val_metrics['ndcg_at_10']:.4f} - "
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
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("FILO-PRIORI V10 - RTPTORRENT DATASET")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Model: V10 Ranking Model")
    logger.info("  - Features: 12 ranking-optimized features")
    logger.info("  - Loss: LambdaRank + ListMLE + ApproxNDCG")
    logger.info("  - Fusion: Residual learning with attention")
    logger.info("")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    logger.info("\n[1/5] Loading data...")
    train_df, test_df = load_rtptorrent_data()

    # Extract ranking features
    logger.info("\n[2/5] Extracting ranking features...")
    extractor = RankingFeatureExtractor(decay_lambda=0.1, recent_window=5)
    extractor.fit(train_df)

    train_features_df = extractor.transform(train_df)
    test_features_df = extractor.transform(test_df)

    feature_cols = [
        'novelty_score', 'base_risk', 'historical_rate', 'recent_rate',
        'very_recent_rate', 'time_decay_score', 'consecutive_failures',
        'failure_trend', 'recency_score', 'volatility',
        'max_consecutive_failures', 'ranking_prior'
    ]

    # Create builds
    train_builds = create_builds(train_features_df, feature_cols)
    test_builds = create_builds(test_features_df, feature_cols)

    train_with_failures = len([b for b in train_builds if b['num_failures'] > 0])
    test_with_failures = len([b for b in test_builds if b['num_failures'] > 0])

    logger.info(f"\nTrain builds: {len(train_builds)} ({train_with_failures} with failures)")
    logger.info(f"Test builds: {len(test_builds)} ({test_with_failures} with failures)")

    # Split train/val
    np.random.shuffle(train_builds)
    split_idx = int(len(train_builds) * 0.85)
    actual_train = train_builds[:split_idx]
    val_builds = train_builds[split_idx:]

    logger.info(f"Train/Val split: {len(actual_train)}/{len(val_builds)}")

    # Baselines
    logger.info("\n[3/5] Computing baselines...")
    rf_apfds = baseline_recently_failed(test_builds)
    fr_apfds = baseline_failure_rate(test_builds)

    logger.info(f"  Recently-Failed: APFD = {np.mean(rf_apfds):.4f} (+/- {np.std(rf_apfds):.4f})")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(fr_apfds):.4f} (+/- {np.std(fr_apfds):.4f})")

    # Train model
    logger.info("\n[4/5] Training V10 model...")
    model = V10RankingModel(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_heads=4,
        dropout=0.2
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_result = train_model(
        model=model,
        train_builds=actual_train,
        val_builds=val_builds,
        device=device,
        num_epochs=50,
        lr=1e-3,
        patience=10
    )

    # Evaluate
    logger.info("\n[5/5] Evaluating on TEST set...")
    test_metrics = evaluate_model(model, test_builds, device)

    # Compute improvements
    rf_mean = np.mean(rf_apfds)
    v10_apfd = test_metrics['apfd']
    improvement = (v10_apfd - rf_mean) / rf_mean * 100 if rf_mean > 0 else 0

    # Statistical test
    min_len = min(len(test_metrics['apfd_values']), len(rf_apfds))
    if min_len >= 5:
        _, p_value = stats.wilcoxon(
            test_metrics['apfd_values'][:min_len],
            rf_apfds[:min_len],
            alternative='greater'
        )
    else:
        p_value = 1.0

    # Results
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS - RTPTORRENT")
    logger.info("=" * 70)
    logger.info(f"V10 Model:         APFD = {v10_apfd:.4f} (+/- {test_metrics['apfd_std']:.4f})")
    logger.info(f"                   NDCG@10 = {test_metrics['ndcg_at_10']:.4f}")
    logger.info(f"Recently-Failed:   APFD = {rf_mean:.4f}")
    logger.info(f"Improvement:       {improvement:+.2f}%")
    logger.info(f"p-value:           {p_value:.4f}")
    logger.info(f"Alpha (learned):   {train_result['final_alpha']:.3f}")
    logger.info("=" * 70)

    # Save results
    results_dir = BASE_DIR / "results" / "experiment_v10_rtptorrent_ranking"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'rtptorrent',
        'model': 'V10_RankingModel',
        'v10': {
            'apfd': float(v10_apfd),
            'apfd_std': float(test_metrics['apfd_std']),
            'ndcg_at_10': float(test_metrics['ndcg_at_10']),
            'num_builds': test_metrics['num_builds'],
            'alpha': train_result['final_alpha']
        },
        'baselines': {
            'recently_failed': {'apfd': float(rf_mean), 'std': float(np.std(rf_apfds))},
            'failure_rate': {'apfd': float(np.mean(fr_apfds)), 'std': float(np.std(fr_apfds))}
        },
        'improvement_vs_rf': improvement,
        'p_value': float(p_value),
        'config': {
            'features': feature_cols,
            'hidden_dim': 128,
            'num_heads': 4,
            'loss': 'LambdaRank+ListMLE+ApproxNDCG',
            'num_epochs': 50
        }
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': results['config'],
        'results': results
    }, results_dir / 'best_model.pt')

    # Save APFD per build
    apfd_df = pd.DataFrame({
        'build_idx': range(len(test_metrics['apfd_values'])),
        'apfd': test_metrics['apfd_values']
    })
    apfd_df.to_csv(results_dir / 'apfd_per_build.csv', index=False)

    logger.info(f"\nResults saved to {results_dir}")

    return results


if __name__ == '__main__':
    main()
