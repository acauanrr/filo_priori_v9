#!/usr/bin/env python3
"""
Filo-Priori V10 Optimized - Industrial Dataset.

This version incorporates the best practices from V9 that achieved APFD=0.64:
1. Sophisticated structural features (similar to v2.5)
2. Deeper model architecture
3. Combined ranking + classification loss
4. Better normalization and regularization

Usage:
    python run_v10_optimized.py --dataset industry
    python run_v10_optimized.py --dataset rtptorrent

Author: Filo-Priori Team
Version: 10.1.0
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SOPHISTICATED FEATURE EXTRACTOR (V2.5 Style)
# =============================================================================

class AdvancedFeatureExtractor:
    """
    Advanced feature extractor inspired by V9's v2.5 structural features.

    Features extracted:
    1. test_age - How long the test has existed
    2. failure_rate - Historical failure rate
    3. recent_failure_rate - Failure rate in recent window
    4. very_recent_failure_rate - Failure rate in very recent window
    5. flakiness_rate - Rate of status changes
    6. consecutive_failures - Current consecutive failures
    7. max_consecutive_failures - Historical max consecutive failures
    8. failure_trend - Trend in failure rate (increasing/decreasing)
    9. test_novelty - How new the test is (inverse of age)
    10. execution_count - Total number of executions
    11. time_since_last_failure - Recency of last failure
    12. failure_velocity - Rate of failure accumulation
    """

    def __init__(self, recent_window: int = 5, very_recent_window: int = 2):
        self.recent_window = recent_window
        self.very_recent_window = very_recent_window
        self.tc_history = defaultdict(lambda: {
            'results': [],
            'first_seen_build': None,
            'build_indices': []
        })
        self.build_index = 0
        self.feature_stats = None

    def fit(self, df: pd.DataFrame):
        """Build history from training data."""
        logger.info("Building test case history...")

        # Sort by build date if available
        if 'Build_Test_Start_Date' in df.columns:
            df = df.copy()
            df['Build_Test_Start_Date'] = pd.to_datetime(df['Build_Test_Start_Date'], errors='coerce')
            df = df.sort_values('Build_Test_Start_Date')

        # Get unique builds in order
        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i for i, b in enumerate(builds['Build_ID'])}

        # Process each row
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building history"):
            tc_key = row['TC_Key']
            result = row['is_failure']
            build_idx = build_order.get(row['Build_ID'], self.build_index)

            history = self.tc_history[tc_key]
            if history['first_seen_build'] is None:
                history['first_seen_build'] = build_idx

            history['results'].append(result)
            history['build_indices'].append(build_idx)

        self.build_index = max(build_order.values()) + 1 if build_order else 0
        logger.info(f"History built: {len(self.tc_history)} test cases, {self.build_index} builds")

    def extract_features(self, tc_key: str, current_build_idx: int) -> np.ndarray:
        """Extract features for a single test case."""
        history = self.tc_history.get(tc_key, None)

        if history is None or len(history['results']) == 0:
            # New test - use defaults
            return np.array([
                0.0,   # test_age
                0.0,   # failure_rate
                0.0,   # recent_failure_rate
                0.0,   # very_recent_failure_rate
                0.0,   # flakiness_rate
                0.0,   # consecutive_failures
                0.0,   # max_consecutive_failures
                0.0,   # failure_trend
                1.0,   # test_novelty (new test)
                0.0,   # execution_count
                999.0, # time_since_last_failure
                0.0,   # failure_velocity
            ], dtype=np.float32)

        results = history['results']
        build_indices = history['build_indices']
        first_seen = history['first_seen_build']

        # Basic counts
        total_executions = len(results)
        total_failures = sum(results)

        # 1. Test age (builds since first seen)
        test_age = current_build_idx - first_seen if first_seen is not None else 0

        # 2. Overall failure rate
        failure_rate = total_failures / total_executions if total_executions > 0 else 0

        # 3. Recent failure rate (last N executions)
        recent = results[-self.recent_window:]
        recent_failure_rate = sum(recent) / len(recent) if recent else 0

        # 4. Very recent failure rate (last 2 executions)
        very_recent = results[-self.very_recent_window:]
        very_recent_failure_rate = sum(very_recent) / len(very_recent) if very_recent else 0

        # 5. Flakiness rate (status changes)
        if len(results) > 1:
            changes = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
            flakiness_rate = changes / (len(results) - 1)
        else:
            flakiness_rate = 0

        # 6. Consecutive failures (current streak)
        consecutive_failures = 0
        for r in reversed(results):
            if r == 1:
                consecutive_failures += 1
            else:
                break

        # 7. Max consecutive failures (historical)
        max_consecutive = 0
        current_streak = 0
        for r in results:
            if r == 1:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0

        # 8. Failure trend (compare recent vs older)
        if len(results) >= self.recent_window * 2:
            older = results[:-self.recent_window]
            newer = results[-self.recent_window:]
            older_rate = sum(older) / len(older) if older else 0
            newer_rate = sum(newer) / len(newer) if newer else 0
            failure_trend = newer_rate - older_rate  # Positive = increasing failures
        else:
            failure_trend = 0

        # 9. Test novelty (inverse of age, normalized)
        test_novelty = 1.0 / (1.0 + test_age) if test_age >= 0 else 1.0

        # 10. Execution count (log transformed)
        execution_count = np.log1p(total_executions)

        # 11. Time since last failure
        time_since_last = 999
        for i, r in enumerate(reversed(results)):
            if r == 1:
                time_since_last = i
                break

        # 12. Failure velocity (failures per build)
        failure_velocity = total_failures / (test_age + 1) if test_age >= 0 else 0

        return np.array([
            test_age,
            failure_rate,
            recent_failure_rate,
            very_recent_failure_rate,
            flakiness_rate,
            consecutive_failures,
            max_consecutive,
            failure_trend,
            test_novelty,
            execution_count,
            time_since_last,
            failure_velocity,
        ], dtype=np.float32)

    def transform(self, df: pd.DataFrame, update_history: bool = False) -> pd.DataFrame:
        """Transform dataframe with advanced features."""
        logger.info("Extracting advanced features...")

        # Get build order
        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds['Build_Test_Start_Date'] = pd.to_datetime(builds['Build_Test_Start_Date'], errors='coerce')
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i + self.build_index for i, b in enumerate(builds['Build_ID'])}

        features_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            tc_key = row['TC_Key']
            build_idx = build_order.get(row['Build_ID'], self.build_index)

            features = self.extract_features(tc_key, build_idx)
            features_list.append({
                'Build_ID': row['Build_ID'],
                'TC_Key': tc_key,
                'is_failure': row['is_failure'],
                'test_age': features[0],
                'failure_rate': features[1],
                'recent_failure_rate': features[2],
                'very_recent_failure_rate': features[3],
                'flakiness_rate': features[4],
                'consecutive_failures': features[5],
                'max_consecutive_failures': features[6],
                'failure_trend': features[7],
                'test_novelty': features[8],
                'execution_count': features[9],
                'time_since_last_failure': features[10],
                'failure_velocity': features[11],
            })

            # Update history if requested (for sequential processing)
            if update_history:
                history = self.tc_history[tc_key]
                if history['first_seen_build'] is None:
                    history['first_seen_build'] = build_idx
                history['results'].append(row['is_failure'])
                history['build_indices'].append(build_idx)

        result_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(result_df)} rows with 12 features")

        return result_df


# =============================================================================
# OPTIMIZED V10 MODEL
# =============================================================================

class FiloPrioriV10Optimized(nn.Module):
    """
    Optimized V10 model with deeper architecture and better fusion.

    Architecture:
    - Feature encoder with residual connections
    - Heuristic scorer with learned weights
    - Adaptive alpha fusion
    - Combined ranking + classification head
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim

        # Feature normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Deep feature encoder with residual
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual projection
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

        # Attention over features
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Heuristic scorer (key features: failure_rate, recent_failure_rate, consecutive_failures)
        self.heuristic_scorer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # Neural scorer
        self.neural_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Learnable alpha for fusion (input-dependent)
        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Global alpha baseline
        self.alpha_baseline = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
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
            scores: [batch, num_tests]
            alpha: [batch, num_tests] - fusion weights
        """
        batch_size, num_tests, _ = features.shape

        # Flatten for processing
        flat_features = features.view(-1, self.input_dim)

        # Normalize features
        if flat_features.size(0) > 1:
            normed_features = self.input_norm(flat_features)
        else:
            normed_features = flat_features

        # Feature attention
        attention = self.feature_attention(flat_features)
        weighted_features = flat_features * attention

        # Encode with residual
        encoded = self.encoder(normed_features)
        residual = self.residual_proj(normed_features)
        encoded = encoded + residual

        # Neural score
        neural_score = self.neural_scorer(encoded).view(batch_size, num_tests)

        # Heuristic score from key features
        # Indices: failure_rate=1, recent_failure_rate=2, consecutive_failures=5, time_since_last=10
        heuristic_features = torch.stack([
            features[:, :, 1],   # failure_rate
            features[:, :, 2],   # recent_failure_rate
            features[:, :, 5],   # consecutive_failures (normalized)
            1.0 / (features[:, :, 10] + 1),  # recency score
        ], dim=-1)

        heuristic_score = self.heuristic_scorer(heuristic_features).squeeze(-1)

        # Input-dependent alpha
        alpha_dynamic = self.alpha_net(flat_features).view(batch_size, num_tests)
        alpha = 0.5 * self.alpha_baseline + 0.5 * alpha_dynamic

        # Fusion
        scores = alpha * heuristic_score + (1 - alpha) * neural_score

        return scores, alpha

    def get_alpha(self) -> float:
        """Get baseline alpha value."""
        return self.alpha_baseline.item()


# =============================================================================
# COMBINED LOSS
# =============================================================================

class CombinedRankingLoss(nn.Module):
    """
    Combined loss: LambdaRank + ListMLE + Focal-style weighting.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        listmle_weight: float = 0.3,
        focal_weight: float = 0.2,
        sigma: float = 1.0
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.listmle_weight = listmle_weight
        self.focal_weight = focal_weight
        self.sigma = sigma

    def forward(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        batch_size, num_items = scores.shape
        device = scores.device

        total_loss = scores.sum() * 0.0  # Ensure gradient

        for b in range(batch_size):
            s, y = scores[b], relevances[b]

            # 1. LambdaRank loss
            pos_idx = (y == 1).nonzero(as_tuple=True)[0]
            neg_idx = (y == 0).nonzero(as_tuple=True)[0]

            if len(pos_idx) > 0 and len(neg_idx) > 0:
                pos_scores = s[pos_idx]
                neg_scores = s[neg_idx]
                diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                lambda_loss = F.softplus(-self.sigma * diff).mean()
                total_loss = total_loss + self.lambda_weight * lambda_loss

            # 2. ListMLE loss (softmax ranking)
            if y.sum() > 0:
                # Sort by relevance
                sorted_idx = y.argsort(descending=True)
                sorted_scores = s[sorted_idx]

                # ListMLE: negative log likelihood of correct ordering
                listmle = 0.0
                for i in range(len(sorted_scores) - 1):
                    remaining = sorted_scores[i:]
                    listmle = listmle - sorted_scores[i] + torch.logsumexp(remaining, dim=0)
                listmle = listmle / max(len(sorted_scores) - 1, 1)
                total_loss = total_loss + self.listmle_weight * listmle

            # 3. Focal-style loss for hard examples
            if len(pos_idx) > 0:
                # Focus on failures that are ranked low
                pos_scores_normalized = torch.sigmoid(s[pos_idx])
                focal_loss = -((1 - pos_scores_normalized) ** 2 * torch.log(pos_scores_normalized + 1e-8)).mean()
                total_loss = total_loss + self.focal_weight * focal_loss

        return total_loss / max(batch_size, 1)


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_industry_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess industrial dataset."""
    logger.info("Loading industrial dataset...")

    train_df = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test_df = pd.read_csv(data_dir / "test.csv", low_memory=False)

    # Convert result to binary
    train_df['is_failure'] = (train_df['TE_Test_Result'] == 'Fail').astype(int)
    test_df['is_failure'] = (test_df['TE_Test_Result'] == 'Fail').astype(int)

    logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

    return train_df, test_df


def load_rtptorrent_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed RTPTorrent dataset."""
    logger.info("Loading RTPTorrent dataset...")

    processed_dir = data_dir / "processed_ranking"
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

    return train_df, test_df


def create_builds(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
    """Group data by build."""
    builds = []

    for build_id, group in df.groupby('Build_ID'):
        features = group[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip extreme values
        features = np.clip(features, -100, 100)

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
    """Evaluate model."""
    model.eval()
    apfds = []

    with torch.no_grad():
        for build in builds:
            if build['num_failures'] == 0:
                continue

            features = build['features'].unsqueeze(0).to(device)
            labels = build['labels'].to(device)

            scores, _ = model(features)
            scores = scores.squeeze(0)

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

def baseline_recently_failed(builds: List[Dict], feature_idx: int = 10) -> List[float]:
    """Recently-failed baseline."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        # Use time_since_last_failure (index 10)
        recency = build['features'][:, feature_idx]
        _, indices = recency.sort()  # Lower = more recent = higher priority
        rankings = torch.zeros_like(recency, dtype=torch.long)
        rankings[indices] = torch.arange(len(recency))
        apfd = compute_apfd(rankings, build['labels'])
        apfds.append(apfd)
    return apfds


def baseline_failure_rate(builds: List[Dict], feature_idx: int = 1) -> List[float]:
    """Failure rate baseline."""
    apfds = []
    for build in builds:
        if build['num_failures'] == 0:
            continue
        fail_rate = build['features'][:, feature_idx]
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
    num_epochs: int = 50,
    lr: float = 3e-4,
    patience: int = 15
) -> Dict:
    """Train the optimized model."""
    model = model.to(device)

    criterion = CombinedRankingLoss(
        lambda_weight=0.5,
        listmle_weight=0.3,
        focal_weight=0.2
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
    history = {'train_loss': [], 'val_apfd': [], 'alpha': []}

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

            # Gradient clipping
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
        history['alpha'].append(model.get_alpha())

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val APFD: {val_metrics['apfd']:.4f} (+/- {val_metrics['apfd_std']:.4f}) - "
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

def run_experiment(dataset_type: str, base_dir: Path) -> Dict:
    """Run optimized experiment."""
    logger.info("=" * 70)
    logger.info(f"FILO-PRIORI V10 OPTIMIZED: {dataset_type.upper()}")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    if dataset_type == 'industry':
        data_dir = base_dir / "datasets/01_industry"
        train_df, test_df = load_industry_data(data_dir)
    else:
        data_dir = base_dir / "datasets/02_rtptorrent"
        train_df, test_df = load_rtptorrent_data(data_dir)

    # Extract advanced features
    logger.info("\nExtracting advanced features...")
    extractor = AdvancedFeatureExtractor(recent_window=5, very_recent_window=2)
    extractor.fit(train_df)

    train_features_df = extractor.transform(train_df, update_history=False)
    test_features_df = extractor.transform(test_df, update_history=False)

    # Feature columns
    feature_cols = [
        'test_age', 'failure_rate', 'recent_failure_rate', 'very_recent_failure_rate',
        'flakiness_rate', 'consecutive_failures', 'max_consecutive_failures',
        'failure_trend', 'test_novelty', 'execution_count', 'time_since_last_failure',
        'failure_velocity'
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
    logger.info("\nComputing baselines...")
    rf_apfds = baseline_recently_failed(test_builds, feature_idx=10)  # time_since_last_failure
    fr_apfds = baseline_failure_rate(test_builds, feature_idx=1)  # failure_rate

    logger.info(f"  Recently-Failed: APFD = {np.mean(rf_apfds):.4f} (+/- {np.std(rf_apfds):.4f})")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(fr_apfds):.4f} (+/- {np.std(fr_apfds):.4f})")

    # Create and train model
    logger.info("\nTraining V10 Optimized model...")
    model = FiloPrioriV10Optimized(
        input_dim=len(feature_cols),
        hidden_dim=128,
        dropout=0.2
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_result = train_model(
        model=model,
        train_builds=actual_train,
        val_builds=val_builds,
        device=device,
        num_epochs=50,
        lr=3e-4,
        patience=15
    )

    # Evaluate
    logger.info("\nEvaluating on TEST set...")
    test_metrics = evaluate_model(model, test_builds, device)

    # Compute improvement
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

    logger.info("\n" + "=" * 70)
    logger.info(f"FINAL RESULTS - {dataset_type.upper()}")
    logger.info("=" * 70)
    logger.info(f"V10 Optimized:     APFD = {v10_apfd:.4f} (+/- {test_metrics['apfd_std']:.4f})")
    logger.info(f"Recently-Failed:   APFD = {rf_mean:.4f}")
    logger.info(f"Improvement:       {improvement:+.2f}%")
    logger.info(f"p-value:           {p_value:.4f}")
    logger.info(f"Alpha (learned):   {train_result['final_alpha']:.3f}")
    logger.info("=" * 70)

    # Save results
    results_dir = base_dir / f"results/experiment_v10_optimized_{dataset_type}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': dataset_type,
        'v10_optimized': {
            'apfd': float(v10_apfd),
            'apfd_std': float(test_metrics['apfd_std']),
            'num_builds': test_metrics['num_builds'],
            'alpha': train_result['final_alpha']
        },
        'baselines': {
            'recently_failed': {'apfd': float(rf_mean), 'std': float(np.std(rf_apfds))},
            'failure_rate': {'apfd': float(np.mean(fr_apfds)), 'std': float(np.std(fr_apfds))}
        },
        'improvement_vs_rf': improvement,
        'p_value': float(p_value)
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'input_dim': len(feature_cols), 'hidden_dim': 128},
        'results': results
    }, results_dir / 'best_model.pt')

    # Save per-build APFD
    apfd_df = pd.DataFrame({
        'build_idx': range(len(test_metrics['apfd_values'])),
        'apfd': test_metrics['apfd_values']
    })
    apfd_df.to_csv(results_dir / 'apfd_per_build.csv', index=False)

    logger.info(f"Results saved to {results_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Filo-Priori V10 Optimized')
    parser.add_argument('--dataset', '-d', type=str, choices=['industry', 'rtptorrent', 'all'],
                       default='industry', help='Dataset to run on')
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

    return all_results


if __name__ == '__main__':
    main()
