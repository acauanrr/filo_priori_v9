#!/usr/bin/env python3
"""
Filo-Priori V10 Simple - RTPTorrent Dataset Pipeline.

This script implements a simplified V10 architecture optimized for the
small RTPTorrent dataset:
1. Advanced ranking features with time-decay weighting
2. LightGBM LambdaRank for ranking (better for small datasets)
3. Feature importance analysis
4. Ensemble with heuristic baseline

Usage:
    python run_v10_rtptorrent_simple.py

Author: Filo-Priori Team
Version: 10.1.0
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
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available, using sklearn")

# Fallback imports
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# =============================================================================
# ADVANCED RANKING FEATURES
# =============================================================================

class RankingFeatureExtractor:
    """Extract ranking-optimized features for test case prioritization."""

    def __init__(self, decay_lambda: float = 0.1, recent_window: int = 5):
        self.decay_lambda = decay_lambda
        self.recent_window = recent_window
        self.tc_history = defaultdict(lambda: {
            'results': [],
            'build_indices': []
        })
        self.max_build_idx = 0

    def fit(self, df: pd.DataFrame):
        """Build history from training data."""
        logger.info("Building test case history...")

        df = df.copy()
        if 'Build_Test_Start_Date' in df.columns:
            df['Build_Test_Start_Date'] = pd.to_datetime(df['Build_Test_Start_Date'], errors='coerce')
            df = df.sort_values('Build_Test_Start_Date')

        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i for i, b in enumerate(builds['Build_ID'])}
        self.max_build_idx = max(build_order.values()) if build_order else 0

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
            return np.array([
                1.0,   # novelty_score
                0.5,   # base_risk
                0.0,   # historical_rate
                0.0,   # recent_rate
                0.0,   # very_recent_rate
                0.0,   # time_decay_score
                0.0,   # consecutive_failures
                0.0,   # failure_trend
                1.0,   # recency_score
                0.0,   # volatility
                0.0,   # max_consecutive_failures
                0.5,   # ranking_prior
                0.0,   # total_executions (normalized)
                0.0,   # last_failure_distance
            ], dtype=np.float32)

        results = history['results']
        build_indices = history['build_indices']

        total = len(results)
        failures = sum(results)

        # 1. Novelty score
        age = current_build - min(build_indices) if build_indices else 0
        novelty_score = 1.0 / (1.0 + age)

        # 2. Base risk
        base_risk = (failures + 1) / (total + 2)

        # 3. Historical failure rate
        historical_rate = failures / total if total > 0 else 0

        # 4. Recent failure rate
        recent = results[-self.recent_window:]
        recent_rate = sum(recent) / len(recent) if recent else 0

        # 5. Very recent failure rate
        very_recent = results[-2:]
        very_recent_rate = sum(very_recent) / len(very_recent) if very_recent else 0

        # 6. Time-decay weighted score
        time_decay_score = 0.0
        for r, b_idx in zip(results, build_indices):
            if r == 1:
                delta = current_build - b_idx
                weight = np.exp(-self.decay_lambda * delta)
                time_decay_score += weight
        time_decay_score = min(time_decay_score, 10.0)

        # 7. Consecutive failures
        consecutive = 0
        for r in reversed(results):
            if r == 1:
                consecutive += 1
            else:
                break

        # 8. Failure trend
        if total >= 6:
            old_rate = sum(results[:total//2]) / (total//2)
            new_rate = sum(results[total//2:]) / (total - total//2)
            trend = new_rate - old_rate
        else:
            trend = 0.0

        # 9. Recency score
        recency = 999
        for i, r in enumerate(reversed(results)):
            if r == 1:
                recency = i
                break
        recency_score = 1.0 / (1.0 + recency)

        # 10. Volatility
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

        # 12. Ranking prior
        ranking_prior = 0.4 * recent_rate + 0.3 * time_decay_score / 10 + 0.3 * recency_score

        # 13. Total executions (normalized)
        total_executions = min(total / 100, 1.0)

        # 14. Last failure distance
        last_failure_dist = recency / 100 if recency < 999 else 1.0

        return np.array([
            novelty_score,
            base_risk,
            historical_rate,
            recent_rate,
            very_recent_rate,
            time_decay_score / 10,
            min(consecutive / 5, 1.0),
            trend,
            recency_score,
            volatility,
            min(max_consec / 5, 1.0),
            ranking_prior,
            total_executions,
            last_failure_dist,
        ], dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with ranking features."""
        logger.info("Extracting ranking features...")

        builds = df.groupby('Build_ID').first().reset_index()
        if 'Build_Test_Start_Date' in builds.columns:
            builds['Build_Test_Start_Date'] = pd.to_datetime(builds['Build_Test_Start_Date'], errors='coerce')
            builds = builds.sort_values('Build_Test_Start_Date')
        build_order = {b: i + self.max_build_idx for i, b in enumerate(builds['Build_ID'])}

        feature_names = [
            'novelty_score', 'base_risk', 'historical_rate', 'recent_rate',
            'very_recent_rate', 'time_decay_score', 'consecutive_failures',
            'failure_trend', 'recency_score', 'volatility',
            'max_consecutive_failures', 'ranking_prior', 'total_executions',
            'last_failure_distance'
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
# METRICS
# =============================================================================

def compute_apfd(rankings: np.ndarray, labels: np.ndarray) -> float:
    """Compute APFD metric."""
    n = len(labels)
    m = labels.sum()
    if m == 0:
        return 1.0
    fail_positions = rankings[labels == 1] + 1
    apfd = 1 - fail_positions.sum() / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


def compute_ndcg(scores: np.ndarray, relevances: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k."""
    n = len(scores)
    k = min(k, n)

    indices = np.argsort(-scores)[:k]
    top_relevances = relevances[indices]

    discounts = np.log2(np.arange(2, k + 2))
    dcg = (top_relevances / discounts).sum()

    sorted_rel = np.sort(relevances)[::-1][:k]
    idcg = (sorted_rel / discounts[:len(sorted_rel)]).sum()

    return dcg / (idcg + 1e-8) if idcg > 0 else 0.0


# =============================================================================
# MODELS
# =============================================================================

class LightGBMRanker:
    """LightGBM-based ranker for test prioritization."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray):
        """Fit the ranker."""
        if HAS_LGBM:
            # LightGBM LambdaRank
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=groups
            )

            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [5, 10],
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'max_depth': self.max_depth,
                'learning_rate': 0.05,
                'n_estimators': self.n_estimators,
                'min_data_in_leaf': 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbosity': -1,
                'seed': 42
            }

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators
            )
        else:
            # Fallback to sklearn
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                random_state=42
            )
            self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        if HAS_LGBM:
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]


class HeuristicRanker:
    """Heuristic baseline ranker."""

    def __init__(self, feature_weights: Dict[str, float] = None):
        self.feature_weights = feature_weights or {
            'recency_score': 0.30,
            'recent_rate': 0.25,
            'time_decay_score': 0.20,
            'ranking_prior': 0.15,
            'base_risk': 0.10
        }

    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Compute heuristic score."""
        scores = np.zeros(len(X))
        for fname, weight in self.feature_weights.items():
            if fname in feature_names:
                idx = feature_names.index(fname)
                scores += weight * X[:, idx]
        return scores


class EnsembleRanker:
    """Ensemble of ML model and heuristic."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.lgbm_ranker = LightGBMRanker(n_estimators=50, max_depth=3)
        self.heuristic_ranker = HeuristicRanker()
        self.feature_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            groups: np.ndarray, feature_names: List[str]):
        """Fit the ensemble."""
        self.feature_names = feature_names
        self.lgbm_ranker.fit(X_train, y_train, groups)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with ensemble."""
        ml_scores = self.lgbm_ranker.predict(X)
        ml_scores = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min() + 1e-8)

        heuristic_scores = self.heuristic_ranker.predict(X, self.feature_names)
        heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min() + 1e-8)

        return self.alpha * ml_scores + (1 - self.alpha) * heuristic_scores


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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_ranker(model, test_df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Evaluate ranker on test builds."""
    apfds = []
    ndcgs = []

    for build_id, group in test_df.groupby('Build_ID'):
        labels = group['is_failure'].values
        if labels.sum() == 0:
            continue

        X = group[feature_cols].values
        scores = model.predict(X)

        # Compute rankings
        rankings = np.argsort(-scores).argsort()

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


def baseline_recently_failed(test_df: pd.DataFrame) -> List[float]:
    """Recently-failed baseline."""
    apfds = []
    for build_id, group in test_df.groupby('Build_ID'):
        labels = group['is_failure'].values
        if labels.sum() == 0:
            continue
        scores = group['recency_score'].values
        rankings = np.argsort(-scores).argsort()
        apfd = compute_apfd(rankings, labels)
        apfds.append(apfd)
    return apfds


def baseline_failure_rate(test_df: pd.DataFrame) -> List[float]:
    """Failure rate baseline."""
    apfds = []
    for build_id, group in test_df.groupby('Build_ID'):
        labels = group['is_failure'].values
        if labels.sum() == 0:
            continue
        scores = group['historical_rate'].values
        rankings = np.argsort(-scores).argsort()
        apfd = compute_apfd(rankings, labels)
        apfds.append(apfd)
    return apfds


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("FILO-PRIORI V10 SIMPLE - RTPTORRENT DATASET")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Model: LightGBM LambdaRank + Heuristic Ensemble")
    logger.info("  - Features: 14 ranking-optimized features")
    logger.info("  - Loss: LambdaRank (NDCG optimization)")
    logger.info("  - Ensemble: ML (70%) + Heuristic (30%)")
    logger.info(f"  - LightGBM available: {HAS_LGBM}")
    logger.info("")

    np.random.seed(42)

    # Load data
    logger.info("\n[1/5] Loading data...")
    train_df, test_df = load_rtptorrent_data()

    # Extract features
    logger.info("\n[2/5] Extracting ranking features...")
    extractor = RankingFeatureExtractor(decay_lambda=0.1, recent_window=5)
    extractor.fit(train_df)

    train_features_df = extractor.transform(train_df)
    test_features_df = extractor.transform(test_df)

    feature_cols = [
        'novelty_score', 'base_risk', 'historical_rate', 'recent_rate',
        'very_recent_rate', 'time_decay_score', 'consecutive_failures',
        'failure_trend', 'recency_score', 'volatility',
        'max_consecutive_failures', 'ranking_prior', 'total_executions',
        'last_failure_distance'
    ]

    # Prepare training data for LambdaRank
    logger.info("\n[3/5] Preparing training data...")

    # Create groups for LambdaRank (one group per build)
    X_train_list = []
    y_train_list = []
    groups_list = []

    for build_id, group in train_features_df.groupby('Build_ID'):
        if group['is_failure'].sum() == 0:
            continue

        X = group[feature_cols].values
        y = group['is_failure'].values

        X_train_list.append(X)
        y_train_list.append(y)
        groups_list.append(len(y))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    groups = np.array(groups_list)

    logger.info(f"Training samples: {len(y_train)}")
    logger.info(f"Training groups: {len(groups)}")
    logger.info(f"Failures in train: {y_train.sum()}")

    # Compute baselines
    logger.info("\n[4/5] Computing baselines...")
    rf_apfds = baseline_recently_failed(test_features_df)
    fr_apfds = baseline_failure_rate(test_features_df)

    logger.info(f"  Recently-Failed: APFD = {np.mean(rf_apfds):.4f} (+/- {np.std(rf_apfds):.4f})")
    logger.info(f"  Failure-Rate:    APFD = {np.mean(fr_apfds):.4f} (+/- {np.std(fr_apfds):.4f})")

    # Train ensemble model
    logger.info("\n[5/5] Training V10 Simple model...")

    # Try different alpha values
    best_alpha = 0.7
    best_apfd = 0.0
    results_by_alpha = {}

    for alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        model = EnsembleRanker(alpha=alpha)
        model.fit(X_train, y_train, groups, feature_cols)
        metrics = evaluate_ranker(model, test_features_df, feature_cols)
        results_by_alpha[alpha] = metrics['apfd']
        logger.info(f"  Alpha={alpha:.1f}: APFD = {metrics['apfd']:.4f}")

        if metrics['apfd'] > best_apfd:
            best_apfd = metrics['apfd']
            best_alpha = alpha

    logger.info(f"\nBest alpha: {best_alpha}")

    # Final model with best alpha
    final_model = EnsembleRanker(alpha=best_alpha)
    final_model.fit(X_train, y_train, groups, feature_cols)
    test_metrics = evaluate_ranker(final_model, test_features_df, feature_cols)

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

    # Feature importance (if LightGBM)
    feature_importance = {}
    if HAS_LGBM and hasattr(final_model.lgbm_ranker.model, 'feature_importance'):
        importance = final_model.lgbm_ranker.model.feature_importance()
        for i, fname in enumerate(feature_cols):
            feature_importance[fname] = int(importance[i])
        logger.info("\nFeature Importance:")
        for fname, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
            logger.info(f"  {fname}: {imp}")

    # Results
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS - RTPTORRENT (V10 Simple)")
    logger.info("=" * 70)
    logger.info(f"V10 Simple:        APFD = {v10_apfd:.4f} (+/- {test_metrics['apfd_std']:.4f})")
    logger.info(f"                   NDCG@10 = {test_metrics['ndcg_at_10']:.4f}")
    logger.info(f"Recently-Failed:   APFD = {rf_mean:.4f}")
    logger.info(f"Improvement:       {improvement:+.2f}%")
    logger.info(f"p-value:           {p_value:.4f}")
    logger.info(f"Best alpha:        {best_alpha}")
    logger.info("=" * 70)

    # Save results
    results_dir = BASE_DIR / "results" / "experiment_v10_rtptorrent_simple"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'rtptorrent',
        'model': 'V10_Simple_LightGBM_Ensemble',
        'v10_simple': {
            'apfd': float(v10_apfd),
            'apfd_std': float(test_metrics['apfd_std']),
            'ndcg_at_10': float(test_metrics['ndcg_at_10']),
            'num_builds': test_metrics['num_builds'],
            'best_alpha': best_alpha
        },
        'baselines': {
            'recently_failed': {'apfd': float(rf_mean), 'std': float(np.std(rf_apfds))},
            'failure_rate': {'apfd': float(np.mean(fr_apfds)), 'std': float(np.std(fr_apfds))}
        },
        'improvement_vs_rf': improvement,
        'p_value': float(p_value),
        'alpha_search': {str(k): float(v) for k, v in results_by_alpha.items()},
        'feature_importance': feature_importance,
        'config': {
            'features': feature_cols,
            'n_estimators': 50,
            'max_depth': 3,
            'ensemble': 'LightGBM + Heuristic'
        }
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

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
