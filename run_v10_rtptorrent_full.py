#!/usr/bin/env python3
"""
Filo-Priori V10 - Complete RTPTorrent Multi-Project Experiment.

This script runs a comprehensive evaluation across ALL 20 projects in the
RTPTorrent dataset, comparing against official baselines.

Pipeline:
1. Load raw data from all 20 projects
2. Extract ranking features for each project
3. Train LightGBM LambdaRank model per project
4. Evaluate against 7 official baselines
5. Compute APFD for each build
6. Generate comprehensive results report

Baselines from RTPTorrent:
- untreated: Original test order
- recently-failed: Alpha=0.8 decay prioritization
- random: Shuffled order
- optimal-failure: Oracle (failures first)
- optimal-failure-duration: Oracle (shortest failures first)
- matrix-naive: File-test failure matrix
- matrix-conditional-prob: Conditional probability matrix

Usage:
    python run_v10_rtptorrent_full.py [--projects all|project1,project2,...]

Author: Filo-Priori Team
Version: 10.2.0
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rtptorrent_experiment.log')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "raw" / "MSR2"

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available - install with: pip install lightgbm")
    from sklearn.ensemble import GradientBoostingClassifier


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProjectData:
    """Container for project data."""
    name: str
    raw_df: pd.DataFrame
    builds: List[Dict]
    train_builds: List[Dict] = field(default_factory=list)
    test_builds: List[Dict] = field(default_factory=list)
    baselines: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    project: str
    model_apfd: float
    model_apfd_std: float
    model_ndcg: float
    baseline_apfds: Dict[str, float]
    num_builds: int
    num_builds_with_failures: int
    improvements: Dict[str, float]
    p_values: Dict[str, float]
    per_build_apfds: List[float]


# =============================================================================
# RANKING FEATURE EXTRACTOR
# =============================================================================

class RankingFeatureExtractor:
    """
    Extract ranking-optimized features for test case prioritization.

    Features:
    1. novelty_score: How new is this test case
    2. base_risk: Prior probability of failure (Laplace smoothed)
    3. historical_failure_rate: Overall failure rate
    4. recent_failure_rate: Failure rate in last N executions
    5. very_recent_rate: Failure rate in last 2 executions
    6. time_decay_score: Exponentially decayed failure signal
    7. consecutive_failures: Current streak of failures
    8. failure_trend: Change in failure rate over time
    9. recency_score: Time since last failure
    10. volatility: Rate of status changes
    11. max_consecutive_failures: Maximum failure streak
    12. ranking_prior: Combined heuristic score
    13. total_executions: Number of times test was executed
    14. last_failure_distance: Builds since last failure
    15. failure_density: Failures per build in recent window
    16. execution_frequency: How often test runs per build
    """

    def __init__(self, decay_lambda: float = 0.1, recent_window: int = 10):
        self.decay_lambda = decay_lambda
        self.recent_window = recent_window
        self.tc_history: Dict[str, Dict] = {}
        self.max_build_idx = 0
        self.build_order: Dict[int, int] = {}

    def fit(self, builds: List[Dict]):
        """Build history from training builds."""
        self.tc_history = defaultdict(lambda: {
            'results': [],
            'build_indices': [],
            'durations': []
        })

        # Process builds in order
        for build_idx, build in enumerate(builds):
            job_id = build['job_id']
            self.build_order[job_id] = build_idx

            for test in build['tests']:
                tc_key = test['name']
                is_failure = 1 if (test['failures'] > 0 or test['errors'] > 0) else 0

                self.tc_history[tc_key]['results'].append(is_failure)
                self.tc_history[tc_key]['build_indices'].append(build_idx)
                self.tc_history[tc_key]['durations'].append(test['duration'])

        self.max_build_idx = len(builds) - 1

    def extract_features(self, tc_key: str, current_build_idx: int) -> np.ndarray:
        """Extract ranking features for a test case."""
        history = self.tc_history.get(tc_key)

        # Default features for new/unknown test cases
        # New tests should have LOW priority (never failed before)
        if history is None or len(history['results']) == 0:
            return np.array([
                1.0,   # novelty_score (new test)
                0.1,   # base_risk (low - unknown)
                0.0,   # historical_failure_rate
                0.0,   # recent_failure_rate
                0.0,   # very_recent_rate
                0.0,   # time_decay_score
                0.0,   # consecutive_failures
                0.0,   # failure_trend
                0.0,   # recency_score (LOW - never failed!)
                0.0,   # volatility
                0.0,   # max_consecutive_failures
                0.0,   # ranking_prior (LOW - no failure history)
                0.0,   # total_executions
                1.0,   # last_failure_distance (HIGH - never failed)
                0.0,   # failure_density
                0.0,   # execution_frequency (new test)
            ], dtype=np.float32)

        # Only use history up to current build
        valid_mask = [b < current_build_idx for b in history['build_indices']]
        if not any(valid_mask):
            # No history before this build - treat as new test
            return np.array([
                1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
            ], dtype=np.float32)

        results = [r for r, v in zip(history['results'], valid_mask) if v]
        build_indices = [b for b, v in zip(history['build_indices'], valid_mask) if v]

        total = len(results)
        failures = sum(results)

        # 1. Novelty score
        age = current_build_idx - min(build_indices) if build_indices else 0
        novelty_score = 1.0 / (1.0 + age)

        # 2. Base risk (Laplace smoothing)
        base_risk = (failures + 1) / (total + 2)

        # 3. Historical failure rate
        historical_rate = failures / total if total > 0 else 0

        # 4. Recent failure rate
        recent = results[-self.recent_window:]
        recent_rate = sum(recent) / len(recent) if recent else 0

        # 5. Very recent failure rate
        very_recent = results[-3:]
        very_recent_rate = sum(very_recent) / len(very_recent) if very_recent else 0

        # 6. Time-decay weighted score
        time_decay_score = 0.0
        for r, b_idx in zip(results, build_indices):
            if r == 1:
                delta = current_build_idx - b_idx
                weight = np.exp(-self.decay_lambda * delta)
                time_decay_score += weight
        time_decay_score = min(time_decay_score, 10.0)

        # 7. Consecutive failures (current streak)
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
        ranking_prior = 0.35 * recent_rate + 0.35 * time_decay_score / 10 + 0.30 * recency_score

        # 13. Total executions (normalized)
        total_executions = min(total / 100, 1.0)

        # 14. Last failure distance
        last_failure_dist = min(recency / 50, 1.0) if recency < 999 else 1.0

        # 15. Failure density (failures per build in recent window)
        recent_builds = set(build_indices[-self.recent_window:])
        failure_density = sum(results[-len(recent_builds):]) / max(len(recent_builds), 1)

        # 16. Execution frequency
        unique_builds = len(set(build_indices))
        execution_frequency = min(unique_builds / (current_build_idx + 1), 1.0) if current_build_idx > 0 else 0.5

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
            failure_density,
            execution_frequency,
        ], dtype=np.float32)

    def update_history(self, build: Dict, build_idx: int):
        """Update history with a new build (for online learning)."""
        for test in build['tests']:
            tc_key = test['name']
            is_failure = 1 if (test['failures'] > 0 or test['errors'] > 0) else 0

            if tc_key not in self.tc_history:
                self.tc_history[tc_key] = {
                    'results': [],
                    'build_indices': [],
                    'durations': []
                }

            self.tc_history[tc_key]['results'].append(is_failure)
            self.tc_history[tc_key]['build_indices'].append(build_idx)
            self.tc_history[tc_key]['durations'].append(test['duration'])


# =============================================================================
# METRICS
# =============================================================================

def compute_apfd(rankings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Average Percentage of Faults Detected (APFD).

    APFD = 1 - (sum of positions of failing tests) / (n * m) + 1/(2n)

    where n = total tests, m = number of failing tests
    """
    n = len(labels)
    m = labels.sum()
    if m == 0:
        return 1.0  # No failures = perfect score

    # Get positions of failing tests (1-indexed)
    fail_positions = rankings[labels == 1] + 1
    apfd = 1 - fail_positions.sum() / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


def compute_ndcg(scores: np.ndarray, relevances: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    n = len(scores)
    k = min(k, n)

    indices = np.argsort(-scores)[:k]
    top_relevances = relevances[indices]

    discounts = np.log2(np.arange(2, k + 2))
    dcg = (top_relevances / discounts).sum()

    sorted_rel = np.sort(relevances)[::-1][:k]
    idcg = (sorted_rel / discounts[:len(sorted_rel)]).sum()

    return dcg / (idcg + 1e-8) if idcg > 0 else 0.0


def compute_apfd_from_order(test_order: List[str], test_results: Dict[str, int]) -> float:
    """Compute APFD from ordered test list and results dict."""
    n = len(test_order)
    failures = [test_results.get(t, 0) for t in test_order]
    m = sum(failures)

    if m == 0:
        return 1.0

    fail_positions = [i + 1 for i, f in enumerate(failures) if f > 0]
    apfd = 1 - sum(fail_positions) / (n * m) + 1 / (2 * n)
    return min(max(apfd, 0.0), 1.0)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_available_projects() -> List[str]:
    """Get list of available projects in RTPTorrent dataset."""
    projects = []
    for item in RAW_DATA_DIR.iterdir():
        if item.is_dir() and '@' in item.name:
            # Check if main data file exists
            project_name = item.name
            data_file = item / f"{project_name}.csv"
            if data_file.exists():
                projects.append(project_name)
    return sorted(projects)


def load_project_data(project_name: str) -> Optional[ProjectData]:
    """Load data for a single project."""
    project_dir = RAW_DATA_DIR / project_name
    data_file = project_dir / f"{project_name}.csv"

    if not data_file.exists():
        logger.warning(f"Data file not found for {project_name}")
        return None

    logger.info(f"Loading {project_name}...")

    try:
        # Load main data
        df = pd.read_csv(data_file)

        # Load baselines
        baselines = {}
        baseline_dir = project_dir / "baseline"
        if baseline_dir.exists():
            short_name = project_name.split('@')[1]
            baseline_files = {
                'untreated': f"{short_name}@untreated.csv",
                'recently_failed': f"{short_name}@recently-failed.csv",
                'random': f"{short_name}@random.csv",
                'optimal_failure': f"{short_name}@optimal-failure.csv",
                'optimal_duration': f"{short_name}@optimal-failure-duration.csv",
                'matrix_naive': f"{short_name}@matrix-naive.csv",
                'matrix_conditional': f"{short_name}@matrix-conditional-prob.csv",
            }

            for baseline_name, filename in baseline_files.items():
                filepath = baseline_dir / filename
                if filepath.exists():
                    baselines[baseline_name] = pd.read_csv(filepath)

        # Group by build/job
        builds = []
        for job_id, group in df.groupby('travisJobId'):
            tests = []
            for _, row in group.iterrows():
                tests.append({
                    'name': row['testName'],
                    'index': row['index'],
                    'duration': row['duration'],
                    'count': row['count'],
                    'failures': row['failures'],
                    'errors': row['errors'],
                    'skipped': row['skipped']
                })

            # Sort by original index
            tests.sort(key=lambda x: x['index'])

            num_failures = sum(1 for t in tests if t['failures'] > 0 or t['errors'] > 0)

            builds.append({
                'job_id': job_id,
                'tests': tests,
                'num_tests': len(tests),
                'num_failures': num_failures
            })

        # Sort builds by job_id (proxy for time order)
        builds.sort(key=lambda x: x['job_id'])

        logger.info(f"  Loaded {len(builds)} builds, {len(df)} test executions")

        return ProjectData(
            name=project_name,
            raw_df=df,
            builds=builds,
            baselines=baselines
        )

    except Exception as e:
        logger.error(f"Error loading {project_name}: {e}")
        return None


def split_train_test(project: ProjectData, test_ratio: float = 0.2) -> ProjectData:
    """
    Split project data into train/test sets.

    IMPORTANT: For fair comparison with RTPTorrent baselines, we only evaluate
    on builds that have baseline data. The baselines use ALL prior history,
    so we should too.

    The split ensures:
    1. Training set has enough history to learn patterns
    2. Test set only includes builds that are in the baseline files
    """
    # Get job IDs that have baseline data
    baseline_job_ids = set()
    if 'recently_failed' in project.baselines:
        baseline_job_ids = set(project.baselines['recently_failed']['travisJobId'].unique())
    elif project.baselines:
        bl_df = list(project.baselines.values())[0]
        if 'travisJobId' in bl_df.columns:
            baseline_job_ids = set(bl_df['travisJobId'].unique())

    if baseline_job_ids:
        # Find builds that have baselines
        builds_with_baseline = [b for b in project.builds if b['job_id'] in baseline_job_ids]
        builds_without_baseline = [b for b in project.builds if b['job_id'] not in baseline_job_ids]

        logger.info(f"  Builds with baselines: {len(builds_with_baseline)}")
        logger.info(f"  Builds without baselines: {len(builds_without_baseline)}")

        # Use first 80% of baseline builds for "warm-up" (history building)
        # Use last 20% for testing
        n_baseline = len(builds_with_baseline)
        n_warmup = int(n_baseline * 0.8)

        # Train = all builds without baseline + first 80% of baseline builds
        project.train_builds = builds_without_baseline + builds_with_baseline[:n_warmup]
        # Test = last 20% of baseline builds
        project.test_builds = builds_with_baseline[n_warmup:]
    else:
        # Fallback to simple split
        n_builds = len(project.builds)
        n_test = max(int(n_builds * test_ratio), 10)
        n_train = n_builds - n_test
        project.train_builds = project.builds[:n_train]
        project.test_builds = project.builds[n_train:]

    logger.info(f"  Final split: {len(project.train_builds)} train, {len(project.test_builds)} test")

    return project


# =============================================================================
# MODEL
# =============================================================================

class OnlineRankingModel:
    """
    Online ranking model that mimics the "recently-failed" baseline.

    Uses exponential moving average (EMA) for test failure scores,
    similar to the RTPTorrent recently-failed baseline (alpha=0.8).
    """

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha  # EMA decay factor (higher = more weight to history)
        self.test_scores: Dict[str, float] = defaultdict(float)
        self.feature_importance = {}

    def update(self, build: Dict):
        """Update scores after observing a build's results."""
        for test in build['tests']:
            test_name = test['name']
            is_fail = 1.0 if (test['failures'] > 0 or test['errors'] > 0) else 0.0
            # EMA update: score = alpha * prev_score + (1-alpha) * current_result
            self.test_scores[test_name] = self.alpha * self.test_scores[test_name] + (1 - self.alpha) * is_fail

    def predict(self, test_names: List[str]) -> np.ndarray:
        """Get ranking scores for tests."""
        return np.array([self.test_scores.get(name, 0.0) for name in test_names])


class RankingModel:
    """LightGBM-based ranking model with ensemble."""

    FEATURE_NAMES = [
        'novelty_score', 'base_risk', 'historical_rate', 'recent_rate',
        'very_recent_rate', 'time_decay_score', 'consecutive_failures',
        'failure_trend', 'recency_score', 'volatility',
        'max_consecutive_failures', 'ranking_prior', 'total_executions',
        'last_failure_distance', 'failure_density', 'execution_frequency'
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, alpha: float = 0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha  # Weight for ML model vs heuristic
        self.model = None
        self.feature_importance = {}
        # Online model component
        self.online_model = OnlineRankingModel(alpha=0.8)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Fit the ranking model."""
        if HAS_LGBM and len(X) > 100:  # Only use ML if enough data
            train_data = lgb.Dataset(X, label=y, group=groups)

            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [5, 10, 20],
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'max_depth': self.max_depth,
                'learning_rate': 0.05,
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

            # Feature importance
            importance = self.model.feature_importance()
            for i, fname in enumerate(self.FEATURE_NAMES):
                self.feature_importance[fname] = int(importance[i])

    def update_online(self, build: Dict):
        """Update online model with build results."""
        self.online_model.update(build)

    def predict(self, X: np.ndarray, test_names: List[str] = None) -> np.ndarray:
        """Predict ranking scores."""
        # Get online scores if test names provided
        if test_names is not None:
            online_scores = self.online_model.predict(test_names)
            # Normalize
            if online_scores.max() > online_scores.min():
                online_scores = (online_scores - online_scores.min()) / (online_scores.max() - online_scores.min() + 1e-8)
        else:
            online_scores = self._heuristic_score(X)

        # If no ML model, return online scores only
        if self.model is None:
            return online_scores

        # Get ML scores
        if HAS_LGBM:
            ml_scores = self.model.predict(X)
        else:
            ml_scores = self._heuristic_score(X)

        # Normalize ML scores
        if ml_scores.max() > ml_scores.min():
            ml_scores = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min() + 1e-8)

        # Ensemble: combine ML with online scores
        return self.alpha * ml_scores + (1 - self.alpha) * online_scores

    def _heuristic_score(self, X: np.ndarray) -> np.ndarray:
        """Compute heuristic ranking score."""
        # Feature indices
        idx_recency = 8   # recency_score
        idx_recent = 3    # recent_rate
        idx_decay = 5     # time_decay_score
        idx_prior = 11    # ranking_prior

        scores = (
            0.35 * X[:, idx_recency] +
            0.30 * X[:, idx_recent] +
            0.20 * X[:, idx_decay] +
            0.15 * X[:, idx_prior]
        )

        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_build(
    model: RankingModel,
    extractor: RankingFeatureExtractor,
    build: Dict,
    build_idx: int
) -> Tuple[float, float, List[float]]:
    """Evaluate model on a single build."""
    tests = build['tests']
    n_tests = len(tests)

    if n_tests == 0:
        return 1.0, 1.0, []

    # Extract features and test names for each test
    features = []
    labels = []
    test_names = []
    for test in tests:
        feat = extractor.extract_features(test['name'], build_idx)
        features.append(feat)
        labels.append(1 if (test['failures'] > 0 or test['errors'] > 0) else 0)
        test_names.append(test['name'])

    X = np.array(features)
    y = np.array(labels)

    if y.sum() == 0:
        return 1.0, 1.0, []  # No failures

    # Predict scores (using both features and online model)
    scores = model.predict(X, test_names)

    # Compute rankings (higher score = higher priority = lower rank number)
    rankings = np.argsort(-scores).argsort()

    # Compute metrics
    apfd = compute_apfd(rankings, y)
    ndcg = compute_ndcg(scores, y, k=10)

    return apfd, ndcg, scores.tolist()


def evaluate_baseline(
    baseline_df: pd.DataFrame,
    project_builds: List[Dict],
    baseline_name: str
) -> Tuple[float, float, List[float], set]:
    """
    Evaluate a baseline on project builds.

    Returns:
        mean_apfd, std_apfd, apfd_list, job_ids_evaluated
    """
    apfds = []
    job_ids_evaluated = set()

    # Group baseline by job
    if 'travisJobId' not in baseline_df.columns:
        return 0.0, 0.0, [], set()

    baseline_by_job = baseline_df.groupby('travisJobId')

    for build in project_builds:
        job_id = build['job_id']

        if job_id not in baseline_by_job.groups:
            continue

        baseline_build = baseline_by_job.get_group(job_id).sort_values('index')

        # Use failures from baseline file (more accurate)
        failures_mask = (baseline_build['failures'] > 0) | (baseline_build['errors'] > 0)
        num_failures = failures_mask.sum()

        # Skip if no failures
        if num_failures == 0:
            continue

        # Compute APFD from baseline ordering
        n = len(baseline_build)
        fail_positions = np.where(failures_mask.values)[0] + 1  # 1-indexed positions
        apfd = 1 - fail_positions.sum() / (n * num_failures) + 1 / (2 * n)
        apfd = min(max(apfd, 0.0), 1.0)

        apfds.append(apfd)
        job_ids_evaluated.add(job_id)

    if not apfds:
        return 0.0, 0.0, [], set()

    return np.mean(apfds), np.std(apfds), apfds, job_ids_evaluated


def evaluate_project(
    project: ProjectData,
    model: RankingModel,
    extractor: RankingFeatureExtractor
) -> EvaluationResult:
    """Evaluate model and baselines on a project."""
    logger.info(f"Evaluating {project.name}...")

    # First, get the set of job IDs that have baselines
    # We'll use the recently_failed baseline as reference
    baseline_job_ids = set()
    if 'recently_failed' in project.baselines:
        rf_df = project.baselines['recently_failed']
        baseline_job_ids = set(rf_df['travisJobId'].unique())
    elif project.baselines:
        # Use any available baseline
        bl_df = list(project.baselines.values())[0]
        if 'travisJobId' in bl_df.columns:
            baseline_job_ids = set(bl_df['travisJobId'].unique())

    logger.info(f"  Baseline job IDs available: {len(baseline_job_ids)}")

    # Filter test builds to only those in baselines
    test_builds_filtered = []
    for build in project.test_builds:
        if build['job_id'] in baseline_job_ids:
            test_builds_filtered.append(build)

    logger.info(f"  Test builds with baselines: {len(test_builds_filtered)}")

    if not test_builds_filtered:
        # Fall back to using all test builds
        test_builds_filtered = project.test_builds
        logger.info(f"  Using all test builds: {len(test_builds_filtered)}")

    # Evaluate baselines first to get the exact jobs to compare
    baseline_apfds = {}
    baseline_apfd_lists = {}
    baseline_job_ids_with_failures = set()

    for baseline_name, baseline_df in project.baselines.items():
        bl_mean, bl_std, bl_apfds, bl_jobs = evaluate_baseline(
            baseline_df, test_builds_filtered, baseline_name
        )

        if bl_mean > 0:
            baseline_apfds[baseline_name] = bl_mean
            baseline_apfd_lists[baseline_name] = bl_apfds
            baseline_job_ids_with_failures.update(bl_jobs)

    logger.info(f"  Jobs with failures in baselines: {len(baseline_job_ids_with_failures)}")

    # Now evaluate model ONLY on the jobs that have failures in baselines
    model_apfds = []
    model_apfd_by_job = {}
    model_ndcgs = []

    # Get the starting index for test builds
    train_size = len(project.train_builds)

    for i, build in enumerate(tqdm(test_builds_filtered, desc=f"  Testing {project.name}")):
        job_id = build['job_id']

        # Only evaluate on jobs that have failures in baseline
        if job_id not in baseline_job_ids_with_failures:
            # Still update history and online model
            build_idx = train_size + i
            extractor.update_history(build, build_idx)
            model.update_online(build)
            continue

        build_idx = train_size + i
        apfd, ndcg, _ = evaluate_build(model, extractor, build, build_idx)
        model_apfds.append(apfd)
        model_apfd_by_job[job_id] = apfd
        model_ndcgs.append(ndcg)

        # Update for next builds (online learning)
        extractor.update_history(build, build_idx)
        model.update_online(build)

    if not model_apfds:
        logger.warning(f"  No builds with failures in test set for {project.name}")
        return EvaluationResult(
            project=project.name,
            model_apfd=0.0,
            model_apfd_std=0.0,
            model_ndcg=0.0,
            baseline_apfds={},
            num_builds=len(test_builds_filtered),
            num_builds_with_failures=0,
            improvements={},
            p_values={},
            per_build_apfds=[]
        )

    model_apfd = np.mean(model_apfds)
    model_apfd_std = np.std(model_apfds)
    model_ndcg = np.mean(model_ndcgs)

    logger.info(f"  Model APFD: {model_apfd:.4f} (+/- {model_apfd_std:.4f})")

    # Compute improvements and p-values
    improvements = {}
    p_values = {}

    for baseline_name, bl_apfds in baseline_apfd_lists.items():
        bl_mean = baseline_apfds[baseline_name]

        # Compute improvement
        improvement = (model_apfd - bl_mean) / bl_mean * 100 if bl_mean > 0 else 0
        improvements[baseline_name] = improvement

        # Statistical test (Wilcoxon signed-rank)
        # Use paired comparison on same builds
        min_len = min(len(model_apfds), len(bl_apfds))
        if min_len >= 10:
            try:
                _, p_val = stats.wilcoxon(
                    model_apfds[:min_len],
                    bl_apfds[:min_len],
                    alternative='greater'
                )
                p_values[baseline_name] = p_val
            except:
                p_values[baseline_name] = 1.0
        else:
            p_values[baseline_name] = 1.0

        logger.info(f"  vs {baseline_name}: {bl_mean:.4f} (improvement: {improvement:+.2f}%)")

    return EvaluationResult(
        project=project.name,
        model_apfd=model_apfd,
        model_apfd_std=model_apfd_std,
        model_ndcg=model_ndcg,
        baseline_apfds=baseline_apfds,
        num_builds=len(test_builds_filtered),
        num_builds_with_failures=len(model_apfds),
        improvements=improvements,
        p_values=p_values,
        per_build_apfds=model_apfds
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_project_model(project: ProjectData) -> Tuple[RankingModel, RankingFeatureExtractor]:
    """Train model for a single project."""
    logger.info(f"Training model for {project.name}...")

    # Initialize feature extractor
    extractor = RankingFeatureExtractor(decay_lambda=0.1, recent_window=10)
    extractor.fit(project.train_builds)

    # Initialize model
    model = RankingModel(n_estimators=100, max_depth=4, alpha=0.5)

    # Update online model with ALL training builds (in order)
    for build in project.train_builds:
        model.update_online(build)

    # Prepare training data for ML model
    X_list = []
    y_list = []
    groups = []

    builds_with_failures = [b for b in project.train_builds if b['num_failures'] > 0]

    for build_idx, build in enumerate(builds_with_failures):
        tests = build['tests']
        build_features = []
        build_labels = []

        for test in tests:
            feat = extractor.extract_features(test['name'], build_idx)
            label = 1 if (test['failures'] > 0 or test['errors'] > 0) else 0
            build_features.append(feat)
            build_labels.append(label)

        if build_features:
            X_list.append(np.array(build_features))
            y_list.append(np.array(build_labels))
            groups.append(len(build_features))

    if not X_list:
        logger.warning(f"  No training data with failures for {project.name}")
        return model, extractor

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.array(groups)

    logger.info(f"  Training samples: {len(y)}, groups: {len(groups)}, failures: {int(y.sum())}")

    # Train ML component
    model.fit(X, y, groups)

    return model, extractor


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_full_experiment(project_names: List[str] = None) -> Dict:
    """Run the complete RTPTorrent experiment."""
    logger.info("=" * 80)
    logger.info("FILO-PRIORI V10 - COMPLETE RTPTORRENT EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"LightGBM available: {HAS_LGBM}")
    logger.info("")

    # Get available projects
    available = get_available_projects()
    logger.info(f"Available projects: {len(available)}")

    if project_names is None or 'all' in project_names:
        projects_to_process = available
    else:
        projects_to_process = [p for p in project_names if p in available]

    logger.info(f"Projects to process: {len(projects_to_process)}")
    logger.info("")

    # Process each project
    all_results: List[EvaluationResult] = []
    project_details = {}

    for project_name in projects_to_process:
        logger.info("-" * 80)
        logger.info(f"PROCESSING: {project_name}")
        logger.info("-" * 80)

        # Load data
        project = load_project_data(project_name)
        if project is None:
            continue

        if len(project.builds) < 50:
            logger.warning(f"  Skipping {project_name}: too few builds ({len(project.builds)})")
            continue

        # Split train/test
        project = split_train_test(project, test_ratio=0.2)

        # Train model
        model, extractor = train_project_model(project)

        # Evaluate
        result = evaluate_project(project, model, extractor)
        all_results.append(result)

        # Store details
        project_details[project_name] = {
            'num_builds': len(project.builds),
            'num_train': len(project.train_builds),
            'num_test': len(project.test_builds),
            'num_test_with_failures': result.num_builds_with_failures,
            'model_apfd': result.model_apfd,
            'model_apfd_std': result.model_apfd_std,
            'baseline_apfds': result.baseline_apfds,
            'improvements': result.improvements,
            'p_values': result.p_values,
            'feature_importance': model.feature_importance
        }

        logger.info("")

    # Aggregate results
    logger.info("=" * 80)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 80)

    if not all_results:
        logger.error("No results to aggregate!")
        return {}

    # Compute weighted averages (by number of builds)
    total_builds = sum(r.num_builds_with_failures for r in all_results)

    weighted_model_apfd = sum(
        r.model_apfd * r.num_builds_with_failures for r in all_results
    ) / total_builds if total_builds > 0 else 0

    # Aggregate baseline comparisons
    baseline_names = ['untreated', 'recently_failed', 'random', 'optimal_failure',
                      'optimal_duration', 'matrix_naive', 'matrix_conditional']

    aggregate_baselines = {}
    aggregate_improvements = {}

    for bl_name in baseline_names:
        bl_values = []
        weights = []
        for r in all_results:
            if bl_name in r.baseline_apfds:
                bl_values.append(r.baseline_apfds[bl_name])
                weights.append(r.num_builds_with_failures)

        if bl_values:
            weighted_avg = sum(v * w for v, w in zip(bl_values, weights)) / sum(weights)
            aggregate_baselines[bl_name] = weighted_avg
            aggregate_improvements[bl_name] = (weighted_model_apfd - weighted_avg) / weighted_avg * 100

    # Print summary
    logger.info(f"\nProcessed {len(all_results)} projects, {total_builds} test builds with failures")
    logger.info(f"\nV10 Model (Weighted Average): APFD = {weighted_model_apfd:.4f}")
    logger.info("\nBaseline Comparisons:")

    for bl_name in baseline_names:
        if bl_name in aggregate_baselines:
            bl_apfd = aggregate_baselines[bl_name]
            improvement = aggregate_improvements[bl_name]
            sig = "*" if improvement > 0 else ""
            logger.info(f"  vs {bl_name:20s}: {bl_apfd:.4f} (improvement: {improvement:+.2f}%{sig})")

    # Per-project summary
    logger.info("\n" + "=" * 80)
    logger.info("PER-PROJECT RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Project':<35} {'APFD':>8} {'vs RF':>10} {'Builds':>8}")
    logger.info("-" * 80)

    for r in sorted(all_results, key=lambda x: -x.model_apfd):
        rf_imp = r.improvements.get('recently_failed', 0)
        logger.info(f"{r.project:<35} {r.model_apfd:>8.4f} {rf_imp:>+10.2f}% {r.num_builds_with_failures:>8}")

    # Save results
    results_dir = BASE_DIR / "results" / "experiment_v10_rtptorrent_full"
    results_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'experiment': 'V10_RTPTorrent_Full',
        'timestamp': datetime.now().isoformat(),
        'num_projects': len(all_results),
        'total_test_builds': total_builds,
        'aggregate': {
            'model_apfd': weighted_model_apfd,
            'baseline_apfds': aggregate_baselines,
            'improvements': aggregate_improvements
        },
        'per_project': project_details
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save CSV summary
    summary_rows = []
    for r in all_results:
        row = {
            'project': r.project,
            'model_apfd': r.model_apfd,
            'model_apfd_std': r.model_apfd_std,
            'model_ndcg': r.model_ndcg,
            'num_builds': r.num_builds_with_failures
        }
        for bl_name in baseline_names:
            row[f'bl_{bl_name}'] = r.baseline_apfds.get(bl_name, np.nan)
            row[f'imp_{bl_name}'] = r.improvements.get(bl_name, np.nan)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / 'summary.csv', index=False)

    # Save all per-build APFDs
    all_apfds = []
    for r in all_results:
        for i, apfd in enumerate(r.per_build_apfds):
            all_apfds.append({
                'project': r.project,
                'build_idx': i,
                'apfd': apfd
            })

    apfd_df = pd.DataFrame(all_apfds)
    apfd_df.to_csv(results_dir / 'all_apfds.csv', index=False)

    logger.info(f"\nResults saved to {results_dir}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return final_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Filo-Priori V10 - Complete RTPTorrent Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_v10_rtptorrent_full.py                    # Run on all projects
    python run_v10_rtptorrent_full.py --projects all    # Run on all projects
    python run_v10_rtptorrent_full.py --projects facebook@buck,square@okhttp
    python run_v10_rtptorrent_full.py --list            # List available projects
        """
    )

    parser.add_argument(
        '--projects', '-p',
        type=str,
        default='all',
        help='Comma-separated list of projects to process, or "all"'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available projects and exit'
    )

    args = parser.parse_args()

    if args.list:
        projects = get_available_projects()
        print(f"\nAvailable projects ({len(projects)}):")
        for p in projects:
            print(f"  - {p}")
        return

    if args.projects == 'all':
        project_list = None
    else:
        project_list = [p.strip() for p in args.projects.split(',')]

    run_full_experiment(project_list)


if __name__ == '__main__':
    main()
