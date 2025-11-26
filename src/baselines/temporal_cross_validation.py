"""
Temporal Cross-Validation for Test Case Prioritization.

Implements time-aware cross-validation strategies:
1. K-Fold Temporal: Expanding window with k folds
2. Sliding Window: Fixed-size sliding window
3. Concept Drift Detection: Monitor model degradation over time

This is critical for demonstrating generalization and detecting concept drift.

Author: Filo-Priori Team
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Container for a cross-validation fold."""
    fold_id: int
    train_builds: List[str]
    val_builds: List[str]
    test_builds: List[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_period: str
    test_period: str


@dataclass
class CVResult:
    """Container for cross-validation results."""
    fold_id: int
    method_name: str
    mean_apfd: float
    std_apfd: float
    median_apfd: float
    n_builds: int
    train_period: str
    test_period: str


class TemporalCrossValidator:
    """
    Temporal Cross-Validation for TCP.

    Ensures no data leakage by maintaining chronological order:
    - Training data always precedes validation/test data
    - Simulates real-world deployment scenario
    """

    def __init__(
        self,
        n_folds: int = 5,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        build_col: str = 'Build_ID',
        date_col: Optional[str] = None,
        min_train_builds: int = 50
    ):
        """
        Initialize temporal cross-validator.

        Args:
            n_folds: Number of cross-validation folds
            val_ratio: Ratio of data for validation within training period
            test_ratio: Ratio of data for test (last portion of each fold)
            build_col: Column name for build ID
            date_col: Column name for date (optional, uses build order if None)
            min_train_builds: Minimum number of training builds per fold
        """
        self.n_folds = n_folds
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.build_col = build_col
        self.date_col = date_col
        self.min_train_builds = min_train_builds

    def _get_build_order(self, df: pd.DataFrame) -> List[str]:
        """Get chronologically ordered list of unique builds."""
        if self.date_col and self.date_col in df.columns:
            # Sort by date
            build_dates = df.groupby(self.build_col)[self.date_col].min()
            return build_dates.sort_values().index.tolist()
        else:
            # Assume build IDs are chronologically ordered
            unique_builds = df[self.build_col].unique()
            # Try to sort numerically if possible
            try:
                return sorted(unique_builds, key=lambda x: int(str(x).split('_')[-1]))
            except (ValueError, IndexError):
                return list(unique_builds)

    def generate_folds(self, df: pd.DataFrame) -> Generator[CVFold, None, None]:
        """
        Generate temporal cross-validation folds.

        Uses expanding window approach:
        - Fold 1: Train [0, 60%], Val [60%, 70%], Test [70%, 80%]
        - Fold 2: Train [0, 65%], Val [65%, 75%], Test [75%, 85%]
        - Fold 3: Train [0, 70%], Val [70%, 80%], Test [80%, 90%]
        - ...

        Yields:
            CVFold objects containing train/val/test splits
        """
        builds = self._get_build_order(df)
        n_builds = len(builds)

        logger.info(f"Generating {self.n_folds} temporal folds from {n_builds} builds")

        # Calculate fold boundaries
        base_train_ratio = 0.6  # Start with 60% training
        increment = (1.0 - base_train_ratio - self.test_ratio) / max(self.n_folds - 1, 1)

        for fold_id in range(self.n_folds):
            # Calculate split points for this fold
            train_end_ratio = base_train_ratio + fold_id * increment
            val_end_ratio = train_end_ratio + self.val_ratio
            test_end_ratio = val_end_ratio + self.test_ratio

            # Ensure we don't exceed 100%
            test_end_ratio = min(test_end_ratio, 1.0)

            # Convert to indices
            train_end_idx = int(train_end_ratio * n_builds)
            val_end_idx = int(val_end_ratio * n_builds)
            test_end_idx = int(test_end_ratio * n_builds)

            # Ensure minimum training size
            if train_end_idx < self.min_train_builds:
                logger.warning(f"Fold {fold_id}: Insufficient training data, skipping")
                continue

            # Split builds
            train_builds = builds[:train_end_idx]
            val_builds = builds[train_end_idx:val_end_idx]
            test_builds = builds[val_end_idx:test_end_idx]

            # Skip if no test builds
            if len(test_builds) == 0:
                logger.warning(f"Fold {fold_id}: No test builds, skipping")
                continue

            # Create DataFrames
            train_df = df[df[self.build_col].isin(train_builds)].copy()
            val_df = df[df[self.build_col].isin(val_builds)].copy()
            test_df = df[df[self.build_col].isin(test_builds)].copy()

            # Period descriptions
            train_period = f"Builds 1-{train_end_idx} ({len(train_builds)} builds)"
            test_period = f"Builds {val_end_idx+1}-{test_end_idx} ({len(test_builds)} builds)"

            logger.info(f"  Fold {fold_id}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

            yield CVFold(
                fold_id=fold_id,
                train_builds=train_builds,
                val_builds=val_builds,
                test_builds=test_builds,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_period=train_period,
                test_period=test_period
            )

    def generate_sliding_window_folds(
        self,
        df: pd.DataFrame,
        window_size: int = 100,
        step_size: int = 20
    ) -> Generator[CVFold, None, None]:
        """
        Generate sliding window folds.

        Uses fixed-size training window that slides through time:
        - Window 1: Train [0, window_size], Test [window_size, window_size + step]
        - Window 2: Train [step, window_size + step], Test [window_size + step, window_size + 2*step]
        - ...

        Args:
            df: Full dataset
            window_size: Number of builds in training window
            step_size: Number of builds to slide per fold

        Yields:
            CVFold objects
        """
        builds = self._get_build_order(df)
        n_builds = len(builds)

        logger.info(f"Generating sliding window folds: window={window_size}, step={step_size}")

        fold_id = 0
        start_idx = 0

        while start_idx + window_size + step_size <= n_builds:
            train_end_idx = start_idx + window_size
            test_end_idx = train_end_idx + step_size

            # Split builds
            train_builds = builds[start_idx:train_end_idx]
            val_builds = train_builds[-int(len(train_builds) * self.val_ratio):]
            train_builds = train_builds[:-len(val_builds)]
            test_builds = builds[train_end_idx:test_end_idx]

            # Create DataFrames
            train_df = df[df[self.build_col].isin(train_builds)].copy()
            val_df = df[df[self.build_col].isin(val_builds)].copy()
            test_df = df[df[self.build_col].isin(test_builds)].copy()

            train_period = f"Builds {start_idx+1}-{train_end_idx}"
            test_period = f"Builds {train_end_idx+1}-{test_end_idx}"

            logger.info(f"  Window {fold_id}: {train_period} -> {test_period}")

            yield CVFold(
                fold_id=fold_id,
                train_builds=list(train_builds),
                val_builds=list(val_builds),
                test_builds=list(test_builds),
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_period=train_period,
                test_period=test_period
            )

            fold_id += 1
            start_idx += step_size


class ConceptDriftDetector:
    """
    Detect concept drift in TCP performance.

    Monitors APFD degradation over time to identify when model retraining is needed.
    """

    def __init__(
        self,
        window_size: int = 10,
        drift_threshold: float = 0.1,
        significance_level: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Number of recent builds to monitor
            drift_threshold: APFD drop threshold to trigger drift alert
            significance_level: P-value threshold for statistical significance
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.apfd_history: List[Tuple[str, float]] = []  # (build_id, apfd)

    def add_observation(self, build_id: str, apfd: float):
        """Add a new APFD observation."""
        self.apfd_history.append((build_id, apfd))

    def detect_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if concept drift has occurred.

        Returns:
            Tuple of (is_drifting, details_dict)
        """
        if len(self.apfd_history) < self.window_size * 2:
            return False, {'reason': 'Insufficient history'}

        # Split into early and recent windows
        early_window = [x[1] for x in self.apfd_history[-2*self.window_size:-self.window_size]]
        recent_window = [x[1] for x in self.apfd_history[-self.window_size:]]

        early_mean = np.mean(early_window)
        recent_mean = np.mean(recent_window)

        # Calculate drop
        drop = early_mean - recent_mean
        relative_drop = drop / early_mean if early_mean > 0 else 0

        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(early_window, recent_window)

        # Drift detection criteria
        is_significant = p_value < self.significance_level
        is_meaningful_drop = relative_drop > self.drift_threshold
        is_drifting = is_significant and is_meaningful_drop

        details = {
            'early_mean': early_mean,
            'recent_mean': recent_mean,
            'absolute_drop': drop,
            'relative_drop': relative_drop,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'is_meaningful_drop': is_meaningful_drop,
            'recommendation': 'Retrain model' if is_drifting else 'Model stable'
        }

        return is_drifting, details

    def get_trend(self) -> Dict[str, Any]:
        """
        Analyze APFD trend over time.

        Returns:
            Dictionary with trend analysis
        """
        if len(self.apfd_history) < 10:
            return {'trend': 'insufficient_data'}

        apfd_values = [x[1] for x in self.apfd_history]

        # Linear regression to find trend
        x = np.arange(len(apfd_values))
        slope, intercept = np.polyfit(x, apfd_values, 1)

        # Determine trend direction
        if slope > 0.001:
            trend = 'improving'
        elif slope < -0.001:
            trend = 'degrading'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'slope': slope,
            'initial_apfd': np.mean(apfd_values[:5]),
            'final_apfd': np.mean(apfd_values[-5:]),
            'overall_mean': np.mean(apfd_values),
            'overall_std': np.std(apfd_values)
        }


def run_temporal_cv(
    df: pd.DataFrame,
    model_factory,
    n_folds: int = 5,
    build_col: str = 'Build_ID',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run temporal cross-validation on a model.

    Args:
        df: Full dataset
        model_factory: Callable that returns a new model instance
        n_folds: Number of folds
        build_col: Build ID column name
        verbose: Print progress

    Returns:
        DataFrame with CV results
    """
    import sys
    sys.path.insert(0, '/home/acauan/ufam/iats/sprint_07/filo_priori_v9')
    from src.evaluation.apfd import calculate_apfd_per_build

    cv = TemporalCrossValidator(n_folds=n_folds, build_col=build_col)
    results = []

    for fold in cv.generate_folds(df):
        if verbose:
            logger.info(f"\n=== Fold {fold.fold_id} ===")
            logger.info(f"  Train: {fold.train_period}")
            logger.info(f"  Test: {fold.test_period}")

        # Create and train model
        model = model_factory()
        model.fit(fold.train_df)

        # Generate rankings
        test_ranked = model.rank_per_build(fold.test_df.copy(), build_col=build_col)

        # Calculate APFD
        test_ranked['label_binary'] = (test_ranked['verdict'].astype(str).str.strip() == 'Fail').astype(int)

        apfd_df = calculate_apfd_per_build(
            test_ranked,
            method_name=model.name,
            build_col=build_col,
            label_col='label_binary',
            rank_col='rank',
            result_col='verdict'
        )

        # Store results
        result = CVResult(
            fold_id=fold.fold_id,
            method_name=model.name,
            mean_apfd=apfd_df['apfd'].mean(),
            std_apfd=apfd_df['apfd'].std(),
            median_apfd=apfd_df['apfd'].median(),
            n_builds=len(apfd_df),
            train_period=fold.train_period,
            test_period=fold.test_period
        )
        results.append(result)

        if verbose:
            logger.info(f"  Mean APFD: {result.mean_apfd:.4f} +/- {result.std_apfd:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'Fold': r.fold_id,
            'Method': r.method_name,
            'Mean APFD': r.mean_apfd,
            'Std APFD': r.std_apfd,
            'Median APFD': r.median_apfd,
            'N Builds': r.n_builds,
            'Train Period': r.train_period,
            'Test Period': r.test_period
        }
        for r in results
    ])

    # Summary
    if verbose:
        logger.info("\n=== Cross-Validation Summary ===")
        logger.info(f"Overall Mean APFD: {results_df['Mean APFD'].mean():.4f} +/- {results_df['Mean APFD'].std():.4f}")
        logger.info(f"Best Fold: {results_df.loc[results_df['Mean APFD'].idxmax(), 'Fold']} "
                   f"(APFD={results_df['Mean APFD'].max():.4f})")
        logger.info(f"Worst Fold: {results_df.loc[results_df['Mean APFD'].idxmin(), 'Fold']} "
                   f"(APFD={results_df['Mean APFD'].min():.4f})")

    return results_df


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Create synthetic dataset with 200 builds
    n_builds = 200
    n_tests_per_build = 50

    data = []
    for b in range(n_builds):
        for t in range(n_tests_per_build):
            # Simulate concept drift: failure rate increases over time
            base_fail_rate = 0.05 + 0.001 * b  # Gradual increase
            is_fail = np.random.random() < base_fail_rate

            data.append({
                'Build_ID': f'Build_{b:04d}',
                'TC_Key': f'TC_{t:03d}',
                'verdict': 'Fail' if is_fail else 'Pass',
                'test_age': b - np.random.randint(0, min(b+1, 50)),
                'failure_rate': base_fail_rate + np.random.normal(0, 0.01)
            })

    demo_df = pd.DataFrame(data)

    print("=" * 60)
    print("TEMPORAL CROSS-VALIDATION DEMO")
    print("=" * 60)

    # Test fold generation
    cv = TemporalCrossValidator(n_folds=5)

    print("\n--- Fold Generation ---")
    for fold in cv.generate_folds(demo_df):
        print(f"Fold {fold.fold_id}: Train={len(fold.train_df)}, Val={len(fold.val_df)}, Test={len(fold.test_df)}")

    # Test concept drift detection
    print("\n--- Concept Drift Detection ---")
    detector = ConceptDriftDetector(window_size=10)

    # Simulate degrading performance
    for i in range(30):
        apfd = 0.7 - 0.01 * i + np.random.normal(0, 0.02)
        detector.add_observation(f'Build_{i}', apfd)

    is_drifting, details = detector.detect_drift()
    print(f"Drift detected: {is_drifting}")
    print(f"Details: {details}")

    trend = detector.get_trend()
    print(f"Trend: {trend}")
