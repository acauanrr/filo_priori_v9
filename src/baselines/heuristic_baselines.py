"""
Heuristic Baselines for Test Case Prioritization.

Implements simple, interpretable baselines that don't require machine learning:
1. Random: Random ordering (expected APFD ≈ 0.5)
2. Recency: Prioritize tests that failed recently
3. Failure-Rate: Prioritize tests with higher historical failure rate
4. Greedy Historical: Combine multiple heuristics

These baselines are critical for scientific comparison as they represent
common industry practices and provide lower bounds for ML approaches.

Author: Filo-Priori Team
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseBaseline(ABC):
    """Abstract base class for all baselines."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseBaseline':
        """Fit the baseline on training data."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability for ranking.

        Returns:
            Array of shape [N,] with failure probabilities (higher = more likely to fail)
        """
        pass

    def rank(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate priority ranks (1 = highest priority).

        Returns:
            Array of shape [N,] with ranks (1-indexed)
        """
        probs = self.predict_proba(df)
        # Higher probability = lower rank number (higher priority)
        ranks = (-probs).argsort().argsort() + 1
        return ranks

    def rank_per_build(self, df: pd.DataFrame, build_col: str = 'Build_ID') -> pd.DataFrame:
        """
        Generate priority ranks per build.

        Returns:
            DataFrame with added 'probability' and 'rank' columns
        """
        df = df.copy()
        df['probability'] = self.predict_proba(df)

        # Rank within each build
        df['rank'] = df.groupby(build_col)['probability'] \
                       .rank(method='first', ascending=False) \
                       .astype(int)

        return df


class RandomBaseline(BaseBaseline):
    """
    Random Ordering Baseline.

    Expected APFD ≈ 0.5 (by definition of random ordering).
    This is the lower bound for any prioritization method.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="Random")
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def fit(self, df: pd.DataFrame) -> 'RandomBaseline':
        """No fitting required for random baseline."""
        self.is_fitted = True
        logger.info(f"RandomBaseline: No fitting required (seed={self.seed})")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return random probabilities."""
        return self.rng.random(len(df))


class RecencyBaseline(BaseBaseline):
    """
    Recency-Based Baseline.

    Prioritizes tests based on how recently they failed.
    Tests that failed more recently get higher priority.

    Score = 1 / (builds_since_last_failure + 1)

    Intuition: Recent failures indicate active bugs that may still exist.
    """

    def __init__(self, decay_factor: float = 0.9):
        super().__init__(name="Recency")
        self.decay_factor = decay_factor
        self.tc_last_failure: Dict[str, int] = {}
        self.current_build_idx: int = 0

    def fit(self, df: pd.DataFrame) -> 'RecencyBaseline':
        """
        Fit by tracking last failure build for each test case.

        Args:
            df: Training DataFrame with columns ['TC_Key', 'Build_ID', 'verdict' or 'TE_Test_Result']
        """
        # Sort by build chronologically
        df_sorted = df.sort_values('Build_ID')

        # Map builds to indices
        unique_builds = df_sorted['Build_ID'].unique()
        build_to_idx = {b: i for i, b in enumerate(unique_builds)}

        # Determine verdict column
        verdict_col = 'verdict' if 'verdict' in df_sorted.columns else 'TE_Test_Result'

        # Track last failure for each TC
        self.tc_last_failure = {}

        for _, row in df_sorted.iterrows():
            tc_key = row.get('TC_Key', row.get('tc_id', str(row.name)))
            build_idx = build_to_idx[row['Build_ID']]
            verdict = str(row.get(verdict_col, '')).strip()

            if verdict == 'Fail':
                self.tc_last_failure[tc_key] = build_idx

        self.current_build_idx = len(unique_builds)
        self.is_fitted = True

        logger.info(f"RecencyBaseline fitted: {len(self.tc_last_failure)} TCs with failure history")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate recency-based failure probability.

        Score = decay_factor ^ (builds_since_last_failure)
        """
        probas = []

        for _, row in df.iterrows():
            tc_key = row.get('TC_Key', row.get('tc_id', str(row.name)))

            if tc_key in self.tc_last_failure:
                builds_since_failure = self.current_build_idx - self.tc_last_failure[tc_key]
                prob = self.decay_factor ** builds_since_failure
            else:
                # No failure history: use baseline probability
                prob = 0.1  # Low but non-zero

            probas.append(prob)

        return np.array(probas)


class FailureRateBaseline(BaseBaseline):
    """
    Failure Rate Baseline.

    Prioritizes tests based on historical failure rate.
    Tests with higher failure rates get higher priority.

    Score = (num_failures + smoothing) / (num_executions + 2*smoothing)

    Intuition: Tests that fail often are likely to fail again.
    """

    def __init__(self, smoothing: float = 1.0, use_recent: bool = False, recent_window: int = 10):
        super().__init__(name="FailureRate" if not use_recent else "RecentFailureRate")
        self.smoothing = smoothing
        self.use_recent = use_recent
        self.recent_window = recent_window
        self.tc_failure_rate: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> 'FailureRateBaseline':
        """
        Fit by calculating failure rate for each test case.

        Args:
            df: Training DataFrame with columns ['TC_Key', 'Build_ID', 'verdict' or 'TE_Test_Result']
        """
        # Sort by build chronologically
        df_sorted = df.sort_values('Build_ID')

        if self.use_recent:
            # Use only recent builds
            unique_builds = df_sorted['Build_ID'].unique()
            recent_builds = set(unique_builds[-self.recent_window:])
            df_sorted = df_sorted[df_sorted['Build_ID'].isin(recent_builds)]

        # Calculate failure rate per TC
        tc_key_col = 'TC_Key' if 'TC_Key' in df_sorted.columns else 'tc_id'
        verdict_col = 'verdict' if 'verdict' in df_sorted.columns else 'TE_Test_Result'

        for tc_key, tc_df in df_sorted.groupby(tc_key_col):
            verdicts = tc_df[verdict_col].astype(str)
            num_failures = (verdicts.str.strip() == 'Fail').sum()
            num_executions = len(tc_df)

            # Laplace smoothing
            rate = (num_failures + self.smoothing) / (num_executions + 2 * self.smoothing)
            self.tc_failure_rate[tc_key] = rate

        self.is_fitted = True

        # Statistics
        rates = list(self.tc_failure_rate.values())
        logger.info(f"FailureRateBaseline fitted: {len(self.tc_failure_rate)} TCs")
        logger.info(f"  Failure rate stats: min={min(rates):.4f}, max={max(rates):.4f}, mean={np.mean(rates):.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return historical failure rates."""
        probas = []
        tc_key_col = 'TC_Key' if 'TC_Key' in df.columns else 'tc_id'

        # Default rate for unknown TCs (slightly below average)
        default_rate = np.mean(list(self.tc_failure_rate.values())) * 0.5 if self.tc_failure_rate else 0.1

        for _, row in df.iterrows():
            tc_key = row.get(tc_key_col, str(row.name))
            prob = self.tc_failure_rate.get(tc_key, default_rate)
            probas.append(prob)

        return np.array(probas)


class GreedyHistoricalBaseline(BaseBaseline):
    """
    Greedy Historical Baseline.

    Combines multiple heuristics with weighted scoring:
    - Failure rate (historical)
    - Recency (recent failures)
    - Streak (consecutive failures)

    Score = w1 * failure_rate + w2 * recency_score + w3 * streak_bonus

    This represents a sophisticated non-ML baseline that combines multiple signals.
    """

    def __init__(self,
                 w_failure_rate: float = 0.5,
                 w_recency: float = 0.3,
                 w_streak: float = 0.2,
                 decay_factor: float = 0.9):
        super().__init__(name="GreedyHistorical")
        self.w_failure_rate = w_failure_rate
        self.w_recency = w_recency
        self.w_streak = w_streak
        self.decay_factor = decay_factor

        # Component baselines
        self.failure_rate_baseline = FailureRateBaseline()
        self.recency_baseline = RecencyBaseline(decay_factor=decay_factor)

        # Streak tracking
        self.tc_streak: Dict[str, int] = {}  # Current failure streak

    def fit(self, df: pd.DataFrame) -> 'GreedyHistoricalBaseline':
        """Fit all component baselines."""
        # Fit components
        self.failure_rate_baseline.fit(df)
        self.recency_baseline.fit(df)

        # Calculate streaks
        df_sorted = df.sort_values('Build_ID')
        tc_key_col = 'TC_Key' if 'TC_Key' in df_sorted.columns else 'tc_id'
        verdict_col = 'verdict' if 'verdict' in df_sorted.columns else 'TE_Test_Result'

        current_streak: Dict[str, int] = {}

        for _, row in df_sorted.iterrows():
            tc_key = row.get(tc_key_col, str(row.name))
            verdict = str(row.get(verdict_col, '')).strip()

            if verdict == 'Fail':
                current_streak[tc_key] = current_streak.get(tc_key, 0) + 1
            else:
                current_streak[tc_key] = 0

        self.tc_streak = current_streak
        self.is_fitted = True

        logger.info(f"GreedyHistoricalBaseline fitted with weights: "
                   f"failure_rate={self.w_failure_rate}, recency={self.w_recency}, streak={self.w_streak}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Combine multiple heuristics into final score."""
        # Get component scores
        failure_rate_scores = self.failure_rate_baseline.predict_proba(df)
        recency_scores = self.recency_baseline.predict_proba(df)

        # Calculate streak bonus
        tc_key_col = 'TC_Key' if 'TC_Key' in df.columns else 'tc_id'
        streak_scores = []

        for _, row in df.iterrows():
            tc_key = row.get(tc_key_col, str(row.name))
            streak = self.tc_streak.get(tc_key, 0)
            # Normalize streak to [0, 1] with saturation at 5 consecutive failures
            streak_score = min(streak / 5.0, 1.0)
            streak_scores.append(streak_score)

        streak_scores = np.array(streak_scores)

        # Weighted combination
        combined = (self.w_failure_rate * failure_rate_scores +
                   self.w_recency * recency_scores +
                   self.w_streak * streak_scores)

        return combined


def evaluate_baseline_apfd(
    baseline: BaseBaseline,
    test_df: pd.DataFrame,
    build_col: str = 'Build_ID',
    label_col: str = 'verdict',
    fail_value: str = 'Fail'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate a baseline on test data and calculate APFD per build.

    Args:
        baseline: Fitted baseline
        test_df: Test DataFrame
        build_col: Build ID column name
        label_col: Label column name
        fail_value: Value indicating failure

    Returns:
        Tuple of (apfd_per_build_df, summary_stats)
    """
    # Generate rankings
    df_ranked = baseline.rank_per_build(test_df, build_col=build_col)

    # Create binary labels
    df_ranked['label_binary'] = (df_ranked[label_col].astype(str).str.strip() == fail_value).astype(int)

    # Import APFD calculation
    import sys
    sys.path.insert(0, '/home/acauan/ufam/iats/sprint_07/filo_priori_v9')
    from src.evaluation.apfd import calculate_apfd_per_build

    # Calculate APFD per build
    apfd_df = calculate_apfd_per_build(
        df_ranked,
        method_name=baseline.name,
        test_scenario="full_test",
        build_col=build_col,
        label_col='label_binary',
        rank_col='rank',
        result_col=label_col
    )

    # Summary statistics
    if len(apfd_df) > 0:
        summary = {
            'method': baseline.name,
            'total_builds': len(apfd_df),
            'mean_apfd': float(apfd_df['apfd'].mean()),
            'median_apfd': float(apfd_df['apfd'].median()),
            'std_apfd': float(apfd_df['apfd'].std()),
            'min_apfd': float(apfd_df['apfd'].min()),
            'max_apfd': float(apfd_df['apfd'].max()),
            'pct_apfd_1_0': float((apfd_df['apfd'] == 1.0).mean() * 100),
            'pct_apfd_gte_0_7': float((apfd_df['apfd'] >= 0.7).mean() * 100),
            'pct_apfd_lt_0_5': float((apfd_df['apfd'] < 0.5).mean() * 100)
        }
    else:
        summary = {'method': baseline.name, 'error': 'No valid builds'}

    return apfd_df, summary


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Create synthetic test data
    n_samples = 1000
    n_builds = 50

    demo_df = pd.DataFrame({
        'TC_Key': [f'TC_{i % 100}' for i in range(n_samples)],
        'Build_ID': [f'Build_{i % n_builds}' for i in range(n_samples)],
        'verdict': np.random.choice(['Pass', 'Fail'], n_samples, p=[0.95, 0.05])
    })

    # Split train/test
    train_builds = [f'Build_{i}' for i in range(40)]
    test_builds = [f'Build_{i}' for i in range(40, 50)]

    train_df = demo_df[demo_df['Build_ID'].isin(train_builds)]
    test_df = demo_df[demo_df['Build_ID'].isin(test_builds)]

    print("=" * 60)
    print("HEURISTIC BASELINES DEMO")
    print("=" * 60)

    # Test each baseline
    baselines = [
        RandomBaseline(seed=42),
        RecencyBaseline(decay_factor=0.9),
        FailureRateBaseline(smoothing=1.0),
        GreedyHistoricalBaseline()
    ]

    for baseline in baselines:
        print(f"\n--- {baseline.name} ---")
        baseline.fit(train_df)
        apfd_df, summary = evaluate_baseline_apfd(baseline, test_df)
        print(f"Mean APFD: {summary.get('mean_apfd', 'N/A'):.4f}")
        print(f"Builds with APFD=1.0: {summary.get('pct_apfd_1_0', 'N/A'):.1f}%")
