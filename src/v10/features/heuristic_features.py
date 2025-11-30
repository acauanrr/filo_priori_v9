"""
Heuristic Feature Extraction for Filo-Priori V10.

This module extracts explicit heuristic features that capture
the signal from the "Recently-Failed" baseline.

Key Insight:
    The Recently-Failed heuristic is so strong because failures
    are temporally clustered (bursty). By making these features
    explicit, we allow the neural network to use them as a
    strong prior while learning to improve on edge cases.

Features:
    1. recency: Transformed time since last failure
    2. fail_rate: Historical failure rate
    3. duration: Average execution time
    4. streak: Consecutive failure count
    5. volatility: Rate of state changes (flakiness)
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch


@dataclass
class TestHistory:
    """Historical execution data for a single test."""
    test_id: str
    total_executions: int = 0
    total_failures: int = 0
    last_failure_build: Optional[int] = None
    last_execution_build: Optional[int] = None
    consecutive_failures: int = 0
    consecutive_passes: int = 0
    state_changes: int = 0  # Pass↔Fail transitions
    execution_times: List[float] = None
    last_result: Optional[str] = None

    def __post_init__(self):
        if self.execution_times is None:
            self.execution_times = []


class RecencyTransform:
    """
    Transforms raw recency (builds since failure) into a
    feature optimized for neural network learning.

    The key insight is that the relationship between recency
    and failure probability is non-linear:
    - Very recent failures (1-2 builds) are very likely to recur
    - Medium recency (3-10 builds) has moderate probability
    - Old failures (>10 builds) have low base rate

    Transform options:
    1. inverse_log: 1 / log(1 + builds)
    2. exponential: exp(-λ × builds)
    3. sigmoid: 1 / (1 + exp(builds - center))
    """

    def __init__(
        self,
        method: str = 'inverse_log',
        decay_rate: float = 0.1,
        center: float = 5.0
    ):
        self.method = method
        self.decay_rate = decay_rate
        self.center = center

    def transform(self, builds_since_failure: int) -> float:
        """
        Transform raw recency to feature value.

        Args:
            builds_since_failure: Number of builds since last failure.
                                  0 = failed in current build.
                                  None/inf = never failed.

        Returns:
            Transformed feature value in (0, 1].
        """
        if builds_since_failure is None or builds_since_failure == float('inf'):
            return 0.0

        if self.method == 'inverse_log':
            # 1 / log(1 + builds + 1)
            # Adds 1 to avoid log(1) = 0
            return 1.0 / math.log(builds_since_failure + 2)

        elif self.method == 'exponential':
            # exp(-λ × builds)
            return math.exp(-self.decay_rate * builds_since_failure)

        elif self.method == 'sigmoid':
            # 1 / (1 + exp(builds - center))
            return 1.0 / (1 + math.exp(builds_since_failure - self.center))

        else:
            return 1.0 / (builds_since_failure + 1)

    def transform_batch(self, values: List[int]) -> torch.Tensor:
        """Transform a batch of recency values."""
        return torch.tensor([self.transform(v) for v in values])


class HeuristicFeatureExtractor:
    """
    Extracts strong heuristic features for residual learning.

    The neural network will learn to improve upon these heuristics,
    not replace them. This is the key to Residual Learning.

    Features extracted:
    1. recency: Transformed time since last failure
    2. fail_rate: Historical P(fail)
    3. recent_fail_rate: P(fail) in last N builds
    4. duration: Normalized execution time
    5. streak: Current consecutive state count
    6. volatility: State change frequency (flakiness)

    These 6 features capture the essence of the Recently-Failed
    baseline and related heuristics.
    """

    def __init__(
        self,
        recency_transform: str = 'inverse_log',
        recent_window: int = 5,
        duration_normalize: str = 'log'
    ):
        self.recency_transform = RecencyTransform(method=recency_transform)
        self.recent_window = recent_window
        self.duration_normalize = duration_normalize

        # Internal state: test_id -> TestHistory
        self._history: Dict[str, TestHistory] = {}
        self._current_build: int = 0
        self._recent_results: Dict[str, List[str]] = defaultdict(list)

    def update(
        self,
        test_id: str,
        result: str,
        build_id: int,
        duration: Optional[float] = None
    ):
        """
        Update history with a new test execution.

        Args:
            test_id: Unique test identifier.
            result: 'Pass' or 'Fail'.
            build_id: Build number (for ordering).
            duration: Optional execution duration in seconds.
        """
        self._current_build = max(self._current_build, build_id)

        if test_id not in self._history:
            self._history[test_id] = TestHistory(test_id=test_id)

        h = self._history[test_id]

        # Update counts
        h.total_executions += 1
        if result == 'Fail':
            h.total_failures += 1
            h.last_failure_build = build_id
            if h.last_result == 'Pass':
                h.consecutive_failures = 1
                h.consecutive_passes = 0
                h.state_changes += 1
            else:
                h.consecutive_failures += 1
        else:  # Pass
            if h.last_result == 'Fail':
                h.consecutive_passes = 1
                h.consecutive_failures = 0
                h.state_changes += 1
            else:
                h.consecutive_passes += 1

        h.last_execution_build = build_id
        h.last_result = result

        if duration is not None:
            h.execution_times.append(duration)

        # Track recent results
        self._recent_results[test_id].append(result)
        if len(self._recent_results[test_id]) > self.recent_window:
            self._recent_results[test_id].pop(0)

    def extract(
        self,
        test_id: str,
        current_build: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract heuristic features for a test.

        Args:
            test_id: Test to extract features for.
            current_build: Current build number (for recency calculation).

        Returns:
            Tensor of shape [6] with features:
            [recency, fail_rate, recent_fail_rate, duration, streak, volatility]
        """
        if current_build is None:
            current_build = self._current_build

        if test_id not in self._history:
            # New test: use conservative defaults
            return torch.tensor([
                0.0,   # recency: never failed
                0.0,   # fail_rate: no history
                0.0,   # recent_fail_rate: no history
                0.5,   # duration: median
                0.0,   # streak: no streak
                0.0    # volatility: no changes
            ])

        h = self._history[test_id]

        # 1. Recency
        if h.last_failure_build is not None:
            builds_since_fail = current_build - h.last_failure_build
        else:
            builds_since_fail = float('inf')
        recency = self.recency_transform.transform(builds_since_fail)

        # 2. Failure rate
        if h.total_executions > 0:
            fail_rate = h.total_failures / h.total_executions
        else:
            fail_rate = 0.0

        # 3. Recent failure rate
        recent = self._recent_results.get(test_id, [])
        if recent:
            recent_fail_rate = sum(1 for r in recent if r == 'Fail') / len(recent)
        else:
            recent_fail_rate = 0.0

        # 4. Duration (normalized)
        if h.execution_times:
            avg_duration = sum(h.execution_times) / len(h.execution_times)
            if self.duration_normalize == 'log':
                duration = math.log(avg_duration + 1) / 10  # Normalize to ~[0, 1]
            else:
                duration = min(avg_duration / 60, 1.0)  # Cap at 60 seconds
        else:
            duration = 0.5  # Default

        # 5. Streak (signed: positive for pass streak, negative for fail streak)
        if h.last_result == 'Fail':
            streak = -h.consecutive_failures / 10  # Normalize
        else:
            streak = h.consecutive_passes / 10

        # 6. Volatility (flakiness)
        if h.total_executions > 1:
            volatility = h.state_changes / (h.total_executions - 1)
        else:
            volatility = 0.0

        return torch.tensor([
            recency,
            fail_rate,
            recent_fail_rate,
            duration,
            streak,
            volatility
        ], dtype=torch.float32)

    def extract_batch(
        self,
        test_ids: List[str],
        current_build: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract features for multiple tests.

        Returns:
            Tensor of shape [num_tests, 6].
        """
        features = [self.extract(tid, current_build) for tid in test_ids]
        return torch.stack(features)

    def get_feature_names(self) -> List[str]:
        """Return names of the extracted features."""
        return [
            'recency',
            'fail_rate',
            'recent_fail_rate',
            'duration',
            'streak',
            'volatility'
        ]

    def reset(self):
        """Clear all history."""
        self._history.clear()
        self._recent_results.clear()
        self._current_build = 0

    def save(self, path: str):
        """Save extractor state."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'history': self._history,
                'recent_results': dict(self._recent_results),
                'current_build': self._current_build
            }, f)

    @classmethod
    def load(cls, path: str) -> 'HeuristicFeatureExtractor':
        """Load extractor state."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        extractor = cls()
        extractor._history = state['history']
        extractor._recent_results = defaultdict(list, state['recent_results'])
        extractor._current_build = state['current_build']
        return extractor


class RecentlyFailedScore:
    """
    Pure Recently-Failed heuristic scorer.

    This is the baseline we're trying to beat.
    Useful for comparison and debugging.
    """

    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        self._last_failure: Dict[str, int] = {}

    def update(self, test_id: str, result: str, build_id: int):
        """Update with a test execution."""
        if result == 'Fail':
            self._last_failure[test_id] = build_id

    def score(self, test_id: str, current_build: int) -> float:
        """
        Score a test by recency of failure.

        Higher score = more recently failed = higher priority.
        """
        if test_id not in self._last_failure:
            return 0.0

        builds_since = current_build - self._last_failure[test_id]
        return math.exp(-self.decay_rate * builds_since)

    def rank(self, test_ids: List[str], current_build: int) -> List[str]:
        """
        Rank tests by Recently-Failed heuristic.

        Returns tests sorted by priority (highest first).
        """
        scores = [(tid, self.score(tid, current_build)) for tid in test_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in scores]
