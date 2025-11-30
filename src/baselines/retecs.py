"""
RETECS: Reinforcement Learning for Automatic Test Case Prioritization
and Selection in Continuous Integration

Reference:
    Spieker, H., Gotlieb, A., Marijan, D., & Mossige, M. (2017).
    Reinforcement learning for automatic test case prioritization and selection
    in continuous integration. ISSTA 2017.

Implementation based on:
    https://github.com/romolodevito/RL_for_TestPrioritization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random


class RETECSAgent:
    """
    RETECS RL Agent for Test Case Prioritization.

    Uses a simple neural network (or table-based) Q-learning approach
    to learn test case prioritization from historical execution data.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.01,
        reward_type: str = 'tcfail'  # 'tcfail', 'timerank', 'reward_plus'
    ):
        """
        Initialize RETECS agent.

        Args:
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            exploration_rate: Initial epsilon for epsilon-greedy
            exploration_decay: Rate of epsilon decay
            min_exploration: Minimum epsilon value
            reward_type: Type of reward function to use
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.reward_type = reward_type

        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Test case history
        self.test_history = {}  # test_id -> {last_exec, last_result, fail_count, exec_count}

    def _get_state(self, test_id: str, build_history: Dict) -> Tuple:
        """
        Get state representation for a test case.

        State features (discretized):
        - Last execution result (0=pass, 1=fail, 2=new)
        - Failure rate bucket (0-4)
        - Recency bucket (0-4)
        """
        if test_id not in self.test_history:
            return (2, 0, 0)  # New test

        hist = self.test_history[test_id]

        # Last result
        last_result = hist.get('last_result', 0)

        # Failure rate bucket
        fail_rate = hist.get('fail_count', 0) / max(hist.get('exec_count', 1), 1)
        fail_bucket = min(int(fail_rate * 5), 4)

        # Recency bucket (builds since last execution)
        recency = hist.get('builds_since_exec', 0)
        recency_bucket = min(recency, 4)

        return (last_result, fail_bucket, recency_bucket)

    def _compute_reward(
        self,
        ranking: List[str],
        verdicts: Dict[str, int],
        durations: Dict[str, float]
    ) -> float:
        """
        Compute reward based on the ranking and actual test results.

        Args:
            ranking: Ordered list of test IDs
            verdicts: Dict mapping test_id -> 0 (pass) or 1 (fail)
            durations: Dict mapping test_id -> execution time

        Returns:
            Reward value
        """
        if self.reward_type == 'tcfail':
            # Reward = 1 for each failure detected in top positions
            reward = 0
            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    # Higher reward for failures found earlier
                    reward += 1.0 / (i + 1)
            return reward

        elif self.reward_type == 'timerank':
            # Time-aware reward considering execution duration
            total_time = sum(durations.values())
            elapsed = 0
            faults_found = 0
            reward = 0

            for test_id in ranking:
                elapsed += durations.get(test_id, 1.0)
                if verdicts.get(test_id, 0) == 1:
                    faults_found += 1
                    # APFD-like reward
                    reward += faults_found / (elapsed / total_time + 0.001)

            return reward

        elif self.reward_type == 'reward_plus':
            # Combined reward: failures early + coverage
            reward = 0
            n_tests = len(ranking)
            n_faults = sum(verdicts.values())

            if n_faults == 0:
                return 0

            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    # Position-based reward (higher for earlier detection)
                    reward += (n_tests - i) / n_tests

            return reward / n_faults

        return 0

    def prioritize(self, test_ids: List[str], build_id: str) -> List[str]:
        """
        Prioritize test cases for a build using learned Q-values.

        Args:
            test_ids: List of test IDs to prioritize
            build_id: Current build identifier

        Returns:
            Ordered list of test IDs (highest priority first)
        """
        # Get Q-values for each test
        test_scores = []

        for test_id in test_ids:
            state = self._get_state(test_id, self.test_history)

            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                score = random.random()
            else:
                # Use Q-value as score (action = prioritize this test)
                score = self.q_table[state].get(test_id, 0.0)

            test_scores.append((test_id, score))

        # Sort by score (descending)
        test_scores.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in test_scores]

    def update(
        self,
        build_id: str,
        ranking: List[str],
        verdicts: Dict[str, int],
        durations: Optional[Dict[str, float]] = None
    ):
        """
        Update Q-values based on observed results.

        Args:
            build_id: Build identifier
            ranking: The ranking that was used
            verdicts: Actual test results (0=pass, 1=fail)
            durations: Test execution durations
        """
        if durations is None:
            durations = {t: 1.0 for t in ranking}

        # Compute reward
        reward = self._compute_reward(ranking, verdicts, durations)

        # Update Q-values for each test in the ranking
        for i, test_id in enumerate(ranking):
            state = self._get_state(test_id, self.test_history)

            # Q-learning update
            old_q = self.q_table[state].get(test_id, 0.0)

            # Simple update: reward weighted by position
            position_weight = 1.0 / (i + 1)
            new_q = old_q + self.lr * (reward * position_weight - old_q)

            self.q_table[state][test_id] = new_q

        # Update test history
        for test_id, verdict in verdicts.items():
            if test_id not in self.test_history:
                self.test_history[test_id] = {
                    'fail_count': 0,
                    'exec_count': 0,
                    'last_result': 0,
                    'builds_since_exec': 0
                }

            self.test_history[test_id]['exec_count'] += 1
            if verdict == 1:
                self.test_history[test_id]['fail_count'] += 1
            self.test_history[test_id]['last_result'] = verdict
            self.test_history[test_id]['builds_since_exec'] = 0

        # Increment recency for non-executed tests
        for test_id in self.test_history:
            if test_id not in verdicts:
                self.test_history[test_id]['builds_since_exec'] += 1

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Reset agent state for a new experiment."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.test_history = {}
        self.epsilon = 0.3


def run_retecs_experiment(
    df: pd.DataFrame,
    build_col: str = 'Build_ID',
    test_col: str = 'TC_Key',
    result_col: str = 'TE_Test_Result',
    duration_col: Optional[str] = None,
    reward_type: str = 'tcfail',
    n_episodes: int = 3
) -> Dict:
    """
    Run RETECS experiment on a dataset.

    Args:
        df: DataFrame with test execution data
        build_col: Column name for build ID
        test_col: Column name for test case ID
        result_col: Column name for test result
        duration_col: Optional column for test duration
        reward_type: Type of reward function
        n_episodes: Number of training episodes

    Returns:
        Dict with APFD scores and statistics
    """
    # Convert result to binary
    df = df.copy()
    if df[result_col].dtype == object:
        df['verdict'] = (df[result_col].str.upper() != 'PASS').astype(int)
    else:
        df['verdict'] = df[result_col].astype(int)

    # Get unique builds in order
    builds = df[build_col].unique().tolist()

    # Split: first 80% for training, last 20% for evaluation
    train_idx = int(len(builds) * 0.8)
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    # Filter to builds with at least one failure
    test_builds_with_failures = []
    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        if build_df['verdict'].sum() > 0:
            test_builds_with_failures.append(build_id)

    print(f"RETECS: Training on {len(train_builds)} builds")
    print(f"RETECS: Evaluating on {len(test_builds_with_failures)} builds with failures")

    # Initialize agent
    agent = RETECSAgent(reward_type=reward_type)

    # Training phase (multiple episodes)
    for episode in range(n_episodes):
        agent.reset()

        for build_id in train_builds:
            build_df = df[df[build_col] == build_id]
            test_ids = build_df[test_col].unique().tolist()

            # Get verdicts
            verdicts = dict(zip(build_df[test_col], build_df['verdict']))

            # Get durations if available
            durations = None
            if duration_col and duration_col in build_df.columns:
                durations = dict(zip(build_df[test_col], build_df[duration_col]))

            # Prioritize
            ranking = agent.prioritize(test_ids, build_id)

            # Update agent
            agent.update(build_id, ranking, verdicts, durations)

    # Evaluation phase
    apfd_scores = []

    for build_id in test_builds_with_failures:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts
        verdicts = dict(zip(build_df[test_col], build_df['verdict']))

        # Prioritize (no exploration)
        agent.epsilon = 0  # Disable exploration for evaluation
        ranking = agent.prioritize(test_ids, build_id)

        # Compute APFD
        n_tests = len(ranking)
        n_faults = sum(verdicts.values())

        if n_faults > 0:
            fault_positions = []
            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    fault_positions.append(i + 1)

            apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
            apfd_scores.append(apfd)

    results = {
        'method': 'RETECS',
        'reward_type': reward_type,
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores) if apfd_scores else 0,
        'std_apfd': np.std(apfd_scores) if apfd_scores else 0,
        'n_builds': len(apfd_scores)
    }

    print(f"RETECS ({reward_type}): Mean APFD = {results['mean_apfd']:.4f} "
          f"(+/- {results['std_apfd']:.4f}) on {results['n_builds']} builds")

    return results


if __name__ == '__main__':
    # Test with sample data
    print("RETECS Baseline Implementation")
    print("Usage: from src.baselines.retecs import run_retecs_experiment")
