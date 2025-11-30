"""
DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing

Reference:
    Chen, J., Bai, Y., Hao, D., Zhang, L., Zhang, L., & Xie, B. (2021).
    DeepOrder: Deep Learning for Test Case Prioritization in Continuous
    Integration Testing. ICSME 2021.

Implementation based on:
    https://github.com/T3AS/DeepOrder-ICSME21
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DeepOrderNet(nn.Module):
    """
    DeepOrder Neural Network for Test Case Prioritization.

    Uses a deep neural network to predict test failure probability
    based on historical execution features.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.2
    ):
        """
        Initialize DeepOrder network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class DeepOrderFeatureExtractor:
    """
    Extract features for DeepOrder from historical test execution data.

    Features (from the paper):
    1. Last execution verdict (0/1)
    2. Last execution duration (normalized)
    3. Failure rate (historical)
    4. Recent failure rate (last N builds)
    5. Execution count
    6. Average duration
    7. Time since last failure
    8. Consecutive failures/passes
    """

    def __init__(self, history_window: int = 10):
        """
        Initialize feature extractor.

        Args:
            history_window: Number of recent builds to consider
        """
        self.history_window = history_window
        self.test_history = {}  # test_id -> list of (build_id, verdict, duration)
        self.build_order = []  # ordered list of build IDs

    def update_history(
        self,
        build_id: str,
        test_results: Dict[str, Tuple[int, float]]
    ):
        """
        Update test history with results from a build.

        Args:
            build_id: Build identifier
            test_results: Dict mapping test_id -> (verdict, duration)
        """
        self.build_order.append(build_id)

        for test_id, (verdict, duration) in test_results.items():
            if test_id not in self.test_history:
                self.test_history[test_id] = []
            self.test_history[test_id].append((build_id, verdict, duration))

    def extract_features(self, test_id: str) -> np.ndarray:
        """
        Extract features for a test case.

        Args:
            test_id: Test case identifier

        Returns:
            Feature vector (8 dimensions)
        """
        if test_id not in self.test_history or len(self.test_history[test_id]) == 0:
            # New test - return default features
            return np.array([
                0.5,  # last_verdict (unknown)
                0.0,  # last_duration
                0.0,  # failure_rate
                0.0,  # recent_failure_rate
                0.0,  # exec_count (normalized)
                0.0,  # avg_duration
                1.0,  # time_since_failure (high = never failed)
                0.0   # consecutive_same
            ], dtype=np.float32)

        history = self.test_history[test_id]

        # Feature 1: Last execution verdict
        last_verdict = history[-1][1]

        # Feature 2: Last execution duration (normalized by max)
        last_duration = history[-1][2]
        max_duration = max(h[2] for h in history) if history else 1.0
        last_duration_norm = last_duration / (max_duration + 1e-6)

        # Feature 3: Historical failure rate
        total_failures = sum(h[1] for h in history)
        failure_rate = total_failures / len(history)

        # Feature 4: Recent failure rate (last N builds)
        recent = history[-self.history_window:] if len(history) >= self.history_window else history
        recent_failures = sum(h[1] for h in recent)
        recent_failure_rate = recent_failures / len(recent)

        # Feature 5: Execution count (normalized)
        exec_count = len(history) / (len(self.build_order) + 1)

        # Feature 6: Average duration (normalized)
        avg_duration = np.mean([h[2] for h in history])
        avg_duration_norm = avg_duration / (max_duration + 1e-6)

        # Feature 7: Time since last failure (normalized)
        time_since_failure = len(history)  # default: never failed
        for i, (_, verdict, _) in enumerate(reversed(history)):
            if verdict == 1:
                time_since_failure = i
                break
        time_since_failure_norm = time_since_failure / (len(history) + 1)

        # Feature 8: Consecutive same results
        consecutive_same = 1
        last_v = history[-1][1]
        for i in range(len(history) - 2, -1, -1):
            if history[i][1] == last_v:
                consecutive_same += 1
            else:
                break
        consecutive_same_norm = consecutive_same / (len(history) + 1)

        return np.array([
            last_verdict,
            last_duration_norm,
            failure_rate,
            recent_failure_rate,
            exec_count,
            avg_duration_norm,
            time_since_failure_norm,
            consecutive_same_norm
        ], dtype=np.float32)


class DeepOrderDataset(Dataset):
    """PyTorch Dataset for DeepOrder training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepOrderModel:
    """
    DeepOrder model for test case prioritization.

    Trains a deep neural network to predict test failure probability
    and uses predictions to rank test cases.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        history_window: int = 10,
        device: str = 'cpu'
    ):
        """
        Initialize DeepOrder model.

        Args:
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Training batch size
            history_window: Window for recent history features
            device: Device to use (cpu/cuda)
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.feature_extractor = DeepOrderFeatureExtractor(history_window)
        self.model = None

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
        build_col: str,
        test_col: str,
        result_col: str,
        duration_col: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from DataFrame."""
        features_list = []
        labels_list = []

        # Process builds in order
        builds = df[build_col].unique().tolist()

        for build_id in builds:
            build_df = df[df[build_col] == build_id]

            # Extract features for each test BEFORE updating history
            for _, row in build_df.iterrows():
                test_id = row[test_col]
                features = self.feature_extractor.extract_features(test_id)
                features_list.append(features)

                # Label: 1 if failed, 0 if passed
                verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
                labels_list.append(verdict)

            # Update history with this build's results
            test_results = {}
            for _, row in build_df.iterrows():
                test_id = row[test_col]
                verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
                duration = row[duration_col] if duration_col and duration_col in row else 1.0
                test_results[test_id] = (verdict, duration)

            self.feature_extractor.update_history(build_id, test_results)

        return np.array(features_list), np.array(labels_list)

    def train(
        self,
        df: pd.DataFrame,
        build_col: str = 'Build_ID',
        test_col: str = 'TC_Key',
        result_col: str = 'TE_Test_Result',
        duration_col: Optional[str] = None
    ):
        """
        Train the DeepOrder model.

        Args:
            df: Training DataFrame
            build_col: Column name for build ID
            test_col: Column name for test case ID
            result_col: Column name for test result
            duration_col: Optional column for test duration
        """
        # Reset feature extractor
        self.feature_extractor = DeepOrderFeatureExtractor()

        # Prepare data
        X, y = self._prepare_training_data(df, build_col, test_col, result_col, duration_col)

        print(f"DeepOrder: Training on {len(X)} samples")
        print(f"DeepOrder: Failure rate = {y.mean():.4f}")

        # Create model
        self.model = DeepOrderNet(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        # Create dataset and dataloader
        dataset = DeepOrderDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss and optimizer
        # Use weighted BCE for class imbalance
        pos_weight = torch.tensor([(1 - y.mean()) / (y.mean() + 1e-6)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"DeepOrder: Epoch {epoch + 1}/{self.epochs}, Loss = {total_loss / len(dataloader):.4f}")

    def prioritize(self, test_ids: List[str]) -> List[str]:
        """
        Prioritize test cases based on predicted failure probability.

        Args:
            test_ids: List of test IDs to prioritize

        Returns:
            Ordered list of test IDs (highest probability first)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        # Extract features for each test
        features_list = []
        for test_id in test_ids:
            features = self.feature_extractor.extract_features(test_id)
            features_list.append(features)

        X = np.array(features_list)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Sort by prediction (descending - highest failure probability first)
        test_scores = list(zip(test_ids, predictions))
        test_scores.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in test_scores]

    def update_history(
        self,
        build_id: str,
        test_results: Dict[str, Tuple[int, float]]
    ):
        """Update history after evaluation."""
        self.feature_extractor.update_history(build_id, test_results)


def run_deeporder_experiment(
    df: pd.DataFrame,
    build_col: str = 'Build_ID',
    test_col: str = 'TC_Key',
    result_col: str = 'TE_Test_Result',
    duration_col: Optional[str] = None,
    epochs: int = 30,
    train_ratio: float = 0.8
) -> Dict:
    """
    Run DeepOrder experiment on a dataset.

    Args:
        df: DataFrame with test execution data
        build_col: Column name for build ID
        test_col: Column name for test case ID
        result_col: Column name for test result
        duration_col: Optional column for test duration
        epochs: Number of training epochs
        train_ratio: Ratio of builds for training

    Returns:
        Dict with APFD scores and statistics
    """
    # Get unique builds in order
    builds = df[build_col].unique().tolist()

    # Split: first X% for training, rest for evaluation
    train_idx = int(len(builds) * train_ratio)
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    # Create train DataFrame
    train_df = df[df[build_col].isin(train_builds)]

    # Filter test builds to those with failures
    test_builds_with_failures = []
    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        if build_df[result_col].apply(lambda x: str(x).upper() != 'PASS').sum() > 0:
            test_builds_with_failures.append(build_id)

    print(f"DeepOrder: Training on {len(train_builds)} builds")
    print(f"DeepOrder: Evaluating on {len(test_builds_with_failures)} builds with failures")

    # Initialize and train model
    model = DeepOrderModel(epochs=epochs)
    model.train(train_df, build_col, test_col, result_col, duration_col)

    # Evaluation phase
    apfd_scores = []

    for build_id in test_builds_with_failures:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts
        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        # Prioritize
        ranking = model.prioritize(test_ids)

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

        # Update history for online learning
        test_results = {}
        for test_id, verdict in verdicts.items():
            duration = 1.0  # Default duration
            if duration_col:
                test_row = build_df[build_df[test_col] == test_id]
                if len(test_row) > 0 and duration_col in test_row.columns:
                    duration = test_row[duration_col].values[0]
            test_results[test_id] = (verdict, duration)
        model.update_history(build_id, test_results)

    results = {
        'method': 'DeepOrder',
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores) if apfd_scores else 0,
        'std_apfd': np.std(apfd_scores) if apfd_scores else 0,
        'n_builds': len(apfd_scores)
    }

    print(f"DeepOrder: Mean APFD = {results['mean_apfd']:.4f} "
          f"(+/- {results['std_apfd']:.4f}) on {results['n_builds']} builds")

    return results


if __name__ == '__main__':
    print("DeepOrder Baseline Implementation")
    print("Usage: from src.baselines.deeporder import run_deeporder_experiment")
