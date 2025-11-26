"""
Machine Learning Baselines for Test Case Prioritization.

Implements ML-based baselines using traditional algorithms:
1. Random Forest: Ensemble of decision trees
2. Logistic Regression: Linear classifier with probability output
3. XGBoost: Gradient boosting (if available)
4. LSTM: Simple sequence model for temporal patterns

These baselines use the same structural features as Filo-Priori
but with simpler models, allowing fair comparison.

Author: Filo-Priori Team
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Import base class
from .heuristic_baselines import BaseBaseline


class MLBaseline(BaseBaseline):
    """Base class for ML-based baselines."""

    def __init__(self, name: str, use_structural_features: bool = True):
        super().__init__(name)
        self.use_structural_features = use_structural_features
        self.model = None
        self.feature_names: List[str] = []
        self.scaler = None

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame.

        Uses the 10 selected structural features from Filo-Priori.
        """
        from sklearn.preprocessing import StandardScaler

        # Define feature columns (matching structural_feature_extractor_v2_5.py)
        feature_cols = [
            'test_age', 'failure_rate', 'recent_failure_rate', 'flakiness_rate',
            'consecutive_failures', 'max_consecutive_failures', 'failure_trend',
            'commit_count', 'test_novelty', 'cr_count'
        ]

        # Check which features are available
        available_features = [f for f in feature_cols if f in df.columns]

        if len(available_features) == 0:
            # Fallback: compute basic features from raw data
            logger.warning("No structural features found. Computing basic features from raw data.")
            return self._compute_basic_features(df)

        self.feature_names = available_features
        X = df[available_features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        return X

    def _compute_basic_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute basic features when structural features are not available.

        This is a fallback that computes features from raw test execution data.
        """
        features = []

        tc_key_col = 'TC_Key' if 'TC_Key' in df.columns else 'tc_id'

        # Group by TC to compute historical features
        tc_history: Dict[str, List[str]] = {}

        for _, row in df.iterrows():
            tc_key = row.get(tc_key_col, str(row.name))
            verdict = str(row.get('verdict', row.get('TE_Test_Result', ''))).strip()

            if tc_key not in tc_history:
                tc_history[tc_key] = []
            tc_history[tc_key].append(verdict)

        # Compute features for each row
        for _, row in df.iterrows():
            tc_key = row.get(tc_key_col, str(row.name))
            history = tc_history.get(tc_key, [])

            # Basic features
            num_executions = len(history)
            num_failures = sum(1 for v in history if v == 'Fail')
            failure_rate = num_failures / max(num_executions, 1)

            # Recent failure rate (last 5)
            recent = history[-5:] if len(history) >= 5 else history
            recent_failures = sum(1 for v in recent if v == 'Fail')
            recent_failure_rate = recent_failures / max(len(recent), 1)

            features.append([
                num_executions,
                failure_rate,
                recent_failure_rate,
                recent_failures,
                num_failures
            ])

        self.feature_names = ['num_executions', 'failure_rate', 'recent_failure_rate',
                             'recent_failures', 'num_failures']

        return np.array(features)

    def _get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract binary labels from DataFrame."""
        if 'verdict' in df.columns:
            return (df['verdict'].astype(str).str.strip() == 'Fail').astype(int).values
        elif 'TE_Test_Result' in df.columns:
            return (df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int).values
        elif 'label_binary' in df.columns:
            return df['label_binary'].astype(int).values
        else:
            raise ValueError("No label column found in DataFrame")


class RandomForestBaseline(MLBaseline):
    """
    Random Forest Baseline.

    Uses ensemble of decision trees for classification.
    Good for handling imbalanced data with class_weight='balanced'.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 class_weight: str = 'balanced',
                 random_state: int = 42):
        super().__init__(name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> 'RandomForestBaseline':
        """Fit Random Forest on training data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        # Prepare features and labels
        X = self._prepare_features(df)
        y = self._get_labels(df)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        self.is_fitted = True

        # Log feature importance
        importances = self.model.feature_importances_
        logger.info(f"RandomForest fitted with {self.n_estimators} trees")
        logger.info(f"  Top features: {sorted(zip(self.feature_names, importances), key=lambda x: -x[1])[:5]}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict failure probability."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Get probability of class 1 (Fail)
        proba = self.model.predict_proba(X_scaled)

        # Handle case where model only predicts one class
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dictionary."""
        if self.model is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))


class LogisticRegressionBaseline(MLBaseline):
    """
    Logistic Regression Baseline.

    Simple linear classifier with probability output.
    Fast training, interpretable coefficients.
    """

    def __init__(self,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 class_weight: str = 'balanced',
                 random_state: int = 42):
        super().__init__(name="LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> 'LogisticRegressionBaseline':
        """Fit Logistic Regression on training data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Prepare features and labels
        X = self._prepare_features(df)
        y = self._get_labels(df)

        # Scale features (important for LR)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver='lbfgs'
        )
        self.model.fit(X_scaled, y)

        self.is_fitted = True

        # Log coefficients
        logger.info(f"LogisticRegression fitted")
        logger.info(f"  Coefficients: {dict(zip(self.feature_names, self.model.coef_[0]))}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict failure probability."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)

        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

    def get_coefficients(self) -> Dict[str, float]:
        """Return coefficient dictionary."""
        if self.model is None:
            return {}
        return dict(zip(self.feature_names, self.model.coef_[0]))


class XGBoostBaseline(MLBaseline):
    """
    XGBoost Baseline.

    Gradient boosting with regularization.
    Often achieves best results among traditional ML methods.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 scale_pos_weight: float = None,  # Auto-computed from data
                 random_state: int = 42):
        super().__init__(name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

    def fit(self, df: pd.DataFrame) -> 'XGBoostBaseline':
        """Fit XGBoost on training data."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not installed. Using GradientBoosting from sklearn instead.")
            return self._fit_gradient_boosting(df)

        from sklearn.preprocessing import StandardScaler

        # Prepare features and labels
        X = self._prepare_features(df)
        y = self._get_labels(df)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Compute class imbalance ratio
        if self.scale_pos_weight is None:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            self.scale_pos_weight = n_neg / max(n_pos, 1)

        # Create and fit model
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info(f"XGBoost fitted with scale_pos_weight={self.scale_pos_weight:.2f}")

        return self

    def _fit_gradient_boosting(self, df: pd.DataFrame) -> 'XGBoostBaseline':
        """Fallback to sklearn GradientBoosting."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        X = self._prepare_features(df)
        y = self._get_labels(df)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        self.name = "GradientBoosting"  # Update name
        logger.info("GradientBoosting (sklearn) fitted as XGBoost fallback")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict failure probability."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)

        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]


class LSTMBaseline(MLBaseline):
    """
    Simple LSTM Baseline.

    Uses recurrent neural network to model temporal patterns.
    Simpler than Filo-Priori but captures sequence information.
    """

    def __init__(self,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32,
                 sequence_length: int = 10,
                 device: str = 'cuda',
                 random_state: int = 42):
        super().__init__(name="LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.random_state = random_state
        self.input_size = None

    def _build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences of test execution history.

        Returns:
            X: [N, seq_len, n_features]
            y: [N,]
        """
        from sklearn.preprocessing import StandardScaler

        # Prepare features
        X_flat = self._prepare_features(df)
        y = self._get_labels(df)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)

        self.input_size = X_scaled.shape[1]

        # Group by TC and build sequences
        tc_key_col = 'TC_Key' if 'TC_Key' in df.columns else 'tc_id'
        df_with_features = df.copy()
        df_with_features['_scaled_features'] = list(X_scaled)
        df_with_features['_label'] = y

        sequences = []
        labels = []

        for tc_key, tc_df in df_with_features.groupby(tc_key_col):
            tc_df = tc_df.sort_values('Build_ID')
            features_list = list(tc_df['_scaled_features'].values)
            labels_list = list(tc_df['_label'].values)

            # Pad or truncate to sequence_length
            for i in range(len(features_list)):
                # Get last sequence_length entries up to i
                start = max(0, i - self.sequence_length + 1)
                seq = features_list[start:i + 1]

                # Pad if needed
                while len(seq) < self.sequence_length:
                    seq.insert(0, np.zeros(self.input_size))

                sequences.append(np.array(seq))
                labels.append(labels_list[i])

        return np.array(sequences), np.array(labels)

    def fit(self, df: pd.DataFrame) -> 'LSTMBaseline':
        """Fit LSTM on training data."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch not available. LSTM baseline requires PyTorch.")
            raise ImportError("PyTorch is required for LSTM baseline")

        # Set seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build sequences
        X_seq, y = self._build_sequences(df)

        # Convert to tensors
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Create dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                out = self.fc(last_hidden)
                return self.sigmoid(out).squeeze()

        self.model = SimpleLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        # Handle class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.is_fitted = True
        self.device_obj = device
        logger.info(f"LSTM fitted: {self.hidden_size} hidden, {self.num_layers} layers")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict failure probability."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Build sequences (using fitted scaler)
        X_flat = self._prepare_features(df)
        X_scaled = self.scaler.transform(X_flat)

        # For inference, we use the last available features as the sequence
        # This is simplified - ideally we'd have full history
        X_seq = np.tile(X_scaled[:, np.newaxis, :], (1, self.sequence_length, 1))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device_obj)

        # Predict
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()

        return proba


# Factory function
def create_ml_baseline(name: str, **kwargs) -> MLBaseline:
    """
    Factory function to create ML baselines.

    Args:
        name: Baseline name ('rf', 'lr', 'xgb', 'lstm')
        **kwargs: Baseline-specific parameters

    Returns:
        Initialized baseline
    """
    baselines = {
        'rf': RandomForestBaseline,
        'random_forest': RandomForestBaseline,
        'lr': LogisticRegressionBaseline,
        'logistic_regression': LogisticRegressionBaseline,
        'xgb': XGBoostBaseline,
        'xgboost': XGBoostBaseline,
        'lstm': LSTMBaseline
    }

    name_lower = name.lower()
    if name_lower not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")

    return baselines[name_lower](**kwargs)


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    n_samples = 1000
    n_features = 10

    # Create synthetic feature data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    feature_names = [f'feature_{i}' for i in range(n_features)]

    demo_df = pd.DataFrame(X, columns=feature_names)
    demo_df['TC_Key'] = [f'TC_{i % 100}' for i in range(n_samples)]
    demo_df['Build_ID'] = [f'Build_{i // 20}' for i in range(n_samples)]
    demo_df['verdict'] = ['Fail' if yi == 1 else 'Pass' for yi in y]

    # Rename features to match expected names
    demo_df = demo_df.rename(columns={
        'feature_0': 'test_age',
        'feature_1': 'failure_rate',
        'feature_2': 'recent_failure_rate',
        'feature_3': 'flakiness_rate',
        'feature_4': 'consecutive_failures',
        'feature_5': 'max_consecutive_failures',
        'feature_6': 'failure_trend',
        'feature_7': 'commit_count',
        'feature_8': 'test_novelty',
        'feature_9': 'cr_count'
    })

    # Split
    train_df = demo_df.iloc[:800]
    test_df = demo_df.iloc[800:]

    print("=" * 60)
    print("ML BASELINES DEMO")
    print("=" * 60)

    # Test each baseline
    baselines = [
        RandomForestBaseline(),
        LogisticRegressionBaseline(),
        XGBoostBaseline()
    ]

    for baseline in baselines:
        print(f"\n--- {baseline.name} ---")
        baseline.fit(train_df)
        proba = baseline.predict_proba(test_df)
        print(f"Prediction range: [{proba.min():.4f}, {proba.max():.4f}]")
        print(f"Mean probability: {proba.mean():.4f}")
