"""
LightGBM LambdaRank Integration for Test Case Prioritization (Proposal #2).

This module provides LightGBM LambdaRank as a secondary ranker that can be used
in ensemble with the neural model (FT-Transformer or dual-stream) to optimize
NDCG/APFD scores.

Key features:
- LambdaRank objective directly optimizes NDCG (highly correlated with APFD)
- Can be used standalone or in stacking/ensemble with neural predictions
- Supports group-wise ranking (per-build ranking)
- Efficient gradient boosting complements neural network predictions

Reference:
- Burges et al., "From RankNet to LambdaRank to LambdaMART: An Overview", MSR-TR-2010-82
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_lightgbm_available() -> bool:
    """Check if LightGBM is available."""
    return LIGHTGBM_AVAILABLE


class LightGBMRanker:
    """
    LightGBM LambdaRank ranker for test case prioritization.

    Uses LambdaRank objective which directly optimizes NDCG,
    a metric highly correlated with APFD.
    """

    def __init__(
        self,
        objective: str = 'lambdarank',
        metric: str = 'ndcg',
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        lambdarank_truncation_level: int = 20,
        early_stopping_rounds: Optional[int] = 50,
        random_state: int = 42,
        verbose: int = -1
    ):
        """
        Initialize LightGBM ranker.

        Args:
            objective: LightGBM objective ('lambdarank' for NDCG optimization)
            metric: Evaluation metric ('ndcg')
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate for boosting
            num_leaves: Maximum number of leaves per tree
            max_depth: Maximum tree depth (-1 for no limit)
            min_child_samples: Minimum samples per leaf
            subsample: Row subsampling ratio
            colsample_bytree: Feature subsampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            lambdarank_truncation_level: Focus on top-k positions for NDCG
            early_stopping_rounds: Early stopping patience (None to disable)
            random_state: Random seed
            verbose: Verbosity level (-1 = silent)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'lambdarank_truncation_level': lambdarank_truncation_level,
            'random_state': random_state,
            'verbose': verbose,
            'force_row_wise': True,  # Prevent memory issues
        }

        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> 'LightGBMRanker':
        """
        Train the LightGBM ranker.

        Args:
            X: Training features [n_samples, n_features]
            y: Relevance labels (1=failure, 0=pass) [n_samples]
            group: Group sizes for each query/build [n_groups]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            group_val: Validation group sizes (optional)
            feature_names: Optional feature names
            categorical_features: Indices of categorical features

        Returns:
            self
        """
        self.feature_names = feature_names

        # Create LightGBM datasets
        train_set = lgb.Dataset(
            X, label=y, group=group,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        valid_sets = [train_set]
        valid_names = ['train']

        if X_val is not None and y_val is not None and group_val is not None:
            val_set = lgb.Dataset(
                X_val, label=y_val, group=group_val,
                reference=train_set,
                feature_name=feature_names,
                categorical_feature=categorical_features
            )
            valid_sets.append(val_set)
            valid_names.append('valid')

        # Configure callbacks
        callbacks = []
        if self.early_stopping_rounds is not None and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=self.params['verbose'] >= 0
            ))

        if self.params['verbose'] >= 0:
            callbacks.append(lgb.log_evaluation(period=10))

        # Train the model
        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else None
        )

        logger.info(f"LightGBM trained with {self.model.num_trees()} trees")

        return self

    def predict(
        self,
        X: np.ndarray,
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict relevance scores.

        Args:
            X: Features [n_samples, n_features]
            group: Optional group sizes (not used in prediction)

        Returns:
            Predicted scores [n_samples]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: 'gain', 'split', or 'cover'

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importance = self.model.feature_importance(importance_type=importance_type)

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}

    def calculate_apfd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray
    ) -> Tuple[float, List[float]]:
        """
        Calculate APFD using model predictions.

        Args:
            X: Features [n_samples, n_features]
            y: True labels (1=failure, 0=pass)
            group: Group sizes per build

        Returns:
            Tuple of (mean_apfd, list_of_per_build_apfd)
        """
        scores = self.predict(X)

        apfd_scores = []
        offset = 0

        for g_size in group:
            g_scores = scores[offset:offset + g_size]
            g_labels = y[offset:offset + g_size]

            # Sort by predicted score (descending)
            order = np.argsort(-g_scores)
            sorted_labels = g_labels[order]

            # Calculate APFD
            n = len(sorted_labels)
            m = sorted_labels.sum()

            if m > 0 and n > 0:
                failure_positions = np.where(sorted_labels == 1)[0] + 1
                apfd = 1.0 - failure_positions.sum() / (n * m) + 1.0 / (2 * n)
                apfd_scores.append(apfd)

            offset += g_size

        mean_apfd = np.mean(apfd_scores) if apfd_scores else 0.0
        return mean_apfd, apfd_scores

    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        self.model.save_model(path)
        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: str) -> 'LightGBMRanker':
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"LightGBM model loaded from {path}")
        return self


class HybridRanker:
    """
    Hybrid ranker combining neural model predictions with LightGBM.

    Supports two modes:
    1. Ensemble: Weighted average of neural and LightGBM predictions
    2. Stacking: Use neural predictions as additional features for LightGBM
    """

    def __init__(
        self,
        mode: str = 'ensemble',
        neural_weight: float = 0.6,
        lgbm_weight: float = 0.4,
        lgbm_config: Optional[Dict] = None
    ):
        """
        Initialize hybrid ranker.

        Args:
            mode: 'ensemble' or 'stacking'
            neural_weight: Weight for neural predictions (ensemble mode)
            lgbm_weight: Weight for LightGBM predictions (ensemble mode)
            lgbm_config: Configuration for LightGBM ranker
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.mode = mode
        self.neural_weight = neural_weight
        self.lgbm_weight = lgbm_weight

        lgbm_config = lgbm_config or {}
        self.lgbm_ranker = LightGBMRanker(**lgbm_config)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        neural_predictions: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
        neural_predictions_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'HybridRanker':
        """
        Train the hybrid ranker.

        Args:
            X: Base features [n_samples, n_features]
            y: Labels (1=failure, 0=pass)
            group: Group sizes per build
            neural_predictions: Neural model predictions (required for stacking)
            X_val, y_val, group_val: Validation data (optional)
            neural_predictions_val: Neural predictions for validation
            feature_names: Feature names

        Returns:
            self
        """
        if self.mode == 'stacking':
            if neural_predictions is None:
                raise ValueError("neural_predictions required for stacking mode")

            # Add neural predictions as additional feature
            X_augmented = np.column_stack([X, neural_predictions])

            if feature_names is not None:
                feature_names = feature_names + ['neural_prob']

            X_val_augmented = None
            if X_val is not None and neural_predictions_val is not None:
                X_val_augmented = np.column_stack([X_val, neural_predictions_val])

            self.lgbm_ranker.fit(
                X_augmented, y, group,
                X_val_augmented, y_val, group_val,
                feature_names=feature_names
            )
        else:
            # Ensemble mode: train LightGBM on base features
            self.lgbm_ranker.fit(
                X, y, group,
                X_val, y_val, group_val,
                feature_names=feature_names
            )

        return self

    def predict(
        self,
        X: np.ndarray,
        neural_predictions: Optional[np.ndarray] = None,
        group: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict using hybrid approach.

        Args:
            X: Features [n_samples, n_features]
            neural_predictions: Neural model predictions
            group: Group sizes (optional)

        Returns:
            Hybrid predictions [n_samples]
        """
        if self.mode == 'stacking':
            if neural_predictions is None:
                raise ValueError("neural_predictions required for stacking mode")

            X_augmented = np.column_stack([X, neural_predictions])
            return self.lgbm_ranker.predict(X_augmented)
        else:
            # Ensemble: weighted average
            lgbm_scores = self.lgbm_ranker.predict(X)

            if neural_predictions is not None:
                # Normalize both to [0, 1] for fair weighting
                lgbm_norm = (lgbm_scores - lgbm_scores.min()) / (lgbm_scores.max() - lgbm_scores.min() + 1e-10)
                neural_norm = neural_predictions

                return self.neural_weight * neural_norm + self.lgbm_weight * lgbm_norm
            else:
                return lgbm_scores

    def calculate_apfd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        neural_predictions: Optional[np.ndarray] = None
    ) -> Tuple[float, List[float]]:
        """
        Calculate APFD using hybrid predictions.

        Returns:
            Tuple of (mean_apfd, list_of_per_build_apfd)
        """
        scores = self.predict(X, neural_predictions, group)

        apfd_scores = []
        offset = 0

        for g_size in group:
            g_scores = scores[offset:offset + g_size]
            g_labels = y[offset:offset + g_size]

            order = np.argsort(-g_scores)
            sorted_labels = g_labels[order]

            n = len(sorted_labels)
            m = sorted_labels.sum()

            if m > 0 and n > 0:
                failure_positions = np.where(sorted_labels == 1)[0] + 1
                apfd = 1.0 - failure_positions.sum() / (n * m) + 1.0 / (2 * n)
                apfd_scores.append(apfd)

            offset += g_size

        mean_apfd = np.mean(apfd_scores) if apfd_scores else 0.0
        return mean_apfd, apfd_scores


def train_lgbm_ranker_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    build_ids: np.ndarray,
    config: Dict,
    val_split: float = 0.2
) -> Tuple[LightGBMRanker, Dict]:
    """
    Train LightGBM ranker directly from embeddings.

    Convenience function for training from neural network embeddings.

    Args:
        embeddings: Feature embeddings [n_samples, embed_dim]
        labels: Binary labels (1=failure, 0=pass)
        build_ids: Build IDs for grouping
        config: Configuration dictionary with 'lightgbm_ranker' section
        val_split: Fraction of data for validation

    Returns:
        Tuple of (trained_ranker, metrics_dict)
    """
    lgbm_config = config.get('lightgbm_ranker', {})

    # Convert build_ids to groups
    unique_builds = np.unique(build_ids)
    n_val = int(len(unique_builds) * val_split)

    val_builds = set(unique_builds[-n_val:]) if n_val > 0 else set()
    train_mask = np.array([b not in val_builds for b in build_ids])
    val_mask = ~train_mask

    # Split data
    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    build_ids_train = build_ids[train_mask]

    X_val = embeddings[val_mask] if val_mask.any() else None
    y_val = labels[val_mask] if val_mask.any() else None
    build_ids_val = build_ids[val_mask] if val_mask.any() else None

    # Compute group sizes
    def compute_groups(bids):
        if bids is None:
            return None
        groups = []
        current_build = bids[0]
        count = 0
        for bid in bids:
            if bid == current_build:
                count += 1
            else:
                groups.append(count)
                current_build = bid
                count = 1
        groups.append(count)
        return np.array(groups)

    group_train = compute_groups(build_ids_train)
    group_val = compute_groups(build_ids_val)

    # Initialize and train ranker
    ranker = LightGBMRanker(
        objective=lgbm_config.get('objective', 'lambdarank'),
        metric=lgbm_config.get('metric', 'ndcg'),
        n_estimators=lgbm_config.get('n_estimators', 300),
        learning_rate=lgbm_config.get('learning_rate', 0.05),
        num_leaves=lgbm_config.get('num_leaves', 31),
        max_depth=lgbm_config.get('max_depth', -1),
        min_child_samples=lgbm_config.get('min_child_samples', 20),
        subsample=lgbm_config.get('subsample', 0.8),
        colsample_bytree=lgbm_config.get('colsample_bytree', 0.8),
        reg_alpha=lgbm_config.get('reg_alpha', 0.1),
        reg_lambda=lgbm_config.get('reg_lambda', 0.1),
        lambdarank_truncation_level=lgbm_config.get('lambdarank_truncation_level', 20),
        early_stopping_rounds=lgbm_config.get('early_stopping_rounds', 50),
        random_state=lgbm_config.get('random_state', 42)
    )

    ranker.fit(
        X_train, y_train, group_train,
        X_val, y_val, group_val
    )

    # Calculate metrics
    metrics = {}

    train_apfd, _ = ranker.calculate_apfd(X_train, y_train, group_train)
    metrics['train_apfd'] = train_apfd

    if X_val is not None:
        val_apfd, _ = ranker.calculate_apfd(X_val, y_val, group_val)
        metrics['val_apfd'] = val_apfd

    metrics['feature_importance'] = ranker.get_feature_importance()

    logger.info(f"LightGBM Training APFD: {train_apfd:.4f}")
    if 'val_apfd' in metrics:
        logger.info(f"LightGBM Validation APFD: {metrics['val_apfd']:.4f}")

    return ranker, metrics
