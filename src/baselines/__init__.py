"""
Baseline Methods for Test Case Prioritization.

This module implements multiple baseline methods for comparison with Filo-Priori v9:
1. Heuristic Baselines: Random, Recency, Failure-Rate, Greedy
2. ML Baselines: Random Forest, Logistic Regression, XGBoost
3. Statistical Validation: Bootstrap CI, t-tests, Cohen's d
4. Temporal Cross-Validation: K-fold temporal, Sliding window, Concept drift

Author: Filo-Priori Team
Date: 2025-11-26
"""

from .heuristic_baselines import (
    RandomBaseline,
    RecencyBaseline,
    FailureRateBaseline,
    GreedyHistoricalBaseline
)

from .ml_baselines import (
    RandomForestBaseline,
    LogisticRegressionBaseline,
    XGBoostBaseline,
    LSTMBaseline
)

from .statistical_validation import (
    bootstrap_confidence_interval,
    paired_ttest,
    wilcoxon_test,
    cohens_d,
    compare_methods,
    generate_comparison_table
)

from .temporal_cross_validation import (
    TemporalCrossValidator,
    ConceptDriftDetector,
    run_temporal_cv
)

__all__ = [
    # Heuristic
    'RandomBaseline',
    'RecencyBaseline',
    'FailureRateBaseline',
    'GreedyHistoricalBaseline',
    # ML
    'RandomForestBaseline',
    'LogisticRegressionBaseline',
    'XGBoostBaseline',
    'LSTMBaseline',
    # Statistical
    'bootstrap_confidence_interval',
    'paired_ttest',
    'wilcoxon_test',
    'cohens_d',
    'compare_methods',
    'generate_comparison_table',
    # Cross-Validation
    'TemporalCrossValidator',
    'ConceptDriftDetector',
    'run_temporal_cv'
]
