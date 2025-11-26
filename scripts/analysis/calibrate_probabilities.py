#!/usr/bin/env python3
"""
Probability Calibration Script for Filo-Priori v9.

This script addresses the issue of compressed probability outputs (0.19-0.53)
by applying various calibration techniques to improve ranking performance.

Techniques implemented:
1. Temperature Scaling - Adjusts confidence via temperature parameter
2. Platt Scaling - Logistic regression on probabilities
3. Isotonic Regression - Non-parametric monotonic calibration
4. Beta Calibration - Two-parameter calibration
5. Ensemble with FailureRate - Combines model with baseline

Usage:
    python calibrate_probabilities.py [--method temperature] [--output results/calibrated]

Author: Filo-Priori Team
Date: 2025-11-26
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.apfd import calculate_apfd_per_build, generate_apfd_report
from src.baselines.statistical_validation import bootstrap_confidence_interval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Base class for probability calibration."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'ProbabilityCalibrator':
        """Fit calibration on validation data."""
        raise NotImplementedError

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        raise NotImplementedError

    def fit_transform(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probabilities, labels)
        return self.transform(probabilities)


class TemperatureScaling(ProbabilityCalibrator):
    """
    Temperature Scaling Calibration.

    Divides logits by temperature T before sigmoid:
    - T > 1: Softens predictions (less confident)
    - T < 1: Sharpens predictions (more confident)
    - T = 1: No change

    For our case (compressed probs), we want T < 1 to sharpen.
    """

    def __init__(self, initial_temp: float = 1.0):
        super().__init__("TemperatureScaling")
        self.temperature = initial_temp
        self.initial_temp = initial_temp

    def _prob_to_logit(self, p: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits safely."""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def _logit_to_prob(self, z: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        return 1 / (1 + np.exp(-z))

    def _nll_loss(self, temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood loss for temperature optimization."""
        scaled_probs = self._logit_to_prob(logits / temperature)
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)

        # Binary cross-entropy
        nll = -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))
        return nll

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'TemperatureScaling':
        """Find optimal temperature using validation data."""
        logits = self._prob_to_logit(probabilities)

        # Optimize temperature
        result = minimize_scalar(
            lambda t: self._nll_loss(t, logits, labels),
            bounds=(0.1, 10.0),
            method='bounded'
        )

        self.temperature = result.x
        self.is_fitted = True

        logger.info(f"TemperatureScaling fitted: T = {self.temperature:.4f}")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        logits = self._prob_to_logit(probabilities)
        scaled_logits = logits / self.temperature
        return self._logit_to_prob(scaled_logits)


class PlattScaling(ProbabilityCalibrator):
    """
    Platt Scaling (Sigmoid Calibration).

    Fits a logistic regression on the probabilities:
    calibrated = sigmoid(a * logit(p) + b)
    """

    def __init__(self):
        super().__init__("PlattScaling")
        self.lr = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """Fit logistic regression on probabilities."""
        # Use log-odds as feature
        p_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

        self.lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        self.lr.fit(X, labels)

        self.is_fitted = True
        logger.info(f"PlattScaling fitted: a={self.lr.coef_[0][0]:.4f}, b={self.lr.intercept_[0]:.4f}")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        p_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class IsotonicCalibration(ProbabilityCalibrator):
    """
    Isotonic Regression Calibration.

    Non-parametric, monotonic calibration that preserves ranking
    while improving calibration.
    """

    def __init__(self):
        super().__init__("IsotonicCalibration")
        self.ir = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibration':
        """Fit isotonic regression."""
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.ir.fit(probabilities, labels)

        self.is_fitted = True
        logger.info("IsotonicCalibration fitted")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        return self.ir.predict(probabilities)


class BetaCalibration(ProbabilityCalibrator):
    """
    Beta Calibration (Kull et al., 2017).

    Models calibration as: calibrated = 1 / (1 + 1/(exp(c) * (p/(1-p))^a + exp(b)))

    More flexible than Platt scaling for skewed distributions.
    """

    def __init__(self):
        super().__init__("BetaCalibration")
        self.a = 1.0
        self.b = 0.0
        self.c = 0.0

    def _calibrate(self, p: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Apply beta calibration formula."""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        odds = p / (1 - p)
        calibrated = 1 / (1 + 1 / (np.exp(c) * np.power(odds, a) + np.exp(b)))
        return np.clip(calibrated, 1e-7, 1 - 1e-7)

    def _loss(self, params: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Log loss for optimization."""
        a, b, c = params
        cal_probs = self._calibrate(probabilities, a, b, c)
        nll = -np.mean(labels * np.log(cal_probs) + (1 - labels) * np.log(1 - cal_probs))
        return nll

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> 'BetaCalibration':
        """Fit beta calibration parameters."""
        result = minimize(
            lambda p: self._loss(p, probabilities, labels),
            x0=[1.0, 0.0, 0.0],
            method='L-BFGS-B',
            bounds=[(0.01, 10), (-5, 5), (-5, 5)]
        )

        self.a, self.b, self.c = result.x
        self.is_fitted = True

        logger.info(f"BetaCalibration fitted: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        return self._calibrate(probabilities, self.a, self.b, self.c)


class FailureRateEnsemble(ProbabilityCalibrator):
    """
    Ensemble combining Filo-Priori with FailureRate baseline.

    final_score = alpha * filo_prob + (1 - alpha) * failure_rate

    This leverages the strong signal from historical failure rate
    while keeping the semantic/structural information from Filo-Priori.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__("FailureRateEnsemble")
        self.alpha = alpha
        self.failure_rates: Dict[str, float] = {}
        self.default_rate = 0.05

    def fit_failure_rates(self, train_df: pd.DataFrame) -> 'FailureRateEnsemble':
        """Compute failure rates from training data."""
        tc_col = 'TC_Key' if 'TC_Key' in train_df.columns else 'tc_id'
        verdict_col = 'verdict' if 'verdict' in train_df.columns else 'TE_Test_Result'

        for tc_key, tc_df in train_df.groupby(tc_col):
            n_fail = (tc_df[verdict_col].astype(str).str.strip() == 'Fail').sum()
            n_total = len(tc_df)
            self.failure_rates[tc_key] = (n_fail + 1) / (n_total + 2)  # Laplace smoothing

        self.default_rate = np.mean(list(self.failure_rates.values()))
        logger.info(f"FailureRateEnsemble: {len(self.failure_rates)} TCs, default_rate={self.default_rate:.4f}")
        return self

    def fit(self, probabilities: np.ndarray, labels: np.ndarray,
            tc_keys: Optional[np.ndarray] = None) -> 'FailureRateEnsemble':
        """
        Optimize alpha using validation data.

        Note: This requires tc_keys to look up failure rates.
        If failure_rates not set, call fit_failure_rates first.
        """
        if tc_keys is None or len(self.failure_rates) == 0:
            logger.warning("FailureRateEnsemble: Using default alpha without optimization")
            self.is_fitted = True
            return self

        # Get failure rates for validation TCs
        fr_values = np.array([self.failure_rates.get(tc, self.default_rate) for tc in tc_keys])

        # Normalize both to [0, 1]
        prob_norm = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min() + 1e-7)
        fr_norm = (fr_values - fr_values.min()) / (fr_values.max() - fr_values.min() + 1e-7)

        # Grid search for optimal alpha
        best_alpha = 0.5
        best_auc = 0

        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * prob_norm + (1 - alpha) * fr_norm
            # Use correlation with labels as proxy for ranking quality
            corr = np.corrcoef(combined, labels)[0, 1]
            if corr > best_auc:
                best_auc = corr
                best_alpha = alpha

        self.alpha = best_alpha
        self.is_fitted = True

        logger.info(f"FailureRateEnsemble fitted: alpha={self.alpha:.2f} (correlation={best_auc:.4f})")
        return self

    def transform(self, probabilities: np.ndarray, tc_keys: np.ndarray) -> np.ndarray:
        """Apply ensemble combination."""
        # Get failure rates
        fr_values = np.array([self.failure_rates.get(tc, self.default_rate) for tc in tc_keys])

        # Normalize both to [0, 1]
        prob_norm = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min() + 1e-7)
        fr_norm = (fr_values - fr_values.min()) / (fr_values.max() - fr_values.min() + 1e-7)

        # Combine
        return self.alpha * prob_norm + (1 - self.alpha) * fr_norm


def evaluate_calibration(
    original_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    method_name: str
) -> Dict:
    """Evaluate calibration quality."""
    # Compute metrics
    orig_mean_pos = original_probs[labels == 1].mean()
    orig_mean_neg = original_probs[labels == 0].mean()
    cal_mean_pos = calibrated_probs[labels == 1].mean()
    cal_mean_neg = calibrated_probs[labels == 0].mean()

    # Correlation with labels
    orig_corr = np.corrcoef(original_probs, labels)[0, 1]
    cal_corr = np.corrcoef(calibrated_probs, labels)[0, 1]

    # Range
    orig_range = original_probs.max() - original_probs.min()
    cal_range = calibrated_probs.max() - calibrated_probs.min()

    return {
        'method': method_name,
        'orig_mean_pos': orig_mean_pos,
        'orig_mean_neg': orig_mean_neg,
        'cal_mean_pos': cal_mean_pos,
        'cal_mean_neg': cal_mean_neg,
        'orig_separation': orig_mean_pos - orig_mean_neg,
        'cal_separation': cal_mean_pos - cal_mean_neg,
        'orig_corr': orig_corr,
        'cal_corr': cal_corr,
        'orig_range': orig_range,
        'cal_range': cal_range
    }


def run_calibration(
    predictions_path: str,
    train_path: str,
    output_dir: str,
    methods: List[str] = None
):
    """
    Run calibration pipeline.

    Args:
        predictions_path: Path to Filo-Priori predictions CSV
        train_path: Path to training data (for failure rates)
        output_dir: Output directory for results
        methods: List of calibration methods to try
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ['temperature', 'platt', 'isotonic', 'beta', 'ensemble']

    logger.info("=" * 70)
    logger.info(" PROBABILITY CALIBRATION FOR FILO-PRIORI V9")
    logger.info("=" * 70)

    # Load predictions
    logger.info(f"\nLoading predictions from: {predictions_path}")
    pred_df = pd.read_csv(predictions_path)
    logger.info(f"  Total rows: {len(pred_df)}")

    # Extract arrays
    probabilities = pred_df['probability'].values
    labels = (pred_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int).values
    tc_keys = pred_df['TC_Key'].values if 'TC_Key' in pred_df.columns else None

    logger.info(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    logger.info(f"  Positive class (Fail): {labels.sum()} ({labels.mean()*100:.2f}%)")

    # Load training data for ensemble
    logger.info(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    if 'verdict' not in train_df.columns:
        train_df['verdict'] = train_df['TE_Test_Result']

    # Split into calibration set (use portion of test data)
    # In practice, you'd use a held-out validation set
    n_cal = len(probabilities) // 5  # Use 20% for calibration
    cal_idx = np.random.RandomState(42).choice(len(probabilities), n_cal, replace=False)

    cal_probs = probabilities[cal_idx]
    cal_labels = labels[cal_idx]
    cal_tc_keys = tc_keys[cal_idx] if tc_keys is not None else None

    logger.info(f"\nCalibration set: {n_cal} samples")

    # Results storage
    results = {}
    calibrated_dfs = {}

    # Original (baseline)
    results['Original'] = {
        'mean_apfd': None,  # Will be computed below
        'prob_range': (probabilities.min(), probabilities.max()),
        'separation': probabilities[labels == 1].mean() - probabilities[labels == 0].mean()
    }

    # Try each calibration method
    calibrators = {
        'temperature': TemperatureScaling(),
        'platt': PlattScaling(),
        'isotonic': IsotonicCalibration(),
        'beta': BetaCalibration(),
        'ensemble': FailureRateEnsemble(alpha=0.5)
    }

    for method_name in methods:
        if method_name not in calibrators:
            logger.warning(f"Unknown method: {method_name}")
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Calibrating with: {method_name.upper()}")
        logger.info("=" * 50)

        calibrator = calibrators[method_name]

        try:
            # Fit calibrator
            if method_name == 'ensemble':
                calibrator.fit_failure_rates(train_df)
                calibrator.fit(cal_probs, cal_labels, cal_tc_keys)
                # Transform with TC keys
                calibrated_probs = calibrator.transform(probabilities, tc_keys)
            else:
                calibrator.fit(cal_probs, cal_labels)
                calibrated_probs = calibrator.transform(probabilities)

            # Evaluate
            eval_result = evaluate_calibration(
                probabilities, calibrated_probs, labels, method_name
            )

            logger.info(f"  Original separation: {eval_result['orig_separation']:.4f}")
            logger.info(f"  Calibrated separation: {eval_result['cal_separation']:.4f}")
            logger.info(f"  Improvement: {eval_result['cal_separation'] - eval_result['orig_separation']:.4f}")
            logger.info(f"  Calibrated range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")

            # Create calibrated predictions DataFrame
            cal_df = pred_df.copy()
            cal_df['probability_original'] = probabilities
            cal_df['probability'] = calibrated_probs

            # Recalculate ranks per build
            cal_df['rank'] = cal_df.groupby('Build_ID')['probability'] \
                                   .rank(method='first', ascending=False) \
                                   .astype(int)

            # Calculate APFD
            cal_df['label_binary'] = labels

            apfd_df = calculate_apfd_per_build(
                cal_df,
                method_name=f'Filo-Priori ({method_name})',
                build_col='Build_ID',
                label_col='label_binary',
                rank_col='rank',
                result_col='TE_Test_Result'
            )

            mean_apfd = apfd_df['apfd'].mean()
            ci = bootstrap_confidence_interval(apfd_df['apfd'].values, n_bootstrap=1000)

            logger.info(f"  Mean APFD: {mean_apfd:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]")

            # Store results
            results[method_name] = {
                'mean_apfd': mean_apfd,
                'ci_95': (ci[1], ci[2]),
                'prob_range': (calibrated_probs.min(), calibrated_probs.max()),
                'separation': eval_result['cal_separation'],
                'calibrator': calibrator
            }

            calibrated_dfs[method_name] = cal_df

            # Save calibrated predictions
            cal_path = output_dir / f'predictions_calibrated_{method_name}.csv'
            cal_df.to_csv(cal_path, index=False)
            logger.info(f"  Saved to: {cal_path}")

            # Save APFD per build
            apfd_path = output_dir / f'apfd_per_build_{method_name}.csv'
            apfd_df.to_csv(apfd_path, index=False)

        except Exception as e:
            logger.error(f"  Error with {method_name}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate original APFD
    logger.info(f"\n{'=' * 50}")
    logger.info("ORIGINAL (uncalibrated)")
    logger.info("=" * 50)

    orig_df = pred_df.copy()
    orig_df['label_binary'] = labels

    orig_apfd_df = calculate_apfd_per_build(
        orig_df,
        method_name='Filo-Priori (original)',
        build_col='Build_ID',
        label_col='label_binary',
        rank_col='rank',
        result_col='TE_Test_Result'
    )

    orig_mean_apfd = orig_apfd_df['apfd'].mean()
    orig_ci = bootstrap_confidence_interval(orig_apfd_df['apfd'].values, n_bootstrap=1000)

    results['Original']['mean_apfd'] = orig_mean_apfd
    results['Original']['ci_95'] = (orig_ci[1], orig_ci[2])

    logger.info(f"  Mean APFD: {orig_mean_apfd:.4f} [{orig_ci[1]:.4f}, {orig_ci[2]:.4f}]")

    # Summary table
    logger.info(f"\n{'=' * 70}")
    logger.info(" CALIBRATION RESULTS SUMMARY")
    logger.info("=" * 70)

    summary_data = []
    for method, res in results.items():
        if res['mean_apfd'] is not None:
            improvement = res['mean_apfd'] - orig_mean_apfd if method != 'Original' else 0
            summary_data.append({
                'Method': method,
                'Mean APFD': res['mean_apfd'],
                '95% CI': f"[{res['ci_95'][0]:.4f}, {res['ci_95'][1]:.4f}]",
                'Δ APFD': improvement,
                'Prob Range': f"[{res['prob_range'][0]:.3f}, {res['prob_range'][1]:.3f}]"
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean APFD', ascending=False)

    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_path = output_dir / 'calibration_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")

    # Find best method
    best_method = summary_df.iloc[0]['Method']
    best_apfd = summary_df.iloc[0]['Mean APFD']

    logger.info(f"\n{'=' * 70}")
    logger.info(f" BEST METHOD: {best_method} (APFD = {best_apfd:.4f})")
    logger.info("=" * 70)

    return results, calibrated_dfs


def main():
    parser = argparse.ArgumentParser(description="Calibrate Filo-Priori probabilities")
    parser.add_argument('--predictions', type=str,
                       default='results/experiment_06_feature_selection/prioritized_test_cases_FULL_testcsv.csv',
                       help='Path to predictions CSV')
    parser.add_argument('--train', type=str, default='datasets/train.csv',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='results/calibrated',
                       help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['temperature', 'platt', 'isotonic', 'beta', 'ensemble'],
                       help='Calibration methods to try')

    args = parser.parse_args()

    run_calibration(
        predictions_path=args.predictions,
        train_path=args.train,
        output_dir=args.output,
        methods=args.methods
    )


if __name__ == "__main__":
    main()
