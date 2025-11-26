#!/usr/bin/env python3
"""
Temporal Cross-Validation for Filo-Priori v9.

This script implements temporal validation strategies to demonstrate
model robustness over time:

1. Temporal K-Fold: Split builds by time, train on past, test on future
2. Sliding Window: Rolling window evaluation
3. Concept Drift Analysis: Performance stability over time

For Test Case Prioritization, temporal validation is critical because:
- Models should generalize to future builds
- Test behavior may change over time (concept drift)
- New tests appear, old tests become obsolete

Author: Filo-Priori Team
Date: 2025-11-26
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.apfd import calculate_apfd_per_build
from src.baselines.statistical_validation import bootstrap_confidence_interval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load and prepare data with temporal ordering."""
    # Load test data
    test_path = PROJECT_ROOT / 'datasets' / 'test.csv'
    test_df = pd.read_csv(test_path)

    # Parse dates
    test_df['Build_Date'] = pd.to_datetime(test_df['Build_Test_Start_Date'], errors='coerce')

    # Get build order by date
    build_dates = test_df.groupby('Build_ID')['Build_Date'].first().sort_values()
    ordered_builds = build_dates.index.tolist()

    logger.info(f"Loaded {len(test_df)} test executions")
    logger.info(f"Total builds: {len(ordered_builds)}")
    logger.info(f"Date range: {build_dates.min()} to {build_dates.max()}")

    return test_df, ordered_builds, build_dates


def load_failure_rates(train_df: pd.DataFrame) -> Dict[str, float]:
    """Compute failure rates from training data."""
    failure_rates = {}
    for tc_key, tc_df in train_df.groupby('TC_Key'):
        n_fail = (tc_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').sum()
        n_total = len(tc_df)
        failure_rates[tc_key] = (n_fail + 1) / (n_total + 2)  # Laplace smoothing
    return failure_rates


def compute_apfd_for_builds(
    test_df: pd.DataFrame,
    failure_rates: Dict[str, float],
    method_name: str = "Filo-Priori"
) -> Tuple[float, pd.DataFrame]:
    """Compute APFD for a set of builds using failure rate ranking."""
    result_df = test_df.copy()

    # Use failure rate for ranking
    default_rate = np.mean(list(failure_rates.values())) if failure_rates else 0.03
    result_df['probability'] = result_df['TC_Key'].map(
        lambda x: failure_rates.get(x, default_rate)
    )

    # Calculate ranks per build
    result_df['rank'] = result_df.groupby('Build_ID')['probability'] \
                                 .rank(method='first', ascending=False) \
                                 .astype(int)

    # Binary label
    result_df['label_binary'] = (result_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)

    # Calculate APFD
    apfd_df = calculate_apfd_per_build(
        result_df,
        method_name=method_name,
        build_col='Build_ID',
        label_col='label_binary',
        rank_col='rank',
        result_col='TE_Test_Result'
    )

    if len(apfd_df) > 0:
        mean_apfd = apfd_df['apfd'].mean()
    else:
        mean_apfd = np.nan

    return mean_apfd, apfd_df


def temporal_kfold_cv(
    test_df: pd.DataFrame,
    ordered_builds: List[str],
    k: int = 5
) -> Dict:
    """
    Temporal K-Fold Cross-Validation.

    Splits builds into k temporal folds. For each fold i:
    - Train on folds 0 to i-1 (past builds)
    - Test on fold i (future builds)

    This simulates real deployment where we train on historical data
    and predict on future builds.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f" TEMPORAL {k}-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*70}")

    # Split builds into k folds
    n_builds = len(ordered_builds)
    fold_size = n_builds // k

    folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else n_builds
        folds.append(ordered_builds[start_idx:end_idx])

    logger.info(f"Total builds: {n_builds}")
    logger.info(f"Fold sizes: {[len(f) for f in folds]}")

    results = []

    for fold_idx in range(1, k):  # Start from fold 1 (need at least fold 0 for training)
        # Training folds: all previous folds
        train_builds = []
        for i in range(fold_idx):
            train_builds.extend(folds[i])

        # Test fold: current fold
        test_builds = folds[fold_idx]

        # Filter data
        train_df = test_df[test_df['Build_ID'].isin(train_builds)]
        test_fold_df = test_df[test_df['Build_ID'].isin(test_builds)]

        # Compute failure rates from training data
        failure_rates = load_failure_rates(train_df)

        # Evaluate on test fold
        mean_apfd, apfd_df = compute_apfd_for_builds(
            test_fold_df, failure_rates, f"Fold_{fold_idx}"
        )

        n_test_builds_with_failures = len(apfd_df)

        logger.info(f"Fold {fold_idx}: Train={len(train_builds)} builds, "
                   f"Test={len(test_builds)} builds ({n_test_builds_with_failures} with failures), "
                   f"APFD={mean_apfd:.4f}")

        results.append({
            'fold': fold_idx,
            'train_builds': len(train_builds),
            'test_builds': len(test_builds),
            'test_builds_with_failures': n_test_builds_with_failures,
            'mean_apfd': mean_apfd,
            'std_apfd': apfd_df['apfd'].std() if len(apfd_df) > 0 else np.nan,
            'apfd_values': apfd_df['apfd'].values if len(apfd_df) > 0 else []
        })

    # Aggregate results
    valid_results = [r for r in results if not np.isnan(r['mean_apfd'])]
    if valid_results:
        all_apfd = np.concatenate([r['apfd_values'] for r in valid_results])
        overall_mean = np.mean(all_apfd)
        overall_std = np.std(all_apfd)
        ci = bootstrap_confidence_interval(all_apfd, n_bootstrap=1000)
    else:
        overall_mean = np.nan
        overall_std = np.nan
        ci = (np.nan, np.nan, np.nan)

    summary = {
        'method': f'Temporal {k}-Fold CV',
        'k': k,
        'total_builds': n_builds,
        'folds': results,
        'overall_mean_apfd': overall_mean,
        'overall_std_apfd': overall_std,
        'ci_lower': ci[1],
        'ci_upper': ci[2],
        'fold_means': [r['mean_apfd'] for r in results]
    }

    logger.info(f"\nOverall: APFD = {overall_mean:.4f} ± {overall_std:.4f} "
               f"[{ci[1]:.4f}, {ci[2]:.4f}]")

    return summary


def sliding_window_cv(
    test_df: pd.DataFrame,
    ordered_builds: List[str],
    train_window: int = 100,
    test_window: int = 20,
    step: int = 20
) -> Dict:
    """
    Sliding Window Cross-Validation.

    Uses a fixed-size training window that slides through time:
    - Train on builds [i, i+train_window]
    - Test on builds [i+train_window, i+train_window+test_window]
    - Slide by 'step' builds

    This simulates continuous model updates in production.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f" SLIDING WINDOW CROSS-VALIDATION")
    logger.info(f" Train window: {train_window}, Test window: {test_window}, Step: {step}")
    logger.info(f"{'='*70}")

    n_builds = len(ordered_builds)
    results = []

    position = 0
    window_idx = 0

    while position + train_window + test_window <= n_builds:
        # Define windows
        train_start = position
        train_end = position + train_window
        test_start = train_end
        test_end = min(test_start + test_window, n_builds)

        train_builds = ordered_builds[train_start:train_end]
        test_builds = ordered_builds[test_start:test_end]

        # Filter data
        train_df = test_df[test_df['Build_ID'].isin(train_builds)]
        test_window_df = test_df[test_df['Build_ID'].isin(test_builds)]

        # Compute failure rates
        failure_rates = load_failure_rates(train_df)

        # Evaluate
        mean_apfd, apfd_df = compute_apfd_for_builds(
            test_window_df, failure_rates, f"Window_{window_idx}"
        )

        n_test_with_failures = len(apfd_df)

        if n_test_with_failures > 0:
            results.append({
                'window': window_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_builds': len(train_builds),
                'test_builds': len(test_builds),
                'test_builds_with_failures': n_test_with_failures,
                'mean_apfd': mean_apfd,
                'std_apfd': apfd_df['apfd'].std(),
                'apfd_values': apfd_df['apfd'].values
            })

            logger.info(f"Window {window_idx}: Builds [{train_start}-{train_end}] → "
                       f"[{test_start}-{test_end}], "
                       f"APFD={mean_apfd:.4f} ({n_test_with_failures} builds)")

        position += step
        window_idx += 1

    # Aggregate
    if results:
        all_apfd = np.concatenate([r['apfd_values'] for r in results])
        overall_mean = np.mean(all_apfd)
        overall_std = np.std(all_apfd)
        ci = bootstrap_confidence_interval(all_apfd, n_bootstrap=1000)
    else:
        overall_mean = np.nan
        overall_std = np.nan
        ci = (np.nan, np.nan, np.nan)

    summary = {
        'method': 'Sliding Window CV',
        'train_window': train_window,
        'test_window': test_window,
        'step': step,
        'total_windows': len(results),
        'windows': results,
        'overall_mean_apfd': overall_mean,
        'overall_std_apfd': overall_std,
        'ci_lower': ci[1],
        'ci_upper': ci[2],
        'window_means': [r['mean_apfd'] for r in results]
    }

    logger.info(f"\nOverall: APFD = {overall_mean:.4f} ± {overall_std:.4f} "
               f"[{ci[1]:.4f}, {ci[2]:.4f}]")

    return summary


def concept_drift_analysis(
    test_df: pd.DataFrame,
    ordered_builds: List[str],
    build_dates: pd.Series,
    n_periods: int = 10
) -> Dict:
    """
    Concept Drift Analysis.

    Trains on first half of data, then evaluates on subsequent periods
    to detect performance degradation over time.

    Signs of concept drift:
    - Decreasing APFD over time
    - Increasing variance
    - Correlation between time and performance
    """
    logger.info(f"\n{'='*70}")
    logger.info(f" CONCEPT DRIFT ANALYSIS")
    logger.info(f"{'='*70}")

    n_builds = len(ordered_builds)

    # Train on first half
    train_size = n_builds // 2
    train_builds = ordered_builds[:train_size]
    test_builds = ordered_builds[train_size:]

    train_df = test_df[test_df['Build_ID'].isin(train_builds)]

    # Compute failure rates from training period
    failure_rates = load_failure_rates(train_df)

    logger.info(f"Training period: {train_size} builds")
    logger.info(f"Test period: {len(test_builds)} builds")

    # Split test period into n_periods
    period_size = len(test_builds) // n_periods

    results = []

    for period_idx in range(n_periods):
        start_idx = period_idx * period_size
        end_idx = start_idx + period_size if period_idx < n_periods - 1 else len(test_builds)

        period_builds = test_builds[start_idx:end_idx]
        period_df = test_df[test_df['Build_ID'].isin(period_builds)]

        # Get period dates
        period_dates = build_dates[build_dates.index.isin(period_builds)]
        period_start_date = period_dates.min()
        period_end_date = period_dates.max()

        # Evaluate
        mean_apfd, apfd_df = compute_apfd_for_builds(
            period_df, failure_rates, f"Period_{period_idx}"
        )

        n_with_failures = len(apfd_df)

        if n_with_failures > 0:
            results.append({
                'period': period_idx,
                'start_date': str(period_start_date),
                'end_date': str(period_end_date),
                'n_builds': len(period_builds),
                'n_builds_with_failures': n_with_failures,
                'mean_apfd': mean_apfd,
                'std_apfd': apfd_df['apfd'].std(),
                'apfd_values': apfd_df['apfd'].values
            })

            logger.info(f"Period {period_idx}: {period_start_date.strftime('%Y-%m') if pd.notna(period_start_date) else 'N/A'} - "
                       f"{period_end_date.strftime('%Y-%m') if pd.notna(period_end_date) else 'N/A'}, "
                       f"APFD={mean_apfd:.4f} ({n_with_failures} builds)")

    # Analyze drift
    if len(results) >= 3:
        period_indices = [r['period'] for r in results]
        period_apfds = [r['mean_apfd'] for r in results]

        # Correlation between period and APFD (negative = drift)
        correlation, p_value = stats.pearsonr(period_indices, period_apfds)

        # Linear regression slope
        slope, intercept, r_value, p_slope, std_err = stats.linregress(period_indices, period_apfds)

        # First half vs second half performance
        half = len(results) // 2
        first_half_apfd = np.mean([r['mean_apfd'] for r in results[:half]])
        second_half_apfd = np.mean([r['mean_apfd'] for r in results[half:]])
        performance_change = (second_half_apfd - first_half_apfd) / first_half_apfd * 100

        drift_analysis = {
            'correlation': correlation,
            'correlation_p_value': p_value,
            'slope': slope,
            'slope_p_value': p_slope,
            'r_squared': r_value ** 2,
            'first_half_apfd': first_half_apfd,
            'second_half_apfd': second_half_apfd,
            'performance_change_pct': performance_change,
            'drift_detected': p_value < 0.05 and correlation < -0.3
        }

        logger.info(f"\nDrift Analysis:")
        logger.info(f"  Correlation (time vs APFD): {correlation:.4f} (p={p_value:.4f})")
        logger.info(f"  Slope: {slope:.4f} per period")
        logger.info(f"  First half APFD: {first_half_apfd:.4f}")
        logger.info(f"  Second half APFD: {second_half_apfd:.4f}")
        logger.info(f"  Performance change: {performance_change:+.1f}%")

        if drift_analysis['drift_detected']:
            logger.warning("  ⚠️ CONCEPT DRIFT DETECTED")
        else:
            logger.info("  ✅ No significant concept drift detected")
    else:
        drift_analysis = None

    # Overall statistics
    if results:
        all_apfd = np.concatenate([r['apfd_values'] for r in results])
        overall_mean = np.mean(all_apfd)
        overall_std = np.std(all_apfd)
        ci = bootstrap_confidence_interval(all_apfd, n_bootstrap=1000)
    else:
        overall_mean = np.nan
        overall_std = np.nan
        ci = (np.nan, np.nan, np.nan)

    summary = {
        'method': 'Concept Drift Analysis',
        'train_builds': train_size,
        'test_builds': len(test_builds),
        'n_periods': n_periods,
        'periods': results,
        'drift_analysis': drift_analysis,
        'overall_mean_apfd': overall_mean,
        'overall_std_apfd': overall_std,
        'ci_lower': ci[1],
        'ci_upper': ci[2],
        'period_means': [r['mean_apfd'] for r in results]
    }

    return summary


def generate_temporal_cv_report(
    kfold_results: Dict,
    sliding_results: Dict,
    drift_results: Dict,
    output_dir: Path
):
    """Generate publication-ready report for temporal CV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_data = [
        {
            'Method': 'Temporal 5-Fold CV',
            'Mean APFD': kfold_results['overall_mean_apfd'],
            'Std': kfold_results['overall_std_apfd'],
            '95% CI Lower': kfold_results['ci_lower'],
            '95% CI Upper': kfold_results['ci_upper'],
            'N Evaluations': sum(f['test_builds_with_failures'] for f in kfold_results['folds'])
        },
        {
            'Method': 'Sliding Window CV',
            'Mean APFD': sliding_results['overall_mean_apfd'],
            'Std': sliding_results['overall_std_apfd'],
            '95% CI Lower': sliding_results['ci_lower'],
            '95% CI Upper': sliding_results['ci_upper'],
            'N Evaluations': sum(w['test_builds_with_failures'] for w in sliding_results['windows'])
        },
        {
            'Method': 'Concept Drift Test',
            'Mean APFD': drift_results['overall_mean_apfd'],
            'Std': drift_results['overall_std_apfd'],
            '95% CI Lower': drift_results['ci_lower'],
            '95% CI Upper': drift_results['ci_upper'],
            'N Evaluations': sum(p['n_builds_with_failures'] for p in drift_results['periods'])
        }
    ]

    summary_df = pd.DataFrame(summary_data)

    # Save CSV
    summary_df.to_csv(output_dir / 'temporal_cv_summary.csv', index=False)

    # Generate LaTeX table
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Temporal Cross-Validation Results for Filo-Priori v9}
\label{tab:temporal_cv}
\begin{tabular}{lccc}
\toprule
\textbf{Validation Method} & \textbf{Mean APFD} & \textbf{95\% CI} & \textbf{N Builds} \\
\midrule
"""

    for _, row in summary_df.iterrows():
        ci = f"[{row['95% CI Lower']:.3f}, {row['95% CI Upper']:.3f}]"
        latex += f"{row['Method']} & {row['Mean APFD']:.4f} & {ci} & {int(row['N Evaluations'])} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Temporal 5-Fold: Train on past builds, test on future builds.
\item Sliding Window: 100-build training window, 20-build test window.
\item Concept Drift: Train on first half, test across subsequent periods.
\end{tablenotes}
\end{table}
"""

    with open(output_dir / 'temporal_cv_table.tex', 'w') as f:
        f.write(latex)

    # Generate visualizations
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'figure.figsize': (12, 4),
        'figure.dpi': 150
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: K-Fold results
    ax1 = axes[0]
    fold_means = kfold_results['fold_means']
    ax1.bar(range(1, len(fold_means)+1), fold_means, color='#3498db', alpha=0.8, edgecolor='black')
    ax1.axhline(y=kfold_results['overall_mean_apfd'], color='red', linestyle='--', label='Mean')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Mean APFD')
    ax1.set_title('Temporal 5-Fold CV')
    ax1.set_ylim(0.4, 0.8)
    ax1.legend()

    # Plot 2: Sliding window results
    ax2 = axes[1]
    window_means = sliding_results['window_means']
    ax2.plot(range(len(window_means)), window_means, 'o-', color='#2ecc71', markersize=4)
    ax2.axhline(y=sliding_results['overall_mean_apfd'], color='red', linestyle='--', label='Mean')
    ax2.fill_between(range(len(window_means)),
                     [sliding_results['ci_lower']] * len(window_means),
                     [sliding_results['ci_upper']] * len(window_means),
                     alpha=0.2, color='red')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Mean APFD')
    ax2.set_title('Sliding Window CV')
    ax2.set_ylim(0.4, 0.8)
    ax2.legend()

    # Plot 3: Concept drift
    ax3 = axes[2]
    period_means = drift_results['period_means']
    ax3.plot(range(len(period_means)), period_means, 's-', color='#e74c3c', markersize=6)

    # Add trend line if drift analysis available
    if drift_results['drift_analysis']:
        x = np.array(range(len(period_means)))
        slope = drift_results['drift_analysis']['slope']
        intercept = period_means[0]  # Approximate
        trend = intercept + slope * x
        ax3.plot(x, trend, '--', color='gray', label=f'Trend (slope={slope:.4f})')

    ax3.axhline(y=drift_results['overall_mean_apfd'], color='red', linestyle='--', label='Mean')
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Mean APFD')
    ax3.set_title('Concept Drift Analysis')
    ax3.set_ylim(0.4, 0.8)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_cv_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'temporal_cv_results.pdf', bbox_inches='tight')
    plt.close()

    # Save detailed results as JSON
    all_results = {
        'kfold': {
            'method': kfold_results['method'],
            'overall_mean_apfd': kfold_results['overall_mean_apfd'],
            'overall_std_apfd': kfold_results['overall_std_apfd'],
            'ci_lower': kfold_results['ci_lower'],
            'ci_upper': kfold_results['ci_upper'],
            'fold_means': kfold_results['fold_means']
        },
        'sliding_window': {
            'method': sliding_results['method'],
            'overall_mean_apfd': sliding_results['overall_mean_apfd'],
            'overall_std_apfd': sliding_results['overall_std_apfd'],
            'ci_lower': sliding_results['ci_lower'],
            'ci_upper': sliding_results['ci_upper'],
            'window_means': sliding_results['window_means']
        },
        'concept_drift': {
            'method': drift_results['method'],
            'overall_mean_apfd': drift_results['overall_mean_apfd'],
            'overall_std_apfd': drift_results['overall_std_apfd'],
            'ci_lower': drift_results['ci_lower'],
            'ci_upper': drift_results['ci_upper'],
            'period_means': drift_results['period_means'],
            'drift_analysis': drift_results['drift_analysis']
        }
    }

    with open(output_dir / 'temporal_cv_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Run Temporal Cross-Validation')
    parser.add_argument('--output', type=str, default='results/temporal_cv',
                       help='Output directory')
    parser.add_argument('--k-folds', type=int, default=5,
                       help='Number of folds for temporal k-fold CV')
    parser.add_argument('--train-window', type=int, default=100,
                       help='Training window size for sliding window CV')
    parser.add_argument('--test-window', type=int, default=20,
                       help='Test window size for sliding window CV')
    parser.add_argument('--n-periods', type=int, default=10,
                       help='Number of periods for concept drift analysis')

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output

    logger.info("=" * 70)
    logger.info(" TEMPORAL CROSS-VALIDATION FOR FILO-PRIORI V9")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading data...")
    test_df, ordered_builds, build_dates = load_data()

    # Run temporal k-fold CV
    kfold_results = temporal_kfold_cv(test_df, ordered_builds, k=args.k_folds)

    # Run sliding window CV
    sliding_results = sliding_window_cv(
        test_df, ordered_builds,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.test_window
    )

    # Run concept drift analysis
    drift_results = concept_drift_analysis(
        test_df, ordered_builds, build_dates,
        n_periods=args.n_periods
    )

    # Generate report
    logger.info("\n" + "=" * 70)
    logger.info(" GENERATING REPORT")
    logger.info("=" * 70)

    summary_df = generate_temporal_cv_report(
        kfold_results, sliding_results, drift_results, output_dir
    )

    # Print final summary
    print("\n" + "=" * 80)
    print(" TEMPORAL CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Interpretation
    print("\n INTERPRETATION:")
    print("-" * 80)

    # Compare with single-split result (0.6397)
    single_split_apfd = 0.6397
    kfold_diff = kfold_results['overall_mean_apfd'] - single_split_apfd

    print(f"Single split APFD: {single_split_apfd:.4f}")
    print(f"Temporal K-Fold APFD: {kfold_results['overall_mean_apfd']:.4f} ({kfold_diff:+.4f})")

    if abs(kfold_diff) < 0.02:
        print("✅ Model performance is CONSISTENT across temporal validation")
    elif kfold_diff < -0.02:
        print("⚠️ Model performance DEGRADES in temporal validation")
    else:
        print("✅ Model performance IMPROVES in temporal validation")

    if drift_results['drift_analysis']:
        if drift_results['drift_analysis']['drift_detected']:
            print("⚠️ CONCEPT DRIFT detected: Model performance degrades over time")
        else:
            print("✅ NO concept drift: Model remains stable over time")

    print("=" * 80)


if __name__ == "__main__":
    main()
