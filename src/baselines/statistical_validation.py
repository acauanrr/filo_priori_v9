"""
Statistical Validation Module for Baseline Comparisons.

Implements rigorous statistical methods for comparing TCP methods:
1. Bootstrap Confidence Intervals (95% CI)
2. Paired t-test for significance
3. Wilcoxon signed-rank test (non-parametric)
4. Cohen's d effect size
5. Multiple comparison correction (Bonferroni, Holm)

This module ensures scientific rigor in comparing Filo-Priori with baselines.

Author: Filo-Priori Team
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    method1: str
    method2: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    significant: bool
    test_type: str


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Array of values
        statistic: 'mean' or 'median'
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n = len(data)

    # Point estimate
    if statistic == 'mean':
        point_estimate = np.mean(data)
        stat_func = np.mean
    elif statistic == 'median':
        point_estimate = np.median(data)
        stat_func = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return point_estimate, ci_lower, ci_upper


def paired_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test.

    Args:
        group1: First group (e.g., Filo-Priori APFD)
        group2: Second group (e.g., baseline APFD)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
    return float(t_stat), float(p_value)


def wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use when data is not normally distributed.

    Args:
        group1: First group
        group2: Second group
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Tuple of (statistic, p_value)
    """
    # Handle identical values
    diff = group1 - group2
    if np.all(diff == 0):
        return 0.0, 1.0

    try:
        stat, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
        return float(stat), float(p_value)
    except ValueError as e:
        # All differences are zero
        logger.warning(f"Wilcoxon test failed: {e}")
        return 0.0, 1.0


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size for paired samples.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: First group (e.g., Filo-Priori APFD)
        group2: Second group (e.g., baseline APFD)

    Returns:
        Tuple of (d_value, interpretation)
    """
    diff = group1 - group2
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(d), interpretation


def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float, bool]:
    """
    Test for normality using Shapiro-Wilk test.

    Args:
        data: Array of values

    Returns:
        Tuple of (statistic, p_value, is_normal)
        is_normal is True if p > 0.05
    """
    if len(data) < 3:
        return 0.0, 1.0, True

    # Shapiro-Wilk has a limit of 5000 samples
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    stat, p_value = stats.shapiro(data)
    is_normal = p_value > 0.05

    return float(stat), float(p_value), is_normal


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    n = len(p_values)
    adjusted_alpha = alpha / n

    results = []
    for p in p_values:
        adjusted_p = min(p * n, 1.0)
        is_significant = p < adjusted_alpha
        results.append((adjusted_p, is_significant))

    return results


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni correction (step-down procedure).

    More powerful than Bonferroni while controlling family-wise error rate.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        List of (adjusted_p, is_significant) tuples
    """
    n = len(p_values)

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    cumulative_max = 0

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted = p * (n - i)
        cumulative_max = max(cumulative_max, adjusted)
        adjusted_p[idx] = min(cumulative_max, 1.0)

    # Determine significance
    results = []
    for i, p in enumerate(p_values):
        is_significant = adjusted_p[i] < alpha
        results.append((adjusted_p[i], is_significant))

    return results


def compare_methods(
    method1_apfd: np.ndarray,
    method2_apfd: np.ndarray,
    method1_name: str,
    method2_name: str,
    alpha: float = 0.05,
    n_bootstrap: int = 1000
) -> StatisticalResult:
    """
    Comprehensive statistical comparison of two methods.

    Performs:
    1. Bootstrap CI for both methods
    2. Normality test
    3. Paired t-test OR Wilcoxon (based on normality)
    4. Cohen's d effect size

    Args:
        method1_apfd: APFD values for method 1
        method2_apfd: APFD values for method 2
        method1_name: Name of method 1
        method2_name: Name of method 2
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples

    Returns:
        StatisticalResult with all statistics
    """
    # Ensure same length
    assert len(method1_apfd) == len(method2_apfd), "Arrays must have same length"

    # Calculate differences
    diff = method1_apfd - method2_apfd

    # Bootstrap CI for difference
    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
        diff, statistic='mean', n_bootstrap=n_bootstrap
    )

    # Check normality
    _, p_normality, is_normal = shapiro_wilk_test(diff)

    # Choose appropriate test
    if is_normal:
        test_type = "paired_ttest"
        _, p_value = paired_ttest(method1_apfd, method2_apfd, alternative='greater')
    else:
        test_type = "wilcoxon"
        _, p_value = wilcoxon_test(method1_apfd, method2_apfd, alternative='greater')

    # Effect size
    effect_size, effect_interpretation = cohens_d(method1_apfd, method2_apfd)

    # Significance
    significant = p_value < alpha

    return StatisticalResult(
        method1=method1_name,
        method2=method2_name,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        effect_size=effect_size,
        effect_interpretation=effect_interpretation,
        significant=significant,
        test_type=test_type
    )


def generate_comparison_table(
    results: Dict[str, np.ndarray],
    reference_method: str = 'Filo-Priori',
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    apply_correction: str = 'holm'
) -> pd.DataFrame:
    """
    Generate comprehensive comparison table.

    Args:
        results: Dictionary mapping method names to APFD arrays
        reference_method: Method to compare others against
        alpha: Significance level
        n_bootstrap: Bootstrap samples
        apply_correction: 'bonferroni', 'holm', or None

    Returns:
        DataFrame with comparison results
    """
    if reference_method not in results:
        raise ValueError(f"Reference method '{reference_method}' not found in results")

    ref_apfd = results[reference_method]

    # Calculate statistics for each method
    rows = []
    p_values = []
    comparisons = []

    for method_name, method_apfd in results.items():
        # Bootstrap CI
        mean_apfd, ci_lower, ci_upper = bootstrap_confidence_interval(
            method_apfd, statistic='mean', n_bootstrap=n_bootstrap
        )

        row = {
            'Method': method_name,
            'Mean APFD': mean_apfd,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Std': np.std(method_apfd),
            'Median': np.median(method_apfd),
            'Min': np.min(method_apfd),
            'Max': np.max(method_apfd),
            'APFD=1.0 (%)': (method_apfd == 1.0).mean() * 100,
            'APFD>=0.7 (%)': (method_apfd >= 0.7).mean() * 100,
            'APFD<0.5 (%)': (method_apfd < 0.5).mean() * 100
        }

        # Comparison with reference
        if method_name != reference_method:
            comparison = compare_methods(
                ref_apfd, method_apfd,
                reference_method, method_name,
                alpha=alpha, n_bootstrap=n_bootstrap
            )

            row['Δ vs ' + reference_method] = comparison.mean_diff
            row['p-value'] = comparison.p_value
            row["Cohen's d"] = comparison.effect_size
            row['Effect'] = comparison.effect_interpretation
            row['Test'] = comparison.test_type

            p_values.append(comparison.p_value)
            comparisons.append(method_name)
        else:
            row['Δ vs ' + reference_method] = 0.0
            row['p-value'] = 1.0
            row["Cohen's d"] = 0.0
            row['Effect'] = '-'
            row['Test'] = '-'

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Apply multiple comparison correction
    if apply_correction and len(p_values) > 1:
        if apply_correction == 'bonferroni':
            corrected = bonferroni_correction(p_values, alpha)
        elif apply_correction == 'holm':
            corrected = holm_correction(p_values, alpha)
        else:
            corrected = [(p, p < alpha) for p in p_values]

        # Update DataFrame with corrected p-values
        for i, method_name in enumerate(comparisons):
            idx = df[df['Method'] == method_name].index[0]
            df.loc[idx, 'p-value (adj)'] = corrected[i][0]
            df.loc[idx, 'Significant'] = '***' if corrected[i][1] else ''

    # Sort by Mean APFD (descending)
    df = df.sort_values('Mean APFD', ascending=False).reset_index(drop=True)

    return df


def print_comparison_table(df: pd.DataFrame, title: str = "Method Comparison"):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)

    # Format for display
    display_cols = ['Method', 'Mean APFD', '95% CI Lower', '95% CI Upper',
                   'Δ vs Filo-Priori', 'p-value', "Cohen's d", 'Effect', 'Significant']

    available_cols = [c for c in display_cols if c in df.columns]

    # Format numeric columns
    formatters = {
        'Mean APFD': '{:.4f}'.format,
        '95% CI Lower': '{:.4f}'.format,
        '95% CI Upper': '{:.4f}'.format,
        'Δ vs Filo-Priori': '{:+.4f}'.format,
        'p-value': lambda x: f'{x:.2e}' if x < 0.001 else f'{x:.4f}',
        "Cohen's d": '{:.3f}'.format
    }

    print(df[available_cols].to_string(index=False, formatters=formatters))
    print("=" * 100)

    # Legend
    print("\nLegend:")
    print("  *** : Statistically significant after correction (p < 0.05)")
    print("  Effect sizes: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>=0.8)")


def generate_latex_table(
    df: pd.DataFrame,
    caption: str = "Comparison of TCP Methods",
    label: str = "tab:comparison"
) -> str:
    """
    Generate LaTeX table for paper.

    Args:
        df: Comparison DataFrame
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    # Select columns for paper
    cols = ['Method', 'Mean APFD', '95% CI Lower', '95% CI Upper',
            'p-value', "Cohen's d", 'Effect']

    available_cols = [c for c in cols if c in df.columns]
    df_subset = df[available_cols].copy()

    # Format CI as single column
    if '95% CI Lower' in df_subset.columns and '95% CI Upper' in df_subset.columns:
        df_subset['95% CI'] = df_subset.apply(
            lambda r: f"[{r['95% CI Lower']:.3f}, {r['95% CI Upper']:.3f}]", axis=1
        )
        df_subset = df_subset.drop(['95% CI Lower', '95% CI Upper'], axis=1)

    # Format p-value
    if 'p-value' in df_subset.columns:
        df_subset['p-value'] = df_subset['p-value'].apply(
            lambda x: f'{x:.2e}' if x < 0.001 else f'{x:.4f}'
        )

    # Format Mean APFD
    if 'Mean APFD' in df_subset.columns:
        df_subset['Mean APFD'] = df_subset['Mean APFD'].apply(lambda x: f'{x:.4f}')

    # Format Cohen's d
    if "Cohen's d" in df_subset.columns:
        df_subset["Cohen's d"] = df_subset["Cohen's d"].apply(lambda x: f'{x:.3f}')

    # Generate LaTeX
    latex = df_subset.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        column_format='l' + 'c' * (len(df_subset.columns) - 1)
    )

    return latex


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    n_builds = 100

    # Simulated APFD results
    results = {
        'Filo-Priori': np.random.beta(8, 3, n_builds),  # Higher performance
        'Random': np.random.uniform(0.4, 0.6, n_builds),  # Around 0.5
        'Recency': np.random.beta(5, 3, n_builds),  # Medium
        'FailureRate': np.random.beta(6, 3, n_builds),  # Medium-high
        'RandomForest': np.random.beta(7, 3, n_builds),  # Good
        'XGBoost': np.random.beta(7.5, 3, n_builds)  # Very good
    }

    print("=" * 60)
    print("STATISTICAL VALIDATION DEMO")
    print("=" * 60)

    # Generate comparison table
    comparison_df = generate_comparison_table(
        results,
        reference_method='Filo-Priori',
        alpha=0.05,
        n_bootstrap=1000,
        apply_correction='holm'
    )

    # Print table
    print_comparison_table(comparison_df)

    # Individual comparison example
    print("\n--- Detailed Comparison: Filo-Priori vs Random ---")
    result = compare_methods(
        results['Filo-Priori'],
        results['Random'],
        'Filo-Priori',
        'Random'
    )
    print(f"Mean difference: {result.mean_diff:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"p-value: {result.p_value:.4e}")
    print(f"Cohen's d: {result.effect_size:.3f} ({result.effect_interpretation})")
    print(f"Test used: {result.test_type}")
    print(f"Significant: {'Yes' if result.significant else 'No'}")

    # LaTeX table
    print("\n--- LaTeX Table ---")
    latex = generate_latex_table(comparison_df)
    print(latex[:500] + "...")
