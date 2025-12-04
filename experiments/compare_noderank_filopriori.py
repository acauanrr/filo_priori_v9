#!/usr/bin/env python3
"""
Comparacao Estatistica: NodeRank vs Filo-Priori

Este script compara os resultados do NodeRank com o Filo-Priori de forma
cientificamente rigorosa, incluindo:
- Testes estatisticos (Wilcoxon signed-rank)
- Effect size (Cohen's d, Cliff's delta)
- Intervalos de confianca
- Visualizacoes comparativas

Uso:
    python experiments/compare_noderank_filopriori.py

Prerequisito:
    Executar primeiro: python experiments/run_noderank_industry.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load APFD results from both methods."""
    noderank_path = PROJECT_ROOT / 'results/noderank_industry/apfd_per_build_FULL_testcsv.csv'
    # Use Filo-Priori V3 results (0.7595 APFD) - the optimized version
    filopriori_path = PROJECT_ROOT / 'results/experiment_industry_optimized_v3/apfd_per_build_FULL_testcsv.csv'

    if not noderank_path.exists():
        print(f"ERROR: NodeRank results not found at {noderank_path}")
        print("Run first: python experiments/run_noderank_industry.py")
        sys.exit(1)

    if not filopriori_path.exists():
        print(f"ERROR: Filo-Priori results not found at {filopriori_path}")
        sys.exit(1)

    noderank_df = pd.read_csv(noderank_path)
    filopriori_df = pd.read_csv(filopriori_path)

    return noderank_df, filopriori_df


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cliff's delta effect size.

    Returns:
        (delta, interpretation)
        where interpretation is: negligible, small, medium, or large
    """
    n1, n2 = len(x), len(y)

    # Count dominance
    more = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)

    delta = (more - less) / (n1 * n2)

    # Interpretation (Romano et al., 2006)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = 'negligible'
    elif abs_delta < 0.33:
        interpretation = 'small'
    elif abs_delta < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return delta, interpretation


def cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size.

    Returns:
        (d, interpretation)
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(x) - np.mean(y)) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return d, interpretation


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean."""
    np.random.seed(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha/2 * 100)
    upper = np.percentile(boot_means, (1 - alpha/2) * 100)

    return lower, upper


def compare_methods(noderank_df: pd.DataFrame, filopriori_df: pd.DataFrame) -> Dict:
    """Perform statistical comparison between methods."""
    # Merge on build_id to ensure paired comparison
    merged = noderank_df[['build_id', 'apfd']].merge(
        filopriori_df[['build_id', 'apfd']],
        on='build_id',
        suffixes=('_noderank', '_filopriori')
    )

    nr_apfd = merged['apfd_noderank'].values
    fp_apfd = merged['apfd_filopriori'].values

    results = {
        'n_builds_compared': len(merged),
        'noderank': {
            'mean': float(np.mean(nr_apfd)),
            'std': float(np.std(nr_apfd)),
            'median': float(np.median(nr_apfd)),
            'ci_95': bootstrap_ci(nr_apfd)
        },
        'filopriori': {
            'mean': float(np.mean(fp_apfd)),
            'std': float(np.std(fp_apfd)),
            'median': float(np.median(fp_apfd)),
            'ci_95': bootstrap_ci(fp_apfd)
        }
    }

    # Difference statistics
    diff = fp_apfd - nr_apfd
    results['difference'] = {
        'mean': float(np.mean(diff)),
        'std': float(np.std(diff)),
        'positive_pct': float(100 * np.mean(diff > 0)),  # % where Filo-Priori is better
        'negative_pct': float(100 * np.mean(diff < 0)),  # % where NodeRank is better
        'equal_pct': float(100 * np.mean(diff == 0))
    }

    # Wilcoxon signed-rank test (paired, non-parametric)
    stat, p_value = stats.wilcoxon(fp_apfd, nr_apfd, alternative='two-sided')
    results['wilcoxon'] = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01
    }

    # Effect sizes
    cliff_d, cliff_interp = cliffs_delta(fp_apfd, nr_apfd)
    cohen_d, cohen_interp = cohens_d(fp_apfd, nr_apfd)

    results['effect_size'] = {
        'cliffs_delta': float(cliff_d),
        'cliffs_interpretation': cliff_interp,
        'cohens_d': float(cohen_d),
        'cohens_interpretation': cohen_interp
    }

    return results


def print_comparison(results: Dict):
    """Print comparison results in a formatted way."""
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON: Filo-Priori vs NodeRank")
    print("="*80)

    print(f"\nBuilds compared: {results['n_builds_compared']}")

    print("\n" + "-"*40)
    print("APFD Summary Statistics")
    print("-"*40)

    nr = results['noderank']
    fp = results['filopriori']

    print(f"                    NodeRank        Filo-Priori")
    print(f"  Mean APFD:        {nr['mean']:.4f}          {fp['mean']:.4f}")
    print(f"  Std:              {nr['std']:.4f}          {fp['std']:.4f}")
    print(f"  Median:           {nr['median']:.4f}          {fp['median']:.4f}")
    print(f"  95% CI:           [{nr['ci_95'][0]:.4f}, {nr['ci_95'][1]:.4f}]  [{fp['ci_95'][0]:.4f}, {fp['ci_95'][1]:.4f}]")

    print("\n" + "-"*40)
    print("Difference Analysis (Filo-Priori - NodeRank)")
    print("-"*40)
    diff = results['difference']
    print(f"  Mean difference:     {diff['mean']:+.4f}")
    print(f"  Std of difference:   {diff['std']:.4f}")
    print(f"  Filo-Priori better:  {diff['positive_pct']:.1f}% of builds")
    print(f"  NodeRank better:     {diff['negative_pct']:.1f}% of builds")
    print(f"  Equal:               {diff['equal_pct']:.1f}% of builds")

    print("\n" + "-"*40)
    print("Statistical Tests")
    print("-"*40)
    wilc = results['wilcoxon']
    print(f"  Wilcoxon signed-rank test:")
    print(f"    Statistic:     {wilc['statistic']:.2f}")
    print(f"    p-value:       {wilc['p_value']:.6f}")
    print(f"    Significant at alpha=0.05: {'YES' if wilc['significant_0.05'] else 'NO'}")
    print(f"    Significant at alpha=0.01: {'YES' if wilc['significant_0.01'] else 'NO'}")

    print("\n" + "-"*40)
    print("Effect Size")
    print("-"*40)
    es = results['effect_size']
    print(f"  Cliff's delta:   {es['cliffs_delta']:+.4f} ({es['cliffs_interpretation']})")
    print(f"  Cohen's d:       {es['cohens_d']:+.4f} ({es['cohens_interpretation']})")

    print("\n" + "-"*40)
    print("Interpretation")
    print("-"*40)

    winner = "Filo-Priori" if diff['mean'] > 0 else "NodeRank"
    abs_diff = abs(diff['mean'])

    if wilc['significant_0.05']:
        print(f"  * {winner} is SIGNIFICANTLY better (p < 0.05)")
        print(f"  * Mean APFD improvement: {abs_diff:.4f} ({100*abs_diff/min(nr['mean'], fp['mean']):.1f}%)")
        print(f"  * Effect size: {es['cliffs_interpretation']}")
    else:
        print(f"  * No significant difference between methods (p = {wilc['p_value']:.4f})")
        print(f"  * Mean difference: {diff['mean']:+.4f}")

    print("="*80)


def save_comparison(results: Dict, noderank_df: pd.DataFrame, filopriori_df: pd.DataFrame):
    """Save comparison results."""
    import json

    output_dir = PROJECT_ROOT / 'results/noderank_industry'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'comparison_statistics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved statistics to: {json_path}")

    # Save merged comparison CSV
    merged = noderank_df[['build_id', 'count_tc', 'apfd']].merge(
        filopriori_df[['build_id', 'apfd']],
        on='build_id',
        suffixes=('_noderank', '_filopriori')
    )
    merged['difference'] = merged['apfd_filopriori'] - merged['apfd_noderank']
    merged['winner'] = merged['difference'].apply(
        lambda x: 'Filo-Priori' if x > 0 else ('NodeRank' if x < 0 else 'Tie')
    )

    csv_path = output_dir / 'per_build_comparison.csv'
    merged.to_csv(csv_path, index=False)
    print(f"Saved per-build comparison to: {csv_path}")


def main():
    """Main entry point."""
    print("\nLoading results...")
    noderank_df, filopriori_df = load_results()

    print(f"  NodeRank builds: {len(noderank_df)}")
    print(f"  Filo-Priori builds: {len(filopriori_df)}")

    print("\nPerforming statistical comparison...")
    results = compare_methods(noderank_df, filopriori_df)

    print_comparison(results)
    save_comparison(results, noderank_df, filopriori_df)


if __name__ == '__main__':
    main()
