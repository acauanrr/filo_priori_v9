#!/usr/bin/env python3
"""
NodeRank vs Filo-priori Comparison Experiment

This script runs a comprehensive comparison between NodeRank and Filo-priori
on the 01_industry dataset, producing scientifically comparable results.

Based on:
    Li, Y., et al. (2024). Test Input Prioritization for Graph Neural Networks.
    IEEE Transactions on Software Engineering, DOI: 10.1109/TSE.2024.3385538
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.noderank import (
    run_noderank_experiment,
    run_random_baseline,
    run_failure_rate_baseline
)

warnings.filterwarnings('ignore')


def load_industry_dataset():
    """Load the 01_industry dataset."""
    data_dir = PROJECT_ROOT / 'datasets' / '01_industry'

    print("Loading 01_industry dataset...")

    # Load train and test data
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset not found in {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine for full dataset
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples:  {len(test_df):,}")
    print(f"  Total samples: {len(full_df):,}")

    # Dataset statistics
    n_builds = full_df['Build_ID'].nunique()
    n_tests = full_df['TC_Key'].nunique()
    fail_rate = (full_df['TE_Test_Result'].str.upper() != 'PASS').mean()

    print(f"  Unique builds: {n_builds}")
    print(f"  Unique tests:  {n_tests}")
    print(f"  Failure rate:  {fail_rate:.4f} ({fail_rate*100:.2f}%)")

    return full_df


def load_filopriori_results():
    """Load existing Filo-priori results for comparison."""
    results_dir = PROJECT_ROOT / 'results' / 'experiment_industry_optimized_v3'
    apfd_path = results_dir / 'apfd_per_build.csv'

    if not apfd_path.exists():
        print("Warning: Filo-priori results not found")
        return None

    df = pd.read_csv(apfd_path)

    results = {
        'method': 'Filo-priori',
        'apfd_scores': df['apfd'].tolist(),
        'mean_apfd': df['apfd'].mean(),
        'std_apfd': df['apfd'].std(),
        'median_apfd': df['apfd'].median(),
        'min_apfd': df['apfd'].min(),
        'max_apfd': df['apfd'].max(),
        'n_builds': len(df),
        'build_ids': df['build_id'].tolist() if 'build_id' in df.columns else None
    }

    print(f"\nFilo-priori Results (from saved experiment):")
    print(f"  Mean APFD:   {results['mean_apfd']:.4f}")
    print(f"  Median APFD: {results['median_apfd']:.4f}")
    print(f"  Std APFD:    {results['std_apfd']:.4f}")
    print(f"  Builds:      {results['n_builds']}")

    return results


def statistical_comparison(results1: dict, results2: dict):
    """
    Perform statistical comparison between two methods.

    Uses:
    - Wilcoxon signed-rank test (non-parametric, paired)
    - Mann-Whitney U test (non-parametric, independent)
    - Effect size (Cliff's delta)
    """
    scores1 = np.array(results1['apfd_scores'])
    scores2 = np.array(results2['apfd_scores'])

    name1 = results1['method']
    name2 = results2['method']

    comparison = {
        'method1': name1,
        'method2': name2,
        'n_samples_1': len(scores1),
        'n_samples_2': len(scores2),
        'mean_diff': results1['mean_apfd'] - results2['mean_apfd'],
        'relative_improvement': (results1['mean_apfd'] - results2['mean_apfd']) / results2['mean_apfd'] * 100
    }

    # Mann-Whitney U test (for independent samples)
    try:
        stat_mw, p_mw = stats.mannwhitneyu(scores1, scores2, alternative='greater')
        comparison['mann_whitney_u'] = stat_mw
        comparison['mann_whitney_p'] = p_mw
        comparison['mann_whitney_significant'] = p_mw < 0.05
    except Exception as e:
        comparison['mann_whitney_error'] = str(e)

    # Wilcoxon test (if samples can be paired)
    if len(scores1) == len(scores2):
        try:
            stat_wx, p_wx = stats.wilcoxon(scores1, scores2, alternative='greater')
            comparison['wilcoxon_stat'] = stat_wx
            comparison['wilcoxon_p'] = p_wx
            comparison['wilcoxon_significant'] = p_wx < 0.05
        except Exception as e:
            comparison['wilcoxon_error'] = str(e)

    # Cliff's delta (effect size)
    def cliffs_delta(x, y):
        """Compute Cliff's delta effect size."""
        n1, n2 = len(x), len(y)
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return (more - less) / (n1 * n2)

    try:
        delta = cliffs_delta(scores1, scores2)
        comparison['cliffs_delta'] = delta

        # Interpret effect size
        if abs(delta) < 0.147:
            comparison['effect_size'] = 'negligible'
        elif abs(delta) < 0.33:
            comparison['effect_size'] = 'small'
        elif abs(delta) < 0.474:
            comparison['effect_size'] = 'medium'
        else:
            comparison['effect_size'] = 'large'
    except Exception as e:
        comparison['cliffs_delta_error'] = str(e)

    return comparison


def print_comparison_table(all_results: list):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    # Header
    print(f"\n{'Method':<25} {'Mean APFD':>12} {'Std':>10} {'Median':>10} {'Builds':>8}")
    print("-"*70)

    # Sort by mean APFD descending
    sorted_results = sorted(all_results, key=lambda x: x['mean_apfd'], reverse=True)

    for r in sorted_results:
        print(f"{r['method']:<25} {r['mean_apfd']:>12.4f} {r['std_apfd']:>10.4f} "
              f"{r['median_apfd']:>10.4f} {r['n_builds']:>8}")

    print("-"*70)

    # Best method
    best = sorted_results[0]
    print(f"\nBest Method: {best['method']} (Mean APFD = {best['mean_apfd']:.4f})")


def run_experiment():
    """Run the full comparison experiment."""
    print("="*80)
    print("NodeRank vs Filo-priori Comparison Experiment")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load dataset
    df = load_industry_dataset()

    # Create results directory
    results_dir = PROJECT_ROOT / 'results' / 'noderank_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Run NodeRank (Full Ensemble)
    print("\n" + "="*60)
    print("Running NodeRank (Full Ensemble)")
    print("="*60)
    noderank_full = run_noderank_experiment(
        df,
        build_col='Build_ID',
        test_col='TC_Key',
        result_col='TE_Test_Result',
        train_ratio=0.8,
        n_gsm=5,
        n_nfm=5,
        n_gmm=4,
        use_ensemble=True,
        seed=42
    )
    all_results.append(noderank_full)

    # 2. Run NodeRank (Single Model - Ablation)
    print("\n" + "="*60)
    print("Running NodeRank (Single Model - Ablation)")
    print("="*60)
    noderank_single = run_noderank_experiment(
        df,
        build_col='Build_ID',
        test_col='TC_Key',
        result_col='TE_Test_Result',
        train_ratio=0.8,
        use_ensemble=False,
        seed=42
    )
    noderank_single['method'] = 'NodeRank (Single)'
    all_results.append(noderank_single)

    # 3. Run Random Baseline
    print("\n" + "="*60)
    print("Running Random Baseline")
    print("="*60)
    random_results = run_random_baseline(
        df, 'Build_ID', 'TC_Key', 'TE_Test_Result', 0.8
    )
    print(f"Random: Mean APFD = {random_results['mean_apfd']:.4f}")
    all_results.append(random_results)

    # 4. Run Failure Rate Baseline
    print("\n" + "="*60)
    print("Running Failure Rate Baseline")
    print("="*60)
    failure_rate_results = run_failure_rate_baseline(
        df, 'Build_ID', 'TC_Key', 'TE_Test_Result', 0.8
    )
    print(f"Failure Rate: Mean APFD = {failure_rate_results['mean_apfd']:.4f}")
    all_results.append(failure_rate_results)

    # 5. Load Filo-priori Results
    print("\n" + "="*60)
    print("Loading Filo-priori Results")
    print("="*60)
    filopriori = load_filopriori_results()
    if filopriori:
        all_results.append(filopriori)

    # Print comparison table
    print_comparison_table(all_results)

    # Statistical comparisons
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    comparisons = []

    # Compare NodeRank vs each baseline
    for baseline in all_results:
        if baseline['method'] != 'NodeRank':
            comparison = statistical_comparison(noderank_full, baseline)
            comparisons.append(comparison)

            print(f"\nNodeRank vs {baseline['method']}:")
            print(f"  Mean difference: {comparison['mean_diff']:.4f}")
            print(f"  Relative improvement: {comparison['relative_improvement']:.2f}%")

            if 'mann_whitney_p' in comparison:
                sig = "Yes" if comparison['mann_whitney_significant'] else "No"
                print(f"  Mann-Whitney U p-value: {comparison['mann_whitney_p']:.6f} (Significant: {sig})")

            if 'cliffs_delta' in comparison:
                print(f"  Cliff's delta: {comparison['cliffs_delta']:.4f} ({comparison['effect_size']})")

    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    # Save APFD per build for NodeRank
    noderank_build_df = pd.DataFrame(noderank_full.get('build_results', []))
    if len(noderank_build_df) > 0:
        noderank_build_df.to_csv(results_dir / 'noderank_apfd_per_build.csv', index=False)
        print(f"  Saved: {results_dir / 'noderank_apfd_per_build.csv'}")

    # Save summary
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'dataset': '01_industry',
        'results': [
            {
                'method': r['method'],
                'mean_apfd': r['mean_apfd'],
                'std_apfd': r['std_apfd'],
                'median_apfd': r.get('median_apfd', np.median(r['apfd_scores']) if r['apfd_scores'] else 0),
                'n_builds': r['n_builds']
            }
            for r in all_results
        ],
        'statistical_comparisons': comparisons,
        'config': noderank_full.get('config', {})
    }

    with open(results_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {results_dir / 'comparison_summary.json'}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'method': r['method'],
            'mean_apfd': r['mean_apfd'],
            'std_apfd': r['std_apfd'],
            'median_apfd': r.get('median_apfd', np.median(r['apfd_scores']) if r['apfd_scores'] else 0),
            'min_apfd': r.get('min_apfd', np.min(r['apfd_scores']) if r['apfd_scores'] else 0),
            'max_apfd': r.get('max_apfd', np.max(r['apfd_scores']) if r['apfd_scores'] else 0),
            'n_builds': r['n_builds']
        }
        for r in all_results
    ])
    comparison_df = comparison_df.sort_values('mean_apfd', ascending=False)
    comparison_df.to_csv(results_dir / 'method_comparison.csv', index=False)
    print(f"  Saved: {results_dir / 'method_comparison.csv'}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)

    return all_results, comparisons


if __name__ == '__main__':
    run_experiment()
