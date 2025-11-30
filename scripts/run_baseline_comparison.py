#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison for TCP

This script runs all relevant baselines on the 01_industry dataset
and compares them with Filo-Priori results.

Baselines:
    1. Random - Random ordering
    2. FailureRate - Historical failure rate heuristic
    3. RETECS - Reinforcement Learning (Spieker et al., ISSTA 2017)
    4. DeepOrder - Deep Neural Network (Chen et al., ICSME 2021)
    5. Filo-Priori - Our approach

Usage:
    python scripts/run_baseline_comparison.py

Output:
    results/baseline_comparison/
    ├── comparison_results.json
    ├── comparison_table.csv
    └── comparison_table.tex
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.retecs import run_retecs_experiment
from src.baselines.deeporder import run_deeporder_experiment


def compute_apfd(ranking: List[str], verdicts: Dict[str, int]) -> float:
    """Compute APFD for a ranking."""
    n_tests = len(ranking)
    n_faults = sum(verdicts.values())

    if n_faults == 0:
        return 0.5  # No faults, APFD undefined, return 0.5

    fault_positions = []
    for i, test_id in enumerate(ranking):
        if verdicts.get(test_id, 0) == 1:
            fault_positions.append(i + 1)

    apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
    return apfd


def run_random_baseline(df: pd.DataFrame, build_col: str, test_col: str, result_col: str) -> Dict:
    """Run random baseline."""
    import random

    builds = df[build_col].unique().tolist()
    train_idx = int(len(builds) * 0.8)
    test_builds = builds[train_idx:]

    apfd_scores = []

    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts
        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        if sum(verdicts.values()) == 0:
            continue  # Skip builds without failures

        # Random ranking
        ranking = test_ids.copy()
        random.shuffle(ranking)

        apfd = compute_apfd(ranking, verdicts)
        apfd_scores.append(apfd)

    return {
        'method': 'Random',
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores),
        'std_apfd': np.std(apfd_scores),
        'n_builds': len(apfd_scores)
    }


def run_failure_rate_baseline(df: pd.DataFrame, build_col: str, test_col: str, result_col: str) -> Dict:
    """Run failure rate heuristic baseline."""
    builds = df[build_col].unique().tolist()
    train_idx = int(len(builds) * 0.8)
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    # Compute failure rates from training data
    train_df = df[df[build_col].isin(train_builds)]
    failure_rates = {}

    for test_id in train_df[test_col].unique():
        test_df = train_df[train_df[test_col] == test_id]
        failures = (test_df[result_col].apply(lambda x: str(x).upper() != 'PASS')).sum()
        total = len(test_df)
        failure_rates[test_id] = failures / total if total > 0 else 0

    apfd_scores = []

    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts
        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        if sum(verdicts.values()) == 0:
            continue  # Skip builds without failures

        # Rank by failure rate (descending)
        ranking = sorted(test_ids, key=lambda t: failure_rates.get(t, 0), reverse=True)

        apfd = compute_apfd(ranking, verdicts)
        apfd_scores.append(apfd)

    return {
        'method': 'FailureRate',
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores),
        'std_apfd': np.std(apfd_scores),
        'n_builds': len(apfd_scores)
    }


def load_filo_priori_results(results_dir: Path) -> Dict:
    """Load Filo-Priori results from existing experiment."""
    apfd_file = results_dir / 'apfd_per_build_FULL_testcsv.csv'

    if not apfd_file.exists():
        print(f"Warning: Filo-Priori results not found at {apfd_file}")
        return None

    df = pd.read_csv(apfd_file)
    apfd_scores = df['apfd'].tolist()

    return {
        'method': 'Filo-Priori',
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores),
        'std_apfd': np.std(apfd_scores),
        'n_builds': len(apfd_scores)
    }


def compute_statistics(results: List[Dict]) -> pd.DataFrame:
    """Compute comparison statistics."""
    data = []

    # Find Filo-Priori for comparison
    filo_priori = next((r for r in results if r['method'] == 'Filo-Priori'), None)

    for result in results:
        row = {
            'Method': result['method'],
            'Mean APFD': result['mean_apfd'],
            'Std APFD': result['std_apfd'],
            'N Builds': result['n_builds']
        }

        # Compute 95% CI
        if result['n_builds'] > 1:
            ci = stats.t.interval(
                0.95,
                result['n_builds'] - 1,
                loc=result['mean_apfd'],
                scale=result['std_apfd'] / np.sqrt(result['n_builds'])
            )
            row['95% CI Lower'] = ci[0]
            row['95% CI Upper'] = ci[1]
        else:
            row['95% CI Lower'] = result['mean_apfd']
            row['95% CI Upper'] = result['mean_apfd']

        # Compute improvement vs Random
        random_result = next((r for r in results if r['method'] == 'Random'), None)
        if random_result:
            improvement = (result['mean_apfd'] - random_result['mean_apfd']) / random_result['mean_apfd'] * 100
            row['vs Random (%)'] = improvement

        # Compute p-value vs Filo-Priori
        if filo_priori and result['method'] != 'Filo-Priori':
            if len(result['apfd_scores']) > 0 and len(filo_priori['apfd_scores']) > 0:
                # Use Wilcoxon signed-rank test for paired comparison
                try:
                    min_len = min(len(result['apfd_scores']), len(filo_priori['apfd_scores']))
                    _, p_value = stats.wilcoxon(
                        result['apfd_scores'][:min_len],
                        filo_priori['apfd_scores'][:min_len]
                    )
                    row['p-value vs Filo-Priori'] = p_value
                except:
                    row['p-value vs Filo-Priori'] = 1.0
            else:
                row['p-value vs Filo-Priori'] = 1.0

        data.append(row)

    # Sort by Mean APFD (descending)
    df = pd.DataFrame(data)
    df = df.sort_values('Mean APFD', ascending=False)

    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for the paper."""
    latex = r"""
\begin{table}[!t]
\centering
\caption{Comparison with State-of-the-Art TCP Methods}
\label{tab:baseline_comparison}
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{APFD} & \textbf{95\% CI} & \textbf{p-value} & \textbf{$\Delta$ vs Random} \\
\midrule
"""

    for _, row in df.iterrows():
        method = row['Method']
        apfd = row['Mean APFD']
        ci_lower = row.get('95% CI Lower', apfd)
        ci_upper = row.get('95% CI Upper', apfd)
        p_value = row.get('p-value vs Filo-Priori', '--')
        delta = row.get('vs Random (%)', 0)

        # Format p-value
        if isinstance(p_value, float):
            if p_value < 0.001:
                p_str = '$<$.001***'
            elif p_value < 0.01:
                p_str = f'{p_value:.3f}**'
            elif p_value < 0.05:
                p_str = f'{p_value:.3f}*'
            else:
                p_str = f'{p_value:.3f}'
        else:
            p_str = '--'

        # Bold Filo-Priori
        if method == 'Filo-Priori':
            method = r'\textbf{Filo-Priori}'
            apfd_str = f'\\textbf{{{apfd:.4f}}}'
        else:
            apfd_str = f'{apfd:.4f}'

        ci_str = f'[{ci_lower:.3f}, {ci_upper:.3f}]'
        delta_str = f'+{delta:.1f}\\%' if delta > 0 else f'{delta:.1f}\\%'

        latex += f'{method} & {apfd_str} & {ci_str} & {p_str} & {delta_str} \\\\\n'

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item * p $<$ 0.05, ** p $<$ 0.01, *** p $<$ 0.001 (Wilcoxon signed-rank test vs Filo-Priori)
\end{tablenotes}
\end{table}
"""
    return latex


def main():
    """Run comprehensive baseline comparison."""
    print("=" * 70)
    print("TCP Baseline Comparison Experiment")
    print("=" * 70)
    print()

    # Setup paths
    data_dir = PROJECT_ROOT / 'datasets' / '01_industry'
    results_dir = PROJECT_ROOT / 'results' / 'baseline_comparison'
    filo_priori_results = PROJECT_ROOT / 'results' / 'experiment_hybrid_phylogenetic'

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_file = data_dir / 'train.csv'
    test_file = data_dir / 'test.csv'

    if not train_file.exists() or not test_file.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Please ensure train.csv and test.csv exist in the 01_industry directory.")
        return

    print("Loading dataset...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Loaded {len(df)} test executions")
    print(f"Unique builds: {df['Build_ID'].nunique()}")
    print(f"Unique test cases: {df['TC_Key'].nunique()}")
    print()

    # Run baselines
    results = []

    # 1. Random baseline
    print("-" * 40)
    print("Running Random baseline...")
    random_result = run_random_baseline(df, 'Build_ID', 'TC_Key', 'TE_Test_Result')
    results.append(random_result)
    print(f"Random: APFD = {random_result['mean_apfd']:.4f} (+/- {random_result['std_apfd']:.4f})")
    print()

    # 2. Failure Rate baseline
    print("-" * 40)
    print("Running FailureRate baseline...")
    fr_result = run_failure_rate_baseline(df, 'Build_ID', 'TC_Key', 'TE_Test_Result')
    results.append(fr_result)
    print(f"FailureRate: APFD = {fr_result['mean_apfd']:.4f} (+/- {fr_result['std_apfd']:.4f})")
    print()

    # 3. RETECS baseline
    print("-" * 40)
    print("Running RETECS baseline...")
    retecs_result = run_retecs_experiment(
        df,
        build_col='Build_ID',
        test_col='TC_Key',
        result_col='TE_Test_Result',
        reward_type='tcfail',
        n_episodes=3
    )
    results.append(retecs_result)
    print()

    # 4. DeepOrder baseline
    print("-" * 40)
    print("Running DeepOrder baseline...")
    deeporder_result = run_deeporder_experiment(
        df,
        build_col='Build_ID',
        test_col='TC_Key',
        result_col='TE_Test_Result',
        epochs=30
    )
    results.append(deeporder_result)
    print()

    # 5. Load Filo-Priori results
    print("-" * 40)
    print("Loading Filo-Priori results...")
    filo_result = load_filo_priori_results(filo_priori_results)
    if filo_result:
        results.append(filo_result)
        print(f"Filo-Priori: APFD = {filo_result['mean_apfd']:.4f} (+/- {filo_result['std_apfd']:.4f})")
    else:
        print("Warning: Could not load Filo-Priori results")
    print()

    # Compute statistics
    print("=" * 70)
    print("Computing comparison statistics...")
    print("=" * 70)

    comparison_df = compute_statistics(results)
    print("\nResults Summary:")
    print(comparison_df.to_string(index=False))

    # Save results
    print("\nSaving results...")

    # JSON
    results_json = {r['method']: {
        'mean_apfd': r['mean_apfd'],
        'std_apfd': r['std_apfd'],
        'n_builds': r['n_builds'],
        'apfd_scores': r['apfd_scores']
    } for r in results}

    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # CSV
    comparison_df.to_csv(results_dir / 'comparison_table.csv', index=False)

    # LaTeX
    latex_table = generate_latex_table(comparison_df)
    with open(results_dir / 'comparison_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"\nResults saved to {results_dir}")
    print("  - comparison_results.json")
    print("  - comparison_table.csv")
    print("  - comparison_table.tex")

    print("\n" + "=" * 70)
    print("Baseline Comparison Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
