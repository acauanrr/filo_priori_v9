"""
Compare V11 (DeepOrder-Enhanced Filo-Priori) with DeepOrder baseline.

This script:
1. Loads V11 experiment results
2. Loads DeepOrder baseline results
3. Performs statistical comparison (Wilcoxon signed-rank test)
4. Generates comparison report

Target: APFD ≥ 0.6550 (beat DeepOrder 0.6500 by significant margin)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_apfd_values(results_dir: str) -> dict:
    """Load APFD values from experiment results."""
    apfd_file = os.path.join(results_dir, "apfd_per_build_FULL_testcsv.csv")

    if os.path.exists(apfd_file):
        df = pd.read_csv(apfd_file)
        if 'apfd' in df.columns:
            return {
                'values': df['apfd'].tolist(),
                'mean': df['apfd'].mean(),
                'std': df['apfd'].std(),
                'median': df['apfd'].median(),
                'builds': len(df)
            }

    # Try loading from test_metrics.json
    metrics_file = os.path.join(results_dir, "test_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            if 'apfd' in metrics:
                return {'mean': metrics['apfd'], 'values': None}

    return None


def bootstrap_ci(values: list, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
    """Calculate bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def compare_methods(v11_results: dict, deeporder_results: dict) -> dict:
    """Perform statistical comparison between V11 and DeepOrder."""
    comparison = {
        'v11_mean': v11_results['mean'],
        'deeporder_mean': deeporder_results['mean'],
        'difference': v11_results['mean'] - deeporder_results['mean'],
        'relative_improvement': (v11_results['mean'] - deeporder_results['mean']) / deeporder_results['mean'] * 100
    }

    # If we have per-build values, do Wilcoxon test
    if v11_results.get('values') and deeporder_results.get('values'):
        v11_vals = np.array(v11_results['values'])
        do_vals = np.array(deeporder_results['values'])

        # Align by build count
        min_len = min(len(v11_vals), len(do_vals))
        v11_vals = v11_vals[:min_len]
        do_vals = do_vals[:min_len]

        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(v11_vals, do_vals)
        comparison['wilcoxon_statistic'] = stat
        comparison['p_value'] = p_value
        comparison['statistically_significant'] = p_value < 0.05

        # Effect size (r = Z / sqrt(N))
        z = stats.norm.ppf(1 - p_value / 2)  # Two-tailed
        r = abs(z) / np.sqrt(min_len)
        comparison['effect_size_r'] = r

        # Bootstrap CI for V11
        lower, upper = bootstrap_ci(v11_vals)
        comparison['v11_ci'] = [lower, upper]

    return comparison


def main():
    print("=" * 70)
    print("V11 vs DeepOrder Comparison")
    print("=" * 70)

    # Paths
    v11_dir = "results/experiment_industry_v11"
    deeporder_dir = "results/baseline_comparison"

    # Load V11 results
    print("\n1. Loading V11 results...")
    v11_results = load_apfd_values(v11_dir)

    if v11_results is None:
        print(f"   ERROR: V11 results not found in {v11_dir}")
        print("   Run experiment first with: python main.py --config configs/experiment_industry_v11.yaml")
        return

    print(f"   V11 APFD: {v11_results['mean']:.4f}")
    if v11_results.get('std'):
        print(f"   V11 Std:  {v11_results['std']:.4f}")
    if v11_results.get('builds'):
        print(f"   Builds:   {v11_results['builds']}")

    # Load DeepOrder baseline (known value)
    deeporder_results = {
        'mean': 0.6500,
        'values': None  # We don't have per-build values for DeepOrder
    }

    # Try to load from baseline comparison
    deeporder_file = os.path.join(deeporder_dir, "deeporder_apfd.json")
    if os.path.exists(deeporder_file):
        with open(deeporder_file, 'r') as f:
            data = json.load(f)
            if 'per_build' in data:
                deeporder_results['values'] = data['per_build']

    print(f"\n2. DeepOrder APFD: {deeporder_results['mean']:.4f} (baseline)")

    # Compare
    print("\n3. Comparison:")
    print("-" * 40)

    diff = v11_results['mean'] - deeporder_results['mean']
    rel_diff = diff / deeporder_results['mean'] * 100

    print(f"   V11 APFD:       {v11_results['mean']:.4f}")
    print(f"   DeepOrder APFD: {deeporder_results['mean']:.4f}")
    print(f"   Difference:     {diff:+.4f} ({rel_diff:+.2f}%)")

    if diff > 0:
        print(f"\n   ✅ V11 OUTPERFORMS DeepOrder by {diff:.4f} ({rel_diff:.2f}%)")

        # Check if we met the target
        if v11_results['mean'] >= 0.6550:
            print(f"   ✅ TARGET MET: APFD ≥ 0.6550")
        else:
            gap = 0.6550 - v11_results['mean']
            print(f"   ⚠️  Target 0.6550 not yet met (gap: {gap:.4f})")
    else:
        print(f"\n   ⚠️  V11 still behind DeepOrder")
        print(f"   Gap to close: {-diff:.4f}")

    # Success criteria summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_criteria = [
        ("APFD > DeepOrder (0.6500)", v11_results['mean'] > 0.6500),
        ("APFD ≥ 0.6550 (target)", v11_results['mean'] >= 0.6550),
        ("Improvement > 1%", rel_diff > 1.0)
    ]

    for criterion, passed in success_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
