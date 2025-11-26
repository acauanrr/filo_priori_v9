#!/usr/bin/env python3
"""
Generate publication-quality figures for the Filo-Priori V9 paper.

Generates:
- fig_rq1_apfd_comparison: Box plot comparing all TCP methods
- fig_rq1_improvement: Bar chart showing improvement over random
- fig_rq2_ablation: Ablation study component contributions
- fig_rq3_temporal: Temporal cross-validation results
- fig_rq4_sensitivity: Hyperparameter sensitivity analysis
- fig_qualitative: Qualitative case study analysis

Author: Filo-Priori V9
Date: 2025-11-26
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

sns.set_style("whitegrid")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PAPER_DIR = BASE_DIR / "paper" / "figures"


def load_baseline_apfd():
    """Load APFD values for all baselines."""
    with open(RESULTS_DIR / "baselines" / "all_apfd_results.json", "r") as f:
        return json.load(f)


def load_experiment_07_apfd():
    """Load APFD values from experiment 07 (ranking optimized)."""
    df = pd.read_csv(RESULTS_DIR / "experiment_07_ranking_optimized" / "apfd_per_build_FULL_testcsv.csv")
    return df['apfd'].tolist()


def generate_fig_rq1_comparison():
    """Generate RQ1: APFD comparison box plot."""
    print("\n" + "="*60)
    print("Generating fig_rq1_apfd_comparison")
    print("="*60)

    # Load data
    baseline_data = load_baseline_apfd()
    exp07_apfd = load_experiment_07_apfd()

    # Order methods by mean APFD
    methods_data = {
        'Filo-Priori\n(Ours)': exp07_apfd,
        'FailureRate': baseline_data['FailureRate'],
        'XGBoost': baseline_data['XGBoost'],
        'GreedyHist.': baseline_data['GreedyHistorical'],
        'LogisticReg.': baseline_data['LogisticRegression'],
        'RandomForest': baseline_data['RandomForest'],
        'Random': baseline_data['Random'],
        'RecentFail.': baseline_data['RecentFailureRate'],
        'Recency': baseline_data['Recency'],
    }

    # Calculate means for ordering
    means = {k: np.mean(v) for k, v in methods_data.items()}
    sorted_methods = sorted(means.keys(), key=lambda x: means[x], reverse=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for boxplot
    data_list = [methods_data[m] for m in sorted_methods]

    # Colors: highlight our method
    colors = ['#2ecc71' if 'Ours' in m else '#3498db' for m in sorted_methods]

    # Create box plot
    bp = ax.boxplot(data_list, patch_artist=True, labels=sorted_methods)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    for i, m in enumerate(sorted_methods):
        ax.scatter(i + 1, means[m], color='red', marker='D', s=50, zorder=5)

    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax.axhline(y=means['Filo-Priori\n(Ours)'], color='#2ecc71', linestyle=':', alpha=0.7)

    ax.set_ylabel('APFD Score', fontweight='bold')
    ax.set_xlabel('TCP Method', fontweight='bold')
    ax.set_title('RQ1: Comparison of TCP Methods on QTA Dataset\n(277 builds with failures)', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Filo-Priori (proposed)')
    blue_patch = mpatches.Patch(color='#3498db', alpha=0.7, label='Baselines')
    red_marker = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8, label='Mean')
    ax.legend(handles=[green_patch, blue_patch, red_marker], loc='lower left')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_rq1_apfd_comparison.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def generate_fig_rq1_improvement():
    """Generate RQ1: Improvement over random bar chart."""
    print("\n" + "="*60)
    print("Generating fig_rq1_improvement")
    print("="*60)

    # Data
    baseline_data = load_baseline_apfd()
    exp07_apfd = load_experiment_07_apfd()

    random_mean = np.mean(baseline_data['Random'])

    methods = {
        'Filo-Priori': np.mean(exp07_apfd),
        'FailureRate': np.mean(baseline_data['FailureRate']),
        'XGBoost': np.mean(baseline_data['XGBoost']),
        'GreedyHist.': np.mean(baseline_data['GreedyHistorical']),
        'LogisticReg.': np.mean(baseline_data['LogisticRegression']),
        'RandomForest': np.mean(baseline_data['RandomForest']),
        'RecentFail.': np.mean(baseline_data['RecentFailureRate']),
        'Recency': np.mean(baseline_data['Recency']),
    }

    # Calculate improvement over random
    improvements = {k: ((v - random_mean) / random_mean) * 100 for k, v in methods.items()}

    # Sort by improvement
    sorted_methods = sorted(improvements.keys(), key=lambda x: improvements[x], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sorted_methods))
    values = [improvements[m] for m in sorted_methods]
    colors = ['#2ecc71' if m == 'Filo-Priori' else '#3498db' if v > 0 else '#e74c3c' for m, v in zip(sorted_methods, values)]

    bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods, rotation=45, ha='right')
    ax.set_ylabel('Improvement over Random (%)', fontweight='bold')
    ax.set_xlabel('TCP Method', fontweight='bold')
    ax.set_title('RQ1: Relative Improvement over Random Ordering', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_rq1_improvement.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def generate_fig_rq2_ablation():
    """Generate RQ2: Ablation study figure."""
    print("\n" + "="*60)
    print("Generating fig_rq2_ablation")
    print("="*60)

    # Ablation data from results
    ablation_data = {
        'Full Model': {'apfd': 0.6397, 'contribution': 0},
        'w/o GATv2': {'apfd': 0.5311, 'contribution': 17.0},
        'w/o Structural': {'apfd': 0.6060, 'contribution': 5.3},
        'w/o Class Weights': {'apfd': 0.6100, 'contribution': 4.6},
        'w/o Ensemble': {'apfd': 0.6171, 'contribution': 3.5},
        'w/o Semantic': {'apfd': 0.6276, 'contribution': 1.9},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: APFD values
    ax = axes[0]
    components = list(ablation_data.keys())
    apfd_values = [ablation_data[c]['apfd'] for c in components]
    colors = ['#2ecc71' if c == 'Full Model' else '#e74c3c' for c in components]

    bars = ax.barh(components, apfd_values, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Mean APFD', fontweight='bold')
    ax.set_title('Ablation Study: APFD by Configuration', fontweight='bold')
    ax.set_xlim(0.5, 0.7)
    ax.axvline(x=0.6397, color='#2ecc71', linestyle='--', alpha=0.7, label='Full Model')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    for bar, val in zip(bars, apfd_values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    # Right: Contribution percentages
    ax = axes[1]
    components_contrib = [c for c in components if c != 'Full Model']
    contributions = [ablation_data[c]['contribution'] for c in components_contrib]

    # Sort by contribution
    sorted_idx = np.argsort(contributions)[::-1]
    components_sorted = [components_contrib[i] for i in sorted_idx]
    contributions_sorted = [contributions[i] for i in sorted_idx]

    # Color gradient based on contribution
    cmap = plt.cm.Reds
    norm_vals = [c / max(contributions) for c in contributions_sorted]
    colors = [cmap(0.3 + 0.6 * v) for v in norm_vals]

    bars = ax.barh(components_sorted, contributions_sorted, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Contribution to Performance (%)', fontweight='bold')
    ax.set_title('Component Contributions (when removed)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, contributions_sorted):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'+{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_rq2_ablation.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def generate_fig_rq3_temporal():
    """Generate RQ3: Temporal cross-validation figure."""
    print("\n" + "="*60)
    print("Generating fig_rq3_temporal")
    print("="*60)

    # Temporal CV data
    temporal_data = {
        'Temporal 5-Fold CV': {'mean': 0.6629, 'ci_low': 0.627, 'ci_high': 0.698, 'n': 215},
        'Sliding Window CV': {'mean': 0.6279, 'ci_low': 0.595, 'ci_high': 0.661, 'n': 248},
        'Concept Drift Test': {'mean': 0.6187, 'ci_low': 0.574, 'ci_high': 0.661, 'n': 152},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(temporal_data.keys())
    means = [temporal_data[m]['mean'] for m in methods]
    ci_lows = [temporal_data[m]['ci_low'] for m in methods]
    ci_highs = [temporal_data[m]['ci_high'] for m in methods]

    x = np.arange(len(methods))
    colors = ['#3498db', '#2ecc71', '#e67e22']

    # Bar with error bars
    bars = ax.bar(x, means, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    # Error bars (CI)
    yerr_low = [m - l for m, l in zip(means, ci_lows)]
    yerr_high = [h - m for m, h in zip(means, ci_highs)]
    ax.errorbar(x, means, yerr=[yerr_low, yerr_high], fmt='none', color='black', capsize=8, capthick=2, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean APFD', fontweight='bold')
    ax.set_xlabel('Validation Method', fontweight='bold')
    ax.set_title('RQ3: Temporal Robustness Across Validation Strategies', fontweight='bold')
    ax.set_ylim(0.5, 0.75)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, m, l, h in zip(bars, means, ci_lows, ci_highs):
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{m:.3f}\n[{l:.3f}, {h:.3f}]', ha='center', va='bottom', fontsize=9)

    # Add annotation
    ax.annotate('Stable performance\nacross all methods\n(range: 0.619-0.663)',
                xy=(1, 0.55), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_rq3_temporal.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def generate_fig_rq4_sensitivity():
    """Generate RQ4: Hyperparameter sensitivity figure."""
    print("\n" + "="*60)
    print("Generating fig_rq4_sensitivity")
    print("="*60)

    # Sensitivity data
    sensitivity_data = {
        'Loss Function': {
            'options': ['Weighted CE', 'Focal', 'CE', 'Ranking'],
            'values': [0.6191, 0.6120, 0.5989, 0.5830],
            'best': 'Weighted CE',
            'delta': 0.036
        },
        'Learning Rate': {
            'options': ['3e-5', '5e-5', '1e-4', '1e-5'],
            'values': [0.6191, 0.5924, 0.5812, 0.5756],
            'best': '3e-5',
            'delta': 0.027
        },
        'GNN Layers': {
            'options': ['1 layer', '2 layers', '3 layers'],
            'values': [0.6191, 0.6012, 0.5920],
            'best': '1 layer',
            'delta': 0.027
        },
        'Features': {
            'options': ['10 (selected)', '6 (baseline)', '29 (expanded)'],
            'values': [0.6191, 0.6210, 0.5997],
            'best': '10 (selected)',
            'delta': 0.021
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    for idx, (param, data) in enumerate(sensitivity_data.items()):
        ax = axes[idx]

        x = np.arange(len(data['options']))
        values = data['values']

        # Highlight best option
        bar_colors = ['#2ecc71' if o == data['best'] else '#3498db' for o in data['options']]

        bars = ax.bar(x, values, color=bar_colors, edgecolor='black', alpha=0.8, linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(data['options'], rotation=15, ha='right')
        ax.set_ylabel('Mean APFD', fontweight='bold')
        ax.set_title(f'{param}\n(Δ = {data["delta"]:.3f})', fontweight='bold')
        ax.set_ylim(0.55, 0.65)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Main title
    fig.suptitle('RQ4: Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_rq4_sensitivity.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def generate_fig_qualitative():
    """Generate qualitative analysis figure."""
    print("\n" + "="*60)
    print("Generating fig_qualitative")
    print("="*60)

    # Load experiment 07 APFD data
    exp07_apfd = load_experiment_07_apfd()
    baseline_data = load_baseline_apfd()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. APFD Distribution (top-left)
    ax = axes[0, 0]
    ax.hist(exp07_apfd, bins=25, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(np.mean(exp07_apfd), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(exp07_apfd):.4f}')
    ax.axvline(np.median(exp07_apfd), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(exp07_apfd):.4f}')
    ax.set_xlabel('APFD Score', fontweight='bold')
    ax.set_ylabel('Number of Builds', fontweight='bold')
    ax.set_title('APFD Distribution (277 Builds)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative Distribution (top-right)
    ax = axes[0, 1]
    sorted_apfd = np.sort(exp07_apfd)
    cumulative = np.arange(1, len(sorted_apfd) + 1) / len(sorted_apfd) * 100

    ax.plot(sorted_apfd, cumulative, linewidth=2, color='#2ecc71')
    ax.fill_between(sorted_apfd, cumulative, alpha=0.3, color='#2ecc71')
    ax.axvline(0.7, color='orange', linestyle='--', alpha=0.7, label='APFD = 0.7')
    ax.set_xlabel('APFD Score', fontweight='bold')
    ax.set_ylabel('Cumulative % of Builds', fontweight='bold')
    ax.set_title('Cumulative APFD Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate statistics
    pct_above_07 = (np.array(exp07_apfd) >= 0.7).sum() / len(exp07_apfd) * 100
    ax.text(0.72, 30, f'{pct_above_07:.1f}% of builds\nachieve APFD ≥ 0.7',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Comparison with Random (bottom-left)
    ax = axes[1, 0]
    random_apfd = baseline_data['Random']

    # Paired comparison
    diff = np.array(exp07_apfd) - np.array(random_apfd)
    wins = (diff > 0).sum()
    losses = (diff < 0).sum()
    ties = (diff == 0).sum()

    ax.hist(diff, bins=25, edgecolor='black', alpha=0.7, color='#9b59b6')
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diff):+.4f}')
    ax.set_xlabel('APFD Difference (Filo-Priori - Random)', fontweight='bold')
    ax.set_ylabel('Number of Builds', fontweight='bold')
    ax.set_title(f'Per-Build Improvement over Random\n(Wins: {wins}, Losses: {losses}, Ties: {ties})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Performance by Build Characteristics (bottom-right)
    ax = axes[1, 1]

    # Categorize by APFD performance
    categories = ['Poor\n(APFD<0.4)', 'Medium\n(0.4-0.7)', 'Good\n(APFD≥0.7)']
    filo_counts = [
        (np.array(exp07_apfd) < 0.4).sum(),
        ((np.array(exp07_apfd) >= 0.4) & (np.array(exp07_apfd) < 0.7)).sum(),
        (np.array(exp07_apfd) >= 0.7).sum()
    ]
    random_counts = [
        (np.array(random_apfd) < 0.4).sum(),
        ((np.array(random_apfd) >= 0.4) & (np.array(random_apfd) < 0.7)).sum(),
        (np.array(random_apfd) >= 0.7).sum()
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, filo_counts, width, label='Filo-Priori', color='#2ecc71', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, random_counts, width, label='Random', color='#e74c3c', edgecolor='black', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Number of Builds', fontweight='bold')
    ax.set_title('Performance Category Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{int(bar.get_height())}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{int(bar.get_height())}', ha='center', fontsize=9)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = PAPER_DIR / f"fig_qualitative.{ext}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def main():
    """Generate all paper figures."""
    print("\n" + "="*70)
    print("FILO-PRIORI V9 - PAPER FIGURES GENERATOR")
    print("="*70)

    # Ensure output directory exists
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    generate_fig_rq1_comparison()
    generate_fig_rq1_improvement()
    generate_fig_rq2_ablation()
    generate_fig_rq3_temporal()
    generate_fig_rq4_sensitivity()
    generate_fig_qualitative()

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput directory: {PAPER_DIR}")
    print("\nGenerated files:")
    for f in sorted(PAPER_DIR.glob("fig_*.png")):
        print(f"  - {f.name}")
    print("="*70)


if __name__ == "__main__":
    main()
