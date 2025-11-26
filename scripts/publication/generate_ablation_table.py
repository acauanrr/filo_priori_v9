#!/usr/bin/env python3
"""
Generate Final Ablation Table for Publication.

Combines results from:
1. Quick ablations (A1-A3, A6)
2. Existing experiment results (A0, A5, A7)

Author: Filo-Priori Team
Date: 2025-11-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent


def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval."""
    values = np.array(values)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return np.mean(values), lower, upper


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return abs(d)


def main():
    # Load quick ablation results
    quick_path = PROJECT_ROOT / 'results' / 'ablation' / 'quick_ablation_apfd_values.json'
    with open(quick_path) as f:
        quick_results = json.load(f)

    # Component descriptions
    components = {
        'A0': {'name': 'Full Model', 'component': 'All components', 'description': 'Complete Filo-Priori v9'},
        'A1': {'name': 'w/o Semantic', 'component': 'Semantic Stream', 'description': 'Without text embeddings'},
        'A2': {'name': 'w/o Structural', 'component': 'Structural Stream', 'description': 'Without structural features'},
        'A3': {'name': 'w/o GATv2', 'component': 'Graph Attention', 'description': 'MLP instead of GAT'},
        'A5': {'name': 'w/o Class Weights', 'component': 'Class Weighting', 'description': 'Standard CE loss'},
        'A6': {'name': 'w/o Cross-Attention', 'component': 'Cross-Attention', 'description': 'Simple concatenation'},
        'A7': {'name': 'w/o Ensemble', 'component': 'Ensemble', 'description': 'Base model only'},
    }

    # Get full model as reference
    full_model = np.array(quick_results['A0'])
    full_mean = np.mean(full_model)

    # Generate table rows
    rows = []
    for ablation_id, values in sorted(quick_results.items()):
        values = np.array(values)
        info = components.get(ablation_id, {'name': ablation_id, 'component': 'Unknown'})

        # Calculate statistics
        mean_apfd, ci_lower, ci_upper = bootstrap_ci(values)
        delta = mean_apfd - full_mean
        contribution = (full_mean - mean_apfd) / full_mean * 100

        # Statistical test (paired t-test)
        if ablation_id != 'A0':
            t_stat, p_val = stats.ttest_rel(full_model, values)
            d = cohens_d(full_model, values)
        else:
            p_val = 1.0
            d = 0.0

        rows.append({
            'ID': ablation_id,
            'Model Variant': info['name'],
            'Removed Component': info['component'],
            'Mean APFD': mean_apfd,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Δ APFD': delta,
            'Contribution (%)': contribution,
            'p-value': p_val,
            "Cohen's d": d,
            'Significant': '***' if p_val < 0.05 else ''
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by contribution (descending)
    df = df.sort_values('Contribution (%)', ascending=False).reset_index(drop=True)

    # Print table
    print("\n" + "=" * 100)
    print(" ABLATION STUDY - FILO-PRIORI V9")
    print("=" * 100)
    print(f"\n{'ID':<5} {'Model Variant':<20} {'Component':<20} {'APFD':<8} {'Δ APFD':<10} {'Contrib.':<10} {'Sig.'}")
    print("-" * 100)

    for _, row in df.iterrows():
        if row['ID'] == 'A0':
            print(f"{row['ID']:<5} {row['Model Variant']:<20} {row['Removed Component']:<20} {row['Mean APFD']:.4f}   {'-':<10} {'-':<10}")
        else:
            print(f"{row['ID']:<5} {row['Model Variant']:<20} {row['Removed Component']:<20} {row['Mean APFD']:.4f}   {row['Δ APFD']:+.4f}    {row['Contribution (%)']:+.1f}%      {row['Significant']}")

    print("-" * 100)

    # Save CSV
    output_dir = PROJECT_ROOT / 'results' / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'ablation_study_final.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Generate LaTeX table
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Component Contributions to Filo-Priori v9 Performance}
\label{tab:ablation}
\begin{tabular}{llccccc}
\toprule
\textbf{ID} & \textbf{Removed Component} & \textbf{Mean APFD} & \textbf{95\% CI} & \textbf{$\Delta$ APFD} & \textbf{Contrib.} & \textbf{Sig.} \\
\midrule
"""

    for _, row in df.iterrows():
        ablation_id = row['ID']
        component = row['Removed Component']
        mean_apfd = f"{row['Mean APFD']:.4f}"
        ci = f"[{row['95% CI Lower']:.3f}, {row['95% CI Upper']:.3f}]"

        if ablation_id == 'A0':
            latex += f"\\textbf{{{ablation_id}}} & \\textbf{{All (Full Model)}} & \\textbf{{{mean_apfd}}} & {ci} & - & - & \\\\\n"
        else:
            delta = f"{row['Δ APFD']:+.4f}"
            contrib = f"{row['Contribution (%)']:+.1f}\\%"
            sig = row['Significant']
            latex += f"{ablation_id} & {component} & {mean_apfd} & {ci} & {delta} & {contrib} & {sig} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\Delta$ APFD shows performance change when component is removed (negative = performance drop).
\item Contribution shows how much each component adds to the full model's performance.
\item *** indicates statistical significance (paired t-test, $p < 0.05$).
\end{tablenotes}
\end{table}
"""

    latex_path = output_dir / 'ablation_study_final.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {latex_path}")

    # Generate visualization
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150
    })

    # Filter out full model for the chart
    ablation_df = df[df['ID'] != 'A0'].copy()
    ablation_df = ablation_df.sort_values('Contribution (%)', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(ablation_df))
    contributions = ablation_df['Contribution (%)'].values
    colors = ['#e74c3c' if sig == '***' else '#3498db' for sig in ablation_df['Significant']]

    bars = ax.barh(y_pos, contributions, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ablation_df['Removed Component'].values)
    ax.set_xlabel('Contribution to APFD (%)')
    ax.set_title('Component Importance - Ablation Study')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, contributions):
        x_pos = val + 0.3 if val >= 0 else val - 1.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:+.1f}%', va='center', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Significant (p < 0.05)'),
        Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study_final.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ablation_study_final.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: ablation_study_final.png/pdf")

    # Summary
    print("\n" + "=" * 70)
    print(" KEY FINDINGS")
    print("=" * 70)
    print(f"Full Model APFD: {full_mean:.4f}")
    print("\nComponent Ranking (by contribution):")

    ranking = ablation_df.sort_values('Contribution (%)', ascending=False)
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        sig = " ***" if row['Significant'] == '***' else ""
        print(f"  {i}. {row['Removed Component']}: +{row['Contribution (%)']:.1f}%{sig}")

    print("\nConclusion: " + ranking.iloc[0]['Removed Component'] + " is the most important component.")
    print("=" * 70)


if __name__ == "__main__":
    main()
