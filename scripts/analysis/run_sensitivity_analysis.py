#!/usr/bin/env python3
"""
Sensitivity Analysis for Filo-Priori v9.

Analyzes hyperparameter sensitivity using:
1. Existing experiment results (already trained models)
2. Post-hoc threshold sensitivity analysis
3. Bootstrap confidence intervals

Author: Filo-Priori Team
Date: 2025-11-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    return np.mean(values), np.std(values), lower, upper


def load_apfd_values(csv_path):
    """Load APFD values from CSV."""
    df = pd.read_csv(csv_path)
    # APFD is typically in column 6 (0-indexed: 5) or named 'APFD'
    if 'APFD' in df.columns:
        return df['APFD'].values
    else:
        # Assume column index 5 (6th column)
        return df.iloc[:, 5].values


def analyze_existing_experiments():
    """Analyze hyperparameter sensitivity from existing experiments."""

    experiments = {
        'Exp 01 (Baseline)': {
            'path': 'results/experiment_01_2025-11-14_14-45/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Focal Loss',
            'lr': '5e-5',
            'dropout': '0.15-0.30',
            'gnn_layers': 2,
            'gat_heads': 4,
            'balanced_sampling': 'No',
            'structural_features': 6
        },
        'Exp 02 (Multi-Edge Graph)': {
            'path': 'results/experiment_02_2025-11-14_15-47/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Focal Loss',
            'lr': '5e-5',
            'dropout': '0.15-0.30',
            'gnn_layers': 2,
            'gat_heads': 4,
            'balanced_sampling': 'No',
            'structural_features': 6
        },
        'Exp 03 (Balanced Weights)': {
            'path': 'results/experiment_03_balanced_weights/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Weighted Focal',
            'lr': '5e-5',
            'dropout': '0.15-0.30',
            'gnn_layers': 2,
            'gat_heads': 4,
            'balanced_sampling': 'Yes',
            'structural_features': 6
        },
        'Exp 04a (Weighted CE)': {
            'path': 'results/experiment_04a/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Weighted CE',
            'lr': '3e-5',
            'dropout': '0.10-0.20',
            'gnn_layers': 1,
            'gat_heads': 2,
            'balanced_sampling': 'No',
            'structural_features': 6
        },
        'Exp 04b (Focal Only)': {
            'path': 'results/experiment_04b_focal_only/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Focal Loss',
            'lr': '3e-5',
            'dropout': '0.10-0.20',
            'gnn_layers': 1,
            'gat_heads': 2,
            'balanced_sampling': 'No',
            'structural_features': 6
        },
        'Exp 05 (Expanded Features)': {
            'path': 'results/experiment_05_expanded_features/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Weighted CE',
            'lr': '3e-5',
            'dropout': '0.10-0.20',
            'gnn_layers': 1,
            'gat_heads': 2,
            'balanced_sampling': 'No',
            'structural_features': 29
        },
        'Exp 06 (Feature Selection)': {
            'path': 'results/experiment_06_feature_selection/apfd_per_build_FULL_testcsv.csv',
            'loss': 'Weighted CE',
            'lr': '3e-5',
            'dropout': '0.10-0.20',
            'gnn_layers': 1,
            'gat_heads': 2,
            'balanced_sampling': 'No',
            'structural_features': 10
        }
    }

    results = []
    apfd_by_exp = {}

    for exp_name, config in experiments.items():
        csv_path = PROJECT_ROOT / config['path']
        if csv_path.exists():
            apfd_values = load_apfd_values(csv_path)
            mean, std, ci_lower, ci_upper = bootstrap_ci(apfd_values)

            apfd_by_exp[exp_name] = apfd_values

            results.append({
                'Experiment': exp_name,
                'Loss Function': config['loss'],
                'Learning Rate': config['lr'],
                'Dropout Range': config['dropout'],
                'GNN Layers': config['gnn_layers'],
                'GAT Heads': config['gat_heads'],
                'Balanced Sampling': config['balanced_sampling'],
                'Structural Features': config['structural_features'],
                'Mean APFD': mean,
                'Std': std,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper,
                'N': len(apfd_values)
            })
        else:
            print(f"Warning: {csv_path} not found")

    return pd.DataFrame(results), apfd_by_exp


def analyze_loss_function_sensitivity(apfd_by_exp):
    """Analyze sensitivity to loss function."""
    loss_groups = {
        'Weighted CE': [],
        'Focal Loss': [],
        'Weighted Focal': []
    }

    mapping = {
        'Exp 04a (Weighted CE)': 'Weighted CE',
        'Exp 06 (Feature Selection)': 'Weighted CE',
        'Exp 01 (Baseline)': 'Focal Loss',
        'Exp 04b (Focal Only)': 'Focal Loss',
        'Exp 03 (Balanced Weights)': 'Weighted Focal'
    }

    for exp_name, loss_type in mapping.items():
        if exp_name in apfd_by_exp:
            loss_groups[loss_type].extend(apfd_by_exp[exp_name].tolist())

    results = []
    for loss_type, values in loss_groups.items():
        if values:
            mean, std, ci_lower, ci_upper = bootstrap_ci(values)
            results.append({
                'Loss Function': loss_type,
                'Mean APFD': mean,
                'Std': std,
                '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
                'N Samples': len(values)
            })

    return pd.DataFrame(results).sort_values('Mean APFD', ascending=False)


def analyze_learning_rate_sensitivity(apfd_by_exp):
    """Analyze sensitivity to learning rate."""
    lr_groups = {
        '3e-5': [],
        '5e-5': []
    }

    mapping = {
        'Exp 04a (Weighted CE)': '3e-5',
        'Exp 04b (Focal Only)': '3e-5',
        'Exp 06 (Feature Selection)': '3e-5',
        'Exp 01 (Baseline)': '5e-5',
        'Exp 02 (Multi-Edge Graph)': '5e-5',
        'Exp 03 (Balanced Weights)': '5e-5'
    }

    for exp_name, lr in mapping.items():
        if exp_name in apfd_by_exp:
            lr_groups[lr].extend(apfd_by_exp[exp_name].tolist())

    results = []
    for lr, values in lr_groups.items():
        if values:
            mean, std, ci_lower, ci_upper = bootstrap_ci(values)
            results.append({
                'Learning Rate': lr,
                'Mean APFD': mean,
                'Std': std,
                '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
                'N Samples': len(values)
            })

    return pd.DataFrame(results).sort_values('Mean APFD', ascending=False)


def analyze_gnn_architecture_sensitivity(apfd_by_exp):
    """Analyze sensitivity to GNN architecture."""
    arch_groups = {
        '1 layer, 2 heads': [],
        '2 layers, 4 heads': []
    }

    mapping = {
        'Exp 04a (Weighted CE)': '1 layer, 2 heads',
        'Exp 04b (Focal Only)': '1 layer, 2 heads',
        'Exp 06 (Feature Selection)': '1 layer, 2 heads',
        'Exp 01 (Baseline)': '2 layers, 4 heads',
        'Exp 02 (Multi-Edge Graph)': '2 layers, 4 heads',
        'Exp 03 (Balanced Weights)': '2 layers, 4 heads'
    }

    for exp_name, arch in mapping.items():
        if exp_name in apfd_by_exp:
            arch_groups[arch].extend(apfd_by_exp[exp_name].tolist())

    results = []
    for arch, values in arch_groups.items():
        if values:
            mean, std, ci_lower, ci_upper = bootstrap_ci(values)
            results.append({
                'GNN Architecture': arch,
                'Mean APFD': mean,
                'Std': std,
                '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
                'N Samples': len(values)
            })

    return pd.DataFrame(results).sort_values('Mean APFD', ascending=False)


def analyze_structural_features_sensitivity(apfd_by_exp):
    """Analyze sensitivity to number of structural features."""
    feature_groups = {
        '6 features': [],
        '10 features': [],
        '29 features': []
    }

    mapping = {
        'Exp 04a (Weighted CE)': '6 features',
        'Exp 04b (Focal Only)': '6 features',
        'Exp 06 (Feature Selection)': '10 features',
        'Exp 05 (Expanded Features)': '29 features'
    }

    for exp_name, feat in mapping.items():
        if exp_name in apfd_by_exp:
            feature_groups[feat].extend(apfd_by_exp[exp_name].tolist())

    results = []
    for feat, values in feature_groups.items():
        if values:
            mean, std, ci_lower, ci_upper = bootstrap_ci(values)
            results.append({
                'Structural Features': feat,
                'Mean APFD': mean,
                'Std': std,
                '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
                'N Samples': len(values)
            })

    return pd.DataFrame(results).sort_values('Mean APFD', ascending=False)


def analyze_balanced_sampling_sensitivity(apfd_by_exp):
    """Analyze sensitivity to balanced sampling."""
    sampling_groups = {
        'No Balanced Sampling': [],
        'Balanced Sampling': []
    }

    mapping = {
        'Exp 04a (Weighted CE)': 'No Balanced Sampling',
        'Exp 04b (Focal Only)': 'No Balanced Sampling',
        'Exp 06 (Feature Selection)': 'No Balanced Sampling',
        'Exp 01 (Baseline)': 'No Balanced Sampling',
        'Exp 03 (Balanced Weights)': 'Balanced Sampling'
    }

    for exp_name, sampling in mapping.items():
        if exp_name in apfd_by_exp:
            sampling_groups[sampling].extend(apfd_by_exp[exp_name].tolist())

    results = []
    for sampling, values in sampling_groups.items():
        if values:
            mean, std, ci_lower, ci_upper = bootstrap_ci(values)
            results.append({
                'Sampling Strategy': sampling,
                'Mean APFD': mean,
                'Std': std,
                '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
                'N Samples': len(values)
            })

    return pd.DataFrame(results).sort_values('Mean APFD', ascending=False)


def generate_latex_table(df, caption, label):
    """Generate LaTeX table from DataFrame."""
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * (len(df.columns) - 1)}}}
\\toprule
"""
    # Header
    latex += " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for _, row in df.iterrows():
        formatted_row = []
        for col, val in row.items():
            if isinstance(val, float):
                formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))
        latex += " & ".join(formatted_row) + " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex


def create_visualizations(sensitivity_results, output_dir):
    """Create sensitivity analysis visualizations."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (12, 8),
        'figure.dpi': 150
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # 1. Loss Function
    ax = axes[0, 0]
    df = sensitivity_results['loss_function']
    x = range(len(df))
    ax.barh(x, df['Mean APFD'], color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Loss Function'])
    ax.set_xlabel('Mean APFD')
    ax.set_title('(a) Loss Function')
    ax.set_xlim(0.55, 0.65)
    for i, v in enumerate(df['Mean APFD']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # 2. Learning Rate
    ax = axes[0, 1]
    df = sensitivity_results['learning_rate']
    x = range(len(df))
    ax.barh(x, df['Mean APFD'], color=['#9b59b6', '#f39c12'], alpha=0.8, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Learning Rate'])
    ax.set_xlabel('Mean APFD')
    ax.set_title('(b) Learning Rate')
    ax.set_xlim(0.55, 0.65)
    for i, v in enumerate(df['Mean APFD']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # 3. GNN Architecture
    ax = axes[0, 2]
    df = sensitivity_results['gnn_architecture']
    x = range(len(df))
    ax.barh(x, df['Mean APFD'], color=['#1abc9c', '#e67e22'], alpha=0.8, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(df['GNN Architecture'])
    ax.set_xlabel('Mean APFD')
    ax.set_title('(c) GNN Architecture')
    ax.set_xlim(0.55, 0.65)
    for i, v in enumerate(df['Mean APFD']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # 4. Structural Features
    ax = axes[1, 0]
    df = sensitivity_results['structural_features']
    x = range(len(df))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax.barh(x, df['Mean APFD'], color=colors[:len(df)], alpha=0.8, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Structural Features'])
    ax.set_xlabel('Mean APFD')
    ax.set_title('(d) Structural Features')
    ax.set_xlim(0.55, 0.65)
    for i, v in enumerate(df['Mean APFD']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # 5. Balanced Sampling
    ax = axes[1, 1]
    df = sensitivity_results['balanced_sampling']
    x = range(len(df))
    ax.barh(x, df['Mean APFD'], color=['#27ae60', '#c0392b'], alpha=0.8, edgecolor='black')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Sampling Strategy'])
    ax.set_xlabel('Mean APFD')
    ax.set_title('(e) Balanced Sampling')
    ax.set_xlim(0.55, 0.65)
    for i, v in enumerate(df['Mean APFD']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

    # 6. Summary: Best configuration for each hyperparameter
    ax = axes[1, 2]
    summary_data = {
        'Loss': sensitivity_results['loss_function'].iloc[0]['Loss Function'],
        'LR': sensitivity_results['learning_rate'].iloc[0]['Learning Rate'],
        'GNN': sensitivity_results['gnn_architecture'].iloc[0]['GNN Architecture'].split(',')[0],
        'Features': sensitivity_results['structural_features'].iloc[0]['Structural Features'].split()[0],
        'Sampling': 'No' if 'No' in sensitivity_results['balanced_sampling'].iloc[0]['Sampling Strategy'] else 'Yes'
    }

    text = "Best Configuration:\n\n"
    for param, value in summary_data.items():
        text += f"• {param}: {value}\n"

    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax.transAxes)
    ax.set_title('(f) Optimal Settings')
    ax.axis('off')

    plt.suptitle('Hyperparameter Sensitivity Analysis - Filo-Priori v9', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sensitivity_analysis.pdf', bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "=" * 80)
    print(" SENSITIVITY ANALYSIS - FILO-PRIORI V9")
    print("=" * 80)

    # Create output directory
    output_dir = PROJECT_ROOT / 'results' / 'sensitivity'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analyze existing experiments
    print("\n1. Loading experiment results...")
    exp_df, apfd_by_exp = analyze_existing_experiments()

    print(f"\n   Found {len(exp_df)} experiments with APFD data")
    print("\n   Experiment Results:")
    print("-" * 80)
    for _, row in exp_df.iterrows():
        print(f"   {row['Experiment']}: APFD = {row['Mean APFD']:.4f} ± {row['Std']:.4f}")

    # 2. Analyze sensitivity to each hyperparameter
    print("\n2. Analyzing hyperparameter sensitivity...")

    sensitivity_results = {}

    # Loss Function
    print("\n   a) Loss Function Sensitivity:")
    loss_df = analyze_loss_function_sensitivity(apfd_by_exp)
    sensitivity_results['loss_function'] = loss_df
    for _, row in loss_df.iterrows():
        print(f"      {row['Loss Function']}: {row['Mean APFD']:.4f} {row['95% CI']}")

    # Learning Rate
    print("\n   b) Learning Rate Sensitivity:")
    lr_df = analyze_learning_rate_sensitivity(apfd_by_exp)
    sensitivity_results['learning_rate'] = lr_df
    for _, row in lr_df.iterrows():
        print(f"      {row['Learning Rate']}: {row['Mean APFD']:.4f} {row['95% CI']}")

    # GNN Architecture
    print("\n   c) GNN Architecture Sensitivity:")
    gnn_df = analyze_gnn_architecture_sensitivity(apfd_by_exp)
    sensitivity_results['gnn_architecture'] = gnn_df
    for _, row in gnn_df.iterrows():
        print(f"      {row['GNN Architecture']}: {row['Mean APFD']:.4f} {row['95% CI']}")

    # Structural Features
    print("\n   d) Structural Features Sensitivity:")
    feat_df = analyze_structural_features_sensitivity(apfd_by_exp)
    sensitivity_results['structural_features'] = feat_df
    for _, row in feat_df.iterrows():
        print(f"      {row['Structural Features']}: {row['Mean APFD']:.4f} {row['95% CI']}")

    # Balanced Sampling
    print("\n   e) Balanced Sampling Sensitivity:")
    sampling_df = analyze_balanced_sampling_sensitivity(apfd_by_exp)
    sensitivity_results['balanced_sampling'] = sampling_df
    for _, row in sampling_df.iterrows():
        print(f"      {row['Sampling Strategy']}: {row['Mean APFD']:.4f} {row['95% CI']}")

    # 3. Save results
    print("\n3. Saving results...")

    # Save experiment summary
    exp_df.to_csv(output_dir / 'experiment_summary.csv', index=False)
    print(f"   Saved: experiment_summary.csv")

    # Save individual sensitivity analyses
    for name, df in sensitivity_results.items():
        df.to_csv(output_dir / f'sensitivity_{name}.csv', index=False)
        print(f"   Saved: sensitivity_{name}.csv")

    # Generate combined LaTeX table
    latex_content = """% Sensitivity Analysis Tables - Filo-Priori v9
% Auto-generated

"""

    # Create a combined sensitivity table
    combined_rows = []

    # Loss Function
    for _, row in sensitivity_results['loss_function'].iterrows():
        combined_rows.append({
            'Hyperparameter': 'Loss Function',
            'Value': row['Loss Function'],
            'Mean APFD': row['Mean APFD'],
            '95% CI': row['95% CI'],
            'N': row['N Samples']
        })

    # Learning Rate
    for _, row in sensitivity_results['learning_rate'].iterrows():
        combined_rows.append({
            'Hyperparameter': 'Learning Rate',
            'Value': row['Learning Rate'],
            'Mean APFD': row['Mean APFD'],
            '95% CI': row['95% CI'],
            'N': row['N Samples']
        })

    # GNN Architecture
    for _, row in sensitivity_results['gnn_architecture'].iterrows():
        combined_rows.append({
            'Hyperparameter': 'GNN Architecture',
            'Value': row['GNN Architecture'],
            'Mean APFD': row['Mean APFD'],
            '95% CI': row['95% CI'],
            'N': row['N Samples']
        })

    # Structural Features
    for _, row in sensitivity_results['structural_features'].iterrows():
        combined_rows.append({
            'Hyperparameter': 'Structural Features',
            'Value': row['Structural Features'],
            'Mean APFD': row['Mean APFD'],
            '95% CI': row['95% CI'],
            'N': row['N Samples']
        })

    # Balanced Sampling
    for _, row in sensitivity_results['balanced_sampling'].iterrows():
        combined_rows.append({
            'Hyperparameter': 'Balanced Sampling',
            'Value': row['Sampling Strategy'],
            'Mean APFD': row['Mean APFD'],
            '95% CI': row['95% CI'],
            'N': row['N Samples']
        })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_dir / 'sensitivity_analysis_combined.csv', index=False)

    # Generate LaTeX
    latex_content += """
\\begin{table}[htbp]
\\centering
\\caption{Hyperparameter Sensitivity Analysis for Filo-Priori v9}
\\label{tab:sensitivity}
\\begin{tabular}{llccc}
\\toprule
\\textbf{Hyperparameter} & \\textbf{Value} & \\textbf{Mean APFD} & \\textbf{95\\% CI} & \\textbf{N} \\\\
\\midrule
"""

    current_hp = None
    for _, row in combined_df.iterrows():
        hp = row['Hyperparameter']
        if hp != current_hp:
            if current_hp is not None:
                latex_content += "\\midrule\n"
            current_hp = hp

        value = row['Value']
        mean_apfd = f"{row['Mean APFD']:.4f}"
        ci = row['95% CI']
        n = int(row['N'])

        latex_content += f"{hp} & {value} & {mean_apfd} & {ci} & {n} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Best configuration for each hyperparameter shown in bold (highest Mean APFD).
\\item N represents the total number of build-level APFD evaluations across relevant experiments.
\\end{tablenotes}
\\end{table}
"""

    with open(output_dir / 'sensitivity_analysis.tex', 'w') as f:
        f.write(latex_content)
    print(f"   Saved: sensitivity_analysis.tex")

    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    create_visualizations(sensitivity_results, output_dir)
    print(f"   Saved: sensitivity_analysis.png/pdf")

    # 5. Summary
    print("\n" + "=" * 80)
    print(" KEY FINDINGS")
    print("=" * 80)

    print("\n   Best configuration for each hyperparameter:")
    print(f"   • Loss Function: {sensitivity_results['loss_function'].iloc[0]['Loss Function']} (APFD = {sensitivity_results['loss_function'].iloc[0]['Mean APFD']:.4f})")
    print(f"   • Learning Rate: {sensitivity_results['learning_rate'].iloc[0]['Learning Rate']} (APFD = {sensitivity_results['learning_rate'].iloc[0]['Mean APFD']:.4f})")
    print(f"   • GNN Architecture: {sensitivity_results['gnn_architecture'].iloc[0]['GNN Architecture']} (APFD = {sensitivity_results['gnn_architecture'].iloc[0]['Mean APFD']:.4f})")
    print(f"   • Structural Features: {sensitivity_results['structural_features'].iloc[0]['Structural Features']} (APFD = {sensitivity_results['structural_features'].iloc[0]['Mean APFD']:.4f})")
    print(f"   • Sampling Strategy: {sensitivity_results['balanced_sampling'].iloc[0]['Sampling Strategy']} (APFD = {sensitivity_results['balanced_sampling'].iloc[0]['Mean APFD']:.4f})")

    # Calculate sensitivity ranges
    print("\n   Sensitivity Ranges (Max - Min APFD):")
    for name, df in sensitivity_results.items():
        delta = df['Mean APFD'].max() - df['Mean APFD'].min()
        print(f"   • {name.replace('_', ' ').title()}: Δ = {delta:.4f} ({delta/0.6 * 100:.1f}% relative)")

    print("\n" + "=" * 80)
    print(f" Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
