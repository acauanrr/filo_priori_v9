#!/usr/bin/env python3
"""
Ablation Study Runner for Filo-Priori v9.

This script runs ablation experiments to analyze the contribution of each
model component. Each ablation removes or disables a specific component
and measures the impact on APFD performance.

Ablations:
A0: Full Model (baseline) - All components enabled
A1: w/o Semantic Stream - Removes text embedding processing
A2: w/o Structural Stream - Removes graph-based structural features
A3: w/o Graph Attention - Uses MLP instead of GATv2
A4: w/o Multi-Edge Graph - Uses only co-failure edges
A5: w/o Class Weights - Uses standard CE instead of weighted CE
A6: w/o Cross-Attention - Uses simple concatenation fusion
A7: w/o Ensemble - Base model without FailureRate combination

Usage:
    python run_ablation_study.py --run-all
    python run_ablation_study.py --ablations A1 A2 A3
    python run_ablation_study.py --generate-table

Author: Filo-Priori Team
Date: 2025-11-26
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.statistical_validation import (
    bootstrap_confidence_interval,
    paired_ttest,
    cohens_d,
    holm_correction
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Ablation configurations
# Note: Some ablations use existing experiment results as proxies
ABLATIONS = {
    'A0_full': {
        'name': 'Full Model',
        'config': 'configs/ablation/ablation_full_model.yaml',
        'description': 'Complete Filo-Priori v9 with all components',
        'component': 'All',
        'results_dir': 'results/ablation/A0_full_model'
    },
    'A1_wo_semantic': {
        'name': 'w/o Semantic',
        'config': 'configs/ablation/ablation_wo_semantic.yaml',
        'description': 'Without text embedding stream',
        'component': 'Semantic Stream',
        'results_dir': 'results/ablation/A1_wo_semantic'
    },
    'A2_wo_structural': {
        'name': 'w/o Structural',
        'config': 'configs/ablation/ablation_wo_structural.yaml',
        'description': 'Without structural/graph features',
        'component': 'Structural Stream',
        'results_dir': 'results/ablation/A2_wo_structural'
    },
    'A3_wo_graph': {
        'name': 'w/o GATv2',
        'config': 'configs/ablation/ablation_wo_graph.yaml',
        'description': 'MLP instead of Graph Attention',
        'component': 'Graph Attention',
        'results_dir': 'results/ablation/A3_wo_graph'
    },
    'A4_wo_multi_edge': {
        'name': 'w/o Multi-Edge',
        'config': 'configs/ablation/ablation_wo_multi_edge.yaml',
        'description': 'Single edge type (co-failure only)',
        'component': 'Multi-Edge Graph',
        'results_dir': 'results/ablation/A4_wo_multi_edge'
    },
    # A5 uses experiment_04b as proxy (Focal Loss instead of Weighted CE)
    'A5_wo_class_weights': {
        'name': 'w/o Class Weights',
        'config': 'configs/ablation/ablation_wo_class_weights.yaml',
        'description': 'Focal Loss instead of Weighted CE (proxy: experiment_04b)',
        'component': 'Class Weighting',
        'results_dir': 'results/experiment_04b_focal_only',  # Use existing experiment as proxy
        'proxy_experiment': 'experiment_04b_focal_only'
    },
    'A6_wo_fusion': {
        'name': 'w/o Cross-Attention',
        'config': 'configs/ablation/ablation_wo_fusion.yaml',
        'description': 'Simple concatenation fusion',
        'component': 'Cross-Attention Fusion',
        'results_dir': 'results/ablation/A6_wo_fusion'
    },
    'A7_wo_ensemble': {
        'name': 'w/o Ensemble',
        'config': None,  # Uses existing base model results
        'description': 'Base model without FailureRate ensemble',
        'component': 'Ensemble (FailureRate)',
        'results_dir': 'results/experiment_06_feature_selection'  # Existing results
    }
}


def load_apfd_results(results_dir: str) -> Optional[np.ndarray]:
    """Load APFD results from a results directory."""
    results_path = Path(results_dir)

    # Try different possible file names
    possible_files = [
        'apfd_per_build_FULL_testcsv.csv',
        'apfd_per_build.csv',
        'apfd_results.csv'
    ]

    for filename in possible_files:
        filepath = results_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'apfd' in df.columns:
                logger.info(f"Loaded {len(df)} APFD values from {filepath}")
                return df['apfd'].values

    # Try JSON format
    json_path = results_path / 'apfd_results.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            if 'apfd_values' in data:
                return np.array(data['apfd_values'])

    return None


def load_existing_results() -> Dict[str, np.ndarray]:
    """Load existing results from previous experiments."""
    results = {}

    # Load baseline results (includes Filo-Priori base)
    baseline_path = PROJECT_ROOT / 'results' / 'baselines' / 'all_apfd_results.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)

        if 'Filo-Priori' in baseline_data:
            results['A7_wo_ensemble'] = np.array(baseline_data['Filo-Priori'])
            logger.info(f"Loaded A7_wo_ensemble (Filo-Priori base): {len(results['A7_wo_ensemble'])} builds")

    # Load ensemble results (our A0 full model with ensemble)
    ensemble_path = PROJECT_ROOT / 'results' / 'calibrated' / 'apfd_per_build_ensemble.csv'
    if ensemble_path.exists():
        df = pd.read_csv(ensemble_path)
        results['A0_full'] = df['apfd'].values
        logger.info(f"Loaded A0_full (Ensemble): {len(results['A0_full'])} builds")

    # Load any existing ablation results
    for ablation_id, config in ABLATIONS.items():
        if ablation_id in results:
            continue  # Already loaded

        apfd_values = load_apfd_results(config['results_dir'])
        if apfd_values is not None:
            results[ablation_id] = apfd_values
            logger.info(f"Loaded {ablation_id}: {len(apfd_values)} builds")

    return results


def run_ablation_experiment(ablation_id: str, force: bool = False) -> bool:
    """Run a single ablation experiment."""
    if ablation_id not in ABLATIONS:
        logger.error(f"Unknown ablation: {ablation_id}")
        return False

    config = ABLATIONS[ablation_id]

    # Check if we need to run (config is None means use existing results)
    if config['config'] is None:
        logger.info(f"Ablation {ablation_id} uses existing results from {config['results_dir']}")
        return True

    # Check if already run
    if not force:
        apfd_values = load_apfd_results(config['results_dir'])
        if apfd_values is not None:
            logger.info(f"Ablation {ablation_id} already has results. Use --force to rerun.")
            return True

    # Run the experiment
    logger.info(f"\n{'='*70}")
    logger.info(f"Running ablation: {ablation_id} - {config['name']}")
    logger.info(f"Config: {config['config']}")
    logger.info(f"{'='*70}")

    config_path = PROJECT_ROOT / config['config']
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    # Run main.py with the ablation config
    cmd = [
        sys.executable, 'main.py',
        '--config', str(config_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Experiment failed:\n{result.stderr}")
            return False

        logger.info(f"Experiment {ablation_id} completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {ablation_id} timed out (>2 hours)")
        return False
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return False


def generate_ablation_table(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Generate ablation study comparison table."""
    if 'A0_full' not in results:
        logger.error("Full model results (A0_full) required as baseline")
        return pd.DataFrame()

    reference = results['A0_full']
    n_builds = len(reference)

    rows = []
    p_values = []
    ablation_ids = []

    for ablation_id, config in ABLATIONS.items():
        if ablation_id not in results:
            logger.warning(f"Missing results for {ablation_id}")
            continue

        values = results[ablation_id]

        # Ensure same length
        if len(values) != n_builds:
            logger.warning(f"{ablation_id} has {len(values)} builds, expected {n_builds}")
            continue

        # Bootstrap CI
        mean_apfd, ci_lower, ci_upper = bootstrap_confidence_interval(values, n_bootstrap=1000)

        row = {
            'Ablation': config['name'],
            'Component': config['component'],
            'Mean APFD': mean_apfd,
            '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            'Std': np.std(values)
        }

        # Statistical comparison with full model
        if ablation_id != 'A0_full':
            # Difference
            delta = mean_apfd - np.mean(reference)
            row['Δ APFD'] = delta

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(reference, values)
            p_val_two_sided = p_val

            # Effect size
            d, d_interp = cohens_d(reference, values)

            row['p-value'] = p_val_two_sided
            row["Cohen's d"] = d
            row['Effect'] = d_interp

            # Contribution (how much this component adds)
            contribution = np.mean(reference) - mean_apfd
            row['Contribution'] = contribution
            row['Contribution %'] = (contribution / np.mean(reference)) * 100

            p_values.append(p_val_two_sided)
            ablation_ids.append(ablation_id)
        else:
            row['Δ APFD'] = 0.0
            row['p-value'] = 1.0
            row["Cohen's d"] = 0.0
            row['Effect'] = '-'
            row['Contribution'] = 0.0
            row['Contribution %'] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply Holm correction
    if len(p_values) > 1:
        corrected = holm_correction(p_values, alpha=0.05)
        for i, ablation_id in enumerate(ablation_ids):
            config = ABLATIONS[ablation_id]
            idx = df[df['Ablation'] == config['name']].index[0]
            df.loc[idx, 'p-value (adj)'] = corrected[i][0]
            df.loc[idx, 'Significant'] = '***' if corrected[i][1] else ''

    # Sort by contribution (descending)
    df = df.sort_values('Contribution', ascending=False).reset_index(drop=True)

    return df


def generate_latex_ablation_table(df: pd.DataFrame) -> str:
    """Generate publication-ready LaTeX table for ablation study."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Contribution of Each Component to Filo-Priori v9 Performance}
\label{tab:ablation}
\begin{tabular}{llcccc}
\toprule
\textbf{Model Variant} & \textbf{Removed Component} & \textbf{Mean APFD} & \textbf{$\Delta$ APFD} & \textbf{Contribution} & \textbf{Sig.} \\
\midrule
"""

    for _, row in df.iterrows():
        name = row['Ablation'].replace('w/o ', 'w/o\\ ')
        component = row['Component']
        mean_apfd = f"{row['Mean APFD']:.4f}"

        if row['Ablation'] == 'Full Model':
            delta = '-'
            contrib = '-'
            sig = ''
            name = r'\textbf{Full Model}'
            mean_apfd = r'\textbf{' + mean_apfd + '}'
        else:
            delta = f"{row['Δ APFD']:+.4f}"
            contrib = f"{row['Contribution %']:+.1f}\\%"
            sig = row.get('Significant', '')

        latex += f"{name} & {component} & {mean_apfd} & {delta} & {contrib} & {sig} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $\Delta$ APFD shows the change compared to the full model (negative means performance drop).
\item Contribution shows how much each component adds to the full model's performance.
\item *** indicates the component's contribution is statistically significant ($\alpha = 0.05$, Holm-corrected).
\end{tablenotes}
\end{table}
"""
    return latex


def generate_component_importance_chart(df: pd.DataFrame, output_dir: Path):
    """Generate visualization of component importance."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    # Filter out full model
    ablation_df = df[df['Ablation'] != 'Full Model'].copy()

    if len(ablation_df) == 0:
        logger.warning("No ablation results to plot")
        return

    # Set style
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

    # Sort by contribution
    ablation_df = ablation_df.sort_values('Contribution', ascending=True)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(ablation_df))
    contributions = ablation_df['Contribution %'].values

    # Color based on significance
    colors = ['#e74c3c' if sig == '***' else '#3498db'
              for sig in ablation_df.get('Significant', [''] * len(ablation_df))]

    bars = ax.barh(y_pos, contributions, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ablation_df['Component'].values)
    ax.set_xlabel('Contribution to APFD (%)')
    ax.set_title('Component Importance (Ablation Study)')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, contributions):
        x_pos = val + 0.2 if val >= 0 else val - 0.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:+.1f}%', va='center', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Significant (p < 0.05)'),
        Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ablation_importance.pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: ablation_importance.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Run Ablation Study for Filo-Priori v9')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all ablation experiments')
    parser.add_argument('--ablations', nargs='+', type=str,
                       help='Specific ablations to run (e.g., A1 A2 A3)')
    parser.add_argument('--generate-table', action='store_true',
                       help='Generate ablation results table from existing results')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun even if results exist')
    parser.add_argument('--output', type=str, default='results/ablation',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(" ABLATION STUDY FOR FILO-PRIORI V9")
    logger.info("=" * 70)

    # Load existing results
    logger.info("\nLoading existing results...")
    results = load_existing_results()
    logger.info(f"Found results for {len(results)} ablations")

    # Run experiments if requested
    if args.run_all:
        ablations_to_run = list(ABLATIONS.keys())
    elif args.ablations:
        # Map short names (A1, A2) to full names (A1_wo_semantic, etc.)
        ablations_to_run = []
        for ablation in args.ablations:
            matched = [k for k in ABLATIONS.keys() if k.startswith(ablation)]
            ablations_to_run.extend(matched)
    else:
        ablations_to_run = []

    if ablations_to_run:
        logger.info(f"\nRunning {len(ablations_to_run)} ablation experiments...")
        for ablation_id in ablations_to_run:
            success = run_ablation_experiment(ablation_id, force=args.force)
            if success:
                # Reload results
                apfd_values = load_apfd_results(ABLATIONS[ablation_id]['results_dir'])
                if apfd_values is not None:
                    results[ablation_id] = apfd_values

    # Generate table
    if args.generate_table or ablations_to_run:
        logger.info("\n" + "=" * 70)
        logger.info(" GENERATING ABLATION RESULTS")
        logger.info("=" * 70)

        if len(results) < 2:
            logger.error("Need at least 2 ablation results to generate comparison")
            logger.info("Available results: " + ", ".join(results.keys()))
            logger.info("\nRun experiments first with:")
            logger.info("  python run_ablation_study.py --run-all")
            return

        # Generate table
        ablation_df = generate_ablation_table(results)

        if len(ablation_df) == 0:
            logger.error("Could not generate ablation table")
            return

        # Print table
        print("\n" + "=" * 100)
        print(" ABLATION STUDY RESULTS")
        print("=" * 100)

        # Select columns for display
        display_cols = ['Ablation', 'Component', 'Mean APFD', 'Δ APFD',
                       'Contribution %', 'p-value', 'Significant']
        display_cols = [c for c in display_cols if c in ablation_df.columns]
        print(ablation_df[display_cols].to_string(index=False))
        print("=" * 100)

        # Save CSV
        csv_path = output_dir / 'ablation_results.csv'
        ablation_df.to_csv(csv_path, index=False)
        logger.info(f"\nSaved: {csv_path}")

        # Generate LaTeX
        latex_table = generate_latex_ablation_table(ablation_df)
        latex_path = output_dir / 'ablation_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"Saved: {latex_path}")

        # Generate visualization
        generate_component_importance_chart(ablation_df, output_dir)

        # Save all APFD values
        apfd_path = output_dir / 'ablation_apfd_values.json'
        with open(apfd_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)
        logger.info(f"Saved: {apfd_path}")

        # Summary
        print("\n" + "=" * 70)
        print(" ABLATION STUDY SUMMARY")
        print("=" * 70)

        if 'A0_full' in results:
            full_apfd = np.mean(results['A0_full'])
            print(f"Full Model APFD: {full_apfd:.4f}")

            # Most important components
            important = ablation_df[ablation_df['Ablation'] != 'Full Model'].nlargest(3, 'Contribution')
            print("\nTop 3 Most Important Components:")
            for _, row in important.iterrows():
                sig = " (***)" if row.get('Significant', '') == '***' else ""
                print(f"  - {row['Component']}: +{row['Contribution %']:.1f}%{sig}")

        print("=" * 70)

    else:
        # Just show what's available
        logger.info("\nAvailable ablation results:")
        for ablation_id in results.keys():
            config = ABLATIONS.get(ablation_id, {})
            name = config.get('name', ablation_id)
            logger.info(f"  - {ablation_id}: {name} ({len(results[ablation_id])} builds)")

        logger.info("\nTo run ablation experiments:")
        logger.info("  python run_ablation_study.py --run-all")
        logger.info("  python run_ablation_study.py --ablations A1 A2 A3")
        logger.info("\nTo generate table from existing results:")
        logger.info("  python run_ablation_study.py --generate-table")


if __name__ == "__main__":
    main()
