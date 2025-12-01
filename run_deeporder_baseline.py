#!/usr/bin/env python3
"""
DeepOrder Baseline Runner for Industry Dataset (01_industry)

This script runs the DeepOrder baseline on the same dataset split
as Filo-Priori for fair comparison.

Usage:
    python run_deeporder_baseline.py --config configs/experiment_industry.yaml
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from baselines.deeporder import DeepOrderModel, run_deeporder_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_apfd(ranking: list, verdicts: dict) -> float:
    """
    Compute APFD (Average Percentage of Faults Detected).

    Args:
        ranking: Ordered list of test case IDs (prioritized order)
        verdicts: Dict mapping test_id -> verdict (1=fail, 0=pass)

    Returns:
        APFD score (0-1, higher is better)
    """
    n_tests = len(ranking)
    n_faults = sum(verdicts.values())

    if n_faults == 0 or n_tests == 0:
        return 0.0

    fault_positions = []
    for i, test_id in enumerate(ranking):
        if verdicts.get(test_id, 0) == 1:
            fault_positions.append(i + 1)

    if not fault_positions:
        return 0.0

    apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
    return apfd


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_deeporder_on_industry(config: dict, results_dir: str):
    """
    Run DeepOrder baseline on industry dataset.

    Uses the same train/test split as Filo-Priori for fair comparison.
    """
    logger.info("=" * 60)
    logger.info("DEEPORDER BASELINE - Industry Dataset")
    logger.info("=" * 60)

    # Load data
    train_path = config['data']['train_path']
    test_path = config['data']['test_path']

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path, low_memory=False)

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, low_memory=False)

    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Column mappings for Industry dataset
    build_col = 'Build_ID'
    test_col = 'TC_Key'
    result_col = 'TE_Test_Result'
    duration_col = None  # Industry dataset doesn't have duration

    # Initialize DeepOrder model
    model = DeepOrderModel(
        hidden_dims=[64, 32, 16],
        dropout=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        history_window=10,
        device='cuda' if config.get('system', {}).get('device') == 'cuda' else 'cpu'
    )

    # Train on training data
    logger.info("Training DeepOrder model...")
    model.train(
        df=train_df,
        build_col=build_col,
        test_col=test_col,
        result_col=result_col,
        duration_col=duration_col
    )

    # Evaluate on test data (by build)
    logger.info("Evaluating on test data...")

    test_builds = test_df[build_col].unique().tolist()
    logger.info(f"Total test builds: {len(test_builds)}")

    apfd_results = []
    builds_with_failures = 0

    for i, build_id in enumerate(test_builds):
        build_df = test_df[test_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts for this build
        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() not in ['PASS', 'PASSED'] else 0
            verdicts[test_id] = verdict

        n_faults = sum(verdicts.values())

        # Only evaluate builds with at least one failure
        if n_faults > 0:
            builds_with_failures += 1

            # Get prioritized ranking from DeepOrder
            ranking = model.prioritize(test_ids)

            # Compute APFD
            apfd = compute_apfd(ranking, verdicts)
            apfd_results.append({
                'build_id': build_id,
                'apfd': apfd,
                'n_tests': len(test_ids),
                'n_faults': n_faults
            })

            # Update history for online learning (important for DeepOrder)
            test_results = {}
            for test_id, verdict in verdicts.items():
                test_results[test_id] = (verdict, 1.0)
            model.update_history(build_id, test_results)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(test_builds)} builds...")

    # Compute statistics
    if apfd_results:
        apfd_values = [r['apfd'] for r in apfd_results]
        mean_apfd = np.mean(apfd_values)
        std_apfd = np.std(apfd_values)
        median_apfd = np.median(apfd_values)
        min_apfd = np.min(apfd_values)
        max_apfd = np.max(apfd_values)
    else:
        mean_apfd = std_apfd = median_apfd = min_apfd = max_apfd = 0.0

    # Results summary
    results = {
        'method': 'DeepOrder',
        'dataset': '01_industry',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mean_apfd': float(mean_apfd),
            'std_apfd': float(std_apfd),
            'median_apfd': float(median_apfd),
            'min_apfd': float(min_apfd),
            'max_apfd': float(max_apfd)
        },
        'statistics': {
            'total_test_builds': len(test_builds),
            'builds_with_failures': builds_with_failures,
            'builds_evaluated': len(apfd_results)
        },
        'per_build_results': apfd_results
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("DEEPORDER RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Builds evaluated: {len(apfd_results)}")
    logger.info(f"Mean APFD: {mean_apfd:.4f} (+/- {std_apfd:.4f})")
    logger.info(f"Median APFD: {median_apfd:.4f}")
    logger.info(f"Min APFD: {min_apfd:.4f}")
    logger.info(f"Max APFD: {max_apfd:.4f}")
    logger.info("=" * 60)

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'deeporder_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save per-build APFD CSV
    apfd_csv_file = os.path.join(results_dir, 'deeporder_apfd_per_build.csv')
    apfd_df = pd.DataFrame(apfd_results)
    apfd_df.to_csv(apfd_csv_file, index=False)
    logger.info(f"Per-build APFD saved to {apfd_csv_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run DeepOrder baseline')
    parser.add_argument('--config', type=str, default='configs/experiment_industry.yaml',
                        help='Path to config file')
    parser.add_argument('--results-dir', type=str, default='results/deeporder_baseline',
                        help='Directory to save results')
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_deeporder_on_industry(config, args.results_dir)

    print(f"\n{'='*60}")
    print(f"DEEPORDER BASELINE COMPLETE")
    print(f"Mean APFD: {results['metrics']['mean_apfd']:.4f}")
    print(f"Results: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
