#!/usr/bin/env python3
"""
Run All Baselines - Complete Comparison Script for Filo-Priori v9.

This script:
1. Loads train/test data
2. Extracts structural features
3. Runs all baseline methods (heuristic + ML)
4. Loads Filo-Priori predictions
5. Calculates APFD per build for all methods
6. Performs statistical comparison
7. Generates publication-ready tables and figures

Usage:
    python run_all_baselines.py [--output-dir results/baselines] [--skip-lstm]

Author: Filo-Priori Team
Date: 2025-11-26
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import baselines
from src.baselines.heuristic_baselines import (
    RandomBaseline, RecencyBaseline, FailureRateBaseline, GreedyHistoricalBaseline
)
from src.baselines.ml_baselines import (
    RandomForestBaseline, LogisticRegressionBaseline, XGBoostBaseline
)
from src.baselines.statistical_validation import (
    generate_comparison_table, print_comparison_table, generate_latex_table,
    bootstrap_confidence_interval
)
from src.evaluation.apfd import (
    calculate_apfd_per_build, calculate_ranks_per_build, generate_apfd_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineRunner:
    """Orchestrates running all baselines and comparisons."""

    def __init__(
        self,
        train_path: str = "datasets/train.csv",
        test_path: str = "datasets/test.csv",
        filo_priori_results: str = "results/experiment_06_feature_selection",
        output_dir: str = "results/baselines",
        cache_dir: str = "cache",
        seed: int = 42
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.filo_priori_results = Path(filo_priori_results)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.seed = seed

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.train_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None

        # Results
        self.apfd_results: Dict[str, np.ndarray] = {}
        self.apfd_per_build: Dict[str, pd.DataFrame] = {}
        self.summary_stats: Dict[str, Dict] = {}

    def load_data(self):
        """Load train and test datasets."""
        logger.info("Loading datasets...")

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        logger.info(f"  Train: {len(self.train_df):,} rows")
        logger.info(f"  Test: {len(self.test_df):,} rows")

        # Ensure required columns
        required_cols = ['Build_ID', 'TC_Key']
        for col in required_cols:
            if col not in self.test_df.columns:
                # Try alternatives
                if col == 'TC_Key' and 'tc_id' in self.test_df.columns:
                    self.train_df['TC_Key'] = self.train_df['tc_id']
                    self.test_df['TC_Key'] = self.test_df['tc_id']
                else:
                    logger.warning(f"Column {col} not found in data")

        # Ensure verdict column (create alias if needed)
        if 'verdict' not in self.test_df.columns:
            if 'TE_Test_Result' in self.test_df.columns:
                self.train_df['verdict'] = self.train_df['TE_Test_Result'].copy()
                self.test_df['verdict'] = self.test_df['TE_Test_Result'].copy()
                logger.info("  Created 'verdict' alias from 'TE_Test_Result'")

    def load_structural_features(self):
        """Load or compute structural features."""
        logger.info("Loading structural features...")

        cache_path = self.cache_dir / "structural_features_v2_5.pkl"

        if cache_path.exists():
            logger.info(f"  Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)

            # The cache contains feature extractor state, not features directly
            # We need to extract features from the loaded data
            from src.preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5

            extractor = StructuralFeatureExtractorV2_5()
            extractor.load_history(cache_path)

            # Get feature names
            feature_names = extractor.get_feature_names()

            # Check if features are already in the dataframe
            if all(f in self.test_df.columns for f in feature_names):
                logger.info("  Features already present in test data")
                self.train_features = self.train_df[feature_names].copy()
                self.test_features = self.test_df[feature_names].copy()
            else:
                logger.info("  Extracting features using cached extractor...")
                train_features = extractor.transform(self.train_df, is_test=False)
                test_features = extractor.transform(self.test_df, is_test=True)

                # Add to dataframes
                for i, name in enumerate(feature_names):
                    self.train_df[name] = train_features[:, i]
                    self.test_df[name] = test_features[:, i]

                self.train_features = pd.DataFrame(train_features, columns=feature_names)
                self.test_features = pd.DataFrame(test_features, columns=feature_names)
        else:
            logger.info("  Computing features from scratch...")
            from src.preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5

            extractor = StructuralFeatureExtractorV2_5()
            train_features = extractor.transform(self.train_df, is_test=False)
            test_features = extractor.transform(self.test_df, is_test=True)

            feature_names = extractor.get_feature_names()

            # Add to dataframes
            for i, name in enumerate(feature_names):
                self.train_df[name] = train_features[:, i]
                self.test_df[name] = test_features[:, i]

            self.train_features = pd.DataFrame(train_features, columns=feature_names)
            self.test_features = pd.DataFrame(test_features, columns=feature_names)

            # Save cache
            extractor.save_history(cache_path)

        logger.info(f"  Features shape: {self.test_features.shape}")

    def load_filo_priori_results(self):
        """Load Filo-Priori predictions and calculate APFD."""
        logger.info("Loading Filo-Priori results...")

        # Look for APFD file
        apfd_file = self.filo_priori_results / "apfd_per_build_FULL_testcsv.csv"

        if apfd_file.exists():
            logger.info(f"  Loading APFD from: {apfd_file}")
            filo_apfd_df = pd.read_csv(apfd_file)

            self.apfd_per_build['Filo-Priori'] = filo_apfd_df
            self.apfd_results['Filo-Priori'] = filo_apfd_df['apfd'].values

            logger.info(f"  Filo-Priori Mean APFD: {filo_apfd_df['apfd'].mean():.4f}")
        else:
            logger.warning(f"  APFD file not found: {apfd_file}")
            logger.warning("  Will compute from predictions if available")

            # Try to load predictions
            pred_file = self.filo_priori_results / "predictions_test.csv"
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file)
                apfd_df, summary = generate_apfd_report(
                    pred_df, method_name="Filo-Priori"
                )
                self.apfd_per_build['Filo-Priori'] = apfd_df
                self.apfd_results['Filo-Priori'] = apfd_df['apfd'].values
            else:
                logger.error("  No Filo-Priori results found!")

    def run_heuristic_baselines(self):
        """Run all heuristic baselines."""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING HEURISTIC BASELINES")
        logger.info("=" * 60)

        baselines = [
            ("Random", RandomBaseline(seed=self.seed)),
            ("Recency", RecencyBaseline(decay_factor=0.9)),
            ("FailureRate", FailureRateBaseline(smoothing=1.0)),
            ("RecentFailureRate", FailureRateBaseline(smoothing=1.0, use_recent=True, recent_window=10)),
            ("GreedyHistorical", GreedyHistoricalBaseline())
        ]

        for name, baseline in baselines:
            logger.info(f"\n--- {name} ---")

            try:
                # Fit on training data
                baseline.fit(self.train_df)

                # Generate rankings for test data
                test_ranked = baseline.rank_per_build(self.test_df.copy(), build_col='Build_ID')

                # Calculate APFD
                test_ranked['label_binary'] = (test_ranked['verdict'].astype(str).str.strip() == 'Fail').astype(int)

                apfd_df = calculate_apfd_per_build(
                    test_ranked,
                    method_name=name,
                    build_col='Build_ID',
                    label_col='label_binary',
                    rank_col='rank',
                    result_col='verdict'
                )

                self.apfd_per_build[name] = apfd_df
                self.apfd_results[name] = apfd_df['apfd'].values

                mean_apfd = apfd_df['apfd'].mean()
                logger.info(f"  Mean APFD: {mean_apfd:.4f}")
                logger.info(f"  Builds with APFD=1.0: {(apfd_df['apfd'] == 1.0).sum()}/{len(apfd_df)}")

            except Exception as e:
                logger.error(f"  Error running {name}: {e}")
                import traceback
                traceback.print_exc()

    def run_ml_baselines(self, skip_lstm: bool = False):
        """Run all ML baselines."""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING ML BASELINES")
        logger.info("=" * 60)

        baselines = [
            ("RandomForest", RandomForestBaseline(n_estimators=100, max_depth=10)),
            ("LogisticRegression", LogisticRegressionBaseline(C=1.0)),
            ("XGBoost", XGBoostBaseline(n_estimators=100, max_depth=6))
        ]

        for name, baseline in baselines:
            logger.info(f"\n--- {name} ---")

            try:
                # Fit on training data
                baseline.fit(self.train_df)

                # Generate rankings for test data
                test_ranked = baseline.rank_per_build(self.test_df.copy(), build_col='Build_ID')

                # Calculate APFD
                test_ranked['label_binary'] = (test_ranked['verdict'].astype(str).str.strip() == 'Fail').astype(int)

                apfd_df = calculate_apfd_per_build(
                    test_ranked,
                    method_name=name,
                    build_col='Build_ID',
                    label_col='label_binary',
                    rank_col='rank',
                    result_col='verdict'
                )

                self.apfd_per_build[name] = apfd_df
                self.apfd_results[name] = apfd_df['apfd'].values

                mean_apfd = apfd_df['apfd'].mean()
                logger.info(f"  Mean APFD: {mean_apfd:.4f}")
                logger.info(f"  Builds with APFD=1.0: {(apfd_df['apfd'] == 1.0).sum()}/{len(apfd_df)}")

                # Feature importance (if available)
                if hasattr(baseline, 'get_feature_importance'):
                    importance = baseline.get_feature_importance()
                    top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
                    logger.info(f"  Top features: {top_features}")

            except Exception as e:
                logger.error(f"  Error running {name}: {e}")
                import traceback
                traceback.print_exc()

    def generate_comparison_report(self):
        """Generate statistical comparison report."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING STATISTICAL COMPARISON")
        logger.info("=" * 60)

        if 'Filo-Priori' not in self.apfd_results:
            logger.error("Filo-Priori results not available for comparison")
            return

        # Ensure all arrays have same length (align by build_id)
        # Get reference build_ids from Filo-Priori
        ref_builds = set(self.apfd_per_build['Filo-Priori']['build_id'].values)

        aligned_results = {}
        for method, apfd_df in self.apfd_per_build.items():
            # Filter to common builds
            common_df = apfd_df[apfd_df['build_id'].isin(ref_builds)].copy()
            common_df = common_df.sort_values('build_id')
            aligned_results[method] = common_df['apfd'].values

        # Verify alignment
        n_builds = len(aligned_results['Filo-Priori'])
        for method, values in aligned_results.items():
            if len(values) != n_builds:
                logger.warning(f"  {method} has {len(values)} builds, expected {n_builds}")
                # Pad or truncate
                if len(values) < n_builds:
                    aligned_results[method] = np.pad(values, (0, n_builds - len(values)), constant_values=0.5)
                else:
                    aligned_results[method] = values[:n_builds]

        # Generate comparison table
        comparison_df = generate_comparison_table(
            aligned_results,
            reference_method='Filo-Priori',
            alpha=0.05,
            n_bootstrap=1000,
            apply_correction='holm'
        )

        # Print table
        print_comparison_table(comparison_df, title="Filo-Priori v9 vs Baselines")

        # Save comparison table
        comparison_path = self.output_dir / "comparison_table.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nComparison table saved to: {comparison_path}")

        # Generate LaTeX table
        latex_table = generate_latex_table(
            comparison_df,
            caption="Comparison of TCP Methods on QTA Dataset",
            label="tab:tcp_comparison"
        )

        latex_path = self.output_dir / "comparison_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to: {latex_path}")

        # Save all APFD results
        all_apfd_path = self.output_dir / "all_apfd_results.json"
        serializable_results = {k: v.tolist() for k, v in aligned_results.items()}
        with open(all_apfd_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"All APFD results saved to: {all_apfd_path}")

        return comparison_df

    def generate_visualizations(self):
        """Generate visualization plots."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 1. Box plot of APFD distributions
            fig, ax = plt.subplots(figsize=(12, 6))

            # Prepare data for plotting
            plot_data = []
            for method, values in self.apfd_results.items():
                for v in values:
                    plot_data.append({'Method': method, 'APFD': v})

            plot_df = pd.DataFrame(plot_data)

            # Order by mean APFD
            method_order = plot_df.groupby('Method')['APFD'].mean().sort_values(ascending=False).index

            sns.boxplot(data=plot_df, x='Method', y='APFD', order=method_order, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('APFD Distribution by Method')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
            ax.legend()

            plt.tight_layout()
            boxplot_path = self.output_dir / "apfd_boxplot.png"
            plt.savefig(boxplot_path, dpi=150)
            plt.close()
            logger.info(f"  Box plot saved to: {boxplot_path}")

            # 2. Bar chart with error bars
            fig, ax = plt.subplots(figsize=(10, 6))

            methods = []
            means = []
            ci_lowers = []
            ci_uppers = []

            for method in method_order:
                values = self.apfd_results[method]
                mean, ci_l, ci_u = bootstrap_confidence_interval(values, n_bootstrap=1000)
                methods.append(method)
                means.append(mean)
                ci_lowers.append(mean - ci_l)
                ci_uppers.append(ci_u - mean)

            x = np.arange(len(methods))
            colors = ['#2ecc71' if m == 'Filo-Priori' else '#3498db' for m in methods]

            ax.bar(x, means, yerr=[ci_lowers, ci_uppers], capsize=5, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel('Mean APFD')
            ax.set_title('Mean APFD with 95% Confidence Intervals')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')

            plt.tight_layout()
            barplot_path = self.output_dir / "apfd_barplot.png"
            plt.savefig(barplot_path, dpi=150)
            plt.close()
            logger.info(f"  Bar plot saved to: {barplot_path}")

            # 3. Improvement over Random
            fig, ax = plt.subplots(figsize=(10, 6))

            random_mean = np.mean(self.apfd_results.get('Random', [0.5]))
            improvements = [(m, (np.mean(v) - random_mean) / random_mean * 100)
                          for m, v in self.apfd_results.items() if m != 'Random']
            improvements.sort(key=lambda x: -x[1])

            methods_imp = [x[0] for x in improvements]
            values_imp = [x[1] for x in improvements]
            colors_imp = ['#2ecc71' if m == 'Filo-Priori' else '#3498db' for m in methods_imp]

            ax.barh(methods_imp, values_imp, color=colors_imp, alpha=0.8)
            ax.set_xlabel('Improvement over Random (%)')
            ax.set_title('APFD Improvement vs Random Baseline')
            ax.axvline(x=0, color='black', linewidth=0.5)

            # Add value labels
            for i, v in enumerate(values_imp):
                ax.text(v + 0.5, i, f'{v:.1f}%', va='center')

            plt.tight_layout()
            improvement_path = self.output_dir / "improvement_vs_random.png"
            plt.savefig(improvement_path, dpi=150)
            plt.close()
            logger.info(f"  Improvement plot saved to: {improvement_path}")

        except ImportError as e:
            logger.warning(f"Visualization skipped (matplotlib not available): {e}")

    def run_all(self, skip_lstm: bool = False):
        """Run complete baseline comparison pipeline."""
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info(" FILO-PRIORI V9 - BASELINE COMPARISON SUITE")
        logger.info("=" * 70)
        logger.info(f"Start time: {start_time}")
        logger.info(f"Output directory: {self.output_dir}")

        # Pipeline
        self.load_data()
        self.load_structural_features()
        self.load_filo_priori_results()
        self.run_heuristic_baselines()
        self.run_ml_baselines(skip_lstm=skip_lstm)
        comparison_df = self.generate_comparison_report()
        self.generate_visualizations()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "=" * 70)
        logger.info(" SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total methods compared: {len(self.apfd_results)}")
        logger.info(f"Total runtime: {duration}")
        logger.info(f"Results saved to: {self.output_dir}")

        # Print final ranking
        logger.info("\nFinal Ranking (by Mean APFD):")
        ranking = sorted(
            [(m, np.mean(v)) for m, v in self.apfd_results.items()],
            key=lambda x: -x[1]
        )
        for i, (method, mean_apfd) in enumerate(ranking, 1):
            marker = " <-- BEST" if i == 1 else ""
            logger.info(f"  {i}. {method}: {mean_apfd:.4f}{marker}")

        return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Run all TCP baselines and compare with Filo-Priori")
    parser.add_argument('--train-path', type=str, default='datasets/train.csv',
                       help='Path to training data')
    parser.add_argument('--test-path', type=str, default='datasets/test.csv',
                       help='Path to test data')
    parser.add_argument('--filo-priori-results', type=str,
                       default='results/experiment_06_feature_selection',
                       help='Path to Filo-Priori results')
    parser.add_argument('--output-dir', type=str, default='results/baselines',
                       help='Output directory')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM baseline (requires PyTorch)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    runner = BaselineRunner(
        train_path=args.train_path,
        test_path=args.test_path,
        filo_priori_results=args.filo_priori_results,
        output_dir=args.output_dir,
        seed=args.seed
    )

    runner.run_all(skip_lstm=args.skip_lstm)


if __name__ == "__main__":
    main()
