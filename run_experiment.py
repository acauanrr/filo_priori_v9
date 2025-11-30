#!/usr/bin/env python3
"""
Filo-Priori Unified Experiment Runner.

This script automatically selects the appropriate model and configuration
based on the target dataset:

- 01_industry: V9 Pipeline (SBERT + GAT + Focal Loss) - APFD ~0.64
- 02_rtptorrent: V10 Pipeline (CodeBERT-style + LambdaRank) - Ranking optimized

Usage:
    python run_experiment.py --dataset industry
    python run_experiment.py --dataset rtptorrent
    python run_experiment.py --dataset all

Author: Filo-Priori Team
Version: 2.0.0
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def run_industry_experiment():
    """
    Run experiment on 01_industry dataset using V9 pipeline.

    V9 Configuration:
    - SBERT embeddings (all-mpnet-base-v2)
    - GAT (Graph Attention Network)
    - 10 structural features (v2.5)
    - Weighted Focal + Ranking Loss
    - Expected APFD: ~0.64
    """
    logger.info("=" * 70)
    logger.info("RUNNING V9 PIPELINE FOR 01_INDUSTRY DATASET")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Model: V9 (Dual-Stream SBERT + GAT)")
    logger.info("  - Embeddings: sentence-transformers/all-mpnet-base-v2")
    logger.info("  - Features: 10 structural features (v2.5)")
    logger.info("  - Loss: Weighted Focal + Ranking Loss")
    logger.info("  - Expected APFD: ~0.64")
    logger.info("")

    config_path = BASE_DIR / "configs" / "experiment_industry.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    cmd = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--config", str(config_path)
    ]

    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info("")

    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running industry experiment: {e}")
        return False


def run_rtptorrent_experiment():
    """
    Run experiment on 02_rtptorrent dataset using V10 pipeline.

    V10 Configuration:
    - Advanced semantic features (ranking-focused)
    - LambdaRank loss optimization
    - Time-decay weighted features
    - Residual learning with heuristic bias
    - Expected APFD: ~0.68+
    """
    logger.info("=" * 70)
    logger.info("RUNNING V10 PIPELINE FOR 02_RTPTORRENT DATASET")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Model: V10 Full (LightGBM LambdaRank + Heuristic Ensemble)")
    logger.info("  - Features: 16 ranking-optimized features")
    logger.info("  - Loss: LambdaRank (NDCG optimization)")
    logger.info("  - Ensemble: ML (70%) + Heuristic (30%)")
    logger.info("  - Evaluation: All 20 projects, comparison with 7 baselines")
    logger.info("")

    # Use the V10 RTPTorrent full multi-project script
    script_path = BASE_DIR / "run_v10_rtptorrent_full.py"

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        logger.info("Creating V10 RTPTorrent script...")
        # Will be created below
        return False

    cmd = [sys.executable, str(script_path)]

    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info("")

    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running rtptorrent experiment: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Filo-Priori Unified Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment.py --dataset industry     # Run V9 on industrial data
    python run_experiment.py --dataset rtptorrent   # Run V10 on RTPTorrent
    python run_experiment.py --dataset all          # Run both experiments

Dataset-Model Mapping:
    01_industry  -> V9 Pipeline (SBERT + GAT + Focal Loss)
    02_rtptorrent -> V10 Pipeline (LambdaRank + Residual Learning)
        """
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['industry', 'rtptorrent', 'all'],
        required=True,
        help='Dataset to run experiment on'
    )

    args = parser.parse_args()

    results = {}

    if args.dataset in ['industry', 'all']:
        success = run_industry_experiment()
        results['industry'] = 'SUCCESS' if success else 'FAILED'

    if args.dataset in ['rtptorrent', 'all']:
        success = run_rtptorrent_experiment()
        results['rtptorrent'] = 'SUCCESS' if success else 'FAILED'

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    for dataset, status in results.items():
        logger.info(f"  {dataset}: {status}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
