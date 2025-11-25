#!/usr/bin/env python3
"""
Test Temporal Leakage Impact
=============================

This script tests if using "all-time" statistics vs "past-only" statistics
makes a significant difference in model performance.

Compares:
1. Current approach: features computed from ALL training data
2. Temporal approach: features computed from PAST data only

Author: Filo-Priori Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_temporal_features(df_train, tc_key, current_build_idx, build_chronology):
    """
    Compute features using ONLY past builds (temporal correctness).

    Args:
        df_train: Training DataFrame
        tc_key: Test case key
        current_build_idx: Index of current build in chronology
        build_chronology: List of builds in chronological order

    Returns:
        dict: Features computed from past only
    """
    # Get only PAST executions
    past_builds = build_chronology[:current_build_idx]
    past_mask = (df_train['TC_Key'] == tc_key) & (df_train['Build_ID'].isin(past_builds))
    past_executions = df_train[past_mask]

    if len(past_executions) == 0:
        return {
            'past_failure_rate': 0.0,
            'past_execution_count': 0,
            'past_last_result': None
        }

    results = past_executions['TE_Test_Result'].values

    return {
        'past_failure_rate': (results == 'Fail').mean(),
        'past_execution_count': len(results),
        'past_last_result': results[-1] if len(results) > 0 else None
    }


def compute_alltime_features(df_train, tc_key):
    """
    Compute features using ALL training data (current approach).

    Args:
        df_train: Training DataFrame
        tc_key: Test case key

    Returns:
        dict: Features computed from all time
    """
    tc_executions = df_train[df_train['TC_Key'] == tc_key]

    if len(tc_executions) == 0:
        return {
            'alltime_failure_rate': 0.0,
            'alltime_execution_count': 0,
            'alltime_last_result': None
        }

    results = tc_executions['TE_Test_Result'].values

    return {
        'alltime_failure_rate': (results == 'Fail').mean(),
        'alltime_execution_count': len(results),
        'alltime_last_result': results[-1] if len(results) > 0 else None
    }


def analyze_leakage_impact(df_train, build_chronology):
    """
    Analyze the impact of temporal leakage by comparing features.

    Computes:
    1. How different are past-only vs all-time features?
    2. What % of cases would have different features?
    3. Is the information gain from future data significant?
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYZING TEMPORAL LEAKAGE IMPACT")
    logger.info("="*80)

    build_to_idx = {build_id: idx for idx, build_id in enumerate(build_chronology)}

    # Sample analysis on subset to avoid long runtime
    sample_size = min(10000, len(df_train))
    df_sample = df_train.sample(n=sample_size, random_state=42)

    logger.info(f"\nAnalyzing {sample_size} sample executions...")

    differences = []
    significant_differences = 0

    for idx, row in df_sample.iterrows():
        tc_key = row['TC_Key']
        build_id = row['Build_ID']

        if build_id not in build_to_idx:
            continue

        current_build_idx = build_to_idx[build_id]

        # Compute both types of features
        past_features = compute_temporal_features(
            df_train, tc_key, current_build_idx, build_chronology
        )
        alltime_features = compute_alltime_features(df_train, tc_key)

        # Compare failure rates
        past_fr = past_features['past_failure_rate']
        alltime_fr = alltime_features['alltime_failure_rate']

        diff = abs(alltime_fr - past_fr)
        differences.append(diff)

        # Significant if difference > 0.1 (10%)
        if diff > 0.1:
            significant_differences += 1

    # Analysis
    differences = np.array(differences)

    logger.info(f"\n📊 LEAKAGE IMPACT ANALYSIS:")
    logger.info(f"   Samples analyzed: {len(differences):,}")
    logger.info(f"   Mean difference (failure_rate): {differences.mean():.4f}")
    logger.info(f"   Median difference: {np.median(differences):.4f}")
    logger.info(f"   Max difference: {differences.max():.4f}")
    logger.info(f"   Std difference: {differences.std():.4f}")

    pct_significant = (significant_differences / len(differences)) * 100
    logger.info(f"\n   Significant differences (>10%): {significant_differences} ({pct_significant:.1f}%)")

    # Interpretation
    logger.info(f"\n💡 INTERPRETATION:")

    if differences.mean() < 0.05:
        logger.info(f"   ✅ LOW IMPACT: Average difference is {differences.mean():.4f}")
        logger.info(f"      The temporal leakage appears to be MINIMAL")
        logger.info(f"      Features are relatively stable across time")
        logger.info(f"      → Evaluation results are likely VALID")
    elif differences.mean() < 0.15:
        logger.info(f"   ⚠️  MODERATE IMPACT: Average difference is {differences.mean():.4f}")
        logger.info(f"      The temporal leakage is MODERATE")
        logger.info(f"      Features change somewhat with future data")
        logger.info(f"      → Evaluation results may be SLIGHTLY OPTIMISTIC")
    else:
        logger.info(f"   ❌ HIGH IMPACT: Average difference is {differences.mean():.4f}")
        logger.info(f"      The temporal leakage is SIGNIFICANT")
        logger.info(f"      Features are very different with future data")
        logger.info(f"      → Evaluation results are likely INFLATED")

    if pct_significant > 30:
        logger.info(f"\n   ⚠️  {pct_significant:.1f}% of cases have >10% difference")
        logger.info(f"      This is concerning - many cases affected by future data")
    else:
        logger.info(f"\n   ✅ Only {pct_significant:.1f}% of cases have >10% difference")
        logger.info(f"      Most cases have similar features with/without future data")

    return differences


def main():
    """Main analysis pipeline."""
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL LEAKAGE IMPACT TEST")
    logger.info("="*80)

    # Load training data
    train_path = Path("datasets/train.csv")

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return

    logger.info(f"\n📂 Loading {train_path}...")
    df_train = pd.read_csv(train_path)
    logger.info(f"   Loaded {len(df_train):,} training samples")

    # Establish chronology
    if 'Build_Test_Start_Date' in df_train.columns:
        build_dates = df_train.groupby('Build_ID')['Build_Test_Start_Date'].first().sort_values()
        build_chronology = build_dates.index.tolist()
        logger.info(f"   Chronology established using Build_Test_Start_Date")
    else:
        build_chronology = df_train['Build_ID'].unique().tolist()
        logger.info(f"   Using order of appearance (no timestamp available)")

    logger.info(f"   Chronology spans {len(build_chronology)} builds")

    # Analyze leakage impact
    differences = analyze_leakage_impact(df_train, build_chronology)

    # Final recommendation
    logger.info(f"\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)

    if differences.mean() < 0.05:
        logger.info("\n✅ EVALUATION APPEARS VALID")
        logger.info("   The current approach (all-time features) has minimal")
        logger.info("   difference from temporal features. Your APFD=0.6171")
        logger.info("   is likely a fair estimate of real performance.")
    else:
        logger.info("\n⚠️  CAUTION RECOMMENDED")
        logger.info("   The current approach may be optimistic. Consider:")
        logger.info("   1. Re-implementing features with temporal correctness")
        logger.info("   2. Re-evaluating model performance")
        logger.info("   3. Reporting this limitation in your thesis")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
