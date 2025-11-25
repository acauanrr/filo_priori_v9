"""
Validation Script for Structural Feature Extractor

This script validates the structural feature extraction module by:
1. Loading a sample of training and test data
2. Extracting structural features
3. Validating feature ranges and distributions
4. Showing detailed examples of computed features

Usage:
    python scripts/validate_structural_features.py --sample-size 10000

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.structural_feature_extractor import (
    StructuralFeatureExtractor,
    extract_structural_features
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_feature_ranges(feature_matrix: np.ndarray,
                            feature_names: list,
                            label: str = "Dataset") -> bool:
    """
    Validate that feature ranges are within expected bounds.

    Returns:
        True if all validations pass
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATING FEATURE RANGES: {label}")
    logger.info(f"{'='*70}")

    all_valid = True

    expected_ranges = {
        'test_age': (0, np.inf),  # Age should be non-negative
        'failure_rate': (0, 1),  # Rate should be between 0 and 1
        'recent_failure_rate': (0, 1),
        'flakiness_rate': (0, 1),
        'commit_count': (1, np.inf),  # At least 1 commit
        'test_novelty': (0, 1)  # Binary flag
    }

    for i, name in enumerate(feature_names):
        col = feature_matrix[:, i]
        min_val, max_val = col.min(), col.max()
        expected_min, expected_max = expected_ranges[name]

        valid = (min_val >= expected_min and max_val <= expected_max)
        status = "✓" if valid else "✗"

        logger.info(f"{status} {name:20s}: [{min_val:8.3f}, {max_val:8.3f}] "
                   f"(expected: [{expected_min}, {expected_max}])")

        if not valid:
            all_valid = False
            logger.warning(f"  WARNING: {name} outside expected range!")

    if all_valid:
        logger.info(f"\n✓ All features within expected ranges")
    else:
        logger.error(f"\n✗ Some features outside expected ranges!")

    return all_valid


def show_feature_statistics(feature_matrix: np.ndarray,
                            feature_names: list,
                            label: str = "Dataset") -> None:
    """
    Show detailed statistics for each feature.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE STATISTICS: {label}")
    logger.info(f"{'='*70}")

    stats_data = []
    for i, name in enumerate(feature_names):
        col = feature_matrix[:, i]
        stats_data.append({
            'Feature': name,
            'Mean': f"{col.mean():.4f}",
            'Std': f"{col.std():.4f}",
            'Min': f"{col.min():.4f}",
            '25%': f"{np.percentile(col, 25):.4f}",
            'Median': f"{np.median(col):.4f}",
            '75%': f"{np.percentile(col, 75):.4f}",
            'Max': f"{col.max():.4f}",
            'Non-Zero': f"{(col != 0).sum()} ({100*(col != 0).mean():.1f}%)"
        })

    stats_df = pd.DataFrame(stats_data)
    logger.info("\n" + stats_df.to_string(index=False))


def show_sample_cases(df: pd.DataFrame,
                     feature_matrix: np.ndarray,
                     feature_names: list,
                     n_samples: int = 10) -> None:
    """
    Show detailed examples of test cases with their features.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"SAMPLE CASES (n={n_samples})")
    logger.info(f"{'='*70}")

    # Select diverse samples
    indices = []

    # Get some new tests
    if 'test_novelty' in feature_names:
        novelty_idx = feature_names.index('test_novelty')
        new_test_indices = np.where(feature_matrix[:, novelty_idx] == 1.0)[0]
        if len(new_test_indices) > 0:
            indices.extend(new_test_indices[:3])

    # Get some old tests with high failure rate
    if 'failure_rate' in feature_names and 'test_age' in feature_names:
        failure_idx = feature_names.index('failure_rate')
        age_idx = feature_names.index('test_age')
        old_failing = np.where((feature_matrix[:, age_idx] > 10) &
                               (feature_matrix[:, failure_idx] > 0.5))[0]
        if len(old_failing) > 0:
            indices.extend(old_failing[:3])

    # Get some flaky tests
    if 'flakiness_rate' in feature_names:
        flaky_idx = feature_names.index('flakiness_rate')
        flaky_tests = np.where(feature_matrix[:, flaky_idx] > 0.3)[0]
        if len(flaky_tests) > 0:
            indices.extend(flaky_tests[:2])

    # Fill remaining with random
    remaining = n_samples - len(indices)
    if remaining > 0:
        random_indices = np.random.choice(len(df), size=remaining, replace=False)
        indices.extend(random_indices)

    indices = indices[:n_samples]

    # Show each case
    for idx in indices:
        row = df.iloc[idx]
        features = feature_matrix[idx]

        logger.info(f"\n--- Sample {idx} ---")
        logger.info(f"TC_Key: {row['TC_Key']}")
        logger.info(f"Build_ID: {row['Build_ID']}")
        logger.info(f"Result: {row['TE_Test_Result']}")

        if 'TE_Summary' in row:
            summary = str(row['TE_Summary'])[:80]
            logger.info(f"Summary: {summary}...")

        logger.info(f"\nExtracted Features:")
        for name, value in zip(feature_names, features):
            logger.info(f"  {name:20s}: {value:.4f}")


def compare_train_test_distributions(train_features: np.ndarray,
                                     test_features: np.ndarray,
                                     feature_names: list) -> None:
    """
    Compare feature distributions between train and test sets.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAIN vs TEST DISTRIBUTION COMPARISON")
    logger.info(f"{'='*70}")

    for i, name in enumerate(feature_names):
        train_col = train_features[:, i]
        test_col = test_features[:, i]

        train_mean = train_col.mean()
        test_mean = test_col.mean()
        diff = test_mean - train_mean
        pct_diff = (diff / train_mean * 100) if train_mean != 0 else 0

        logger.info(f"\n{name}:")
        logger.info(f"  Train: mean={train_mean:.4f}, std={train_col.std():.4f}")
        logger.info(f"  Test:  mean={test_mean:.4f}, std={test_col.std():.4f}")
        logger.info(f"  Diff:  {diff:+.4f} ({pct_diff:+.1f}%)")


def validate_business_logic(df: pd.DataFrame,
                            feature_matrix: np.ndarray,
                            feature_names: list) -> bool:
    """
    Validate business logic rules.

    Returns:
        True if all validations pass
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATING BUSINESS LOGIC")
    logger.info(f"{'='*70}")

    all_valid = True

    # Rule 1: Test novelty should be 1.0 only for first occurrence in build
    if 'test_novelty' in feature_names:
        novelty_idx = feature_names.index('test_novelty')
        novel_tests = feature_matrix[:, novelty_idx] == 1.0

        logger.info(f"\nRule 1: Test Novelty")
        logger.info(f"  Novel tests: {novel_tests.sum()} ({100*novel_tests.mean():.1f}%)")

        # Check if any TC_Key has multiple novel flags (should not happen)
        novel_tc_keys = df[novel_tests]['TC_Key'].value_counts()
        duplicate_novels = (novel_tc_keys > 1).sum()

        if duplicate_novels > 0:
            logger.error(f"  ✗ Found {duplicate_novels} TC_Keys with multiple novel flags!")
            all_valid = False
        else:
            logger.info(f"  ✓ No duplicate novel flags")

    # Rule 2: Failure rate should be 0 for new tests (no history)
    if 'failure_rate' in feature_names and 'test_age' in feature_names:
        failure_idx = feature_names.index('failure_rate')
        age_idx = feature_names.index('test_age')

        new_tests = feature_matrix[:, age_idx] == 0
        new_test_failures = feature_matrix[new_tests, failure_idx]

        logger.info(f"\nRule 2: New Tests Should Have Zero Failure Rate")
        logger.info(f"  New tests: {new_tests.sum()}")
        logger.info(f"  New tests with failure_rate=0: {(new_test_failures == 0).sum()}")

        if not np.all(new_test_failures == 0):
            logger.warning(f"  ⚠ Some new tests have non-zero failure rate!")
            logger.warning(f"    Max failure rate for new tests: {new_test_failures.max():.4f}")
            # This is a warning, not an error (might be due to data issues)

    # Rule 3: Commit count should be at least 1
    if 'commit_count' in feature_names:
        commit_idx = feature_names.index('commit_count')
        commit_counts = feature_matrix[:, commit_idx]

        logger.info(f"\nRule 3: Commit Count Should Be >= 1")
        logger.info(f"  Min commit count: {commit_counts.min():.0f}")

        if commit_counts.min() < 1:
            logger.error(f"  ✗ Found samples with commit_count < 1!")
            all_valid = False
        else:
            logger.info(f"  ✓ All samples have commit_count >= 1")

    if all_valid:
        logger.info(f"\n✓ All business logic rules validated")
    else:
        logger.error(f"\n✗ Some business logic rules failed!")

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description='Validate structural feature extraction'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Number of samples to load from each dataset (default: 10000, use -1 for all)'
    )
    parser.add_argument(
        '--recent-window',
        type=int,
        default=5,
        help='Window size for recent failure rate (default: 5)'
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=10,
        help='Number of example cases to show (default: 10)'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("STRUCTURAL FEATURE EXTRACTOR VALIDATION")
    logger.info("="*70)
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Recent window: {args.recent_window}")
    logger.info("")

    # Load data
    logger.info("Loading datasets...")
    df_train = pd.read_csv('datasets/train.csv')
    df_test = pd.read_csv('datasets/test.csv')

    logger.info(f"  Train: {len(df_train)} rows")
    logger.info(f"  Test: {len(df_test)} rows")

    # Sample if requested
    if args.sample_size > 0 and args.sample_size < len(df_train):
        logger.info(f"\nSampling {args.sample_size} rows from each dataset...")
        df_train = df_train.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(n=min(args.sample_size, len(df_test)),
                                 random_state=42).reset_index(drop=True)

    # Extract features
    logger.info("\nExtracting structural features...")
    train_features, _, test_features = extract_structural_features(
        df_train,
        df_val=None,
        df_test=df_test,
        recent_window=args.recent_window,
        cache_path=None  # Don't cache during validation
    )

    # Get feature names
    extractor = StructuralFeatureExtractor(recent_window=args.recent_window)
    feature_names = extractor.get_feature_names()

    # Validate feature ranges
    train_valid = validate_feature_ranges(train_features, feature_names, "TRAIN")
    test_valid = validate_feature_ranges(test_features, feature_names, "TEST")

    # Show statistics
    show_feature_statistics(train_features, feature_names, "TRAIN")
    show_feature_statistics(test_features, feature_names, "TEST")

    # Compare distributions
    compare_train_test_distributions(train_features, test_features, feature_names)

    # Show sample cases
    show_sample_cases(df_train, train_features, feature_names, n_samples=args.n_examples)

    # Validate business logic
    logic_valid = validate_business_logic(df_train, train_features, feature_names)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)

    all_valid = train_valid and test_valid and logic_valid

    if all_valid:
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("\nThe structural feature extractor is working correctly!")
        logger.info("\nNext steps:")
        logger.info("  1. Integrate extractor into main.py pipeline")
        logger.info("  2. Modify Structural Stream to accept these features")
        logger.info("  3. Train and evaluate V8 model")
        return 0
    else:
        logger.error("✗ SOME VALIDATIONS FAILED")
        logger.error("\nPlease review the errors above before proceeding.")
        return 1


if __name__ == '__main__':
    import os
    os.chdir('/home/acauan/ufam/iats/sprint_07/filo_priori_v8')
    sys.exit(main())
