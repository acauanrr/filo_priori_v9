#!/usr/bin/env python3
"""
Verification Script: Data Leakage Detection
============================================

This script verifies that there is NO data leakage between train and test sets.

Checks:
1. Temporal separation: Test builds come AFTER train builds
2. No overlap: Same (TC_Key, Build_ID) pairs don't appear in both sets
3. Feature computation: Structural features use only PAST data
4. Graph construction: Graph built ONLY from train data

Author: Filo-Priori Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def check_temporal_separation(df_train, df_test):
    """
    Check if test builds come chronologically AFTER train builds.

    Returns:
        bool: True if temporal separation is correct, False otherwise
    """
    print("\n" + "="*80)
    print("CHECK 1: TEMPORAL SEPARATION")
    print("="*80)

    if 'Build_ID' not in df_train.columns or 'Build_ID' not in df_test.columns:
        print("⚠️  WARNING: Build_ID column not found!")
        return False

    max_train_build = df_train['Build_ID'].max()
    min_test_build = df_test['Build_ID'].min()

    print(f"\n📊 Build Statistics:")
    print(f"   Train builds: {df_train['Build_ID'].min()} to {max_train_build}")
    print(f"   Test builds:  {min_test_build} to {df_test['Build_ID'].max()}")

    if min_test_build > max_train_build:
        print(f"\n✅ PASS: Temporal separation is CORRECT")
        print(f"   All test builds ({min_test_build}+) come AFTER train builds (≤{max_train_build})")
        return True
    else:
        print(f"\n❌ FAIL: Temporal separation is INCORRECT")
        print(f"   Test builds ({min_test_build}) overlap with train builds ({max_train_build})")
        print(f"   ⚠️  This could cause DATA LEAKAGE!")
        return False


def check_no_overlap(df_train, df_test):
    """
    Check that no (TC_Key, Build_ID) pair appears in both train and test.

    Returns:
        bool: True if no overlap, False otherwise
    """
    print("\n" + "="*80)
    print("CHECK 2: NO EXECUTION OVERLAP")
    print("="*80)

    # Create unique identifiers for each execution
    train_executions = set(zip(df_train['TC_Key'], df_train['Build_ID']))
    test_executions = set(zip(df_test['TC_Key'], df_test['Build_ID']))

    overlap = train_executions & test_executions

    print(f"\n📊 Execution Statistics:")
    print(f"   Train executions: {len(train_executions):,}")
    print(f"   Test executions:  {len(test_executions):,}")
    print(f"   Overlap:          {len(overlap):,}")

    if len(overlap) == 0:
        print(f"\n✅ PASS: No execution overlap")
        print(f"   No (TC_Key, Build_ID) pair appears in both train and test")
        return True
    else:
        print(f"\n❌ FAIL: Found {len(overlap)} overlapping executions")
        print(f"   ⚠️  CRITICAL DATA LEAKAGE DETECTED!")
        print(f"\n   First 5 overlapping executions:")
        for i, (tc_key, build_id) in enumerate(list(overlap)[:5]):
            print(f"      {i+1}. TC_Key={tc_key}, Build_ID={build_id}")
        return False


def check_known_vs_orphan(df_train, df_test):
    """
    Analyze known vs orphan test cases in test set.

    Returns:
        dict: Statistics about known/orphan distribution
    """
    print("\n" + "="*80)
    print("CHECK 3: KNOWN vs ORPHAN TEST CASES")
    print("="*80)

    train_tc_keys = set(df_train['TC_Key'].unique())
    test_tc_keys = set(df_test['TC_Key'].unique())

    known_tc_keys = test_tc_keys & train_tc_keys
    orphan_tc_keys = test_tc_keys - train_tc_keys

    # Count executions for each category
    test_known_mask = df_test['TC_Key'].isin(known_tc_keys)
    num_known_executions = test_known_mask.sum()
    num_orphan_executions = (~test_known_mask).sum()

    total_test_executions = len(df_test)
    pct_known = (num_known_executions / total_test_executions) * 100
    pct_orphan = (num_orphan_executions / total_test_executions) * 100

    print(f"\n📊 Test Case Statistics:")
    print(f"   Train unique TC_Keys: {len(train_tc_keys):,}")
    print(f"   Test unique TC_Keys:  {len(test_tc_keys):,}")
    print(f"   Known TC_Keys:        {len(known_tc_keys):,}")
    print(f"   Orphan TC_Keys:       {len(orphan_tc_keys):,}")

    print(f"\n📊 Test Execution Statistics:")
    print(f"   Total test executions:  {total_test_executions:,}")
    print(f"   Known executions:       {num_known_executions:,} ({pct_known:.1f}%)")
    print(f"   Orphan executions:      {num_orphan_executions:,} ({pct_orphan:.1f}%)")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if pct_known > 70:
        print(f"   ✅ {pct_known:.1f}% of test executions are from KNOWN test cases")
        print(f"      These test cases existed in training, but executions are NEW")
        print(f"      Model uses learned patterns (NOT memorization)")
        print(f"      This is VALID if temporal separation is correct (CHECK 1)")
    else:
        print(f"   ⚠️  Only {pct_known:.1f}% of test executions are from known test cases")
        print(f"      This is unusual - most datasets have >70% known cases")

    print(f"\n   ℹ️  {pct_orphan:.1f}% of test executions are from ORPHAN test cases")
    print(f"      These test cases NEVER appeared in training")
    print(f"      Model has NO historical information about them")
    print(f"      These test the model's ability to generalize to completely new tests")

    return {
        'train_tc_keys': len(train_tc_keys),
        'test_tc_keys': len(test_tc_keys),
        'known_tc_keys': len(known_tc_keys),
        'orphan_tc_keys': len(orphan_tc_keys),
        'known_executions': num_known_executions,
        'orphan_executions': num_orphan_executions,
        'pct_known': pct_known,
        'pct_orphan': pct_orphan
    }


def check_feature_leakage_risk(df_train, df_test):
    """
    Check if there's a risk of feature computation leakage.

    This is a heuristic check - manual code review is still needed.
    """
    print("\n" + "="*80)
    print("CHECK 4: FEATURE COMPUTATION LEAKAGE RISK")
    print("="*80)

    print("\n⚠️  MANUAL VERIFICATION REQUIRED:")
    print("\n1. Check StructuralFeatureExtractor:")
    print("   File: src/preprocessing/structural_feature_extractor*.py")
    print("   Verify: Features use ONLY past executions (Build_ID < current_build)")
    print("   Example CORRECT code:")
    print("     past_mask = (df['TC_Key'] == tc_key) & (df['Build_ID'] < current_build)")
    print("     failure_rate = past_executions[past_mask]['TE_Test_Result'].eq('Fail').mean()")

    print("\n2. Check Graph Construction:")
    print("   File: src/phylogenetic/multi_edge_graph_builder.py")
    print("   Verify: Graph built ONLY from df_train (not df_train + df_test)")
    print("   Example CORRECT code:")
    print("     graph = build_multi_edge_graph(df_train)  # ONLY train data")

    print("\n3. Check Main Pipeline:")
    print("   File: main.py")
    print("   Verify: Test features computed INDEPENDENTLY from training")
    print("   No information from test set should leak into train features")

    print("\n📝 Recommendation:")
    print("   Run: grep -n 'Build_ID' src/preprocessing/structural_feature_extractor*.py")
    print("   Look for: Filters that exclude current build")


def main():
    """Main verification pipeline."""
    print("\n" + "="*80)
    print("DATA LEAKAGE VERIFICATION SCRIPT")
    print("="*80)
    print("\nThis script checks for data leakage between train and test sets.")
    print("It performs 4 critical checks to ensure evaluation validity.")

    # Load datasets
    print("\n📂 Loading datasets...")

    # Try to load from cache or raw data
    train_path = Path("datasets/train.csv")
    test_path = Path("datasets/test.csv")

    if not train_path.exists() or not test_path.exists():
        print(f"\n❌ ERROR: Dataset files not found!")
        print(f"   Expected: {train_path.absolute()}")
        print(f"   Expected: {test_path.absolute()}")
        print(f"\n   Please ensure train.csv and test.csv are in the datasets/ directory")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"   ✅ Loaded train.csv: {len(df_train):,} rows")
    print(f"   ✅ Loaded test.csv:  {len(df_test):,} rows")

    # Run checks
    results = {}

    results['temporal_separation'] = check_temporal_separation(df_train, df_test)
    results['no_overlap'] = check_no_overlap(df_train, df_test)
    results['known_orphan_stats'] = check_known_vs_orphan(df_train, df_test)
    check_feature_leakage_risk(df_train, df_test)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_passed = results['temporal_separation'] and results['no_overlap']

    if all_passed:
        print("\n✅ ALL AUTOMATIC CHECKS PASSED!")
        print("\n   Your train/test split appears to be VALID:")
        print("   • Temporal separation: CORRECT")
        print("   • No execution overlap: CORRECT")
        print(f"   • Known test cases: {results['known_orphan_stats']['pct_known']:.1f}%")
        print(f"   • Orphan test cases: {results['known_orphan_stats']['pct_orphan']:.1f}%")

        print("\n⚠️  MANUAL VERIFICATION STILL REQUIRED:")
        print("   • Check structural feature computation (CHECK 4)")
        print("   • Review graph construction code")
        print("   • Ensure no test data leaks into training")
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("\n   Issues detected:")
        if not results['temporal_separation']:
            print("   ❌ Temporal separation: FAILED")
        if not results['no_overlap']:
            print("   ❌ Execution overlap: FAILED")

        print("\n   ⚠️  CRITICAL: Your evaluation may be INVALID due to data leakage!")
        print("   Please review your train/test split methodology.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
