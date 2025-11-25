"""
Script para recalcular APFD com regra corrigida: count_tc=1 => APFD=1.0
"""
import pandas as pd
import numpy as np
import sys

def calculate_apfd_single_build_FIXED(ranks: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate APFD for a single build with FIXED rule: count_tc=1 => APFD=1.0
    """
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)

    n_tests = int(len(labels_arr))
    fail_indices = np.where(labels_arr.astype(int) != 0)[0]
    n_failures = len(fail_indices)

    # Business rule: if no failures, APFD is undefined (skip this build)
    if n_failures == 0:
        return None

    # CRITICAL FIX: if only 1 test case, APFD = 1.0
    # This MUST be checked BEFORE the formula calculation
    if n_tests == 1:
        print(f"  [FIX] count_tc=1 detected, returning APFD=1.0 (was 0.5)")
        return 1.0

    # Get ranks of failures
    failure_ranks = ranks_arr[fail_indices]

    # Calculate APFD
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) + 1.0 / float(2.0 * n_tests)

    return float(np.clip(apfd, 0.0, 1.0))


def recalculate_apfd_from_prioritized_csv(prioritized_csv: str, output_csv: str):
    """
    Recalculate APFD per build from prioritized test cases CSV
    """
    print(f"\n{'='*70}")
    print(f"RECALCULATING APFD WITH FIX: count_tc=1 => APFD=1.0")
    print(f"{'='*70}")
    print(f"Input:  {prioritized_csv}")
    print(f"Output: {output_csv}")

    # Load prioritized test cases
    df = pd.read_csv(prioritized_csv)
    print(f"\nLoaded {len(df)} test cases from {df['Build_ID'].nunique()} builds")

    results = []

    # Group by Build_ID
    grouped = df.groupby('Build_ID')

    builds_fixed = 0
    builds_unchanged = 0

    for build_id, build_df in grouped:
        # Count unique test cases
        if 'TC_Key' in build_df.columns:
            count_tc = build_df['TC_Key'].nunique()
        else:
            count_tc = len(build_df)

        # Only include builds with at least one "Fail" result
        if 'TE_Test_Result' in build_df.columns:
            fail_mask = (build_df['TE_Test_Result'].astype(str).str.strip() == "Fail")
            if not fail_mask.any():
                continue
        else:
            # Fallback: use label_binary
            if 'label_binary' not in build_df.columns:
                continue
            fail_mask = (build_df['label_binary'].astype(int) != 0)
            if not fail_mask.any():
                continue

        # CRITICAL FIX: For count_tc=1, directly return APFD=1.0
        # This bypasses the function entirely to ensure correctness
        if count_tc == 1:
            builds_fixed += 1
            apfd = 1.0
            print(f"[FIXING] Build {build_id}: count_tc={count_tc} -> APFD=1.0 (was 0.5)")
        else:
            builds_unchanged += 1
            # Get ranks and labels
            ranks = build_df['rank'].values if 'rank' in build_df.columns else None
            if ranks is None:
                continue

            labels = fail_mask.astype(int).values

            # Calculate APFD with FIXED function
            apfd = calculate_apfd_single_build_FIXED(ranks, labels)

        if apfd is None:
            continue

        # Count commits (placeholder)
        count_commits = 0

        # Add to results
        results.append({
            'method_name': 'dual_stream_gnn_exp_17_FULL_testcsv',
            'build_id': build_id,
            'test_scenario': 'full_test_csv_277_builds',
            'count_tc': count_tc,
            'count_commits': count_commits,
            'apfd': apfd,
            'time': 0.0
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('build_id').reset_index(drop=True)

    # Save
    results_df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print(f"RECALCULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total builds: {len(results_df)}")
    print(f"Builds fixed (count_tc=1): {builds_fixed}")
    print(f"Builds unchanged: {builds_unchanged}")
    print(f"\nMean APFD (OLD): <calculated from file>")
    print(f"Mean APFD (NEW): {results_df['apfd'].mean():.6f}")
    print(f"\nOutput saved to: {output_csv}")
    print(f"{'='*70}\n")

    return results_df


if __name__ == "__main__":
    # Recalculate for experiment_017_ranking_corrected_03
    prioritized_csv = "results/experiment_017_ranking_corrected_03/prioritized_test_cases_FULL_testcsv.csv"
    output_csv = "results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv_FIXED.csv"

    results_df = recalculate_apfd_from_prioritized_csv(prioritized_csv, output_csv)

    # Show comparison
    old_csv = "results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv.csv"
    old_df = pd.read_csv(old_csv)

    print(f"\n{'='*70}")
    print(f"COMPARISON: OLD vs NEW")
    print(f"{'='*70}")
    print(f"Mean APFD (OLD): {old_df['apfd'].mean():.6f}")
    print(f"Mean APFD (NEW): {results_df['apfd'].mean():.6f}")
    print(f"Difference:      {results_df['apfd'].mean() - old_df['apfd'].mean():.6f}")
    print(f"\nBuilds with APFD=1.0 (OLD): {(old_df['apfd']==1.0).sum()}")
    print(f"Builds with APFD=1.0 (NEW): {(results_df['apfd']==1.0).sum()}")
    print(f"{'='*70}\n")
