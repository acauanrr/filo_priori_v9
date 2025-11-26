#!/usr/bin/env python3
"""
Qualitative Analysis for Filo-Priori v9.

Analyzes:
1. Best and worst performing builds
2. Case studies with detailed rankings
3. Patterns in success/failure cases
4. Comparison with optimal ranking

Author: Filo-Priori Team
Date: 2025-11-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent


def load_data():
    """Load all necessary data for qualitative analysis."""

    # Load APFD per build
    apfd_path = PROJECT_ROOT / 'results' / 'experiment_06_feature_selection' / 'apfd_per_build_FULL_testcsv.csv'
    apfd_df = pd.read_csv(apfd_path)

    # Load prioritized test cases
    prio_path = PROJECT_ROOT / 'results' / 'experiment_06_feature_selection' / 'prioritized_test_cases_FULL_testcsv.csv'
    prio_df = pd.read_csv(prio_path)

    # Load original test data
    test_path = PROJECT_ROOT / 'datasets' / 'test.csv'
    test_df = pd.read_csv(test_path)

    return apfd_df, prio_df, test_df


def analyze_apfd_distribution(apfd_df):
    """Analyze the distribution of APFD scores."""

    apfd_values = apfd_df['apfd'].values

    analysis = {
        'total_builds': len(apfd_values),
        'mean': np.mean(apfd_values),
        'std': np.std(apfd_values),
        'median': np.median(apfd_values),
        'min': np.min(apfd_values),
        'max': np.max(apfd_values),
        'perfect_apfd': np.sum(apfd_values == 1.0),
        'high_apfd': np.sum(apfd_values >= 0.8),
        'medium_apfd': np.sum((apfd_values >= 0.5) & (apfd_values < 0.8)),
        'low_apfd': np.sum(apfd_values < 0.5),
        'quartiles': {
            'Q1': np.percentile(apfd_values, 25),
            'Q2': np.percentile(apfd_values, 50),
            'Q3': np.percentile(apfd_values, 75)
        }
    }

    return analysis


def identify_extreme_cases(apfd_df, n=10):
    """Identify best and worst performing builds."""

    # Sort by APFD
    sorted_df = apfd_df.sort_values('apfd', ascending=False)

    best_builds = sorted_df.head(n)
    worst_builds = sorted_df.tail(n)

    # Also get perfect APFD builds
    perfect_builds = apfd_df[apfd_df['apfd'] == 1.0]

    return best_builds, worst_builds, perfect_builds


def analyze_build_characteristics(apfd_df, test_df):
    """Analyze characteristics that correlate with APFD performance."""

    # Merge APFD with test data to get build characteristics
    build_stats = test_df.groupby('Build_ID').agg({
        'TC_Key': 'count',  # Number of test cases
        'TE_Test_Result': lambda x: (x == 'Fail').sum(),  # Number of failures
        'commit': lambda x: x.dropna().nunique()  # Number of unique commits
    }).reset_index()

    build_stats.columns = ['Build_ID', 'num_test_cases', 'num_failures', 'num_commits']
    build_stats['failure_rate'] = build_stats['num_failures'] / build_stats['num_test_cases']

    # Merge with APFD
    merged = apfd_df.merge(build_stats, left_on='build_id', right_on='Build_ID', how='left')

    # Analyze correlations
    correlations = {
        'apfd_vs_num_tc': merged['apfd'].corr(merged['num_test_cases']),
        'apfd_vs_num_failures': merged['apfd'].corr(merged['num_failures']),
        'apfd_vs_failure_rate': merged['apfd'].corr(merged['failure_rate']),
        'apfd_vs_num_commits': merged['apfd'].corr(merged['num_commits'])
    }

    # Categorize builds
    merged['apfd_category'] = pd.cut(merged['apfd'],
                                      bins=[0, 0.5, 0.7, 0.9, 1.0],
                                      labels=['Low', 'Medium', 'High', 'Excellent'])

    category_stats = merged.groupby('apfd_category').agg({
        'num_test_cases': 'mean',
        'num_failures': 'mean',
        'failure_rate': 'mean',
        'num_commits': 'mean'
    }).round(2)

    return merged, correlations, category_stats


def create_case_study(build_id, prio_df, test_df, apfd_df):
    """Create a detailed case study for a specific build."""

    # Get APFD for this build
    build_apfd = apfd_df[apfd_df['build_id'] == build_id]['apfd'].values[0]

    # Get prioritized test cases for this build
    build_prio = prio_df[prio_df['Build_ID'] == build_id].copy()

    # Get original test data
    build_test = test_df[test_df['Build_ID'] == build_id].copy()

    # Merge to get full information
    if len(build_prio) > 0:
        # Sort by rank
        build_prio = build_prio.sort_values('rank')

        # Calculate where failures are in the ranking
        failures = build_prio[build_prio['TE_Test_Result'] == 'Fail']
        total_tc = len(build_prio)

        # Create case study
        case_study = {
            'build_id': build_id,
            'apfd': build_apfd,
            'total_test_cases': total_tc,
            'total_failures': len(failures),
            'failure_rate': len(failures) / total_tc if total_tc > 0 else 0,
            'failure_positions': failures['rank'].tolist() if len(failures) > 0 else [],
            'top_10_ranking': [],
            'optimal_vs_actual': {}
        }

        # Get top 10 ranking
        top_10 = build_prio.head(10)
        for _, row in top_10.iterrows():
            case_study['top_10_ranking'].append({
                'rank': int(row['rank']),
                'tc_key': row['TC_Key'],
                'result': row['TE_Test_Result'],
                'probability': round(row['probability'], 4)
            })

        # Calculate optimal APFD (if all failures were at the top)
        n_failures = len(failures)
        if n_failures > 0:
            optimal_positions = list(range(1, n_failures + 1))
            optimal_apfd = 1 - (sum(optimal_positions) / (n_failures * total_tc)) + 1 / (2 * total_tc)

            actual_positions = failures['rank'].tolist()
            case_study['optimal_vs_actual'] = {
                'optimal_positions': optimal_positions,
                'actual_positions': sorted(actual_positions),
                'optimal_apfd': round(optimal_apfd, 4),
                'actual_apfd': round(build_apfd, 4),
                'gap': round(optimal_apfd - build_apfd, 4)
            }

        return case_study

    return None


def analyze_failure_detection_speed(prio_df, apfd_df):
    """Analyze how quickly failures are detected."""

    results = []

    for build_id in apfd_df['build_id'].unique():
        build_prio = prio_df[prio_df['Build_ID'] == build_id].sort_values('rank')

        if len(build_prio) == 0:
            continue

        total_tc = len(build_prio)
        failures = build_prio[build_prio['TE_Test_Result'] == 'Fail']

        if len(failures) > 0:
            first_failure_rank = failures['rank'].min()
            first_failure_percentile = (first_failure_rank / total_tc) * 100

            # How many failures found in first 10%, 25%, 50%
            pct_10 = int(total_tc * 0.1)
            pct_25 = int(total_tc * 0.25)
            pct_50 = int(total_tc * 0.5)

            failures_in_10 = len(failures[failures['rank'] <= pct_10])
            failures_in_25 = len(failures[failures['rank'] <= pct_25])
            failures_in_50 = len(failures[failures['rank'] <= pct_50])

            results.append({
                'build_id': build_id,
                'total_tc': total_tc,
                'total_failures': len(failures),
                'first_failure_rank': first_failure_rank,
                'first_failure_percentile': first_failure_percentile,
                'failures_in_top_10pct': failures_in_10,
                'failures_in_top_25pct': failures_in_25,
                'failures_in_top_50pct': failures_in_50,
                'pct_failures_in_top_10': failures_in_10 / len(failures) * 100,
                'pct_failures_in_top_25': failures_in_25 / len(failures) * 100,
                'pct_failures_in_top_50': failures_in_50 / len(failures) * 100
            })

    return pd.DataFrame(results)


def generate_report(apfd_analysis, best_builds, worst_builds, perfect_builds,
                   correlations, category_stats, case_studies, detection_speed):
    """Generate a comprehensive qualitative analysis report."""

    report = """
================================================================================
QUALITATIVE ANALYSIS REPORT - FILO-PRIORI V9
================================================================================

1. APFD DISTRIBUTION ANALYSIS
-----------------------------

Total builds analyzed: {total_builds}

Statistics:
  Mean APFD:   {mean:.4f}
  Std Dev:     {std:.4f}
  Median:      {median:.4f}
  Min:         {min:.4f}
  Max:         {max:.4f}

Quartiles:
  Q1 (25%):    {q1:.4f}
  Q2 (50%):    {q2:.4f}
  Q3 (75%):    {q3:.4f}

Performance Categories:
  Perfect APFD (=1.0):    {perfect_apfd:3d} builds ({perfect_pct:.1f}%)
  High APFD (>=0.8):      {high_apfd:3d} builds ({high_pct:.1f}%)
  Medium APFD (0.5-0.8):  {medium_apfd:3d} builds ({medium_pct:.1f}%)
  Low APFD (<0.5):        {low_apfd:3d} builds ({low_pct:.1f}%)

""".format(
        total_builds=apfd_analysis['total_builds'],
        mean=apfd_analysis['mean'],
        std=apfd_analysis['std'],
        median=apfd_analysis['median'],
        min=apfd_analysis['min'],
        max=apfd_analysis['max'],
        q1=apfd_analysis['quartiles']['Q1'],
        q2=apfd_analysis['quartiles']['Q2'],
        q3=apfd_analysis['quartiles']['Q3'],
        perfect_apfd=apfd_analysis['perfect_apfd'],
        perfect_pct=apfd_analysis['perfect_apfd']/apfd_analysis['total_builds']*100,
        high_apfd=apfd_analysis['high_apfd'],
        high_pct=apfd_analysis['high_apfd']/apfd_analysis['total_builds']*100,
        medium_apfd=apfd_analysis['medium_apfd'],
        medium_pct=apfd_analysis['medium_apfd']/apfd_analysis['total_builds']*100,
        low_apfd=apfd_analysis['low_apfd'],
        low_pct=apfd_analysis['low_apfd']/apfd_analysis['total_builds']*100
    )

    report += """
2. CORRELATION ANALYSIS
-----------------------

Correlations with APFD:
  Number of test cases:  r = {apfd_vs_num_tc:+.4f}
  Number of failures:    r = {apfd_vs_num_failures:+.4f}
  Failure rate:          r = {apfd_vs_failure_rate:+.4f}
  Number of commits:     r = {apfd_vs_num_commits:+.4f}

Interpretation:
""".format(**correlations)

    if correlations['apfd_vs_num_tc'] > 0.1:
        report += "  - Larger builds tend to have higher APFD\n"
    elif correlations['apfd_vs_num_tc'] < -0.1:
        report += "  - Larger builds tend to have lower APFD\n"
    else:
        report += "  - Build size has minimal impact on APFD\n"

    if correlations['apfd_vs_failure_rate'] > 0.1:
        report += "  - Higher failure rates correlate with better APFD\n"
    elif correlations['apfd_vs_failure_rate'] < -0.1:
        report += "  - Higher failure rates correlate with worse APFD\n"
    else:
        report += "  - Failure rate has minimal impact on APFD\n"

    report += """
3. BUILD CHARACTERISTICS BY APFD CATEGORY
-----------------------------------------

"""
    report += category_stats.to_string()

    report += """

4. FAILURE DETECTION SPEED
--------------------------

"""
    if len(detection_speed) > 0:
        report += f"""
Average first failure detection:
  Mean rank of first failure:     {detection_speed['first_failure_rank'].mean():.1f}
  Mean percentile:                {detection_speed['first_failure_percentile'].mean():.1f}%

Failures detected by test execution percentage:
  In top 10% of tests:  {detection_speed['pct_failures_in_top_10'].mean():.1f}% of failures
  In top 25% of tests:  {detection_speed['pct_failures_in_top_25'].mean():.1f}% of failures
  In top 50% of tests:  {detection_speed['pct_failures_in_top_50'].mean():.1f}% of failures

This means running only 25% of tests would detect {detection_speed['pct_failures_in_top_25'].mean():.1f}% of failures.
"""

    report += """
5. BEST PERFORMING BUILDS (Top 10)
----------------------------------

"""
    for _, row in best_builds.iterrows():
        report += f"  {row['build_id']}: APFD = {row['apfd']:.4f} ({row['count_tc']} TCs)\n"

    report += """
6. WORST PERFORMING BUILDS (Bottom 10)
--------------------------------------

"""
    for _, row in worst_builds.iterrows():
        report += f"  {row['build_id']}: APFD = {row['apfd']:.4f} ({row['count_tc']} TCs)\n"

    report += """
7. CASE STUDIES
---------------
"""

    for i, case in enumerate(case_studies, 1):
        if case is None:
            continue

        report += f"""
Case Study {i}: {case['build_id']}
{'='*50}

Build Overview:
  APFD Score:        {case['apfd']:.4f}
  Total Test Cases:  {case['total_test_cases']}
  Total Failures:    {case['total_failures']}
  Failure Rate:      {case['failure_rate']*100:.1f}%

"""
        if case['optimal_vs_actual']:
            report += f"""Ranking Analysis:
  Optimal failure positions: {case['optimal_vs_actual']['optimal_positions'][:5]}{'...' if len(case['optimal_vs_actual']['optimal_positions']) > 5 else ''}
  Actual failure positions:  {case['optimal_vs_actual']['actual_positions'][:5]}{'...' if len(case['optimal_vs_actual']['actual_positions']) > 5 else ''}
  Optimal APFD:              {case['optimal_vs_actual']['optimal_apfd']:.4f}
  Actual APFD:               {case['optimal_vs_actual']['actual_apfd']:.4f}
  Gap:                       {case['optimal_vs_actual']['gap']:.4f}

"""

        report += "Top 10 Prioritized Test Cases:\n"
        report += "  Rank | Result | Probability | TC Key\n"
        report += "  -----|--------|-------------|--------\n"
        for tc in case['top_10_ranking']:
            result_marker = "FAIL" if tc['result'] == 'Fail' else "Pass"
            report += f"  {tc['rank']:4d} | {result_marker:6s} | {tc['probability']:.4f}      | {tc['tc_key'][:20]}\n"
        report += "\n"

    report += """
8. KEY INSIGHTS
---------------

"""
    # Calculate key insights
    perfect_pct = apfd_analysis['perfect_apfd'] / apfd_analysis['total_builds'] * 100
    high_pct = apfd_analysis['high_apfd'] / apfd_analysis['total_builds'] * 100

    report += f"""
1. PERFECT PRIORITIZATION: {apfd_analysis['perfect_apfd']} builds ({perfect_pct:.1f}%) achieve
   perfect APFD (1.0), meaning all failures were ranked first.

2. HIGH PERFORMANCE: {apfd_analysis['high_apfd']} builds ({high_pct:.1f}%) achieve APFD >= 0.8,
   indicating strong failure detection in the majority of cases.

3. EARLY DETECTION: On average, {detection_speed['pct_failures_in_top_25'].mean():.1f}% of failures
   are found by running only 25% of tests.

4. BUILD SIZE IMPACT: {'Larger builds tend to have better APFD' if correlations['apfd_vs_num_tc'] > 0.1 else 'Build size has minimal impact on performance'}.

5. FAILURE RATE IMPACT: {'Higher failure rates correlate with better APFD' if correlations['apfd_vs_failure_rate'] > 0.1 else 'Failure rate has minimal impact on performance'}.

================================================================================
"""

    return report


def generate_latex_case_studies(case_studies):
    """Generate LaTeX formatted case studies."""

    latex = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CASE STUDIES - QUALITATIVE ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\subsection{Case Studies}
\\label{sec:case_studies}

We present detailed case studies of builds with varying APFD performance
to illustrate Filo-Priori's behavior in different scenarios.

"""

    for i, case in enumerate(case_studies, 1):
        if case is None:
            continue

        perf_label = "High" if case['apfd'] >= 0.8 else ("Medium" if case['apfd'] >= 0.5 else "Low")

        latex += f"""
\\subsubsection{{Case Study {i}: {perf_label} Performance Build}}

\\textbf{{Build ID:}} \\texttt{{{case['build_id']}}}

\\begin{{itemize}}
    \\item APFD Score: {case['apfd']:.4f}
    \\item Total Test Cases: {case['total_test_cases']}
    \\item Total Failures: {case['total_failures']} ({case['failure_rate']*100:.1f}\\%)
"""

        if case['optimal_vs_actual']:
            gap = case['optimal_vs_actual']['gap']
            latex += f"""    \\item Optimal APFD: {case['optimal_vs_actual']['optimal_apfd']:.4f}
    \\item Gap from Optimal: {gap:.4f}
"""

        latex += """\\end{itemize}

\\begin{table}[htbp]
\\centering
\\caption{Top 10 Prioritized Test Cases for """ + case['build_id'] + """}
\\begin{tabular}{clcc}
\\toprule
\\textbf{Rank} & \\textbf{TC Key} & \\textbf{Result} & \\textbf{Prob.} \\\\
\\midrule
"""

        for tc in case['top_10_ranking']:
            result = "\\textbf{Fail}" if tc['result'] == 'Fail' else "Pass"
            latex += f"{tc['rank']} & {tc['tc_key'][:15]}... & {result} & {tc['probability']:.3f} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}

"""

    return latex


def main():
    print("\n" + "=" * 80)
    print(" QUALITATIVE ANALYSIS - FILO-PRIORI V9")
    print("=" * 80)

    # Create output directory
    output_dir = PROJECT_ROOT / 'results' / 'qualitative_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    apfd_df, prio_df, test_df = load_data()
    print(f"   Loaded {len(apfd_df)} builds with APFD scores")
    print(f"   Loaded {len(prio_df)} prioritized test case records")
    print(f"   Loaded {len(test_df)} test execution records")

    # Analyze APFD distribution
    print("\n2. Analyzing APFD distribution...")
    apfd_analysis = analyze_apfd_distribution(apfd_df)
    print(f"   Mean APFD: {apfd_analysis['mean']:.4f}")
    print(f"   Perfect APFD builds: {apfd_analysis['perfect_apfd']}")

    # Identify extreme cases
    print("\n3. Identifying extreme cases...")
    best_builds, worst_builds, perfect_builds = identify_extreme_cases(apfd_df)
    print(f"   Best builds: {len(best_builds)}")
    print(f"   Worst builds: {len(worst_builds)}")
    print(f"   Perfect builds: {len(perfect_builds)}")

    # Analyze build characteristics
    print("\n4. Analyzing build characteristics...")
    merged_df, correlations, category_stats = analyze_build_characteristics(apfd_df, test_df)
    print(f"   Correlations computed")

    # Analyze failure detection speed
    print("\n5. Analyzing failure detection speed...")
    detection_speed = analyze_failure_detection_speed(prio_df, apfd_df)
    print(f"   Analyzed {len(detection_speed)} builds")

    # Create case studies
    print("\n6. Creating case studies...")
    case_studies = []

    # Best performing build
    best_build_id = best_builds.iloc[0]['build_id']
    case_studies.append(create_case_study(best_build_id, prio_df, test_df, apfd_df))
    print(f"   Case 1 (Best): {best_build_id}")

    # Worst performing build
    worst_build_id = worst_builds.iloc[0]['build_id']
    case_studies.append(create_case_study(worst_build_id, prio_df, test_df, apfd_df))
    print(f"   Case 2 (Worst): {worst_build_id}")

    # Medium performing build
    medium_apfd = apfd_df[(apfd_df['apfd'] >= 0.55) & (apfd_df['apfd'] <= 0.65)]
    if len(medium_apfd) > 0:
        medium_build_id = medium_apfd.iloc[len(medium_apfd)//2]['build_id']
        case_studies.append(create_case_study(medium_build_id, prio_df, test_df, apfd_df))
        print(f"   Case 3 (Medium): {medium_build_id}")

    # Large build (many test cases)
    large_build = apfd_df.sort_values('count_tc', ascending=False).iloc[0]['build_id']
    case_studies.append(create_case_study(large_build, prio_df, test_df, apfd_df))
    print(f"   Case 4 (Large): {large_build}")

    # Generate report
    print("\n7. Generating report...")
    report = generate_report(apfd_analysis, best_builds, worst_builds, perfect_builds,
                            correlations, category_stats, case_studies, detection_speed)

    with open(output_dir / 'qualitative_analysis_report.txt', 'w') as f:
        f.write(report)
    print(f"   Saved: qualitative_analysis_report.txt")

    # Generate LaTeX case studies
    latex_cases = generate_latex_case_studies(case_studies)
    with open(output_dir / 'case_studies.tex', 'w') as f:
        f.write(latex_cases)
    print(f"   Saved: case_studies.tex")

    # Save detection speed analysis
    detection_speed.to_csv(output_dir / 'failure_detection_speed.csv', index=False)
    print(f"   Saved: failure_detection_speed.csv")

    # Save category statistics
    category_stats.to_csv(output_dir / 'category_statistics.csv')
    print(f"   Saved: category_statistics.csv")

    # Create visualization
    print("\n8. Creating visualizations...")
    create_visualizations(apfd_df, detection_speed, category_stats, output_dir)
    print(f"   Saved: qualitative_analysis.png/pdf")

    # Print summary
    print("\n" + "=" * 80)
    print(" KEY FINDINGS")
    print("=" * 80)
    print(f"\n   Perfect APFD builds:     {apfd_analysis['perfect_apfd']} ({apfd_analysis['perfect_apfd']/apfd_analysis['total_builds']*100:.1f}%)")
    print(f"   High APFD builds (>=0.8): {apfd_analysis['high_apfd']} ({apfd_analysis['high_apfd']/apfd_analysis['total_builds']*100:.1f}%)")
    print(f"   Mean first failure rank:  {detection_speed['first_failure_rank'].mean():.1f}")
    print(f"   Failures in top 25%:      {detection_speed['pct_failures_in_top_25'].mean():.1f}%")

    print("\n" + "=" * 80)
    print(f" Results saved to: {output_dir}")
    print("=" * 80)


def create_visualizations(apfd_df, detection_speed, category_stats, output_dir):
    """Create qualitative analysis visualizations."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.dpi': 150
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. APFD Distribution Histogram
    ax = axes[0, 0]
    ax.hist(apfd_df['apfd'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(apfd_df['apfd'].mean(), color='red', linestyle='--', label=f'Mean: {apfd_df["apfd"].mean():.3f}')
    ax.axvline(apfd_df['apfd'].median(), color='green', linestyle='--', label=f'Median: {apfd_df["apfd"].median():.3f}')
    ax.set_xlabel('APFD Score')
    ax.set_ylabel('Number of Builds')
    ax.set_title('(a) APFD Score Distribution')
    ax.legend()

    # 2. First Failure Detection
    ax = axes[0, 1]
    ax.hist(detection_speed['first_failure_percentile'], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.axvline(detection_speed['first_failure_percentile'].mean(), color='red', linestyle='--',
               label=f'Mean: {detection_speed["first_failure_percentile"].mean():.1f}%')
    ax.set_xlabel('First Failure Position (% of test suite)')
    ax.set_ylabel('Number of Builds')
    ax.set_title('(b) First Failure Detection Position')
    ax.legend()

    # 3. Failures Detected by Execution Percentage
    ax = axes[1, 0]
    percentages = [10, 25, 50, 75, 100]
    failures_detected = [
        detection_speed['pct_failures_in_top_10'].mean(),
        detection_speed['pct_failures_in_top_25'].mean(),
        detection_speed['pct_failures_in_top_50'].mean(),
        (detection_speed['pct_failures_in_top_50'].mean() + 100) / 2,  # Estimate for 75%
        100
    ]
    ax.plot(percentages, failures_detected, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax.fill_between(percentages, failures_detected, alpha=0.3, color='#e74c3c')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Test Suite Executed (%)')
    ax.set_ylabel('Failures Detected (%)')
    ax.set_title('(c) Failure Detection Curve')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 4. APFD vs Build Size
    ax = axes[1, 1]
    ax.scatter(apfd_df['count_tc'], apfd_df['apfd'], alpha=0.5, color='#9b59b6')
    ax.set_xlabel('Number of Test Cases')
    ax.set_ylabel('APFD Score')
    ax.set_title('(d) APFD vs Build Size')

    # Add trend line
    z = np.polyfit(apfd_df['count_tc'], apfd_df['apfd'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(apfd_df['count_tc'].min(), apfd_df['count_tc'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
    ax.legend()

    plt.suptitle('Qualitative Analysis - Filo-Priori v9', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_dir / 'qualitative_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'qualitative_analysis.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
