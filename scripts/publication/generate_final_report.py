#!/usr/bin/env python3
"""
Generate Consolidated Final Report for Filo-Priori v9.

Combines all experimental results from phases 1-6 into a comprehensive
publication-ready report.

Author: Filo-Priori Team
Date: 2025-11-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def load_all_results():
    """Load all experimental results from all phases."""
    results = {}

    # Phase 1: Baselines
    baseline_path = PROJECT_ROOT / 'results' / 'baselines' / 'comparison_table.csv'
    if baseline_path.exists():
        results['baselines'] = pd.read_csv(baseline_path)

    # Phase 2: Ablation
    ablation_path = PROJECT_ROOT / 'results' / 'ablation' / 'ablation_study_final.csv'
    if ablation_path.exists():
        results['ablation'] = pd.read_csv(ablation_path)

    # Phase 3: Temporal CV
    temporal_path = PROJECT_ROOT / 'results' / 'temporal_cv' / 'temporal_cv_summary.csv'
    if temporal_path.exists():
        results['temporal_cv'] = pd.read_csv(temporal_path)

    # Phase 4: Sensitivity
    sensitivity_path = PROJECT_ROOT / 'results' / 'sensitivity' / 'sensitivity_analysis_combined.csv'
    if sensitivity_path.exists():
        results['sensitivity'] = pd.read_csv(sensitivity_path)

    # Phase 6: Qualitative
    detection_path = PROJECT_ROOT / 'results' / 'qualitative_analysis' / 'failure_detection_speed.csv'
    if detection_path.exists():
        results['detection_speed'] = pd.read_csv(detection_path)

    category_path = PROJECT_ROOT / 'results' / 'qualitative_analysis' / 'category_statistics.csv'
    if category_path.exists():
        results['category_stats'] = pd.read_csv(category_path)

    # APFD per build
    apfd_path = PROJECT_ROOT / 'results' / 'experiment_06_feature_selection' / 'apfd_per_build_FULL_testcsv.csv'
    if apfd_path.exists():
        results['apfd_per_build'] = pd.read_csv(apfd_path)

    return results


def generate_executive_summary(results):
    """Generate executive summary."""

    summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    FILO-PRIORI V9 - EXECUTIVE SUMMARY                        ║
║                                                                              ║
║           Deep Learning-based Test Case Prioritization System                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: {date}

┌──────────────────────────────────────────────────────────────────────────────┐
│ MAIN RESULT                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

  Filo-Priori v9 achieves APFD = 0.6171 on 277 builds with failures,
  representing a +10.3% improvement over random ordering (p < 0.001).

┌──────────────────────────────────────────────────────────────────────────────┐
│ KEY METRICS                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┬────────────────────────────────────────────────────┐
  │ Metric              │ Value                                              │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Mean APFD           │ 0.6171 [0.586, 0.648]                              │
  │ vs Random           │ +10.3% (p < 0.001) ***                             │
  │ vs FailureRate      │ -1.9% (p = 0.36, not significant)                  │
  │ vs XGBoost          │ 0.0% (comparable)                                  │
  │ Perfect APFD builds │ 23 (8.3%)                                          │
  │ High APFD builds    │ 84 (30.3%)                                         │
  │ Failures in top 25% │ 33.2%                                              │
  └─────────────────────┴────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ ARCHITECTURAL INSIGHTS (Ablation Study)                                      │
└──────────────────────────────────────────────────────────────────────────────┘

  Component Contributions:

  ████████████████████████████████████  GATv2 (Graph Attention): +17.0% ***
  ██████████                            Structural Stream:        +5.3% ***
  █████████                             Class Weighting:          +4.6% ***
  ███████                               Ensemble:                 +3.5% ***
  ████                                  Semantic Stream:          +1.9%
  ▒▒                                    Cross-Attention:          -1.1%

  → Graph Attention Networks are the MOST CRITICAL component

┌──────────────────────────────────────────────────────────────────────────────┐
│ TEMPORAL ROBUSTNESS                                                          │
└──────────────────────────────────────────────────────────────────────────────┘

  Validation Method         APFD     95% CI
  ─────────────────────────────────────────
  Temporal 5-Fold CV        0.6629   [0.627, 0.698]
  Sliding Window CV         0.6279   [0.595, 0.661]
  Concept Drift Test        0.6187   [0.574, 0.661]

  → Model is STABLE across temporal validation (range: 0.619-0.663)
  → No significant concept drift detected

┌──────────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED CONFIGURATION                                                    │
└──────────────────────────────────────────────────────────────────────────────┘

  Parameter              Recommended Value    Impact
  ──────────────────────────────────────────────────
  Loss Function          Weighted CE          High (Δ=5.9%)
  Learning Rate          3e-5                 Medium (Δ=4.5%)
  GNN Architecture       1 layer, 2 heads     Medium (Δ=4.5%)
  Structural Features    10 (selected)        Low (Δ=2.9%)
  Balanced Sampling      No                   Medium (Δ=5.3%)

""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return summary


def generate_detailed_report(results):
    """Generate detailed consolidated report."""

    report = """
================================================================================
                     FILO-PRIORI V9 - CONSOLIDATED FINAL REPORT
================================================================================
                    Deep Learning-based Test Case Prioritization
================================================================================

Generated: {date}
Dataset: QTA (Industrial Mobile Testing)
Builds with failures: 277
Total test executions: 8,847

================================================================================
                              TABLE OF CONTENTS
================================================================================

1. Executive Summary
2. Phase 1: Baseline Comparison
3. Phase 2: Ablation Study
4. Phase 3: Temporal Cross-Validation
5. Phase 4: Hyperparameter Sensitivity
6. Phase 5: Paper Sections (Generated)
7. Phase 6: Qualitative Analysis
8. Consolidated Statistics
9. Recommendations
10. Files Generated

================================================================================
                         1. EXECUTIVE SUMMARY
================================================================================

Filo-Priori v9 is a dual-stream deep learning architecture for Test Case
Prioritization (TCP) that combines:

• Semantic Stream: SBERT embeddings of test case descriptions and commit messages
• Structural Stream: Historical test execution patterns (10 engineered features)
• Graph Attention: GATv2 on phylogenetic graph (co-failure relationships)
• Fusion Module: Cross-attention mechanism for stream integration

MAIN FINDINGS:
─────────────
1. Achieves APFD = 0.6171 (+10.3% vs Random, p < 0.001)
2. Graph Attention Networks contribute +17.0% to performance
3. Model is temporally robust (APFD range: 0.619-0.663)
4. Running 25% of tests detects 33.2% of failures
5. 8.3% of builds achieve perfect prioritization (APFD = 1.0)

""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Phase 1: Baselines
    report += """
================================================================================
                      2. PHASE 1: BASELINE COMPARISON
================================================================================

"""
    if 'baselines' in results:
        df = results['baselines']
        report += "Method                  Mean APFD    95% CI              Δ vs FP     p-value    Sig.\n"
        report += "─" * 90 + "\n"

        filo_apfd = df[df['Method'] == 'Filo-Priori']['Mean APFD'].values[0]

        for _, row in df.iterrows():
            method = row['Method']
            mean = row['Mean APFD']
            ci_l = row['95% CI Lower']
            ci_u = row['95% CI Upper']
            delta = mean - filo_apfd
            pval = row['p-value']
            sig = row.get('Significant', '')

            if method == 'Filo-Priori':
                report += f"{method:<22} {mean:.4f}       [{ci_l:.3f}, {ci_u:.3f}]       -           -          (reference)\n"
            else:
                pval_str = f"{pval:.4f}" if pval > 0.0001 else f"{pval:.2e}"
                report += f"{method:<22} {mean:.4f}       [{ci_l:.3f}, {ci_u:.3f}]       {delta:+.4f}      {pval_str:<10} {sig}\n"

        report += "\nKey Findings:\n"
        report += "• Filo-Priori significantly outperforms Random, Recency, and RecentFailureRate (p < 0.001)\n"
        report += "• Comparable performance to FailureRate and XGBoost (no significant difference)\n"
        report += "• Effect sizes are small to negligible for most comparisons\n"

    # Phase 2: Ablation
    report += """
================================================================================
                        3. PHASE 2: ABLATION STUDY
================================================================================

"""
    if 'ablation' in results:
        df = results['ablation']
        report += "ID    Component Removed      Mean APFD    Δ APFD      Contribution    Sig.\n"
        report += "─" * 80 + "\n"

        for _, row in df.iterrows():
            aid = row['ID']
            comp = row['Removed Component']
            mean = row['Mean APFD']
            delta = row['Δ APFD']
            contrib = row['Contribution (%)']
            sig = row.get('Significant', '')

            if aid == 'A0':
                report += f"{aid:<5} {'Full Model':<22} {mean:.4f}       -           -               (baseline)\n"
            else:
                report += f"{aid:<5} {comp:<22} {mean:.4f}       {delta:+.4f}      {contrib:+.1f}%           {sig}\n"

        report += "\nKey Findings:\n"
        report += "• GATv2 (Graph Attention) is the MOST CRITICAL component (+17.0%)\n"
        report += "• Structural Stream contributes +5.3% (second most important)\n"
        report += "• Class Weighting provides +4.6% improvement\n"
        report += "• Cross-Attention has NEGATIVE contribution (-1.1%), suggesting simpler fusion may work better\n"

    # Phase 3: Temporal CV
    report += """
================================================================================
                   4. PHASE 3: TEMPORAL CROSS-VALIDATION
================================================================================

"""
    if 'temporal_cv' in results:
        df = results['temporal_cv']
        report += "Validation Method        Mean APFD    Std        95% CI              N Builds\n"
        report += "─" * 80 + "\n"

        for _, row in df.iterrows():
            method = row['Method']
            mean = row['Mean APFD']
            std = row['Std']
            ci_l = row['95% CI Lower']
            ci_u = row['95% CI Upper']
            n = int(row['N Evaluations'])
            report += f"{method:<24} {mean:.4f}       {std:.4f}     [{ci_l:.3f}, {ci_u:.3f}]       {n}\n"

        report += "\nKey Findings:\n"
        report += "• Model maintains stable performance across all temporal validation methods\n"
        report += "• APFD range (0.619-0.663) indicates robustness to temporal distribution shift\n"
        report += "• No significant concept drift detected\n"
        report += "• Temporal K-Fold shows highest APFD (0.663), suggesting good generalization\n"

    # Phase 4: Sensitivity
    report += """
================================================================================
                  5. PHASE 4: HYPERPARAMETER SENSITIVITY
================================================================================

"""
    if 'sensitivity' in results:
        df = results['sensitivity']
        report += "Hyperparameter          Value                    Mean APFD    95% CI\n"
        report += "─" * 80 + "\n"

        current_hp = None
        for _, row in df.iterrows():
            hp = row['Hyperparameter']
            if hp != current_hp:
                if current_hp is not None:
                    report += "─" * 80 + "\n"
                current_hp = hp

            value = row['Value']
            mean = row['Mean APFD']
            ci = row['95% CI']

            # Mark best
            hp_df = df[df['Hyperparameter'] == hp]
            is_best = mean == hp_df['Mean APFD'].max()
            marker = " ← BEST" if is_best else ""

            report += f"{hp:<23} {value:<24} {mean:.4f}       {ci}{marker}\n"

        report += "\nSensitivity Ranges:\n"
        for hp in df['Hyperparameter'].unique():
            hp_df = df[df['Hyperparameter'] == hp]
            delta = hp_df['Mean APFD'].max() - hp_df['Mean APFD'].min()
            report += f"• {hp}: Δ = {delta:.4f} ({delta/0.6*100:.1f}% relative)\n"

        report += "\nKey Findings:\n"
        report += "• Loss Function has the highest impact (5.9% relative variation)\n"
        report += "• Simpler GNN architecture (1 layer, 2 heads) outperforms deeper models\n"
        report += "• Balanced sampling DECREASES performance (avoid for ranking tasks)\n"

    # Phase 6: Qualitative
    report += """
================================================================================
                      6. PHASE 6: QUALITATIVE ANALYSIS
================================================================================

"""
    if 'apfd_per_build' in results:
        apfd = results['apfd_per_build']['apfd']

        report += "APFD Distribution:\n"
        report += "─" * 40 + "\n"
        report += f"  Mean:     {apfd.mean():.4f}\n"
        report += f"  Median:   {apfd.median():.4f}\n"
        report += f"  Std:      {apfd.std():.4f}\n"
        report += f"  Min:      {apfd.min():.4f}\n"
        report += f"  Max:      {apfd.max():.4f}\n"
        report += f"  Q1:       {apfd.quantile(0.25):.4f}\n"
        report += f"  Q3:       {apfd.quantile(0.75):.4f}\n"

        perfect = (apfd == 1.0).sum()
        high = (apfd >= 0.8).sum()
        medium = ((apfd >= 0.5) & (apfd < 0.8)).sum()
        low = (apfd < 0.5).sum()
        total = len(apfd)

        report += f"\nPerformance Categories:\n"
        report += "─" * 40 + "\n"
        report += f"  Perfect (=1.0):    {perfect:3d} builds ({perfect/total*100:.1f}%)\n"
        report += f"  High (≥0.8):       {high:3d} builds ({high/total*100:.1f}%)\n"
        report += f"  Medium (0.5-0.8):  {medium:3d} builds ({medium/total*100:.1f}%)\n"
        report += f"  Low (<0.5):        {low:3d} builds ({low/total*100:.1f}%)\n"

    if 'detection_speed' in results:
        ds = results['detection_speed']

        report += f"\nFailure Detection Speed:\n"
        report += "─" * 40 + "\n"
        report += f"  Mean first failure rank:     {ds['first_failure_rank'].mean():.1f}\n"
        report += f"  Mean first failure pct:      {ds['first_failure_percentile'].mean():.1f}%\n"
        report += f"  Failures in top 10%:         {ds['pct_failures_in_top_10'].mean():.1f}%\n"
        report += f"  Failures in top 25%:         {ds['pct_failures_in_top_25'].mean():.1f}%\n"
        report += f"  Failures in top 50%:         {ds['pct_failures_in_top_50'].mean():.1f}%\n"

        report += "\nKey Findings:\n"
        report += "• 8.3% of builds achieve perfect prioritization (all failures first)\n"
        report += "• Running only 25% of tests detects 33.2% of failures on average\n"
        report += "• Higher failure rates correlate with better APFD (r = +0.37)\n"
        report += "• Model struggles with very low failure rate builds (<3%)\n"

    # Consolidated Statistics
    report += """
================================================================================
                       7. CONSOLIDATED STATISTICS
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUMMARY TABLE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Metric                                    │ Value                           │
├───────────────────────────────────────────┼─────────────────────────────────┤
│ Dataset                                   │ QTA (Industrial)                │
│ Builds with failures                      │ 277                             │
│ Total test executions                     │ 8,847                           │
├───────────────────────────────────────────┼─────────────────────────────────┤
│ Main APFD                                 │ 0.6171 [0.586, 0.648]           │
│ Improvement vs Random                     │ +10.3% (p < 0.001)              │
│ Temporal CV range                         │ 0.619 - 0.663                   │
├───────────────────────────────────────────┼─────────────────────────────────┤
│ Most important component                  │ GATv2 (+17.0%)                  │
│ Second most important                     │ Structural Stream (+5.3%)       │
│ Most sensitive hyperparameter             │ Loss Function (Δ=5.9%)          │
├───────────────────────────────────────────┼─────────────────────────────────┤
│ Perfect APFD builds                       │ 23 (8.3%)                       │
│ High APFD builds (≥0.8)                   │ 84 (30.3%)                      │
│ Failures detected in top 25%             │ 33.2%                           │
└───────────────────────────────────────────┴─────────────────────────────────┘

"""

    # Recommendations
    report += """
================================================================================
                          8. RECOMMENDATIONS
================================================================================

FOR PRACTITIONERS:
─────────────────
1. Use Filo-Priori when you have:
   • At least 100 builds with failure history
   • Test case descriptions/documentation
   • Commit information linked to builds

2. Recommended configuration:
   • Loss: Weighted Cross-Entropy
   • Learning Rate: 3e-5
   • GNN: 1 layer, 2 attention heads
   • Features: Use feature selection (10 features)
   • DO NOT use balanced sampling for ranking tasks

3. Expected benefits:
   • ~10% improvement over random ordering
   • 33% of failures detected in first 25% of tests
   • Stable performance over time (no retraining needed frequently)

4. Limitations to consider:
   • Cold start problem for new test cases
   • Lower performance on builds with <3% failure rate
   • Requires GPU for training (30-60 minutes)

FOR RESEARCHERS:
────────────────
1. Graph Attention Networks are crucial for TCP - explore other GNN variants
2. Simpler architectures often outperform complex ones - avoid over-engineering
3. Cross-attention fusion shows negative contribution - investigate alternatives
4. Temporal robustness is achievable with proper training strategies

"""

    # Files Generated
    report += """
================================================================================
                         9. FILES GENERATED
================================================================================

Phase 1 - Baselines:
  results/baselines/comparison_table.csv
  results/baselines/comparison_table.tex
  results/baselines/apfd_boxplot.png

Phase 2 - Ablation:
  results/ablation/ablation_study_final.csv
  results/ablation/ablation_study_final.tex
  results/ablation/ablation_study_final.png

Phase 3 - Temporal CV:
  results/temporal_cv/temporal_cv_summary.csv
  results/temporal_cv/temporal_cv_table.tex
  results/temporal_cv/temporal_cv_results.png

Phase 4 - Sensitivity:
  results/sensitivity/sensitivity_analysis_combined.csv
  results/sensitivity/sensitivity_analysis.tex
  results/sensitivity/sensitivity_analysis.png

Phase 5 - Paper Sections:
  results/paper_sections/section_results.tex
  results/paper_sections/section_discussion.tex
  results/paper_sections/section_threats.tex
  results/paper_sections/all_tables.tex
  results/paper_sections/paper_sections_combined.tex

Phase 6 - Qualitative:
  results/qualitative_analysis/qualitative_analysis_report.txt
  results/qualitative_analysis/case_studies.tex
  results/qualitative_analysis/qualitative_analysis.png

Final Report:
  results/final_report/consolidated_report.txt
  results/final_report/consolidated_report.md
  results/final_report/executive_summary.txt

================================================================================
                              END OF REPORT
================================================================================
"""

    return report


def generate_markdown_report(results):
    """Generate Markdown version of the report."""

    md = """# Filo-Priori v9 - Consolidated Final Report

**Deep Learning-based Test Case Prioritization System**

Generated: {date}

---

## Executive Summary

Filo-Priori v9 achieves **APFD = 0.6171** on 277 builds with failures, representing a **+10.3% improvement** over random ordering (p < 0.001).

### Key Metrics

| Metric | Value |
|--------|-------|
| Mean APFD | 0.6171 [0.586, 0.648] |
| vs Random | +10.3% (p < 0.001) *** |
| vs FailureRate | -1.9% (not significant) |
| Perfect APFD builds | 23 (8.3%) |
| High APFD builds (≥0.8) | 84 (30.3%) |
| Failures in top 25% | 33.2% |

---

## Phase Results Summary

### Phase 1: Baseline Comparison

""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if 'baselines' in results:
        df = results['baselines']
        md += "| Method | Mean APFD | 95% CI | Significance |\n"
        md += "|--------|-----------|--------|-------------|\n"

        for _, row in df.iterrows():
            method = row['Method']
            mean = row['Mean APFD']
            ci_l = row['95% CI Lower']
            ci_u = row['95% CI Upper']
            sig = row.get('Significant', '')
            md += f"| {method} | {mean:.4f} | [{ci_l:.3f}, {ci_u:.3f}] | {sig} |\n"

    md += """
### Phase 2: Ablation Study

| Component | Contribution | Significance |
|-----------|--------------|--------------|
| GATv2 (Graph Attention) | +17.0% | *** |
| Structural Stream | +5.3% | *** |
| Class Weighting | +4.6% | *** |
| Ensemble | +3.5% | *** |
| Semantic Stream | +1.9% | - |
| Cross-Attention | -1.1% | - |

**Key Finding:** Graph Attention Networks are the most critical component.

### Phase 3: Temporal Cross-Validation

| Method | Mean APFD | 95% CI |
|--------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

**Key Finding:** Model is temporally robust (range: 0.619-0.663).

### Phase 4: Hyperparameter Sensitivity

| Hyperparameter | Best Value | Impact |
|----------------|------------|--------|
| Loss Function | Weighted CE | 5.9% |
| Learning Rate | 3e-5 | 4.5% |
| GNN Architecture | 1 layer, 2 heads | 4.5% |
| Structural Features | 10 (selected) | 2.9% |
| Balanced Sampling | No | 5.3% |

### Phase 6: Qualitative Analysis

| Category | Builds | Percentage |
|----------|--------|------------|
| Perfect (=1.0) | 23 | 8.3% |
| High (≥0.8) | 84 | 30.3% |
| Medium (0.5-0.8) | 100 | 36.1% |
| Low (<0.5) | 93 | 33.6% |

**Failure Detection:** Running 25% of tests detects 33.2% of failures.

---

## Recommendations

### For Practitioners

1. **Use Filo-Priori when you have:**
   - At least 100 builds with failure history
   - Test case descriptions/documentation
   - Commit information linked to builds

2. **Recommended configuration:**
   - Loss: Weighted Cross-Entropy
   - Learning Rate: 3e-5
   - GNN: 1 layer, 2 attention heads
   - DO NOT use balanced sampling

3. **Expected benefits:**
   - ~10% improvement over random ordering
   - 33% of failures detected in first 25% of tests

### For Researchers

1. Graph Attention Networks are crucial for TCP
2. Simpler architectures often outperform complex ones
3. Cross-attention fusion shows negative contribution - investigate alternatives

---

## Citation

If you use Filo-Priori v9, please cite:

```bibtex
@article{filo_priori_v9,
  title={Filo-Priori: Deep Learning-based Test Case Prioritization with Graph Attention Networks},
  author={...},
  journal={...},
  year={2025}
}
```

---

*Report generated automatically by Filo-Priori v9 analysis pipeline.*
"""

    return md


def main():
    print("\n" + "=" * 80)
    print(" GENERATING CONSOLIDATED FINAL REPORT")
    print("=" * 80)

    # Create output directory
    output_dir = PROJECT_ROOT / 'results' / 'final_report'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("\n1. Loading all experimental results...")
    results = load_all_results()
    print(f"   Loaded: {list(results.keys())}")

    # Generate executive summary
    print("\n2. Generating executive summary...")
    exec_summary = generate_executive_summary(results)
    with open(output_dir / 'executive_summary.txt', 'w') as f:
        f.write(exec_summary)
    print(f"   Saved: executive_summary.txt")

    # Generate detailed report
    print("\n3. Generating detailed report...")
    detailed_report = generate_detailed_report(results)
    with open(output_dir / 'consolidated_report.txt', 'w') as f:
        f.write(detailed_report)
    print(f"   Saved: consolidated_report.txt")

    # Generate Markdown report
    print("\n4. Generating Markdown report...")
    md_report = generate_markdown_report(results)
    with open(output_dir / 'consolidated_report.md', 'w') as f:
        f.write(md_report)
    print(f"   Saved: consolidated_report.md")

    # Copy all LaTeX tables to final report
    print("\n5. Compiling all LaTeX files...")
    latex_files = [
        'results/baselines/comparison_table.tex',
        'results/ablation/ablation_study_final.tex',
        'results/temporal_cv/temporal_cv_table.tex',
        'results/sensitivity/sensitivity_analysis.tex',
        'results/qualitative_analysis/case_studies.tex',
        'results/paper_sections/all_tables.tex',
        'results/paper_sections/section_results.tex',
        'results/paper_sections/section_discussion.tex',
        'results/paper_sections/section_threats.tex'
    ]

    combined_latex = """% FILO-PRIORI V9 - ALL LATEX CONTENT
% Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
%
% This file contains all LaTeX tables and sections for the paper.
%

"""

    for latex_file in latex_files:
        path = PROJECT_ROOT / latex_file
        if path.exists():
            with open(path) as f:
                content = f.read()
            combined_latex += f"\n% === {latex_file} ===\n"
            combined_latex += content + "\n"

    with open(output_dir / 'all_latex_content.tex', 'w') as f:
        f.write(combined_latex)
    print(f"   Saved: all_latex_content.tex")

    # Print executive summary
    print("\n" + exec_summary)

    print("\n" + "=" * 80)
    print(" REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n   Output directory: {output_dir}")
    print("\n   Files generated:")
    for f in sorted(output_dir.glob('*')):
        print(f"      - {f.name}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
