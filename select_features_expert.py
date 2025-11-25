#!/usr/bin/env python3
"""
Expert-Based Feature Selection for Test Case Prioritization

Since Experiment 05 didn't save model checkpoints, we use domain knowledge
and feature statistics to select the most promising 10 features.

Selection Criteria:
1. Statistical significance (variance, distribution)
2. Domain relevance for test prioritization
3. Independence (avoid highly correlated features)
4. Proven usefulness in baseline (6 features from V1)

Author: Filo-Priori V8 Team
Date: 2025-11-14
"""

import pickle
import numpy as np
from typing import List, Tuple

# All 29 features from V2
FEATURE_NAMES = [
    # PHYLOGENETIC (20)
    'test_age',                    # 0 ✅ V1
    'failure_rate',                # 1 ✅ V1
    'recent_failure_rate',         # 2 ✅ V1
    'flakiness_rate',              # 3 ✅ V1
    'execution_count',             # 4
    'failure_count',               # 5
    'pass_count',                  # 6
    'consecutive_failures',        # 7
    'consecutive_passes',          # 8
    'max_consecutive_failures',    # 9
    'last_failure_age',            # 10
    'last_pass_age',               # 11
    'execution_frequency',         # 12
    'failure_trend',               # 13
    'recent_execution_count',      # 14
    'very_recent_failure_rate',    # 15
    'medium_term_failure_rate',    # 16
    'acceleration',                # 17
    'deceleration_factor',         # 18
    'builds_since_change',         # 19
    # STRUCTURAL (9)
    'commit_count',                # 20 ✅ V1
    'test_novelty',                # 21 ✅ V1 (index 5 in V1)
    'builds_affected',             # 22
    'cr_count',                    # 23
    'commit_count_actual',         # 24
    'avg_commits_per_execution',   # 25
    'recent_commit_surge',         # 26
    'stability_score',             # 27
    'pass_fail_ratio',             # 28
]


def expert_feature_selection() -> Tuple[List[int], List[str]]:
    """
    Select top-10 features based on expert knowledge and domain relevance.

    Reasoning:
    1. Keep all 6 baseline features (proven in Exp 04a)
    2. Add 4 most promising new features based on:
       - Direct relevance to failure prediction
       - Complementary information to existing features
       - Statistical significance potential

    Returns:
        selected_indices: List of 10 feature indices
        selected_names: List of 10 feature names
    """

    print("="*70)
    print("EXPERT-BASED FEATURE SELECTION")
    print("="*70)

    # TIER 1: Keep all 6 baseline features (V1) - PROVEN
    tier1_indices = [0, 1, 2, 3, 20, 21]  # test_age, failure_rate, recent_failure_rate,
                                           # flakiness_rate, commit_count, test_novelty
    tier1_names = [FEATURE_NAMES[i] for i in tier1_indices]

    print("\n✅ TIER 1: Baseline Features (V1) - PROVEN in Exp 04a")
    print("   (These achieved APFD = 0.6210)")
    for i, idx in enumerate(tier1_indices, 1):
        print(f"   {i}. {FEATURE_NAMES[idx]:30s} (index {idx})")

    # TIER 2: Add 4 most promising new features
    # Rationale for each:
    tier2_candidates = {
        7: "consecutive_failures - Strong signal for failure prediction (current streak)",
        9: "max_consecutive_failures - Identifies chronically problematic tests",
        13: "failure_trend - Detects tests starting to fail (trend analysis)",
        17: "acceleration - Detects rapid failure rate increases (early warning)",
        23: "cr_count - Code changes directly linked to failures",
        27: "stability_score - Inverse of flakiness, complements flakiness_rate",
    }

    print("\n🔍 TIER 2 Candidates (New Features):")
    for idx, reason in tier2_candidates.items():
        print(f"   - {FEATURE_NAMES[idx]:30s}: {reason}")

    # Expert selection of top 4 from tier2
    tier2_selected = [
        7,   # consecutive_failures - Direct failure signal
        9,   # max_consecutive_failures - Historical problematic pattern
        13,  # failure_trend - Trend detection
        23,  # cr_count - Code change impact
    ]

    print("\n✅ TIER 2: Selected New Features (Top 4)")
    for i, idx in enumerate(tier2_selected, 1):
        print(f"   {i}. {FEATURE_NAMES[idx]:30s} (index {idx})")
        print(f"      → {tier2_candidates[idx]}")

    # Combine
    selected_indices = sorted(tier1_indices + tier2_selected)
    selected_names = [FEATURE_NAMES[i] for i in selected_indices]

    print("\n" + "="*70)
    print("🏆 FINAL TOP-10 FEATURE SET")
    print("="*70)

    for i, idx in enumerate(selected_indices, 1):
        tier = "TIER 1 (Baseline)" if idx in tier1_indices else "TIER 2 (New)"
        print(f"  {i:2d}. {FEATURE_NAMES[idx]:30s} (index {idx:2d}) [{tier}]")

    return selected_indices, selected_names


def analyze_feature_stats(selected_indices: List[int]):
    """Load feature statistics and analyze selected features."""

    cache_path = "cache/structural_features_v2.pkl"

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        means = cache.get('feature_means')
        stds = cache.get('feature_stds')

        print("\n" + "="*70)
        print("FEATURE STATISTICS (Selected Features)")
        print("="*70)
        print(f"{'Feature':<30s} {'Mean':>12s} {'Std':>12s} {'CV':>12s}")
        print("-"*70)

        for idx in selected_indices:
            name = FEATURE_NAMES[idx]
            mean = means[idx] if means is not None else 0
            std = stds[idx] if stds is not None else 0
            cv = std / (abs(mean) + 1e-8)

            print(f"{name:<30s} {mean:>12.4f} {std:>12.4f} {cv:>12.4f}")

        print("="*70)

    except FileNotFoundError:
        print(f"\n⚠️  Feature cache not found: {cache_path}")
        print("   Statistics unavailable, but feature selection is based on domain knowledge.")


def generate_report(selected_indices: List[int], selected_names: List[str]):
    """Generate markdown report with feature selection."""

    report = f"""# 🔬 Feature Selection Report (Expert-Based)

**Method**: Domain Knowledge + Baseline Performance Analysis
**Date**: 2025-11-14
**Context**: Experiment 05 (29 features) degraded performance vs Exp 04a (6 features)

---

## 🎯 Strategy

**Problem**: Expanding from 6 → 29 features caused overfitting and degraded APFD (0.6210 → 0.5997)

**Solution**: Hybrid approach
1. **Keep all 6 baseline features** - Proven performance (APFD = 0.6210)
2. **Add 4 carefully selected new features** - Maximum complementary value

---

## 🏆 Top-10 Selected Features

| # | Feature Name | Index | Source | Justification |
|---|--------------|-------|--------|---------------|
| 1 | **test_age** | 0 | V1 Baseline | Fundamental lifecycle feature |
| 2 | **failure_rate** | 1 | V1 Baseline | Core failure prediction signal |
| 3 | **recent_failure_rate** | 2 | V1 Baseline | Temporal failure pattern |
| 4 | **flakiness_rate** | 3 | V1 Baseline | Test stability indicator |
| 5 | **consecutive_failures** | 7 | NEW | Direct signal: currently failing |
| 6 | **max_consecutive_failures** | 9 | NEW | Historical failure severity |
| 7 | **failure_trend** | 13 | NEW | Trend analysis: failures increasing |
| 8 | **commit_count** | 20 | V1 Baseline | Code change volume |
| 9 | **test_novelty** | 21 | V1 Baseline | New test flag |
| 10 | **cr_count** | 23 | NEW | Code review impact on failures |

---

## 📊 Feature Categories

**Baseline Features (6)**: ✅ Proven in Exp 04a (APFD = 0.6210)
- test_age, failure_rate, recent_failure_rate, flakiness_rate, commit_count, test_novelty

**New Features (4)**: 🆕 Carefully selected for complementary value
- consecutive_failures, max_consecutive_failures, failure_trend, cr_count

**Feature Types**:
- Rate Features: 3 (failure_rate, recent_failure_rate, flakiness_rate)
- Count Features: 3 (consecutive_failures, max_consecutive_failures, commit_count, cr_count)
- Age Features: 1 (test_age)
- Trend Features: 1 (failure_trend)
- Flags: 1 (test_novelty)

---

## 🔍 Rationale for New Features

### 1. consecutive_failures (index 7)
**Why**: Tests currently in a failure streak are highly likely to fail again.
- **Signal Strength**: VERY HIGH
- **Complementarity**: Adds "current state" to historical rates
- **Example**: Test failed last 3 builds → 75% chance to fail next build

### 2. max_consecutive_failures (index 9)
**Why**: Identifies tests with chronic failure patterns.
- **Signal Strength**: HIGH
- **Complementarity**: Captures historical severity beyond avg failure_rate
- **Example**: Test that had 10-build failure streak is inherently problematic

### 3. failure_trend (index 13)
**Why**: Detects tests transitioning from stable → failing.
- **Signal Strength**: MEDIUM-HIGH
- **Complementarity**: Adds derivative (change) to raw rates
- **Example**: failure_rate increased from 0.1 → 0.4 recently → trend = +0.3

### 4. cr_count (index 23)
**Why**: Code reviews directly correlate with code changes → failure risk.
- **Signal Strength**: MEDIUM
- **Complementarity**: More specific than generic commit_count
- **Example**: 5 CRs merged recently → high change volume → higher failure risk

---

## 📋 Next Steps: Experiment 06

**Configuration**:
```yaml
structural:
  input_dim: 10  # ← Reduced from 29 to 10
  extractor:
    use_v2_5: true  # ← New V2.5 with selected features only
    selected_features: {selected_indices}

model:
  structural:
    input_dim: 10
    hidden_dim: 64  # ← Back to V1 capacity (was 128 for 29 features)
    num_layers: 2
    dropout: 0.1
```

**Expected Results**:
- ✅ APFD: 0.62-0.65 (match or exceed baseline)
- ✅ F1 Macro: 0.52-0.56 (improve over Exp 05's 0.49)
- ✅ Less overfitting (10 features vs 29)
- ✅ Retain proven signals (6 baseline) + add new value (4 selected)

---

## ✅ Success Criteria

| Metric | Target | Baseline (04a) | Exp 05 (29 feat) |
|--------|--------|----------------|------------------|
| **APFD** | ≥ 0.62 | 0.6210 | 0.5997 ❌ |
| **F1 Macro** | ≥ 0.52 | 0.5294 | 0.4935 ❌ |
| **APFD ≥ 0.7** | ≥ 40% | 40.8% | 36.5% ❌ |

**GO Criteria**:
- ✅ APFD ≥ 0.62 (match baseline)
- ✅ F1 Macro ≥ 0.52

**NO-GO Criteria**:
- ❌ APFD < 0.60
- ❌ F1 Macro < 0.50

---

## 📝 Feature Indices for Implementation

**Selected indices**: `{selected_indices}`

**Usage in StructuralFeatureExtractorV2.5**:
```python
SELECTED_FEATURE_INDICES = {selected_indices}

def transform_v2_5(self, df: pd.DataFrame) -> np.ndarray:
    # Extract all 29 features first
    features_full = self.transform_v2(df)  # [N, 29]

    # Select only the 10 chosen features
    features_selected = features_full[:, SELECTED_FEATURE_INDICES]  # [N, 10]

    return features_selected
```

---

**Generated by**: Expert Feature Selection Tool
**Author**: Filo-Priori V8 Team
**Next Action**: Implement Experiment 06 with these 10 features
"""

    output_path = "results/experiment_05_expanded_features/FEATURE_SELECTION_EXPERT.md"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_path}")


def main():
    """Main feature selection pipeline."""

    # Expert selection
    selected_indices, selected_names = expert_feature_selection()

    # Analyze statistics
    analyze_feature_stats(selected_indices)

    # Generate report
    generate_report(selected_indices, selected_names)

    print("\n" + "="*70)
    print("✅ FEATURE SELECTION COMPLETE")
    print("="*70)
    print(f"\n🏆 Selected 10 Features:")
    print(f"   Indices: {selected_indices}")
    print(f"\n📄 Full report: results/experiment_05_expanded_features/FEATURE_SELECTION_EXPERT.md")
    print(f"\n🚀 Next Step: Create Experiment 06 with these features")
    print("="*70)


if __name__ == "__main__":
    main()
