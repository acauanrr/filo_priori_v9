# Filo-Priori Baseline Results

This document tracks the official baseline results for the Filo-Priori model.
All future experiments should be compared against these benchmarks.

---

## Current Baseline (V3 - Validated December 2025)

### Industrial Dataset (01_industry)

| Metric | Value | Notes |
|--------|-------|-------|
| **Mean APFD (277 builds)** | **0.7595** | Primary metric - all builds with failures |
| **Median APFD** | **0.7944** | Robust central tendency |
| **Std APFD** | 0.1894 | Standard deviation |
| **Min APFD** | 0.0833 | Lowest performing build |
| **Max APFD** | 1.0000 | Perfect prioritization (23 builds) |
| **APFD (test split)** | **0.6966** | 64 builds from validation split |
| **Val F1 Macro** | **0.5899** | Classification performance |
| **Test F1 Macro** | **0.5870** | Generalization |
| **Optimal Threshold** | 0.2777 | F-beta optimization |

### APFD Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| APFD = 1.0 (perfect) | 23 | 8.3% |
| APFD ≥ 0.7 (high) | 188 | 67.9% |
| APFD ≥ 0.5 (acceptable) | 247 | 89.2% |
| APFD < 0.5 (low) | 30 | 10.8% |

### Validation Summary

- ✅ **277 builds** with failures verified against test.csv
- ✅ **5,085 test cases** total (mean 18.4 per build)
- ✅ **No data leakage** (grouped splits by Build_ID)
- ✅ **All build IDs unique** and verified against source

### Configuration

- **Config File**: `configs/experiment_industry_optimized_v3.yaml`
- **Model Type**: `dual_stream` (DualStreamModelV8)
- **Date**: December 2025

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 3e-5 | Proven optimal for this task |
| Batch Size | 16 | Memory efficient |
| Epochs | 50 | With early stopping |
| Loss Type | `weighted_focal` | Best for class imbalance |
| `use_class_weights` | **false** | Single balancing mechanism |
| `focal_alpha` | 0.5 | Neutral (no class preference in focal) |
| `focal_gamma` | 2.0 | Moderate hard example focus |
| `use_balanced_sampling` | **true** | Primary balancing mechanism |
| `minority_weight` | 1.0 | |
| `majority_weight` | 0.1 | 10:1 effective ratio |
| Dropout | 0.15-0.20 | Moderate regularization |

### Critical Insight: Single Balancing Mechanism

**Problem Solved**: Previous versions had "mode collapse" issues:
- V1: Model predicted all Pass (minority class ignored)
- V2: Model predicted all Fail (over-compensation)

**Root Cause**: Triple compensation from:
1. `class_weights` (~19x for minority)
2. `balanced_sampling` (20x oversampling)
3. `focal_alpha` (0.85 = more weight to minority)

Combined effect: ~323x weight to minority class.

**Solution (V3)**: Use **ONLY ONE** balancing mechanism:
- `balanced_sampling` with 10:1 ratio
- `use_class_weights: false`
- `focal_alpha: 0.5` (neutral)

---

## Historical Baselines

### V1 - Original (November 2025)

| Metric | Value |
|--------|-------|
| APFD | 0.6503 |
| F1 Macro | ~0.50 |
| Recall (Fail) | ~3% |

**Issues**: Mode collapse to Pass class

### V2 - Balanced Sampling Attempt (November 2025)

| Metric | Value |
|--------|-------|
| APFD | ~0.55 |
| Recall (Fail) | ~100% |

**Issues**: Mode collapse to Fail class (triple compensation)

---

## Baseline Comparison (Same Dataset)

| Method | APFD | vs Filo-Priori V3 |
|--------|------|-------------------|
| **Filo-Priori V3 (Latest)** | **0.7595** | -- |
| Filo-Priori V1 | 0.6503 | -14.4% |
| FailureRate | 0.6289 | -17.2% |
| XGBoost | 0.6171 | -18.7% |
| Random | 0.5596 | -26.3% |

---

## Key Improvements in V3 (What Drove the APFD Gain)

1. **Dense Multi-Edge Graph**: semantic_top_k=10, threshold=0.65, temporal/component edges → fewer orphans, better message passing
2. **High-Variance Orphan Scorer**: k=20, euclidean, structural blend, temperature → eliminated flat scores; orphans now differentiated
3. **Balanced Sampling + Tuned Threshold**: Two-phase search with f_beta 0.8 → improved early-fail capture
4. **DeepOrder Features + Priority History**: Informative structural priors for rarely failing tests
5. **Strict Build-Level Split**: No leakage; metrics reflect genuine generalization

---

## Outlier Analysis

### Builds with Low APFD (< 0.3) - 7 builds

| Build ID | APFD | Test Cases |
|----------|------|------------|
| T2SR33.54 | 0.0833 | 6 |
| U3UX34.1 | 0.1167 | 29 |
| S3SG32.39-90-1 | 0.1406 | 31 |
| UTPN34.176 | 0.1667 | 3 |
| UTP34.79 | 0.2000 | 15 |
| T3TDC33.3 | 0.2500 | 2 |
| T1TH33.75-12-6 | 0.2778 | 9 |

### Builds with Perfect APFD (= 1.0) - 23 builds

All 23 builds with APFD = 1.0 have exactly **1 test case** each, which is expected behavior (single failing test ranked first = perfect APFD).

---

## How to Compare Against Baseline

```bash
# Run baseline configuration
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Expected results (validated December 2025):
# - Mean APFD (277 builds): 0.7595
# - Median APFD: 0.7944
# - APFD (test split, 64 builds): 0.6966
# - Val F1 Macro: 0.5899
# - Test F1 Macro: 0.5870
# - APFD >= 0.7: 67.9% (188/277)
# - APFD >= 0.5: 89.2% (247/277)
```

### Metrics to Report

1. **Mean APFD (277 builds)** - Primary metric for prioritization
2. **Median APFD** - Robust central tendency
3. **APFD Distribution** - % of builds with APFD ≥ 0.7 and ≥ 0.5
4. **F1 Macro** - Classification balance
5. **Test Split APFD** - Generalization check

---

## Version History

| Version | Date | Mean APFD | Median APFD | Key Changes |
|---------|------|-----------|-------------|-------------|
| **V3 (Current)** | Dec 2025 | **0.7595** | **0.7944** | Dense graph, high-variance orphan KNN, DeepOrder features |
| V2 | Nov 2025 | ~0.55 | -- | Added balanced sampling (broken - mode collapse) |
| V1 | Nov 2025 | 0.6503 | -- | Original dual_stream |

---

*Last Updated: December 2025 (Validated)*
