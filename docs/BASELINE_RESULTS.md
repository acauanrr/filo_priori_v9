# Filo-Priori Baseline Results

This document tracks the official baseline results for the Filo-Priori model.
All future experiments should be compared against these benchmarks.

---

## Current Baseline (V3 - December 2025)

### Industrial Dataset (01_industry)

| Metric | Value | Notes |
|--------|-------|-------|
| **APFD (277 builds)** | **0.6661** | Full test.csv with all failure builds |
| **APFD (test split)** | **0.7086** | 64 builds from validation split |
| **F1 Macro** | **0.5875** | Balanced metric |
| **Precision (Fail)** | 0.2000 | |
| **Recall (Fail)** | 0.3023 | Significantly improved from 0-3% |
| **F1 (Fail)** | 0.2407 | |
| **AUROC** | 0.7141 | |
| **AUPRC Macro** | 0.5889 | |
| **Accuracy** | 0.9251 | |
| **Optimal Threshold** | 0.44 | For F1 Macro optimization |

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

| Method | APFD | vs Baseline |
|--------|------|-------------|
| **Filo-Priori V3** | **0.6661** | **+3.8%** vs V1 |
| Filo-Priori V1 | 0.6503 | baseline |
| FailureRate | 0.6289 | -2.6% |
| XGBoost | 0.6171 | -5.1% |
| Random | 0.5596 | -14.2% |

---

## Improvement Opportunities (Future Work)

### High Priority

1. **KNN Orphan Strategy Variance**
   - Current: All orphans get similar scores (~0.2011)
   - Impact: 22.7% of test samples affected
   - Suggestion: Increase `k_neighbors` or use different similarity metric

2. **Recall vs Precision Trade-off**
   - Current: 30% recall, 20% precision for Fail class
   - Suggestion: Adjust decision threshold or sampling ratio

### Medium Priority

3. **Graph Connectivity**
   - Many test cases are "orphans" (not in training graph)
   - Suggestion: Lower `semantic_threshold` or increase `semantic_top_k`

4. **Threshold Optimization**
   - Current search: [0.1, 0.9] with step 0.02
   - Suggestion: Finer search around optimal (0.44)

### Low Priority

5. **Model Architecture**
   - Current warning: hidden_dim mismatch
   - Not affecting results but should be fixed

---

## How to Compare Against Baseline

```bash
# Run baseline configuration
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Expected results:
# - APFD (full test): ~0.6661
# - APFD (test split): ~0.7086
# - F1 Macro: ~0.5875
```

### Metrics to Report

1. **APFD (277 builds)** - Primary metric for prioritization
2. **F1 Macro** - Classification balance
3. **Recall (Fail)** - Fault detection sensitivity
4. **AUROC** - Discrimination ability

---

## Version History

| Version | Date | APFD | Key Changes |
|---------|------|------|-------------|
| V3 | Dec 2025 | **0.6661** | Single balancing mechanism |
| V2 | Nov 2025 | ~0.55 | Added balanced sampling (broken) |
| V1 | Nov 2025 | 0.6503 | Original dual_stream |

---

*Last Updated: December 2025*
