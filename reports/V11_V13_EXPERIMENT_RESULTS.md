# Filo-Priori V11-V13 Experiment Results Report

## Executive Summary

This report documents the experiments conducted to improve Filo-Priori's APFD performance on the 01_industry dataset to surpass DeepOrder (0.6500 APFD).

**Current Status**: All experiments achieved ~0.638 APFD, remaining **1.7-1.8% behind DeepOrder**.

## Results Summary

| Version | Structural Features | Ranking Weight | APFD | vs DeepOrder |
|---------|---------------------|----------------|------|--------------|
| Baseline | V2.5 (10 features) | 0.3 | 0.6383 | -1.80% |
| V11 | V3 (14 features) | 0.5 | 0.6386 | -1.75% |
| V12 | V3 (14 features) | 0.2 | 0.6388 | -1.72% |
| V13 | V3 (14 features) | 0.4 | 0.6384 | -1.78% |
| **DeepOrder** | - | - | **0.6500** | - |

**Best Result**: V12 with 0.6388 APFD (+0.0005 over baseline)

## What Was Implemented

### 1. V3 Structural Feature Extractor

New file: `src/preprocessing/structural_feature_extractor_v3.py`

**14 features** (4 new over V2.5's 10):

| Feature | Description | Source |
|---------|-------------|--------|
| test_age | Build count since first execution | V2.5 |
| failure_rate | Overall failure rate | V2.5 |
| recent_failure_rate | Failure rate in last N builds | V2.5 |
| **weighted_failure_rate** | Decay-weighted failure rate | **NEW** |
| flakiness_rate | Pass/fail transition frequency | V2.5 |
| failure_trend | Recent vs overall failure rate | V2.5 |
| **last_verdict** | 0/1 - Last execution result | **NEW** |
| **time_since_failure** | Normalized builds since last fail | **NEW** |
| consecutive_failures | Current failure streak | V2.5 |
| max_consecutive_failures | Max historical streak | V2.5 |
| **execution_frequency** | Executions per build span | **NEW** |
| test_novelty | Is this a new test? | V2.5 |
| commit_count | Recent commits count | V2.5 |
| cr_count | Change request count | V2.5 |

### 2. Configuration Changes

**V11** (aggressive ranking):
- V3 features (14 dim)
- `recent_window: 10` (extended from 5)
- `ranking.weight: 0.5`
- Result: Training instability, early stopping at epoch 16

**V12** (stable training):
- V3 features (14 dim)
- `ranking.weight: 0.2` (conservative)
- `warmup_epochs: 8`, `patience: 20`
- Result: Stable training, slight improvement

**V13** (balanced ranking):
- V3 features (14 dim)
- `ranking.weight: 0.4`
- `margin: 0.7`, `hard_negative_percent: 0.3`
- Result: Still unstable, no improvement

## Key Observations

### 1. Training Instability with High Ranking Weight
- Ranking weight > 0.3 causes validation F1 to oscillate
- Model alternates between predicting mostly Pass and mostly Fail
- Best model often saved in epoch 1 (before ranking loss is active)

### 2. New Features Have Minimal Impact
- V3's 14 features vs V2.5's 10 features: ~0.0005 APFD difference
- The new DeepOrder-style features don't provide significant lift
- Possible explanations:
  - Features are correlated with existing features
  - Graph propagation already captures similar information
  - Different data characteristics than DeepOrder's evaluation

### 3. APFD Distribution Stability
```
APFD>=0.9: 21-22% across all experiments
APFD>=0.7: 43-44% across all experiments
APFD>=0.5: 69-71% across all experiments
```
The distribution is remarkably stable regardless of configuration.

## Gap Analysis: Filo-Priori vs DeepOrder

| Aspect | Filo-Priori | DeepOrder |
|--------|-------------|-----------|
| Architecture | GAT + Dual-stream | RNN/LSTM |
| Feature Type | Static per-test | Sequential per-build |
| Ranking | Pairwise ranking loss | Pointwise RNN output |
| Graph | Co-failure relationships | N/A |
| Semantic | SBERT embeddings | N/A |

**Potential Reasons for Gap**:
1. DeepOrder models temporal sequences explicitly (RNN/LSTM)
2. DeepOrder may use different evaluation protocol
3. Per-build vs per-test prediction granularity

## Recommendations for Future Work

### Short-term (to beat 0.65 APFD)
1. **Temporal modeling**: Add LSTM/Transformer layer for sequence modeling
2. **Ensemble methods**: Combine multiple model predictions
3. **Threshold optimization**: Better post-hoc ranking calibration

### Medium-term (research directions)
1. Investigate per-build prediction architecture
2. Study feature importance and ablation
3. Compare directly with DeepOrder implementation

## Files Created/Modified

```
NEW FILES:
- src/preprocessing/structural_feature_extractor_v3.py (605 lines)
- configs/experiment_industry_v11.yaml
- configs/experiment_industry_v12.yaml
- configs/experiment_industry_v13.yaml
- scripts/compare_v11_deeporder.py
- reports/V11_V13_EXPERIMENT_RESULTS.md

MODIFIED:
- main.py (added V3 extractor support)
```

## Conclusion

The V11-V13 experiments show that:
1. Adding DeepOrder-style features provides marginal improvement (+0.0005)
2. The ranking loss weight has minimal impact on final APFD
3. A fundamentally different approach (e.g., temporal modeling) may be needed to surpass DeepOrder

**Best Configuration**: V12 with V3 features and ranking weight 0.2

---
*Report generated: 2025-11-29*
*Experiments conducted by Filo-Priori V11 Team*
