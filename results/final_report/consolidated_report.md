# Filo-Priori v9 - Consolidated Final Report

**Deep Learning-based Test Case Prioritization System**

Generated: 2025-11-26 13:27:18

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

| Method | Mean APFD | 95% CI | Significance |
|--------|-----------|--------|-------------|
| FailureRate | 0.6289 | [0.597, 0.660] | nan |
| XGBoost | 0.6171 | [0.583, 0.649] | nan |
| Filo-Priori | 0.6171 | [0.586, 0.648] | nan |
| GreedyHistorical | 0.6138 | [0.582, 0.647] | nan |
| LogisticRegression | 0.5964 | [0.559, 0.631] | nan |
| RandomForest | 0.5910 | [0.556, 0.627] | nan |
| Random | 0.5596 | [0.533, 0.584] | *** |
| RecentFailureRate | 0.5454 | [0.515, 0.578] | *** |
| Recency | 0.5240 | [0.490, 0.557] | *** |

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
