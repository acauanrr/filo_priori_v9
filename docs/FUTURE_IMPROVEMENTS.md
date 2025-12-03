# Future Improvements - Filo-Priori

This document tracks identified improvement opportunities for the Filo-Priori model.
Organized by priority based on potential impact.

---

## High Priority

### 1. KNN Orphan Strategy Low Variance - ✅ RESOLVED

**Problem**: The KNN strategy for handling orphan test cases (tests not in the training graph) produces nearly identical scores for all orphans.

**Evidence from V3 log (BEFORE FIX)**:
```
KNN orphan scores computed: 22 samples
  Min=0.2011, Max=0.2011, Mean=0.2011, Std=0.0000
```

**AFTER FIX** (validated December 2025):
```
Orphan stats: mean 0.3717, std 0.0462 (no flat 0.2011 plateaus)
```

**Solution Implemented**:
- New shared `compute_orphan_scores` (KNN + structural blend + temperature scaling + alpha blend)
- Config: k=20, euclidean metric, structural_weight=0.35, temperature=0.7, alpha_blend=0.55
- Result: Orphan scores now differentiated with healthy variance

**Config (current)**:
```yaml
ranking:
  orphan_strategy:
    k_neighbors: 20
    similarity_metric: "euclidean"
    structural_weight: 0.35
    temperature: 1.5
```

---

### 2. High Imputation Rate (22.7%) - ✅ RESOLVED

**Problem**: Nearly a quarter of test samples in full test.csv have scores around 0.5 (imputed).

**Solution Implemented**:
- **Denser multi-edge graph**: semantic_threshold=0.65, semantic_top_k=10, temporal/component edges
- **Result**: 77.4% in-graph coverage (reduced orphans significantly)
- **Orphan scoring**: KNN + structural blend replaces flat 0.5 imputations

**Current Config**:
```yaml
graph:
  semantic_threshold: 0.65
  semantic_top_k: 10
  edge_types: [co_failure, co_success, semantic, temporal, component]
```

---

### 3. Recall vs Precision Trade-off - ✅ IMPROVED

**Solution Implemented**:
- **Two-phase threshold search** (coarse 0.05, fine 0.01) with f_beta target (beta=0.8)
- **Optimal threshold**: 0.2777 (from F-beta optimization)
- **Result**: Better early-fail capture while maintaining precision balance

**Remaining Opportunity**: Dynamic threshold per build based on historical failure rates (TODO)

---

## Medium Priority

### 4. Graph Connectivity Improvement

**Problem**: Many tests are disconnected from the graph (orphans), limiting GAT's effectiveness.

**Suggested Solutions**:
1. **Add more edge types**:
   - Module co-location edges
   - Author/committer similarity
   - Execution time similarity
2. **Lower `min_co_occurrences`** threshold
3. **Use transitive closure** for co-failure edges

---

### 5. Threshold Search Optimization

**Current**: Search [0.1, 0.9] with step 0.02

**Problem**: Coarse search may miss optimal threshold.

**Suggested Solutions**:
1. **Two-phase search**: Coarse (0.05 step) then fine (0.005 step)
2. **Different metrics per threshold**: Optimize for recall at low thresholds, precision at high
3. **Build-specific thresholds**: Based on historical failure rate

---

### 6. Warning: hidden_dim Mismatch

**Warning from log**:
```
WARNING: hidden_dim (256) differs from gnn.hidden_dim (128)
```

**Problem**: Model configuration inconsistency between fusion and GNN dimensions.

**Solution**: Align dimensions in config:
```yaml
model:
  fusion:
    input_dim: 320  # Should be gnn_hidden_dim * num_heads + semantic_hidden_dim
```

---

## Low Priority

### 7. Batch Size Optimization

**Current**: batch_size=16

**Consideration**: Larger batches (32, 64) might:
- Provide more stable gradients
- Better utilize GPU
- Improve generalization

**Trade-off**: Larger batch may require lower learning rate.

---

### 8. Learning Rate Scheduler

**Current**: Cosine annealing with warmup

**Alternative approaches**:
1. OneCycleLR for faster convergence
2. ReduceLROnPlateau for adaptive learning
3. Warmup with step decay

---

### 9. Embedding Model Upgrade

**Current**: `sentence-transformers/all-mpnet-base-v2`

**Alternatives**:
1. `sentence-transformers/all-MiniLM-L12-v2` (faster, smaller)
2. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (if multilingual needed)
3. Domain-adapted SBERT (fine-tune on test case descriptions)

---

## Experimental Ideas

### A. Ensemble Models

Combine multiple models with different configurations:
- Model 1: High recall config
- Model 2: High precision config
- Final: Weighted average based on confidence

### B. Temporal Features

Add time-based features:
- Day of week
- Time since last CI run
- Sprint/release cycle phase

### C. Attention Analysis

Analyze GAT attention weights to:
- Understand which test relationships matter most
- Prune low-importance edges
- Guide feature engineering

### D. Active Learning

Implement online learning:
- Update model after each build
- Prioritize labeling ambiguous cases
- Adapt to concept drift

---

## Implementation Checklist

| Improvement | Difficulty | Expected Impact | Status | Result |
|-------------|------------|-----------------|--------|--------|
| KNN variance fix | Medium | High | ✅ DONE | Orphan scores: std=0.0462 (was 0.0) |
| Lower semantic threshold | Easy | Medium | ✅ DONE | 77.4% in-graph coverage |
| hidden_dim warning | Easy | Low | ✅ DONE | Auto-align in main.py |
| Threshold fine search | Easy | Low | ✅ DONE | Two-phase coarse/fine |
| Dense multi-edge graph | Medium | High | ✅ DONE | semantic_top_k=10, temporal/component edges |
| DeepOrder features | Medium | Medium | ✅ DONE | 19 total features (10 base + 9 DeepOrder) |
| More edge types | Hard | Medium | ✅ DONE | Temporal + component edges added |
| Dynamic threshold | Medium | Medium | TODO | -- |

---

## Validated Results (December 2025)

The improvements above achieved:

| Metric | Before (V1) | After (V3) | Improvement |
|--------|-------------|------------|-------------|
| **Mean APFD** | 0.6503 | **0.7595** | **+16.8%** |
| **Median APFD** | -- | **0.7944** | -- |
| APFD ≥ 0.7 | ~50% | **67.9%** | +17.9pp |
| APFD ≥ 0.5 | ~75% | **89.2%** | +14.2pp |
| Orphan score std | 0.0 | 0.0462 | Variance restored |

---

*Last Updated: December 2025 (Validated)*
