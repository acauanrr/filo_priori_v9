# Future Improvements - Filo-Priori

This document tracks identified improvement opportunities for the Filo-Priori model.
Organized by priority based on potential impact.

---

## High Priority

### 1. KNN Orphan Strategy Low Variance

**Problem**: The KNN strategy for handling orphan test cases (tests not in the training graph) produces nearly identical scores for all orphans.

**Evidence from V3 log**:
```
KNN orphan scores computed: 22 samples
  Min=0.2011, Max=0.2011, Mean=0.2011, Std=0.0000
```

All 22 orphans in a batch receive the same score (0.2011), which defeats the purpose of using KNN to differentiate them.

**Impact**: 22.7% of full test samples require imputation (scores around 0.5), suggesting many orphans.

**Suggested Solutions**:
1. **Increase `k_neighbors`** (currently 10) to get more diverse neighbor sets
2. **Use different distance metrics** (L2 instead of cosine)
3. **Apply temperature scaling** to similarity scores before weighting
4. **Blend with structural features** instead of relying solely on semantic similarity

**Config to modify**:
```yaml
ranking:
  orphan_strategy:
    k_neighbors: 20  # Increase from 10
    similarity_metric: "euclidean"  # Try different metric
```

---

### 2. High Imputation Rate (22.7%)

**Problem**: Nearly a quarter of test samples in full test.csv have scores around 0.5 (imputed).

**Evidence**:
```
FULL TEST SET EVALUATION:
  Samples: 47006
  Scores around 0.5 (imputed): 10690 (22.7%)
  Scores < 0.4: 26066 (55.5%)
  Scores > 0.6: 10250 (21.8%)
```

**Root Cause**: Test cases that:
- Were not in the training set
- Have no graph edges to other test cases
- Have very different semantic embeddings

**Suggested Solutions**:
1. **Lower semantic threshold** for graph construction:
   ```yaml
   graph:
     semantic_threshold: 0.65  # From 0.75
   ```
2. **Increase `semantic_top_k`** to create more edges:
   ```yaml
   graph:
     semantic_top_k: 10  # From 5
   ```
3. **Use fallback heuristic** for orphans based on:
   - Test case age
   - Similar test name patterns
   - Module/package similarity

---

### 3. Recall vs Precision Trade-off

**Problem**: Current optimal threshold (0.44) gives 30% recall but only 20% precision for Fail class.

**Current Metrics** (threshold=0.44):
- Precision (Fail): 0.2000
- Recall (Fail): 0.3023
- F1 (Fail): 0.2407

**Suggested Solutions**:
1. **Dynamic threshold per build**: Use different thresholds based on build characteristics
2. **Adjust sampling ratio**: Try 15:1 instead of 10:1
3. **Threshold ensemble**: Average predictions from multiple thresholds

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

| Improvement | Difficulty | Expected Impact | Status |
|-------------|------------|-----------------|--------|
| KNN variance fix | Medium | High | TODO |
| Lower semantic threshold | Easy | Medium | TODO |
| hidden_dim warning | Easy | Low | TODO |
| Threshold fine search | Easy | Low | TODO |
| More edge types | Hard | Medium | TODO |
| Dynamic threshold | Medium | Medium | TODO |

---

*Last Updated: December 2025*
