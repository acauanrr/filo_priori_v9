# Filo-Priori V10 - Consolidated Final Report

**Deep Learning-based Test Case Prioritization System**

Generated: 2025-11-29

---

## Executive Summary

Filo-Priori achieves state-of-the-art results on two complementary datasets:

| Dataset | APFD | Improvement | Test Builds |
|---------|------|-------------|-------------|
| **01_industry** | **0.6413** | +14.6% vs random | 277 |
| **02_rtptorrent** | **0.8376** | +2.02% vs recently_failed | 1,250 |

---

## Dataset 1: Industrial QTA (Classification Mode)

### Key Metrics

| Metric | Value |
|--------|-------|
| Mean APFD | **0.6413** [0.612, 0.672] |
| vs Random | **+14.6%** (p < 0.001) *** |
| vs FailureRate | **+2.0%** |
| Total Builds | 277 |
| Total Executions | 52,102 |
| Pass:Fail Ratio | 37:1 |

### Baseline Comparison

| Method | Mean APFD | 95% CI | vs Random |
|--------|-----------|--------|-----------|
| **Filo-Priori** | **0.6413** | [0.612, 0.672] | **+14.6%** |
| FailureRate | 0.6289 | [0.601, 0.658] | +12.4% |
| XGBoost | 0.6171 | [0.589, 0.646] | +10.3% |
| GreedyHistorical | 0.6138 | [0.585, 0.643] | +9.7% |
| LogisticRegression | 0.5964 | [0.568, 0.625] | +6.6% |
| RandomForest | 0.5910 | [0.563, 0.620] | +5.6% |
| Random | 0.5596 | [0.531, 0.588] | baseline |
| RecentFailureRate | 0.5454 | [0.517, 0.574] | -2.5% |
| Recency | 0.5240 | [0.496, 0.553] | -6.4% |

### Ablation Study

| Component | Contribution | p-value |
|-----------|--------------|---------|
| **Graph Attention (GAT)** | **+17.0%** | < 0.001*** |
| Structural Stream | +5.3% | < 0.001*** |
| Focal Loss | +4.6% | < 0.001*** |
| Class Weighting | +3.5% | 0.002** |
| Semantic Stream | +1.9% | 0.087 |

**Key Finding:** Graph Attention Networks are the most critical component (+17.0%).

### Temporal Cross-Validation

| Method | Mean APFD | 95% CI |
|--------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

**Key Finding:** Model is temporally robust (APFD range: 0.619-0.663).

---

## Dataset 2: RTPTorrent (V10 Full - Learning-to-Rank Mode)

### Aggregate Results (20 Projects)

| Metric | Value |
|--------|-------|
| Mean APFD | **0.8376** |
| Projects | 20 Java projects |
| Test Builds | 1,250 |
| Model | LightGBM LambdaRank |
| Features | 16 ranking-optimized |

### Baseline Comparison

| Baseline | APFD | Model Improvement |
|----------|------|-------------------|
| **Filo-Priori V10** | **0.8376** | -- |
| recently_failed | 0.8209 | **+2.02%** |
| random | 0.4940 | +69.56% |
| untreated | 0.3574 | +134.32% |
| matrix_naive | 0.5693 | +47.11% |
| matrix_conditional | 0.5132 | +63.21% |
| optimal_duration | 0.5934 | +41.15% |
| optimal_failure (oracle) | 0.9249 | -9.45% |

**Key Finding:** Model outperforms **6 out of 7 baselines**.

### Top-5 Projects by APFD

| Project | APFD | Test Builds | vs recently_failed |
|---------|------|-------------|-------------------|
| apache/sling | 0.9922 | 163 | +2.18% |
| neuland/jade4j | 0.9799 | 20 | +0.88% |
| eclipse/jetty.project | 0.9789 | 66 | +1.27% |
| facebook/buck | 0.9722 | 69 | +1.97% |
| deeplearning4j/dl4j | 0.9277 | 114 | +0.46% |

### Feature Importance (Aggregate)

| Feature | Importance |
|---------|------------|
| novelty_score | High |
| base_risk | High |
| execution_frequency | High |
| time_decay_score | Medium |
| historical_rate | Medium |
| total_executions | Medium |

---

## Model Configurations

### Industry Dataset (V9 Classification)

```yaml
model: DualStreamModelV8
semantic_dim: 1536  # SBERT all-mpnet-base-v2
structural_features: 10
gat_layers: 1
gat_heads: 2
loss: WeightedFocalLoss (alpha=0.75, gamma=2.5)
optimizer: AdamW (lr=3e-5)
```

### RTPTorrent Dataset (V10 LambdaRank)

```yaml
model: LightGBM LambdaRank
features: 16 ranking-optimized
online_learning: EMA (alpha=0.8)
objective: lambdarank
n_estimators: 100
max_depth: 5
```

---

## Key Insights

### What Works

1. **Graph Attention Networks** are crucial for TCP (+17.0% contribution)
2. **Historical features** (failure_rate, time_decay) are most predictive
3. **Simpler architectures** (1 GAT layer) outperform complex ones
4. **LambdaRank** is effective for ranking optimization

### What Doesn't Work

1. **Deep GNN architectures** overfit on test relationship graphs
2. **Balanced sampling** degrades performance (use class weighting instead)
3. **Cross-attention fusion** shows marginal/negative contribution

---

## Reproducibility

### Running Experiments

```bash
# Industry dataset
python run_experiment.py --dataset industry

# RTPTorrent dataset (all 20 projects)
python run_experiment.py --dataset rtptorrent

# Or run both
python run_experiment.py --dataset all
```

### Results Location

| Dataset | Results Path |
|---------|--------------|
| Industry | `results/experiment_hybrid_phylogenetic/` |
| RTPTorrent | `results/experiment_v10_rtptorrent_full/` |

---

## Citation

```bibtex
@article{filo_priori_v10_2025,
  title={Filo-Priori: A Dual-Stream Deep Learning Approach to
         Test Case Prioritization},
  author={Ribeiro, Acauan C.},
  journal={IEEE Transactions on Software Engineering},
  year={2025},
  note={Under Review}
}
```

---

*Report generated by Filo-Priori V10 analysis pipeline.*
*Last updated: 2025-11-29*
