# Filo-Priori V10: Dual-Stream Deep Learning for Test Case Prioritization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![APFD](https://img.shields.io/badge/APFD-0.7595-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A dual-stream deep learning approach combining **semantic understanding** (SBERT) with **structural patterns** (Graph Attention Networks) for intelligent test case prioritization in CI/CD pipelines.

---

## Key Results (Validated December 2025)

### Industrial Dataset - 277 Builds with Failures

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.7595** | Primary ranking metric |
| **Median APFD** | **0.7944** | Robust central tendency |
| **APFD ≥ 0.7** | **67.9%** | 188/277 builds |
| **APFD ≥ 0.5** | **89.2%** | 247/277 builds |
| **APFD = 1.0** | **8.3%** | 23/277 builds (perfect) |
| **vs Random** | **+35.7%** | Significant improvement |
| **vs FailureRate** | **+20.8%** | Beats traditional baseline |

### Improvement Over Previous Versions

| Version | APFD | Change |
|---------|------|--------|
| V1 (baseline) | 0.6503 | - |
| **V10 (current)** | **0.7595** | **+16.8%** |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run best configuration
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Expected output:
# Mean APFD (277 builds): 0.7595
# Median APFD: 0.7944
# APFD >= 0.7: 67.9%
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FILO-PRIORI V10 ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐                      ┌──────────────────┐        │
│  │  SEMANTIC INPUT  │                      │ STRUCTURAL INPUT │        │
│  │  TC_Summary      │                      │  19 Features     │        │
│  │  TC_Steps        │                      │  (10 base +      │        │
│  │  Commit Messages │                      │   9 DeepOrder)   │        │
│  └────────┬─────────┘                      └────────┬─────────┘        │
│           │                                         │                   │
│           ▼                                         ▼                   │
│  ┌──────────────────┐                      ┌──────────────────┐        │
│  │  SBERT Encoder   │                      │  Multi-Edge      │        │
│  │  all-mpnet-base  │                      │  Test Graph      │        │
│  │  1536-dim output │                      │  (~32K edges)    │        │
│  └────────┬─────────┘                      └────────┬─────────┘        │
│           │                                         │                   │
│           ▼                                         ▼                   │
│  ┌──────────────────┐                      ┌──────────────────┐        │
│  │  SEMANTIC STREAM │                      │ STRUCTURAL STREAM│        │
│  │  FFN [1536→256]  │                      │  GAT [19→256]    │        │
│  └────────┬─────────┘                      └────────┬─────────┘        │
│           │                                         │                   │
│           └─────────────────┬───────────────────────┘                   │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │ CROSS-ATTENTION  │                                 │
│                    │ FUSION (512-dim) │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │   CLASSIFIER     │                                 │
│                    │  MLP [512→2]     │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │ ORPHAN HANDLING  │◄─── 22.7% of tests              │
│                    │ KNN + Structural │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
│                             ▼                                           │
│                       P(Fail) per test → Ranked List                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Makes V10 Different?

### 1. Dense Multi-Edge Graph

Connects tests through 5 relationship types:

| Edge Type | Weight | Purpose |
|-----------|--------|---------|
| Co-failure | 1.0 | Tests that fail together |
| Co-success | 0.5 | Tests that pass together |
| Semantic | 0.3 | Similar descriptions (top-10) |
| Temporal | 0.2 | Sequential execution patterns |
| Component | 0.4 | Same module/package |

**Config:** `semantic_top_k: 10`, `semantic_threshold: 0.65`

### 2. Advanced Orphan Handling

Tests not in the training graph (22.7%) receive intelligent scoring:

```
Orphan Pipeline:
├── KNN Similarity (k=20, euclidean)
├── Structural Blend (35% weight)
├── Temperature Scaling (T=0.7)
└── Alpha Blend (55% KNN + 45% base)
```

**Result:** Orphan variance restored from 0.0 to 0.0462

### 3. Single Balancing Mechanism

Avoids mode collapse by using ONLY balanced sampling:

```yaml
loss:
  use_class_weights: false  # DISABLED
  focal_alpha: 0.5          # NEUTRAL

sampling:
  use_balanced_sampling: true
  minority_weight: 1.0
  majority_weight: 0.07     # ~15:1 ratio
```

### 4. DeepOrder Features

9 additional history-based features:

- `execution_status_last_[1,2,3,5,10]`
- `distance`, `status_changes`
- `cycles_since_last_fail`, `fail_rate_last_10`

---

## Component Contributions

| Component | Impact | Description |
|-----------|--------|-------------|
| Multi-Edge Graph | +6-8% | Denser connectivity, better GAT propagation |
| Orphan KNN | +4-5% | Differentiated orphan scores via KNN + structural blend |
| Single Balancing | +2-3% | Fixed mode collapse |
| DeepOrder Features | +1-2% | Temporal patterns |
| Threshold Optimization | +0.5-1% | Two-phase F-beta search |

---

## Project Structure

```
filo-priori-v10/
├── main.py                          # Main entry point
├── configs/
│   └── experiment_industry_optimized_v3.yaml  # Best config
├── src/
│   ├── models/
│   │   └── dual_stream_v8.py        # Main model
│   ├── evaluation/
│   │   └── orphan_ranker.py         # KNN scoring
│   ├── phylogenetic/
│   │   └── multi_edge_graph_builder.py
│   └── preprocessing/
│       └── structural_feature_extractor_v2_5.py
├── results/
│   └── experiment_industry_optimized_v3/
│       ├── apfd_per_build_FULL_testcsv.csv
│       ├── prioritized_test_cases_FULL_testcsv.csv
│       └── best_model.pt
└── docs/
    ├── TECHNICAL_REPORT_APFD_0.7595.md   # Detailed analysis
    ├── BASELINE_RESULTS.md
    └── PIPELINE_ARCHITECTURE.md          # Mermaid diagrams
```

---

## Training Configuration

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 16 |
| Epochs | 50 (early stop patience=15) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | Cosine Annealing |
| GAT Layers | 1 |
| GAT Heads | 2 |
| Dropout | 0.15 |

### Loss Function

```
Weighted Focal Loss: L = -α(1-p_t)^γ log(p_t)

α = 0.5 (neutral)
γ = 2.0 (focus on hard examples)
```

---

## Orphan Handling Details

### What Are Orphans?

Tests not present in the training graph (~22.7% of test set). Without special handling, they all receive score 0.5.

### Solution: 4-Stage KNN Pipeline

```python
# Stage 1: KNN Similarity
similarities = exp(-euclidean_distance(orphan, in_graph_tests))
top_k = select_top_k(similarities, k=20)

# Stage 2: Structural Blend
combined = 0.65 * semantic_sim + 0.35 * structural_sim

# Stage 3: Temperature-Scaled Softmax
weights = softmax(combined / temperature)  # T=0.7
knn_score = dot(weights, in_graph_scores)

# Stage 4: Alpha Blend
final = 0.55 * knn_score + 0.45 * base_score
```

### Configuration

```yaml
orphan_strategy:
  k_neighbors: 20
  similarity_metric: "euclidean"
  structural_weight: 0.35
  temperature: 0.7
  alpha_blend: 0.55
```

---

## Datasets

### Industrial QTA Dataset

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 |

### RTPTorrent (Optional)

| Statistic | Value |
|-----------|-------|
| Projects | 20 Java |
| APFD | 0.8376 |
| vs recently_failed | +2.02% |

---

## Baseline Comparison

| Method | Mean APFD | vs Filo-Priori |
|--------|-----------|----------------|
| **Filo-Priori V10** | **0.7595** | - |
| Filo-Priori V1 | 0.6503 | -14.4% |
| FailureRate | 0.6289 | -17.2% |
| XGBoost | 0.6171 | -18.7% |
| Random | 0.5596 | -26.3% |

---

## Validation Summary

```
======================================================================
VALIDATION - experiment_industry_optimized_v3
======================================================================

Builds:        277 ✓ (verified against test.csv)
Test Cases:    5,085 ✓
Mean APFD:     0.7595 ✓
Median APFD:   0.7944 ✓
Std APFD:      0.1894
Min APFD:      0.0833
Max APFD:      1.0000

Distribution:
  APFD = 1.0:   23 (8.3%)   - all have 1 test case
  APFD >= 0.7:  188 (67.9%)
  APFD >= 0.5:  247 (89.2%)
  APFD < 0.5:   30 (10.8%)

Data Integrity:
  Invalid values: 0 ✓
  Null values: 0 ✓
  Unique build IDs: 277 ✓

======================================================================
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [TECHNICAL_REPORT_APFD_0.7595.md](docs/TECHNICAL_REPORT_APFD_0.7595.md) | Detailed analysis of how APFD 0.7595 was achieved |
| [BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) | Baseline comparison and metrics |
| [PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) | Visual diagrams (Mermaid) |
| [APFD_SUCCESS_REPORT.md](docs/APFD_SUCCESS_REPORT.md) | End-to-end pipeline explanation |
| [FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md) | Roadmap and implemented fixes |

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| GPU VRAM | 8GB | 12GB+ |
| CUDA | 11.8+ | 12.1+ |

### Software

```
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.2
transformers>=4.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
PyYAML>=6.0
```

---

## Citation

```bibtex
@article{filo_priori_2025,
  title={Filo-Priori: A Dual-Stream Deep Learning Approach to
         Test Case Prioritization},
  author={Ribeiro, Acauan C.},
  journal={IEEE Transactions on Software Engineering},
  year={2025},
  note={Under Review}
}
```

---

## References

- **GAT**: Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.
- **Focal Loss**: Lin, T., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- **SBERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.
- **DeepOrder**: Wang, R., et al. (2020). Deep Learning for Test Case Prioritization.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Changelog

### V10.2 (December 2025) - VALIDATED

- **Mean APFD: 0.7595** (+16.8% from V1)
- Dense multi-edge graph (semantic_top_k=10, threshold=0.65)
- Advanced orphan handling (KNN k=20 + structural blend)
- DeepOrder features (9 additional)
- Two-phase threshold optimization
- All 277 builds validated against source data

### V10.1 (December 2025)

- Fixed mode collapse (single balancing mechanism)
- Config bug fix (`use_class_weights` now respected)

### V10.0 (November 2025)

- RTPTorrent support (APFD 0.8376)
- Learning-to-Rank pipeline

---

| Status | Version | APFD | Updated |
|--------|---------|------|---------|
| **Production Ready** | **V10.2** | **0.7595** | December 2025 |
