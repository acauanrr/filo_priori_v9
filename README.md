# Filo-Priori V9: A Dual-Stream Deep Learning Approach to Test Case Prioritization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-IEEE--TSE--submission-brightgreen.svg)

**A dual-stream deep learning approach that combines semantic understanding of test cases with structural patterns learned from execution history using Graph Attention Networks.**

## The Challenge: Test Case Relationships

Traditional TCP approaches treat test cases as independent entities. We propose combining two complementary information sources:

| Feature Type | Description | Examples |
|--------------|-------------|----------|
| **Semantic** | What tests do | Test descriptions, commit messages |
| **Structural** | How tests behave | Failure rates, co-failure patterns, trends |

The key insight: tests that fail together often indicate related functionality, and Graph Attention Networks can learn to propagate failure signals through these relationships.

---

## Filo-Priori Method: Industrial Dataset (01_industry)

This section provides a comprehensive description of the Filo-Priori approach specifically optimized for the Industrial QTA dataset.

### Problem Statement

**Test Case Prioritization (TCP)** aims to order test cases such that those most likely to fail are executed first. This is critical in Continuous Integration (CI) environments where:

- Test suites are large (2,347 unique test cases)
- Build frequency is high (1,339 builds)
- Time constraints require executing only a subset of tests
- Early fault detection accelerates debugging

**The Industrial Challenge:**

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Extreme Class Imbalance** | 37:1 Pass:Fail ratio | Model tends to predict all Pass |
| **Sparse Failures** | Only 20.7% of builds have failures | Limited positive training examples |
| **Test Interdependencies** | Tests share functionality | Independent treatment loses information |
| **Concept Drift** | Software evolves over time | Historical patterns may become stale |

### The Filo-Priori Approach

Filo-Priori introduces a **dual-stream architecture** that processes two complementary information sources:

```
                    FILO-PRIORI: DUAL-STREAM APPROACH
    ================================================================

    INPUT SOURCES                    PROCESSING                OUTPUT
    -------------                    ----------                ------

    ┌─────────────────┐
    │  SEMANTIC INFO  │
    │  - TC_Summary   │      ┌──────────────┐
    │  - TC_Steps     │ ───> │   SBERT      │ ───> 768-dim ──┐
    │  - Commit Msgs  │      │  Encoder     │                │
    └─────────────────┘      └──────────────┘                │
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                             1536-dim combined ──>  │  SEMANTIC       │
                                                    │  STREAM (FFN)   │
                                                    │  [1536→256→256] │
                                                    └────────┬────────┘
                                                             │
    ┌─────────────────┐                                      │
    │ STRUCTURAL INFO │      ┌──────────────┐                │
    │  - Failure rate │ ───> │  Feature     │                │
    │  - Co-failures  │      │  Extractor   │                ▼
    │  - Test age     │      │  V2.5        │      ┌─────────────────┐
    │  - Trends       │      └──────────────┘      │  CROSS-ATTENTION│
    └─────────────────┘              │             │     FUSION      │
            │                        │             │  [512→256→256]  │
            │                        ▼             └────────┬────────┘
            │               ┌──────────────┐                │
            │               │  TEST GRAPH  │                │
            └──────────────>│  (Multi-Edge)│                │
                            │  - Co-fail   │                │
                            │  - Co-pass   │                ▼
                            │  - Semantic  │      ┌─────────────────┐
                            └──────┬───────┘      │   CLASSIFIER    │
                                   │              │   MLP [256→128  │
                                   ▼              │        →64→2]   │
                            ┌──────────────┐      └────────┬────────┘
                            │  STRUCTURAL  │               │
                            │  STREAM (GAT)│               │
                            │  [19→128→256]│ ──────────────┘
                            └──────────────┘
                                                           │
                                                           ▼
                                                    P(Fail) for each
                                                      test case
```

### Component Details

#### 1. Semantic Stream

The semantic stream captures **what tests do** through natural language understanding:

**Input Processing:**
```
Test Case Text:
├── TC_Summary: "Verify login functionality with valid credentials"
├── TC_Steps: "1. Navigate to login page\n2. Enter username\n3. Enter password..."
└── Commit Messages: "Fix authentication bug in login module"

         │
         ▼ SBERT (all-mpnet-base-v2)
         │
    768-dim embedding (test) + 768-dim embedding (commit)
         │
         ▼ Concatenation
         │
    1536-dim combined semantic embedding
```

**Architecture:**
| Layer | Input | Output | Activation |
|-------|-------|--------|------------|
| Linear + LayerNorm | 1536 | 256 | GELU |
| Dropout | 256 | 256 | - |
| Linear + LayerNorm | 256 | 256 | GELU |
| Residual Connection | 256 | 256 | - |

#### 2. Structural Stream with Graph Attention

The structural stream captures **how tests behave** through historical patterns and test relationships:

**Feature Engineering (19 features):**

| Category | Features | Description |
|----------|----------|-------------|
| **Historical** | `failure_rate`, `recent_failure_rate`, `flakiness_rate` | Failure patterns |
| **Temporal** | `test_age`, `test_novelty`, `consecutive_failures` | Time-based patterns |
| **Trend** | `failure_trend`, `max_consecutive_failures` | Direction of change |
| **Context** | `commit_count`, `cr_count` | Code change association |
| **DeepOrder** | `execution_status_last_[1,2,3,5,10]`, `distance`, `status_changes`, `cycles_since_last_fail`, `fail_rate_last_10` | Execution history |

**Multi-Edge Test Graph:**

The graph captures relationships between test cases:

```
Test Graph Construction:
========================

For each pair of tests (Ti, Tj):

1. CO-FAILURE EDGE (weight: 1.0)
   - Added if Ti and Tj failed together in the same build
   - Edge weight = number of co-failures / total builds
   - Captures: "Tests that fail together are likely related"

2. CO-SUCCESS EDGE (weight: 0.5)
   - Added if Ti and Tj passed together consistently
   - Lower weight because co-success is less informative
   - Captures: "Tests that always pass together may test similar code"

3. SEMANTIC EDGE (weight: 0.3)
   - Added if cosine_similarity(embed(Ti), embed(Tj)) > 0.75
   - Top-K (5) most similar tests connected
   - Captures: "Tests with similar descriptions may be related"

Graph Statistics (01_industry):
├── Nodes: 2,347 (unique test cases)
├── Co-failure edges: ~12,000
├── Co-success edges: ~8,000
├── Semantic edges: ~11,700
└── Average degree: ~13.5 edges per test
```

**Graph Attention Network (GAT):**

```python
# GAT Layer processing
h_i = σ(Σ_j α_ij · W · h_j)

where:
- h_i: representation of test i
- α_ij: attention weight between tests i and j
- W: learnable weight matrix
- σ: ELU activation

Attention computation:
α_ij = softmax(LeakyReLU(a^T · [Wh_i || Wh_j]))
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 1 | Deeper GAT overfits on sparse graph |
| Heads | 2 | Multi-head captures different relationships |
| Hidden dim | 128 | Balance expressiveness vs. overfitting |
| Dropout | 0.15 | Regularization |

#### 3. Cross-Attention Fusion

The fusion module combines semantic and structural representations using bidirectional attention:

```
Semantic (256-dim) ─────┐
                        ▼
              ┌─────────────────┐
              │  Cross-Attention │
              │                  │
              │  Q = Semantic    │
              │  K,V = Structural│
              │         +        │
              │  Q = Structural  │
              │  K,V = Semantic  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Concatenation   │
              │  [256 + 256]     │
              └────────┬────────┘
                       │
                       ▼
              512-dim fused representation
```

This allows the model to:
- Weight semantic features based on structural context
- Weight structural features based on semantic similarity
- Dynamically balance both modalities per test case

#### 4. Classification Head

The final classifier predicts P(Fail) for each test:

```
Fused (512-dim) → Linear(256) → GELU → Dropout(0.2)
                → Linear(128) → GELU → Dropout(0.2)
                → Linear(64)  → GELU → Dropout(0.2)
                → Linear(2)   → Softmax
                                  │
                                  ▼
                           [P(Pass), P(Fail)]
```

### Handling Extreme Class Imbalance (37:1)

The industrial dataset has severe class imbalance, which caused significant challenges:

#### Evolution of Balancing Strategies

| Version | Strategy | Result | Problem |
|---------|----------|--------|---------|
| **V1** | `class_weights` only | APFD: 0.6503 | Recall (Fail): ~3% |
| **V2** | `class_weights` + `balanced_sampling` + high `focal_alpha` | APFD: ~0.55 | Mode collapse to Fail |
| **V3** | `balanced_sampling` only | APFD: **0.6661** | **Balanced predictions** |

#### The Mode Collapse Problem

**V2 Failure Analysis:**

```
V2 Configuration (BROKEN):
├── class_weights: [1.0, 19.0]  → 19x weight to Fail
├── balanced_sampling: minority_weight=1.0, majority_weight=0.05  → 20x oversampling
└── focal_alpha: 0.85  → Additional ~1.7x preference for Fail

Combined effect: 19 × 20 × 1.7 ≈ 323x weight to Fail class!

Result: Model predicts ALL samples as Fail
        Recall (Fail) = 100%, Precision (Fail) = 2.7%
        F1 Macro ≈ 0.50 (degenerate)
```

**V3 Solution: Single Balancing Mechanism**

```yaml
# V3 Configuration (CORRECT)
training:
  loss:
    type: "weighted_focal"
    use_class_weights: false    # DISABLED
    focal_alpha: 0.5            # NEUTRAL
    focal_gamma: 2.0            # Focus on hard examples

  sampling:
    use_balanced_sampling: true
    minority_weight: 1.0
    majority_weight: 0.1        # 10:1 ratio (moderate)
```

**Key Insight:** Use **ONLY ONE** mechanism to handle class imbalance.

#### Weighted Focal Loss

The loss function combines focal modulation with optional class weighting:

```python
Focal Loss: L = -α_t · (1 - p_t)^γ · log(p_t)

where:
- p_t: probability of correct class
- α_t: class weight (0.5 = neutral in V3)
- γ: focusing parameter (2.0)

Effect of γ:
- Well-classified (p_t > 0.9): weight ≈ 0.01 (ignored)
- Uncertain (p_t ≈ 0.5): weight ≈ 0.25 (moderate)
- Misclassified (p_t < 0.1): weight ≈ 0.81 (emphasized)
```

#### Balanced Sampling

Instead of modifying loss weights, V3 uses `WeightedRandomSampler`:

```python
# Sampling probabilities
P(sample Fail) = 1.0 / N_fail    # High probability for minority
P(sample Pass) = 0.1 / N_pass    # Low probability for majority

# Effective batch composition:
# - ~50% Fail samples (vs. 2.7% in original data)
# - ~50% Pass samples
```

### Orphan Handling with KNN

Test cases not present in the training graph ("orphans") require special handling:

```
Orphan Detection:
├── Test case not in training set
├── No graph edges to other tests
└── Score ≈ 0.5 (uncertain)

KNN Imputation:
1. Compute semantic similarity to all in-graph tests
2. Select K=10 nearest neighbors
3. Weight neighbors by similarity
4. Estimate P(Fail) = Σ(similarity_i × P(Fail)_i) / Σ(similarity_i)
```

**Configuration:**
```yaml
ranking:
  orphan_strategy:
    enabled: true
    method: "knn_pfail"
    k_neighbors: 10
    alpha_blend: 0.5  # 50% KNN + 50% own P(Fail)
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA PREPARATION                                            │
│     ├── Load train.csv, test.csv                                │
│     ├── Split: 80% train / 10% val / 10% test                   │
│     ├── Extract semantic embeddings (cached)                    │
│     ├── Extract structural features (19 features)               │
│     └── Build multi-edge graph                                  │
│                                                                 │
│  2. MODEL INITIALIZATION                                        │
│     ├── SemanticStream: FFN [1536→256]                          │
│     ├── StructuralStream: GAT [19→128→256]                      │
│     ├── CrossAttentionFusion: [512→256]                         │
│     └── Classifier: MLP [256→128→64→2]                          │
│                                                                 │
│  3. TRAINING LOOP (50 epochs)                                   │
│     ├── Balanced sampling (10:1 ratio)                          │
│     ├── Forward pass through dual-stream                        │
│     ├── Weighted Focal Loss (γ=2.0, α=0.5)                      │
│     ├── AdamW optimizer (lr=3e-5, wd=1e-4)                      │
│     ├── Cosine annealing scheduler                              │
│     ├── Gradient clipping (max_norm=1.0)                        │
│     └── Early stopping (patience=15, monitor=val_f1_macro)      │
│                                                                 │
│  4. EVALUATION                                                  │
│     ├── Threshold optimization (search [0.1, 0.9])              │
│     ├── Classification metrics (F1, Precision, Recall)          │
│     ├── Ranking metrics (APFD per build)                        │
│     └── Full test.csv evaluation (277 builds with failures)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Improvements in V3

| Improvement | Before (V1/V2) | After (V3) | Impact |
|-------------|----------------|------------|--------|
| **Class Balancing** | Multiple mechanisms | Single (sampling) | Fixed mode collapse |
| **Recall (Fail)** | 3% (V1) / 100% (V2) | **30.2%** | Balanced detection |
| **APFD** | 0.6503 (V1) | **0.6661** | +2.4% improvement |
| **F1 Macro** | ~0.50 | **0.5875** | +17.5% improvement |
| **Bug Fix** | `use_class_weights` ignored | Config respected | Correct behavior |

### Reproducibility

To reproduce the V3 baseline results:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Run with best configuration
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Expected output:
# ├── APFD (277 builds): ~0.6661
# ├── APFD (test split): ~0.7086
# ├── F1 Macro: ~0.5875
# └── Recall (Fail): ~30.2%
```

---

## Key Results

### Industrial Dataset (Classification Mode)

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6661** | Average Percentage of Faults Detected (277 builds) |
| **APFD Test Split** | **0.7086** | 64 builds from test split |
| **F1 Macro** | **0.5875** | Balanced classification metric |
| **Recall (Fail)** | **30.2%** | Fault detection sensitivity |
| **vs FailureRate** | **+5.9%** | Beats the strongest baseline |
| **vs Random** | **+19.0%** | Statistically significant |
| **Graph Attention** | **+17.0%** | Most critical component (ablation) |

> **Baseline Config**: `configs/experiment_industry_optimized_v3.yaml`
> See [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) for detailed benchmark documentation.

### RTPTorrent Dataset (V10 Full - 20 Projects)

| Metric | Value |
|--------|-------|
| **Mean APFD** | **0.8376** |
| **Projects** | 20 Java projects |
| **Test Builds** | 1,250 |
| **vs recently_failed** | **+2.02%** |
| **vs random** | +69.56% |
| **vs untreated** | +134.32% |

**Top-5 Projects:**

| Project | APFD | vs recently_failed |
|---------|------|-------------------|
| apache/sling | 0.9922 | +2.18% |
| neuland/jade4j | 0.9799 | +0.88% |
| eclipse/jetty.project | 0.9789 | +1.27% |
| facebook/buck | 0.9722 | +1.97% |
| deeplearning4j/dl4j | 0.9277 | +0.46% |

---

## Overview

Filo-Priori V9 supports **two operational modes** for different datasets:

### Classification Mode (Industrial Dataset)

| Component | Description |
|-----------|-------------|
| **Semantic Stream** | FFN processing SBERT embeddings (1536-dim) |
| **Structural Stream** | GAT over multi-edge test relationship graph |
| **Cross-Attention Fusion** | Bidirectional attention combining modalities |
| **Weighted Focal Loss** | Handles 37:1 class imbalance |

### Learning-to-Rank Mode (RTPTorrent)

| Component | Description |
|-----------|-------------|
| **Semantic Stream** | MLP processing SBERT embeddings (768-dim) |
| **Structural Stream** | MLP over historical features (2x weight) |
| **Concatenation Fusion** | Simple feature fusion |
| **ListNet Loss** | Listwise ranking loss |

### Key Contributions

1. **Multi-Edge Test Relationship Graph**: Captures co-failure, co-success, and semantic similarity edges
2. **Dual-Stream Architecture**: Combines SBERT semantic embeddings with GAT-based structural learning
3. **Cross-Attention Fusion**: Bidirectional attention for dynamic modality combination
4. **Weighted Focal Loss**: Addresses severe class imbalance (37:1 Pass:Fail ratio)
5. **Feature Engineering**: 10 discriminative features selected from 29 candidates
6. **Learning-to-Rank Pipeline**: Dedicated L2R mode with ListNet loss for ranking-focused datasets
7. **Multi-Dataset Support**: Unified framework supporting Industrial and RTPTorrent datasets

---

## Architecture

```
                    FILO-PRIORI V9: DUAL-STREAM ARCHITECTURE
    =====================================================================

    INPUTS                        STREAMS                       OUTPUT
    ------                        -------                       ------

    [Semantic Input]             +------------------+
    TC_Summary +                 |  SEMANTIC        |
    TC_Steps +       ---------> |  STREAM          | ----+
    Commit_Msg                   |  FFN [1536→256]  |     |
                                 +------------------+     |
                                                          |
                                                          |  CROSS-ATTENTION
    [Structural Input]           +------------------+     |     FUSION
    10 Features +                |  STRUCTURAL      |     |  +-----------+
    Test Graph   -------------> |  STREAM (GAT)    | ----+->| Bi-dir    |
    (multi-edge)                 |  [10→256]        |     |  | Attention |
                                 +------------------+     |  +-----------+
                                                          |       |
                                                          +-------+
                                                                  |
                                                                  v
                                                          +---------------+
                                                          | CLASSIFIER    |
                                                          | MLP [512→128→ |
                                                          |      64→2]    |
                                                          +---------------+
                                                                  |
                                                                  v
                                                          [Prioritized List]
                                                          T' = {t1, t2, ..., tn}

    LOSS FUNCTION:
    L = Weighted Focal Loss (alpha=0.75, gamma=2.5)
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/filo-priori-v9.git
cd filo-priori-v9

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on Industrial Dataset with BEST configuration (APFD 0.6661)
python main.py --config configs/experiment_industry_optimized_v3.yaml

# Train on Industrial Dataset (default - Classification Mode)
python main.py --config configs/experiment_industry.yaml

# Train on RTPTorrent Dataset (Learning-to-Rank Mode)
python scripts/preprocessing/preprocess_rtptorrent_ranking.py  # Preprocess for L2R
python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml

# Cross-dataset evaluation (Train Industry, Test RTPTorrent)
python main.py --config configs/experiment_cross_dataset.yaml
```

### Results

Results are saved to `results/<experiment_name>/`:

| File | Description |
|------|-------------|
| `apfd_per_build_FULL_testcsv.csv` | Per-build APFD scores |
| `prioritized_test_cases_FULL_testcsv.csv` | Ranked test cases with probabilities |
| `optimal_threshold.txt` | Best decision threshold |
| `best_model.pt` | Model checkpoint |

---

## Project Structure

```
filo-priori-v9/
├── main.py                      # Main entry point - Classification mode (Industry)
├── main_rtptorrent.py           # Main entry point - Learning-to-Rank mode (RTPTorrent)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── configs/                     # Experiment configurations (YAML)
│   ├── experiment_industry.yaml          # Industry dataset config
│   ├── experiment_rtptorrent_l2r.yaml    # RTPTorrent L2R config
│   ├── experiment_07_ranking_optimized.yaml  # Best Industry config
│   └── ...
│
├── src/                         # Source code
│   ├── models/                  # Neural network architectures
│   │   ├── dual_stream_v8.py    # Main model (DualStreamModelV8)
│   │   ├── model_factory.py     # Unified model factory
│   │   └── ablation_model.py    # For ablation studies
│   │
│   ├── preprocessing/           # Data loading and feature extraction
│   │   ├── data_loader.py       # CSV loading and splitting
│   │   ├── structural_feature_extractor_v2_5.py  # 10 selected features
│   │   └── commit_extractor.py  # Commit message parsing
│   │
│   ├── phylogenetic/            # Graph construction
│   │   └── multi_edge_graph_builder.py  # Co-failure + co-success + semantic
│   │
│   ├── embeddings/              # SBERT embeddings with caching
│   │   ├── embedding_manager.py # High-level interface
│   │   ├── sbert_encoder.py     # SBERT encoding
│   │   └── embedding_cache.py   # Cache management
│   │
│   ├── evaluation/              # Metrics calculation
│   │   ├── apfd.py              # APFD (main ranking metric)
│   │   ├── metrics.py           # Classification metrics
│   │   ├── threshold_optimizer.py
│   │   └── rtptorrent_evaluator.py  # RTPTorrent baselines comparison
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py           # Training loops
│   │   ├── losses.py            # Focal Loss, Weighted Focal Loss
│   │   └── ranking_losses.py    # L2R losses (ListNet, ListMLE, LambdaRank)
│   │
│   ├── baselines/               # Baseline implementations
│   │   ├── heuristic_baselines.py  # Random, Recency, FailureRate
│   │   └── ml_baselines.py         # LogReg, RF, XGBoost
│   │
│   └── utils/                   # Utilities
│
├── scripts/                     # Analysis and visualization scripts
│   ├── analysis/                # Experimental analysis
│   ├── preprocessing/           # Dataset preprocessing
│   │   └── preprocess_rtptorrent_ranking.py  # RTPTorrent L2R preprocessing
│   └── publication/             # Paper generation
│
├── paper/                       # Publication materials (LaTeX)
│   ├── main_ieee_tse.tex        # Paper (IEEE TSE format)
│   ├── references_ieee.bib      # Bibliography
│   ├── figures/                 # PDF + PNG figures
│   └── sections/                # Paper sections
│
├── results/                     # Experimental results
│   ├── experiment_hybrid_phylogenetic/   # Best Industry results (APFD 0.6413)
│   ├── experiment_rtptorrent_l2r/        # RTPTorrent L2R results
│   ├── baselines/               # Baseline comparison
│   ├── ablation/                # Ablation study
│   └── temporal_cv/             # Temporal cross-validation
│
├── datasets/                    # Data files (gitignored)
│   ├── 01_industry/             # Industrial QTA dataset
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── README.md
│   ├── 02_rtptorrent/           # RTPTorrent open-source dataset
│   │   ├── raw/MSR2/            # Original downloaded data
│   │   ├── processed_ranking/   # Preprocessed for L2R
│   │   └── README.md
│   └── README.md
│
└── cache/                       # Embeddings cache (gitignored)
```

---

## Data Pipeline

### 1. Data Loading
```
datasets/train.csv, test.csv
    │
    ↓ DataLoader
    │
Split: Train (80%) / Val (10%) / Test (10%)
```

### 2. Semantic Feature Extraction
```
Test Case Text                    Commit Messages
(TC_Summary + TC_Steps)           (commit_msg + diff)
    │                                  │
    ↓ SBERT (all-mpnet-base-v2)       ↓
    │                                  │
768-dim embedding                  768-dim embedding
    │                                  │
    └──────────┬───────────────────────┘
               │
               ↓
        1536-dim combined embedding
```

### 3. Structural Feature Extraction (10 features)
```
Historical Execution Data
    │
    ↓ StructuralFeatureExtractor V2.5
    │
10 Selected Features:
├── test_age              (builds since first appearance)
├── failure_rate          (historical failure %)
├── recent_failure_rate   (last 5 builds)
├── flakiness_rate        (pass/fail oscillation)
├── commit_count          (associated commits)
├── test_novelty          (first appearance flag)
├── consecutive_failures  (current streak)
├── max_consecutive_failures
├── failure_trend         (improving/degrading)
└── cr_count              (change requests)
```

### 4. Graph Construction
```
Historical Execution Data
    │
    ↓ MultiEdgeGraphBuilder
    │
Multi-Edge Test Relationship Graph:
├── Co-Failure edges   (weight: 1.0) - Tests that fail together
├── Co-Success edges   (weight: 0.5) - Tests that pass together
└── Semantic edges     (weight: 0.3) - Semantically similar tests
```

---

## Training Configuration

### Loss Function

```python
Weighted Focal Loss: L = -alpha * (1 - p_t)^gamma * log(p_t)
```

| Parameter | V3 Value | Purpose |
|-----------|----------|---------|
| **alpha** | 0.5 | Neutral (no class preference in focal term) |
| **gamma** | 2.0 | Focusing parameter (down-weights easy examples) |
| **class_weights** | disabled | Single balancing mechanism via sampling |

> **Key Insight**: Use ONLY ONE balancing mechanism. V3 uses `balanced_sampling` (10:1 ratio) instead of class weights to avoid mode collapse.

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 50 | With early stopping |
| Batch Size | 32 | Grouped by Build_ID |
| Learning Rate | 3e-5 | Conservative, proven optimal |
| Weight Decay | 1e-4 | L2 regularization |
| Optimizer | AdamW | Adaptive + weight decay |
| Scheduler | Cosine Annealing | eta_min=1e-6 |
| Early Stopping | patience=15 | Monitor: val_f1_macro |
| Gradient Clipping | 1.0 | Prevent exploding gradients |
| GAT Layers | 1 | Simpler architecture works best |
| GAT Heads | 2 | Multi-head attention |
| Dropout | 0.3-0.4 | Regularization |

---

## Experimental Results Summary

### RQ1: Effectiveness (Baseline Comparison)

| Method | Mean APFD | vs Random |
|--------|-----------|-----------|
| **Filo-Priori V3** | **0.6661** | **+19.0%** |
| Filo-Priori V1 | 0.6503 | +16.2% |
| FailureRate | 0.6289 | +12.4% |
| XGBoost | 0.6171 | +10.3% |
| GreedyHistorical | 0.6138 | +9.7% |
| LogisticRegression | 0.5964 | +6.6% |
| RandomForest | 0.5910 | +5.6% |
| Random | 0.5596 | baseline |
| RecentFailureRate | 0.5454 | -2.5% |
| Recency | 0.5240 | -6.4% |

### RQ2: Component Contributions (Ablation Study)

| Component | Contribution | p-value |
|-----------|-------------|---------|
| **Graph Attention (GAT)** | **+17.0%** | < 0.001*** |
| Structural Stream | +5.3% | < 0.001*** |
| Focal Loss | +4.6% | < 0.001*** |
| Class Weighting | +3.5% | 0.002** |
| Semantic Stream | +1.9% | 0.087 |

### RQ3: Temporal Robustness

| Validation Method | Mean APFD | 95% CI |
|-------------------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

### RQ4: Key Findings

- **Graph Attention**: Most critical component (+17.0% from ablation)
- **Weighted Focal Loss**: Essential for handling class imbalance
- **Feature Selection**: 10 features sufficient (V2.5 extractor)
- **Simple Architecture**: 1-layer GAT outperforms deeper models
- **Learning Rate**: 3e-5 proven optimal (very sensitive)

---

## Datasets

Filo-Priori V9 supports multiple datasets for comprehensive evaluation:

### Dataset 1: Industrial QTA (Qodo Test Automation)

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 (highly imbalanced) |
| **Semantic Info** | Rich (test descriptions, commit messages) |

### Dataset 2: RTPTorrent (Open-Source)

| Statistic | Value |
|-----------|-------|
| Projects | 20 Java projects |
| Total Executions | 23.1M |
| Unique Builds | ~110K |
| Failure Rate | 0.20% (very sparse) |
| Source | Travis CI build logs |
| Reference | Mattis et al., MSR 2020 |
| DOI | https://doi.org/10.1145/3379597.3387458 |
| License | CC BY 4.0 |
| **Semantic Info** | Limited (test class names only) |

**Data Fields (Unified Format):**
- `Build_ID`: Build identifier
- `TC_Key`: Test case identifier
- `TE_Summary`: Test execution summary
- `TC_Steps`: Test case steps (optional)
- `TE_Test_Result`: Pass/Fail verdict
- `commit`: Associated commit messages

---

## Learning-to-Rank Mode (RTPTorrent)

Due to the different characteristics of RTPTorrent (sparse failures, limited semantic info), we provide a dedicated **Learning-to-Rank pipeline** optimized for this dataset.

### Key Differences from Classification Mode

| Aspect | Classification (Industry) | Ranking (RTPTorrent) |
|--------|---------------------------|----------------------|
| **Objective** | Predict Pass/Fail | Rank tests by failure likelihood |
| **Loss Function** | Weighted Focal Loss | ListNet (listwise cross-entropy) |
| **Evaluation** | F1, Accuracy, APFD | APFD only (per-build) |
| **Semantic Features** | Rich (descriptions, commits) | Limited (class names) |
| **Structural Features** | 10 features | 9 historical features |
| **Model** | DualStreamV8 + GAT | Two-stream MLP (semantic + structural) |

### L2R Architecture

```
                    FILO-PRIORI L2R: TWO-STREAM RANKING MODEL
    =====================================================================

    INPUTS                        STREAMS                       OUTPUT
    ------                        -------                       ------

    [Semantic Input]             +------------------+
    Test Class Name   ---------> |  SEMANTIC MLP    | ----+
    (SBERT 768-dim)              |  [768→128→64]    |     |
                                 +------------------+     |
                                                          |     +------------+
                                                          +---->| FUSION     |
    [Structural Input]           +------------------+     |     | Concat     |
    9 Historical     ----------> |  STRUCTURAL MLP  | ----+     | [192→128]  |
    Features                     |  [9→64→64]       |           +------------+
    (weight: 2x)                 +------------------+                  |
                                                                       v
                                                               +---------------+
                                                               | SCORER        |
                                                               | MLP [128→64→1]|
                                                               +---------------+
                                                                       |
                                                                       v
                                                               [Ranking Score]
                                                               (per test)

    LOSS: ListNet = -sum(P_true * log(P_pred))
    where P = softmax(scores) over tests in each build
```

### Historical Features (L2R)

```
Per-test features extracted from execution history:
├── total_executions      (build count for this test)
├── total_failures        (cumulative failure count)
├── failure_rate          (total_failures / total_executions)
├── recent_failures       (failures in last 5 builds)
├── recent_executions     (executions in last 5 builds)
├── avg_duration          (average test duration)
├── last_failure_recency  (builds since last failure)
├── is_new_test           (first appearance flag)
└── duration              (current execution duration)
```

### Available L2R Loss Functions

| Loss | Type | Description |
|------|------|-------------|
| **ListNet** | Listwise | Cross-entropy on top-1 probabilities (default) |
| **ListMLE** | Listwise | Maximum likelihood for permutation |
| **LambdaRank** | Pairwise | NDCG-aware gradient weighting |
| **ApproxNDCG** | Listwise | Differentiable NDCG approximation |
| **MSE** | Pointwise | Direct regression to relevance |

### RTPTorrent Baselines (7 strategies)

The RTPTorrent dataset comes with 7 pre-computed baseline orderings:

| Baseline | Description |
|----------|-------------|
| `untreated` | Original test order |
| `random` | Random permutation |
| `recently-failed` | Tests that failed recently first |
| `optimal-failure` | Oracle: perfect failure ordering |
| `optimal-failure-duration` | Oracle: failure + short duration |
| `matrix-naive` | Co-failure matrix approach |
| `matrix-conditional-prob` | Conditional probability matrix |

### V10 Full Results (All 20 Projects)

| Baseline | APFD | Model Improvement |
|----------|------|-------------------|
| **Filo-Priori V10** | **0.8376** | -- |
| recently_failed | 0.8209 | +2.02% |
| random | 0.4940 | +69.56% |
| untreated | 0.3574 | +134.32% |
| matrix_naive | 0.5693 | +47.11% |
| matrix_conditional | 0.5132 | +63.21% |
| optimal_duration | 0.5934 | +41.15% |
| optimal_failure (oracle) | 0.9249 | -9.45% |

The model outperforms **6 out of 7 baselines**, with only the oracle (optimal_failure) achieving higher APFD.

---

## Requirements

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 16GB | 32GB |
| GPU VRAM | 8GB | 12GB+ |
| GPU Model | Any NVIDIA | RTX 3090+ |
| CUDA | 11.8+ | 12.1+ |
| Storage | 10GB | 20GB |
| Training Time | ~2-3 hours | ~1 hour |

### Software Dependencies

**Core ML:**
- torch>=2.0.0
- torch-geometric>=2.3.0
- sentence-transformers>=2.2.2
- transformers>=4.30.0

**Data Processing:**
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- scipy>=1.10.0

**Utilities:**
- PyYAML>=6.0
- tqdm>=4.65.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

---

## Paper Compilation

The `paper/` directory contains all materials ready for submission:

```bash
cd paper/
pdflatex main_ieee_tse.tex
bibtex main_ieee_tse
pdflatex main_ieee_tse.tex
pdflatex main_ieee_tse.tex
```

**Target Journal:** IEEE Transactions on Software Engineering (IEEE TSE)

---

## Scripts

### Run Training
```bash
# Best configuration
python main.py --config configs/experiment_07_ranking_optimized.yaml

# Or use convenience script
./run_experiment.sh
```

### Compare Experiments
```bash
./scripts/compare_experiments_quick.sh
```

### Generate Publication Figures
```bash
python scripts/publication/generate_paper_figures.py
```

### Precompute Embeddings
```bash
python scripts/precompute_embeddings_sbert.py
```

### Prepare RTPTorrent Dataset (Learning-to-Rank)
```bash
# Download dataset (4.1GB) - place in datasets/02_rtptorrent/raw/MSR2/

# Preprocess for Learning-to-Rank
python scripts/preprocessing/preprocess_rtptorrent_ranking.py

# Run L2R experiment
python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml

# Results include per-project reports with baseline comparisons
ls results/experiment_rtptorrent_l2r/
# report_*.txt - Human-readable comparisons
# results_*.json - Detailed metrics and statistics
```

---

## Model Architecture Details

### DualStreamModelV8

The main model (`src/models/dual_stream_v8.py`) consists of:

```python
class DualStreamModelV8(nn.Module):
    """
    Dual-stream architecture for test case prioritization.

    Streams:
    - SemanticStream: FFN processing SBERT embeddings
    - StructuralStreamV8: GAT over test relationship graph

    Fusion:
    - CrossAttentionFusion: Bidirectional attention

    Output:
    - SimpleClassifier: MLP for binary classification
    """
```

| Component | Input Dim | Output Dim | Details |
|-----------|-----------|------------|---------|
| SemanticStream | 1536 | 256 | 2 FFN blocks with residual |
| StructuralStreamV8 | 10 | 256 | GAT (4 heads → 1 head) |
| CrossAttentionFusion | 256 + 256 | 512 | 4-head bidirectional |
| SimpleClassifier | 512 | 2 | MLP [128, 64] |

---

## Citation

```bibtex
@article{filo_priori_v9_2025,
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
- **GATv2**: Brody, S., et al. (2022). How Attentive are Graph Attention Networks? ICLR.
- **Focal Loss**: Lin, T., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- **SBERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

| Status | Version | Last Updated |
|--------|---------|--------------|
| Publication Ready | V10.1 | December 2025 |

### Changelog

**V10.1 (December 2025)**
- **New Industrial Dataset Baseline**: APFD **0.6661** (+3.8% improvement)
- Fixed critical bug in `losses.py`: `use_class_weights` config was being ignored
- **Single Balancing Mechanism**: Solved mode collapse by using only `balanced_sampling`
- Improved Recall (Fail) from 3% to **30.2%**
- New baseline config: `configs/experiment_industry_optimized_v3.yaml`
- Added baseline documentation: `docs/BASELINE_RESULTS.md`

**V10.0 (November 2025)**
- **Complete RTPTorrent Evaluation**: Full experiment across all 20 projects
- **APFD 0.8376** on RTPTorrent, outperforming 6/7 baselines
- LightGBM LambdaRank with 16 ranking-optimized features
- Online learning with EMA (alpha=0.8)
- New `run_v10_rtptorrent_full.py` for comprehensive multi-project evaluation

**V9.2 (November 2025)**
- Added Learning-to-Rank mode for RTPTorrent dataset
- Implemented ListNet, ListMLE, LambdaRank, ApproxNDCG losses
- Added RTPTorrent evaluator with 7 baseline comparisons
- New `main_rtptorrent.py` for L2R experiments

**V9.1 (November 2025)**
- Initial publication-ready version
- Dual-stream architecture with GAT
- Industrial dataset support (APFD 0.6413)
