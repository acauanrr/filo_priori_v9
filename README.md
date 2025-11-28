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

## Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6413** | Average Percentage of Faults Detected |
| **vs FailureRate** | **+2.0%** | Beats the strongest baseline |
| **vs Random** | **+14.6%** | Statistically significant (p < 0.001) |
| **Graph Attention** | **+17.0%** | Most critical component (ablation) |

---

## Overview

Filo-Priori V9 uses a **dual-stream architecture** for TCP:

| Component | Description |
|-----------|-------------|
| **Semantic Stream** | FFN processing SBERT embeddings (1536-dim) |
| **Structural Stream** | GAT over multi-edge test relationship graph |
| **Cross-Attention Fusion** | Bidirectional attention combining modalities |
| **Weighted Focal Loss** | Handles 37:1 class imbalance |

### Key Contributions

1. **Multi-Edge Test Relationship Graph**: Captures co-failure, co-success, and semantic similarity edges
2. **Dual-Stream Architecture**: Combines SBERT semantic embeddings with GAT-based structural learning
3. **Cross-Attention Fusion**: Bidirectional attention for dynamic modality combination
4. **Weighted Focal Loss**: Addresses severe class imbalance (37:1 Pass:Fail ratio)
5. **Feature Engineering**: 10 discriminative features selected from 29 candidates

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
# Train on Industrial Dataset (default)
python main.py --config configs/experiment_industry.yaml

# Train on RTPTorrent Dataset (download first)
python scripts/preprocessing/download_rtptorrent.py  # Download 4.1GB
python scripts/preprocessing/preprocess_rtptorrent.py  # Convert format
python main.py --config configs/experiment_rtptorrent.yaml

# Cross-dataset evaluation (Train Industry, Test RTPTorrent)
python main.py --config configs/experiment_cross_dataset.yaml

# Or use best configuration
python main.py --config configs/experiment_07_ranking_optimized.yaml
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
├── main.py                      # Main entry point (training + evaluation)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── configs/                     # Experiment configurations (YAML)
│   ├── experiment_07_ranking_optimized.yaml  # Best config
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
│   │   └── threshold_optimizer.py
│   │
│   ├── training/                # Training pipeline
│   │   ├── trainer.py           # Training loops
│   │   └── losses.py            # Focal Loss, Weighted Focal Loss
│   │
│   ├── baselines/               # Baseline implementations
│   │   ├── heuristic_baselines.py  # Random, Recency, FailureRate
│   │   └── ml_baselines.py         # LogReg, RF, XGBoost
│   │
│   └── utils/                   # Utilities
│
├── scripts/                     # Analysis and visualization scripts
│   ├── analysis/                # Experimental analysis
│   └── publication/             # Paper generation
│
├── paper/                       # Publication materials (LaTeX)
│   ├── main_ieee_tse.tex        # Paper (IEEE TSE format)
│   ├── references_ieee.bib      # Bibliography
│   ├── figures/                 # PDF + PNG figures
│   └── sections/                # Paper sections
│
├── results/                     # Experimental results
│   ├── experiment_hybrid_phylogenetic/   # Best results (APFD 0.6413)
│   ├── experiment_07_ranking_optimized/  # GATv2 results
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
│   │   ├── raw/                 # Original downloaded data
│   │   ├── processed/           # Converted to Filo-Priori format
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
Weighted Focal Loss: L = -alpha * w_t * (1 - p_t)^gamma * log(p_t)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **alpha** | 0.75 | Class balancing weight |
| **gamma** | 2.5 | Focusing parameter (down-weights easy examples) |
| **w_t** | inverse freq | Additional class weighting |

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
| **Filo-Priori** | **0.6413** | **+14.6%** |
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
| Source | Travis CI build logs |
| Reference | Mattis et al., MSR 2020 |
| DOI | https://doi.org/10.1145/3379597.3387458 |
| License | CC BY 4.0 |
| **Semantic Info** | Limited (test names only) |

**Data Fields (Unified Format):**
- `Build_ID`: Build identifier
- `TC_Key`: Test case identifier
- `TE_Summary`: Test execution summary
- `TC_Steps`: Test case steps (optional)
- `TE_Test_Result`: Pass/Fail verdict
- `commit`: Associated commit messages

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

### Prepare RTPTorrent Dataset
```bash
# Download from Zenodo (4.1GB)
python scripts/preprocessing/download_rtptorrent.py

# Convert to Filo-Priori format
python scripts/preprocessing/preprocess_rtptorrent.py

# Optional: Process specific project only
python scripts/preprocessing/preprocess_rtptorrent.py --project commons-math
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
| Publication Ready | V9.1 | November 2025 |
