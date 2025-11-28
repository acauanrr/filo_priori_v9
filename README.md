# Filo-Priori V9: A Phylogenetic Approach to Test Case Prioritization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-IEEE--TSE--submission-brightgreen.svg)

**A bio-inspired deep learning approach that treats software evolution as a phylogenetic tree, using the Git DAG to model evolutionary relationships between commits for intelligent test case prioritization.**

## Paradigm Shift: From Linear History to Phylogenetic Trees

Traditional TCP approaches treat software history as a linear time series. We propose a fundamental reconceptualization: **software evolution is a phylogenetic tree**, where commits are taxa and the Git DAG captures evolutionary relationships.

| Biological Concept | Software Equivalent |
|--------------------|---------------------|
| Taxon/Species | Commit/Version |
| DNA Sequence | Source Code / AST |
| Mutation (SNP) | Code Diff |
| Phylogenetic Tree | Git DAG |
| Phylogenetic Signal | Failure Autocorrelation |

---

## Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6413** | Average Percentage of Faults Detected |
| **vs FailureRate** | **+2.0%** | Beats the strongest baseline |
| **vs Random** | **+14.6%** | Statistically significant (p < 0.001) |
| **PhyloEncoder + GATv2** | **Hybrid** | Best of both worlds architecture |

---

## Overview

Filo-Priori V9 introduces a **bio-inspired hybrid architecture** for TCP:

| Component | Description |
|-----------|-------------|
| **Phylo-Encoder LITE** | GGNN (2 layers, 128-dim) over Git DAG with learnable temperature |
| **GATv2 Encoder** | Graph Attention Network over test co-failure graph |
| **Cross-Attention Fusion** | Combines phylo + structural + semantic features |
| **Ranking Loss** | RankNet-style pairwise loss aligned with APFD metric |

### Scientific Contributions

1. **Phylogenetic Metaphor**: First application of computational phylogenetics to TCP, treating Git DAG as evolutionary tree
2. **Phylogenetic Distance Kernel**: Novel distance metric with learnable temperature parameter
3. **Phylo-Encoder (GGNN)**: Gated Graph Neural Network for failure propagation through commit history
4. **Phylogenetic Regularization**: Loss component encouraging evolutionary consistency in predictions
5. **Hybrid Architecture**: Combines proven GATv2 with novel phylogenetic encoding for best results

---

## Architecture

```
                    FILO-PRIORI: HYBRID ARCHITECTURE (BEST)
    =====================================================================

    INPUTS                        ENCODERS                      OUTPUT
    ------                        --------                      ------

    [Git DAG]                +------------------+
    Commits +                |  PHYLO-ENCODER   |
    Branches +  -----------> |  LITE (2 layers) | ----+
    Merges                   |  [768 → 128]     |     |
                             +------------------+     |
                                                      |  (element-wise sum)
    [Test Graph]             +------------------+     |     +---------------+
    Co-Failure +             |  GATv2 STREAM    |     +---> | CROSS-        |
    Co-Success + ----------> |  (2 heads, 256)  | ----+     | ATTENTION     |
    Semantic                 |  Structural [10] |           | FUSION        |
                             +------------------+           +-------+-------+
                                                                    |
    [Semantic]               +------------------+                   |
    TC_Summary +             |  SEMANTIC        |                   |
    TC_Steps +   ----------> |  STREAM          | -----------------+
    Commits                  |  (SBERT → 256)   |                   |
                             +------------------+                   v
                                                            +---------------+
                                                            | CLASSIFIER    |
                                                            | [512→128→64→2]|
                                                            +---------------+
                                                                    |
                                                                    v
                                                            [Prioritized List]
                                                            T' = {t1, t2, ..., tn}

    LOSS FUNCTION:
    L = 0.7 × L_focal + 0.3 × L_rank + 0.05 × L_phylo_reg
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
# Train model with optimal configuration (HYBRID - BEST RESULTS)
python main.py --config configs/experiment_hybrid_phylogenetic.yaml

# Or use the convenience script
./run_experiment_hybrid.sh

# Alternative: ranking-optimized (without PhyloEncoder)
python main.py --config configs/experiment_07_ranking_optimized.yaml
```

### Results

Results are saved to `results/experiment_hybrid_phylogenetic/`:

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
│   ├── experiment_hybrid_phylogenetic.yaml   # BEST config (HYBRID)
│   ├── experiment_07_ranking_optimized.yaml  # GATv2 only
│   ├── experiment_phylogenetic.yaml          # Full phylogenetic
│   └── ...
│
├── src/                         # Source code (51 Python modules)
│   ├── models/                  # Neural network architectures
│   │   ├── phylogenetic_dual_stream.py  # Hybrid model (PhyloEncoder + GATv2)
│   │   ├── phylo_encoder.py     # GGNN-based PhyloEncoder
│   │   ├── dual_stream_v8.py    # GATv2-only model
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
│   │   └── losses.py            # Focal Loss + Ranking Loss
│   │
│   ├── baselines/               # Baseline implementations
│   │   ├── heuristic_baselines.py  # Random, Recency, FailureRate
│   │   └── ml_baselines.py         # LogReg, RF, XGBoost
│   │
│   ├── layers/                  # Custom neural network layers
│   │   └── gatv2.py             # GATv2 implementation
│   │
│   └── utils/                   # Utilities
│
├── scripts/                     # Analysis and visualization scripts
│   ├── analysis/                # Experimental analysis
│   └── publication/             # Paper generation
│
├── paper/                       # Publication materials (LaTeX)
│   ├── main.tex                 # Paper template (EMSE/IST format)
│   ├── references.bib           # Bibliography
│   ├── figures/                 # PDF + PNG figures
│   └── tables/                  # LaTeX tables
│
├── results/                     # Experimental results
│   ├── experiment_hybrid_phylogenetic/   # BEST results (APFD 0.6413)
│   ├── experiment_phylogenetic_v9/       # Full phylogenetic results
│   ├── experiment_07_ranking_optimized/  # GATv2-only results
│   ├── baselines/               # Baseline comparison
│   ├── ablation/                # Ablation study
│   └── temporal_cv/             # Temporal cross-validation
│
├── docs/                        # Documentation
│   └── TECHNICAL_GUIDE.md       # Detailed technical documentation
│
├── datasets/                    # Data files (gitignored)
│   ├── train.csv
│   └── test.csv
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
Multi-Edge Phylogenetic Graph:
├── Co-Failure edges   (weight: 1.0) - Tests that fail together
├── Co-Success edges   (weight: 0.5) - Tests that pass together
└── Semantic edges     (weight: 0.3) - Semantically similar tests
```

---

## Training Configuration

### Loss Function (Combined)

```python
Total Loss = 0.7 × Focal + 0.3 × Ranking + 0.05 × PhyloReg
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Focal Loss** | 0.7 | Handles 37:1 class imbalance (α=0.75, γ=2.5) |
| **Ranking Loss** | 0.3 | RankNet pairwise loss aligned with APFD |
| **Phylo Regularization** | 0.05 | Encourages evolutionary consistency |

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

---

## Experimental Results Summary

### RQ1: Effectiveness (Baseline Comparison)

| Method | Mean APFD | vs Baseline | vs Random |
|--------|-----------|-------------|-----------|
| **Filo-Priori (Hybrid)** | **0.6413** | - | **+14.6%** |
| Filo-Priori (GATv2 only) | 0.6379 | -0.53% | +14.0% |
| Filo-Priori (Full Phylo) | 0.6316 | -1.5% | +12.9% |
| FailureRate | 0.6289 | -2.0% | +12.4% |
| XGBoost | 0.6171 | -3.8% | +10.3% |
| GreedyHistorical | 0.6138 | -4.3% | +9.7% |
| LogisticRegression | 0.5964 | -7.0% | +6.6% |
| RandomForest | 0.5910 | -7.8% | +5.6% |
| Random | 0.5596 | -12.7% | baseline |

### RQ2: Component Contributions (Ablation Study)

| Component | Contribution | Significance |
|-----------|-------------|--------------|
| **Graph Attention (GATv2)** | **+17.0%** | *** (most critical) |
| **PhyloEncoder LITE** | **+0.5%** | * (novel contribution) |
| Structural Stream | +5.3% | *** |
| Class Weighting | +4.6% | *** |
| Semantic Stream | +1.9% | - |

### RQ3: Temporal Robustness

| Validation Method | Mean APFD | 95% CI |
|-------------------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

### RQ4: Key Findings

- **Hybrid Architecture**: PhyloEncoder + GATv2 achieves best results (APFD 0.6413)
- **Ranking Loss**: Critical for APFD optimization (+3.4% improvement)
- **GATv2**: Most important component (+17.0% from ablation)
- **PhyloEncoder LITE**: Adds +0.5% while providing scientific novelty
- **Feature Selection**: 10 features sufficient (V2.5 extractor)
- **Learning Rate**: 3e-5 proven optimal (very sensitive)

---

## Dataset

**QTA Dataset (Qodo Test Automation)**

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 (highly imbalanced) |

**Data Fields:**
- `TE_Summary`: Test execution summary
- `TC_Steps`: Test case steps
- `Commit_Message`: Associated commit messages
- `TE_Test_Result`: Pass/Fail verdict
- `Build_ID`: Build identifier

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
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Target Journal:** IEEE Transactions on Software Engineering (IEEE TSE)

---

## Scripts

### Run Training
```bash
# Best configuration (Hybrid - APFD 0.6413)
python main.py --config configs/experiment_hybrid_phylogenetic.yaml

# Or use convenience script
./run_experiment_hybrid.sh
```

### Compare Experiments
```bash
./scripts/compare_experiments_quick.sh
```

### Generate Publication Figures
```bash
python scripts/generate_publication_visualizations.py
```

### Precompute Embeddings
```bash
python scripts/precompute_embeddings_sbert.py
```

---

## Citation

```bibtex
@software{filo_priori_v9_2025,
  title={Filo-Priori: Deep Learning-based Test Case Prioritization
         with Graph Attention Networks and Ranking-Aware Training},
  author={Ribeiro, Acauan C.},
  year={2025},
  version={9.0},
  institution={IComp/UFAM}
}
```

---

## References

- **GATv2**: Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? ICLR.
- **RankNet**: Burges, C., et al. (2005). Learning to Rank using Gradient Descent. ICML.
- **Focal Loss**: Lin, T., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- **SBERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

| Status | Version | Last Updated |
|--------|---------|--------------|
| Publication Ready | V9.1 (Hybrid) | November 2025 |
