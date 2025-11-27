# Filo-Priori V9: Deep Learning-Based Test Case Prioritization with Graph Attention Networks

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-publication--ready-brightgreen.svg)

**A deep learning approach for intelligent test case prioritization in CI/CD pipelines, combining semantic embeddings, structural features, and Graph Attention Networks (GATv2) with ranking-aware training.**

---

## Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6379** | Average Percentage of Faults Detected |
| **vs FailureRate** | **+1.4%** | Beats the strongest baseline |
| **vs Random** | **+14.0%** | Statistically significant (p < 0.001) |
| **GATv2 Contribution** | **+17.0%** | Most critical component (ablation study) |

---

## Overview

Filo-Priori V9 combines **semantic understanding** of test cases with **phylogenetic (historical execution) patterns** through a **Dual-Stream Neural Network** and **Multi-Edge Phylogenetic Graph** architecture with **GATv2 attention** and **ranking-aware training**.

### Scientific Contributions

1. **Multi-Edge Phylogenetic Graph**: First application of multi-edge graphs in TCP combining co-failure, co-success, and semantic edges
2. **Dual-Stream Architecture**: Solves dimensional imbalance between semantic (1536-dim) and structural (10-dim) features
3. **GATv2 Attention**: Dynamic attention mechanism for test relationships (Brody et al., 2022)
4. **Ranking-Aware Training**: RankNet-style pairwise loss aligned with APFD metric (Burges et al., 2005)
5. **Comprehensive Evaluation**: Ablation study, temporal cross-validation, sensitivity analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   FILO-PRIORI V9 ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT DATA                                                      │
│  ├── Test Case Text (TC_Summary + TC_Steps)                     │
│  ├── Commit Messages + Diffs                                     │
│  └── Historical Execution Data                                   │
│                                                                  │
│       ↓                    ↓                    ↓                │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │   SBERT      │   │   SBERT      │   │   Structural     │     │
│  │  Encoder     │   │  Encoder     │   │   Extractor      │     │
│  │  (768-dim)   │   │  (768-dim)   │   │   (10 features)  │     │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘     │
│         │                  │                    │                │
│         └────────┬─────────┘                    │                │
│                  │                              │                │
│                  ↓                              ↓                │
│  ┌───────────────────────┐        ┌───────────────────────────┐ │
│  │   SEMANTIC STREAM     │        │   STRUCTURAL STREAM       │ │
│  │                       │        │                           │ │
│  │   Combined: 1536-dim  │        │   Features + Graph        │ │
│  │   MLP: 256-dim out    │        │   GATv2: 128-dim out      │ │
│  │   2 layers + GELU     │        │   2 heads attention       │ │
│  └───────────┬───────────┘        └─────────────┬─────────────┘ │
│              │                                  │                │
│              └────────────┬─────────────────────┘                │
│                           │                                      │
│                           ↓                                      │
│              ┌───────────────────────┐                          │
│              │     FUSION LAYER      │                          │
│              │   [256 + 64] = 320    │                          │
│              │   → 256-dim hidden    │                          │
│              │   2 layers + GELU     │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│                          ↓                                       │
│              ┌───────────────────────┐                          │
│              │     CLASSIFIER        │                          │
│              │   256 → 128 → 2       │                          │
│              │   Pass/Fail logits    │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│                          ↓                                       │
│              ┌───────────────────────┐                          │
│              │   TEST CASE RANKING   │                          │
│              │   by Fail probability │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
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
# Train model with optimal configuration (ranking-optimized)
python main.py --config configs/experiment_07_ranking_optimized.yaml

# Force regenerate embeddings if needed
python main.py --config configs/experiment_07_ranking_optimized.yaml --force-regen-embeddings
```

### Results

Results are saved to `results/experiment_07_ranking_optimized/`:

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
│   ├── experiment_07_ranking_optimized.yaml  # Best config (recommended)
│   ├── experiment_06_feature_selection.yaml
│   └── ...
│
├── src/                         # Source code (51 Python modules)
│   ├── models/                  # Neural network architectures
│   │   ├── dual_stream_v8.py    # Main model (Dual-Stream + GATv2)
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
│   ├── experiment_07_ranking_optimized/  # Best results
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
Total Loss = 0.7 × Weighted Focal Loss + 0.3 × Ranking Loss
```

**Weighted Focal Loss** (Lin et al., 2017):
- Handles 37:1 class imbalance
- α = [0.15, 0.85] (per-class weights)
- γ = 2.5 (focusing parameter)

**Ranking Loss** (RankNet, Burges et al., 2005):
- Creates (Fail, Pass) pairs within same Build_ID
- Hard Negative Mining: Top-5 hardest Pass cases
- Margin: 0.5

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

| Method | Mean APFD | p-value | vs Random |
|--------|-----------|---------|-----------|
| **Filo-Priori** | **0.6379** | - | **+14.0%** |
| FailureRate | 0.6289 | 0.363 | +12.4% |
| XGBoost | 0.6171 | 0.577 | +10.3% |
| GreedyHistorical | 0.6138 | 0.096 | +9.7% |
| LogisticRegression | 0.5964 | 0.185 | +6.6% |
| RandomForest | 0.5910 | 0.094 | +5.6% |
| Random | 0.5596 | <0.001 | baseline |

### RQ2: Component Contributions (Ablation Study)

| Component | Contribution | Significance |
|-----------|-------------|--------------|
| **Graph Attention (GATv2)** | **+17.0%** | *** (most critical) |
| Structural Stream | +5.3% | *** |
| Class Weighting | +4.6% | *** |
| Ensemble | +3.5% | *** |
| Semantic Stream | +1.9% | - |

### RQ3: Temporal Robustness

| Validation Method | Mean APFD | 95% CI |
|-------------------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

### RQ4: Key Findings

- **Ranking Loss**: Critical for APFD optimization (+3.4% improvement)
- **GATv2**: Most important component (+17.0% from ablation)
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

**Target Journals:** EMSE, IST (Qualis A)

---

## Scripts

### Run Training
```bash
python main.py --config configs/experiment_07_ranking_optimized.yaml
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
| Publication Ready | V9.0 | November 2025 |
