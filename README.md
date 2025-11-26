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

1. **Multi-Edge Phylogenetic Graph**: First application of multi-edge graphs in TCP
2. **Dual-Stream Architecture**: Solves dimensional imbalance between semantic (1536-dim) and structural (10-dim) features
3. **GATv2 Attention**: Dynamic attention mechanism for test relationships
4. **Ranking-Aware Training**: RankNet-style pairwise loss aligned with APFD metric
5. **Comprehensive Evaluation**: Ablation study, temporal cross-validation, sensitivity analysis

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
```

### Results

Results are saved to `results/experiment_07_ranking_optimized/`

---

## Project Structure

```
filo-priori-v9/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── configs/               # Experiment configurations
│   ├── experiment_06_feature_selection.yaml
│   └── experiment_07_ranking_optimized.yaml  # Best config
│
├── src/                   # Source code
│   ├── models/            # Neural network models
│   ├── preprocessing/     # Feature extraction
│   ├── phylogenetic/      # Graph construction
│   ├── evaluation/        # Metrics (APFD, F1, etc.)
│   ├── training/          # Training utilities
│   ├── embeddings/        # SBERT embeddings
│   ├── baselines/         # Baseline methods
│   └── utils/             # Utility functions
│
├── scripts/               # Analysis scripts
│   ├── analysis/          # Experimental analysis
│   └── publication/       # Paper generation
│
├── paper/                 # Publication materials
│   ├── main.tex           # Paper template (EMSE/IST)
│   ├── references.bib     # Bibliography
│   ├── figures/           # PDF + PNG figures
│   ├── tables/            # LaTeX tables
│   └── sections/          # Paper sections
│
├── results/               # Experimental results
│   ├── experiment_07_ranking_optimized/  # Best model
│   ├── baselines/         # Baseline comparison
│   ├── ablation/          # Ablation study
│   └── temporal_cv/       # Temporal validation
│
├── docs/                  # Documentation
├── datasets/              # Data files (gitignored)
└── cache/                 # Embeddings cache (gitignored)
```

---

## Experimental Results Summary

### RQ1: Effectiveness (Baseline Comparison)

| Method | Mean APFD | p-value | Effect Size |
|--------|-----------|---------|-------------|
| **Filo-Priori** | **0.6379** | - | - |
| FailureRate | 0.6289 | 0.314 | negligible |
| XGBoost | 0.6171 | 0.089 | negligible |
| GreedyHistorical | 0.6138 | 0.041 | negligible |
| Random | 0.5596 | <0.001 | small |
| Recency | 0.5240 | <0.001 | medium |

### RQ2: Component Contributions (Ablation Study)

| Component | Contribution | Significance |
|-----------|-------------|--------------|
| **Graph Attention (GATv2)** | **+17.0%** | *** |
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

### RQ4: Hyperparameter Sensitivity

- **Ranking Loss**: Critical for APFD optimization (+3.4% improvement)
- **Loss Function**: Weighted Focal + Ranking Loss achieves best results
- **Best Configuration**: LR 3e-5, 1-layer GNN, 10 features, ranking weight 0.3

---

## Key Innovation: Ranking-Aware Training

The main improvement in V9 comes from **aligning the training objective with the evaluation metric**:

| Aspect | Previous (Exp 06) | Current (Exp 07) |
|--------|-------------------|------------------|
| Loss | Cross-entropy (classification) | Weighted Focal + Ranking Loss |
| Objective | Minimize prediction error | Rank failures before passes |
| APFD | 0.6171 | **0.6379** (+3.4%) |

**Scientific basis**: RankNet (Burges et al., ICML 2005), Focal Loss (Lin et al., ICCV 2017)

---

## Paper Compilation

The `paper/` directory contains all materials ready for submission to EMSE/IST:

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Requirements

### Hardware
- **RAM**: 16GB minimum
- **VRAM**: 8GB+ recommended (CUDA 11.8+)
- **Storage**: ~10GB for embeddings cache

### Software
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- Sentence-Transformers 2.2+

---

## Dataset

QTA Dataset statistics:

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 |

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

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

| Status | Version | Last Updated |
|--------|---------|--------------|
| Publication Ready | V9.0 | November 2025 |

**Target Journals**: EMSE, IST (Qualis A)
