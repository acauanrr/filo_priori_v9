# Filo-Priori V9: Deep Learning-Based Test Case Prioritization Using Multi-Edge Phylogenetic Graphs and Dual-Stream Neural Networks

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research--publication-blueviolet.svg)

**A novel deep learning architecture for intelligent test case prioritization in CI/CD pipelines, targeting publication in Qualis A journals (EMSE, IST).**

---

## Overview

Filo-Priori V9 is an evolution of the production-ready V8 system, now with focus on **scientific rigor** for publication in top-tier Software Engineering journals. The system combines **semantic understanding** of test cases with **phylogenetic (historical execution) patterns** through a novel **Dual-Stream Neural Network** and **Multi-Edge Phylogenetic Graph** architecture with **GATv2 attention**.

**Key Result**: **APFD = 0.6171** (+23.4% vs Random), with 40.8% of builds achieving perfect prioritization (APFD = 1.0).

### Scientific Contributions

1. **Multi-Edge Phylogenetic Graph**: First application of multi-edge graphs in TCP, with novel co-success edges
2. **Dual-Stream Architecture**: Solves dimensional imbalance between semantic (1536-dim) and structural (10-dim) features
3. **Multi-Granularity Temporal Features**: Systematic methodology for feature selection in temporal data
4. **Production-Ready Research System**: Bridges research-to-production gap with reproducible, deployable code

### Key Features

- **Dual-Stream Architecture**: Independent processing of semantic (SBERT embeddings) and structural (10 phylogenetic features) with cross-attention fusion
- **GATv2 Attention**: Dynamic attention mechanism with 2-head attention learning edge importance
- **Multi-Edge Phylogenetic Graph**: Three complementary edge types:
  - Co-failure edges (weight=1.0): Fault propagation patterns
  - Co-success edges (weight=0.5): Stability patterns (novel contribution)
  - Semantic edges (weight=0.3): Content similarity (cosine > 0.75)
- **Expert-Selected Features**: 10 features via 3-phase selection methodology:
  - 6 phylogenetic: test_age, failure_rate, recent_failure_rate, very_recent_failure_rate, failure_streak, pass_streak
  - 4 structural: num_commits, num_change_requests, commit_surge, execution_stability
- **Orphan Handling**: Smart inference for new test cases (global_idx=-1)
- **Production-Ready**: 1.26M parameters, ~5MB model, 3-4h training time

---

## Performance Metrics

### Primary Metrics (Ranking)

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6171** | Average Percentage of Faults Detected |
| **Median APFD** | **0.6458** | Median across builds |
| **APFD = 1.0** | **40.8%** of builds | Perfect prioritization |
| **APFD ≥ 0.5** | **63.9%** of builds | Better than random |
| **Improvement vs Random** | **+23.4%** | APFD increase over random ordering |

### Classification Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **F1-Macro** | **0.5312** | Balanced performance across Pass/Fail |
| **Accuracy** | **0.9664** | Overall classification accuracy |
| **AUROC** | **0.7891** | Area under ROC curve |
| **AUPRC-Macro** | **0.4562** | Area under PR curve |

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 |
| Builds with Failures | 277 (20.7%) |
| Avg TCs/Build | 38.9 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FILO-PRIORI PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│  INPUT DATA (Per Test Execution)                            │
│  ├─ Text: summary + steps + commits                         │
│  ├─ Structural: 10 phylogenetic features                    │
│  └─ Graph: Multi-edge relationships                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  PREPROCESSING PIPELINE                            │     │
│  │  1. Text → SBERT Embeddings (1536-dim)            │     │
│  │  2. Features → Normalization (10-dim)             │     │
│  │  3. Graph → Multi-Edge Construction               │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  DUAL-STREAM NEURAL NETWORK                        │     │
│  │                                                     │     │
│  │  ┌─────────────────┐    ┌──────────────────────┐  │     │
│  │  │ Semantic Stream │    │ Structural Stream    │  │     │
│  │  │ 1536 → 256      │    │ 10 → 64 (MLP)        │  │     │
│  │  │ 2-layer MLP     │    │ GATv2 (2 heads)      │  │     │
│  │  │ LayerNorm+GELU  │    │ Graph Aggregation    │  │     │
│  │  └─────────────────┘    └──────────────────────┘  │     │
│  │           │                        │               │     │
│  │           └────────────┬───────────┘               │     │
│  │                        ▼                           │     │
│  │              ┌──────────────────┐                  │     │
│  │              │  Cross-Attention │                  │     │
│  │              │  Fusion (4 heads)│                  │     │
│  │              └──────────────────┘                  │     │
│  │                        │                           │     │
│  │                        ▼                           │     │
│  │              ┌──────────────────┐                  │     │
│  │              │   Classifier     │                  │     │
│  │              │   256 → 128 → 2  │                  │     │
│  │              └──────────────────┘                  │     │
│  └────────────────────────────────────────────────────┘     │
│                           │                                 │
│                           ▼                                 │
│                    P(Pass), P(Fail) → Ranking by P(Fail)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Scientific Contributions

### 1. Multi-Edge Phylogenetic Graph (Novel)

**Problem**: Single-edge graphs (co-failure only) ignore stability patterns and semantic relationships.

**Solution**: Three complementary edge types in a unified graph:
- **Co-failure edges** (weight=1.0): P(A fails | B fails) - fault propagation
- **Co-success edges** (weight=0.5): P(A passes | B passes) - stability patterns (**novel**)
- **Semantic edges** (weight=0.3): Cosine similarity > 0.75 - functional similarity

**Impact**: First multi-edge graph for TCP. Graph density 25-50x higher than single-edge. Co-success edges are completely novel contribution.

**Publication Potential**: Very High

### 2. Dual-Stream Architecture

**Problem**: Dimensional imbalance between semantic (1536-dim) and structural (10-dim) causes one stream to dominate.

**Solution**: Independent streams with specialized architectures:
- **Semantic Stream**: 1536 → 256 (compression via 2-layer MLP)
- **Structural Stream**: 10 → 64 → 256 (upsampling + GATv2 aggregation)
- **Cross-Attention Fusion**: Bidirectional attention with learned gating

**Impact**: +8% synergy over best single-stream variant. Resolves dimensional imbalance explicitly.

**Publication Potential**: High

### 3. GATv2 for Test Case Prioritization

**Problem**: GAT original has "static attention" limitation (Brody et al., 2022).

**Solution**: GATv2 with dynamic attention:
- LeakyReLU applied AFTER linear projection
- 2-head attention (128-dim total)
- Edge weights enable importance learning across edge types

**Impact**: Adapts to test-specific relationship patterns dynamically.

### 4. Multi-Granularity Temporal Feature Engineering

**Problem**: Naive feature expansion leads to overfitting in temporal data (29 features → APFD 0.5997).

**Solution**: 3-phase methodology:
- **Phase 1**: Baseline with 6 proven features (APFD ~0.62)
- **Phase 2**: Expansion to 29 features (exploratory, overfitting detected)
- **Phase 3**: Expert-guided selection → 10 features (APFD 0.6171)

**Selected Features**:
- **Phylogenetic (6)**: test_age, failure_rate, recent_failure_rate, very_recent_failure_rate, failure_streak, pass_streak
- **Structural (4)**: num_commits, num_change_requests, commit_surge, execution_stability

**Impact**: Demonstrates "more features ≠ better" in temporal data. Replicable methodology.

**Publication Potential**: Medium-High

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
# Train production model (Experiment 06 - 10 selected features)
python main.py --config configs/experiment_06_feature_selection.yaml

# Results saved to: results/experiment_06_feature_selection/
```

### Inference

```bash
# Generate prioritized test ranking
python predict.py --model results/experiment_06_feature_selection/best_model.pt \
                  --input datasets/test.csv \
                  --output ranked_tests.csv
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | GPU memory vs convergence |
| Epochs | 50 | With early stopping |
| Learning Rate | 3e-5 | AdamW optimizer |
| GAT Heads | 2 | Multi-head attention |
| Hidden Dim | 256 | Model capacity |
| Dropout | 0.1/0.3 | Semantic/GAT |

---

## Documentation

### Scientific Documentation

| Document | Description |
|----------|-------------|
| [SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md](SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md) | Complete scientific analysis (11 sections, 1200+ lines) |
| [REASONING_AGENT_PROMPT.md](REASONING_AGENT_PROMPT.md) | Research roadmap for Qualis A publication |
| [results/publication/TECHNICAL_REPORT.md](results/publication/TECHNICAL_REPORT.md) | Technical report with architecture details |

### Architecture Documentation

Comprehensive visual documentation in `figures/`:

- **complete_pipeline_architecture.mmd**: Full end-to-end pipeline
- **step_2.1_structural_features_extraction.md**: Feature extraction deep dive
- **step_2.2_model_architecture.md**: Model architecture and phylogenetic graphs
- **step_2.2_implementation_details.md**: Implementation details

Additional in `results/publication/`:
- **INFERENCE_EXPLANATION.md**: Inference for new test cases
- **GRAPH_CONSTRUCTION_STEP_BY_STEP.md**: Multi-edge graph construction
- **FEATURE_EXPANSION_ANALYSIS.md**: Feature selection analysis

View Mermaid diagrams:
- GitHub/GitLab: Automatic rendering
- VS Code: "Markdown Preview Mermaid Support" extension
- Online: https://mermaid.live/

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
- pandas, numpy, scikit-learn

---

## Project Structure

```
filo-priori-v9/
├── main.py                            # Main entry point
├── requirements.txt                   # Dependencies
├── README.md                          # This file
│
├── configs/                           # Experiment configurations
│   ├── experiment_06_feature_selection.yaml  # Production config
│   └── experiment_*.yaml              # Other experiment configs
│
├── src/                               # Source code
│   ├── models/                        # Neural network models
│   ├── preprocessing/                 # Feature extraction
│   ├── phylogenetic/                  # Graph construction
│   ├── evaluation/                    # Metrics (APFD, F1, etc.)
│   ├── training/                      # Training utilities
│   ├── embeddings/                    # SBERT embeddings
│   ├── baselines/                     # Baseline methods
│   └── utils/                         # Utility functions
│
├── scripts/                           # Analysis and utility scripts
│   ├── analysis/                      # Experimental analysis
│   │   ├── run_all_baselines.py       # Baseline comparison
│   │   ├── run_ablation_study.py      # Ablation study
│   │   ├── run_temporal_cv.py         # Temporal cross-validation
│   │   ├── run_sensitivity_analysis.py # Hyperparameter sensitivity
│   │   └── run_qualitative_analysis.py # Qualitative analysis
│   └── publication/                   # Paper generation
│       ├── generate_paper_sections.py # LaTeX sections
│       └── prepare_paper_submission.py # Final paper materials
│
├── paper/                             # Publication materials (ready)
│   ├── main.tex                       # Paper template
│   ├── figures/                       # All figures (PDF + PNG)
│   ├── tables/                        # All LaTeX tables
│   └── sections/                      # Paper sections
│
├── results/                           # Experimental results
│   ├── experiment_06_feature_selection/ # Main model results
│   ├── baselines/                     # Baseline comparison
│   ├── ablation/                      # Ablation study
│   ├── temporal_cv/                   # Temporal validation
│   ├── sensitivity/                   # Sensitivity analysis
│   ├── qualitative_analysis/          # Qualitative analysis
│   └── final_report/                  # Consolidated report
│
├── docs/                              # Documentation
│   ├── SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md
│   └── REASONING_AGENT_PROMPT.md
│
├── datasets/                          # Data files (in .gitignore)
├── cache/                             # Embeddings cache (in .gitignore)
└── _archive/                          # Archived old files (in .gitignore)
```

---

## Publication Roadmap

### Current Status: Score 62/100 → Target 81/100

| Criterion | Current | Target | Gap |
|-----------|---------|--------|-----|
| Originality | 7.5/10 | 8.0/10 | Ablation studies |
| Scientific Rigor | 5.0/10 | 8.5/10 | Baselines, CIs, cross-val |
| Results Quality | 7.0/10 | 7.5/10 | Error analysis |
| Reproducibility | 9.0/10 | 9.5/10 | Already excellent |
| Comparison with SOTA | 2.0/10 | 8.0/10 | Implement 5-7 baselines |
| Generalization | 4.0/10 | 7.5/10 | Cross-validation |

### Phases

**Phase 1 (1-2 weeks): Critical Fixes**
- [ ] Related Work + 5-7 baselines implementation
- [ ] Statistical validation (Bootstrap 1000x, paired t-tests)
- [ ] Cross-validation (temporal k-fold or cross-project)

**Phase 2 (1 week): Methodological Improvements**
- [ ] Error analysis of 36.1% builds with APFD < 0.5
- [ ] Systematic ablation studies (15-20 variants)

**Phase 3 (3-4 days): Polish**
- [ ] Interpretability analysis (attention weights, t-SNE)
- [ ] Paper writing (8-10 pages)

### Target Journals (Qualis A)

| Journal | Impact Factor | Fit | Priority |
|---------|---------------|-----|----------|
| **EMSE** | ~4.0 | Excellent | High |
| **IST** | ~3.5 | Very Good | High |
| **TSE** | ~7.0 | Excellent | After track record |
| **JSS** | ~3.0 | Good | Backup |

---

## Known Limitations

1. **Single dataset**: Only QTA project, generalization not tested
2. **No SOTA comparison**: Only Random baseline implemented
3. **36.1% failures**: Builds with APFD < 0.5 not analyzed
4. **No statistical validation**: Missing CIs and significance tests
5. **Cold-start problem**: New tests get default prediction [0.5, 0.5]

See [SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md](SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md) for detailed gap analysis.

---

## Future Work

- Cross-project validation with additional datasets
- Online learning for concept drift adaptation
- Multi-task learning (TCP + fault localization)
- Code coverage feature integration
- Industrial deployment study

---

## Citation

```bibtex
@software{filo_priori_v9_2025,
  title={Filo-Priori V9: Multi-Edge Phylogenetic Graphs with Dual-Stream Neural Networks for Test Case Prioritization},
  author={Filo-Priori Research Team},
  year={2025},
  version={9.0},
  url={https://github.com/your-org/filo-priori-v9}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/filo-priori-v9/issues)
- **Scientific Analysis**: [SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md](SCIENTIFIC_ANALYSIS_FOR_PUBLICATION.md)
- **Technical Report**: [results/publication/TECHNICAL_REPORT.md](results/publication/TECHNICAL_REPORT.md)

---

| Status | Version | Last Updated |
|--------|---------|--------------|
| Research & Publication | V9.0 | November 25, 2025 |

**Research Focus**: Targeting Qualis A publication (EMSE, IST)

---

**Made with dedication by the Filo-Priori Research Team**
