# Filo-Priori V8: Phylogenetic Graph-Based Test Case Prioritization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

**A state-of-the-art deep learning system for intelligent test case prioritization in CI/CD pipelines.**

---

## Overview

Filo-Priori V8 is a novel neural network architecture that combines **semantic understanding** of test cases with **phylogenetic (historical execution) patterns** to predict test failures and prioritize test execution order. The model achieves **APFD = 0.6171** (61.71% of faults detected by halfway point), representing a **23.4% improvement over random** test ordering.

### Key Features

- **Dual-Stream Architecture**: Separate processing of semantic (SBERT embeddings) and structural (10 phylogenetic features) information
- **Graph Attention Networks (GAT)**: 2-head attention learns dynamic importance of test relationships
- **Multi-Edge Phylogenetic Graph**: Three complementary edge types:
  - Co-failure edges (weight=1.0): Tests failing together
  - Co-success edges (weight=0.5): Tests passing together (stability patterns)
  - Semantic edges (weight=0.3): Content similarity (cosine > 0.75)
- **Expert-Selected Features**: 10 carefully chosen features from initial 29 candidates
  - 6 phylogenetic: test_age, recent_failure_rate, very_recent_failure_rate, medium_term_failure_rate, failure_streak, pass_streak
  - 4 structural: num_commits, num_change_requests, commit_surge, execution_stability
- **Orphan Handling**: Smart inference for new test cases never seen during training
- **Production-Ready**: Optimized for real-world CI/CD deployment with caching

---

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean APFD** | **0.6171** | Average Percentage of Faults Detected |
| **F1-Macro** | **0.5312** | Balanced performance across Pass/Fail |
| **APFD ≥ 0.7** | **40.8%** of builds | High-quality prioritization |
| **Improvement vs Random** | **+23.4%** | APFD increase over random ordering |

**Dataset**: 52,102 test executions, 1,339 builds, 2,347 unique test cases

---

## Scientific Contributions

### 1. Dual-Stream Architecture

**Novelty**: Orthogonal information sources for test case prioritization:
- **Semantic Stream**: SBERT embeddings (1536-dim) from all-mpnet-base-v2
  - Dual-field encoding: (summary+steps) || commits
  - Captures test content and intent
- **Structural Stream**: 10 phylogenetic features from execution history
  - Temporal patterns, failure rates, commit metadata
  - Prevents semantic echo chamber problem

**Impact**: Enables model to learn both "what the test does" (semantic) and "how it behaves" (structural).

### 2. Multi-Edge Phylogenetic Graph

**Novelty**: Three complementary relationship types in a unified graph:
- **Co-failure edges** (weight=1.0): P(A fails | B fails) - captures fault propagation
- **Co-success edges** (weight=0.5): P(A passes | B passes) - captures stability patterns
- **Semantic edges** (weight=0.3): Cosine similarity > 0.75 - captures conceptual similarity

**Impact**: Richer relationship modeling enables GAT to learn which connections matter for failure prediction. First multi-edge approach for TCP.

### 3. Graph Attention Networks for TCP

**Novelty**: Dynamic attention mechanism learns edge importance:
- 2-head GAT (128-dim total, 64 per head)
- 1 layer with ELU activation
- Learns to weight different edge types differently per test case
- **Orphan handling**: New test cases (global_idx=-1) get default prediction [0.5, 0.5]

**Impact**: Outperforms static graph convolution by adapting to test-specific relationship patterns.

### 4. Expert-Guided Feature Selection Methodology

**Novelty**: Systematic ablation-based feature selection:
- **Phase 1**: Baseline with 6 proven features (APFD ≈ 0.62)
- **Phase 2**: Expansion to 29 features (exploratory)
- **Phase 3**: Selection of 10 features via expert analysis + ablation (APFD = 0.6171)

**Selected Features**:
- **Phylogenetic (6)**: test_age, recent_failure_rate, very_recent_failure_rate, medium_term_failure_rate, failure_streak, pass_streak
- **Structural (4)**: num_commits, num_change_requests, commit_surge, execution_stability

**Impact**: Achieves production performance while preventing overfitting. Replicable methodology for feature selection in temporal data.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/filo-priori-v8.git
cd filo-priori-v8

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train production model (Experiment 06)
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

---

## Documentation

**Comprehensive Technical Report**: [results/publication/TECHNICAL_REPORT.md](results/publication/TECHNICAL_REPORT.md)

Includes:
- Detailed architecture explanation
- Mathematical formulations
- Scientific contributions
- Ablation studies
- Reproducibility guidelines

---

## Architecture Diagrams

Comprehensive visual documentation is available in the `figures/` directory:

- **complete_pipeline_architecture.mmd**: Full end-to-end pipeline (both phases)
- **step_2.1_structural_features_extraction.md**: Feature extraction deep dive
- **step_2.1_data_flow.md**: Technical data flow and implementation
- **step_2.2_model_architecture.md**: Model architecture and phylogenetic graphs
- **step_2.2_implementation_details.md**: Implementation details and configuration

Additional visualizations in `results/publication/`:
- **INFERENCE_EXPLANATION.md**: How inference works for new test cases
- **GRAPH_CONSTRUCTION_STEP_BY_STEP.md**: Multi-edge graph construction
- **FEATURE_EXPANSION_ANALYSIS.md**: Feature selection analysis
- **multi_edge_phylogenetic_graph_interactive.html**: Interactive graph visualization

View diagrams:
- GitHub/GitLab: Automatic Mermaid rendering
- VS Code: Install "Markdown Preview Mermaid Support"
- Online: https://mermaid.live/

---

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB RAM minimum
- 8GB+ VRAM recommended

**Key Dependencies**:
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- Sentence-Transformers 2.2+
- pandas, numpy, scikit-learn

---

## Project Structure

```
filo-priori-v8/
├── configs/                    # Experiment configurations
├── src/                        # Source code
│   ├── models/                 # Neural network models
│   ├── preprocessing/          # Feature extraction
│   ├── phylogenetic/           # Graph construction
│   └── evaluation/             # Metrics
├── results/                    # Experimental results
│   ├── experiment_06_feature_selection/  # Production model
│   └── publication/            # Technical report + visualizations
├── main.py                     # Training script
└── predict.py                  # Inference script
```

---

## Citation

```bibtex
@software{filo_priori_v8_2025,
  title={Filo-Priori V8: Phylogenetic Graph-Based Dual-Stream Neural Network for Test Case Prioritization},
  author={Filo-Priori Team},
  year={2025},
  version={1.0}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/filo-priori-v8/issues)
- **Documentation**: [Technical Report](results/publication/TECHNICAL_REPORT.md)

---

**Status**: Production-Ready ✅
**Version**: 1.0
**Last Updated**: November 14, 2025

---

**Made with ❤️ by the Filo-Priori Team**
