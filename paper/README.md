# Filo-Priori V9 - Paper Submission Materials

## Paper Title

**Filo-Priori: A Dual-Stream Deep Learning Approach to Test Case Prioritization**

Target Journal: IEEE Transactions on Software Engineering (IEEE TSE)

## Directory Structure

```
paper/
├── main_ieee_tse.tex      # Main paper (IEEE TSE format)
├── references_ieee.bib    # Bibliography
├── figures.tex            # Figure inclusion code
├── figures/               # All figures (PDF + PNG)
│   ├── fig_rq1_apfd_comparison.pdf
│   ├── fig_rq1_improvement.pdf
│   ├── fig_rq2_ablation.pdf
│   ├── fig_rq3_temporal.pdf
│   ├── fig_rq4_sensitivity.pdf
│   └── fig_qualitative.pdf
└── sections/              # Paper sections
    ├── results_ieee.tex
    ├── discussion_ieee.tex
    └── threats_ieee.tex
```

## Compilation

```bash
cd paper/
pdflatex main_ieee_tse.tex
bibtex main_ieee_tse
pdflatex main_ieee_tse.tex
pdflatex main_ieee_tse.tex
```

## Key Results

| Metric | Value |
|--------|-------|
| **Mean APFD** | **0.6413** [0.612, 0.672] |
| vs Random | **+14.6%** (p < 0.001) |
| vs FailureRate | **+2.0%** (beats strongest baseline) |
| Architecture | Dual-Stream (Semantic + Structural) |
| Most important component | Graph Attention (+17.0%) |
| Temporal robustness | 0.619-0.663 |
| Loss function | Weighted Focal Loss |

## Architecture Overview

The paper describes a **dual-stream architecture**:

1. **Semantic Stream**: FFN processing SBERT embeddings (1536-dim)
2. **Structural Stream**: GAT over multi-edge test relationship graph
3. **Cross-Attention Fusion**: Bidirectional attention combining modalities
4. **Classifier**: MLP for binary classification

## Key Contributions

1. **Multi-Edge Test Relationship Graph**: Co-failure + co-success + semantic similarity
2. **Dual-Stream Architecture**: SBERT + GAT combination
3. **Cross-Attention Fusion**: Bidirectional attention for modality combination
4. **Weighted Focal Loss**: Addresses 37:1 class imbalance

## Research Questions

| RQ | Question | Answer |
|----|----------|--------|
| RQ1 | How effective is Filo-Priori? | APFD 0.6413, +14.6% vs Random |
| RQ2 | What components contribute most? | GAT (+17.0%), Structural (+5.3%) |
| RQ3 | Is it temporally robust? | Yes, APFD 0.619-0.663 |
| RQ4 | How sensitive to hyperparameters? | Loss function most important |

## Figures

| Figure | Description |
|--------|-------------|
| fig_rq1_apfd_comparison | Box plot of APFD comparison across methods |
| fig_rq1_improvement | Bar chart of improvement over random |
| fig_rq2_ablation | Ablation study component contributions |
| fig_rq3_temporal | Temporal cross-validation results |
| fig_rq4_sensitivity | Hyperparameter sensitivity analysis |
| fig_qualitative | Qualitative analysis and distributions |

## Regenerating Figures

```bash
cd ..  # Go to project root
python scripts/publication/generate_paper_figures.py
```

## Ablation Study Components

| Component | Contribution | p-value |
|-----------|-------------|---------|
| Graph Attention | +17.0% | < 0.001*** |
| Structural Stream | +5.3% | < 0.001*** |
| Focal Loss | +4.6% | < 0.001*** |
| Class Weighting | +3.5% | 0.002** |
| Semantic Stream | +1.9% | 0.087 |

---

Generated: 2025-11-28 (updated to reflect actual implementation)
