# Filo-Priori v9 - Paper Submission Materials

## Directory Structure

```
paper/
├── main.tex              # Main paper template
├── figures.tex           # Figure inclusion code
├── figures/              # All figures (PDF + PNG)
│   ├── fig_rq1_apfd_comparison.pdf
│   ├── fig_rq1_improvement.pdf
│   ├── fig_rq2_ablation.pdf
│   ├── fig_rq3_temporal.pdf
│   ├── fig_rq4_sensitivity.pdf
│   └── fig_qualitative.pdf
├── tables/               # All LaTeX tables
│   ├── tab_comparison.tex
│   ├── tab_ablation.tex
│   ├── tab_temporal_cv.tex
│   ├── tab_sensitivity.tex
│   └── tab_case_studies.tex
└── sections/             # Paper sections
    ├── results.tex
    ├── discussion.tex
    └── threats.tex
```

## Usage

1. Copy this directory to your paper project
2. Include sections using `\input{sections/results}` etc.
3. Include tables using `\input{tables/tab_comparison}` etc.
4. Include figures using the code in `figures.tex`

## Key Results

| Metric | Value |
|--------|-------|
| **Mean APFD** | **0.6413** [0.612, 0.672] |
| vs Random | **+14.6%** (p < 0.001) |
| vs FailureRate | **+2.0%** (beats strongest baseline) |
| Architecture | Hybrid (PhyloEncoder LITE + GATv2) |
| Most important component | GATv2 (+17.0%) |
| Temporal robustness | 0.619-0.663 |
| Key innovation | Phylogenetic encoding + Ranking-aware training |

## Figures

1. **fig_rq1_apfd_comparison** - Box plot of APFD comparison
2. **fig_rq1_improvement** - Improvement over random
3. **fig_rq2_ablation** - Ablation study results
4. **fig_rq3_temporal** - Temporal cross-validation
5. **fig_rq4_sensitivity** - Hyperparameter sensitivity
6. **fig_qualitative** - Qualitative analysis

## Tables

1. **tab_comparison** - TCP method comparison (9 methods)
2. **tab_ablation** - Ablation study (7 components)
3. **tab_temporal_cv** - Temporal CV results
4. **tab_sensitivity** - Hyperparameter sensitivity
5. **tab_case_studies** - Case studies

Generated: 2025-11-26 (updated)
