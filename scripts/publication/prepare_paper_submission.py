#!/usr/bin/env python3
"""
Prepare Paper Submission for Filo-Priori v9.

1. Reviews and fixes LaTeX files
2. Organizes figures for paper
3. Creates proper directory structure
4. Generates figure inclusion code

Author: Filo-Priori Team
Date: 2025-11-26
"""

import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def fix_latex_tables():
    """Fix issues in LaTeX tables."""

    all_tables_path = PROJECT_ROOT / 'results' / 'paper_sections' / 'all_tables.tex'

    if all_tables_path.exists():
        with open(all_tables_path, 'r') as f:
            content = f.read()

        # Fix 'nan' values
        content = content.replace(' nan ', ' ')
        content = content.replace('nan \\\\', ' \\\\')
        content = content.replace('& nan', '& -')

        # Save fixed version
        with open(all_tables_path, 'w') as f:
            f.write(content)

        print(f"   Fixed: {all_tables_path.name}")

    return True


def create_paper_structure():
    """Create proper paper directory structure."""

    paper_dir = PROJECT_ROOT / 'paper'

    # Create directories
    dirs = [
        paper_dir,
        paper_dir / 'figures',
        paper_dir / 'tables',
        paper_dir / 'sections',
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"   Created paper directory structure at: {paper_dir}")

    return paper_dir


def collect_figures(paper_dir):
    """Collect and organize all figures for the paper."""

    figures_dir = paper_dir / 'figures'

    # Define figures to include with their new names
    figures = {
        # RQ1: Baseline Comparison
        'fig_rq1_apfd_comparison': 'results/publication_final/apfd_boxplot_publication.pdf',
        'fig_rq1_improvement': 'results/publication_final/improvement_publication.pdf',

        # RQ2: Ablation Study
        'fig_rq2_ablation': 'results/ablation/ablation_study_final.pdf',

        # RQ3: Temporal CV
        'fig_rq3_temporal': 'results/temporal_cv/temporal_cv_results.pdf',

        # RQ4: Sensitivity
        'fig_rq4_sensitivity': 'results/sensitivity/sensitivity_analysis.pdf',

        # Qualitative Analysis
        'fig_qualitative': 'results/qualitative_analysis/qualitative_analysis.pdf',
    }

    # Also collect PNG versions
    png_figures = {
        'fig_rq1_apfd_comparison': 'results/publication_final/apfd_boxplot_publication.png',
        'fig_rq1_improvement': 'results/publication_final/improvement_publication.png',
        'fig_rq2_ablation': 'results/ablation/ablation_study_final.png',
        'fig_rq3_temporal': 'results/temporal_cv/temporal_cv_results.png',
        'fig_rq4_sensitivity': 'results/sensitivity/sensitivity_analysis.png',
        'fig_qualitative': 'results/qualitative_analysis/qualitative_analysis.png',
    }

    collected = []

    # Copy PDF figures
    for new_name, src_path in figures.items():
        src = PROJECT_ROOT / src_path
        if src.exists():
            dst = figures_dir / f"{new_name}.pdf"
            shutil.copy2(src, dst)
            collected.append(dst.name)
        else:
            print(f"   Warning: {src_path} not found")

    # Copy PNG figures
    for new_name, src_path in png_figures.items():
        src = PROJECT_ROOT / src_path
        if src.exists():
            dst = figures_dir / f"{new_name}.png"
            shutil.copy2(src, dst)

    print(f"   Collected {len(collected)} figures")

    return collected


def collect_tables(paper_dir):
    """Collect all LaTeX tables."""

    tables_dir = paper_dir / 'tables'

    # Source tables
    tables = {
        'tab_comparison.tex': 'results/baselines/comparison_table.tex',
        'tab_ablation.tex': 'results/ablation/ablation_study_final.tex',
        'tab_temporal_cv.tex': 'results/temporal_cv/temporal_cv_table.tex',
        'tab_sensitivity.tex': 'results/sensitivity/sensitivity_analysis.tex',
        'tab_case_studies.tex': 'results/qualitative_analysis/case_studies.tex',
    }

    collected = []

    for new_name, src_path in tables.items():
        src = PROJECT_ROOT / src_path
        if src.exists():
            # Read and fix content
            with open(src, 'r') as f:
                content = f.read()

            # Fix common issues
            content = content.replace(' nan ', ' - ')
            content = content.replace('nan \\\\', '- \\\\')
            content = content.replace('& nan', '& -')

            # Save to new location
            dst = tables_dir / new_name
            with open(dst, 'w') as f:
                f.write(content)

            collected.append(new_name)

    print(f"   Collected {len(collected)} tables")

    return collected


def collect_sections(paper_dir):
    """Collect all LaTeX sections."""

    sections_dir = paper_dir / 'sections'

    sections = {
        'results.tex': 'results/paper_sections/section_results.tex',
        'discussion.tex': 'results/paper_sections/section_discussion.tex',
        'threats.tex': 'results/paper_sections/section_threats.tex',
    }

    collected = []

    for new_name, src_path in sections.items():
        src = PROJECT_ROOT / src_path
        if src.exists():
            # Read content
            with open(src, 'r') as f:
                content = f.read()

            # Fix table/figure references to use new paths
            content = content.replace('\\input{tables/', '\\input{tables/tab_')

            # Save to new location
            dst = sections_dir / new_name
            with open(dst, 'w') as f:
                f.write(content)

            collected.append(new_name)

    print(f"   Collected {len(collected)} sections")

    return collected


def generate_figures_latex(paper_dir):
    """Generate LaTeX code for including figures."""

    latex = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURES - FILO-PRIORI V9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auto-generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
%
% Usage: \\input{figures.tex} in your main document
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------------------------------------------------------------
% Figure 1: APFD Comparison (RQ1)
%------------------------------------------------------------------------------
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{figures/fig_rq1_apfd_comparison.pdf}
\\caption{Box plot comparison of APFD scores across TCP methods on QTA dataset
(N=277 builds). Filo-Priori achieves competitive performance with traditional
baselines while significantly outperforming recency-based approaches
($p < 0.001$, Wilcoxon signed-rank test). Whiskers extend to 1.5 IQR.}
\\label{fig:apfd_comparison}
\\end{figure}

%------------------------------------------------------------------------------
% Figure 2: Improvement over Random (RQ1)
%------------------------------------------------------------------------------
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.85\\textwidth]{figures/fig_rq1_improvement.pdf}
\\caption{Relative improvement in APFD over random ordering for each TCP method.
Filo-Priori achieves +10.3\\% improvement, significantly outperforming Random
($p < 0.001$). Error bars represent 95\\% bootstrap confidence intervals.}
\\label{fig:improvement}
\\end{figure}

%------------------------------------------------------------------------------
% Figure 3: Ablation Study (RQ2)
%------------------------------------------------------------------------------
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{figures/fig_rq2_ablation.pdf}
\\caption{Component contributions to Filo-Priori performance. Graph Attention
(GATv2) provides the largest contribution (+17.0\\%), followed by the Structural
Stream (+5.3\\%) and Class Weighting (+4.6\\%). Red bars indicate statistically
significant contributions ($p < 0.05$, paired t-test).}
\\label{fig:ablation}
\\end{figure}

%------------------------------------------------------------------------------
% Figure 4: Temporal Cross-Validation (RQ3)
%------------------------------------------------------------------------------
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.85\\textwidth]{figures/fig_rq3_temporal.pdf}
\\caption{Temporal cross-validation results showing consistent APFD performance
across different validation strategies. The narrow range (0.619--0.663) indicates
temporal robustness and minimal concept drift impact. Error bars represent
95\\% bootstrap confidence intervals.}
\\label{fig:temporal_cv}
\\end{figure}

%------------------------------------------------------------------------------
% Figure 5: Hyperparameter Sensitivity (RQ4)
%------------------------------------------------------------------------------
\\begin{figure*}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{figures/fig_rq4_sensitivity.pdf}
\\caption{Hyperparameter sensitivity analysis. (a) Loss function has the highest
impact on performance ($\\Delta$ = 5.9\\%). (b) Lower learning rate (3e-5)
outperforms higher rate. (c) Simpler GNN architecture performs better.
(d) 10 selected features optimal. (e) Balanced sampling degrades performance.
(f) Summary of optimal configuration.}
\\label{fig:sensitivity}
\\end{figure*}

%------------------------------------------------------------------------------
% Figure 6: Qualitative Analysis
%------------------------------------------------------------------------------
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{figures/fig_qualitative.pdf}
\\caption{Qualitative analysis of Filo-Priori performance. (a) APFD score
distribution across 277 builds. (b) First failure detection position distribution.
(c) Failure detection curve showing that running 25\\% of tests detects 33.2\\%
of failures. (d) Relationship between build size and APFD score.}
\\label{fig:qualitative}
\\end{figure}

"""

    figures_path = paper_dir / 'figures.tex'
    with open(figures_path, 'w') as f:
        f.write(latex)

    print(f"   Generated: figures.tex")

    return figures_path


def generate_main_template(paper_dir):
    """Generate main paper template."""

    template = r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILO-PRIORI V9 - PAPER TEMPLATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Target: EMSE / IST (Qualis A)
%
% Structure:
%   - sections/results.tex
%   - sections/discussion.tex
%   - sections/threats.tex
%   - tables/*.tex
%   - figures/*.pdf
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[review]{elsarticle}

\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{xcolor}

% For table notes
\usepackage{threeparttable}
\newenvironment{tablenotes}{\begin{tablenotes}}{\end{tablenotes}}

\journal{Empirical Software Engineering}

\begin{document}

\begin{frontmatter}

\title{Filo-Priori: Deep Learning-based Test Case Prioritization with Graph Attention Networks}

\author[inst1]{Author One}
\author[inst1]{Author Two}
\author[inst2]{Author Three}

\affiliation[inst1]{organization={University One},
            city={City},
            country={Country}}

\affiliation[inst2]{organization={University Two},
            city={City},
            country={Country}}

\begin{abstract}
Test Case Prioritization (TCP) aims to order test cases to maximize early fault
detection in Continuous Integration environments. We present Filo-Priori, a
deep learning approach that combines semantic embeddings, structural features,
and Graph Attention Networks (GATv2) to model test case relationships through
a phylogenetic graph. Our evaluation on an industrial dataset (277 builds,
8,847 test executions) shows that Filo-Priori achieves APFD = 0.6171,
representing a 10.3\% improvement over random ordering ($p < 0.001$).
An ablation study reveals that Graph Attention Networks contribute +17.0\%
to performance. Temporal cross-validation confirms robustness across different
time periods (APFD range: 0.619--0.663). Running only 25\% of tests detects
33.2\% of failures on average.
\end{abstract}

\begin{keyword}
Test Case Prioritization \sep
Deep Learning \sep
Graph Neural Networks \sep
Continuous Integration \sep
Software Testing
\end{keyword}

\end{frontmatter}

%------------------------------------------------------------------------------
% RESULTS SECTION
%------------------------------------------------------------------------------
\input{sections/results}

%------------------------------------------------------------------------------
% DISCUSSION SECTION
%------------------------------------------------------------------------------
\input{sections/discussion}

%------------------------------------------------------------------------------
% THREATS TO VALIDITY
%------------------------------------------------------------------------------
\input{sections/threats}

%------------------------------------------------------------------------------
% FIGURES
%------------------------------------------------------------------------------
\input{figures}

%------------------------------------------------------------------------------
% REFERENCES
%------------------------------------------------------------------------------
\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}
"""

    template_path = paper_dir / 'main.tex'
    with open(template_path, 'w') as f:
        f.write(template)

    print(f"   Generated: main.tex (paper template)")

    return template_path


def generate_readme(paper_dir):
    """Generate README for paper directory."""

    readme = """# Filo-Priori v9 - Paper Submission Materials

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
2. Include sections using `\\input{sections/results}` etc.
3. Include tables using `\\input{tables/tab_comparison}` etc.
4. Include figures using the code in `figures.tex`

## Key Results

| Metric | Value |
|--------|-------|
| Mean APFD | 0.6171 [0.586, 0.648] |
| vs Random | +10.3% (p < 0.001) |
| Most important component | GATv2 (+17.0%) |
| Temporal robustness | 0.619-0.663 |
| Failures in top 25% | 33.2% |

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

Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""

    readme_path = paper_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"   Generated: README.md")

    return readme_path


def main():
    print("\n" + "=" * 80)
    print(" PREPARING PAPER SUBMISSION MATERIALS")
    print("=" * 80)

    # 1. Fix LaTeX tables
    print("\n1. Fixing LaTeX tables...")
    fix_latex_tables()

    # 2. Create paper structure
    print("\n2. Creating paper directory structure...")
    paper_dir = create_paper_structure()

    # 3. Collect figures
    print("\n3. Collecting figures...")
    figures = collect_figures(paper_dir)

    # 4. Collect tables
    print("\n4. Collecting tables...")
    tables = collect_tables(paper_dir)

    # 5. Collect sections
    print("\n5. Collecting sections...")
    sections = collect_sections(paper_dir)

    # 6. Generate figures LaTeX
    print("\n6. Generating figures.tex...")
    generate_figures_latex(paper_dir)

    # 7. Generate main template
    print("\n7. Generating main.tex template...")
    generate_main_template(paper_dir)

    # 8. Generate README
    print("\n8. Generating README...")
    generate_readme(paper_dir)

    # Summary
    print("\n" + "=" * 80)
    print(" PAPER SUBMISSION MATERIALS READY")
    print("=" * 80)

    print(f"\n   Output directory: {paper_dir}")
    print("\n   Contents:")

    for item in sorted(paper_dir.rglob('*')):
        if item.is_file():
            rel_path = item.relative_to(paper_dir)
            print(f"      {rel_path}")

    print("\n" + "=" * 80)
    print(" NEXT STEPS")
    print("=" * 80)
    print("""
   1. Copy 'paper/' directory to your LaTeX project
   2. Include sections: \\input{sections/results}
   3. Include tables: \\input{tables/tab_comparison}
   4. Include figures: See figures.tex for code
   5. Compile with: pdflatex main.tex

   Target journals:
   - Empirical Software Engineering (EMSE)
   - Information and Software Technology (IST)
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
