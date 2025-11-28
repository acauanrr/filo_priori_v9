# Scripts - scripts/

**Last Updated:** 2025-11-28

---

## Directory Structure

```
scripts/
├── preprocessing/         # Dataset preprocessing scripts
│   ├── download_rtptorrent.py     # Download RTPTorrent from Zenodo
│   └── preprocess_rtptorrent.py   # Convert to Filo-Priori format
├── analysis/              # Experimental analysis scripts
├── publication/           # Paper generation scripts
│   ├── generate_paper_figures.py
│   └── generate_paper_sections.py
├── cleanup_project.sh     # Project cleanup utility
├── compare_experiments_quick.sh  # Quick metrics comparison
└── README.md              # This file
```

---

## Preprocessing Scripts

### RTPTorrent Dataset

```bash
# Step 1: Download from Zenodo (4.1 GB)
python scripts/preprocessing/download_rtptorrent.py

# Step 2: Convert to Filo-Priori format
python scripts/preprocessing/preprocess_rtptorrent.py

# Optional: Process specific project
python scripts/preprocessing/preprocess_rtptorrent.py --project commons-math

# List available projects
python scripts/preprocessing/preprocess_rtptorrent.py --list-projects
```

Output location: `datasets/02_rtptorrent/processed/`

---

## Publication Scripts

### Generate Paper Figures

```bash
python scripts/publication/generate_paper_figures.py
```

Generates all figures for the IEEE TSE paper:
- `fig_rq1_apfd_comparison.pdf` - APFD comparison box plot
- `fig_rq1_improvement.pdf` - Improvement over random
- `fig_rq2_ablation.pdf` - Ablation study results
- `fig_rq3_temporal.pdf` - Temporal cross-validation
- `fig_rq4_sensitivity.pdf` - Hyperparameter sensitivity
- `fig_qualitative.pdf` - Qualitative analysis

Output directory: `paper/figures/`

---

## Utility Scripts

### Compare Experiments

```bash
./scripts/compare_experiments_quick.sh
```

Quickly compare metrics across experiments:
- Test Accuracy
- F1 Macro
- Mean APFD

### Cleanup Project

```bash
./scripts/cleanup_project.sh
```

Removes:
- `__pycache__/` directories
- `*.pyc` files
- Temporary logs
- `.DS_Store` files

---

## Analysis Scripts

Located in `scripts/analysis/`:

| Script | Purpose |
|--------|---------|
| `extract_all_metrics.py` | Extract metrics from all experiments |
| `validate_experiment.py` | Validate experiment results |

---

## Usage Examples

### Generate All Publication Materials

```bash
# Generate figures
python scripts/publication/generate_paper_figures.py

# Check figures were created
ls paper/figures/*.pdf
```

### Compare Experiment Results

```bash
# Quick comparison
./scripts/compare_experiments_quick.sh

# Detailed extraction
python scripts/analysis/extract_all_metrics.py
```

### Clean Before Commit

```bash
# Remove temporary files
./scripts/cleanup_project.sh

# Verify clean state
git status
```

---

## Creating New Analysis Scripts

1. Create script in appropriate directory:
   - `scripts/analysis/` for experiment analysis
   - `scripts/publication/` for paper materials

2. Use consistent naming: `<action>_<target>.py`

3. Add docstring at top of file

4. Update this README

---

**Maintained by:** Filo-Priori Team
