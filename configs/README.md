# Configurations - configs/

**Last Updated:** 2025-11-28

---

## Datasets Configurations

Filo-Priori supports multiple datasets for evaluation:

| Config | Dataset | Description |
|--------|---------|-------------|
| `experiment_industry.yaml` | Industrial QTA | Main industrial dataset (52K executions) |
| `experiment_rtptorrent.yaml` | RTPTorrent | Open-source MSR 2020 dataset (20 Java projects) |
| `experiment_cross_dataset.yaml` | Cross-Dataset | Train Industry / Test RTPTorrent |

### Data Paths:

```yaml
# Industrial Dataset
data:
  train_path: "datasets/01_industry/train.csv"
  test_path: "datasets/01_industry/test.csv"

# RTPTorrent Dataset (after preprocessing)
data:
  train_path: "datasets/02_rtptorrent/processed/train.csv"
  test_path: "datasets/02_rtptorrent/processed/test.csv"
```

---

## Current Best Configuration

**File:** `experiment_07_ranking_optimized.yaml`

This configuration achieves the best results (APFD = 0.6413) on the Industrial dataset.

### Key Settings:

| Category | Setting | Value |
|----------|---------|-------|
| **Model** | Type | `dual_stream_v8` |
| **Semantic** | SBERT model | `all-mpnet-base-v2` |
| **Structural** | Layer type | GATConv |
| **Structural** | Layers | 1 |
| **Structural** | Heads | 2 |
| **Loss** | Type | `weighted_focal` |
| **Loss** | Alpha | 0.75 |
| **Loss** | Gamma | 2.5 |
| **Training** | Learning rate | 3e-5 |
| **Training** | Weight decay | 1e-4 |
| **Training** | Batch size | 32 |
| **Training** | Epochs | 50 |
| **Training** | Early stopping | patience=15 |

---

## Usage

### Run on Industrial Dataset (default):

```bash
python main.py --config configs/experiment_industry.yaml
```

### Run on RTPTorrent Dataset:

```bash
# First, download and preprocess the dataset
python scripts/preprocessing/download_rtptorrent.py
python scripts/preprocessing/preprocess_rtptorrent.py

# Then run the experiment
python main.py --config configs/experiment_rtptorrent.yaml
```

### Run Cross-Dataset Evaluation:

```bash
python main.py --config configs/experiment_cross_dataset.yaml
```

---

## Configuration Files

| File | Purpose | Dataset |
|------|---------|---------|
| `experiment_industry.yaml` | Industrial evaluation | 01_industry |
| `experiment_rtptorrent.yaml` | RTPTorrent evaluation | 02_rtptorrent |
| `experiment_cross_dataset.yaml` | Cross-domain evaluation | Industry -> RTPTorrent |
| `experiment_improved.yaml` | General improved settings | 01_industry |
| `experiment_07_ranking_optimized.yaml` | Best performing config | 01_industry |
| `experiment_phylogenetic.yaml` | Phylogenetic variant (experimental) | 01_industry |

---

## Configuration Structure

```yaml
# Experiment metadata
experiment:
  name: "experiment_name"
  version: "2.0.0"

# Data paths
data:
  train_path: "datasets/01_industry/train.csv"
  test_path: "datasets/01_industry/test.csv"

# Embedding configuration
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  cache_dir: "cache/01_industry"

# Model architecture
model:
  type: "dual_stream"
  semantic:
    input_dim: 1536
    hidden_dim: 256
  structural:
    input_dim: 6
    hidden_dim: 64
  gnn:
    type: "GAT"
    num_layers: 1
    num_heads: 2

# Training configuration
training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 3.0e-5
  loss:
    type: "weighted_focal"
    focal_alpha: 0.75
    focal_gamma: 2.5

# Output paths
output:
  results_dir: "results/experiment_name"
```

---

## Model Types

| Type | Description | File |
|------|-------------|------|
| `dual_stream` | Main model (SBERT + GAT + CrossAttention) | `src/models/dual_stream_v8.py` |

---

## Loss Types

| Type | Description | Usage |
|------|-------------|-------|
| `weighted_focal` | Weighted Focal Loss | Best for class imbalance (37:1) |
| `focal` | Focal Loss | Alternative without class weights |
| `ce` | Cross Entropy | Standard classification |
| `weighted_ce` | Weighted Cross Entropy | With class weights |

---

## Dataset-Specific Recommendations

### Industrial Dataset:
- Rich semantic information (test descriptions, commit messages)
- 37:1 class imbalance
- Recommended: `focal_gamma: 2.5-3.0`, balanced sampling

### RTPTorrent Dataset:
- Limited semantic information (test names only)
- Varies by project
- Recommended: Rely more on structural features
- Use smaller embedding batch sizes

### Cross-Dataset:
- Higher dropout for regularization
- Label smoothing (0.05)
- Separate cache directories per dataset

---

## Creating New Experiments

1. Copy a base configuration:
   ```bash
   cp configs/experiment_industry.yaml configs/experiment_NEW.yaml
   ```

2. Modify parameters as needed

3. Update `output.results_dir` to new results directory

4. Update `cache_dir` paths if needed

5. Run:
   ```bash
   python main.py --config configs/experiment_NEW.yaml
   ```

---

**Maintained by:** Filo-Priori Team
