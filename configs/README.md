# Configurations - configs/

**Last Updated:** 2025-11-28

---

## Current Best Configuration

**File:** `experiment_07_ranking_optimized.yaml`

This configuration achieves the best results (APFD = 0.6413) and is recommended for new experiments.

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

### Run Best Configuration:

```bash
python main.py --config configs/experiment_07_ranking_optimized.yaml
```

### Run Specific Experiment:

```bash
python main.py --config configs/<experiment_name>.yaml
```

---

## Configuration Structure

```yaml
# Data paths
data:
  train_path: "datasets/train.csv"
  test_path: "datasets/test.csv"
  output_dir: "results/<experiment_name>/"

# Embedding configuration
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  tc_fields: ["TE_Summary", "TC_Steps"]
  commit_fields: ["Commit_Message"]

# Model architecture
model:
  type: "dual_stream_v8"
  semantic_dim: 256
  structural_dim: 256
  fusion_type: "cross_attention"
  structural_stream:
    layer_type: "gat"
    num_layers: 1
    num_heads: 2
    dropout: 0.3

# Training configuration
training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 3.0e-5
  weight_decay: 1.0e-4
  loss:
    type: "weighted_focal"
    focal_alpha: 0.75
    focal_gamma: 2.5
  early_stopping:
    patience: 15
    monitor: "val_f1_macro"

# Hardware
hardware:
  device: "cuda"
```

---

## Model Types

| Type | Description | File |
|------|-------------|------|
| `dual_stream_v8` | Main model (SBERT + GAT + CrossAttention) | `src/models/dual_stream_v8.py` |

---

## Loss Types

| Type | Description | Usage |
|------|-------------|-------|
| `weighted_focal` | Weighted Focal Loss | Best for class imbalance (37:1) |
| `focal` | Focal Loss | Alternative without class weights |
| `ce` | Cross Entropy | Standard classification |
| `weighted_ce` | Weighted Cross Entropy | With class weights |

---

## Hyperparameter Sensitivity

Based on ablation study (see paper):

| Parameter | Most Sensitive | Recommended |
|-----------|---------------|-------------|
| Loss type | Yes (+5.9%) | weighted_focal |
| Focal gamma | Yes (+5.5%) | 2.5 |
| Learning rate | Yes (+4.4%) | 3e-5 |
| GNN layers | Medium (+4.4%) | 1 |
| GNN heads | Low (+2.9%) | 2 |

---

## Creating New Experiments

1. Copy the best configuration:
   ```bash
   cp configs/experiment_07_ranking_optimized.yaml configs/experiment_NEW.yaml
   ```

2. Modify parameters as needed

3. Update `output_dir` to new results directory

4. Run:
   ```bash
   python main.py --config configs/experiment_NEW.yaml
   ```

---

**Maintained by:** Filo-Priori Team
