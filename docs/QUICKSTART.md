# Quick Start Guide

Get up and running with Filo-Priori in minutes.

---

## Installation (First Time Only)

```bash
# 1. Navigate to project
cd filo_priori_v8

# 2. Run experiment (auto-installs everything)
./run_experiment.sh
```

That's it! The script handles:
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Embedding generation & caching
- ✅ Model training
- ✅ Evaluation & results

**First run:** ~5-10 minutes (includes embedding generation)
**Subsequent runs:** ~2-5 minutes (uses cached embeddings)

---

## Basic Usage

### Run Experiment (Recommended)

```bash
# Use cached embeddings (fast)
./run_experiment.sh
```

### Force Regenerate Embeddings

```bash
# If you updated the data
./run_experiment.sh --force-regen
```

### Clear Everything and Start Fresh

```bash
# Nuclear option
./run_experiment.sh --clear-cache
```

---

## Understanding the Output

### During Execution

```
======================================================================
               FILO-PRIORI EXPERIMENT RUNNER
======================================================================

Activating virtual environment...
✓ Virtual environment activated

Checking dependencies...
✓ Dependencies already installed

Cache Status:
  ✓ Embeddings cached (3.5M, created: 2024-11-14)
  → Will reuse cached embeddings

======================================================================
                    STARTING EXPERIMENT
======================================================================

Epoch 1/50
  train_loss: 0.6234  val_loss: 0.5891  val_f1_macro: 0.4521
Epoch 2/50
  train_loss: 0.5789  val_loss: 0.5456  val_f1_macro: 0.4892
...
```

### Results

Check `results/<experiment_name>/`:

```
results/filo_priori_sbert_2024-11-14_10-30-45/
├── test_metrics.json           ← Final performance metrics
├── predictions.csv             ← Test predictions
├── rankings.csv                ← Test case rankings
├── apfd_per_build.csv          ← APFD scores
├── confusion_matrix.png        ← Visual results
└── best_model.pt               ← Trained model
```

**Key Metrics to Check:**
- `test_metrics.json` → Look for `accuracy`, `f1_macro`, `auprc_macro`
- `apfd_per_build.csv` → Average APFD score

---

## Common Tasks

### 1. View Cached Embeddings Info

```bash
ls -lh cache/
cat cache/embeddings_metadata.txt
```

### 2. Test Embedding System

```bash
./venv/bin/python scripts/test_sbert_encoding.py
```

### 3. Generate Embeddings Only

```bash
./venv/bin/python scripts/precompute_embeddings_sbert.py \
    --config configs/experiment.yaml \
    --output cache/embeddings.npz \
    --batch_size 128 \
    --device cuda
```

### 4. View Results

```bash
# List all experiments
ls -lt results/

# View latest metrics
cat results/$(ls -t results/ | head -1)/test_metrics.json
```

---

## Configuration Basics

Edit `configs/experiment.yaml`:

### Change Embedding Batch Size

```yaml
embedding:
  batch_size: 64  # Reduce if GPU memory issues
```

### Change Training Parameters

```yaml
training:
  epochs: 30         # Reduce for faster experiments
  batch_size: 16     # Reduce if GPU memory issues
  learning_rate: 1e-4  # Increase for faster convergence
```

### Disable GPU (Use CPU)

```yaml
embedding:
  device: "cpu"

system:
  device: "cpu"
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# In configs/experiment.yaml
embedding:
  batch_size: 32  # Reduce from 128
training:
  batch_size: 8   # Reduce from 32
```

### Issue: "Embeddings taking too long"

**Check:**
1. Is cache enabled? → `embedding.use_cache: true`
2. Using GPU? → `embedding.device: "cuda"`
3. First run? → Embedding generation is normal (~1-2 min)
4. Subsequent run slow? → Try `--force-regen` to rebuild cache

### Issue: "Results poor (low F1)"

**Try:**
```yaml
training:
  epochs: 100           # More training
  learning_rate: 5e-5   # Tune LR
  early_stopping:
    patience: 20        # More patience
```

### Issue: "Cache not updating after data change"

**Solution:**
```bash
# Force regeneration
./run_experiment.sh --force-regen

# Or clear cache
./run_experiment.sh --clear-cache
```

---

## Next Steps

1. **Read the README**: `README.md`
2. **Technical Details**: `docs/TECHNICAL_GUIDE.md`
3. **Customize Config**: `configs/experiment.yaml`
4. **Run Experiments**: `./run_experiment.sh`
5. **Analyze Results**: `results/<experiment>/`

---

## Tips for Best Results

### 1. Always Use Cache

The embedding cache saves **massive** time:
- First run: ~2 minutes
- Cached runs: ~instant

Don't regenerate unless data actually changed!

### 2. Monitor Training

Watch for:
- `val_f1_macro` increasing
- `val_loss` decreasing
- Early stopping at appropriate epoch

### 3. Experiment Incrementally

Start with default config, then tune:
1. Run baseline experiment
2. Check results
3. Adjust one parameter
4. Re-run and compare
5. Iterate

### 4. Use Appropriate Hardware

Recommended:
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 3060 or better)
- **RAM**: 8GB+ system RAM
- **Storage**: 1GB+ free space

Can run on CPU but will be slower (3-5x).

---

## Example Workflow

```bash
# Day 1: Initial setup
./run_experiment.sh
# → Generates embeddings, trains model, ~5-10 min

# Day 2: Tune hyperparameters
# Edit configs/experiment.yaml (change learning_rate, etc.)
./run_experiment.sh
# → Uses cached embeddings, trains model, ~2-5 min

# Day 3: New data added
# Update datasets/train.csv
./run_experiment.sh --force-regen
# → Regenerates embeddings (data changed), trains, ~5-10 min

# Day 4: Quick test
./run_experiment.sh
# → Uses cache, fast run, ~2-5 min
```

---

## Help

```bash
# Show script options
./run_experiment.sh --help
```

For more help:
- Check `docs/TECHNICAL_GUIDE.md`
- Open an issue on GitHub
- Review logs in `logs/`

---

Happy experimenting! 🚀
