# Quick Run Guide - Filo-Priori V8

**Status:** ✅ Pipeline Working!

---

## Running Experiments

### 1. Quick Test (Recommended First)
```bash
./run_experiment.sh
```
- Uses sample_size from config (default: full dataset)
- Embeddings cached automatically
- Results in `results/`

### 2. Small Sample Test (Fast - 30 seconds)
```bash
./venv/bin/python main.py --config configs/experiment.yaml --sample-size 500
```
- Quick validation that everything works
- Good for debugging

### 3. Full Experiment (Production)
```bash
./venv/bin/python main.py --config configs/experiment.yaml
```
- Uses all data (~30K samples)
- Takes ~5-10 minutes
- Best model quality

### 4. Force Regenerate Embeddings
```bash
./venv/bin/python main.py --config configs/experiment.yaml --force-regen-embeddings
```
- Use if you changed data or want fresh embeddings
- Otherwise, cache is automatically reused

---

## What to Expect

### Console Output
```
INFO:utils.config_validator:Configuration validation passed!
INFO:__main__:Using device: cuda

STEP 1: DATA PREPARATION
  Train: X samples
  Val: Y samples
  Test: Z samples

1.2: Extracting semantic embeddings with SBERT...
  Generating/loading embeddings...
  ✓ Encoded in ~1 second

STEP 2: MODEL INITIALIZATION
  Dual-Stream Model V8 initialized

STEP 3: TRAINING
  Epoch 1/50: Train Loss=X.XX, Val Loss=Y.YY, Val F1=Z.ZZ
  ...

STEP 4: TEST EVALUATION
  Test F1: X.XXXX
  Test Accuracy: X.XXXX

STEP 5: RANKING & APFD
  Mean APFD: X.XXXX ⭐

TRAINING COMPLETE!
```

### Expected Metrics (sample_size=500)
```
Best Val F1:  ~0.40-0.50
Test F1:      ~0.30-0.40
Mean APFD:    ~0.45-0.50
```

### Expected Metrics (Full Data)
```
Best Val F1:  ~0.50-0.60 (expected higher)
Test F1:      ~0.45-0.55
Mean APFD:    ~0.55-0.65
```

---

## Output Files

After running, check:

```bash
ls results/
```

You should see:
- `apfd_per_build.csv` - APFD metrics per build
- `prioritized_test_cases.csv` - Ranked test cases
- `best_model_v8.pt` - Best model checkpoint (if val improved)

### View APFD Results
```bash
head results/apfd_per_build.csv
```

### View Rankings
```bash
head results/prioritized_test_cases.csv
```

---

## Troubleshooting

### If you see "Config validation failed"
- Check `configs/experiment.yaml` is correct
- Make sure all required fields are present
- See `PIPELINE_SUCCESS_SUMMARY.md` for details

### If embeddings take too long
- First run generates embeddings (~1-2 minutes)
- Subsequent runs load from cache (~2 seconds)
- Use `--force-regen-embeddings` only if needed

### If CUDA out of memory
- Reduce `batch_size` in config (default: 32 → try 16)
- Or use CPU: `--device cpu` (slower but works)

### If validation F1 = 0.0 throughout
- Normal for small sample_size (e.g., 500)
- Validation set might be too small or imbalanced
- Use full dataset for better validation

---

## Performance Tips

1. **Use cache** - Don't regenerate embeddings unless needed
2. **Start small** - Test with `--sample-size 500` first
3. **Monitor GPU** - Check `nvidia-smi` for memory usage
4. **Check logs** - Read console output for warnings

---

## Next Experiments

### Try Different Configurations

Edit `configs/experiment.yaml`:

```yaml
# Training
training:
  learning_rate: 0.0001  # Try different values
  num_epochs: 100        # More epochs

# Loss
loss:
  focal:
    alpha: 0.5           # Adjust class weights
    gamma: 2.0           # Focus on hard examples

# Model
model:
  semantic:
    hidden_dim: 512      # Larger hidden layer
```

Then run:
```bash
./run_experiment.sh
```

---

## Success Checklist

When experiment completes successfully, you should have:

- [x] "Configuration validation passed!" message
- [x] Training completed (all epochs or early stopped)
- [x] Test metrics calculated (F1, Accuracy, etc.)
- [x] Mean APFD reported
- [x] `results/apfd_per_build.csv` file created
- [x] `results/prioritized_test_cases.csv` file created

If all checked, **congratulations!** Your experiment ran successfully! 🎉

---

## Common Commands Reference

```bash
# Run experiment
./run_experiment.sh

# Run with custom config
./run_experiment.sh --config configs/my_config.yaml

# Quick test
./venv/bin/python main.py --config configs/experiment.yaml --sample-size 500

# Force regenerate embeddings
./run_experiment.sh --force-regen

# Clear cache
rm -rf cache/

# Check results
ls -lh results/
head results/apfd_per_build.csv

# Monitor GPU
watch -n 1 nvidia-smi

# View full log
tail -f /tmp/experiment_test.log  # If running in background
```

---

**Quick Start:**
```bash
./run_experiment.sh
```

**That's it!** The pipeline handles everything else automatically.

---

*For detailed information, see `PIPELINE_SUCCESS_SUMMARY.md`*
