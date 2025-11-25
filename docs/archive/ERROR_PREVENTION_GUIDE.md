# Error Prevention and Troubleshooting Guide

## Overview

This document describes the comprehensive error prevention system implemented in Filo-Priori V8/V9 to prevent runtime errors when running experiments on remote servers.

## Critical Issues Fixed

### 1. Missing Import in qodo_encoder.py ✓ FIXED

**Issue**: The `qodo_encoder.py` file used `os.makedirs()` and `os.path.join()` without importing the `os` module.

**Symptom**:
```python
NameError: name 'os' is not defined
```

**Fix**: Added `import os` to the file imports.

**Location**: `src/embeddings/qodo_encoder.py:8`

---

### 2. Configuration Validation ✓ FIXED

**Issue**: No validation of configuration files before execution, leading to KeyError and type mismatches during runtime.

**Symptoms**:
- `KeyError: 'semantic'`
- `TypeError: expected int, got str`
- Dimension mismatches between config and model

**Fix**: Created comprehensive configuration validation system.

**New Files**:
- `src/utils/config_validator.py` - Schema-based validation
- `preflight_check.py` - Pre-flight validation script

**Features**:
- Required field validation
- Type checking (int, float, str, dict, bool)
- Range validation (min/max for numeric values)
- Dimension consistency checks
- Dataset file existence validation
- Split ratio validation (must sum to 1.0)

---

### 3. Pre-Flight Validation System ✓ NEW

**Purpose**: Catch ALL potential errors before experiment execution starts.

**Script**: `preflight_check.py`

**Checks Performed**:

1. **Python Environment**
   - Python version ≥ 3.8
   - Virtual environment activation

2. **Configuration File**
   - File exists
   - Valid YAML syntax
   - Schema compliance

3. **Required Dependencies**
   - torch, transformers, sentence-transformers
   - torch-geometric, pandas, numpy
   - scikit-learn, scipy, yaml, tqdm, networkx

4. **Optional Dependencies**
   - tensorboard, matplotlib, seaborn
   - imbalanced-learn

5. **PyTorch & CUDA**
   - PyTorch installation
   - CUDA availability
   - GPU device detection
   - CUDA test tensor creation

6. **Dataset Files**
   - train.csv and test.csv exist
   - File sizes reasonable (>1 MB)
   - CSV readable
   - Columns present

7. **Directory Structure**
   - Required: src/, configs/
   - Optional: cache/, results/, logs/, models/
   - Auto-creates missing optional directories

8. **GPU Availability**
   - GPU memory check
   - Memory estimation for model
   - Warnings if insufficient

9. **Configuration Schema**
   - Runs full config validation
   - Reports all errors at once

10. **Memory Requirements**
    - System RAM check
    - Warns if < 16 GB available

---

### 4. Enhanced Error Handling in main.py ✓ IMPROVED

**Changes**:

1. **Config Loading**:
   - File existence check
   - YAML syntax validation
   - Automatic config validation on load

2. **Better Error Messages**:
   - Clear indication of what failed
   - Suggested fixes
   - Validation errors collected and reported together

---

## Usage Guide

### Before Running Experiments (RECOMMENDED)

Always run the pre-flight check before starting experiments:

```bash
# Basic usage (uses default config)
python preflight_check.py

# With custom config
python preflight_check.py --config configs/custom.yaml

# Make it executable (one-time)
chmod +x preflight_check.py
./preflight_check.py
```

**Example Output**:

```
======================================================================
FILO-PRIORI PRE-FLIGHT VALIDATION
======================================================================

======================================================================
CHECK: Python Environment
======================================================================
  Python version: 3.10.12
✓ Python Environment passed

======================================================================
CHECK: Configuration File
======================================================================
  Config file: configs/experiment.yaml
  Config sections: experiment, data, text, commit, semantic, structural, ...
✓ Configuration File passed

======================================================================
CHECK: Required Dependencies
======================================================================
Checking required packages...
  ✓ torch
  ✓ transformers
  ✓ sentence-transformers
  ✓ torch-geometric
  ✓ pandas
  ...

======================================================================
CHECK: PyTorch & CUDA
======================================================================
  PyTorch version: 2.0.1+cu118
  CUDA available: True
  CUDA version (PyTorch): 11.8
  Number of GPUs: 1
    GPU 0: Quadro RTX 8000 (48.0 GB)
✓ CUDA test tensor creation successful
✓ PyTorch & CUDA passed

======================================================================
CHECK: Dataset Files
======================================================================
  train_path: datasets/train.csv (1.68 GB)
    Columns (12): TC_Key, Build_ID, TE_Summary, TC_Steps, ...
    Sample rows: 5
  test_path: datasets/test.csv (581.23 MB)
    Columns (12): TC_Key, Build_ID, TE_Summary, TC_Steps, ...
    Sample rows: 5
✓ Dataset Files passed

======================================================================
VALIDATION SUMMARY
======================================================================
  Total checks: 9
  Passed: 9
  Failed: 0
  Errors: 0
  Warnings: 0

✓ ALL CHECKS PASSED! Ready to run experiment.
```

### Running Experiments

#### Option 1: Manual Validation (Recommended for Development)

```bash
# 1. Run pre-flight check
python preflight_check.py --config configs/experiment.yaml

# 2. If all checks pass, run experiment
python main.py --config configs/experiment.yaml
```

#### Option 2: Quick Run (Config Validation Automatic)

```bash
# Config validation runs automatically in main.py
python main.py --config configs/experiment.yaml
```

#### Option 3: Using Experiment Runner Script

```bash
# The run_experiment.sh script now includes validation
./run_experiment.sh --config configs/experiment.yaml
```

---

## Common Error Scenarios & Solutions

### Scenario 1: Missing Dataset Files

**Error**:
```
✗ Dataset Files failed
  - Dataset not found: datasets/train.csv
```

**Solution**:
```bash
# Ensure datasets are in correct location
ls -lh datasets/
# Should show:
# -rw-r--r-- 1 user group 1.7G train.csv
# -rw-r--r-- 1 user group 581M test.csv
```

---

### Scenario 2: Configuration Dimension Mismatch

**Error**:
```
✗ Configuration Schema failed
  - Model semantic input_dim (1024) doesn't match semantic combined_embedding_dim (3072)
```

**Solution**:
Edit `configs/experiment.yaml`:
```yaml
model:
  semantic:
    input_dim: 3072  # Must match semantic.combined_embedding_dim
```

---

### Scenario 3: CUDA Not Available

**Error**:
```
✗ PyTorch & CUDA failed
  - Config requires CUDA but no GPU detected
```

**Solutions**:

**Option A**: Fix GPU/CUDA installation
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Option B**: Use CPU (slow!)
```yaml
# configs/experiment.yaml
hardware:
  device: "cpu"  # Change from "cuda" to "cpu"
```

---

### Scenario 4: Missing Dependencies

**Error**:
```
✗ Required Dependencies failed
  - Missing required package: sentence-transformers
  - Missing required package: torch-geometric
```

**Solution**:
```bash
# Activate venv
source venv/bin/activate

# Install missing packages
pip install -r requirements.txt

# Or install individually
pip install sentence-transformers
pip install torch-geometric
```

---

### Scenario 5: Invalid YAML Syntax

**Error**:
```
✗ Configuration File failed
  - Invalid YAML syntax: mapping values are not allowed here
    in "configs/experiment.yaml", line 45, column 12
```

**Solution**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/experiment.yaml'))"

# Common issues:
# - Wrong indentation (use 2 spaces, not tabs)
# - Missing colons after keys
# - Unquoted strings with special characters
```

---

### Scenario 6: GPU Out of Memory

**Warning**:
```
⚠ GPU may have insufficient memory (8.0 GB available, ~8.0 GB recommended)
```

**Solutions**:

**Option A**: Reduce batch size
```yaml
# configs/experiment.yaml
semantic:
  batch_size: 16  # Reduce from 32

training:
  batch_size: 16  # Reduce from 32
```

**Option B**: Reduce max sequence length
```yaml
# configs/experiment.yaml
semantic:
  max_length: 256  # Reduce from 512
```

**Option C**: Use gradient accumulation (future feature)

---

## Server Deployment Checklist

Use this checklist when deploying to IATS/UFAM server:

### Pre-Deployment

- [ ] Copy entire project to server
- [ ] Verify datasets are present (`datasets/train.csv`, `datasets/test.csv`)
- [ ] Activate virtual environment or create new one
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set environment variables (already in code):
  ```bash
  export PYTORCH_NO_NVML=1
  export CUDA_LAUNCH_BLOCKING=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

### Validation

- [ ] Run pre-flight check: `python preflight_check.py`
- [ ] Verify GPU detected: `nvidia-smi`
- [ ] Check CUDA version matches PyTorch
- [ ] Verify disk space: `df -h` (need ~10 GB for results/cache)

### Execution

- [ ] Test with sample data: `python main.py --sample 1000`
- [ ] If sample succeeds, run full experiment
- [ ] Monitor GPU usage: `watch -n 1 nvidia-smi`
- [ ] Monitor logs: `tail -f results/experiment_XXX/output.log`

### Post-Execution

- [ ] Verify results directory created
- [ ] Check APFD results: `results/experiment_XXX/apfd_per_build_FULL_testcsv.csv`
- [ ] Verify model checkpoint saved
- [ ] Review logs for warnings/errors

---

## Configuration Validation Schema

The validation system checks for the following structure:

```yaml
experiment:
  name: str (required)
  version: str
  seed: int (default: 42)

data:
  train_path: str (required, file must exist)
  test_path: str (required, file must exist)
  train_split: float (0.0-1.0, required)
  val_split: float (0.0-1.0, required)
  test_split: float (0.0-1.0, required)
  # Note: train_split + val_split + test_split must = 1.0

semantic:
  model_name: str (required)
  embedding_dim: int (min: 128, required)
  combined_embedding_dim: int (min: 256, required)
  # Note: combined_embedding_dim should = 2 * embedding_dim
  max_length: int (32-32768, required)
  batch_size: int (1-512, required)
  cache_path: str (optional)

structural:
  extractor: dict (required)
  input_dim: int (min: 1, required)

model:
  type: str (required)
  semantic: dict (required)
    input_dim: int (must match semantic.combined_embedding_dim)
  structural: dict (required)
  classifier: dict (required)
  num_classes: int (min: 2, required)

training:
  num_epochs: int (1-1000, required)
  batch_size: int (1-512, required)
  learning_rate: float (1e-7 to 1.0, required)
  weight_decay: float (0.0-1.0, optional)

hardware:
  device: str ("cuda" or "cpu", required)
  num_workers: int (0-32, default: 4)
  pin_memory: bool (default: true)
```

---

## Testing the Fixes

### Test 1: Config Validation

```bash
# This should pass
python preflight_check.py --config configs/experiment.yaml

# Test invalid config
echo "invalid: [broken yaml" > test_bad.yaml
python preflight_check.py --config test_bad.yaml
# Should show error

# Clean up
rm test_bad.yaml
```

### Test 2: Missing Import Fix

```bash
# Test qodo encoder import
python -c "from src.embeddings.qodo_encoder import QodoEncoder; print('Import successful')"
```

### Test 3: Dataset Validation

```bash
# Test with missing dataset
mv datasets/train.csv datasets/train.csv.backup
python preflight_check.py
# Should show error about missing file

# Restore
mv datasets/train.csv.backup datasets/train.csv
```

### Test 4: Full Pipeline (Sample)

```bash
# Run with small sample to test entire pipeline
python main.py --config configs/experiment.yaml --sample 100

# Should complete without errors
```

---

## Continuous Monitoring

### During Experiment Execution

Monitor these to detect issues early:

```bash
# GPU usage (separate terminal)
watch -n 1 nvidia-smi

# Memory usage
watch -n 1 free -h

# Log output
tail -f results/experiment_XXX/output.log

# Disk space
watch -n 60 df -h
```

### Red Flags to Watch For

1. **GPU Memory**: Should stay < 90% of total
2. **System RAM**: Should stay < 80% of total
3. **Disk Space**: Need ~10 GB free for cache/results
4. **Temperature**: GPU temp should stay < 85°C
5. **Log Errors**: Any ERROR or CRITICAL messages

---

## Integration with Existing Workflow

The validation system integrates seamlessly:

### Existing Workflow
```bash
./run_experiment.sh
```

### Enhanced Workflow (Recommended)
```bash
# 1. Pre-flight check
python preflight_check.py

# 2. If passed, run experiment
./run_experiment.sh
```

### Automated Workflow
```bash
# Add to run_experiment.sh (after line 105):
print_info "Running pre-flight checks..."
python preflight_check.py --config "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    print_error "Pre-flight checks failed. Aborting."
    exit 1
fi
```

---

## Summary of Changes

### Files Added
1. `src/utils/config_validator.py` - Configuration validation system
2. `src/utils/__init__.py` - Utils package initialization
3. `preflight_check.py` - Pre-flight validation script
4. `ERROR_PREVENTION_GUIDE.md` - This document

### Files Modified
1. `src/embeddings/qodo_encoder.py` - Added missing `import os`
2. `main.py` - Added config validation on startup

### No Breaking Changes
All changes are backward compatible. The system works with or without validation.

---

## Support

If you encounter issues not covered here:

1. Run pre-flight check with full output:
   ```bash
   python preflight_check.py --config configs/experiment.yaml 2>&1 | tee preflight_log.txt
   ```

2. Check the generated log file

3. Review the specific error messages and refer to this guide

4. For GPU/CUDA issues, also check:
   - `nvidia-smi`
   - `nvcc --version`
   - `python -c "import torch; print(torch.version.cuda)"`

---

## Appendix: Server Specifications Reference

Your IATS/UFAM server specs:

- **CPU**: Intel Xeon W-2235 @ 3.80GHz (12 cores)
- **RAM**: 125 GB (79 GB available)
- **GPU**: Quadro RTX 8000 (48 GB VRAM)
- **CUDA**: 12.2
- **Driver**: 535.247.01
- **Disk**: 937 GB total, 756 GB available

This hardware is excellent for deep learning. The validation system ensures you use it efficiently.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Tested On**: Ubuntu 22.04 LTS, Python 3.10, PyTorch 2.0.1, CUDA 11.8/12.2
