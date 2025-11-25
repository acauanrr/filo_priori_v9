# Troubleshooting Guide

## Common Issues and Solutions

### 1. NVML/CUDA Errors

**Symptoms:**
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
UserWarning: Can't initialize NVML
```

**Cause:**
- NVIDIA driver/NVML instability
- CUDA/PyTorch incompatibility
- GPU memory fragmentation during large encodings

**Solutions:**

#### Option 1: GPU Recovery (Default Behaviour)
The encoder now keeps everything on CUDA and automatically retries with smaller batches, full cache flush, and model reloads:
```bash
# cuda_retries controls how many recovery attempts are made (default 3)
python main.py --config configs/experiment.yaml
```
Monitor logs for:
- `Clearing CUDA cache (recovery attempt X)`
- `Reloading Qodo model on CUDA`

If retries are exhausted, fix the environment before re-running.

> ℹ️ The runner and encoder automatically set `PYTORCH_NO_NVML=1`, which tells PyTorch to skip NVML entirely. This avoids the `nvmlInit_v2` crash on hosts where NVML is unavailable (e.g., some WSL2 setups).

#### Option 2: Manual GPU Diagnostics
```bash
nvidia-smi                 # Check GPU utilization & NVML availability
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
./run_experiment.sh --device cuda
```
Update GPU drivers if `nvidia-smi` fails.

#### Option 3: CPU Execution (No Longer Supported)
Automatic CPU fallback was removed. If CUDA is unavailable or misconfigured the encoder aborts immediately:
```text
RuntimeError: CUDA device unavailable during embedding generation.
```
Fix the GPU environment (drivers/NVML) before rerunning; CPU-only runs are not supported for Qodo embeddings.

---

### 2. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

#### Reduce Batch Size
Edit `configs/experiment.yaml`:
```yaml
training:
  batch_size: 16  # Default: 32, try 16 or 8
```

#### Reduce Max Length
Edit `configs/experiment.yaml`:
```yaml
semantic:
  max_length: 256  # Default: 512
```

#### Use CPU for Encoding
Force CPU for the memory-intensive encoding step:
```bash
./run_experiment.sh --device cpu
```

---

### 3. Model Download Fails

**Symptoms:**
```
OSError: Can't load model from Qodo/Qodo-Embed-1-1.5B
Connection timeout
```

**Solutions:**

#### Check Internet Connection
```bash
ping huggingface.co
```

#### Manual Download
```bash
# Download model manually
git lfs install
git clone https://huggingface.co/Qodo/Qodo-Embed-1-1.5B models/Qodo-Embed-1-1.5B

# Update config to use local path
vim configs/experiment.yaml
# Change: model_name: "models/Qodo-Embed-1-1.5B"
```

#### Use Cached Model
If model was downloaded before:
```bash
ls ~/.cache/huggingface/hub/
# Model should be there
```

---

### 4. Dependencies Missing

**Symptoms:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
ImportError: cannot import name 'GATConv'
```

**Solutions:**

#### Re-run Setup
```bash
./setup_experiment.sh
```

#### Manual Installation
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### Check Installation
```bash
source venv/bin/activate
python -c "import sentence_transformers; print('OK')"
python -c "import torch_geometric; print('OK')"
```

---

### 5. Virtual Environment Issues

**Symptoms:**
```
externally-managed-environment
pip: command not found
```

**Solutions:**

#### Use venv
```bash
# Delete old venv
rm -rf venv

# Re-run setup
./setup_experiment.sh
```

#### Verify Activation
```bash
source venv/bin/activate
which python  # Should point to venv/bin/python
```

---

### 6. Dataset Not Found

**Symptoms:**
```
FileNotFoundError: datasets/train.csv
```

**Solutions:**

#### Check Dataset Location
```bash
ls datasets/
# Should show: train.csv, test.csv
```

#### Verify Path in Config
```yaml
data:
  train_path: "datasets/train.csv"
  test_path: "datasets/test.csv"
```

---

### 7. Experiment Number Conflicts

**Symptoms:**
```
Error: results/experiment_001 already exists
```

**Solutions:**

#### Auto-Increment Works
The script automatically finds the next number:
```bash
./run_experiment.sh  # Creates experiment_002 if 001 exists
```

#### Manual Cleanup
```bash
# Archive old experiments
mkdir -p archive/old_results
mv results/experiment_0* archive/old_results/
```

---

### 8. Slow Training

**Symptoms:**
- Training takes hours
- Each epoch > 10 minutes

**Solutions:**

#### Test on Sample
```bash
./run_experiment.sh --sample 1000  # Use 1000 samples only
```

#### Use GPU
```bash
# Check GPU usage
nvidia-smi

# Ensure using CUDA
./run_experiment.sh --device cuda
```

#### Reduce Dataset Size
Edit config:
```yaml
data:
  train_split: 0.1  # Use 10% of data for testing
```

---

### 9. Config Errors

**Symptoms:**
```
KeyError: 'semantic'
yaml.scanner.ScannerError
```

**Solutions:**

#### Validate YAML
```bash
python -c "import yaml; yaml.safe_load(open('configs/experiment.yaml'))"
```

#### Use Default Config
```bash
cp archive/configs/experiment_v9_qodo.yaml configs/experiment.yaml
```

#### Check Indentation
YAML is indentation-sensitive:
```yaml
# WRONG
semantic:
model_name: "Qodo/Qodo-Embed-1-1.5B"

# CORRECT
semantic:
  model_name: "Qodo/Qodo-Embed-1-1.5B"
```

---

### 10. Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
bash: ./run_experiment.sh: Permission denied
```

**Solutions:**

#### Make Executable
```bash
chmod +x setup_experiment.sh run_experiment.sh
```

#### Check File Ownership
```bash
ls -la *.sh
# Should be owned by you
```

---

## Performance Tips

### Faster Encoding
```yaml
semantic:
  batch_size: 64  # Increase from 32 (if memory allows)
  max_length: 256  # Reduce from 512
```

### Faster Training
```yaml
training:
  num_epochs: 20  # Reduce from 50 for testing
  batch_size: 64  # Increase if memory allows
```

### Use Mixed Precision
```yaml
training:
  use_amp: true  # Automatic Mixed Precision
```

---

## Debug Mode

### Verbose Logging
```bash
# Edit main.py or add to config
logging:
  level: "DEBUG"  # Instead of "INFO"
```

### Sample Run
```bash
# Test with small sample
./run_experiment.sh --sample 100
```

### Check Intermediate Outputs
```bash
# Cache should contain embeddings
ls -lh cache/embeddings_qodo/

# Results should have logs
cat results/experiment_XXX/output.log | grep ERROR
```

---

## Getting Help

If none of these solutions work:

1. **Check Logs:**
   ```bash
   cat results/experiment_XXX/output.log | grep -i error
   ```

2. **Check System:**
   ```bash
   nvidia-smi  # GPU status
   free -h     # RAM usage
   df -h       # Disk space
   ```

3. **Provide Info:**
   - Python version: `python --version`
   - CUDA version: `nvidia-smi`
   - Error message
   - Config file
   - Last 50 lines of log

4. **Contact:**
   - Open GitHub issue
   - Include: error, config, system info

---

**Last Updated:** 2025-11-10
