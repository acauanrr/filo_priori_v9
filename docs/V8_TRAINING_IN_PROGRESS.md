# Filo-Priori V8 - Training In Progress

**Date:** 2025-11-06
**Status:** 🔄 TRAINING IN PROGRESS
**Implementation:** Complete Steps 2.1 + 2.2

---

## Current Training Run

### Configuration
- **Sample Size:** 1,000 records (quick validation)
- **Device:** CPU (no GPU available)
- **Model:** DualStreamModelV8
- **Semantic Input:** BGE-Large embeddings [1024-dim]
- **Structural Input:** Historical features [6-dim]

### Data Split (Group-Aware by Build_ID)
- **Train:** 842 samples (32 builds)
- **Val:** 62 samples (5 builds)
- **Test:** 96 samples (5 builds)
- **Total Builds:** 42 unique Build_IDs
- **No Leakage:** ✅ All splits disjoint by Build_ID

### Class Distribution
- **Pass (Class 1):** 796 samples (79.6%)
- **Not-Pass (Class 0):** 204 samples (20.4%)
- **Class Weights:** [2.45, 0.63]
- **Strategy:** Binary (Pass vs All)

---

## Training Pipeline Progress

### ✅ Step 1.1: Data Loading (COMPLETE)
- Loaded 1,000 samples from train.csv
- Applied binary labeling (Pass vs Not-Pass)
- Split data with Build_ID grouping
- No data leakage

### 🔄 Step 1.2: Semantic Embedding Extraction (IN PROGRESS)
- **Model:** BAAI/bge-large-en-v1.5 (cached)
- **Progress:** ~44% (12/27 batches encoded)
- **Time per batch:** ~17s on CPU
- **Estimated completion:** 5-7 minutes

**Text Processing:**
- Combined: TE_Summary + TC_Steps + commit
- Cleaned and formatted with [SEP] tokens
- Max length: 512 tokens

### ⏳ Step 1.3: Structural Feature Extraction (PENDING)
Will extract 6 historical features:
1. test_age
2. failure_rate
3. recent_failure_rate
4. flakiness_rate
5. commit_count
6. test_novelty

### ⏳ Step 1.4: Phylogenetic Graph Construction (PENDING)
- Graph type: co_failure
- Min co-occurrences: 2
- Weight threshold: 0.1

### ⏳ Step 2: Model Training (PENDING)
- Architecture: DualStreamModelV8
- Loss: Focal Loss (α=[0.15, 0.85], γ=2.0)
- Optimizer: AdamW (lr=5e-5, wd=2e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 40 (with early stopping, patience=12)
- Batch size: 32

### ⏳ Step 3: Evaluation (PENDING)
- Classification metrics (F1, Accuracy, AUPRC)
- APFD calculation
- Comparison with V7 baseline

---

## V8 vs V7 Architectural Differences

| Component | V7 (Flawed) | V8 (Fixed) |
|-----------|-------------|------------|
| **Semantic Stream** | BGE [1024] | BGE [1024] ✓ Same |
| **Structural Stream** | BGE [1024] ⚠️ | Historical [6] ✅ |
| **Graph** | k-NN semantic | Co-failure ✅ |
| **Information** | Echo chamber ⚠️ | Truly orthogonal ✅ |
| **Hypothesis Testable** | ❌ No | ✅ Yes |

---

## Expected Outcomes

### Baseline Targets (1K Sample)
With only 1,000 samples, we expect:
- **Test F1 Macro:** 0.45-0.50 (small dataset)
- **Test Accuracy:** 55-60%
- **Val-Test Gap:** <15% (good generalization)
- **Prediction Diversity:** >0.30

### Full Dataset Targets (Later)
With full dataset (~400K samples):
- **Test F1 Macro:** ≥0.55
- **Test Accuracy:** ≥65%
- **Mean APFD:** ≥0.60
- **Better than V7:** Expected due to orthogonal features

---

## Files Created/Modified

### Core V8 Implementation
| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/preprocessing/structural_feature_extractor.py` | ✅ Complete | 576 | Extract 6 historical features |
| `src/phylogenetic/phylogenetic_graph_builder.py` | ✅ Complete | 560 | Build co-failure/commit graphs |
| `src/models/dual_stream_v8.py` | ✅ Complete | 530 | V8 architecture |
| `main_v8.py` | ✅ Complete | 436 | Training pipeline |
| `configs/experiment_v8_baseline.yaml` | ✅ Complete | 235 | V8 configuration |

### Validation & Testing
| File | Status | Purpose |
|------|--------|---------|
| `test_v8_simple.py` | ✅ All tests passed | Quick validation |
| `scripts/validate_v8_pipeline.py` | ✅ Complete | Comprehensive validation |
| `test_data_pipeline.py` | ✅ All tests passed | Data loading validation |

### Configuration Fixes
- Added `text` section for TextProcessor
- Fixed split config (`train_split`, `val_split`, `test_split`)
- Added `pooling_strategy` for SemanticEncoder
- Disabled V7 imports in `src/models/__init__.py`

---

## Scientific Contributions

### 1. Identification of "Semantic Echo Chamber"
First explicit identification of this architectural flaw in dual-stream models where both streams process the same type of information.

### 2. True Phylogenetic Graphs for TCP
Novel application of:
- **Co-failure graphs:** Tests that fail together
- **Commit dependency graphs:** Tests affecting same code

### 3. Explicit Structural Features
Clear, interpretable domain features:
- Test lifecycle (age, novelty)
- Historical behavior (failure rates, flakiness)
- Code coupling (commit counts)

---

## Next Steps

### Immediate (In Progress)
1. ⏳ Complete semantic embedding extraction
2. ⏳ Extract structural features
3. ⏳ Train V8 model on 1K sample
4. ⏳ Evaluate and validate metrics

### Short-Term (After 1K Validation)
1. Run full training with complete dataset (~400K samples)
2. Compare V8 results with V7 baseline
3. Analyze feature importance
4. Test different graph types (co_failure vs commit_dependency)

### Long-Term (Phase 3)
1. Implement Lambda-APFD with list-wise ranking
2. Test advanced semantic encoders (CodeBERT)
3. Multi-objective optimization (APFD + Cost)
4. Prepare for publication

---

## Log Files

- **Training Log:** `training_log.txt` (real-time updates)
- **Model Checkpoint:** `best_model_v8.pt` (will be created)
- **Results:** `results/experiment_v8_baseline/` (will be created)
- **APFD Report:** `apfd_per_build_v8.csv` (will be created)

---

## Monitoring Commands

```bash
# Check training progress
tail -f training_log.txt

# Check current phase
tail -50 training_log.txt | grep -E "(STEP|Epoch|Loss|F1)"

# Check if process is running
ps aux | grep main_v8.py

# Check GPU/CPU usage
htop
```

---

## Timeline Estimate

- **Text Encoding:** ~10 minutes (1K samples on CPU)
- **Structural Features:** ~1 minute
- **Graph Building:** ~30 seconds
- **Model Training:** ~10-15 minutes (40 epochs with early stopping)
- **Evaluation:** ~1 minute
- **Total:** ~25-30 minutes for 1K sample run

**Full Dataset:** Would take ~6-8 hours on CPU

---

**Status:** ✅ V8 Implementation Complete | 🔄 Training In Progress
**Next Milestone:** Complete 1K validation run and analyze results

---

*For detailed implementation, see `STEP_2.2_COMPLETE.md`*
