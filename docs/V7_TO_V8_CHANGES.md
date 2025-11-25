# Filo-Priori V7 → V8: Major Architectural Changes

**Date:** 2025-11-06
**Status:** 🚧 In Progress (Step 2.1 Complete)

---

## Critical Issue Identified in V7

### The "Semantic Echo Chamber" Problem

**V7 Architecture Flaw:**
```
Both Semantic and Structural streams used the same BGE embeddings as input:

Semantic Stream:  BGE embeddings → FFN
Structural Stream: BGE embeddings → k-NN Graph → GNN

Problem: The "structural" stream was just processing a topological view
of semantic similarity, NOT true structural information.
```

**Impact on Thesis:**
- Cannot validate hypothesis that "semantic + structural fusion improves performance"
- Both streams see the same information (text-based)
- Thesis reviewers would correctly reject the premise

---

## V8 Solution: True Structural Features

### New Architecture

```
Semantic Stream:  BGE embeddings [1024] → FFN → features [256]
                  ↓
                  Text-based semantic information

Structural Stream: Historical features [6] → Projection → features [256]
                   ↓
                   True structural information:
                   - test_age
                   - failure_rate
                   - recent_failure_rate
                   - flakiness_rate
                   - commit_count
                   - test_novelty
```

### Key Differences

| Aspect | V7 | V8 |
|--------|----|----|
| **Semantic Stream Input** | BGE embeddings [1024] | BGE embeddings [1024] ✓ Same |
| **Structural Stream Input** | BGE embeddings [1024] | Historical features [6] ✅ NEW |
| **Graph Construction** | k-NN from BGE embeddings | No graph needed ✅ Simplified |
| **Information Orthogonality** | ❌ Both streams semantic | ✅ Truly orthogonal |
| **Thesis Validation** | ❌ Cannot validate | ✅ Can validate |

---

## Implementation Progress

### ✅ Phase 1: Structural Feature Extraction (Step 2.1) - COMPLETE

**Implemented:**
1. `src/preprocessing/structural_feature_extractor.py` (576 lines)
   - Extracts 6 structural/phylogenetic features
   - Learns from training history
   - No data leakage

2. `scripts/validate_structural_features.py` (391 lines)
   - Comprehensive validation
   - Multiple sanity checks
   - Example outputs

**Validation Results:**
- ✅ All features in expected ranges
- ✅ Statistics make sense (20k sample test)
- ✅ No data leakage confirmed

**Documentation:**
- `docs/V8_IMPLEMENTATION_STEP_2.1_COMPLETE.md` - Full report

---

### 🚧 Phase 2: Model Architecture Updates (Step 2.2) - PENDING

**To Be Implemented:**

1. **Modify Structural Stream** (`src/models/dual_stream.py`)
   ```python
   # Current V7
   class StructuralStream(nn.Module):
       def __init__(self, input_dim=1024, ...):  # BGE dim
           self.projection = nn.Linear(1024, 256)
           self.message_passing = ...

   # New V8
   class StructuralStream(nn.Module):
       def __init__(self, input_dim=6, ...):  # Structural features
           self.projection = nn.Linear(6, 256)
           self.ffn = ...  # No message passing needed
   ```

2. **Update Data Pipeline** (`main.py`)
   - Extract structural features
   - Pass to model alongside embeddings

3. **Remove Graph Construction**
   - Structural stream no longer needs k-NN graph
   - Simplifies pipeline

---

### 📋 Phase 3: List-Wise Ranking (Step 2.3+) - FUTURE

After fixing the architecture, implement Lambda-APFD:
- Convert Phase 1 from classifier to feature extractor
- Add Phase 2 list-wise ranker
- Optimize APFD directly

---

## Breaking Changes

### Code That Will Change

1. **Model initialization** (`main.py`)
   ```python
   # OLD V7
   model = DualStreamModel(
       semantic_input_dim=1024,
       structural_input_dim=1024,  # Same as semantic!
       ...
   )

   # NEW V8
   from src.preprocessing.structural_feature_extractor import extract_structural_features

   # Extract structural features
   train_struct, val_struct, test_struct = extract_structural_features(...)

   model = DualStreamModel(
       semantic_input_dim=1024,
       structural_input_dim=6,  # Different!
       ...
   )
   ```

2. **Training loop** (`src/training/trainer.py`)
   ```python
   # OLD V7
   logits = model(embeddings, edge_index, edge_weights)

   # NEW V8
   logits = model(
       semantic_input=embeddings,
       structural_input=struct_features
   )
   ```

3. **Graph construction** (`src/phylogenetic/tree_builder.py`)
   - Still needed for semantic stream (optional)
   - NOT needed for structural stream anymore

### Config Changes

```yaml
# OLD V7 config
model:
  semantic_input_dim: 1024
  structural_input_dim: 1024  # Same!
  use_graph_rewiring: true

# NEW V8 config
model:
  semantic_input_dim: 1024
  structural_input_dim: 6  # Different!
  structural_features:
    - test_age
    - failure_rate
    - recent_failure_rate
    - flakiness_rate
    - commit_count
    - test_novelty
  use_graph_rewiring: false  # Not needed for structural stream
```

---

## Migration Path

### For Existing V7 Users

1. **Keep V7 code for comparison**
   ```bash
   # V7 will remain as baseline
   git tag v7-final
   ```

2. **Install V8 alongside V7**
   ```bash
   cd ..
   git clone filo_priori_v7 filo_priori_v8
   cd filo_priori_v8
   # Apply V8 changes
   ```

3. **Run experiments side-by-side**
   - V7: Semantic echo chamber (baseline)
   - V8: True structural features (hypothesis test)

4. **Compare results**
   - If V8 ≤ V7: Structural features don't help (negative result, still publishable)
   - If V8 > V7: Validates thesis hypothesis ✅

---

## Expected Improvements

### Hypothesis

Adding true structural features will improve:
1. **Classification**: Better failure prediction (F1, AUPRC)
2. **Prioritization**: Better APFD scores
3. **Interpretability**: Clear feature importance

### Metrics to Track

| Metric | V7 Baseline | V8 Target | Status |
|--------|-------------|-----------|--------|
| Test F1 Macro | 0.50-0.55 | ≥0.55 | 🔄 TBD |
| Test Accuracy | 60-65% | ≥65% | 🔄 TBD |
| Mean APFD | 0.597 | ≥0.60 | 🔄 TBD |
| Prediction Diversity | 0.30-0.40 | ≥0.40 | 🔄 TBD |

---

## Scientific Value

### Why This Matters

1. **Thesis Validity**: Can now properly test the core hypothesis

2. **Reproducibility**: Clear separation of information sources

3. **Interpretability**: Can attribute improvements to specific features
   - *"How much does test_age contribute vs flakiness_rate?"*

4. **Literature Alignment**: Matches best practices in TCP research

---

## Timeline

| Phase | Status | Completion |
|-------|--------|------------|
| 2.1: Structural Feature Extraction | ✅ Complete | 2025-11-06 |
| 2.2: Model Architecture Update | 🚧 In Progress | TBD |
| 2.3: Integration Testing | 📋 Planned | TBD |
| 2.4: Comparative Evaluation | 📋 Planned | TBD |
| 3.0: List-Wise Ranking | 📋 Future | TBD |

---

## Quick Start (Current V8 State)

```bash
# Setup
cd filo_priori_v8
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy

# Validate structural features
python scripts/validate_structural_features.py --sample-size 10000

# Expected output
# ✓ All features within expected ranges
# ✓ All validations passed
```

---

## Questions & Answers

**Q: Can I still use V7?**
A: Yes! V7 remains as a baseline for comparison.

**Q: Will V8 be slower?**
A: No. Removing graph construction from structural stream actually speeds up training.

**Q: Do I need to retrain from scratch?**
A: Yes. The model architecture changes require retraining.

**Q: What if V8 doesn't improve results?**
A: Still valuable! Proves that structural info doesn't help, which is a valid scientific result.

---

## References

- **Original Plan**: `docs/V8_ACTION_PLAN.md` (full strategic analysis)
- **Step 2.1 Report**: `docs/V8_IMPLEMENTATION_STEP_2.1_COMPLETE.md`
- **Validation Script**: `scripts/validate_structural_features.py`
- **New Module**: `src/preprocessing/structural_feature_extractor.py`

---

**Current Status**: ✅ **Step 2.1 Complete, Ready for Step 2.2**

**Next Action**: Modify Structural Stream to accept 6D input instead of 1024D
