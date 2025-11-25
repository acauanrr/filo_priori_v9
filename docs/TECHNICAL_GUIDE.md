# Filo-Priori Technical Guide

Complete technical documentation for the Filo-Priori system.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Embedding System](#embedding-system)
3. [Caching Mechanism](#caching-mechanism)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Metrics](#evaluation-metrics)

---

## System Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Loading                            │
│  (train.csv, test.csv) → pandas DataFrames                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              Embedding Manager (with Cache)                 │
│  • Checks cache validity                                     │
│  • Loads from cache OR generates new embeddings             │
│  • Saves to cache for future use                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
      ↓               ↓               ↓
┌──────────┐  ┌──────────────┐  ┌──────────┐
│   TC     │  │   Commit     │  │Structural│
│Embeddings│  │  Embeddings  │  │ Features │
│ (768)    │  │    (768)     │  │   (6)    │
└────┬─────┘  └──────┬───────┘  └────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │   Dual-Stream Model   │
         │  ┌─────────────────┐  │
         │  │ Semantic Branch │  │
         │  │   (256-dim)     │  │
         │  └─────────────────┘  │
         │  ┌─────────────────┐  │
         │  │Structural Branch│  │
         │  │   (64-dim)      │  │
         │  └─────────────────┘  │
         │  ┌─────────────────┐  │
         │  │  Graph Branch   │  │
         │  │  (128-dim GAT)  │  │
         │  └─────────────────┘  │
         └───────────┬───────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │    Fusion Layer       │
         │      (256-dim)        │
         └───────────┬───────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │     Classifier        │
         │  (Binary: Pass/Fail)  │
         └───────────┬───────────┘
                     │
                     ↓
         ┌───────────────────────┐
         │  Failure Probability  │
         │  → Test Case Ranking  │
         └───────────────────────┘
```

---

## Embedding System

### SBERT Encoder (all-mpnet-base-v2)

**Model Specifications:**
- Parameters: 110M
- Embedding Dimension: 768
- Max Sequence Length: 384 tokens (recommended), 512 max
- VRAM Usage: ~200MB
- Architecture: MPNet-base (12 layers, 768 hidden size)

**Why SBERT over Qodo?**

| Aspect | SBERT | Qodo-Embed-1-1.5B |
|--------|-------|-------------------|
| Size | 110M params | 1.5B params |
| VRAM | ~200MB | ~3GB |
| Speed | 4,628/s | ~100/s |
| Optimized For | **General text** | Code retrieval |
| Our Data Type | **Test descriptions** | Code snippets |
| Stability | ✅ Perfect | ❌ NVML errors |

**Key Insight:** Test case descriptions and commit messages are natural language text, not code. SBERT is the optimal choice.

### Text Preparation

**Test Cases:**
```python
def prepare_tc_text(row):
    summary = row['tc_summary']
    steps = row['tc_steps']
    return f"Summary: {summary}\nSteps: {steps}"
```

**Commits:**
```python
def prepare_commit_text(row):
    msg = row['commit_msg']
    diff = row['commit_diff'][:2000]  # Truncate to fit in context
    return f"Commit Message: {msg}\n\nDiff:\n{diff}"
```

### Encoding Process

```python
# Initialize encoder
encoder = SBERTEncoder(config, device='cuda')

# Encode in chunks for stability
embeddings = encoder.encode_texts_chunked(
    texts,
    chunk_size=1280,  # 10 batches
    desc="Encoding TCs"
)
# Output: numpy array of shape (N, 768)
```

---

## Caching Mechanism

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EmbeddingCache                        │
│                                                         │
│  cache/                                                 │
│  ├── embeddings.npz          ← Compressed embeddings   │
│  └── embeddings_metadata.txt ← Hash + metadata         │
│                                                         │
│  Functions:                                             │
│  • exists() → bool                                      │
│  • is_valid(train_df, test_df) → bool                  │
│  • load() → (embeddings, dim, model)                   │
│  • save(embeddings, metadata)                          │
│  • clear()                                             │
└─────────────────────────────────────────────────────────┘
```

### Cache Validation

The cache uses **MD5 hashing** to detect data changes:

```python
def _compute_data_hash(train_df, test_df):
    # Hash based on:
    # 1. Data shape (num rows)
    # 2. First row content
    # 3. Last row content
    hash_input = f"{len(train_df)}_{len(test_df)}"
    hash_input += str(train_df.iloc[0].to_dict())
    hash_input += str(train_df.iloc[-1].to_dict())
    return hashlib.md5(hash_input.encode()).hexdigest()
```

**When cache is invalidated:**
- Data shape changed (rows added/removed)
- First or last row content changed
- Cache file missing or corrupted

### EmbeddingManager Workflow

```python
def get_embeddings(train_df, test_df):
    # 1. Check cache
    if force_regenerate:
        → Generate new embeddings
    elif cache.exists() and cache.is_valid(train_df, test_df):
        → Load from cache
    else:
        → Generate new embeddings

    # 2. Save if newly generated
    if generated:
        cache.save(embeddings, metadata)

    # 3. Return embeddings
    return {
        'train_tc': train_tc_emb,
        'test_tc': test_tc_emb,
        'train_commit': train_commit_emb,
        'test_commit': test_commit_emb,
        'embedding_dim': 768,
        'model_name': 'all-mpnet-base-v2'
    }
```

---

## Model Architecture

### Dual-Stream Network

**Input:**
- Semantic: TC embeddings (768) + Commit embeddings (768) = 1536
- Structural: 6 features (pass/fail rates, temporal info)
- Graph: Adjacency matrix from phylogenetic graph

**Semantic Branch:**
```python
Input: (1536-dim)
   ↓
[Linear + GELU + Dropout]
   ↓
[Linear + GELU + Dropout]
   ↓
Output: (256-dim)
```

**Structural Branch:**
```python
Input: (6-dim)
   ↓
[Linear + GELU + Dropout]
   ↓
[Linear + GELU + Dropout]
   ↓
Output: (64-dim)
```

**Graph Branch (GAT):**
```python
Input: Node features + Adjacency
   ↓
[GAT Layer 1] (4 heads, 128-dim)
   ↓
[ELU + Dropout]
   ↓
[GAT Layer 2] (4 heads, 128-dim)
   ↓
Output: (128-dim)
```

**Fusion + Classification:**
```python
Concat: (256 + 64 + 128 = 448-dim)
   ↓
[Linear + GELU + Dropout] (256-dim)
   ↓
[Linear + GELU + Dropout] (128-dim)
   ↓
[Linear] (2 classes: Pass/Fail)
   ↓
Softmax → Probabilities
```

---

## Training Pipeline

### Data Flow

```
1. Load Data
   ├── train.csv (69,169 samples)
   └── test.csv (31,333 samples)

2. Split Train → Train/Val (80%/20%)

3. Generate/Load Embeddings
   ├── Check cache
   ├── Load OR generate
   └── Save to cache

4. Create DataLoaders
   ├── Batch size: 32
   ├── Shuffle: True (train only)
   └── Pin memory: True (if CUDA)

5. Training Loop (up to 50 epochs)
   ├── Forward pass
   ├── Compute loss (Focal Loss)
   ├── Backward pass
   ├── Optimizer step (AdamW)
   ├── LR scheduling (Cosine)
   └── Early stopping (patience=12)

6. Evaluation
   ├── Validation metrics (every epoch)
   ├── Best model selection (F1 Macro)
   └── Final test evaluation

7. Save Results
   ├── Model checkpoint
   ├── Metrics JSON
   ├── Predictions CSV
   ├── Rankings CSV
   └── Plots (confusion matrix, PR curves)
```

### Loss Function

**Focal Loss** for handling class imbalance:

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
  α = 0.25  (class weight)
  γ = 2.0   (focusing parameter)
```

Benefits:
- Down-weights easy examples
- Focuses on hard, misclassified examples
- Better than Cross-Entropy for imbalanced data

---

## Evaluation Metrics

### Classification Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision & Recall:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**F1 Scores:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

F1 Macro = average(F1_Pass, F1_Fail)
F1 Weighted = weighted_average by class support
```

**AUPRC (Area Under Precision-Recall Curve):**
- More informative than AUC-ROC for imbalanced data
- Macro: average across classes

### Ranking Metrics

**APFD (Average Percentage of Faults Detected):**

```
APFD = 1 - (TF₁ + TF₂ + ... + TFₘ) / (n × m) + 1/(2n)

where:
  n = total number of test cases
  m = number of faults
  TFᵢ = position of first test case that detects fault i
```

Interpretation:
- APFD = 1.0: Perfect (all faults detected first)
- APFD = 0.5: Random ordering
- APFD > 0.7: Good prioritization

---

## Configuration Reference

### Key Configuration Parameters

**Embeddings:**
```yaml
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  batch_size: 128      # Larger = faster but more VRAM
  use_cache: true      # Enable caching
  device: "cuda"       # or "cpu"
```

**Model:**
```yaml
model:
  semantic:
    hidden_dim: 256    # Semantic representation size
    num_layers: 2      # Depth
    dropout: 0.15      # Regularization
```

**Training:**
```yaml
training:
  epochs: 50
  learning_rate: 5e-5
  weight_decay: 1e-4
  early_stopping:
    patience: 12       # Stop if no improvement
    monitor: "val_f1_macro"
```

---

## Performance Optimization

### GPU Memory

**Reduce batch size if OOM:**
```yaml
embedding:
  batch_size: 64  # or 32
training:
  batch_size: 16  # or 8
```

**Use gradient accumulation:**
```yaml
training:
  batch_size: 8
  accumulation_steps: 4  # Effective batch = 32
```

### Speed

**Increase batch size if memory allows:**
```yaml
embedding:
  batch_size: 256  # 2x faster
training:
  batch_size: 64   # Faster convergence
```

**Disable unnecessary features:**
```yaml
graph:
  build_graph: false  # Skip if not needed
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch_size in config
```

**2. Slow Embedding Generation**
```
Check:
- Using GPU? (device: "cuda")
- Batch size too small? (increase to 128)
- Cache enabled? (use_cache: true)
```

**3. Cache Not Being Used**
```
Check:
- Data changed? (will auto-regenerate)
- Cache corrupted? (use --clear-cache)
- force_regenerate enabled? (remove flag)
```

**4. Poor Model Performance**
```
Check:
- Sufficient training epochs? (50+)
- Appropriate learning rate? (5e-5)
- Early stopping too aggressive? (increase patience)
```

---

## API Reference

### EmbeddingManager

```python
from src.embeddings import EmbeddingManager

manager = EmbeddingManager(
    config: Dict,
    force_regenerate: bool = False,
    cache_dir: str = 'cache'
)

# Get embeddings (auto-cached)
embeddings = manager.get_embeddings(train_df, test_df)

# Clear cache
manager.clear_cache()

# Get cache info
info = manager.cache_info()
```

### SBERTEncoder

```python
from src.embeddings import SBERTEncoder

encoder = SBERTEncoder(config, device='cuda')

# Encode batch
embeddings = encoder.encode_texts_batch(texts)

# Encode with chunking
embeddings = encoder.encode_texts_chunked(
    texts,
    chunk_size=1280,
    desc="Processing"
)

# Get embedding dimension
dim = encoder.get_embedding_dim()  # 768
```

---

## Development

### Adding New Features

**1. Custom Embedding Model:**
```python
# src/embeddings/custom_encoder.py
class CustomEncoder:
    def __init__(self, config, device):
        # Load your model
        pass

    def encode_texts_batch(self, texts):
        # Return numpy array (N, embedding_dim)
        pass
```

**2. Custom Metrics:**
```python
# src/utils/metrics.py
def custom_metric(y_true, y_pred):
    # Compute your metric
    return score
```

**3. Custom Loss:**
```python
# src/model/losses.py
class CustomLoss(nn.Module):
    def forward(self, outputs, targets):
        # Compute loss
        return loss
```

---

## References

- [SBERT Paper](https://arxiv.org/abs/1908.10084)
- [GAT Paper](https://arxiv.org/abs/1710.10903)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)

---

**Last Updated:** 2024-11-14
