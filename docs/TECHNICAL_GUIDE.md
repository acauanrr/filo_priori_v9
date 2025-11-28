# Filo-Priori V9 Technical Guide

Complete technical documentation for the Filo-Priori V9 system.

**Paradigm:** Bio-inspired Phylogenetic Approach to Test Case Prioritization

**Last Updated:** November 2025

---

## Table of Contents

1. [Phylogenetic Approach](#phylogenetic-approach)
2. [System Architecture](#system-architecture)
3. [Embedding System](#embedding-system)
4. [Caching Mechanism](#caching-mechanism)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Loss Functions](#loss-functions)
9. [Graph Construction](#graph-construction)

---

## Phylogenetic Approach

### Conceptual Foundation

Filo-Priori treats software evolution as a **phylogenetic tree**, drawing from computational biology. This paradigm shift provides a principled framework for modeling software history.

### Mapping: Biology → Software Engineering

| Biological Concept | Software Equivalent | Implementation |
|--------------------|---------------------|----------------|
| Taxon/Species | Commit/Version | Node in Git DAG |
| DNA Sequence | Source Code / AST | Code embeddings |
| Mutation (SNP) | Code Diff | Edge weights |
| Phylogenetic Tree | Git DAG | Graph structure |
| Phylogenetic Signal | Failure Autocorrelation | Attention weights |
| Common Ancestor | Merge Base | Synchronization point |

### Phylogenetic Distance Kernel

The evolutionary distance between commits is computed as:

```
d_phylo(c_i, c_j) = shortest_path(c_i, c_j) × β^(n_merges)
```

Where:
- `shortest_path(c_i, c_j)`: Shortest path in the Git DAG
- `n_merges`: Number of merge commits on the path
- `β = 0.9`: Decay factor (merges "reset" divergence)

### Key Insight

The phylogenetic metaphor provides:
1. **Mathematical foundations** (distance kernels, signal propagation)
2. **Inductive bias** (evolutionary proximity ⟹ behavioral similarity)
3. **Principled regularization** (predictions should be consistent with evolution)

---

## System Architecture

### Overview: Phylogenetic Neural Architecture

```
                    FILO-PRIORI: PHYLOGENETIC ARCHITECTURE
    =====================================================================

    INPUTS                        ENCODERS                      OUTPUT
    ------                        --------                      ------

    [Git DAG]                +------------------+
    Commits +                |  PHYLO-ENCODER   |
    Branches +  -----------> |  (GGNN Temporal) | ----+
    Merges                   |  Phylo Distance  |     |
                             +------------------+     |
                                                      |     +---------------+
    [Source Code]            +------------------+     +---> | CROSS-        |
    AST/CFG +                |  CODE-ENCODER    |     |     | ATTENTION     |
    Diffs +     -----------> |  (GATv2+CodeBERT)| ----+     | FUSION        |
    Coverage                 |  Semantic Embed. |           |               |
                             +------------------+           +-------+-------+
                                                                    |
    [Test History]           +------------------+                   v
    Pass/Fail +              | HIERARCHICAL     |           +---------------+
    Flakiness + -----------> | ATTENTION        | ----+     | RANKING       |
    Exec Time                | (Micro/Meso/Macro|     +---> | MODULE        |
                             +------------------+           | P(fail|test)  |
                                                            +---------------+
                                                                    |
                                                                    v
                                                            [Prioritized List]
                                                            T' = {t1, t2, ..., tn}
```

### Module Details

**1. Phylo-Encoder (GGNN Temporal)**
- Processes Git DAG as phylogenetic tree
- Computes evolutionary distances between commits
- Propagates failure signals weighted by phylogenetic distance

**2. Code-Encoder (GATv2 + CodeBERT)**
- Semantic embeddings via CodeBERT/SBERT
- Multi-edge graph (co-failure, co-success, semantic)
- Dynamic attention (GATv2) over test relationships

**3. Hierarchical Attention**
- Micro: Token-level attention in code
- Meso: Method/call-graph attention
- Macro: Commit history attention

**4. Ranking Module**
- Cross-attention fusion of all representations
- Combined loss: Focal + RankNet + Phylo-Regularization
- Outputs failure probability for ranking

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

**Structural Stream (with GATv2):**
```python
Input: (10-dim features) + Phylogenetic Graph
   ↓
[Linear Projection: 10 → 128]
   ↓
[GATv2 Layer] (2 heads, 64-dim per head)
   ↓
[LeakyReLU + Dropout]
   ↓
Output: (128-dim → projected to 64-dim)
```

**GATv2 vs Standard GAT:**
- **GAT:** attention = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))
- **GATv2:** attention = softmax(a^T * LeakyReLU(W*[h_i || h_j]))
- GATv2 applies LeakyReLU AFTER projection (Brody et al., 2022)
- More expressive, feature-dependent attention

**Fusion + Classification:**
```python
Concat: (256 + 64 = 320-dim)
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

## Loss Functions

### Combined Loss (Phylogenetic)

The V9 system uses a three-component loss function:

```python
Total Loss = λ₁ × Focal Loss + λ₂ × Ranking Loss + λ₃ × Phylo Regularization

Where: λ₁ = 0.6, λ₂ = 0.3, λ₃ = 0.1
```

### 1. Weighted Focal Loss (Classification)

**Purpose:** Handle extreme class imbalance (37:1 Pass:Fail ratio)

**Formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

**Parameters:**
- α = [0.15, 0.85] (class weights: low for Pass, high for Fail)
- γ = 2.5 (focusing parameter)

**Benefits:**
- Down-weights easy examples (well-classified samples)
- Focuses learning on hard, misclassified examples
- Superior to standard Cross-Entropy for imbalanced data

**Reference:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

### 2. Ranking Loss (RankNet-style)

**Purpose:** Align training objective with APFD evaluation metric

**Formula:**
```
L_rank = log(1 + exp(-(s_fail - s_pass - margin)))
```

**Key Features:**
- Creates (Fail, Pass) pairs within same Build_ID
- Uses logits for stability (not probabilities)
- Margin: 0.5 (minimum separation between classes)

**Hard Negative Mining:**
- Selects top-5 hardest Pass examples per build
- 20% hard negative sampling ratio
- Maximum 50 pairs per build
- Starts at epoch 3 (after warmup)

**Reference:** Burges et al., "Learning to Rank using Gradient Descent" (ICML 2005)

### 3. Phylogenetic Regularization

**Purpose:** Encourage predictions consistent with evolutionary structure

**Formula:**
```
L_phylo = Σ w_phylo(c_i, c_j) × |p(c_i) - p(c_j)|

Where:
- (c_i, c_j) ∈ edges of Git DAG
- w_phylo = exp(-d_phylo(c_i, c_j))
- p(c) = predicted failure probability
```

**Key Insight:**
- Phylogenetically close commits should have similar failure predictions
- Encodes inductive bias: evolutionary proximity ⟹ behavioral similarity
- Helps smooth predictions across the evolutionary tree

**Reference:** Inspired by phylogenetic comparative methods (Felsenstein, 2004)

---

## Graph Construction

### Multi-Edge Phylogenetic Graph

The V9 system constructs a multi-edge graph capturing test relationships:

```
┌─────────────────────────────────────────┐
│       Multi-Edge Graph Builder          │
├─────────────────────────────────────────┤
│                                         │
│  Historical Execution Data              │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────────────┐    │
│  │ 1. Co-Failure Edges (w=1.0)    │    │
│  │    Tests that fail together     │    │
│  │    in the same build            │    │
│  └─────────────────────────────────┘    │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────────────┐    │
│  │ 2. Co-Success Edges (w=0.5)    │    │
│  │    Tests that pass together     │    │
│  │    in the same build            │    │
│  └─────────────────────────────────┘    │
│           │                             │
│           ↓                             │
│  ┌─────────────────────────────────┐    │
│  │ 3. Semantic Edges (w=0.3)      │    │
│  │    Top-k semantically similar   │    │
│  │    tests (SBERT similarity)     │    │
│  └─────────────────────────────────┘    │
│           │                             │
│           ↓                             │
│      Combined Graph                     │
│      (PyG Data object)                  │
└─────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| min_co_occurrences | 1 | Minimum co-occurrences for edge |
| weight_threshold | 0.05 | Minimum edge weight |
| semantic_top_k | 5 | Top-k semantic neighbors |
| semantic_threshold | 0.75 | Minimum semantic similarity |

### Graph Impact

- **Before multi-edge:** ~0.02% density (sparse, limited information)
- **After multi-edge:** 0.5-1.0% density (richer connections)
- **GATv2 benefit:** More edges = better attention propagation

---

## Structural Features (V2.5)

### 10 Selected Features

| Feature | Description | Type |
|---------|-------------|------|
| test_age | Builds since first appearance | Numerical |
| failure_rate | Historical failure percentage | Numerical |
| recent_failure_rate | Failures in last 5 builds | Numerical |
| flakiness_rate | Pass/Fail oscillation frequency | Numerical |
| commit_count | Number of associated commits | Numerical |
| test_novelty | First appearance flag (0/1) | Binary |
| consecutive_failures | Current failure streak | Numerical |
| max_consecutive_failures | Maximum observed streak | Numerical |
| failure_trend | Trend analysis (-1/0/+1) | Categorical |
| cr_count | Associated change requests | Numerical |

### Temporal Windows

- **Recent:** Last 5 builds
- **Very recent:** Last 2 builds
- **Medium-term:** Last 10 builds

### Feature Selection Process

Features selected from original 29 V2 features based on:
1. Feature importance analysis
2. Correlation analysis (remove redundant features)
3. APFD impact evaluation

---

## References

### Core Technologies
- [SBERT Paper](https://arxiv.org/abs/1908.10084) - Reimers & Gurevych (2019)
- [GAT Paper](https://arxiv.org/abs/1710.10903) - Veličković et al. (2018)
- [GATv2 Paper](https://arxiv.org/abs/2105.14491) - Brody et al. (2022)
- [GGNN Paper](https://arxiv.org/abs/1511.05493) - Li et al. (2016)

### Loss Functions
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - Lin et al. (2017)
- [RankNet Paper](https://icml.cc/Conferences/2005/proceedings/papers/012_Learning_BursgesEtAl.pdf) - Burges et al. (2005)

### Phylogenetic Foundations
- [Inferring Phylogenies](https://www.sinauer.com/media/wysiwyg/samples/InferringPhylogenies.pdf) - Felsenstein (2004)
- [Change Impact Graphs](https://doi.org/10.1016/j.infsof.2009.06.002) - German et al. (2009)
- [Origin Analysis](https://doi.org/10.1109/TSE.2005.32) - Godfrey & Zou (2005)

### Code Intelligence
- [CodeBERT](https://arxiv.org/abs/2002.08155) - Feng et al. (2020)
- [GraphCodeBERT](https://arxiv.org/abs/2009.08366) - Guo et al. (2021)

### TCP State-of-the-Art
- [RETECS](https://doi.org/10.1145/3092703.3092709) - Spieker et al. (2017)
- [NodeRank](https://doi.org/10.1007/s10664-024-10453-4) - van Soest et al. (2024)

---

**Last Updated:** November 2025
**Paradigm Version:** Phylogenetic V1.0
