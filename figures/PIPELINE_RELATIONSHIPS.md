# Filo-Priori V7: Complete Pipeline Relationships (End-to-End)

**Generated:** 2025-11-06
**Diagram:** `figures/complete_pipeline_architecture.mmd`
**Project:** Test Case Prioritization using Dual-Stream Graph-Semantic Model

---

## Table of Contents

1. [Overview](#overview)
2. [PHASE 1: Classification](#phase-1-test-case-failure-classification)
3. [PHASE 2: Prioritization](#phase-2-test-case-prioritization--apfd)
4. [Key Libraries and Technologies](#key-libraries-and-technologies)
5. [Critical Parameters Summary](#critical-parameters-summary)
6. [File Mappings](#file-mappings)

---

## Overview

The Filo-Priori V7 pipeline consists of **two main phases**:

### **PHASE 1: TEST CASE FAILURE CLASSIFICATION** (Machine Learning)
- **Goal:** Train a model to predict the probability that a test case will fail
- **Input:** Historical test case data (summary, steps, commits, results)
- **Output:** Failure probability P(Fail) for each test case
- **Duration:** ~2-3 hours (training)

### **PHASE 2: TEST CASE PRIORITIZATION & APFD** (Ranking + Evaluation)
- **Goal:** Rank test cases by failure probability and evaluate prioritization quality
- **Input:** Failure probabilities from Phase 1
- **Output:** Prioritized test cases + APFD metrics per build
- **Duration:** ~5-10 minutes (inference + ranking)

---

# PHASE 1: TEST CASE FAILURE CLASSIFICATION

## STEP 1: DATA INPUT & PREPROCESSING

### 1.1 DataLoader: Load and Prepare Data

**File:** `src/preprocessing/data_loader.py`
**Class:** `DataLoader`
**Function:** `prepare_dataset(sample_size=None)`

#### **Input:**
```python
{
    'train.csv': DataFrame with columns [
        'TE_Summary',        # Test execution summary
        'TC_Steps',          # Test case steps
        'commit',            # Related commits (string or list)
        'CR',                # Change requests
        'CR_Type',           # Type of change request
        'CR_Component',      # Affected component
        'TE_Test_Result',    # Result: 'Pass', 'Fail', 'Delete', 'Blocked'
        'Build_ID',          # Build identifier
        'TC_Key'             # Test case unique key
    ],
    'test.csv': Same structure (31,333 samples, 277 builds)
}
```

#### **Processing:**
```python
def prepare_dataset(self, sample_size=None):
    # 1. Load CSV files
    train_df = pd.read_csv('datasets/train.csv')
    test_df = pd.read_csv('datasets/test.csv')

    # 2. Clean text fields (remove NAs, strip whitespace)
    train_df['TE_Summary'] = train_df['TE_Summary'].fillna('').str.strip()
    train_df['TC_Steps'] = train_df['TC_Steps'].fillna('').str.strip()

    # 3. Process commit lists (convert string to list if needed)
    train_df['commit_processed'] = train_df['commit'].apply(self._process_commits)

    # 4. Binary labeling based on strategy
    # Strategy: 'pass_vs_all' with binary_positive_class='Pass'
    # → Pass=1 (positive), Not-Pass=0 (Fail/Delete/Blocked)
    train_df['label'] = (train_df['TE_Test_Result'] == 'Pass').astype(int)

    # 5. Split data: 80% train, 10% val, 10% test
    train, val, test = self._split_data(train_df, test_size=0.2, val_size=0.1)

    # 6. Compute class weights for loss function
    class_counts = train['label'].value_counts()
    class_weights = len(train) / (len(class_counts) * class_counts)

    return {
        'train': train,
        'val': val,
        'test': test,
        'class_weights': class_weights.values,
        'label_mapping': {0: 'Not-Pass', 1: 'Pass'}
    }
```

#### **Output:**
```python
data_dict = {
    'train': DataFrame [~55,000 rows],
    'val': DataFrame [~7,000 rows],
    'test': DataFrame [~7,000 rows],
    'class_weights': array([0.15, 0.85]),  # Inverse frequency weights
    'label_mapping': {0: 'Not-Pass', 1: 'Pass'}
}
```

**Key Library:** `pandas`

---

### 1.2 TextProcessor: Combine and Clean Text

**File:** `src/preprocessing/text_processor.py`
**Class:** `TextProcessor`
**Function:** `prepare_batch_texts(summaries, steps, commits)`

#### **Input:**
```python
{
    'summaries': List[str],  # Test execution summaries
    'steps': List[str],      # Test case steps
    'commits': List[str]     # Commit messages (top 3)
}
```

#### **Processing:**
```python
def prepare_batch_texts(self, summaries, steps, commits):
    """Combine multiple text fields into single strings"""
    combined = []

    for summary, step, commit in zip(summaries, steps, commits):
        # 1. Clean each field
        summary_clean = self._clean_text(summary)
        step_clean = self._clean_text(step)
        commit_clean = self._process_commit_list(commit)  # Extract top 3

        # 2. Combine with separator token
        combined_text = f"{summary_clean} [SEP] {step_clean} [SEP] {commit_clean}"

        # 3. Truncate if too long
        combined_text = self._truncate(combined_text, max_length=512)

        combined.append(combined_text)

    return combined
```

#### **Example:**
```
Input:
  summary: "Test login functionality"
  steps: "1. Open app 2. Enter credentials 3. Click login"
  commits: "Fix auth bug, Update UI, Add validation"

Output:
  "Test login functionality [SEP] 1. Open app 2. Enter credentials 3. Click login [SEP] Fix auth bug, Update UI, Add validation"
```

#### **Output:**
```python
combined_texts: List[str]  # One combined text per sample
```

**Key Library:** `re` (regex), string operations

---

### 1.3 SMOTE: Balance Training Data (Optional)

**File:** `src/preprocessing/data_loader.py`
**Function:** `apply_smote_to_embeddings(embeddings, labels, config)`

#### **Input:**
```python
{
    'embeddings': np.ndarray shape [N, 1024],  # Train embeddings
    'labels': np.ndarray shape [N],            # Train labels (0 or 1)
    'config': dict with 'data.smote' settings
}
```

#### **Processing:**
```python
from imblearn.over_sampling import SMOTE

def apply_smote_to_embeddings(embeddings, labels, config):
    # 1. Get SMOTE parameters from config
    k_neighbors = config['data']['smote'].get('k_neighbors', 5)
    sampling_strategy = config['data']['smote'].get('sampling_strategy', 'auto')

    # 2. Apply SMOTE
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=42
    )

    # 3. Generate synthetic samples
    embeddings_resampled, labels_resampled = smote.fit_resample(
        embeddings,
        labels
    )

    # 4. Log statistics
    logger.info(f"Before SMOTE: {Counter(labels)}")
    logger.info(f"After SMOTE: {Counter(labels_resampled)}")

    return embeddings_resampled, labels_resampled
```

#### **Example:**
```
Before SMOTE:
  Class 0 (Not-Pass): 7,000 samples
  Class 1 (Pass): 48,000 samples

After SMOTE:
  Class 0 (Not-Pass): 48,000 samples (41,000 synthetic)
  Class 1 (Pass): 48,000 samples
```

#### **Output:**
```python
{
    'embeddings': np.ndarray shape [M, 1024],  # M > N (balanced)
    'labels': np.ndarray shape [M]              # Balanced labels
}
```

**Key Library:** `imbalanced-learn` (SMOTE)

---

## STEP 2: SEMANTIC EMBEDDING EXTRACTION

### 2.1 BGE-Large Encoder: Generate Embeddings

**File:** `src/embeddings/semantic_encoder.py`
**Class:** `SemanticEncoder`
**Function:** `encode_dataset(texts, cache_path=None)`

#### **Input:**
```python
{
    'texts': List[str],  # Combined text strings from TextProcessor
    'cache_path': str    # Optional path to cache embeddings
}
```

#### **Processing:**
```python
from sentence_transformers import SentenceTransformer

class SemanticEncoder:
    def __init__(self, config, device='cuda'):
        # Load pre-trained model
        self.model = SentenceTransformer(
            'BAAI/bge-large-en-v1.5',  # 1024-dim embeddings
            device=device
        )
        self.max_length = 512
        self.batch_size = 32

    def encode_dataset(self, texts, cache_path=None):
        # 1. Check cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        # 2. Encode texts in batches
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization
            convert_to_numpy=True
        )

        # 3. Save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"Embeddings cached to {cache_path}")

        return embeddings
```

#### **Output:**
```python
embeddings: np.ndarray shape [N, 1024]
# Each row is a normalized 1024-dimensional vector representing the semantic meaning
```

**Key Library:** `sentence-transformers` (based on transformers + PyTorch)

---

## STEP 3: GRAPH STRUCTURE CONSTRUCTION

### 3.1 k-NN Graph Builder: Create Graph from Embeddings

**File:** `src/phylogenetic/tree_builder.py`
**Class:** `PhylogeneticTreeBuilder`
**Function:** `build_knn_graph(embeddings, k=10)`

#### **Input:**
```python
{
    'embeddings': np.ndarray shape [N, 1024],  # Node features
    'k': int  # Number of nearest neighbors
}
```

#### **Processing:**
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_knn_graph(embeddings, k=10):
    # 1. Compute k-NN using cosine distance
    nn = NearestNeighbors(
        n_neighbors=k,
        metric='cosine',  # 1 - cosine_similarity
        algorithm='auto'
    )
    nn.fit(embeddings)

    # 2. Find k nearest neighbors for each node
    distances, indices = nn.kneighbors(embeddings)

    # 3. Build edge list (source, target)
    edge_list = []
    weight_list = []

    for i in range(len(embeddings)):
        for j in range(k):
            neighbor_idx = indices[i, j]
            distance = distances[i, j]

            # Add edge: i -> neighbor_idx
            edge_list.append([i, neighbor_idx])

            # Compute edge weight: 1 / (1 + distance)
            # Higher weight for closer nodes
            weight = 1.0 / (1.0 + distance)
            weight_list.append(weight)

    # 4. Convert to numpy arrays
    edge_index = np.array(edge_list).T  # Shape: [2, E]
    edge_weights = np.array(weight_list)  # Shape: [E]

    logger.info(f"Built k-NN graph: {len(embeddings)} nodes, {edge_index.shape[1]} edges")

    return edge_index, edge_weights
```

#### **Output:**
```python
{
    'edge_index': np.ndarray shape [2, num_edges],
    # edge_index[0] = source nodes
    # edge_index[1] = target nodes

    'edge_weights': np.ndarray shape [num_edges],
    # Weight for each edge (1/(1+distance))
}

# Example:
# edge_index = [[0, 0, 0, 1, 1, 1, ...],  # Source nodes
#               [5, 12, 8, 3, 9, 15, ...]] # Target nodes
# edge_weights = [0.95, 0.87, 0.92, ...]
```

**Key Library:** `scikit-learn` (NearestNeighbors)

---

## STEP 4: DUAL-STREAM MODEL ARCHITECTURE

### 4.1 Semantic Stream: Extract Text Features

**File:** `src/models/dual_stream.py`
**Class:** `SemanticStream`
**Function:** `forward(x)`

#### **Input:**
```python
x: torch.Tensor shape [batch, 1024]  # BGE embeddings
```

#### **Processing:**
```python
class SemanticStream(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, dropout=0.4):
        super().__init__()

        # Projection layer
        self.projection = nn.Linear(input_dim, hidden_dim)

        # FFN blocks
        self.ffn_blocks = nn.ModuleList([
            FFNBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 1. Project to hidden dimension
        x = self.projection(x)  # [batch, 1024] → [batch, 256]

        # 2. Apply FFN blocks with residuals
        for ffn in self.ffn_blocks:
            x = ffn(x) + x  # Residual connection

        # 3. Final normalization
        x = self.layer_norm(x)

        return x  # [batch, 256]

class FFNBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)        # [batch, 256] → [batch, 1024]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)        # [batch, 1024] → [batch, 256]
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x
```

#### **Output:**
```python
semantic_features: torch.Tensor shape [batch, 256]
# Contextual semantic representation of each test case
```

**Key Library:** `torch.nn` (PyTorch)

---

### 4.2 Structural Stream: Extract Graph Features

**File:** `src/models/dual_stream.py`
**Class:** `StructuralStream`
**Function:** `forward(x, edge_index, edge_weights)`

#### **Input:**
```python
{
    'x': torch.Tensor shape [N, 1024],        # Node features (all nodes in batch)
    'edge_index': torch.LongTensor shape [2, E],  # Graph connectivity
    'edge_weights': torch.Tensor shape [E]    # Edge weights
}
```

#### **Processing:**
```python
class StructuralStream(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, dropout=0.4):
        super().__init__()

        # Projection
        self.projection = nn.Linear(input_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_weights):
        # 1. Project node features
        x = self.projection(x)  # [N, 1024] → [N, 256]

        # 2. Apply message passing layers
        for mp in self.mp_layers:
            x = mp(x, edge_index, edge_weights)

        return x  # [N, 256]

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_fn = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_weights):
        residual = x

        # 1. Gather source and target node features
        src_nodes = edge_index[0]  # Source node indices
        dst_nodes = edge_index[1]  # Target node indices

        src_features = x[src_nodes]  # [E, 256]
        dst_features = x[dst_nodes]  # [E, 256]

        # 2. Concatenate features
        concat_features = torch.cat([src_features, dst_features], dim=-1)  # [E, 512]

        # 3. Compute messages
        messages = self.message_fn(concat_features)  # [E, 256]
        messages = self.activation(messages)

        # 4. Weight messages by edge weights
        messages = messages * edge_weights.unsqueeze(-1)  # [E, 256]

        # 5. Aggregate messages (mean aggregation)
        aggregated = torch.zeros_like(x)
        count = torch.zeros(x.size(0), 1, device=x.device)

        aggregated.index_add_(0, dst_nodes, messages)
        count.index_add_(0, dst_nodes, torch.ones_like(edge_weights).unsqueeze(-1))

        aggregated = aggregated / (count + 1e-8)  # Avoid division by zero

        # 6. Update node features
        x = self.update_fn(aggregated)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x
```

#### **Output:**
```python
structural_features: torch.Tensor shape [N, 256]
# Graph-aware representation of each node (test case)
```

**Key Library:** `torch.nn` (PyTorch), custom message passing

---

### 4.3 Fusion Layer: Combine Semantic + Structural

**File:** `src/models/cross_attention.py`
**Class:** `CrossAttentionFusion`
**Function:** `forward(semantic_features, structural_features)`

#### **Input:**
```python
{
    'semantic_features': torch.Tensor shape [batch, 256],
    'structural_features': torch.Tensor shape [batch, 256]
}
```

#### **Processing:**
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()

        # Cross-attention: semantic → structural
        self.cross_attn_sem2struct = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: structural → semantic
        self.cross_attn_struct2sem = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_sem = nn.LayerNorm(hidden_dim)
        self.norm_struct = nn.LayerNorm(hidden_dim)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )

    def forward(self, semantic_features, structural_features):
        batch_size = semantic_features.size(0)

        # 1. Add sequence dimension (length=1)
        sem_seq = semantic_features.unsqueeze(1)      # [batch, 1, 256]
        struct_seq = structural_features.unsqueeze(1)  # [batch, 1, 256]

        # 2. Cross-attention: semantic attends to structural
        sem_attended, _ = self.cross_attn_sem2struct(
            query=sem_seq,
            key=struct_seq,
            value=struct_seq
        )  # [batch, 1, 256]
        sem_attended = sem_attended.squeeze(1)  # [batch, 256]
        semantic_enhanced = self.norm_sem(semantic_features + sem_attended)

        # 3. Cross-attention: structural attends to semantic
        struct_attended, _ = self.cross_attn_struct2sem(
            query=struct_seq,
            key=sem_seq,
            value=sem_seq
        )  # [batch, 1, 256]
        struct_attended = struct_attended.squeeze(1)  # [batch, 256]
        structural_enhanced = self.norm_struct(structural_features + struct_attended)

        # 4. Concatenate and fuse
        concat = torch.cat([semantic_enhanced, structural_enhanced], dim=-1)  # [batch, 512]
        fused = self.fusion_gate(concat)  # [batch, 256]

        return fused
```

#### **Output:**
```python
fused_features: torch.Tensor shape [batch, 512]
# Or [batch, 256] depending on fusion configuration
# Combined representation with both semantic and structural information
```

**Key Library:** `torch.nn.MultiheadAttention` (PyTorch)

---

### 4.4 Classifier Head: Predict Failure Probability

**File:** `src/models/dual_stream.py`
**Function:** `forward(fused_features)`

#### **Input:**
```python
fused_features: torch.Tensor shape [batch, 512]
```

#### **Processing:**
```python
class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[128, 64], num_classes=2, dropout=0.4):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.mlp(x)
        return logits

# Full forward pass
def model_forward(embeddings, edge_index, edge_weights):
    # 1. Semantic stream
    semantic_features = semantic_stream(embeddings)  # [batch, 256]

    # 2. Structural stream (on all nodes)
    structural_features_all = structural_stream(
        embeddings, edge_index, edge_weights
    )  # [N, 256]

    # 3. Pool structural features to batch level
    structural_features = global_mean_pool(
        structural_features_all,
        batch_indices
    )  # [batch, 256]

    # 4. Fusion
    fused = fusion_layer(semantic_features, structural_features)  # [batch, 512]

    # 5. Classification
    logits = classifier(fused)  # [batch, 2]

    return logits
```

#### **Output:**
```python
logits: torch.Tensor shape [batch, 2]
# Raw scores for each class
# logits[:, 0] = Not-Pass (Fail) score
# logits[:, 1] = Pass score
```

**Key Library:** `torch.nn` (PyTorch)

---

## STEP 5: MODEL TRAINING

### 5.1 Focal Loss: Handle Class Imbalance

**File:** `src/training/losses.py`
**Class:** `FocalLoss`
**Function:** `forward(inputs, targets)`

#### **Input:**
```python
{
    'inputs': torch.Tensor shape [batch, 2],  # Logits from classifier
    'targets': torch.LongTensor shape [batch]  # Ground truth labels (0 or 1)
}
```

#### **Processing:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.15, 0.85], gamma=2.0):
        """
        Focal Loss for imbalanced classification

        Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

        Args:
            alpha: Class weights [weight_class0, weight_class1]
                   Higher weight = more focus on that class
            gamma: Focusing parameter (0 = CE loss, 2 = strong focus on hard examples)
        """
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 1. Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. Compute p_t: probability of true class
        p_t = torch.exp(-ce_loss)  # Probability of correct class

        # 3. Get alpha for each sample based on its true class
        alpha_t = self.alpha[targets]

        # 4. Apply focal term: (1 - p_t)^gamma
        # Easy examples (p_t close to 1): focal_weight ≈ 0 (downweight)
        # Hard examples (p_t close to 0): focal_weight ≈ 1 (keep weight)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # 5. Final loss
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()
```

#### **Example:**
```
Sample 1: Predicted P(Pass)=0.95, True=Pass
  → p_t = 0.95 (confident correct)
  → focal_weight = 0.85 * (1-0.95)^2 = 0.85 * 0.0025 = 0.002
  → Loss contribution ≈ 0 (easy example, downweighted)

Sample 2: Predicted P(Pass)=0.55, True=Pass
  → p_t = 0.55 (uncertain correct)
  → focal_weight = 0.85 * (1-0.55)^2 = 0.85 * 0.2025 = 0.172
  → Loss contribution = moderate (harder example)

Sample 3: Predicted P(Pass)=0.10, True=Fail
  → p_t = 0.90 (confident correct)
  → focal_weight = 0.15 * (1-0.90)^2 = 0.15 * 0.01 = 0.0015
  → Loss contribution ≈ 0 (easy example for minority class)
```

#### **Output:**
```python
loss: torch.Tensor shape []  # Scalar loss value
```

**Key Library:** `torch.nn.functional` (PyTorch)

---

### 5.2 Training Loop: Optimize Model

**File:** `src/training/trainer.py`
**Class:** `Trainer`
**Function:** `train()`

#### **Processing:**
```python
class Trainer:
    def __init__(self, model, config, train_data, val_data, graph_data, device):
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],      # 5e-5
            weight_decay=config['training']['weight_decay']  # 2e-4
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )

        # Loss function
        self.criterion = FocalLoss(alpha=[0.15, 0.85], gamma=2.0)

        # Early stopping
        self.patience = config['training']['early_stopping']['patience']  # 12
        self.best_metric = 0.0
        self.patience_counter = 0

    def train(self):
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in self.train_loader:
                # 1. Build k-NN graph for this batch
                edge_index, edge_weights = self._build_batch_knn_graph(batch)

                # 2. Forward pass
                logits = self.model(batch.x, edge_index, edge_weights)

                # 3. Compute loss
                loss = self.criterion(logits, batch.y)

                # 4. Backward pass
                loss.backward()

                # 5. Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                # 6. Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()

            # 7. Scheduler step
            self.scheduler.step()

            # Validation phase
            val_metrics = self._validate()

            # 8. Early stopping check
            val_f1 = val_metrics['f1_macro']
            if val_f1 > self.best_metric:
                self.best_metric = val_f1
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
```

#### **Output:**
```python
{
    'best_model.pt': Checkpoint file with best validation F1,
    'last_model.pt': Checkpoint file from last epoch,
    'training_history': {
        'train_loss': [0.652, 0.589, 0.531, ...],
        'val_f1_macro': [0.423, 0.456, 0.492, ...]
    }
}
```

**Key Libraries:** `torch.optim` (PyTorch)

---

## STEP 6: MODEL EVALUATION

### 6.1 Generate Predictions

**File:** `main.py` (evaluation section)
**Function:** Inline in `evaluate_model()`

#### **Input:**
```python
{
    'model': Trained model,
    'test_embeddings': torch.Tensor [N, 1024],
    'test_edge_index': torch.LongTensor [2, E],
    'test_edge_weights': torch.Tensor [E],
    'test_labels': np.ndarray [N]
}
```

#### **Processing:**
```python
def generate_predictions(model, embeddings, edge_index, edge_weights, T=1.0, threshold=0.5):
    model.eval()

    with torch.no_grad():
        # 1. Forward pass
        logits = model(embeddings, edge_index, edge_weights)  # [N, 2]

        # 2. Apply temperature scaling (calibration)
        # T > 1: softer probabilities (more uncertain)
        # T < 1: harder probabilities (more confident)
        # T = 1: no scaling
        probabilities = torch.softmax(logits / T, dim=1)  # [N, 2]

        # 3. Apply optimal threshold (found during validation)
        # Binary classification: compare P(Pass) to threshold
        predictions = (probabilities[:, 1] >= threshold).long()  # [N]

        # 4. Convert to numpy
        probabilities = probabilities.cpu().numpy()
        predictions = predictions.cpu().numpy()

    return predictions, probabilities

# Usage
predictions, probabilities = generate_predictions(
    model,
    test_embeddings,
    test_edge_index,
    test_edge_weights,
    T=1.05,  # Loaded from best_model.pt
    threshold=0.52  # Loaded from best_model.pt
)
```

#### **Output:**
```python
{
    'predictions': np.ndarray shape [N],  # Binary predictions (0 or 1)
    'probabilities': np.ndarray shape [N, 2],  # Class probabilities
    # probabilities[:, 0] = P(Not-Pass/Fail) → Used for prioritization
    # probabilities[:, 1] = P(Pass)
}

# Example:
# predictions = [1, 0, 1, 1, 0, ...]  # Pass, Fail, Pass, Pass, Fail, ...
# probabilities = [[0.15, 0.85],  # P(Fail)=0.15, P(Pass)=0.85
#                  [0.72, 0.28],  # P(Fail)=0.72, P(Pass)=0.28
#                  [0.18, 0.82],  # P(Fail)=0.18, P(Pass)=0.82
#                  ...]
```

**Key Library:** `torch` (PyTorch), `numpy`

---

### 6.2 Compute Classification Metrics

**File:** `src/evaluation/metrics.py`
**Function:** `compute_metrics(y_true, y_pred, num_classes, label_names, probabilities)`

#### **Input:**
```python
{
    'y_true': np.ndarray [N],  # Ground truth labels
    'y_pred': np.ndarray [N],  # Predicted labels
    'num_classes': int,        # Number of classes (2)
    'label_names': List[str],  # ['Not-Pass', 'Pass']
    'probabilities': np.ndarray [N, 2]  # Class probabilities
}
```

#### **Processing:**
```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc
)

def compute_metrics(y_true, y_pred, num_classes, label_names, probabilities):
    # 1. Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # 2. F1 Score (macro average: unweighted mean of per-class F1)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 3. F1 Score (weighted average: weighted by support)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # 4. Precision and Recall per class
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # cm = [[TN, FP],
    #       [FN, TP]]

    # 6. AUPRC (Area Under Precision-Recall Curve)
    auprc_scores = []
    for class_idx in range(num_classes):
        # Binarize: class vs rest
        y_true_binary = (y_true == class_idx).astype(int)
        y_score = probabilities[:, class_idx]

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        auprc = auc(recall, precision)
        auprc_scores.append(auprc)

    # 7. Prediction diversity (minority class proportion)
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique, counts))
    min_count = min(pred_dist.values())
    max_count = max(pred_dist.values())
    diversity = min_count / max_count if max_count > 0 else 0.0

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'auprc_per_class': auprc_scores,
        'auprc_macro': np.mean(auprc_scores),
        'prediction_diversity': diversity
    }
```

#### **Output:**
```python
metrics = {
    'accuracy': 0.625,
    'f1_macro': 0.587,
    'f1_weighted': 0.612,
    'precision_per_class': [0.65, 0.60],  # [Not-Pass, Pass]
    'recall_per_class': [0.55, 0.68],     # [Not-Pass, Pass]
    'confusion_matrix': [[1250, 750],  # TN, FP
                         [450, 1550]], # FN, TP
    'auprc_per_class': [0.62, 0.71],
    'auprc_macro': 0.665,
    'prediction_diversity': 0.45
}
```

**Key Library:** `scikit-learn`

---

# PHASE 2: TEST CASE PRIORITIZATION & APFD

## STEP 7: TEST CASE RANKING

### 7.1 Rank Test Cases per Build

**File:** `src/evaluation/apfd.py`
**Function:** `calculate_ranks_per_build(df, probability_col, build_col)`

#### **Input:**
```python
{
    'df': pd.DataFrame with columns [
        'Build_ID',        # Build identifier
        'TC_Key',          # Test case key
        'TE_Test_Result',  # Ground truth result
        'probability'      # P(Fail) from Phase 1
    ],
    'probability_col': str = 'probability',
    'build_col': str = 'Build_ID'
}
```

#### **Processing:**
```python
def calculate_ranks_per_build(df, probability_col='probability', build_col='Build_ID'):
    """
    Calculate priority ranks per build based on failure probabilities.

    Ranks are 1-indexed within each build, where:
    - rank=1 is the highest priority (highest P(Fail))
    - rank=n is the lowest priority (lowest P(Fail))
    """
    df = df.copy()

    # Calculate ranks per build
    # Higher probability = lower rank number (higher priority)
    df['rank'] = df.groupby(build_col)[probability_col] \
                   .rank(method='first', ascending=False) \
                   .astype(int)

    logger.info(f"Ranks calculated per build (rank range: {df['rank'].min()}-{df['rank'].max()})")

    return df

# Example:
# Before ranking:
# Build_ID  TC_Key    TE_Test_Result  probability
# B1        TC001     Pass            0.15
# B1        TC002     Fail            0.72
# B1        TC003     Pass            0.18
# B2        TC004     Pass            0.22
# B2        TC005     Fail            0.65

# After ranking:
# Build_ID  TC_Key    TE_Test_Result  probability  rank
# B1        TC001     Pass            0.15         3  (lowest priority)
# B1        TC002     Fail            0.72         1  (highest priority)
# B1        TC003     Pass            0.18         2
# B2        TC004     Pass            0.22         2
# B2        TC005     Fail            0.65         1  (highest priority in B2)
```

#### **Output:**
```python
df_with_ranks: pd.DataFrame
# Same DataFrame with added 'rank' column
# Ranks are calculated independently for each Build_ID
```

**Key Library:** `pandas`

---

### 7.2 Save Prioritized Test Cases

**File:** `src/evaluation/apfd.py`
**Function:** `generate_prioritized_csv(df, output_path, probability_col, label_col, build_col)`

#### **Input:**
```python
{
    'df': pd.DataFrame with ranks,
    'output_path': str,  # Path to save CSV
    'probability_col': str = 'probability',
    'label_col': str = 'label_binary',
    'build_col': str = 'Build_ID'
}
```

#### **Processing:**
```python
def generate_prioritized_csv(df, output_path, probability_col='probability',
                            label_col='label_binary', build_col='Build_ID'):
    """
    Generate prioritized test cases CSV with ranks per build.
    """
    # 1. Calculate ranks if not present
    if 'rank' not in df.columns:
        df = calculate_ranks_per_build(df, probability_col, build_col)

    # 2. Calculate priority score (can include diversity if available)
    df['diversity_score'] = 0.0  # Placeholder
    df['priority_score'] = df[probability_col]

    # 3. Select and order columns for output
    output_cols = [
        build_col,
        'TC_Key',
        'TE_Test_Result',
        label_col,
        probability_col,
        'diversity_score',
        'priority_score',
        'rank'
    ]

    output_cols = [col for col in output_cols if col in df.columns]

    # 4. Save to CSV
    df[output_cols].to_csv(output_path, index=False)
    logger.info(f"Prioritized test cases saved to: {output_path}")

    return df
```

#### **Output:**
```
File: prioritized_test_cases.csv

Build_ID,TC_Key,TE_Test_Result,label_binary,probability,diversity_score,priority_score,rank
B1,TC002,Fail,1,0.72,0.0,0.72,1
B1,TC003,Pass,0,0.18,0.0,0.18,2
B1,TC001,Pass,0,0.15,0.0,0.15,3
B2,TC005,Fail,1,0.65,0.0,0.65,1
B2,TC004,Pass,0,0.22,0.0,0.22,2
...
```

**Key Library:** `pandas`

---

## STEP 8: APFD CALCULATION

### 8.1 Calculate APFD for Single Build

**File:** `src/evaluation/apfd.py`
**Function:** `calculate_apfd_single_build(ranks, labels)`

#### **Input:**
```python
{
    'ranks': np.ndarray [n],  # Ranks within build (1-indexed)
    'labels': np.ndarray [n]  # Binary labels (1=Fail, 0=Pass)
}
```

#### **Processing:**
```python
def calculate_apfd_single_build(ranks, labels):
    """
    Calculate APFD for a single build.

    Formula:
        APFD = 1 - (sum of failure ranks) / (n_failures * n_tests) + 1 / (2 * n_tests)

    Special cases:
        - n_tests = 1: APFD = 1.0 (no ordering possible)
        - n_failures = 0: APFD = None (skip build)
    """
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)

    n_tests = int(len(labels_arr))
    fail_indices = np.where(labels_arr != 0)[0]  # Indices of failures
    n_failures = len(fail_indices)

    # Business rule: if no failures, APFD is undefined (skip this build)
    if n_failures == 0:
        return None

    # Business rule: if only 1 test case, APFD = 1.0
    if n_tests == 1:
        return 1.0

    # Get ranks of failures
    failure_ranks = ranks_arr[fail_indices]

    # Calculate APFD
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) \
           + 1.0 / float(2.0 * n_tests)

    return float(np.clip(apfd, 0.0, 1.0))

# Example:
# Build with 10 tests, 2 failures at ranks [2, 5]
#
# ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# labels = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#          ^     ^        ^
#          Pass  Fail     Fail
#
# n_tests = 10
# n_failures = 2
# failure_ranks = [2, 5]
# sum(failure_ranks) = 7
#
# APFD = 1 - (7 / (2 * 10)) + 1 / (2 * 10)
#      = 1 - 0.35 + 0.05
#      = 0.70
#
# Interpretation: 70% of the failures were detected in the first 70% of tests
```

#### **Output:**
```python
apfd: float  # Score between 0.0 and 1.0 (higher is better)
# 1.0 = Perfect prioritization (all failures at the top)
# 0.5 = Random ordering
# 0.0 = Worst prioritization (all failures at the bottom)
```

**Key Library:** `numpy`

---

### 8.2 Calculate APFD for All Builds

**File:** `src/evaluation/apfd.py`
**Function:** `calculate_apfd_per_build(df, method_name, test_scenario, build_col, label_col, rank_col, result_col)`

#### **Input:**
```python
{
    'df': pd.DataFrame with columns [
        'Build_ID',        # Build identifier
        'TC_Key',          # Test case unique key
        'label_binary',    # Binary label (1=Fail, 0=Pass)
        'rank',            # Priority rank (1=highest)
        'TE_Test_Result'   # Original test result
    ],
    'method_name': str,    # Name of prioritization method
    'test_scenario': str,  # Type of test scenario
    'build_col': str = 'Build_ID',
    'label_col': str = 'label_binary',
    'rank_col': str = 'rank',
    'result_col': str = 'TE_Test_Result'
}
```

#### **Processing:**
```python
def calculate_apfd_per_build(df, method_name, test_scenario, build_col='Build_ID',
                            label_col='label_binary', rank_col='rank',
                            result_col='TE_Test_Result'):
    """
    Calculate APFD per build for the entire test set.

    BUSINESS RULE: Only builds with at least one test with result "Fail" are included.
    This should result in exactly 277 builds (as per project requirements).

    BUSINESS RULE (count_tc=1): Builds with only 1 unique test case MUST have APFD=1.0
    """
    results = []

    # Group by Build_ID
    grouped = df.groupby(build_col)

    logger.info(f"Calculating APFD for {len(grouped)} total builds...")

    builds_with_failures = 0
    builds_skipped = 0

    for build_id, build_df in grouped:
        # Count UNIQUE test cases
        if 'TC_Key' in build_df.columns:
            count_tc = build_df['TC_Key'].nunique()
        else:
            count_tc = len(build_df)

        # CRITICAL BUSINESS RULE: count_tc=1 → APFD=1.0
        if count_tc == 1:
            # Still need to verify this build has at least one failure
            fail_mask = (build_df[result_col].astype(str).str.strip() == "Fail")
            if not fail_mask.any():
                builds_skipped += 1
                continue

            builds_with_failures += 1

            # Count unique commits
            try:
                count_commits = count_total_commits(build_df)
            except Exception as e:
                count_commits = 0

            # Add to results with APFD=1.0
            results.append({
                'method_name': method_name,
                'build_id': build_id,
                'test_scenario': test_scenario,
                'count_tc': count_tc,
                'count_commits': count_commits,
                'apfd': 1.0,  # Business rule: count_tc=1 → APFD=1.0
                'time': 0.0
            })
            continue  # Skip standard APFD calculation

        # CRITICAL BUSINESS RULE: Only include builds with at least one "Fail" result
        fail_mask = (build_df[result_col].astype(str).str.strip() == "Fail")
        if not fail_mask.any():
            builds_skipped += 1
            continue

        builds_with_failures += 1

        # Count unique commits
        try:
            count_commits = count_total_commits(build_df)
        except Exception as e:
            count_commits = 0

        # Get ranks and labels for this build
        ranks = build_df[rank_col].values
        labels = fail_mask.astype(int).values

        # Calculate APFD for this build
        apfd = calculate_apfd_single_build(ranks, labels)

        # Skip if APFD is None
        if apfd is None:
            continue

        # Add to results
        results.append({
            'method_name': method_name,
            'build_id': build_id,
            'test_scenario': test_scenario,
            'count_tc': count_tc,
            'count_commits': count_commits,
            'apfd': apfd,
            'time': 0.0
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by build_id for consistency
    if len(results_df) > 0:
        results_df = results_df.sort_values('build_id').reset_index(drop=True)

    logger.info(f"APFD calculated for {len(results_df)} builds with 'Fail' results")
    logger.info(f"   Builds included: {builds_with_failures}")
    logger.info(f"   Builds skipped (no failures): {builds_skipped}")
    logger.info(f"   Expected: 277 builds (as per project requirements)")

    if len(results_df) != 277:
        logger.warning(f"⚠️  WARNING: Expected 277 builds but got {len(results_df)}")

    return results_df
```

#### **Output:**
```
File: apfd_per_build.csv

method_name,build_id,test_scenario,count_tc,count_commits,apfd,time
dual_stream_gnn_exp_17,B1,full_test,25,63,0.7500,0.0
dual_stream_gnn_exp_17,B2,full_test,10,37,0.6667,0.0
dual_stream_gnn_exp_17,B3,full_test,107,240,0.3266,0.0
...
dual_stream_gnn_exp_17,B277,full_test,18,12,0.5812,0.0

(277 rows total)
```

**Key Library:** `pandas`, `numpy`

---

### 8.3 Count Unique Commits

**File:** `src/evaluation/apfd.py`
**Function:** `count_total_commits(df_build)`

#### **Input:**
```python
df_build: pd.DataFrame for a single build
# Must have columns: 'commit', 'CR' (or 'CR_y')
```

#### **Processing:**
```python
import ast

def count_total_commits(df_build):
    """
    Count total commits for a build (including CRs).
    Works with both test.csv and test_filtered.csv structures.
    """
    total_commits = set()

    # Count commits from 'commit' column
    if 'commit' in df_build.columns:
        for commit_str in df_build['commit'].dropna():
            try:
                # Parse string representation of list
                commits = ast.literal_eval(commit_str)
                if isinstance(commits, list):
                    total_commits.update(commits)
                else:
                    total_commits.add(str(commit_str))
            except:
                total_commits.add(str(commit_str))

    # Count CRs (works with both CR and CR_y columns)
    cr_column = 'CR_y' if 'CR_y' in df_build.columns else 'CR' if 'CR' in df_build.columns else None
    if cr_column:
        for cr_str in df_build[cr_column].dropna():
            try:
                crs = ast.literal_eval(cr_str)
                if isinstance(crs, list):
                    for cr in crs:
                        total_commits.add(f"CR_{cr}")
            except:
                total_commits.add(f"CR_{cr_str}")

    return max(len(total_commits), 1)

# Example:
# Build DataFrame:
#   commit: "['abc123', 'def456', 'ghi789']"
#   CR: "['CR001', 'CR002']"
#
# Result: 5 unique commits (3 commits + 2 CRs)
```

#### **Output:**
```python
count_commits: int  # Number of unique commits + CRs
```

**Key Library:** `ast` (Abstract Syntax Tree)

---

## STEP 9: APFD SUMMARY STATISTICS

### 9.1 Generate APFD Report

**File:** `src/evaluation/apfd.py`
**Function:** `generate_apfd_report(df, method_name, test_scenario, output_path)`

#### **Input:**
```python
{
    'df': pd.DataFrame with ranks (from Step 7),
    'method_name': str,    # Prioritization method name
    'test_scenario': str,  # Test scenario type
    'output_path': str     # Optional path to save CSV report
}
```

#### **Processing:**
```python
def generate_apfd_report(df, method_name, test_scenario, output_path=None):
    """
    Generate complete APFD report with summary statistics.
    """
    # 1. Calculate APFD per build
    results_df = calculate_apfd_per_build(df, method_name, test_scenario)

    if len(results_df) == 0:
        logger.warning("No builds with failures found. APFD cannot be calculated.")
        return results_df, {}

    # 2. Calculate summary statistics
    summary_stats = {
        'total_builds': len(results_df),
        'mean_apfd': float(results_df['apfd'].mean()),
        'median_apfd': float(results_df['apfd'].median()),
        'std_apfd': float(results_df['apfd'].std()),
        'min_apfd': float(results_df['apfd'].min()),
        'max_apfd': float(results_df['apfd'].max()),
        'total_test_cases': int(results_df['count_tc'].sum()),
        'mean_tc_per_build': float(results_df['count_tc'].mean()),
        'builds_apfd_1.0': int((results_df['apfd'] == 1.0).sum()),
        'builds_apfd_gte_0.7': int((results_df['apfd'] >= 0.7).sum()),
        'builds_apfd_gte_0.5': int((results_df['apfd'] >= 0.5).sum()),
        'builds_apfd_lt_0.5': int((results_df['apfd'] < 0.5).sum())
    }

    # 3. Save to CSV if path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"APFD per-build report saved to: {output_path}")

    return results_df, summary_stats
```

#### **Output:**
```python
(results_df, summary_stats) = (
    pd.DataFrame with APFD per build,
    {
        'total_builds': 277,
        'mean_apfd': 0.596664,       # ⭐ PRIMARY METRIC
        'median_apfd': 0.583400,
        'std_apfd': 0.287543,
        'min_apfd': 0.041667,
        'max_apfd': 1.000000,
        'total_test_cases': 5247,
        'mean_tc_per_build': 18.94,
        'builds_apfd_1.0': 23,
        'builds_apfd_gte_0.7': 95,
        'builds_apfd_gte_0.5': 156,
        'builds_apfd_lt_0.5': 121
    }
)
```

**Key Library:** `pandas`, `numpy`

---

### 9.2 Print APFD Summary

**File:** `src/evaluation/apfd.py`
**Function:** `print_apfd_summary(summary_stats)`

#### **Input:**
```python
summary_stats: Dict  # Summary statistics from generate_apfd_report()
```

#### **Processing:**
```python
def print_apfd_summary(summary_stats):
    """Print formatted APFD summary statistics."""
    if not summary_stats:
        print("\n" + "="*70)
        print("APFD PER BUILD - NO DATA")
        print("="*70)
        print("No builds with failures found.")
        return

    print("\n" + "="*70)
    print("APFD PER BUILD - SUMMARY STATISTICS")
    print("="*70)
    print(f"Total builds analyzed: {summary_stats['total_builds']}")
    print(f"Total test cases: {summary_stats['total_test_cases']}")
    print(f"Mean TCs per build: {summary_stats['mean_tc_per_build']:.1f}")
    print(f"\nAPFD Statistics:")
    print(f"  Mean:   {summary_stats['mean_apfd']:.4f} ⭐ PRIMARY METRIC")
    print(f"  Median: {summary_stats['median_apfd']:.4f}")
    print(f"  Std:    {summary_stats['std_apfd']:.4f}")
    print(f"  Min:    {summary_stats['min_apfd']:.4f}")
    print(f"  Max:    {summary_stats['max_apfd']:.4f}")
    print(f"\nAPFD Distribution:")

    total = summary_stats['total_builds']
    pct_1_0 = summary_stats['builds_apfd_1.0']/total*100
    pct_gte_0_7 = summary_stats['builds_apfd_gte_0.7']/total*100
    pct_gte_0_5 = summary_stats['builds_apfd_gte_0.5']/total*100
    pct_lt_0_5 = summary_stats['builds_apfd_lt_0.5']/total*100

    print(f"  Builds with APFD = 1.0:  {summary_stats['builds_apfd_1.0']:3d} ({pct_1_0:5.1f}%)")
    print(f"  Builds with APFD ≥ 0.7:  {summary_stats['builds_apfd_gte_0.7']:3d} ({pct_gte_0_7:5.1f}%)")
    print(f"  Builds with APFD ≥ 0.5:  {summary_stats['builds_apfd_gte_0.5']:3d} ({pct_gte_0_5:5.1f}%)")
    print(f"  Builds with APFD < 0.5:  {summary_stats['builds_apfd_lt_0.5']:3d} ({pct_lt_0_5:5.1f}%)")
    print("="*70)
```

#### **Output (Console):**
```
======================================================================
APFD PER BUILD - SUMMARY STATISTICS
======================================================================
Total builds analyzed: 277
Total test cases: 5247
Mean TCs per build: 18.9

APFD Statistics:
  Mean:   0.5967 ⭐ PRIMARY METRIC
  Median: 0.5834
  Std:    0.2875
  Min:    0.0417
  Max:    1.0000

APFD Distribution:
  Builds with APFD = 1.0:   23 (  8.3%)
  Builds with APFD ≥ 0.7:   95 ( 34.3%)
  Builds with APFD ≥ 0.5:  156 ( 56.3%)
  Builds with APFD < 0.5:  121 ( 43.7%)
======================================================================
```

---

# KEY LIBRARIES AND TECHNOLOGIES

## Core Libraries

| Library | Version | Purpose | Used In |
|---------|---------|---------|---------|
| **PyTorch** | ≥1.12.0 | Deep learning framework | Model, Training, Inference |
| **sentence-transformers** | ≥2.2.0 | Pre-trained embeddings (BGE) | Semantic encoding |
| **scikit-learn** | ≥1.1.0 | k-NN, metrics | Graph construction, Evaluation |
| **pandas** | ≥1.4.0 | Data manipulation | Data loading, APFD calculation |
| **numpy** | ≥1.21.0 | Numerical operations | All stages |
| **imbalanced-learn** | ≥0.9.0 | SMOTE | Class balancing (optional) |

## Model-Specific Libraries

| Library | Purpose |
|---------|---------|
| `torch.nn` | Neural network layers (Linear, LayerNorm, Dropout, GELU) |
| `torch.nn.functional` | Functional API (softmax, cross_entropy) |
| `torch.optim` | Optimizers (AdamW) and schedulers (CosineAnnealingLR) |

## Pre-trained Model

**BGE-Large-EN-v1.5:**
- **Source:** BAAI (Beijing Academy of AI)
- **HuggingFace ID:** `BAAI/bge-large-en-v1.5`
- **Architecture:** BERT-based encoder
- **Embedding Dimension:** 1024
- **Max Sequence Length:** 512 tokens
- **Training:** Contrastive learning on 1B+ sentence pairs
- **Performance:** State-of-the-art on MTEB benchmark

---

# CRITICAL PARAMETERS SUMMARY

## Data Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `train_val_test_split` | 80/10/10 | Data split ratio |
| `binary_positive_class` | 'Pass' | Positive class (Class 1) |
| `use_smote` | false (default) | Enable/disable SMOTE |
| `sample_size` | None | Sample data (None = use all) |

## Embedding Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `model_name` | BAAI/bge-large-en-v1.5 | Pre-trained model |
| `embedding_dim` | 1024 | Embedding dimension |
| `max_length` | 512 | Max tokens per text |
| `batch_size` | 32 | Batch size for encoding |
| `normalize_embeddings` | true | L2 normalization |

## Graph Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `k_neighbors` | 10 | Number of nearest neighbors |
| `distance_metric` | 'cosine' | Distance metric for k-NN |
| `edge_weight_formula` | 1/(1+distance) | Edge weight calculation |

## Model Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Semantic Stream:** | | |
| `num_layers` | 2 | FFN blocks |
| `hidden_dim` | 256 | Hidden dimension |
| `dropout` | 0.3-0.4 | Dropout rate |
| **Structural Stream:** | | |
| `num_layers` | 2 | Message passing layers |
| `hidden_dim` | 256 | Hidden dimension |
| **Fusion:** | | |
| `fusion_type` | 'cross_attention' | Fusion method |
| `num_heads` | 4 | Attention heads |
| **Classifier:** | | |
| `hidden_dims` | [128, 64] | MLP hidden dimensions |
| `num_classes` | 2 | Binary classification |

## Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Loss:** | | |
| `loss_type` | 'focal' | Loss function |
| `focal_alpha` | [0.15, 0.85] | Class weights |
| `focal_gamma` | 2.0 | Focusing parameter |
| **Optimizer:** | | |
| `optimizer` | 'AdamW' | Optimizer type |
| `learning_rate` | 5e-5 | Initial learning rate |
| `weight_decay` | 2e-4 | L2 regularization |
| **Scheduler:** | | |
| `scheduler` | 'cosine' | LR scheduler |
| `min_lr` | 1e-6 | Minimum learning rate |
| **Training:** | | |
| `num_epochs` | 80 | Maximum epochs |
| `batch_size` | 32 | Training batch size |
| `grad_clip_norm` | 1.0 | Gradient clipping |
| `patience` | 12 | Early stopping patience |
| `monitor_metric` | 'val_f1_macro' | Metric to monitor |

## APFD Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `expected_builds` | 277 | Expected builds with failures |
| `count_tc_1_rule` | APFD=1.0 | Special case for 1 TC |
| `only_fail_results` | true | Only count 'Fail' status |

---

# FILE MAPPINGS

## Code Files

| Component | File | Main Classes/Functions |
|-----------|------|----------------------|
| **Main Pipeline** | `main.py` | Main execution flow |
| **Data Loading** | `src/preprocessing/data_loader.py` | `DataLoader.prepare_dataset()` |
| **Text Processing** | `src/preprocessing/text_processor.py` | `TextProcessor.prepare_batch_texts()` |
| **Semantic Encoding** | `src/embeddings/semantic_encoder.py` | `SemanticEncoder.encode_dataset()` |
| **Graph Construction** | `src/phylogenetic/tree_builder.py` | `PhylogeneticTreeBuilder.build_knn_graph()` |
| **Model** | `src/models/dual_stream.py` | `DualStreamPhylogeneticTransformer` |
| **Cross-Attention** | `src/models/cross_attention.py` | `CrossAttentionFusion` |
| **Loss Function** | `src/training/losses.py` | `FocalLoss` |
| **Trainer** | `src/training/trainer.py` | `Trainer.train()` |
| **Classification Metrics** | `src/evaluation/metrics.py` | `compute_metrics()` |
| **APFD Calculation** | `src/evaluation/apfd.py` | `calculate_apfd_per_build()`, `generate_apfd_report()` |

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/experiment_017_ranking_corrected.yaml` | Experiment configuration |
| `configs/experiment_013_pass_vs_fail.yaml` | Binary classification config (reference) |

## Output Files

| File | Description |
|------|-------------|
| `results/{exp}/best_model.pt` | Best model checkpoint |
| `results/{exp}/predictions.npz` | Test predictions + probabilities |
| `results/{exp}/confusion_matrix.png` | Confusion matrix visualization |
| `results/{exp}/precision_recall_curves.png` | PR curves |
| `results/{exp}/prioritized_test_cases.csv` | Ranked test cases per build |
| `results/{exp}/apfd_per_build.csv` | APFD scores per build |
| `results/{exp}/metrics.json` | All evaluation metrics |

---

## Quick Reference: Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **F1 Macro** | Unweighted average of per-class F1 | ≥ 0.50 |
| **Accuracy** | Correct predictions / Total | ≥ 0.60 |
| **Mean APFD** | Average APFD across 277 builds | ≥ 0.55 ⭐ |
| **APFD ≥ 0.5** | Builds with decent prioritization | ≥ 55% |

---

**Last Updated:** 2025-11-06
**Version:** Filo-Priori V7
**Experiment:** 017 (Ranking Corrected)
