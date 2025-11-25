%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e1f5ff','primaryTextColor':'#000','primaryBorderColor':'#0277bd','lineColor':'#0288d1','secondaryColor':'#fff3e0','tertiaryColor':'#f1f8e9'}}}%%

flowchart TB
    %% Title
    TITLE["<b>🕸️ GAT INFERENCE: How Graph Attention Works on Test.csv</b><br/>Focus: KNOWN test cases (76.7% of test.csv)"]

    %% Training Graph (Static)
    subgraph TRAIN_GRAPH ["📦 TRAINING GRAPH (Built once, static)"]
        direction LR
        TG_INFO["<b>Graph Construction (from train.csv)</b><br/>─────────────────────────────────<br/>Nodes: 2,347 unique TC_Keys<br/>Edges: 461,493 total<br/>  • Co-failure: 495 (0.1%)<br/>  • Co-success: 207,913 (45.1%)<br/>  • Semantic: 253,085 (54.8%)<br/><br/>Stored as: edge_index [2, 461493]<br/>Stored as: edge_weights [461493]"]

        TG_EXAMPLE["<b>Example Subgraph:</b><br/>─────────────────<br/>MCA-1015 (node 0)<br/>  ├─ co-failure → MCA-101956 (0.85)<br/>  ├─ co-success → MCA-102345 (0.72)<br/>  └─ semantic → MCA-201567 (0.68)<br/><br/>MCA-101956 (node 1)<br/>  ├─ co-failure → MCA-1015 (0.85)<br/>  └─ semantic → MCA-999888 (0.71)"]
    end

    %% Inference Batch
    subgraph BATCH ["🔄 INFERENCE BATCH (Build_789 example)"]
        direction TB

        B_INPUT["<b>Input Batch (4 test cases)</b><br/>─────────────────────────────<br/>1. MCA-1015 (global_idx=0) ✅<br/>2. MCA-NEW-123 (global_idx=-1) ❌<br/>3. MCA-101956 (global_idx=1) ✅<br/>4. MCA-NEW-456 (global_idx=-1) ❌"]

        B_FILTER["<b>STEP 1: Filter Orphans</b><br/>─────────────────────────────<br/>valid_mask = (global_indices != -1)<br/>→ [True, False, True, False]<br/><br/>Valid samples: [MCA-1015, MCA-101956]<br/>Orphan samples: [MCA-NEW-123, MCA-NEW-456]"]

        B_SUBGRAPH["<b>STEP 2: Extract Subgraph</b><br/>─────────────────────────────<br/>subset = [0, 1]  (global indices)<br/><br/>subgraph() extracts edges connecting<br/>nodes 0 and 1 from training graph<br/><br/>Result:<br/>  edge_index: [[0, 1], [1, 0]]<br/>  edge_weights: [0.85, 0.85]<br/>  (co-failure bidirectional)<br/><br/>Relabeled to batch indices: [0, 1]"]

        B_FEATURES["<b>STEP 3: Prepare Features</b><br/>─────────────────────────────<br/><b>Semantic:</b> [2, 1536] embeddings<br/>  • MCA-1015: [0.12, -0.45, ...]<br/>  • MCA-101956: [-0.56, 0.23, ...]<br/><br/><b>Structural:</b> [2, 6] features<br/>  • MCA-1015: [45, 0.23, 0.15, 0.08, 3, 0]<br/>  • MCA-101956: [30, 0.08, 0.05, 0.02, 2, 0]"]
    end

    %% GAT Processing
    subgraph GAT ["🧠 GAT LAYERS (Structural Stream)"]
        direction TB

        GAT_INPUT["<b>INPUT</b><br/>───────────────<br/>x: [2, 6] structural features<br/>edge_index: [[0,1], [1,0]]<br/>edge_weights: [0.85, 0.85]"]

        GAT_L1["<b>GAT Layer 1 (Multi-head: 4 heads)</b><br/>────────────────────────────────────<br/><b>For each node:</b><br/>1. Self-attention: Compute attention scores<br/>   α(i,j) = attention_coef(h_i, h_j, e_ij)<br/><br/>2. Weighted aggregation:<br/>   h'_i = Σ α(i,j) × W × h_j<br/><br/>Example for MCA-1015 (node 0):<br/>  • Self: α(0,0) × W × h_0<br/>  • Neighbor: α(0,1) × 0.85 × W × h_1<br/>  • Result: [128] (32 per head × 4 heads)<br/><br/>Output: [2, 128]"]

        GAT_L2["<b>GAT Layer 2 (Single head)</b><br/>────────────────────────────────────<br/>Repeat attention mechanism:<br/>  • Input: [2, 128]<br/>  • Attention on refined features<br/>  • Edge weights still incorporated<br/><br/>Output: [2, 256] structural features"]

        GAT_OUTPUT["<b>STRUCTURAL FEATURES</b><br/>────────────────────────────<br/>[2, 256] graph-aware features<br/><br/>• MCA-1015: [0.34, -0.12, ..., 0.56]<br/>• MCA-101956: [-0.23, 0.45, ..., -0.34]<br/><br/><b>These capture:</b><br/>✓ Node's own history<br/>✓ Neighbor behavior patterns<br/>✓ Edge strength (co-failure, etc.)"]
    end

    %% Dual Stream Fusion
    subgraph FUSION ["🔗 DUAL-STREAM FUSION"]
        direction TB

        F_SEMANTIC["<b>Semantic Stream Output</b><br/>────────────────────────────<br/>[2, 256] semantic features<br/>from SBERT + MLP"]

        F_STRUCTURAL["<b>Structural Stream Output</b><br/>────────────────────────────<br/>[2, 256] graph features<br/>from GAT"]

        F_CROSS["<b>Cross-Attention Fusion</b><br/>────────────────────────────<br/>Query: Semantic features<br/>Key/Value: Structural features<br/><br/>Attention weights learned to<br/>combine both streams<br/><br/>Output: [2, 512] fused features"]

        F_CLASSIFIER["<b>Classifier</b><br/>────────────────────────────<br/>Linear(512 → 2)<br/>↓<br/>Logits: [2, 2]<br/>↓<br/>Softmax<br/>↓<br/>Probabilities: [2, 2]"]

        F_RESULT["<b>PREDICTIONS (Valid nodes)</b><br/>────────────────────────────<br/>MCA-1015: [0.28, 0.72]<br/>  P(Pass)=0.28, P(Fail)=0.72<br/><br/>MCA-101956: [0.88, 0.12]<br/>  P(Pass)=0.88, P(Fail)=0.12"]
    end

    %% Fill Orphans
    FILL["<b>STEP 4: Fill Orphan Predictions</b><br/>────────────────────────────────────<br/>full_probs = np.full((4, 2), 0.5)<br/>full_probs[[0,2]] = [[0.28,0.72], [0.88,0.12]]<br/><br/><b>Final batch predictions:</b><br/>1. MCA-1015: [0.28, 0.72] ← REAL<br/>2. MCA-NEW-123: [0.5, 0.5] ← DEFAULT<br/>3. MCA-101956: [0.88, 0.12] ← REAL<br/>4. MCA-NEW-456: [0.5, 0.5] ← DEFAULT"]

    %% Ranking
    RANK["<b>STEP 5: Ranking</b><br/>────────────────────────────<br/>Sort by P(Fail) descending:<br/><br/>1. MCA-1015 (0.72) 🔴<br/>2. MCA-NEW-123 (0.50)<br/>3. MCA-NEW-456 (0.50)<br/>4. MCA-101956 (0.12) 🟢"]

    %% Key Insight
    INSIGHT["<b>💡 KEY INSIGHTS</b><br/>──────────────────────────────────────────────<br/><b>1. GAT processes REAL graph during inference</b><br/>   • Same graph structure from training<br/>   • Extracts relevant subgraph per batch<br/>   • Edge weights preserved<br/><br/><b>2. Structural features are NOT simulated</b><br/>   • Known TCs: REAL historical stats<br/>   • Extracted from train.csv executions<br/>   • test_age, failure_rate, etc. are actual values<br/><br/><b>3. GAT impact is significant</b><br/>   • 76.7% of test.csv uses GAT<br/>   • Captures co-failure patterns<br/>   • Neighbor aggregation adds context<br/><br/><b>4. Orphans get conservative treatment</b><br/>   • [0.5, 0.5] = maximum uncertainty<br/>   • Ranked in middle by default<br/>   • Only 23.3% of samples"]

    %% Flow
    TITLE --> TRAIN_GRAPH
    TITLE --> BATCH

    TG_INFO --> TG_EXAMPLE
    TG_EXAMPLE --> B_SUBGRAPH

    B_INPUT --> B_FILTER
    B_FILTER --> B_SUBGRAPH
    B_SUBGRAPH --> B_FEATURES

    B_FEATURES --> GAT_INPUT
    GAT_INPUT --> GAT_L1
    GAT_L1 --> GAT_L2
    GAT_L2 --> GAT_OUTPUT

    GAT_OUTPUT --> F_STRUCTURAL
    B_FEATURES --> F_SEMANTIC

    F_SEMANTIC --> F_CROSS
    F_STRUCTURAL --> F_CROSS
    F_CROSS --> F_CLASSIFIER
    F_CLASSIFIER --> F_RESULT

    F_RESULT --> FILL
    FILL --> RANK
    RANK --> INSIGHT

    %% Styling
    classDef titleStyle fill:#1a237e,stroke:#fff,stroke-width:3px,color:#fff,font-weight:bold
    classDef graphStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    classDef batchStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef gatStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:3px,color:#000
    classDef fusionStyle fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    classDef resultStyle fill:#fff9c4,stroke:#f9a825,stroke-width:3px,color:#000
    classDef fillStyle fill:#ffccbc,stroke:#d84315,stroke-width:2px,color:#000
    classDef rankStyle fill:#c8e6c9,stroke:#388e3c,stroke-width:3px,color:#000
    classDef insightStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:4px,color:#000,font-weight:bold

    class TITLE titleStyle
    class TG_INFO,TG_EXAMPLE graphStyle
    class B_INPUT,B_FILTER,B_SUBGRAPH,B_FEATURES batchStyle
    class GAT_INPUT,GAT_L1,GAT_L2,GAT_OUTPUT gatStyle
    class F_SEMANTIC,F_STRUCTURAL,F_CROSS,F_CLASSIFIER fusionStyle
    class F_RESULT resultStyle
    class FILL fillStyle
    class RANK rankStyle
    class INSIGHT insightStyle

    style TRAIN_GRAPH fill:#f1f8f1,stroke:#388e3c,stroke-width:4px
    style BATCH fill:#e8f4fd,stroke:#1565c0,stroke-width:4px
    style GAT fill:#fce4ec,stroke:#880e4f,stroke-width:4px
    style FUSION fill:#fffde7,stroke:#f57f17,stroke-width:4px
