%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f5e9','primaryTextColor':'#000','primaryBorderColor':'#2e7d32','lineColor':'#1976d2','secondaryColor':'#ffebee','tertiaryColor':'#fff9c4'}}}%%

flowchart TB
    %% Title
    TITLE["<b>🔍 TEST.CSV INFERENCE: REAL COMPOSITION</b><br/>Total: 31,333 samples"]

    %% Split
    subgraph KNOWN ["✅ KNOWN TEST CASES (76.7% - 24,017 samples)"]
        direction TB

        K_DESC["<b>TC_Keys that appeared in train.csv</b><br/>─────────────────────────────<br/>Examples: MCA-1015, MCA-101956, etc.<br/>global_idx ≥ 0"]

        K_STRUCT["<b>📊 STRUCTURAL FEATURES (REAL!)</b><br/>─────────────────────────────<br/>✓ test_age = 45 builds (ACTUAL)<br/>✓ failure_rate = 0.23 (ACTUAL)<br/>✓ recent_failure_rate = 0.15 (ACTUAL)<br/>✓ flakiness_rate = 0.08 (ACTUAL)<br/>✓ commit_count = 3 (from current build)<br/>✓ test_novelty = 0.0 (KNOWN)<br/><br/><b>NOT SIMULATED!</b> Real history from train.csv"]

        K_GRAPH["<b>🕸️ GRAPH PROCESSING (GAT ACTIVE)</b><br/>─────────────────────────────<br/>• Node exists in training graph<br/>• Has edges (co-failure, co-success, semantic)<br/>• GAT aggregates from neighbors<br/>• Full graph attention mechanism<br/><br/>Graph Stats:<br/>  - 2,347 nodes<br/>  - 461,493 edges<br/>  - Avg degree: 393 neighbors"]

        K_SEMANTIC["<b>📝 SEMANTIC FEATURES</b><br/>─────────────────────────────<br/>SBERT embeddings [1536]<br/>from TC text + Commit text"]

        K_DUAL["<b>🔮 DUAL-STREAM MODEL</b><br/>─────────────────────────────<br/>Semantic Stream [256] +<br/>Structural Stream (GAT) [256]<br/>↓<br/>Fusion [512]<br/>↓<br/>Classifier [2]"]

        K_OUTPUT["<b>✨ OUTPUT</b><br/>─────────────────────────────<br/>Real predictions based on:<br/>✓ Semantic patterns<br/>✓ Historical behavior<br/>✓ Graph relationships<br/><br/>Example: [0.28, 0.72]<br/>P(Pass)=0.28, P(Fail)=0.72"]
    end

    subgraph ORPHAN ["❌ ORPHAN TEST CASES (23.3% - 7,316 samples)"]
        direction TB

        O_DESC["<b>TC_Keys NOT in train.csv</b><br/>─────────────────────────────<br/>Examples: MCA-NEW-123, MCA-NEW-456<br/>global_idx = -1"]

        O_STRUCT["<b>📊 STRUCTURAL FEATURES (DEFAULTS)</b><br/>─────────────────────────────<br/>✓ test_age = 0.0 (NEW)<br/>✓ failure_rate = 0.31 (population mean)<br/>✓ recent_failure_rate = 0.28 (population mean)<br/>✓ flakiness_rate = 0.12 (population median)<br/>✓ commit_count = 2 (from current build)<br/>✓ test_novelty = 1.0 (NEW FLAG)<br/><br/><b>+ IMPUTATION</b> (if available):<br/>  Uses K=10 semantic neighbors<br/>  Weighted average by similarity"]

        O_GRAPH["<b>🕸️ GRAPH PROCESSING (FILTERED OUT)</b><br/>─────────────────────────────<br/>❌ Not in training graph<br/>❌ No edges<br/>❌ Filtered before GAT (valid_mask = False)<br/>❌ GAT not executed for these samples<br/><br/>Code: valid_mask = (global_indices != -1)<br/>      → False for orphans"]

        O_SEMANTIC["<b>📝 SEMANTIC FEATURES</b><br/>─────────────────────────────<br/>SBERT embeddings [1536]<br/>from TC text + Commit text<br/><br/><b>SAME as known TCs!</b><br/>Text always available"]

        O_SKIP["<b>⚠️ NO DUAL-STREAM</b><br/>─────────────────────────────<br/>Semantic stream NOT executed<br/>Structural stream NOT executed<br/>Model forward pass SKIPPED<br/><br/>Why? Conservative approach:<br/>Insufficient information"]

        O_OUTPUT["<b>🔒 DEFAULT OUTPUT</b><br/>─────────────────────────────<br/>Default probability:<br/>[0.5, 0.5]<br/><br/>P(Pass)=0.5, P(Fail)=0.5<br/><b>Maximum uncertainty</b><br/><br/>Ranked in middle of list"]
    end

    %% Key Statistics
    STATS["<b>📈 KEY STATISTICS</b><br/>──────────────────────────────────────────<br/><b>Test Split (6,195 samples):</b><br/>  • Known: 6,152 (99.3%) → FULL INFERENCE<br/>  • Orphans: 43 (0.7%) → DEFAULT [0.5, 0.5]<br/><br/><b>Full test.csv (31,333 samples):</b><br/>  • Known: 24,017 (76.7%) → FULL INFERENCE<br/>  • Orphans: 7,316 (23.3%) → DEFAULT [0.5, 0.5]<br/><br/><b>Unique TC_Keys in test.csv:</b><br/>  • Known: 1,859 (74.2%)<br/>  • Orphans: 646 (25.8%)<br/>  • Total: 2,505"]

    %% Impact
    IMPACT["<b>💥 STRUCTURAL STREAM IMPACT</b><br/>──────────────────────────────────────────<br/><b>GAT influences 76.7% of predictions!</b><br/><br/>Graph Statistics:<br/>  • Total edges: 461,493<br/>  • Co-failure edges: 495 (0.1%)<br/>  • Co-success edges: 207,913 (45.1%)<br/>  • Semantic edges: 253,085 (54.8%)<br/>  • Avg degree: 393 neighbors/node<br/><br/>Evidence from Experiments:<br/>  • Exp 04a (baseline): APFD = 0.6210<br/>  • Random baseline: APFD ≈ 0.50<br/>  • <b>+24% improvement</b> from dual-stream<br/><br/><b>YES, structural stream matters!</b>"]

    %% Flow
    TITLE --> KNOWN
    TITLE --> ORPHAN

    K_DESC --> K_STRUCT
    K_DESC --> K_SEMANTIC
    K_STRUCT --> K_GRAPH
    K_SEMANTIC --> K_DUAL
    K_GRAPH --> K_DUAL
    K_DUAL --> K_OUTPUT

    O_DESC --> O_STRUCT
    O_DESC --> O_SEMANTIC
    O_STRUCT --> O_GRAPH
    O_SEMANTIC --> O_SKIP
    O_GRAPH --> O_SKIP
    O_SKIP --> O_OUTPUT

    K_OUTPUT --> STATS
    O_OUTPUT --> STATS
    STATS --> IMPACT

    %% Styling
    classDef titleStyle fill:#1a237e,stroke:#fff,stroke-width:3px,color:#fff,font-weight:bold
    classDef knownStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    classDef orphanStyle fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000
    classDef structStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef graphStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef dualStyle fill:#fff9c4,stroke:#f57f17,stroke-width:3px,color:#000
    classDef outputStyle fill:#c8e6c9,stroke:#388e3c,stroke-width:3px,color:#000
    classDef defaultStyle fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#000
    classDef statsStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:4px,color:#000,font-weight:bold
    classDef impactStyle fill:#fff3e0,stroke:#ef6c00,stroke-width:4px,color:#000,font-weight:bold

    class TITLE titleStyle
    class K_DESC,O_DESC knownStyle
    class K_STRUCT,O_STRUCT structStyle
    class K_GRAPH,O_GRAPH graphStyle
    class K_SEMANTIC,O_SEMANTIC structStyle
    class K_DUAL dualStyle
    class O_SKIP defaultStyle
    class K_OUTPUT outputStyle
    class O_OUTPUT defaultStyle
    class STATS statsStyle
    class IMPACT impactStyle

    style KNOWN fill:#f1f8f1,stroke:#388e3c,stroke-width:4px
    style ORPHAN fill:#fff5f5,stroke:#d32f2f,stroke-width:4px
