%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e1f5ff','primaryTextColor':'#000','primaryBorderColor':'#0277bd','lineColor':'#0277bd','secondaryColor':'#fff3e0','tertiaryColor':'#f1f8e9'}}}%%

graph TB
    %% Input Sources
    subgraph INPUT["📥 INPUT: Test Case Data"]
        TC_DATA["<b>Test Case Metadata</b><br/>─────────────────<br/>• tc_summary<br/>• tc_steps<br/>• commit_messages<br/>• diff"]
    end

    %% Text Preparation
    subgraph PREP["📝 TEXT PREPARATION"]
        TC_TEXT["<b>TC Text Construction</b><br/>─────────────────<br/>Summary: {tc_summary}<br/>Steps: {tc_steps}<br/><br/>Example:<br/>'Summary: Verify login<br/>Steps: 1. Open app<br/>2. Enter credentials<br/>3. Click OK'"]

        COMMIT_TEXT["<b>Commit Text Construction</b><br/>─────────────────<br/>Message: {commit_messages}<br/>Diff: {diff}<br/><br/>Example:<br/>'Message: Fix auth bug<br/>Diff: - if user == null<br/>+ if user != null'"]
    end

    %% SBERT Processing
    subgraph SBERT["🧠 SBERT MODEL (all-mpnet-base-v2)"]
        SBERT_INFO["<b>Sentence-BERT Encoder</b><br/>─────────────────<br/>• Microsoft MPNet architecture<br/>• Pre-trained on 1B+ sentence pairs<br/>• State-of-the-art semantic similarity<br/>• Captures meaning, not just keywords"]

        TC_EMBED["<b>TC Embedding</b><br/>─────────────────<br/>SBERT(TC Text)<br/>↓<br/>[768 dimensions]<br/><br/>Captures semantic meaning<br/>of test case"]

        COMMIT_EMBED["<b>Commit Embedding</b><br/>─────────────────<br/>SBERT(Commit Text)<br/>↓<br/>[768 dimensions]<br/><br/>Captures semantic meaning<br/>of code changes"]
    end

    %% Concatenation
    CONCAT["<b>📊 CONCATENATION</b><br/>─────────────────<br/>TC Embedding ⊕ Commit Embedding<br/>↓<br/>[1536 dimensions]<br/><br/>Rich semantic representation"]

    %% Semantic Stream Neural Network
    subgraph STREAM["🔮 SEMANTIC STREAM (Neural Network)"]
        direction TB

        LAYER1["<b>Layer 1: Feature Extraction</b><br/>─────────────────<br/>Linear(1536 → 512)<br/>BatchNorm1d(512)<br/>ReLU Activation<br/>Dropout(0.3)<br/><br/>[batch, 1536] → [batch, 512]"]

        LAYER2["<b>Layer 2: Feature Refinement</b><br/>─────────────────<br/>Linear(512 → 256)<br/>BatchNorm1d(256)<br/>ReLU Activation<br/>Dropout(0.3)<br/><br/>[batch, 512] → [batch, 256]"]

        OUTPUT["<b>✨ Semantic Features</b><br/>─────────────────<br/>[batch, 256]<br/><br/>High-level semantic patterns<br/>learned during training"]
    end

    %% What it learns
    LEARNING["<b>💡 WHAT IT LEARNS</b><br/>─────────────────<br/>Patterns like:<br/>• 'login', 'auth' → high failure risk<br/>• 'database', 'connection' → medium risk<br/>• 'UI', 'cosmetic' → low risk<br/><br/>✓ Works for NEW tests!<br/>✓ No historical data needed<br/>✓ Pure semantic understanding"]

    %% Key Advantage
    ADVANTAGE["<b>🎯 KEY ADVANTAGE</b><br/>─────────────────<br/>Brand new test cases<br/>(never seen in training)<br/>↓<br/>Still get meaningful predictions<br/>based on semantic similarity<br/>to known patterns"]

    %% Flow connections
    TC_DATA --> TC_TEXT
    TC_DATA --> COMMIT_TEXT

    TC_TEXT --> SBERT_INFO
    COMMIT_TEXT --> SBERT_INFO

    SBERT_INFO --> TC_EMBED
    SBERT_INFO --> COMMIT_EMBED

    TC_EMBED --> CONCAT
    COMMIT_EMBED --> CONCAT

    CONCAT --> LAYER1
    LAYER1 --> LAYER2
    LAYER2 --> OUTPUT

    OUTPUT --> LEARNING
    LEARNING --> ADVANTAGE

    %% Styling
    classDef inputStyle fill:#e1f5ff,stroke:#0277bd,stroke-width:3px,color:#000
    classDef prepStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef sbertStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000
    classDef concatStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:3px,color:#000
    classDef streamStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef outputStyle fill:#fff9c4,stroke:#f9a825,stroke-width:3px,color:#000
    classDef learnStyle fill:#e0f2f1,stroke:#00897b,stroke-width:2px,color:#000
    classDef advStyle fill:#ffebee,stroke:#d32f2f,stroke-width:3px,color:#000

    class TC_DATA inputStyle
    class TC_TEXT,COMMIT_TEXT prepStyle
    class SBERT_INFO,TC_EMBED,COMMIT_EMBED sbertStyle
    class CONCAT concatStyle
    class LAYER1,LAYER2 streamStyle
    class OUTPUT outputStyle
    class LEARNING learnStyle
    class ADVANTAGE advStyle
