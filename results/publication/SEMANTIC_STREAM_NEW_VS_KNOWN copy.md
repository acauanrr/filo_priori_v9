%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e1f5ff','primaryTextColor':'#000','primaryBorderColor':'#0277bd','lineColor':'#0288d1','secondaryColor':'#fff3e0','tertiaryColor':'#f1f8e9'}}}%%

flowchart TB
    %% Title
    TITLE["<b>🔮 SEMANTIC STREAM</b><br/>'What is this test ABOUT?'<br/>────────────────"]

    %% Two parallel paths
    subgraph NEW ["🆕 BRAND NEW TEST CASE"]
        direction TB
        NEW_TC["<b>MCA-NEW-123</b><br/>❌ NOT in training data<br/>❌ NO historical features<br/>❌ NO graph connections"]

        NEW_TEXT["<b>Text Data</b><br/>✓ Summary: 'Verify login'<br/>✓ Steps: '1. Open app...'<br/>✓ Commit: 'Fix auth bug...'"]

        NEW_SBERT["<b>SBERT Encoding</b><br/>📊 [1536 dims]"]

        NEW_NN["<b>Neural Network</b><br/>1536→512→256"]

        NEW_OUT["<b>✨ Semantic Features</b><br/>[256]<br/><br/>Predictions based on<br/>semantic patterns:<br/>'login' → high risk"]
    end

    subgraph KNOWN ["✅ KNOWN TEST CASE"]
        direction TB
        KNOWN_TC["<b>MCA-1015</b><br/>✓ In training data<br/>✓ HAS historical features<br/>✓ HAS graph connections"]

        KNOWN_TEXT["<b>Text Data</b><br/>✓ Summary: 'Test database'<br/>✓ Steps: '1. Connect...'<br/>✓ Commit: 'Update schema...'"]

        KNOWN_SBERT["<b>SBERT Encoding</b><br/>📊 [1536 dims]"]

        KNOWN_NN["<b>Neural Network</b><br/>1536→512→256"]

        KNOWN_OUT["<b>✨ Semantic Features</b><br/>[256]<br/><br/>Predictions based on<br/>semantic patterns:<br/>'database' → medium risk"]
    end

    %% Key Point
    KEYPOINT["<b>🎯 KEY INSIGHT</b><br/>────────────────────────────────────<br/><b>SAME PROCESS FOR BOTH!</b><br/>────────────────────────────────────<br/>✓ Text always available (summary, steps, commit)<br/>✓ SBERT generates embeddings (1536 dims)<br/>✓ Neural network extracts semantic patterns<br/>✓ Works WITHOUT historical data<br/>✓ Captures meaning, not just keywords<br/><br/><b>NEW tests get meaningful predictions!</b>"]

    %% Model Info
    MODEL["<b>🧠 SBERT Model</b><br/>─────────────────<br/>all-mpnet-base-v2<br/>Pre-trained on 1B+ pairs<br/>State-of-the-art semantics"]

    %% Flow
    TITLE --> NEW_TC
    TITLE --> KNOWN_TC

    NEW_TC --> NEW_TEXT
    NEW_TEXT --> NEW_SBERT
    NEW_SBERT --> NEW_NN
    NEW_NN --> NEW_OUT

    KNOWN_TC --> KNOWN_TEXT
    KNOWN_TEXT --> KNOWN_SBERT
    KNOWN_SBERT --> KNOWN_NN
    KNOWN_NN --> KNOWN_OUT

    NEW_OUT --> KEYPOINT
    KNOWN_OUT --> KEYPOINT

    MODEL -.-> NEW_SBERT
    MODEL -.-> KNOWN_SBERT

    %% Styling
    classDef titleStyle fill:#1a237e,stroke:#fff,stroke-width:3px,color:#fff,font-weight:bold,font-size:16px
    classDef newTcStyle fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000,font-weight:bold
    classDef knownTcStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef textStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef sbertStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef nnStyle fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    classDef outStyle fill:#fff9c4,stroke:#f9a825,stroke-width:3px,color:#000
    classDef keyStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:4px,color:#000,font-weight:bold
    classDef modelStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000

    class TITLE titleStyle
    class NEW_TC newTcStyle
    class KNOWN_TC knownTcStyle
    class NEW_TEXT,KNOWN_TEXT textStyle
    class NEW_SBERT,KNOWN_SBERT sbertStyle
    class NEW_NN,KNOWN_NN nnStyle
    class NEW_OUT,KNOWN_OUT outStyle
    class KEYPOINT keyStyle
    class MODEL modelStyle

    style NEW fill:#fff5f5,stroke:#d32f2f,stroke-width:3px,stroke-dasharray: 5 5
    style KNOWN fill:#f1f8f1,stroke:#388e3c,stroke-width:3px,stroke-dasharray: 5 5
