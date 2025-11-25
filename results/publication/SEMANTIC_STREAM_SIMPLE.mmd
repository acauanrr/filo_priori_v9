%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#bbdefb','primaryTextColor':'#000','primaryBorderColor':'#1976d2','lineColor':'#1976d2','secondaryColor':'#c8e6c9','tertiaryColor':'#fff9c4'}}}%%

flowchart LR
    %% Question
    QUESTION["❓ <b>QUESTION</b><br/>'What is this test ABOUT?'"]

    %% Input Data
    subgraph INPUT [" "]
        direction TB
        TC["<b>Test Case</b><br/>📝<br/>summary<br/>steps"]
        COMMIT["<b>Commit</b><br/>💾<br/>message<br/>diff"]
    end

    %% SBERT Processing
    subgraph SBERT ["🧠 SBERT (all-mpnet-base-v2)"]
        direction TB
        ENCODE["<b>Text → Vector</b><br/>Semantic encoding"]
        TC_VEC["TC Vector<br/>[768]"]
        COMMIT_VEC["Commit Vector<br/>[768]"]
    end

    %% Concatenation
    MERGE["<b>⊕</b><br/>Concatenate<br/>[1536]"]

    %% Neural Network
    subgraph NN ["🔮 Neural Network (Semantic Stream)"]
        direction TB
        L1["Layer 1<br/>1536→512"]
        L2["Layer 2<br/>512→256"]
    end

    %% Output
    OUTPUT["<b>✨ Semantic Features</b><br/>[256]<br/><br/>Learned patterns:<br/>'login'→high risk<br/>'UI'→low risk"]

    %% Advantage
    ADVANTAGE["<b>🎯 Works for<br/>NEW tests!</b><br/><br/>No history needed<br/>Pure semantics"]

    %% Flow
    QUESTION --> INPUT
    TC --> ENCODE
    COMMIT --> ENCODE
    ENCODE --> TC_VEC
    ENCODE --> COMMIT_VEC
    TC_VEC --> MERGE
    COMMIT_VEC --> MERGE
    MERGE --> L1
    L1 --> L2
    L2 --> OUTPUT
    OUTPUT --> ADVANTAGE

    %% Styling
    classDef questionStyle fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000,font-weight:bold
    classDef inputStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef sbertStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:3px,color:#000
    classDef mergeStyle fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000,font-size:20px
    classDef nnStyle fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    classDef outputStyle fill:#fff9c4,stroke:#f9a825,stroke-width:3px,color:#000
    classDef advStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:#000,font-weight:bold

    class QUESTION questionStyle
    class TC,COMMIT inputStyle
    class ENCODE,TC_VEC,COMMIT_VEC sbertStyle
    class MERGE mergeStyle
    class L1,L2 nnStyle
    class OUTPUT outputStyle
    class ADVANTAGE advStyle

    style INPUT fill:#f5f5f5,stroke:#bdbdbd,stroke-width:2px
    style SBERT fill:#fce4ec,stroke:#880e4f,stroke-width:3px
    style NN fill:#fffde7,stroke:#f57f17,stroke-width:3px
