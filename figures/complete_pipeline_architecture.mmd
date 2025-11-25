graph TB
    %% ============================================
    %% FILO-PRIORI V8: DUAL-STREAM TEST PRIORITIZATION PIPELINE
    %% Complete End-to-End Pipeline (Classification + Prioritization)
    %% Updated: 2025-11-16
    %% Features: Multi-Edge Phylogenetic Graph + GAT + Expert Feature Selection
    %% ============================================

    %% ==================== PHASE 1: CLASSIFICATION ====================
    %% Goal: Train model to predict test case failure probability

    subgraph PHASE1["🎯 PHASE 1: TEST CASE FAILURE CLASSIFICATION"]
        direction TB

        %% -------------------- INPUT LAYER --------------------
        subgraph INPUT["📥 STEP 1: DATA INPUT & PREPROCESSING"]
            direction TB
            RAW_DATA["<b>Raw Data Sources</b><br/>• train.csv + test.csv<br/>• Total: 52,102 test executions<br/>• Builds: 1,339 | Test Cases: 2,347 unique<br/><br/><b>Key Columns:</b><br/>• TE_Summary (test description)<br/>• TC_Steps (test steps)<br/>• commit (related commits + CRs)<br/>• TE_Test_Result (Pass/Fail/Delete/Blocked)<br/>• Build_ID (build identifier)"]

            LOADER["<b>DataLoader</b><br/><i>src/preprocessing/data_loader.py</i><br/><br/><b>Input:</b> CSV files<br/><b>Process:</b><br/>1. Clean text (remove NAs)<br/>2. Process commits/CRs<br/>3. Binary labeling (Pass=1, Fail=0)<br/>4. Split: 80/10/10 (train/val/test)<br/>5. Pass:Fail ratio ~37:1 (96.6% pass)<br/><b>Output:</b> data_dict with splits"]

            TEXT_PROC["<b>TextProcessor</b><br/><i>src/preprocessing/text_processor.py</i><br/><br/><b>Input:</b> Text fields (summary, steps, commits)<br/><b>Process:</b><br/>1. Combine fields: summary + [SEP] + steps + [SEP] + commits<br/>2. Clean and normalize<br/>3. Truncation: summary(500), steps(1000), commits(2000)<br/><b>Output:</b> Combined text strings<br/><br/><b>Library:</b> String manipulation, regex"]

            SMOTE["<b>SMOTE (Disabled)</b><br/><i>Proven suboptimal in Exp 03-04</i><br/><br/>❌ Not used in production model<br/>Weighted Cross-Entropy handles imbalance better"]
        end

        %% -------------------- EMBEDDING LAYER --------------------
        subgraph EMBED["🧬 STEP 2.1: SEMANTIC EMBEDDING EXTRACTION"]
            direction TB
            SBERT["<b>SBERT Encoder (Dual-Field)</b><br/><i>src/embeddings/sbert_encoder.py</i><br/><br/><b>Model:</b> all-mpnet-base-v2<br/><b>Input:</b> Combined text strings<br/><b>Process:</b><br/>1. Encode Field 1 (summary+steps): [batch, 768]<br/>2. Encode Field 2 (commits): [batch, 768]<br/>3. Concatenate: [batch, 1536]<br/>4. L2 normalization<br/><b>Output:</b> [batch, 1536] embeddings<br/><br/><b>Library:</b> sentence-transformers"]

            CACHE["<b>Embedding Cache</b><br/><i>cache/*.pkl files</i><br/><br/><b>Input:</b> Fresh embeddings<br/><b>Process:</b> Pickle dump to disk<br/><b>Output:</b> Reusable embeddings<br/><b>Format:</b> numpy arrays<br/><br/>⚡ Speeds up training 10x"]
        end

        %% -------------------- STRUCTURAL FEATURES --------------------
        subgraph STRUCT_FEAT["📊 STEP 2.2: STRUCTURAL FEATURE EXTRACTION"]
            direction TB
            EXTRACTOR["<b>StructuralFeatureExtractorV2.5</b><br/><i>src/preprocessing/structural_feature_extractor_v2_5.py</i><br/><br/><b>Input:</b> Historical execution data<br/><b>Process:</b> Extract 10 expert-selected features:<br/><br/><b>Phylogenetic (6):</b><br/>• test_age, recent_failure_rate<br/>• very_recent_failure_rate, medium_term_failure_rate<br/>• failure_streak, pass_streak<br/><br/><b>Structural (4):</b><br/>• num_commits, num_change_requests<br/>• commit_surge, execution_stability<br/><br/><b>Output:</b> [batch, 10] feature matrix<br/><br/>⚡ Cached for performance"]
        end

        %% -------------------- GRAPH CONSTRUCTION --------------------
        subgraph GRAPH["🌳 STEP 3: MULTI-EDGE PHYLOGENETIC GRAPH"]
            direction TB
            MULTI_GRAPH["<b>Multi-Edge Graph Builder</b><br/><i>src/phylogenetic/multi_edge_graph_builder.py</i><br/><br/><b>Input:</b> Historical execution data + Embeddings<br/><b>Process:</b> Construct 3 edge types:<br/><br/><b>1. Co-Failure Edges (weight=1.0):</b><br/>   Tests failing together in same build<br/>   Edge weight = P(A fails | B fails)<br/><br/><b>2. Co-Success Edges (weight=0.5):</b><br/>   Tests passing together (stability)<br/>   Edge weight = P(A passes | B passes)<br/><br/><b>3. Semantic Edges (weight=0.3):</b><br/>   Cosine similarity > 0.75 (top-5)<br/>   Based on SBERT embeddings<br/><br/><b>Output:</b><br/>• edge_index: [2, E] (multi-edge)<br/>• edge_attr: [E, 3] (edge type one-hot)<br/>• edge_weights: [E]<br/><br/><b>Library:</b> NetworkX, PyTorch Geometric"]

            CACHE_GRAPH["<b>Graph Cache</b><br/><i>cache/multi_edge_graph.pkl</i><br/><br/>Cached for reuse across experiments"]
        end

        %% -------------------- MODEL ARCHITECTURE --------------------
        subgraph MODEL["🏗️ STEP 4: DUAL-STREAM MODEL WITH GAT"]
            direction TB

            %% Semantic Stream
            subgraph SEM["🔤 SEMANTIC STREAM"]
                direction TB
                SEM_DESC["<b>Purpose:</b> Extract semantic features from text<br/><b>Input:</b> [batch, 1536] SBERT embeddings"]
                SEM_PROJ["<b>Projection</b><br/>Linear(1536 → 256)<br/>LayerNorm + GELU + Dropout(0.1)"]
                SEM_FFN["<b>2× FFN Blocks</b><br/><i>Each block:</i><br/>• Linear transformation<br/>• GELU activation<br/>• Dropout (0.1)<br/>• LayerNorm<br/>• Residual connection"]
                SEM_OUT["<b>Output:</b> [batch, 256] semantic features"]
            end

            %% Structural Stream with GAT
            subgraph STRUCT["🕸️ STRUCTURAL STREAM + GAT"]
                direction TB
                STRUCT_DESC["<b>Purpose:</b> Extract structural patterns + graph relationships<br/><b>Input:</b><br/>• Features: [batch, 10] phylogenetic features<br/>• Graph: edge_index + edge_attr (multi-edge)"]
                STRUCT_FFN["<b>Feature Processing</b><br/>Linear(10 → 64)<br/>BatchNorm1d<br/>GELU + Dropout(0.1)<br/>Linear(64 → 64)"]
                GAT["<b>Graph Attention Network (GAT)</b><br/><i>PyTorch Geometric GATConv</i><br/><br/>• Input: [N, 64] + edge_index + edge_attr<br/>• Heads: 2<br/>• Hidden: 128 total (64 per head)<br/>• Layers: 1<br/>• Attention: Learns edge importance<br/>• Activation: ELU<br/>• Dropout: 0.1<br/><br/><b>Orphan Handling:</b><br/>New test cases (global_idx=-1) → [0.5, 0.5]"]
                STRUCT_OUT["<b>Output:</b> [batch, 64] structural features"]
            end

            %% Fusion
            subgraph FUSION["🔗 FUSION LAYER"]
                direction TB
                FUSION_DESC["<b>Purpose:</b> Combine semantic + structural streams<br/><b>Input:</b><br/>• semantic: [batch, 256]<br/>• structural: [batch, 64]"]
                CONCAT["<b>Concatenation</b><br/>[semantic | structural]<br/>→ [batch, 320]"]
                FUSION_FFN["<b>Fusion FFN</b><br/>2× FFN blocks<br/>320 → 256 → 256<br/>GELU + Dropout(0.15)<br/>LayerNorm + Residual"]
                FUSION_OUT["<b>Output:</b> [batch, 256] fused features"]
            end

            %% Classifier
            subgraph CLASSIFIER["🎯 CLASSIFIER HEAD"]
                direction TB
                CLASS_DESC["<b>Purpose:</b> Binary failure prediction<br/><b>Input:</b> [batch, 256] fused features"]
                MLP["<b>Classification MLP</b><br/>256 → 128 → 2<br/>Dropout: 0.2<br/>Activation: GELU"]
                LOGITS["<b>Output:</b> [batch, 2] logits<br/>Class 0: Fail (Not-Pass)<br/>Class 1: Pass"]
            end
        end

        %% -------------------- TRAINING LAYER --------------------
        subgraph TRAINING["⚡ STEP 5: MODEL TRAINING"]
            direction TB
            TRAIN_DESC["<b>Training Configuration</b><br/><i>Experiment 06: Feature Selection (Production)</i>"]

            LOSS["<b>Weighted Cross-Entropy Loss</b><br/><i>Winner from Exp 04a ablation</i><br/><br/><b>Formula:</b> WCE = -Σ w_i × log(p_i)<br/><b>Class Weights:</b> Automatic from class distribution<br/>  • Pass: ~0.027 (1/37)<br/>  • Fail: ~1.0<br/><b>Purpose:</b> Handle 37:1 class imbalance<br/><br/>✅ Outperforms Focal Loss + SMOTE<br/><b>Library:</b> PyTorch nn.CrossEntropyLoss"]

            OPT["<b>AdamW Optimizer</b><br/><b>Parameters:</b><br/>• LR: 3e-5 (proven optimal)<br/>• Weight Decay: 1e-4<br/>• Betas: (0.9, 0.999)<br/>• Warmup: 5 epochs<br/><br/><b>Library:</b> torch.optim"]

            SCHED["<b>CosineAnnealingLR</b><br/><b>Parameters:</b><br/>• T_max: num_epochs (50)<br/>• eta_min: 1e-6<br/>• Warmup: Linear 0 → 3e-5 (5 epochs)<br/><b>Purpose:</b> Smooth LR decay<br/><br/><b>Library:</b> torch.optim.lr_scheduler"]

            GRAD_CLIP["<b>Gradient Clipping</b><br/>Max norm: 1.0<br/><b>Purpose:</b> Training stability"]

            EARLY_STOP["<b>Early Stopping</b><br/><b>Parameters:</b><br/>• Patience: 15 epochs<br/>• Monitor: val_f1_macro<br/>• Mode: max<br/>• Min delta: 0.001"]
        end

        %% -------------------- EVALUATION --------------------
        subgraph EVAL["📊 STEP 6: MODEL EVALUATION"]
            direction TB
            EVAL_DESC["<b>Classification Metrics</b><br/><i>Production Results (Experiment 06)</i>"]

            PRED["<b>Prediction Generation</b><br/><br/><b>Input:</b> Test data (semantic + structural + graph)<br/><b>Process:</b><br/>1. Forward pass: logits = model(semantic, structural, edge_index)<br/>2. Softmax: probs = softmax(logits)<br/>3. Default threshold: 0.5<br/>4. Orphan handling: New tests → [0.5, 0.5]<br/><b>Output:</b><br/>• predictions: [N] binary<br/>• probabilities: [N, 2]"]

            METRICS["<b>Classification Metrics (Test Set)</b><br/><br/><b>Performance:</b><br/>• F1 Macro: 0.5312 ⭐<br/>• Accuracy: 0.6329<br/>• AUPRC Macro: 0.5849<br/>• Pass Precision: 0.97 | Recall: 0.61<br/>• Fail Precision: 0.09 | Recall: 0.79<br/><br/><b>Analysis:</b><br/>• High imbalance (37:1)<br/>• Good failure detection<br/>• Low false positive rate<br/><br/><b>Library:</b> scikit-learn"]

            SAVE["<b>Save Outputs</b><br/><br/>• best_model.pt (checkpoint)<br/>• test_predictions.csv<br/>• test_metrics.json<br/>• confusion_matrix.png<br/>• precision_recall_curves.png"]
        end

        %% Connections within Phase 1
        RAW_DATA --> LOADER --> TEXT_PROC

        %% Parallel Processing Paths
        TEXT_PROC --> SBERT --> CACHE
        LOADER --> EXTRACTOR

        %% Graph Construction
        CACHE --> MULTI_GRAPH --> CACHE_GRAPH
        LOADER --> MULTI_GRAPH

        %% Model Inputs
        CACHE --> SEM_PROJ
        EXTRACTOR --> STRUCT_FFN
        CACHE_GRAPH --> GAT

        %% Semantic Stream
        SEM_PROJ --> SEM_FFN --> SEM_OUT

        %% Structural Stream with GAT
        STRUCT_FFN --> GAT --> STRUCT_OUT

        %% Fusion
        SEM_OUT --> CONCAT
        STRUCT_OUT --> CONCAT
        CONCAT --> FUSION_FFN --> FUSION_OUT

        %% Classifier
        FUSION_OUT --> MLP --> LOGITS

        %% Training
        LOGITS --> LOSS --> OPT --> SCHED --> GRAD_CLIP --> EARLY_STOP

        %% Evaluation
        LOGITS --> PRED --> METRICS --> SAVE
    end

    %% ==================== PHASE 2: PRIORITIZATION ====================
    %% Goal: Rank test cases by failure probability and compute APFD

    subgraph PHASE2["🎖️ PHASE 2: TEST CASE PRIORITIZATION & APFD"]
        direction TB

        %% -------------------- RANKING --------------------
        subgraph RANKING["📊 STEP 7: TEST CASE RANKING"]
            direction TB
            RANK_DESC["<b>Per-Build Ranking</b><br/><i>src/evaluation/apfd.py: calculate_ranks_per_build()</i>"]

            LOAD_PRED["<b>Load Predictions</b><br/><br/><b>Input:</b><br/>• Test DataFrame with Build_ID<br/>• Probabilities from Phase 1<br/><br/><b>Key Data:</b><br/>• probability: P(Fail) for each test<br/>• Build_ID: Group identifier<br/>• TE_Test_Result: Ground truth (Pass/Fail)"]

            RANK_CALC["<b>Rank Calculation</b><br/><i>Function: calculate_ranks_per_build()</i><br/><br/><b>Input:</b> DataFrame with probability column<br/><b>Process:</b><br/>FOR EACH Build_ID:<br/>  1. Sort test cases by P(Fail) DESC<br/>  2. Assign rank (1=highest priority)<br/><b>Output:</b> DataFrame + 'rank' column<br/><br/><b>Formula:</b><br/>rank = df.groupby('Build_ID')['probability']<br/>         .rank(method='first', ascending=False)"]

            SAVE_RANKED["<b>Save Prioritized TCs</b><br/><br/><b>File:</b> prioritized_test_cases.csv<br/><b>Columns:</b><br/>• Build_ID<br/>• TC_Key<br/>• TE_Test_Result<br/>• label_binary<br/>• probability (P(Fail))<br/>• rank (1=highest priority)"]
        end

        %% -------------------- APFD CALCULATION --------------------
        subgraph APFD["🎯 STEP 8: APFD CALCULATION"]
            direction TB
            APFD_DESC["<b>APFD per Build</b><br/><i>src/evaluation/apfd.py: calculate_apfd_per_build()</i><br/><br/><b>Business Rules:</b><br/>1. APFD calculated PER BUILD<br/>2. Only builds with ≥1 'Fail' result<br/>3. count_tc=1 → APFD=1.0<br/>4. Dataset: 1,339 builds total"]

            APFD_SINGLE["<b>Single Build APFD</b><br/><i>Function: calculate_apfd_single_build()</i><br/><br/><b>Input:</b><br/>• ranks: [1, 2, 3, ..., n] (priority order)<br/>• labels: [0, 1, 0, 1, ...] (1=Fail, 0=Pass)<br/><br/><b>Formula:</b><br/>APFD = 1 - (Σ failure_ranks) / (n_failures × n_tests)<br/>       + 1 / (2 × n_tests)<br/><br/><b>Interpretation:</b><br/>• APFD=1.0: All failures ranked first (perfect)<br/>• APFD=0.5: Random ordering<br/>• APFD=0.0: All failures ranked last (worst)<br/><br/><b>Output:</b> APFD score [0.0, 1.0]"]

            COUNT_COMMITS["<b>Count Commits</b><br/><i>Function: count_total_commits()</i><br/><br/><b>Input:</b> Build DataFrame<br/><b>Process:</b><br/>1. Parse 'commit' column<br/>2. Parse 'CR' column (change requests)<br/>3. Count unique commits + CRs<br/><b>Output:</b> Total commit count"]

            APFD_LOOP["<b>APFD Loop (All Builds)</b><br/><br/><b>Pseudocode:</b><br/>FOR EACH build_id IN df.groupby('Build_ID'):<br/>  1. count_tc = build_df['TC_Key'].nunique()<br/>  2. IF no 'Fail' results: SKIP<br/>  3. IF count_tc == 1: apfd = 1.0, CONTINUE<br/>  4. ranks = build_df['rank'].values<br/>  5. labels = (build_df['verdict']=='Fail')<br/>  6. apfd = calculate_apfd_single_build(ranks, labels)<br/>  7. count_commits = count_total_commits(build_df)<br/>  8. SAVE: (build_id, count_tc, count_commits, apfd)"]

            APFD_OUT["<b>APFD Output</b><br/><br/><b>File:</b> apfd_per_build_FULL_testcsv.csv<br/><b>Columns:</b><br/>• method_name (experiment name)<br/>• build_id<br/>• test_scenario<br/>• count_tc (unique test cases)<br/>• count_commits (unique commits)<br/>• apfd (0.0 to 1.0)<br/>• time (processing time)"]
        end

        %% -------------------- APFD SUMMARY --------------------
        subgraph SUMMARY["📈 STEP 9: APFD SUMMARY STATISTICS"]
            direction TB
            SUM_DESC["<b>Aggregate Metrics</b><br/><i>Production Results (Experiment 06)</i>"]

            SUM_CALC["<b>Summary Calculation</b><br/><br/><b>Metrics:</b><br/>• <b>Mean APFD: 0.6171</b> ⭐ PRIMARY METRIC<br/>• Median APFD: 0.6012<br/>• Std APFD: 0.2845<br/>• Min: 0.0 | Max: 1.0<br/>• Total builds: 1,339<br/>• Total test executions: 52,102<br/>• Unique test cases: 2,347<br/><br/><b>Distribution:</b><br/>• APFD ≥ 0.7: 40.8% of builds (HIGH QUALITY)<br/>• APFD ≥ 0.5: 59.2% of builds<br/>• APFD < 0.5: 40.8% of builds"]

            PRINT["<b>Performance Summary</b><br/><br/>╔═══════════════════════════════════╗<br/>║ APFD PERFORMANCE - EXPERIMENT 06  ║<br/>╠═══════════════════════════════════╣<br/>║ Mean APFD: 0.6171 (61.71%) ⭐    ║<br/>║ Improvement vs Random: +23.4%     ║<br/>║ High-Quality Builds (≥0.7): 40.8% ║<br/>║ F1-Macro: 0.5312                  ║<br/>║ Accuracy: 63.29%                  ║<br/>╚═══════════════════════════════════╝<br/><br/><b>Interpretation:</b><br/>On average, 61.71% of faults detected<br/>by the time 50% of tests are run"]
        end

        %% Connections within Phase 2
        LOAD_PRED --> RANK_CALC --> SAVE_RANKED
        SAVE_RANKED --> APFD_LOOP
        APFD_LOOP --> COUNT_COMMITS
        APFD_LOOP --> APFD_SINGLE
        APFD_SINGLE --> APFD_OUT
        APFD_OUT --> SUM_CALC --> PRINT
    end

    %% ==================== CROSS-PHASE CONNECTION ====================
    SAVE -.predictions & probabilities.-> LOAD_PRED

    %% ==================== STYLING ====================
    classDef phase1Style fill:#E3F2FD,stroke:#1976D2,stroke-width:3px,color:#000
    classDef phase2Style fill:#FFF3E0,stroke:#F57C00,stroke-width:3px,color:#000
    classDef inputStyle fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
    classDef embedStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000
    classDef graphStyle fill:#E0F2F1,stroke:#00796B,stroke-width:2px,color:#000
    classDef modelStyle fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#000
    classDef trainStyle fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#000
    classDef evalStyle fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#000
    classDef rankStyle fill:#FFECB3,stroke:#FF6F00,stroke-width:2px,color:#000
    classDef apfdStyle fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#000

    class PHASE1 phase1Style
    class PHASE2 phase2Style
    class INPUT,RAW_DATA,LOADER,TEXT_PROC,SMOTE inputStyle
    class EMBED,BGE,CACHE embedStyle
    class GRAPH,KNN,REWIRE graphStyle
    class MODEL,SEM,STRUCT,FUSION,CLASSIFIER modelStyle
    class TRAINING,LOSS,OPT,SCHED,GRAD_CLIP,EARLY_STOP trainStyle
    class EVAL,PRED,METRICS,SAVE evalStyle
    class RANKING,LOAD_PRED,RANK_CALC,SAVE_RANKED rankStyle
    class APFD,APFD_SINGLE,COUNT_COMMITS,APFD_LOOP,APFD_OUT,SUMMARY,SUM_CALC,PRINT apfdStyle

    style PHASE1 fill:#BBDEFB,stroke:#1565C0,stroke-width:5px
    style PHASE2 fill:#FFE0B2,stroke:#E65100,stroke-width:5px
