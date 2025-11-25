# STEP 2.2: Implementation Details & Technical Flow

## Complete Data Flow: From Raw Data to Predictions

```mermaid
flowchart TB
    subgraph raw_data["📁 RAW DATA"]
        train_csv["train.csv<br/>~69K records"]
        test_csv["test.csv<br/>~31K records"]

        columns["Required Columns:<br/>• Build_ID<br/>• TC_Key<br/>• TE_Test_Result<br/>• TE_Summary<br/>• TC_Steps<br/>• commit<br/>• Build_Test_Start_Date"]
    end

    subgraph step1["STEP 1: Feature Extraction"]
        direction TB

        subgraph semantic["🔵 Semantic Pipeline"]
            text_combine["Combine 3 fields:<br/>TE_Summary +<br/>TC_Steps +<br/>commit"]
            bge_encode["BGE Encoder<br/>encode_dataset()"]
            sem_cache["Cache:<br/>embeddings/*.npy"]
            sem_tensor["Tensor[N, 1024]<br/>dtype: float32"]

            text_combine --> bge_encode --> sem_cache --> sem_tensor
        end

        subgraph structural["🟢 Structural Pipeline"]
            extractor_init["StructuralFeatureExtractor<br/>recent_window=5"]
            extractor_fit["fit(df_train)<br/>Learn history"]
            extractor_transform["transform(df)<br/>Extract 6 features"]
            struct_cache["Cache:<br/>structural_features.pkl"]
            struct_tensor["Tensor[N, 6]<br/>dtype: float32"]

            extractor_init --> extractor_fit --> extractor_transform --> struct_cache --> struct_tensor
        end

        subgraph graph_build["🌳 Graph Pipeline"]
            phylo_init["PhylogeneticGraphBuilder<br/>type='co_failure'"]
            phylo_build["Build graph from<br/>df_train"]
            graph_cache["Cache:<br/>phylogenetic_graph.pkl"]
            graph_obj["NetworkX Graph<br/>(Optional for V8)"]

            phylo_init --> phylo_build --> graph_cache --> graph_obj
        end
    end

    subgraph step2["STEP 2: Model Creation"]
        direction TB

        config_load["Load config:<br/>experiment_v8_baseline.yaml"]

        model_create["create_model_v8(config)"]

        subgraph model_components["Model Components"]
            direction LR
            sem_stream_comp["SemanticStream<br/>(1024→256)"]
            struct_stream_comp["StructuralStreamV8<br/>(6→256)"]
            fusion_comp["CrossAttentionFusion<br/>(512)"]
            classifier_comp["Classifier<br/>(512→2)"]
        end

        device_move["Move to device<br/>(cuda/cpu)"]

        config_load --> model_create --> model_components --> device_move
    end

    subgraph step3["STEP 3: Training"]
        direction TB

        dataloader["DataLoader<br/>batch_size=32"]

        subgraph epoch_loop["Epoch Loop (1-40)"]
            direction TB

            subgraph batch_process["Batch Processing"]
                direction LR
                get_batch["Get batch:<br/>sem[32,1024]<br/>struct[32,6]<br/>labels[32]"]
                forward["Forward:<br/>model(sem, struct)"]
                loss_comp["Focal Loss<br/>alpha=[0.15,0.85]<br/>gamma=2.0"]
                backward["loss.backward()"]
                grad_clip["Clip gradients<br/>max_norm=1.0"]
                optim_step["optimizer.step()"]

                get_batch --> forward --> loss_comp --> backward --> grad_clip --> optim_step
            end

            val_eval["Validation"]
            early_check["Early Stopping<br/>patience=12"]
            save_best["Save best model<br/>if val_f1 improved"]

            batch_process --> val_eval --> early_check --> save_best
        end

        dataloader --> epoch_loop
    end

    subgraph step4["STEP 4: Evaluation"]
        direction TB

        load_best["Load best model"]
        test_eval["Test evaluation"]

        subgraph metrics["Metrics Computation"]
            direction LR
            classification["Classification:<br/>• F1 Macro<br/>• Accuracy<br/>• Precision<br/>• Recall<br/>• AUPRC"]
            apfd["APFD:<br/>• Per-build<br/>• Mean APFD<br/>• Test prioritization"]
        end

        save_results["Save results:<br/>• test_metrics.json<br/>• confusion_matrix.png<br/>• prioritized_tests.csv"]

        load_best --> test_eval --> metrics --> save_results
    end

    raw_data --> step1
    step1 --> step2
    step2 --> step3
    step3 --> step4

    style semantic fill:#e1f5ff,stroke:#0077cc,stroke-width:2px
    style structural fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style graph_build fill:#fff4e1,stroke:#cc7700,stroke-width:2px
```

## Phylogenetic Graph: Edge Weight Computation

```mermaid
flowchart TB
    subgraph co_failure_detail["Co-Failure Graph Weight Computation"]
        direction TB

        example_data["Example Data:<br/>Build_1: [TC_A fails, TC_B fails, TC_C passes]<br/>Build_2: [TC_A fails, TC_B passes]<br/>Build_3: [TC_A fails, TC_B fails]"]

        step1["Step 1: Count co-failures<br/>Edge (A,B): 2 builds<br/>(Build_1 and Build_3)"]

        step2["Step 2: Count total failures<br/>TC_A: 3 builds<br/>TC_B: 2 builds"]

        step3["Step 3: Compute conditional probability<br/>P(A fails | B fails) = co_failures / total_B_failures<br/>= 2 / 2 = 1.0<br/><br/>P(B fails | A fails) = co_failures / total_A_failures<br/>= 2 / 3 = 0.67"]

        step4["Step 4: Choose weight<br/>Use symmetric: max(P(A|B), P(B|A)) = 1.0<br/>Or directed: separate edges with different weights"]

        step5["Step 5: Apply thresholds<br/>if co_failures ≥ min_co_occurrences (e.g., 2) ✓<br/>if weight ≥ weight_threshold (e.g., 0.1) ✓<br/>Then: Add edge (A,B) with weight=1.0"]

        example_data --> step1 --> step2 --> step3 --> step4 --> step5
    end

    subgraph commit_dep_detail["Commit Dependency Graph Weight Computation"]
        direction TB

        commit_data["Example Data:<br/>Commit_1: [TC_A, TC_B, TC_C]<br/>Commit_2: [TC_A, TC_D]<br/>Commit_3: [TC_B, TC_C]"]

        c_step1["Step 1: Count shared commits<br/>Edge (A,B): 1 commit (Commit_1)<br/>Edge (B,C): 2 commits (Commit_1, Commit_3)"]

        c_step2["Step 2: Normalize by max<br/>max_shared = 2<br/>Weight(A,B) = 1/2 = 0.5<br/>Weight(B,C) = 2/2 = 1.0"]

        c_step3["Step 3: Apply thresholds<br/>Filter by weight_threshold"]

        commit_data --> c_step1 --> c_step2 --> c_step3
    end

    style co_failure_detail fill:#e1f5ff,stroke:#0077cc,stroke-width:2px
    style commit_dep_detail fill:#fff4e1,stroke:#cc7700,stroke-width:2px
```

## StructuralStreamV8: Architecture Deep Dive

```mermaid
flowchart TB
    subgraph input_layer["INPUT LAYER"]
        input["Input Tensor<br/>[batch=32, features=6]<br/><br/>Feature breakdown:<br/>[0] test_age<br/>[1] failure_rate<br/>[2] recent_failure_rate<br/>[3] flakiness_rate<br/>[4] commit_count<br/>[5] test_novelty"]
    end

    subgraph layer1["LAYER 1: Normalization + Expansion"]
        direction TB

        bn1["BatchNorm1d(6)<br/>Stabilizes small input<br/>Learned params: γ, β"]

        linear1["Linear(6 → 128)<br/>Weight: [6, 128]<br/>Bias: [128]"]

        gelu1["GELU Activation<br/>Non-linearity"]

        dropout1["Dropout(0.3)<br/>Regularization"]

        bn1 --> linear1 --> gelu1 --> dropout1

        note1["Purpose:<br/>Expand from 6 features<br/>to 128-dim space"]
    end

    subgraph layer2["LAYER 2: Normalization + Projection"]
        direction TB

        bn2["BatchNorm1d(128)<br/>Stabilizes activations"]

        linear2["Linear(128 → 256)<br/>Weight: [128, 256]<br/>Bias: [256]"]

        gelu2["GELU Activation"]

        dropout2["Dropout(0.3)"]

        bn2 --> linear2 --> gelu2 --> dropout2

        note2["Purpose:<br/>Project to 256-dim<br/>semantic-aligned space"]
    end

    subgraph output_layer["OUTPUT LAYER"]
        output["Output Tensor<br/>[batch=32, features=256]<br/><br/>Ready for fusion with<br/>semantic features [32, 256]"]
    end

    subgraph params["PARAMETERS"]
        direction TB

        total["Total Parameters:<br/>• BatchNorm1(6): 12<br/>• Linear1(6→128): 896<br/>• BatchNorm2(128): 256<br/>• Linear2(128→256): 33,024<br/><br/>TOTAL: ~34K parameters"]

        comparison["Comparison:<br/>V7 Structural: ~260K params<br/>V8 Structural: ~34K params<br/><br/>87% reduction!"]
    end

    input_layer --> layer1 --> layer2 --> output_layer
    params -.->|"Info"| layer1

    style layer1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style layer2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style note1 fill:#fff3cd,stroke:#856404
    style note2 fill:#fff3cd,stroke:#856404
```

## Forward Pass: Detailed Execution Flow

```mermaid
sequenceDiagram
    participant Input as Input Data
    participant SemStream as SemanticStream
    participant StructStream as StructuralStreamV8
    participant Fusion as CrossAttentionFusion
    participant Classifier as Classifier
    participant Output as Output

    Note over Input,Output: FORWARD PASS for batch_size=32

    Input->>SemStream: semantic_input [32, 1024]
    activate SemStream

    SemStream->>SemStream: LayerNorm([32, 1024])
    SemStream->>SemStream: FFN1: Linear(1024→512)
    SemStream->>SemStream: GELU + Dropout
    SemStream->>SemStream: FFN2: Linear(512→256)
    SemStream->>SemStream: GELU + Dropout

    SemStream-->>Fusion: semantic_features [32, 256]
    deactivate SemStream

    Input->>StructStream: structural_input [32, 6]
    activate StructStream

    StructStream->>StructStream: BatchNorm1d([32, 6])
    StructStream->>StructStream: Linear(6→128)
    StructStream->>StructStream: GELU + Dropout
    StructStream->>StructStream: BatchNorm1d([32, 128])
    StructStream->>StructStream: Linear(128→256)
    StructStream->>StructStream: GELU + Dropout

    StructStream-->>Fusion: structural_features [32, 256]
    deactivate StructStream

    activate Fusion

    Fusion->>Fusion: Sem→Struct Attention
    Note right of Fusion: Q: semantic [32, 256]<br/>K,V: structural [32, 256]<br/>Output: [32, 256]

    Fusion->>Fusion: Struct→Sem Attention
    Note right of Fusion: Q: structural [32, 256]<br/>K,V: semantic [32, 256]<br/>Output: [32, 256]

    Fusion->>Fusion: Concatenate
    Note right of Fusion: [32, 256] + [32, 256]<br/>= [32, 512]

    Fusion->>Fusion: LayerNorm([32, 512])

    Fusion-->>Classifier: fused_features [32, 512]
    deactivate Fusion

    activate Classifier

    Classifier->>Classifier: Linear(512→128)
    Classifier->>Classifier: GELU + Dropout(0.4)
    Classifier->>Classifier: Linear(128→64)
    Classifier->>Classifier: GELU + Dropout(0.4)
    Classifier->>Classifier: Linear(64→2)

    Classifier-->>Output: logits [32, 2]
    deactivate Classifier

    Note over Output: Apply softmax for probabilities<br/>Apply argmax for predictions
```

## Configuration System: YAML Deep Dive

```mermaid
graph TB
    subgraph config["📋 experiment_v8_baseline.yaml"]
        direction TB

        subgraph experiment_config["experiment"]
            exp_name["name: 'v8_baseline'"]
            exp_version["version: '8.0.0'"]
            exp_seed["seed: 42"]
        end

        subgraph data_config["data"]
            train_path["train_path: 'datasets/train.csv'"]
            test_path["test_path: 'datasets/test.csv'"]
            splits["train_split: 0.8<br/>val_split: 0.1<br/>test_split: 0.1"]
            binary["binary_classification: true<br/>positive_class: 'Pass'"]
        end

        subgraph semantic_config["semantic"]
            sem_model["model_name: 'BAAI/bge-large-en-v1.5'"]
            sem_dim["embedding_dim: 1024"]
            sem_fields["fields:<br/>  - TE_Summary<br/>  - TC_Steps<br/>  - commit"]
            sem_cache["cache_path: 'cache/embeddings'"]
        end

        subgraph structural_config["structural (NEW!)"]
            struct_window["recent_window: 5"]
            struct_dim["input_dim: 6"]
            struct_cache["cache_path: 'cache/structural_features.pkl'"]
        end

        subgraph graph_config["graph (NEW!)"]
            graph_type["type: 'co_failure'"]
            graph_min["min_co_occurrences: 2"]
            graph_thresh["weight_threshold: 0.1"]
            graph_cache["cache_path: 'cache/phylogenetic_graph.pkl'"]
            graph_build["build_graph: true"]
        end

        subgraph model_config["model"]
            model_type["type: 'dual_stream_v8'"]

            subgraph sem_stream_config["semantic"]
                sem_input["input_dim: 1024"]
                sem_hidden["hidden_dim: 256"]
                sem_layers["num_layers: 2"]
            end

            subgraph struct_stream_config["structural"]
                struct_input["input_dim: 6"]
                struct_hidden["hidden_dim: 256"]
                struct_bn["use_batch_norm: true"]
            end

            subgraph fusion_config["fusion"]
                fusion_heads["num_heads: 4"]
                fusion_dropout["dropout: 0.1"]
            end

            model_type --> sem_stream_config
            model_type --> struct_stream_config
            model_type --> fusion_config
        end

        subgraph training_config["training"]
            epochs["num_epochs: 40"]
            batch["batch_size: 32"]
            lr["learning_rate: 5e-5"]
            wd["weight_decay: 2e-4"]
            early["early_stopping:<br/>  patience: 12<br/>  monitor: 'val_f1_macro'"]
        end

        subgraph loss_config["loss"]
            loss_type["type: 'focal'"]
            focal_alpha["alpha: [0.15, 0.85]"]
            focal_gamma["gamma: 2.0"]
        end
    end

    subgraph usage["📝 Usage in Code"]
        direction TB

        load["config = load_config('configs/experiment_v8_baseline.yaml')"]
        access1["semantic_dim = config['semantic']['embedding_dim']  # 1024"]
        access2["struct_dim = config['structural']['input_dim']  # 6"]
        access3["model_cfg = config['model']  # Dict for create_model_v8()"]
    end

    config --> usage

    style structural_config fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style graph_config fill:#fff4e1,stroke:#cc7700,stroke-width:3px
```

## Training Loop: Iteration Details

```mermaid
flowchart TB
    subgraph epoch["EPOCH LOOP (epoch=1 to 40)"]
        direction TB

        start_epoch["Start Epoch {epoch}"]

        subgraph train_phase["TRAINING PHASE"]
            direction TB

            set_train["model.train()"]
            init_loss["total_loss = 0"]

            subgraph batch_iteration["BATCH ITERATION"]
                direction LR

                for_batch["for batch in train_loader:"]
                unpack["sem, struct, labels = batch<br/>sem: [32, 1024]<br/>struct: [32, 6]<br/>labels: [32]"]
                to_device["Move to device:<br/>sem.to(device)<br/>struct.to(device)<br/>labels.to(device)"]
                forward_pass["logits = model(sem, struct)<br/>logits: [32, 2]"]
                compute_loss["loss = criterion(logits, labels)<br/>Focal Loss"]
                zero_grad["optimizer.zero_grad()"]
                backward_pass["loss.backward()"]
                clip_grad["torch.nn.utils.clip_grad_norm_<br/>  (model.parameters(), 1.0)"]
                optimizer_step["optimizer.step()"]
                accumulate["total_loss += loss.item()"]

                for_batch --> unpack --> to_device --> forward_pass
                forward_pass --> compute_loss --> zero_grad --> backward_pass
                backward_pass --> clip_grad --> optimizer_step --> accumulate
            end

            avg_loss["avg_train_loss = total_loss / num_batches"]

            set_train --> init_loss --> batch_iteration --> avg_loss
        end

        subgraph val_phase["VALIDATION PHASE"]
            direction TB

            set_eval["model.eval()"]
            no_grad["with torch.no_grad():"]

            subgraph val_iteration["VALIDATION ITERATION"]
                val_forward["logits = model(sem, struct)"]
                val_loss["loss = criterion(logits, labels)"]
                collect_preds["predictions.append(preds)"]
            end

            compute_metrics["metrics = compute_metrics(<br/>  predictions, labels)"]
            val_f1["val_f1_macro = metrics['f1_macro']"]

            set_eval --> no_grad --> val_iteration --> compute_metrics --> val_f1
        end

        subgraph scheduler_step["LEARNING RATE UPDATE"]
            scheduler["scheduler.step()<br/>CosineAnnealingLR"]
            lr_decay["Learning rate decays<br/>from 5e-5 to 1e-6"]
        end

        subgraph early_stopping["EARLY STOPPING CHECK"]
            direction TB

            check_improve{"val_f1 ><br/>best_val_f1?"}
            save_model["Save best model:<br/>torch.save(model.state_dict(),<br/>  'best_model_v8.pt')"]
            reset_patience["patience_counter = 0"]
            increment_patience["patience_counter += 1"]

            check_stop{"patience_counter<br/>>= 12?"}
            stop_training["STOP TRAINING<br/>Early stopping triggered"]

            check_improve -->|Yes| save_model --> reset_patience
            check_improve -->|No| increment_patience --> check_stop
            check_stop -->|Yes| stop_training
        end

        start_epoch --> train_phase --> val_phase --> scheduler_step --> early_stopping
    end

    style train_phase fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style val_phase fill:#e1f5ff,stroke:#0077cc,stroke-width:2px
    style early_stopping fill:#fff3cd,stroke:#856404,stroke-width:2px
```

## Focal Loss Computation

```mermaid
flowchart TB
    subgraph inputs["INPUTS"]
        logits["Logits: [batch=32, classes=2]<br/>Raw model output"]
        targets["Targets: [batch=32]<br/>True labels (0 or 1)"]
        alpha["Alpha: [0.15, 0.85]<br/>Class weights"]
        gamma["Gamma: 2.0<br/>Focusing parameter"]
    end

    subgraph computation["COMPUTATION STEPS"]
        direction TB

        step1["Step 1: Get probabilities<br/>p = softmax(logits, dim=-1)<br/>p: [32, 2]"]

        step2["Step 2: Get class probabilities<br/>p_t = p[range(32), targets]<br/>p_t: [32]"]

        step3["Step 3: Compute focal weight<br/>focal_weight = (1 - p_t)^gamma<br/>focal_weight: [32]"]

        step4["Step 4: Apply alpha weighting<br/>alpha_t = alpha[targets]<br/>alpha_t: [32]"]

        step5["Step 5: Compute CE loss<br/>ce_loss = -log(p_t)<br/>ce_loss: [32]"]

        step6["Step 6: Combine<br/>focal_loss = alpha_t * focal_weight * ce_loss<br/>focal_loss: [32]"]

        step7["Step 7: Reduce<br/>loss = focal_loss.mean()<br/>loss: scalar"]

        step1 --> step2 --> step3 --> step4 --> step5 --> step6 --> step7
    end

    subgraph example["EXAMPLE"]
        direction TB

        ex_input["Example batch:<br/>targets = [0, 1, 1, 0]<br/>p_t = [0.9, 0.3, 0.8, 0.7]"]

        ex_focal["Focal weights:<br/>(1-0.9)^2 = 0.01  (easy)<br/>(1-0.3)^2 = 0.49  (hard)<br/>(1-0.8)^2 = 0.04  (easy)<br/>(1-0.7)^2 = 0.09  (medium)"]

        ex_alpha["Alpha weights:<br/>0.15 (class 0)<br/>0.85 (class 1)<br/>0.85 (class 1)<br/>0.15 (class 0)"]

        ex_result["Hard examples get<br/>higher loss weight!<br/><br/>This focuses learning<br/>on difficult cases"]

        ex_input --> ex_focal --> ex_alpha --> ex_result
    end

    inputs --> computation
    computation -.->|"Illustration"| example

    style computation fill:#e1f5ff,stroke:#0077cc,stroke-width:2px
    style example fill:#fff3cd,stroke:#856404,stroke-width:2px
```

## Comparison Table: Implementation Details

| Component | V7 Implementation | V8 Implementation | Change Type |
|-----------|------------------|-------------------|-------------|
| **Structural Input** | embeddings[1024] | features[6] | 🔴 Breaking |
| **Input Processing** | LayerNorm → FFN | BatchNorm → FFN | 🟡 Modified |
| **Parameter Count** | ~260K | ~34K | 🟢 Reduced 87% |
| **Graph Usage** | Required (k-NN) | Optional (phylogenetic) | 🟡 Modified |
| **Graph Type** | Semantic similarity | Co-failure/Commit | 🔴 Breaking |
| **BatchNorm** | Not used | Used (stability) | 🟢 Added |
| **Forward Signature** | (emb, edge_idx, edge_w) | (sem_in, struct_in) | 🔴 Breaking |
| **Config Keys** | Different | New keys added | 🔴 Breaking |
| **Cache Files** | embeddings only | embeddings + structural + graph | 🟢 Added |
| **Validation** | Manual | Automated scripts | 🟢 Added |

---

## Quick Reference: File Locations

```mermaid
graph TB
    subgraph core["Core Implementation"]
        phylo["src/phylogenetic/<br/>phylogenetic_graph_builder.py<br/>(560 lines)"]
        model["src/models/<br/>dual_stream_v8.py<br/>(530 lines)"]
        extractor["src/preprocessing/<br/>structural_feature_extractor.py<br/>(576 lines)"]
    end

    subgraph scripts["Scripts & Entry Points"]
        main["main_v8.py<br/>(400 lines)"]
        validate["scripts/<br/>validate_v8_pipeline.py<br/>(330 lines)"]
        test["test_v8_simple.py<br/>(125 lines)"]
    end

    subgraph config["Configuration"]
        yaml["configs/<br/>experiment_v8_baseline.yaml<br/>(140 lines)"]
    end

    subgraph cache["Cache Files"]
        emb_cache["cache/embeddings/<br/>*.npy"]
        struct_cache["cache/<br/>structural_features.pkl"]
        graph_cache["cache/<br/>phylogenetic_graph.pkl"]
    end

    subgraph outputs["Outputs"]
        model_out["results/<br/>best_model.pt"]
        metrics_out["results/<br/>test_metrics.json"]
        plots_out["results/<br/>*.png"]
        csv_out["results/<br/>*.csv"]
    end

    core --> scripts
    config --> scripts
    scripts --> cache
    scripts --> outputs

    style core fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style config fill:#fff3cd,stroke:#856404,stroke-width:2px
    style cache fill:#e1f5ff,stroke:#0077cc,stroke-width:2px
```

---

**Status**: ✅ **STEP 2.2 Implementation Details Complete**
