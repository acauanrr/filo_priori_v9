# Figures and Diagrams - Filo-Priori V8

This directory contains technical diagrams and architecture visualizations for Filo-Priori V8.

**Updated**: 2025-11-16
**Status**: Production-ready (Experiment 06)
**APFD**: 0.6171 | **F1-Macro**: 0.5312

## 📊 Available Diagrams

### 🎯 FOR PRESENTATIONS (NEW!)

#### [dual_stream_architecture_slide.mmd](dual_stream_architecture_slide.mmd)
**Ultra-compact horizontal flow diagram - BEST FOR SLIDES**

Perfect for:
- Conference presentations
- Thesis defense slides
- Quick architecture overview

Shows:
- Horizontal left-to-right flow
- All 5 main components with specs
- Parameter counts per component
- Performance metrics summary
- Color-coded by component type

**Dimensions**: Optimized for 16:9 slide format

---

#### [dual_stream_architecture_compact.mmd](dual_stream_architecture_compact.mmd)
**Detailed vertical flow diagram - GOOD FOR TECHNICAL SLIDES**

Perfect for:
- Technical deep-dives
- Architecture walkthroughs
- Detailed component explanation

Shows:
- Vertical top-to-bottom flow
- Detailed specifications per component
- Parameter breakdown with percentages
- Input/output dimensions
- Model summary box

**Dimensions**: Optimized for detailed technical presentations

---

#### [architecture_specification_table.md](architecture_specification_table.md)
**Table-based specification with diagrams - BEST FOR DOCUMENTATION**

Perfect for:
- Technical reports
- Architecture documentation
- Quick reference guide

Contains:
- Simple horizontal flow diagram
- Complete specification table
- Performance metrics table
- Parameter distribution pie chart
- Key design principles

**Format**: Markdown with embedded Mermaid diagrams

---

### 1. Complete Pipeline Architecture

#### [complete_pipeline_architecture.mmd](complete_pipeline_architecture.mmd)
**Main diagram: End-to-end pipeline (Phase 1: Classification + Phase 2: Prioritization)**

**Updated**: 2025-11-16 with latest improvements

Contains:
- 📥 **Phase 1: Test Case Failure Classification**
  - Step 1: Data Input & Preprocessing (52,102 executions, 1,339 builds)
  - Step 2.1: Semantic Embedding Extraction (SBERT all-mpnet-base-v2, 1536-dim)
  - Step 2.2: Structural Feature Extraction (10 expert-selected features)
  - Step 3: Multi-Edge Phylogenetic Graph (co-failure, co-success, semantic)
  - Step 4: Dual-Stream Model with GAT (2-head attention)
  - Step 5: Training (Weighted CE Loss, AdamW, Cosine Annealing)
  - Step 6: Evaluation (F1=0.5312, Accuracy=63.29%)

- 🎖️ **Phase 2: Test Case Prioritization & APFD**
  - Step 7: Per-Build Ranking (probability-based)
  - Step 8: APFD Calculation (per-build scores)
  - Step 9: APFD Summary (Mean=0.6171, +23.4% vs random)

**Key Features Shown**:
- Multi-edge graph construction with 3 edge types
- GAT with orphan handling for new test cases
- Parallel processing of semantic and structural paths
- Production metrics from Experiment 06

**When to use**: Understanding the complete system architecture and data flow

---

### 2. Structural Features Extraction (STEP 2.1)

#### [step_2.1_structural_features_extraction.md](step_2.1_structural_features_extraction.md)
**Main diagram for structural feature extraction**

Contains:
- 🔴 **V7 Problem**: Semantic Echo Chamber - both streams used BGE
- 🟢 **V8 Solution**: True orthogonal information (semantic + structural)
- 🔧 **Extraction Pipeline**: Complete StructuralFeatureExtractorV2.5 flow
- 📊 **10 Selected Features**: 6 phylogenetic + 4 structural
- 🧮 **Computation Details**: How each feature is calculated
- 🏗️ **V8 Integration**: How features connect to dual-stream model
- ✅ **Validation**: Test results with production data
- 🎯 **Key Achievements**: Mind map of main results
- ⚡ **Performance Metrics**: Time, memory, quality

**When to use**: Understanding feature extraction architecture and purpose

#### [step_2.1_data_flow.md](step_2.1_data_flow.md)
**Technical data flow and detailed implementation**

Contains:
- 📁 **Complete Data Flow**: From raw CSV to model input
- 🔄 **Processing Pipeline**: Parallel semantic + structural processing
- 📊 **Feature Matrix Structure**: Exact format [N, 10]
- 🎯 **Model Input Format**: Tensors, dtypes, devices
- 📈 **Historical Computation**: Step-by-step sequence diagram
- 📉 **Feature Distributions**: Validated statistics
- 💾 **Cache Strategy**: Performance with and without caching
- 🛡️ **Data Leakage Prevention**: Correct vs incorrect comparison
- 💻 **Example Code**: Real Python integration
- 📋 **Comparison Table**: V7 vs V8 (before/after)

**When to use**: Technical implementation, debugging, or code integration

---

### 3. Model Architecture & Phylogenetic Graph (STEP 2.2)

#### [step_2.2_model_architecture.md](step_2.2_model_architecture.md)
**Main diagram: V8 Model Architecture and Phylogenetic Graphs**

Contains:
- ❌ **V7 vs ✅ V8**: Complete visual comparison of Echo Chamber vs True Dual-Stream
- 🌳 **Phylogenetic Graph Builder**: Multi-edge graph with 3 edge types
  - Co-failure edges: Tests failing together (weight=1.0)
  - Co-success edges: Tests passing together (weight=0.5)
  - Semantic edges: Content similarity (weight=0.3)
- 📊 **Co-Failure Sequence**: Sequence diagram of graph construction
- 🏗️ **V8 Model Layer-by-Layer**: Each layer explained in detail
  - Semantic Stream: 1536 → 256
  - Structural Stream: 10 → 64 (with GAT)
  - Fusion Layer: 320 → 256
  - Classifier: 256 → 2
- 🔄 **Breaking Changes**: V7 → V8 incompatible changes
- 🚀 **Training Pipeline**: Complete training flow
- ✅ **Validation Tests**: 5 tests performed
- 🎯 **Scientific Impact**: Mind map of contributions
- 📅 **Roadmap**: Next steps (immediate, short-term, long-term)

**When to use**: Understanding the new V8 architecture and phylogenetic graphs

#### [step_2.2_implementation_details.md](step_2.2_implementation_details.md)
**End-to-end technical flow and implementation details**

Contains:
- 📁 **Complete Data Flow**: Raw CSV → Predictions
- 🌳 **Edge Weight Computation**: Multi-edge graph construction in detail
  - Co-failure: P(A fails | B fails)
  - Co-success: P(A passes | B passes)
  - Semantic: Cosine similarity with threshold
- 🏗️ **Model Deep Dive**: Layer-by-layer architecture with parameters
  - Semantic stream: FFN blocks with residual connections
  - Structural stream: BatchNorm + GAT
  - Fusion: Concatenation + FFN
- 🔄 **Forward Pass Sequence**: Complete sequence diagram
- ⚙️ **Config System YAML**: Detailed configuration structure (Experiment 06)
- 🔁 **Training Loop**: Detailed iteration with early stopping
- 📉 **Weighted CE Loss**: Step-by-step computation with class weights
- 📋 **Comparison Table**: V7 vs V8 implementation
- 📂 **File Locations**: Where to find each file

**When to use**: Implementation, debugging, or understanding technical details

---

## 🎨 Como Visualizar os Diagramas

### Opção 1: GitHub/GitLab (Recomendado)
Os arquivos .md com diagramas Mermaid são renderizados automaticamente.

### Opção 2: VS Code
Instale "Markdown Preview Mermaid Support" e pressione Ctrl+Shift+V

### Opção 3: Mermaid Live Editor
Acesse https://mermaid.live/ e cole o código

---

## 📊 Documentation Statistics

### Complete Pipeline
- **File**: complete_pipeline_architecture.mmd
- **Last Updated**: 2025-11-16
- **Diagrams**: 1 comprehensive end-to-end Mermaid diagram
- **Size**: ~9 KB
- **Focus**: Full pipeline (Phase 1: Classification + Phase 2: Prioritization)
- **Coverage**: 9 steps across 2 phases

### STEP 2.1: Structural Features Extraction
- **Files**: 2 markdown files
- **Diagrams**: ~8 Mermaid diagrams
- **Size**: ~22 KB total
- **Focus**: Feature extraction, validation, pipeline
- **Features**: 10 expert-selected features (6 phylogenetic + 4 structural)

### STEP 2.2: Model Architecture & Phylogenetic Graphs
- **Files**: 2 markdown files
- **Diagrams**: ~15 Mermaid diagrams
- **Size**: ~34 KB total
- **Focus**: Model architecture, multi-edge graphs, training, implementation
- **Key Components**: GAT, multi-edge graph, dual-stream fusion

### Total
- **Files**: 5 diagram files (1 .mmd + 4 .md) + README
- **Diagrams**: ~24 Mermaid diagrams
- **Coverage**: Complete end-to-end pipeline + detailed component breakdowns
- **Lines of Code Documented**: ~2,000+ lines across preprocessing, models, and phylogenetic modules

---

## 🔗 Diagram → Implementation Mapping

| Diagram | Implementation Files | Key Components |
|---------|---------------------|----------------|
| `complete_pipeline_architecture.mmd` | `main.py` (full pipeline) | End-to-end flow with all 9 steps |
| `step_2.1_structural_features_extraction.md` | `src/preprocessing/structural_feature_extractor_v2_5.py` | 10 feature extraction |
| `step_2.1_data_flow.md` | `src/preprocessing/data_loader.py` + `src/embeddings/` | Data flow + caching |
| `step_2.2_model_architecture.md` | `src/models/dual_stream.py` | Dual-stream model + GAT |
| `step_2.2_implementation_details.md` | `src/phylogenetic/multi_edge_graph_builder.py` | Multi-edge graph construction |

---

## 🎯 Production Status

**Current Version**: Experiment 06 (Feature Selection)
**Status**: ✅ Production-Ready
**Performance**:
- Mean APFD: 0.6171 (61.71%)
- F1-Macro: 0.5312
- Accuracy: 63.29%
- Improvement vs Random: +23.4%

**Documentation**: ✅ Complete visual documentation for all pipeline components

---

**For additional visualizations, see**: `results/publication/` directory
- INFERENCE_EXPLANATION.md
- GRAPH_CONSTRUCTION_STEP_BY_STEP.md
- FEATURE_EXPANSION_ANALYSIS.md
- multi_edge_phylogenetic_graph_interactive.html
