# Filo-Priori V9 - Documentação Completa do Sistema

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Estrutura do Projeto](#2-estrutura-do-projeto)
3. [Pipeline de Dados](#3-pipeline-de-dados)
4. [Arquitetura do Modelo](#4-arquitetura-do-modelo)
5. [Estratégias para Datasets](#5-estratégias-para-datasets)
6. [Sistema de Treinamento](#6-sistema-de-treinamento)
7. [Sistema de Avaliação](#7-sistema-de-avaliação)
8. [Relacionamento entre Componentes](#8-relacionamento-entre-componentes)
9. [Configurações](#9-configurações)
10. [Resultados e Métricas](#10-resultados-e-métricas)

---

## 1. Visão Geral

### 1.1 Objetivo do Sistema

O **Filo-Priori V9** é um sistema de **Priorização de Casos de Teste (TCP - Test Case Prioritization)** baseado em Deep Learning que combina:

- **Stream Semântico**: Embeddings SBERT de descrições de testes e commits
- **Stream Estrutural**: Features históricas processadas por Graph Attention Networks (GAT)
- **Fusão Cross-Attention**: Combinação dinâmica das duas modalidades

### 1.2 Métricas Principais

| Métrica | Valor | Dataset |
|---------|-------|---------|
| **Mean APFD** | 0.6413 | Industrial QTA |
| **vs FailureRate** | +2.0% | Baseline |
| **vs Random** | +14.6% | Baseline |

### 1.3 Inovações Técnicas

1. **Grafo Multi-Edge**: Arestas co-failure + co-success + similaridade semântica
2. **Arquitetura Dual-Stream**: SBERT (semântica) + GAT (estrutura)
3. **Cross-Attention Fusion**: Atenção bidirecional entre modalidades
4. **Weighted Focal Loss**: Para desbalanceamento extremo (37:1)
5. **Feature Selection V2.5**: 10 features discriminativas selecionadas

---

## 2. Estrutura do Projeto

### 2.1 Árvore de Diretórios

```
filo_priori_v9/
├── main.py                          # Pipeline principal (Classificação)
├── main_rtptorrent.py               # Pipeline L2R (RTPTorrent)
├── requirements.txt                 # Dependências Python
├── README.md                        # Documentação principal
│
├── configs/                         # Configurações YAML
│   ├── experiment_industry.yaml     # Dataset Industrial
│   ├── experiment_rtptorrent_l2r.yaml  # Dataset RTPTorrent
│   ├── experiment_07_ranking_optimized.yaml  # Melhor config (APFD 0.6413)
│   └── ...
│
├── src/                             # Código-fonte principal
│   ├── models/                      # Arquiteturas neurais
│   │   ├── dual_stream_v8.py        # Modelo principal
│   │   ├── phylogenetic_dual_stream.py  # Variante filogenética
│   │   ├── phylo_encoder.py         # Encoder GGNN
│   │   └── cross_attention.py       # Módulo de fusão
│   │
│   ├── embeddings/                  # Sistema de embeddings SBERT
│   │   ├── embedding_manager.py     # Interface de alto nível
│   │   ├── sbert_encoder.py         # Codificação SBERT
│   │   └── embedding_cache.py       # Cache persistente
│   │
│   ├── preprocessing/               # Carregamento e features
│   │   ├── data_loader.py           # Carregamento CSV
│   │   ├── structural_feature_extractor_v2.py    # 29 features
│   │   ├── structural_feature_extractor_v2_5.py  # 10 features selecionadas
│   │   └── structural_feature_imputation.py      # Imputação
│   │
│   ├── phylogenetic/                # Construção de grafos
│   │   ├── multi_edge_graph_builder.py   # Grafo multi-edge
│   │   └── phylogenetic_graph_builder.py # Grafo filogenético
│   │
│   ├── training/                    # Funções de perda
│   │   ├── losses.py                # Focal Loss, Weighted CE
│   │   └── ranking_losses.py        # ListNet, LambdaRank
│   │
│   ├── evaluation/                  # Métricas e avaliação
│   │   ├── apfd.py                  # Cálculo APFD
│   │   ├── metrics.py               # F1, Precision, Recall
│   │   └── threshold_optimizer.py   # Otimização de threshold
│   │
│   └── baselines/                   # Implementações baseline
│       ├── heuristic_baselines.py   # Random, Recency
│       └── ml_baselines.py          # LogReg, RF, XGBoost
│
├── scripts/                         # Scripts de análise
│   ├── analysis/                    # Estudos de ablação, CV temporal
│   ├── preprocessing/               # Pré-processamento
│   └── publication/                 # Geração de figuras/tabelas
│
├── datasets/                        # Dados de entrada
│   ├── 01_industry/                 # Dataset Industrial QTA
│   │   ├── train.csv                # 41,680 execuções
│   │   └── test.csv                 # 10,420 execuções
│   └── 02_rtptorrent/               # Dataset RTPTorrent
│       ├── raw/MSR2/                # 20 projetos Java
│       └── processed/               # Dados processados
│
├── cache/                           # Cache de embeddings
│   ├── 01_industry/
│   └── 02_rtptorrent/
│
├── results/                         # Resultados experimentais
│   ├── experiment_07_ranking_optimized/  # Melhor resultado
│   ├── baselines/
│   ├── temporal_cv/
│   └── ablation/
│
└── paper/                           # Materiais de publicação
    ├── main_ieee_tse.tex
    ├── figures/
    └── tables/
```

### 2.2 Módulos Principais

| Módulo | Linhas | Propósito |
|--------|--------|-----------|
| `dual_stream_v8.py` | ~23K | Modelo principal de produção |
| `data_loader.py` | ~500 | Carregamento e splits |
| `apfd.py` | ~500 | Métrica principal de ranking |
| `multi_edge_graph_builder.py` | ~400 | Construção do grafo |
| `structural_feature_extractor_v2_5.py` | ~600 | Extração de 10 features |

---

## 3. Pipeline de Dados

### 3.1 Fluxo Geral

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DE DADOS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CSV (train.csv, test.csv)                                            │
│        │                                                               │
│        ▼                                                               │
│   ┌────────────────────────────────────────┐                           │
│   │         DATA LOADER                     │                           │
│   │  • Carrega CSVs                        │                           │
│   │  • Limpa valores ausentes              │                           │
│   │  • Processa commits/CRs                │                           │
│   │  • Codifica labels                     │                           │
│   └────────────────────────────────────────┘                           │
│        │                                                               │
│        ▼                                                               │
│   ┌────────────────────────────────────────┐                           │
│   │      TRAIN/VAL/TEST SPLIT              │                           │
│   │  • GroupShuffleSplit por Build_ID      │                           │
│   │  • 80% / 10% / 10%                     │                           │
│   │  • Anti-vazamento temporal             │                           │
│   └────────────────────────────────────────┘                           │
│        │                                                               │
│        ├──────────────────┬──────────────────┐                         │
│        ▼                  ▼                  ▼                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   SEMANTIC   │  │  STRUCTURAL  │  │    GRAPH     │                  │
│  │  EMBEDDINGS  │  │   FEATURES   │  │ CONSTRUCTION │                  │
│  │              │  │              │  │              │                  │
│  │ • SBERT      │  │ • V2.5       │  │ • Co-failure │                  │
│  │ • TC: 768d   │  │ • 10 feat.   │  │ • Co-success │                  │
│  │ • Commit:768d│  │ • Imputação  │  │ • Semântico  │                  │
│  │ • Total:1536d│  │              │  │              │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│        │                  │                  │                         │
│        └──────────────────┼──────────────────┘                         │
│                           ▼                                            │
│                  ┌──────────────────┐                                  │
│                  │   MODEL INPUT    │                                  │
│                  │ • Embeddings     │                                  │
│                  │ • Features       │                                  │
│                  │ • Graph edges    │                                  │
│                  │ • Labels         │                                  │
│                  └──────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Carregamento de Dados

**Arquivo**: `src/preprocessing/data_loader.py`

```python
class DataLoader:
    def load_data(train_path, test_path) -> (DataFrame, DataFrame)
    def clean_data(df) -> DataFrame
    def encode_labels(df, binary=True, strategy='pass_vs_fail') -> DataFrame
    def compute_class_weights(df) -> dict
    def split_data(df, train_split=0.8, val_split=0.1) -> (df_train, df_val, df_test)
```

**Campos do CSV**:
| Campo | Descrição | Uso |
|-------|-----------|-----|
| `Build_ID` | Identificador da build | Agrupamento |
| `TC_Key` | ID único do caso de teste | Identificação |
| `TE_Summary` | Descrição do teste | Embedding semântico |
| `TC_Steps` | Passos detalhados | Embedding semântico |
| `TE_Test_Result` | Pass/Fail | Label |
| `commit` | Mensagens de commit | Embedding semântico |
| `CR` | Change Request | Feature estrutural |

### 3.3 Split Anti-Vazamento

O sistema usa **GroupShuffleSplit** com `Build_ID` como grupo:

```python
# Todos os testes da mesma build permanecem juntos
# Isso força o modelo a generalizar para builds não vistas

splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(splitter.split(X, y, groups=Build_ID))
```

**Verificação de Vazamento**:
- Checa overlap de Build_IDs entre splits
- Loga qualquer detecção de vazamento

### 3.4 Extração de Embeddings Semânticos

**Arquivo**: `src/embeddings/embedding_manager.py`

```
┌────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS SBERT                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Modelo: sentence-transformers/all-mpnet-base-v2               │
│  Dimensão: 768 por stream                                      │
│                                                                │
│  STREAM 1: Caso de Teste                                       │
│  ┌────────────────────────────────────┐                        │
│  │ "Summary: [TE_Summary]             │                        │
│  │  Steps: [TC_Steps]"                │ ──► 768-dim            │
│  └────────────────────────────────────┘                        │
│                                                                │
│  STREAM 2: Commits                                             │
│  ┌────────────────────────────────────┐                        │
│  │ "Commit Message: [commit]          │                        │
│  │  Diff: [diff[:2000]]"              │ ──► 768-dim            │
│  └────────────────────────────────────┘                        │
│                                                                │
│  COMBINADO: concatenate([TC, Commit]) ──► 1536-dim             │
│                                                                │
│  Cache: cache/01_industry/all-mpnet-base-v2.pkl                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.5 Extração de Features Estruturais

**Arquivo**: `src/preprocessing/structural_feature_extractor_v2_5.py`

**10 Features Selecionadas (V2.5)**:

| # | Feature | Descrição | Tipo |
|---|---------|-----------|------|
| 1 | `test_age` | Builds desde primeira aparição | Temporal |
| 2 | `failure_rate` | Taxa histórica de falha | Estatística |
| 3 | `recent_failure_rate` | Taxa nas últimas 5 builds | Tendência |
| 4 | `flakiness_rate` | Oscilação Pass/Fail | Estabilidade |
| 5 | `consecutive_failures` | Streak atual de falhas | Padrão |
| 6 | `max_consecutive_failures` | Maior streak histórico | Padrão |
| 7 | `failure_trend` | recent - overall rate | Tendência |
| 8 | `commit_count` | Commits associados | Cobertura |
| 9 | `test_novelty` | Flag de primeira aparição | Novidade |
| 10 | `cr_count` | Change Requests | Mudanças |

**Janelas Temporais**:
```python
recent_window = 5 builds
very_recent_window = 2 builds
medium_term_window = 10 builds
min_history = 2 execuções
```

### 3.6 Construção do Grafo

**Arquivo**: `src/phylogenetic/multi_edge_graph_builder.py`

**Três Tipos de Arestas**:

| Tipo | Peso | Definição |
|------|------|-----------|
| **Co-Failure** | 1.0 | Testes que falham juntos na mesma build |
| **Co-Success** | 0.5 | Testes que passam juntos |
| **Semântico** | 0.3 | Similaridade de embedding > 0.75 |

```python
# Construção do grafo
graph_builder = MultiEdgeGraphBuilder(
    edge_types=['co_failure', 'co_success', 'semantic'],
    edge_weights={'co_failure': 1.0, 'co_success': 0.5, 'semantic': 0.3},
    min_co_occurrences=1,
    semantic_threshold=0.75
)
graph = graph_builder.fit(df_train, embeddings)
```

---

## 4. Arquitetura do Modelo

### 4.1 Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DualStreamModelV8 - ARQUITETURA                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐      ┌─────────────────────────┐          │
│  │    SEMANTIC STREAM      │      │   STRUCTURAL STREAM     │          │
│  │                         │      │                         │          │
│  │  Input: [B, 1536]       │      │  Input: [N, 10]        │          │
│  │         ▼               │      │         ▼               │          │
│  │  Linear(1536 → 256)     │      │  Linear(10 → 64)       │          │
│  │         ▼               │      │         ▼               │          │
│  │  FFN Layer 1            │      │  GAT Layer 1           │          │
│  │  • 256 → 1024 → 256     │      │  • 64 → 256            │          │
│  │  • GELU + Dropout       │      │  • 4 heads             │          │
│  │  • Residual             │      │  • ELU + Dropout       │          │
│  │         ▼               │      │         ▼               │          │
│  │  FFN Layer 2            │      │  GAT Layer 2 (opt)     │          │
│  │  • Mesmo padrão         │      │  • 1 head              │          │
│  │         ▼               │      │         ▼               │          │
│  │  LayerNorm              │      │  Output: [N, 256]      │          │
│  │         ▼               │      │                         │          │
│  │  Output: [B, 256]       │      │                         │          │
│  └────────────┬────────────┘      └────────────┬────────────┘          │
│               │                                │                        │
│               └────────────────┬───────────────┘                        │
│                                ▼                                        │
│               ┌────────────────────────────────────┐                    │
│               │      CROSS-ATTENTION FUSION        │                    │
│               │                                    │                    │
│               │  Semantic ◄──► Structural          │                    │
│               │  (Atenção bidirecional)            │                    │
│               │                                    │                    │
│               │  Output: [B, 512]                  │                    │
│               └────────────────┬───────────────────┘                    │
│                                ▼                                        │
│               ┌────────────────────────────────────┐                    │
│               │         CLASSIFIER MLP             │                    │
│               │                                    │                    │
│               │  512 → 128 → 64 → 2               │                    │
│               │  GELU + Dropout(0.4)               │                    │
│               │                                    │                    │
│               │  Output: [B, 2] (Pass/Fail probs)  │                    │
│               └────────────────────────────────────┘                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Stream Semântico

**Arquivo**: `src/models/dual_stream_v8.py` (linhas 35-86)

```python
class SemanticStream(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, num_layers=2, dropout=0.3):
        # Projeção de entrada
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # N camadas FFN com residuais
        self.layers = nn.ModuleList([
            FFNLayer(hidden_dim, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)  # [B, 1536] → [B, 256]
        for layer in self.layers:
            x = x + layer(x)  # Residual
        return self.norm(x)  # [B, 256]
```

### 4.3 Stream Estrutural (GAT)

**Arquivo**: `src/models/dual_stream_v8.py` (linhas 88-223)

```python
class StructuralStreamV8(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, num_heads=4, dropout=0.3):
        # Projeção inicial
        self.input_proj = nn.Linear(input_dim, 64)

        # GAT Layer 1: multi-head com concatenação
        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,  # Concatena heads
            edge_dim=1    # Suporta pesos de aresta
        )

        # GAT Layer 2: single-head com média
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index, edge_weights):
        x = self.input_proj(x)  # [N, 10] → [N, 64]
        x = self.gat1(x, edge_index, edge_attr=edge_weights)  # [N, 1024]
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_weights)  # [N, 256]
        return x
```

### 4.4 Cross-Attention Fusion

**Arquivo**: `src/models/dual_stream_v8.py` (linhas 225-311)

```python
class CrossAttentionFusion(nn.Module):
    """
    Fusão bidirecional por atenção cruzada.

    O stream semântico "pergunta" ao estrutural e vice-versa.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        self.sem_to_struct = nn.MultiheadAttention(embed_dim, num_heads)
        self.struct_to_sem = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, semantic, structural):
        # Semântico atende ao estrutural
        sem_enhanced, _ = self.sem_to_struct(semantic, structural, structural)
        sem_enhanced = self.norm1(semantic + sem_enhanced)

        # Estrutural atende ao semântico
        struct_enhanced, _ = self.struct_to_sem(structural, semantic, semantic)
        struct_enhanced = self.norm2(structural + struct_enhanced)

        # Concatena
        return torch.cat([sem_enhanced, struct_enhanced], dim=-1)  # [B, 512]
```

### 4.5 Gated Fusion (Alternativa)

```python
class GatedFusionUnit(nn.Module):
    """
    Arbitração dinâmica entre modalidades.

    z ≈ 1 → usa semântico
    z ≈ 0 → usa estrutural
    """
    def forward(self, semantic, structural):
        combined = torch.cat([semantic, structural], dim=-1)
        z = torch.sigmoid(self.gate_net(combined))  # [B, 256]
        fused = z * semantic + (1 - z) * structural
        return self.output_proj(fused)  # [B, 512]
```

### 4.6 Classificador

```python
class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[128, 64], num_classes=2):
        self.layers = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(64, 2)  # Binary: Pass vs Fail
        )

    def forward(self, x):
        return self.layers(x)  # Logits [B, 2]
```

### 4.7 Funções de Perda

**Arquivo**: `src/training/losses.py`

| Perda | Fórmula | Uso |
|-------|---------|-----|
| **CrossEntropy** | `-log(p_t)` | Balanceado |
| **WeightedCE** | `-w_c * log(p_t)` | Desbalanceado |
| **FocalLoss** | `-α(1-p_t)^γ * log(p_t)` | Hard examples |
| **WeightedFocal** | Combina os 3 | **37:1 imbalance** |

**Configuração Otimizada**:
```yaml
loss:
  type: "weighted_focal"
  focal_alpha: 0.75      # Peso da classe minoritária
  focal_gamma: 2.5       # Foco em exemplos difíceis
  label_smoothing: 0.0
```

---

## 5. Estratégias para Datasets

### 5.1 Comparação Geral

| Aspecto | Industrial QTA | RTPTorrent |
|---------|----------------|------------|
| **Tamanho** | 52K execuções | 23M execuções |
| **Projetos** | 1 (proprietário) | 20 (open-source) |
| **Taxa de Falha** | 2.7% (37:1) | 0.2% (500:1) |
| **Info Semântica** | Rica | Limitada |
| **Modo** | Classificação | Learning-to-Rank |
| **Perda** | Weighted Focal | ListNet |
| **Grafo** | GAT multi-edge | Nenhum |

### 5.2 Dataset Industrial (Classificação)

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE INDUSTRIAL (main.py)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DADOS                           MODELO                         │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ • 52K execuções     │        │ DualStreamModelV8   │        │
│  │ • 1,339 builds      │        │ + Cross-Attention   │        │
│  │ • 2,347 TCs únicos  │        │ + GAT multi-edge    │        │
│  │ • TE_Summary rica   │        │                     │        │
│  │ • Commits + CRs     │        │ Output: P(Fail)     │        │
│  └─────────────────────┘        └─────────────────────┘        │
│                                                                 │
│  EMBEDDINGS                      FEATURES                       │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ TC: 768-dim         │        │ 10 V2.5 features    │        │
│  │ Commit: 768-dim     │        │ • failure_rate      │        │
│  │ Combined: 1536-dim  │        │ • recent_failure    │        │
│  └─────────────────────┘        │ • flakiness         │        │
│                                 │ • etc.              │        │
│  GRAFO                          └─────────────────────┘        │
│  ┌─────────────────────┐                                       │
│  │ Co-failure: w=1.0   │        PERDA                          │
│  │ Co-success: w=0.5   │        ┌─────────────────────┐        │
│  │ Semantic: w=0.3     │        │ WeightedFocalLoss   │        │
│  └─────────────────────┘        │ α=0.75, γ=2.5       │        │
│                                 └─────────────────────┘        │
│                                                                 │
│  RESULTADO: APFD = 0.6413                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Por que Classificação funciona:**
1. **Info semântica rica**: Descrições + commits fornecem contexto
2. **Desbalanceamento manejável**: 37:1 é severo mas tratável com Focal Loss
3. **Suite pequena**: 2,347 TCs permite aprendizado de grafo
4. **Cross-Attention valioso**: Streams semântico e estrutural são balanceados

### 5.3 Dataset RTPTorrent (Learning-to-Rank)

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE RTPTORRENT (main_rtptorrent.py)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DADOS                           MODELO                         │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ • 23M execuções     │        │ Two-Stream MLP      │        │
│  │ • 110K builds       │        │ + Concatenation     │        │
│  │ • 20 projetos       │        │ (sem grafo)         │        │
│  │ • Class names only  │        │                     │        │
│  │ • 0.2% failure rate │        │ Output: Score       │        │
│  └─────────────────────┘        └─────────────────────┘        │
│                                                                 │
│  EMBEDDINGS                      FEATURES                       │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ Class name: 768-dim │        │ 9 features          │        │
│  │ (single stream)     │        │ • total_executions  │        │
│  │                     │        │ • failure_rate      │        │
│  │ Peso estrutural: 2x │        │ • recent_failures   │        │
│  └─────────────────────┘        │ • duration          │        │
│                                 └─────────────────────┘        │
│                                                                 │
│  PERDA: ListNet (listwise)                                     │
│  ┌─────────────────────┐                                       │
│  │ P_pred vs P_true    │        BASELINES: 7 estratégias       │
│  │ Cross-entropy       │        • untreated, random            │
│  │ distribuições       │        • recently-failed              │
│  └─────────────────────┘        • optimal-failure, etc.        │
│                                                                 │
│  RESULTADO: APFD = 0.7113 (HikariCP)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Por que Learning-to-Rank funciona:**
1. **Esparsidade extrema**: 0.2% torna classificação quase impossível
2. **Info semântica limitada**: Apenas nomes de classe
3. **Métrica de ranking**: APFD é inerentemente um ranking
4. **Múltiplos projetos**: Permite generalização cross-projeto

### 5.4 Diferenças de Configuração

**experiment_industry.yaml:**
```yaml
model:
  type: "dual_stream"
  semantic:
    input_dim: 1536     # Dual-field
  gnn:
    type: "GAT"
    num_layers: 1
    num_heads: 2

training:
  loss:
    type: "weighted_focal"
    focal_alpha: 0.75
```

**experiment_rtptorrent_l2r.yaml:**
```yaml
model:
  type: "ranking_model"
  semantic_dim: 768     # Single-field
  structural_stream:
    weight: 2.0         # Dobra peso

training:
  ranking_loss:
    type: "listnet"
```

---

## 6. Sistema de Treinamento

### 6.1 Loop de Treinamento

**Arquivo**: `main.py` (linhas 1020-1060)

```python
# Pseudo-código do loop de treinamento
for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        # 1. Extrai subgrafo do batch
        subgraph = extract_subgraph(batch_tc_keys, full_graph)

        # 2. Forward pass
        logits = model(
            semantic_input=batch['embeddings'],
            structural_input=batch['features'],
            edge_index=subgraph.edge_index,
            edge_weights=subgraph.edge_weights
        )

        # 3. Calcula perda
        loss = criterion(logits, batch['labels'])

        # 4. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 5. Validação
    val_metrics = evaluate(model, val_loader)

    # 6. Early stopping
    if val_metrics['f1_macro'] > best_f1:
        save_checkpoint(model)
        best_f1 = val_metrics['f1_macro']
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

    scheduler.step()
```

### 6.2 Otimizador e Scheduler

```python
# Configuração padrão
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

### 6.3 Early Stopping

```python
early_stopping = EarlyStopping(
    patience=12,
    monitor='val_f1_macro',
    mode='max',
    min_delta=0.001
)
```

### 6.4 Amostragem Balanceada

```python
# Pesos para WeightedRandomSampler
minority_weight = 1.0
majority_weight = 0.05  # Ratio 20:1

sample_weights = [
    minority_weight if label == 1 else majority_weight
    for label in labels
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
```

---

## 7. Sistema de Avaliação

### 7.1 Métrica Principal: APFD

**Arquivo**: `src/evaluation/apfd.py`

**Fórmula APFD:**
```
APFD = 1 - (Σ TFi) / (n_failures × n_tests) + 1/(2 × n_tests)

onde:
- TFi = posição do failure i (1-indexed, menor = maior prioridade)
- n_failures = número de testes com falha na build
- n_tests = total de testes na build
```

**Interpretação:**
- APFD = 1.0: Todos os failures no topo (perfeito)
- APFD = 0.5: Ranking aleatório
- APFD = 0.0: Todos os failures no final (pior caso)

### 7.2 Cálculo por Build

```python
def calculate_apfd_per_build(df, probability_col='probability'):
    """
    Calcula APFD para cada build com falhas.

    Regra: Só inclui builds com TE_Test_Result == 'Fail'
    Esperado: 277 builds com falhas
    """
    results = []

    for build_id, group in df.groupby('Build_ID'):
        # Só builds com falhas
        if 'Fail' not in group['TE_Test_Result'].values:
            continue

        # Rankeia por probabilidade (desc)
        group = group.sort_values(probability_col, ascending=False)
        group['rank'] = range(1, len(group) + 1)

        # Posições dos failures
        failure_positions = group[group['TE_Test_Result'] == 'Fail']['rank']

        # Calcula APFD
        n_tests = len(group)
        n_failures = len(failure_positions)
        apfd = 1 - (failure_positions.sum() / (n_failures * n_tests)) + 1/(2*n_tests)

        results.append({'Build_ID': build_id, 'APFD': apfd})

    return pd.DataFrame(results)
```

### 7.3 Otimização de Threshold

**Arquivo**: `src/evaluation/threshold_optimizer.py`

```python
def optimize_threshold_for_minority(
    probabilities,
    labels,
    strategy='f1_macro',
    threshold_range=(0.2, 0.8),
    step=0.05
):
    """
    Busca exaustiva pelo melhor threshold.

    Estratégias:
    - f1_macro: Otimiza F1 Macro
    - f1_minority: Otimiza F1 da classe minoritária
    - recall_minority: Maximiza recall de falhas
    - balanced_accuracy: Balanceia sensibilidade/especificidade
    """
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(*threshold_range, step):
        predictions = (probabilities >= threshold).astype(int)

        if strategy == 'f1_macro':
            score = f1_score(labels, predictions, average='macro')
        elif strategy == 'recall_minority':
            score = recall_score(labels, predictions, pos_label=1)
        # ...

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 7.4 Validação Cruzada Temporal

**Arquivo**: `scripts/analysis/run_temporal_cv.py`

**3 Estratégias de CV:**

1. **Temporal K-Fold**:
   - Treina no passado, testa no futuro
   - K=5 folds ordenados por data

2. **Sliding Window**:
   - Janela deslizante de treino/teste
   - Simula updates contínuos

3. **Concept Drift Analysis**:
   - Detecta degradação ao longo do tempo
   - Correlação tempo × APFD

```python
# Detecção de drift
correlation = pearsonr(time_indices, apfd_values)
has_drift = (correlation.pvalue < 0.05) and (correlation.statistic < -0.3)
```

### 7.5 Comparação com Baselines

**Baselines Implementados:**

| Categoria | Método | Descrição |
|-----------|--------|-----------|
| Heurístico | Random | Ordem aleatória |
| Heurístico | Recency | Recência de falha |
| Heurístico | FailureRate | Taxa histórica |
| ML | LogisticRegression | Modelo linear |
| ML | RandomForest | Ensemble de árvores |
| ML | XGBoost | Gradient boosting |

---

## 8. Relacionamento entre Componentes

### 8.1 Diagrama de Dependências

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RELACIONAMENTO ENTRE MÓDULOS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐          ┌─────────────┐         ┌─────────────┐      │
│  │   CONFIGS   │──────────│    MAIN     │─────────│   RESULTS   │      │
│  │   (YAML)    │          │   (main.py) │         │   (CSVs)    │      │
│  └─────────────┘          └──────┬──────┘         └─────────────┘      │
│                                  │                                      │
│         ┌────────────────────────┼────────────────────────┐            │
│         │                        │                        │            │
│         ▼                        ▼                        ▼            │
│  ┌─────────────┐          ┌─────────────┐         ┌─────────────┐      │
│  │ DATA_LOADER │          │   MODELS    │         │ EVALUATION  │      │
│  │             │          │             │         │             │      │
│  │ • load      │          │ • forward   │         │ • APFD      │      │
│  │ • split     │          │ • loss      │         │ • metrics   │      │
│  │ • encode    │          │             │         │ • threshold │      │
│  └──────┬──────┘          └──────┬──────┘         └─────────────┘      │
│         │                        │                                      │
│         ├────────────────────────┤                                      │
│         │                        │                                      │
│         ▼                        ▼                                      │
│  ┌─────────────┐          ┌─────────────┐                              │
│  │ EMBEDDINGS  │          │  TRAINING   │                              │
│  │             │          │             │                              │
│  │ • SBERT     │          │ • losses    │                              │
│  │ • cache     │          │ • optimizer │                              │
│  └─────────────┘          └─────────────┘                              │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐          ┌─────────────┐                              │
│  │  FEATURES   │──────────│   GRAPHS    │                              │
│  │             │          │             │                              │
│  │ • V2.5      │  builds  │ • co-fail   │                              │
│  │ • impute    │──edges───│ • co-succ   │                              │
│  └─────────────┘          │ • semantic  │                              │
│                           └─────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Fluxo de Dados Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FLUXO DE DADOS END-TO-END                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ENTRADA                                                                │
│  ────────                                                               │
│  train.csv ─────┬──────────────────────────────────────────────────────│
│  test.csv  ─────┘                                                       │
│        │                                                               │
│        ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: CARREGAMENTO                                            │  │
│  │  data_loader.load_data() → clean_data() → encode_labels()        │  │
│  │  compute_class_weights() → split_data()                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│        │                                                               │
│        ├─────────────────────┬──────────────────────┐                  │
│        ▼                     ▼                      ▼                  │
│  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐          │
│  │   STEP 2A    │     │   STEP 2B    │      │   STEP 2C    │          │
│  │  Embeddings  │     │   Features   │      │    Graph     │          │
│  │              │     │              │      │              │          │
│  │ SBERT encode │     │ V2.5 extract │      │ Build edges  │          │
│  │ Cache check  │     │ Imputation   │      │ 3 types      │          │
│  │ 1536-dim     │     │ 10-dim       │      │ edge_index   │          │
│  └──────┬───────┘     └──────┬───────┘      └──────┬───────┘          │
│         │                    │                     │                   │
│         └────────────────────┼─────────────────────┘                   │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: CRIAÇÃO DE DATALOADERS                                  │  │
│  │  • PyTorch DataLoader com sampler balanceado                     │  │
│  │  • Batch size: 32                                                │  │
│  │  • TC_Key → global_index mapping                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│        │                                                               │
│        ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 4: TREINAMENTO                                             │  │
│  │  for epoch in range(50):                                         │  │
│  │      for batch in train_loader:                                  │  │
│  │          subgraph = extract_subgraph(batch)                      │  │
│  │          logits = model(embed, feat, subgraph)                   │  │
│  │          loss = criterion(logits, labels)                        │  │
│  │          loss.backward()                                         │  │
│  │          optimizer.step()                                        │  │
│  │      val_metrics = evaluate()                                    │  │
│  │      early_stopping.check()                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│        │                                                               │
│        ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 5: OTIMIZAÇÃO DE THRESHOLD                                 │  │
│  │  • Grid search [0.2, 0.8] step 0.05                              │  │
│  │  • Otimiza para f1_macro no validation                           │  │
│  │  • Threshold ótimo: ~0.35-0.45                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│        │                                                               │
│        ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  STEP 6: AVALIAÇÃO FINAL                                         │  │
│  │  • Aplica threshold otimizado no test set                        │  │
│  │  • Gera rankings por build                                       │  │
│  │  • Calcula APFD para 277 builds com falhas                       │  │
│  │  • Compara com baselines                                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│        │                                                               │
│        ▼                                                               │
│  SAÍDA                                                                 │
│  ─────                                                                 │
│  • best_model.pt              (checkpoint)                             │
│  • optimal_threshold.txt      (threshold + métricas)                   │
│  • prioritized_test_cases.csv (rankings)                               │
│  • apfd_per_build.csv         (APFD por build)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Interações entre Módulos

| Módulo Origem | Módulo Destino | Dados Transferidos |
|---------------|----------------|-------------------|
| `data_loader` | `embedding_manager` | Textos (TE_Summary, TC_Steps, commit) |
| `data_loader` | `feature_extractor` | DataFrame com histórico |
| `data_loader` | `graph_builder` | DataFrame de treino |
| `embedding_manager` | `model` | Tensor [N, 1536] |
| `feature_extractor` | `model` | Tensor [N, 10] |
| `graph_builder` | `model` | edge_index [2, E], edge_weights [E] |
| `model` | `losses` | Logits [B, 2] |
| `model` | `evaluation` | Probabilidades [N, 2] |
| `evaluation/apfd` | `results` | APFD por build, rankings |

---

## 9. Configurações

### 9.1 Configuração Otimizada (APFD 0.6413)

**Arquivo**: `configs/experiment_07_ranking_optimized.yaml`

```yaml
experiment:
  name: "experiment_07_ranking_optimized"
  version: "1.0"
  description: "Configuração otimizada para APFD"

data:
  train_path: "datasets/01_industry/train.csv"
  test_path: "datasets/01_industry/test.csv"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  binary_classification: true
  binary_strategy: "pass_vs_fail"
  random_seed: 42

embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"
  batch_size: 32
  max_length: 384
  use_cache: true
  cache_dir: "cache/01_industry"

model:
  type: "dual_stream_v8"

  semantic:
    input_dim: 1536
    hidden_dim: 256
    num_layers: 2
    dropout: 0.3

  structural:
    input_dim: 10  # V2.5 features
    hidden_dim: 256
    dropout: 0.3

  gnn:
    type: "GAT"
    num_layers: 1
    num_heads: 2

  fusion:
    type: "cross_attention"
    num_heads: 4
    dropout: 0.1

  classifier:
    hidden_dims: [128, 64]
    dropout: 0.4

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.00005
  weight_decay: 0.0001

  loss:
    type: "weighted_focal"
    focal_alpha: 0.75
    focal_gamma: 2.5

  early_stopping:
    patience: 12
    monitor: "val_f1_macro"
    mode: "max"

  scheduler:
    type: "cosine"
    eta_min: 0.000001

graph:
  build_graph: true
  use_multi_edge: true
  edge_types: ["co_failure", "co_success", "semantic"]
  edge_weights:
    co_failure: 1.0
    co_success: 0.5
    semantic: 0.3
  semantic_threshold: 0.75

evaluation:
  threshold_search:
    enabled: true
    range: [0.2, 0.8]
    step: 0.05
    optimize_for: "f1_macro"
```

### 9.2 Parâmetros Críticos

| Parâmetro | Valor | Impacto |
|-----------|-------|---------|
| `focal_alpha` | 0.75 | Peso da classe minoritária |
| `focal_gamma` | 2.5 | Foco em exemplos difíceis |
| `learning_rate` | 5e-5 | Provado ótimo empiricamente |
| `gnn.num_layers` | 1 | 1 layer GAT supera arquiteturas mais profundas |
| `gnn.num_heads` | 2 | Balanceamento custo/benefício |
| `threshold_range` | [0.2, 0.8] | 3x amplitude padrão |

---

## 10. Resultados e Métricas

### 10.1 Resultados Principais

| Experimento | APFD | vs Baseline | Dataset |
|-------------|------|-------------|---------|
| **Exp 07 (Otimizado)** | **0.6413** | +2.0% | Industrial |
| Exp 06 (Feature Selection) | 0.6289 | baseline | Industrial |
| Exp Phylogenetic | 0.6350 | +0.9% | Industrial |
| RTPTorrent (HikariCP) | 0.7113 | +43.9% vs untreated | RTPTorrent |

### 10.2 Estudo de Ablação

| Componente | Contribuição | p-valor |
|------------|--------------|---------|
| **Graph Attention (GAT)** | **+17.0%** | < 0.001 |
| Structural Stream | +5.3% | < 0.001 |
| Focal Loss | +4.6% | < 0.001 |
| Class Weighting | +3.5% | 0.002 |
| Semantic Stream | +1.9% | 0.087 |

**Conclusão**: GAT é o componente mais crítico do sistema.

### 10.3 Robustez Temporal

| Validação | Mean APFD | 95% CI |
|-----------|-----------|--------|
| Temporal 5-Fold CV | 0.6629 | [0.627, 0.698] |
| Sliding Window CV | 0.6279 | [0.595, 0.661] |
| Concept Drift Test | 0.6187 | [0.574, 0.661] |

### 10.4 Estrutura de Resultados

```
results/experiment_07_ranking_optimized/
├── apfd_per_build_FULL_testcsv.csv      # APFD por build (277 builds)
├── prioritized_test_cases_FULL_testcsv.csv  # Rankings com probabilidades
├── optimal_threshold.txt                 # Threshold ótimo
├── best_model.pt                         # Checkpoint do modelo
├── metrics_summary.json                  # Todas as métricas
└── logs/                                 # Logs de treinamento
```

---

## Apêndice A: Comandos de Execução

### Treinar Modelo (Industrial)

```bash
python main.py --config configs/experiment_07_ranking_optimized.yaml
```

### Treinar Modelo (RTPTorrent)

```bash
python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml
```

### Executar Ablação

```bash
python scripts/analysis/run_ablation_study.py
```

### Executar CV Temporal

```bash
python scripts/analysis/run_temporal_cv.py
```

### Comparar com Baselines

```bash
python scripts/analysis/run_all_baselines.py
```

---

## Apêndice B: Dependências

```
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
sentence-transformers>=2.2.2
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
PyYAML>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## Apêndice C: Referências

1. **SBERT**: Reimers & Gurevych (2019). Sentence-BERT.
2. **GATv2**: Brody et al. (2022). How Attentive are Graph Attention Networks?
3. **Focal Loss**: Lin et al. (2017). Focal Loss for Dense Object Detection.
4. **RTPTorrent**: Mattis et al. (2020). RTorrent Dataset.
5. **APFD**: Rothermel et al. (1999). Test Case Prioritization.

---

*Documento gerado automaticamente - Filo-Priori V9*
*Última atualização: Novembro 2025*
