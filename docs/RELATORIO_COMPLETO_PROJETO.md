# Relatório Técnico Completo: Filo-Priori V9

**Sistema de Priorização Inteligente de Casos de Teste com Graph Neural Networks**

**Versão:** 9.0.0
**Data:** 2025-11-12
**Autor:** Equipe Filo-Priori

---

## Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Problema e Motivação](#2-problema-e-motivação)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Pipeline de Execução Detalhado](#4-pipeline-de-execução-detalhado)
5. [Componentes Técnicos](#5-componentes-técnicos)
6. [Configurações e Hiperparâmetros](#6-configurações-e-hiperparâmetros)
7. [Métricas de Avaliação](#7-métricas-de-avaliação)
8. [Estrutura do Projeto](#8-estrutura-do-projeto)
9. [Workflow de Experimentação](#9-workflow-de-experimentação)
10. [Evolução do Projeto (V8 → V9)](#10-evolução-do-projeto-v8--v9)
11. [Resultados Esperados](#11-resultados-esperados)
12. [Guia de Uso](#12-guia-de-uso)

---

## 1. Visão Geral do Projeto

### 1.1 Objetivo

O **Filo-Priori** é um sistema de Deep Learning para **priorização inteligente de casos de teste (Test Case Prioritization - TCP)** em ambientes de integração contínua (CI/CD). O objetivo é ordenar casos de teste pela probabilidade de falha, permitindo detecção antecipada de defeitos e otimizando o tempo de feedback para desenvolvedores.

### 1.2 Problema Resolvido

Em projetos de software grandes:
- **Milhares de casos de teste** podem levar horas para executar
- **Feedback lento** atrasa a detecção de bugs
- **Recursos limitados** (tempo, máquinas CI/CD)
- **Testes importantes** podem ser executados por último

**Solução:** Priorizar testes com maior probabilidade de falhar, maximizando a detecção precoce de defeitos (métrica APFD).

### 1.3 Abordagem

O Filo-Priori utiliza uma **arquitetura dual-stream** que combina:
1. **Informação Semântica:** Descrições de testes e commits (texto)
2. **Informação Estrutural:** Histórico de execução e relações entre testes (grafo)

Diferentemente de abordagens tradicionais baseadas apenas em histórico, o sistema aprende representações profundas que capturam:
- Semântica dos testes (o que o teste faz)
- Padrões de co-falha (testes que falham juntos)
- Contexto de commits (mudanças no código)

---

## 2. Problema e Motivação

### 2.1 Contexto de Test Case Prioritization (TCP)

**Definição:** Ordenar casos de teste para maximizar objetivos específicos (detecção de falhas, cobertura, etc.).

**Métrica Principal - APFD (Average Percentage of Faults Detected):**

```
APFD = 1 - (TF1 + TF2 + ... + TFm) / (n * m) + 1/(2n)

Onde:
- n = número total de casos de teste
- m = número de falhas detectadas
- TFi = posição do primeiro teste que detecta a falha i
```

**Interpretação:** APFD próximo de 1.0 significa que falhas são detectadas cedo na execução.

### 2.2 Desafios

1. **Esparsidade de Falhas:** Apenas ~10-15% dos testes falham tipicamente
2. **Desbalanceamento de Classes:** 85% Pass vs 15% Fail
3. **Dados Multimodais:** Texto + Histórico + Grafo
4. **Relações Complexas:** Co-falhas, dependências de commits
5. **Generalização:** Modelo deve funcionar em builds futuros (não vistos)

### 2.3 Motivação para Arquitetura Dual-Stream

**Por que combinar Semântica + Estrutura?**

- **Semântica (texto):** Captura *o que* o teste faz e *quais* mudanças foram feitas
  - "Test login with invalid credentials" + "Fix authentication bug"
  - Embeddings densos (1536 dims cada)

- **Estrutura (grafo):** Captura *como* testes se relacionam e *padrões históricos*
  - Teste A e B falham juntos frequentemente
  - Taxa de falha, flakiness, idade do teste

**Resultado:** Predição mais robusta combinando evidências complementares.

---

## 3. Arquitetura do Sistema

### 3.1 Visão Geral - Dual-Stream Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Test Cases (TC):                                                   │
│    - TC_Key, TE_Summary, TC_Steps                                   │
│  Commits:                                                           │
│    - Commit messages, CR (Change Requests)                          │
│  Historical Data:                                                   │
│    - Build_ID, TE_Test_Result, Execution history                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   DATA PREPROCESSING & FEATURE EXTRACTION│
        └─────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  SEMANTIC STREAM      │   │  STRUCTURAL STREAM    │
    │                       │   │                       │
    │ Qodo-Embed-1-1.5B     │   │ Phylogenetic Graph    │
    │ (1.5B parameters)     │   │ + GAT (Graph          │
    │                       │   │   Attention Network)  │
    │ TC Embedding: 1536    │   │                       │
    │ Commit Emb:   1536    │   │ Structural Features:  │
    │ ─────────────────     │   │ - Failure rate        │
    │ Combined:     3072    │   │ - Test age            │
    │                       │   │ - Flakiness           │
    │ Output: [B, 256]      │   │ - Recent history      │
    │                       │   │                       │
    │                       │   │ Output: [B, 256]      │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  FUSION LAYER       │
                    │  (Cross-Attention)  │
                    │                     │
                    │  4-head attention   │
                    │  Query: Semantic    │
                    │  Key/Value: Struct  │
                    │                     │
                    │  Output: [B, 256]   │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  CLASSIFIER         │
                    │                     │
                    │  256 → 128 → 64 → 2 │
                    │  (Pass / Fail)      │
                    │                     │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  OUTPUT             │
                    │                     │
                    │  - Risk scores      │
                    │  - Test ranking     │
                    │  - APFD calculation │
                    └─────────────────────┘
```

### 3.2 Componentes Principais

#### 3.2.1 Semantic Stream (Fluxo Semântico)

**Função:** Processar informação textual (testes e commits)

**Modelo:** Qodo-Embed-1-1.5B (1.5 bilhões de parâmetros)
- Estado da arte em embeddings semânticos
- Suporta textos até 32k tokens
- Pré-treinado em código e documentação técnica

**Encoding Separado (Novidade V9):**

1. **Test Case Encoding:**
   ```
   Input:  TE_Summary + TC_Steps
   Output: [batch_size, 1536]
   ```

2. **Commit Encoding:**
   ```
   Input:  Preprocessed commit messages
   Output: [batch_size, 1536]
   ```

3. **Concatenação:**
   ```
   Combined: [batch_size, 3072]
   ```

**Arquitetura do Stream:**
```python
SemanticStream:
  Input Projection: 3072 → 256
  FFN Layers (x2):
    - Linear: 256 → 1024
    - GELU activation
    - Dropout (0.15)
    - Linear: 1024 → 256
    - LayerNorm
    - Residual connection
  Output: [batch_size, 256]
```

**Por que Encoding Separado?**
- Preserva estrutura semântica de cada tipo de informação
- Evita mistura prematura de contextos diferentes
- Dobra capacidade informacional (1024 → 3072 dims)

#### 3.2.2 Structural Stream (Fluxo Estrutural)

**Função:** Processar features históricas e relações em grafo

**Componente A - Features Estruturais (6 features):**

1. **Failure Rate:** Taxa de falha histórica do teste
   ```
   failure_rate = num_failures / total_executions
   ```

2. **Recent Failure Rate:** Taxa de falha nas últimas N execuções
   ```
   recent_failure_rate = failures_last_5 / 5
   ```

3. **Test Age:** Idade do teste (número de builds desde criação)
   ```
   test_age = current_build_id - first_build_id
   ```

4. **Avg Time Since Last Execution:** Tempo médio desde última execução
   ```
   avg_time = mean(time_between_executions)
   ```

5. **Flakiness Score:** Medida de instabilidade (Pass → Fail → Pass)
   ```
   flakiness = transitions / total_executions
   ```

6. **Last Result:** Resultado da última execução (0=Pass, 1=Fail)

**Componente B - Phylogenetic Graph (Grafo Filogenético):**

**Tipo:** Co-Failure Graph
- **Nós:** Test Cases (TC_Key)
- **Arestas:** Existem se testes falharam juntos no mesmo Build_ID
- **Pesos:** Probabilidade condicional P(A falha | B falha)

```
weight(A, B) = co_failures(A, B) / total_failures(B)

Onde:
- co_failures(A, B) = número de builds onde A e B falharam juntos
- total_failures(B) = número total de falhas de B
```

**Construção do Grafo:**
```python
1. Para cada Build_ID:
   - Identificar testes que falharam
   - Criar arestas entre todos pares de testes que falharam

2. Agregar co-falhas:
   - Contar co-ocorrências
   - Calcular pesos (probabilidade condicional)

3. Filtrar arestas:
   - Manter apenas arestas com min_co_occurrences ≥ 2
   - Manter apenas arestas com weight ≥ 0.1
```

**Arquitetura - Graph Attention Network (GAT):**

```python
StructuralStreamV8:
  Input Features: [num_nodes, 6]

  GAT Layer 1:
    - 4 attention heads
    - 6 → 256 dimensions
    - Edge weights incorporated
    - ELU activation
    - Dropout (0.15)

  GAT Layer 2:
    - 4 attention heads
    - 256 → 256 dimensions
    - ELU activation
    - Dropout (0.15)

  Output: [batch_size, 256]
```

**Atenção Multi-Head (GAT):**
- Cada head aprende padrões diferentes de co-falha
- Agregação ponderada de vizinhos no grafo
- Permite capturar relações complexas entre testes

#### 3.2.3 Fusion Layer (Camada de Fusão)

**Função:** Combinar informação semântica e estrutural

**Tipo:** Cross-Attention (Atenção Cruzada)

```python
CrossAttentionFusion:
  Query:     Semantic features [B, 256]
  Key:       Structural features [B, 256]
  Value:     Structural features [B, 256]

  Attention:
    - 4 attention heads
    - Scaled dot-product attention
    - Dropout (0.1)

  Output:
    - Attended features [B, 256]
    - Residual connection
    - Layer normalization
```

**Intuição:**
- "Quais features estruturais são relevantes dado o contexto semântico?"
- Semantic features fazem Query: "O que é importante?"
- Structural features respondem: "Aqui está o padrão histórico relevante"

#### 3.2.4 Classifier (Classificador)

**Função:** Predizer probabilidade de falha

```python
Classifier:
  Input: [batch_size, 256]

  Hidden Layer 1: 256 → 128
    - Linear transformation
    - ReLU activation
    - Dropout (0.25)

  Hidden Layer 2: 128 → 64
    - Linear transformation
    - ReLU activation
    - Dropout (0.25)

  Output Layer: 64 → 2 (Pass / Fail)
    - Linear transformation
    - Softmax (applied during loss)

  Output: [batch_size, 2] (logits)
```

**Saída Final:**
```python
logits = model(semantic_emb, structural_features, edge_index)
probs = softmax(logits)  # [batch_size, 2]

# Risk score para ranking
risk_score = probs[:, 1]  # Probabilidade de Fail

# Ordenar testes por risk_score (maior primeiro)
ranked_tests = argsort(risk_score, descending=True)
```

---

## 4. Pipeline de Execução Detalhado

### 4.1 Fase 1: Preparação de Dados

#### Passo 1.1: Carregamento de Dados

```python
# Entrada: datasets/train.csv, datasets/test.csv
DataLoader:
  - Carregar CSVs com pandas
  - Validar colunas obrigatórias:
    * TC_Key (identificador único do teste)
    * Build_ID (identificador do build)
    * TE_Summary (descrição do teste)
    * TC_Steps (passos do teste)
    * TE_Test_Result (Pass / Fail)
    * commit (lista de commits)
    * CR (Change Requests)

  - Split dataset:
    * Train: 80%
    * Val: 10%
    * Test: 10%
```

**Formato de Dados:**
```csv
TC_Key,Build_ID,TE_Summary,TC_Steps,TE_Test_Result,commit,CR
TC_001,B_100,"Test login","1. Open app\n2. Enter credentials",Pass,"['abc123', 'def456']","['CR-001']"
TC_002,B_100,"Test logout","1. Click logout",Fail,"['abc123']","[]"
...
```

#### Passo 1.2: Extração de Commits

```python
CommitExtractor:
  Input: df['commit'] (lista de commits como string)

  Process:
    1. Parse string para lista: "['abc', 'def']" → ['abc', 'def']
    2. Limitar a max_commits_per_tc (default: 10)
    3. Concatenar commits: "abc def"
    4. Truncar para max_commit_length (default: 1024 chars)

  Output: Lista de strings de commits processados
```

#### Passo 1.3: Extração de Features Estruturais

```python
StructuralFeatureExtractor:
  Input: DataFrame com histórico completo

  Para cada TC_Key:
    1. Coletar execuções históricas (window=5 builds recentes)
    2. Calcular:
       - failure_rate
       - recent_failure_rate
       - test_age
       - avg_time_since_last_exec
       - flakiness_score
       - last_result
    3. Imputar valores faltantes (mean/median)

  Output: [num_tests, 6] numpy array
```

**Exemplo de Feature Extraction:**
```python
TC_001 execution history:
  Build_100: Pass
  Build_101: Fail
  Build_102: Pass
  Build_103: Pass
  Build_104: Fail

Features:
  failure_rate = 2/5 = 0.4
  recent_failure_rate = 2/5 = 0.4
  test_age = 104 - 100 = 4
  flakiness_score = 2/5 = 0.4  (2 transitions)
  last_result = 1 (Fail)
```

#### Passo 1.4: Construção do Grafo Filogenético

```python
PhylogeneticGraphBuilder:
  Input: Training DataFrame

  1. Identificar co-falhas:
     Para cada Build_ID:
       - failed_tests = df[df.TE_Test_Result == 'Fail'].TC_Key
       - Para cada par (A, B) em failed_tests:
           co_failure_count[(A, B)] += 1

  2. Calcular pesos:
     Para cada par (A, B):
       weight = co_failure_count[(A, B)] / total_failures[B]

  3. Filtrar:
     - Manter se co_failure_count ≥ min_co_occurrences (2)
     - Manter se weight ≥ weight_threshold (0.1)

  4. Converter para PyTorch Geometric:
     edge_index = [[src_nodes], [dst_nodes]]  # [2, num_edges]
     edge_weight = [weights]  # [num_edges]

  Output: edge_index, edge_weight
```

**Exemplo de Grafo:**
```
Build_100: TC_001 (Fail), TC_002 (Fail), TC_003 (Pass)
Build_101: TC_001 (Fail), TC_002 (Fail), TC_004 (Fail)
Build_102: TC_001 (Fail), TC_003 (Fail)

Co-failures:
  (TC_001, TC_002): 2 builds
  (TC_001, TC_004): 1 build
  (TC_001, TC_003): 1 build
  (TC_002, TC_004): 1 build

Edges (min_co_occurrences=2):
  TC_001 <-> TC_002 (weight=2/2=1.0)
```

### 4.2 Fase 2: Extração de Embeddings Semânticos

```python
QodoEncoder:
  Model: Qodo-Embed-1-1.5B (SentenceTransformer)
  Device: CUDA

  1. Encode Test Cases:
     Input: TE_Summary + TC_Steps
     Example: "Test login functionality\n1. Open app\n2. Enter credentials\n3. Click login"
     Output: [num_tests, 1536]

  2. Encode Commits:
     Input: Preprocessed commit messages
     Example: "Fix authentication bug in login module"
     Output: [num_tests, 1536]

  3. Concatenate:
     combined_embedding = concat([tc_emb, commit_emb], dim=1)
     Output: [num_tests, 3072]

  4. Cache:
     - Salvar em cache/embeddings_qodo/
     - Formato: .npy (numpy array)
     - Reutilizar em próximas execuções
```

**Processamento em Batch:**
```python
batch_size = 32

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(
        batch,
        normalize_embeddings=True,  # L2 normalization
        show_progress_bar=True
    )
```

### 4.3 Fase 3: Treinamento do Modelo

#### Passo 3.1: Inicialização

```python
Model Initialization:
  - SemanticStream(input_dim=3072, hidden_dim=256)
  - StructuralStreamV8(input_dim=6, hidden_dim=256, num_heads=4)
  - CrossAttentionFusion(dim=256, num_heads=4)
  - Classifier(hidden_dims=[128, 64], num_classes=2)

Optimizer: AdamW
  - learning_rate = 7.5e-5
  - weight_decay = 3e-5
  - betas = (0.9, 0.999)

Scheduler: CosineAnnealingLR
  - T_max = 50 epochs
  - eta_min = 1e-6

Loss: Weighted Cross-Entropy
  - class_weights = [1.0, 1.0]  # Balanced (V9)
  - label_smoothing = 0.0
```

#### Passo 3.2: Training Loop

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        # Forward pass
        semantic_emb = batch['semantic_embeddings']  # [B, 3072]
        structural_features = batch['structural_features']  # [B, 6]
        edge_index = batch['edge_index']  # [2, num_edges]
        labels = batch['labels']  # [B]

        # Get subgraph for current batch
        batch_edge_index, batch_edge_weight = subgraph(
            batch['node_indices'],
            edge_index,
            edge_weight
        )

        # Model forward
        logits = model(
            semantic_emb,
            structural_features,
            batch_edge_index,
            batch_edge_weight
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(...)
            val_loss = criterion(logits, labels)

            # Compute metrics
            preds = torch.argmax(logits, dim=1)
            accuracy, f1, precision, recall = compute_metrics(preds, labels)

    # Learning rate scheduling
    scheduler.step()

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_checkpoint(model, 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 4.4 Fase 4: Avaliação e Ranking

#### Passo 4.1: Predição

```python
model.eval()
with torch.no_grad():
    logits = model(test_semantic_emb, test_structural_features, test_edge_index)
    probs = F.softmax(logits, dim=1)
    risk_scores = probs[:, 1]  # Probabilidade de Fail
```

#### Passo 4.2: Threshold Optimization

```python
# Search for best threshold on validation set
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.10, 0.60, 0.05):
    preds = (val_risk_scores >= threshold).long()
    f1 = f1_score(val_labels, preds, average='macro')

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply to test set
test_preds = (test_risk_scores >= best_threshold).long()
```

#### Passo 4.3: Test Case Ranking

```python
# Criar DataFrame com resultados
results_df = pd.DataFrame({
    'TC_Key': test_tc_keys,
    'Build_ID': test_build_ids,
    'risk_score': test_risk_scores.cpu().numpy(),
    'predicted': test_preds.cpu().numpy(),
    'actual': test_labels.cpu().numpy()
})

# Ordenar por Build_ID e risk_score
results_df = results_df.sort_values(['Build_ID', 'risk_score'], ascending=[True, False])

# Salvar ranking
results_df.to_csv('prioritized_test_cases.csv', index=False)
```

#### Passo 4.4: APFD Calculation

```python
# Calcular APFD por build
apfd_results = []

for build_id in results_df['Build_ID'].unique():
    df_build = results_df[results_df['Build_ID'] == build_id]

    # Skip se não há falhas
    if df_build['actual'].sum() == 0:
        continue

    # Business rule: 1 TC → APFD = 1.0
    if len(df_build) == 1:
        apfd = 1.0
    else:
        # Calcular APFD
        ranks = np.arange(1, len(df_build) + 1)
        labels = df_build['actual'].values

        fault_positions = ranks[labels == 1]
        n = len(df_build)
        m = len(fault_positions)

        if m == 0:
            apfd = None
        else:
            apfd = 1 - (fault_positions.sum() / (n * m)) + 1/(2*n)

    apfd_results.append({
        'Build_ID': build_id,
        'APFD': apfd,
        'num_tests': len(df_build),
        'num_failures': df_build['actual'].sum()
    })

# Mean APFD
mean_apfd = np.mean([r['APFD'] for r in apfd_results if r['APFD'] is not None])
```

---

## 5. Componentes Técnicos

### 5.1 Módulos de Pré-processamento

#### 5.1.1 DataLoader (`src/preprocessing/data_loader.py`)

**Responsabilidades:**
- Carregar datasets (train.csv, test.csv)
- Validar schema de dados
- Split train/val/test
- Calcular class weights para balanceamento

**API:**
```python
data_loader = DataLoader(config)
data_dict = data_loader.prepare_dataset(sample_size=None)

# Output:
{
    'train': DataFrame,
    'val': DataFrame,
    'test': DataFrame,
    'class_weights': Tensor
}
```

#### 5.1.2 CommitExtractor (`src/preprocessing/commit_extractor.py`)

**Responsabilidades:**
- Extrair textos de commits de DataFrame
- Preprocessar (concatenar, truncar)
- Lidar com diferentes formatos (string, lista)

**API:**
```python
extractor = CommitExtractor(config)
commits = extractor.extract_from_dataframe(df, column='commit')

# Output: List[str]
```

#### 5.1.3 StructuralFeatureExtractor (`src/preprocessing/structural_feature_extractor.py`)

**Responsabilidades:**
- Extrair 6 features estruturais
- Calcular estatísticas por TC_Key
- Imputar valores faltantes

**Features Extraídas:**
1. `failure_rate`
2. `recent_failure_rate`
3. `test_age`
4. `avg_time_since_last_exec`
5. `flakiness_score`
6. `last_result`

**API:**
```python
features = extract_structural_features(
    df=df_train,
    recent_window=5,
    min_history=2
)

# Output: np.ndarray [num_tests, 6]
```

### 5.2 Módulos de Embeddings

#### 5.2.1 QodoEncoder (`src/embeddings/qodo_encoder.py`)

**Responsabilidades:**
- Carregar modelo Qodo-Embed-1-1.5B
- Encoding separado de TCs e Commits
- Caching de embeddings
- Gerenciamento de GPU/CUDA

**API:**
```python
encoder = QodoEncoder(config, device='cuda')

# Encoding separado
tc_embeddings, commit_embeddings = encoder.encode_dataset_separate(
    summaries=summaries,
    steps=steps,
    commit_texts=commits,
    cache_dir='cache/embeddings_qodo',
    split_name='train'
)

# Output:
# tc_embeddings: [num_tests, 1536]
# commit_embeddings: [num_tests, 1536]
```

**Features:**
- Batch processing (batch_size=32)
- L2 normalization
- CUDA error handling
- Progress bar
- Automatic caching

### 5.3 Módulos de Grafo

#### 5.3.1 PhylogeneticGraphBuilder (`src/phylogenetic/phylogenetic_graph_builder.py`)

**Responsabilidades:**
- Construir grafo de co-falhas
- Calcular pesos (probabilidade condicional)
- Filtrar arestas espúrias
- Gerar estruturas PyTorch Geometric

**API:**
```python
builder = PhylogeneticGraphBuilder(
    graph_type='co_failure',
    min_co_occurrences=2,
    weight_threshold=0.1
)

# Fit no training set
builder.fit(df_train)

# Transform para obter edge_index
edge_index, edge_weight = builder.transform(df_train)

# Output:
# edge_index: [2, num_edges]
# edge_weight: [num_edges]
```

**Tipos de Grafo Suportados:**
- `co_failure`: Testes que falharam juntos
- `commit_dependency`: Testes afetados pelos mesmos commits
- `hybrid`: Combinação de ambos

### 5.4 Módulos de Modelo

#### 5.4.1 DualStreamV8 (`src/models/dual_stream_v8.py`)

**Arquitetura Completa:**

```python
class DualStreamModelV8(nn.Module):
    def __init__(self, config):
        # Semantic stream
        self.semantic_stream = SemanticStream(
            input_dim=3072,
            hidden_dim=256,
            num_layers=2,
            dropout=0.15
        )

        # Structural stream
        self.structural_stream = StructuralStreamV8(
            input_dim=6,
            hidden_dim=256,
            num_heads=4,
            dropout=0.15
        )

        # Fusion layer
        self.fusion = CrossAttentionFusion(
            dim=256,
            num_heads=4,
            dropout=0.1
        )

        # Classifier
        self.classifier = Classifier(
            input_dim=256,
            hidden_dims=[128, 64],
            num_classes=2,
            dropout=0.25
        )

    def forward(self, semantic_emb, structural_features, edge_index, edge_weight=None):
        # Process semantic
        sem_out = self.semantic_stream(semantic_emb)

        # Process structural with graph
        struct_out = self.structural_stream(
            structural_features,
            edge_index,
            edge_weight
        )

        # Fuse
        fused = self.fusion(sem_out, struct_out)

        # Classify
        logits = self.classifier(fused)

        return logits
```

#### 5.4.2 CrossAttentionFusion (`src/models/cross_attention.py`)

**Implementação:**

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_features, structural_features):
        # semantic_features: [B, 256] - Query
        # structural_features: [B, 256] - Key, Value

        # Expand dims for attention
        sem = semantic_features.unsqueeze(1)  # [B, 1, 256]
        struct = structural_features.unsqueeze(1)  # [B, 1, 256]

        # Cross-attention
        attn_out, attn_weights = self.attention(
            query=sem,
            key=struct,
            value=struct
        )

        # Residual + norm
        out = self.norm(sem + self.dropout(attn_out))

        return out.squeeze(1)  # [B, 256]
```

### 5.5 Módulos de Treinamento

#### 5.5.1 Losses (`src/training/losses.py`)

**Weighted Cross-Entropy:**
```python
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
```

**Focal Loss (alternativa):**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        # Implementação do Focal Loss para lidar com desbalanceamento
        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

### 5.6 Módulos de Avaliação

#### 5.6.1 Metrics (`src/evaluation/metrics.py`)

**Métricas Implementadas:**

```python
def compute_metrics(preds, labels, probs=None):
    """
    Compute comprehensive metrics

    Returns:
        {
            'accuracy': float,
            'f1_macro': float,
            'f1_weighted': float,
            'precision_macro': float,
            'recall_macro': float,
            'auprc_macro': float,  # Area Under Precision-Recall Curve
            'confusion_matrix': np.ndarray,
            'prediction_diversity': float  # % de predições em cada classe
        }
    """
```

**Métricas Principais:**

1. **Accuracy:** Acurácia geral
2. **F1 Macro:** F1 médio entre classes (balanceado)
3. **F1 Weighted:** F1 ponderado por suporte
4. **Precision/Recall:** Por classe
5. **AUPRC:** Área sob curva Precision-Recall
6. **Prediction Diversity:** Evitar colapso de classe única

#### 5.6.2 APFD (`src/evaluation/apfd.py`)

**Funções Principais:**

```python
def calculate_apfd_single_build(ranks, labels):
    """
    Calculate APFD for a single build

    Args:
        ranks: Test positions [1, 2, 3, ...]
        labels: Test results [0, 1, 0, ...]  (0=Pass, 1=Fail)

    Returns:
        APFD score [0.0, 1.0]
    """

def generate_apfd_report(df_results, expected_builds=277):
    """
    Generate APFD report for all builds

    Returns:
        DataFrame with columns:
        - method_name
        - build_id
        - test_scenario
        - count_tc
        - count_commits
        - apfd
        - time
    """

def print_apfd_summary(apfd_report):
    """
    Print summary statistics:
    - Mean APFD
    - Median APFD
    - Std APFD
    - Min/Max APFD
    - Number of builds evaluated
    """
```

---

## 6. Configurações e Hiperparâmetros

### 6.1 Arquivo de Configuração (`configs/experiment.yaml`)

#### 6.1.1 Experiment Settings

```yaml
experiment:
  name: "v9_qodo"
  version: "9.0.0"
  description: "V9 with Qodo-Embed-1-1.5B - separate TC and Commit encodings"
  seed: 42  # Reprodutibilidade
```

#### 6.1.2 Data Configuration

```yaml
data:
  train_path: "datasets/train.csv"
  test_path: "datasets/test.csv"

  # Splits
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42

  # Binary classification
  binary_classification: true
  binary_strategy: "pass_vs_fail"
  binary_positive_class: "Pass"
  binary_negative_class: "Fail"

  # SMOTE (desabilitado em V9)
  smote:
    enabled: false
```

#### 6.1.3 Text Processing

```yaml
text:
  num_commits_to_keep: 5
  max_commit_length: 1024  # Qodo suporta até 32k
  max_summary_length: 512
  max_steps_length: 1024
```

#### 6.1.4 Semantic Embeddings (Qodo-Embed)

```yaml
semantic:
  model_name: "Qodo/Qodo-Embed-1-1.5B"
  embedding_dim: 1536  # Single embedding
  combined_embedding_dim: 3072  # TC + Commit
  max_length: 512
  batch_size: 32
  cuda_retries: 3
  normalize_embeddings: true
  cache_path: "cache/embeddings_qodo"

  # Separate encoding (NOVIDADE V9)
  separate_encoding: true
  encode_tc_separately: true
  encode_commit_separately: true
```

#### 6.1.5 Structural Features

```yaml
structural:
  extractor:
    recent_window: 5  # Últimas 5 execuções
    min_history: 2    # Mínimo de histórico
    cache_path: "cache/structural_features.pkl"
  input_dim: 6  # 6 features
```

#### 6.1.6 Phylogenetic Graph

```yaml
graph:
  type: "co_failure"
  min_co_occurrences: 2  # Mínimo 2 co-falhas
  weight_threshold: 0.1  # Peso mínimo 0.1
  cache_path: "cache/phylogenetic_graph.pkl"
  build_graph: true
```

#### 6.1.7 Model Architecture

```yaml
model:
  type: "dual_stream_v9"

  semantic:
    input_dim: 3072
    hidden_dim: 256
    num_layers: 2
    dropout: 0.15
    activation: "gelu"

  structural:
    input_dim: 6
    hidden_dim: 256
    num_heads: 4  # GAT multi-head
    dropout: 0.15
    activation: "elu"
    use_edge_weights: true

  fusion:
    type: "cross_attention"
    num_heads: 4
    dropout: 0.1

  classifier:
    hidden_dims: [128, 64]
    dropout: 0.25

  num_classes: 2
```

#### 6.1.8 Training Hyperparameters

```yaml
training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 7.5e-5  # AdamW
  weight_decay: 3e-5

  optimizer:
    type: "AdamW"
    betas: [0.9, 0.999]
    eps: 1e-8

  scheduler:
    type: "cosine"  # Cosine annealing
    T_max: 50
    eta_min: 1e-6

  grad_clip_norm: 1.0  # Gradient clipping

  early_stopping:
    enabled: true
    patience: 20
    monitor: "val_f1_macro"
    mode: "max"
    min_delta: 0.001

  use_amp: false  # Mixed precision (desabilitado)
```

#### 6.1.9 Loss Configuration

```yaml
loss:
  type: "weighted_ce"

  weighted_ce:
    use_class_weights: false  # Balanceado em V9
    class_weights: [60.0, 1.0]  # Não usado
    label_smoothing: 0.0
```

#### 6.1.10 Evaluation Settings

```yaml
evaluation:
  metrics:
    - "accuracy"
    - "f1_macro"
    - "f1_weighted"
    - "precision"
    - "recall"
    - "auprc"
    - "confusion_matrix"
    - "prediction_diversity"

  temperature_scaling:
    enabled: true
    initial_T: 1.0

  threshold_search:
    enabled: true
    search_range: [0.10, 0.60]
    search_step: 0.05
    metric: "f1_macro"
```

#### 6.1.11 APFD Configuration

```yaml
apfd:
  expected_builds: 277  # Total de builds esperado
  count_tc_1_rule: true  # APFD=1.0 se apenas 1 TC
  only_fail_results: true  # Considerar apenas builds com falhas
  save_prioritized_tests: true
  save_apfd_per_build: true
```

### 6.2 Hiperparâmetros Críticos

#### 6.2.1 Learning Rate

**Valor:** 7.5e-5

**Justificativa:**
- Modelo grande (1.5B params no encoder)
- Fine-tuning suave necessário
- Cosine annealing para convergência estável

#### 6.2.2 Dropout Rates

**Valores:**
- Semantic stream: 0.15
- Structural stream: 0.15
- Fusion layer: 0.1
- Classifier: 0.25

**Justificativa:**
- Menor dropout em fusion (informação crítica)
- Maior dropout em classifier (prevenir overfitting)

#### 6.2.3 Batch Size

**Valor:** 32

**Trade-offs:**
- Maior batch: Gradientes mais estáveis, mais memória GPU
- Menor batch: Menos memória, convergência mais ruidosa
- 32: Balanço ideal para GPU 11GB

#### 6.2.4 Number of Epochs

**Valor:** 50 (com early stopping)

**Early Stopping:**
- Patience: 20 épocas
- Monitor: val_f1_macro
- Evita overfitting

---

## 7. Métricas de Avaliação

### 7.1 Métricas de Classificação

#### 7.1.1 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Limitação:** Pode ser enganosa com dados desbalanceados

**Exemplo:**
- 90% Pass, 10% Fail
- Modelo que sempre prediz Pass: 90% accuracy (ruim!)

#### 7.1.2 F1 Score (Macro)

```
F1_macro = (F1_Pass + F1_Fail) / 2

Onde:
F1_class = 2 * (Precision * Recall) / (Precision + Recall)
```

**Por que Macro?**
- Trata ambas classes igualmente
- Não é dominada pela classe majoritária
- Melhor métrica para dados desbalanceados

**Exemplo:**
```
Pass:
  Precision = 0.95, Recall = 0.92
  F1_Pass = 2 * (0.95 * 0.92) / (0.95 + 0.92) = 0.935

Fail:
  Precision = 0.30, Recall = 0.40
  F1_Fail = 2 * (0.30 * 0.40) / (0.30 + 0.40) = 0.343

F1_Macro = (0.935 + 0.343) / 2 = 0.639
```

#### 7.1.3 Precision e Recall

```
Precision = TP / (TP + FP)  # "Quantos preditos Fail realmente falharam?"
Recall = TP / (TP + FN)     # "Quantas falhas reais foram detectadas?"
```

**Trade-off:**
- Alta Precision: Poucos falsos positivos
- Alto Recall: Detecta mais falhas reais

**No contexto de TCP:**
- Alto Recall para Fail: Importante! (detectar mais falhas)
- Precision: Menos crítico (executar alguns passes extras é aceitável)

#### 7.1.4 AUPRC (Area Under Precision-Recall Curve)

```
AUPRC = ∫ Precision(Recall) dRecall
```

**Vantagens:**
- Melhor que AUC-ROC para dados desbalanceados
- Foca na classe minoritária (Fail)
- Range: [0.0, 1.0]

**Interpretação:**
- AUPRC = 0.5: Bom modelo
- AUPRC = 0.7: Muito bom
- AUPRC > 0.8: Excelente

#### 7.1.5 Prediction Diversity

```
Diversity = min(% Predições Pass, % Predições Fail)
```

**Função:** Detectar colapso de predição

**Exemplo:**
- Modelo prediz 100% Pass: Diversity = 0% (colapso!)
- Modelo prediz 70% Pass, 30% Fail: Diversity = 30% (ok)

**Critério:** Diversity ≥ 10% para ambas classes

### 7.2 Métrica Principal: APFD

#### 7.2.1 Definição Formal

```
APFD = 1 - (TF1 + TF2 + ... + TFm) / (n * m) + 1/(2n)

Onde:
- n = número total de casos de teste no build
- m = número de falhas detectadas
- TFi = posição (rank) do primeiro teste que detecta a falha i
```

#### 7.2.2 Interpretação

**Range:** [0.0, 1.0]

**Valores:**
- **APFD = 1.0:** Todas falhas detectadas no início (ideal)
- **APFD = 0.5:** Detecção aleatória (baseline)
- **APFD < 0.5:** Pior que aleatório

**Exemplo Prático:**

Build com 10 testes, 3 falhas:

```
Ordenação 1 (Ruim):
  Posição: 1  2  3  4  5  6  7  8  9  10
  Resultado: P  P  P  P  P  F  F  P  F  P
  Falhas em: 6, 7, 9

  APFD = 1 - (6 + 7 + 9) / (10 * 3) + 1/20
       = 1 - 22/30 + 0.05
       = 0.317

Ordenação 2 (Boa):
  Posição: 1  2  3  4  5  6  7  8  9  10
  Resultado: F  F  P  F  P  P  P  P  P  P
  Falhas em: 1, 2, 4

  APFD = 1 - (1 + 2 + 4) / (10 * 3) + 1/20
       = 1 - 7/30 + 0.05
       = 0.817
```

#### 7.2.3 Regras de Negócio (Business Rules)

1. **APFD calculado POR BUILD:**
   - Não global
   - Cada build é independente

2. **Apenas builds com falhas:**
   - Builds 100% Pass: Ignorados
   - Justificativa: APFD não aplicável sem falhas

3. **Build com 1 único TC:**
   - APFD = 1.0 (regra de negócio)
   - Justificativa: Única escolha possível

4. **Agregação:**
   - Mean APFD = média sobre todos builds válidos
   - Mediana, Std também reportados

#### 7.2.4 Cálculo no Sistema

```python
def calculate_apfd_per_build(df_results):
    apfd_scores = []

    for build_id in df_results['Build_ID'].unique():
        df_build = df_results[df_results['Build_ID'] == build_id]

        # Ordenar por risk_score (já priorizado)
        df_build = df_build.sort_values('risk_score', ascending=False)

        # Pegar ranks e labels
        ranks = np.arange(1, len(df_build) + 1)
        labels = df_build['actual'].values

        # Contar falhas
        num_failures = labels.sum()

        # Regra 1: Skip se sem falhas
        if num_failures == 0:
            continue

        # Regra 2: APFD = 1.0 se apenas 1 TC
        if len(df_build) == 1:
            apfd = 1.0
        else:
            # Calcular APFD normal
            fault_positions = ranks[labels == 1]
            n = len(df_build)
            m = num_failures

            apfd = 1 - (fault_positions.sum() / (n * m)) + 1/(2*n)

        apfd_scores.append({
            'Build_ID': build_id,
            'APFD': apfd,
            'num_tests': len(df_build),
            'num_failures': num_failures
        })

    # Mean APFD
    mean_apfd = np.mean([s['APFD'] for s in apfd_scores])

    return apfd_scores, mean_apfd
```

### 7.3 Critérios de Sucesso

#### 7.3.1 Métricas de Classificação (Test Set)

| Métrica | Baseline (Random) | Target V9 | Excelente |
|---------|-------------------|-----------|-----------|
| Accuracy | ~0.85 | ≥ 0.60 | ≥ 0.70 |
| F1 Macro | ~0.50 | ≥ 0.54 | ≥ 0.60 |
| Recall (Fail) | ~0.15 | ≥ 0.20 | ≥ 0.40 |
| Precision (Fail) | ~0.15 | ≥ 0.25 | ≥ 0.35 |
| AUPRC | ~0.15 | ≥ 0.30 | ≥ 0.50 |
| Prediction Diversity | 0-100% | ≥ 10% | ≥ 20% |

#### 7.3.2 Métrica APFD

| Configuração | Mean APFD | Status |
|--------------|-----------|--------|
| Random Baseline | ~0.50 | Baseline |
| V8 (BGE fine-tuned) | 0.5481 | Previous |
| V8 Improved | 0.5967 | Best Previous |
| **V9 Target** | **≥ 0.58** | **Target** |
| Excellent | ≥ 0.65 | Stretch Goal |

#### 7.3.3 Critérios de Falha (Experiment Rejection)

**Rejeitar experimento se:**

1. **Colapso de Predição:**
   - Prediction Diversity < 5% para qualquer classe
   - Modelo predizendo apenas uma classe

2. **Performance Inferior a Baseline:**
   - Mean APFD < 0.50 (pior que aleatório)
   - F1 Macro < 0.45

3. **Recall Crítico:**
   - Recall (Fail) < 10%
   - Modelo não detecta falhas

4. **Instabilidade:**
   - Train-Val gap > 0.20 (overfitting severo)
   - Loss divergindo (NaN ou Inf)

---

## 8. Estrutura do Projeto

### 8.1 Organização de Diretórios

```
filo_priori_v8/
│
├── main.py                          # Entry point principal
├── setup_experiment.sh              # Script de setup do ambiente
├── run_experiment.sh                # Script de execução de experimentos
├── requirements.txt                 # Dependências do projeto
│
├── configs/                         # Configurações
│   └── experiment.yaml              # Config ativa (ÚNICA)
│
├── src/                             # Código fonte
│   ├── __init__.py
│   │
│   ├── embeddings/                  # Módulos de embeddings
│   │   ├── __init__.py
│   │   ├── qodo_encoder.py          # Qodo-Embed-1-1.5B (V9)
│   │   ├── semantic_encoder.py      # BGE encoder (legacy)
│   │   └── commit_extractor.py      # Extração de commits
│   │
│   ├── preprocessing/               # Pré-processamento
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Carregamento de dados
│   │   ├── text_processor.py        # Processamento de texto
│   │   ├── commit_extractor.py      # Extração de commits
│   │   ├── structural_feature_extractor.py  # Features estruturais
│   │   └── structural_feature_imputation.py # Imputação de features
│   │
│   ├── phylogenetic/                # Grafos filogenéticos
│   │   ├── __init__.py
│   │   ├── phylogenetic_graph_builder.py  # Construção de grafos
│   │   ├── graph_rewiring.py        # Rewiring de grafos
│   │   └── link_prediction.py       # Predição de links
│   │
│   ├── models/                      # Modelos de rede neural
│   │   ├── __init__.py
│   │   ├── dual_stream_v8.py        # Modelo principal (V8/V9)
│   │   ├── dual_stream.py           # Versão anterior
│   │   └── cross_attention.py       # Camada de fusão
│   │
│   ├── layers/                      # Camadas customizadas
│   │   ├── gatv2.py                 # Graph Attention V2
│   │   ├── attention_pooling.py     # Attention pooling
│   │   └── denoising_gate.py        # Denoising gates
│   │
│   ├── training/                    # Módulos de treinamento
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   └── losses.py                # Loss functions
│   │
│   ├── evaluation/                  # Avaliação
│   │   ├── __init__.py
│   │   ├── metrics.py               # Métricas (F1, Precision, etc.)
│   │   └── apfd.py                  # Cálculo de APFD
│   │
│   └── monitoring/                  # Monitoramento
│       ├── __init__.py
│       ├── advanced_monitor.py      # Monitoramento avançado
│       └── metrics_composer.py      # Composição de métricas
│
├── datasets/                        # Dados de entrada
│   ├── train.csv                    # Dados de treino
│   └── test.csv                     # Dados de teste
│
├── cache/                           # Cache de features/embeddings
│   ├── embeddings_qodo/             # Embeddings do Qodo
│   │   ├── train_tc_embeddings.npy
│   │   ├── train_commit_embeddings.npy
│   │   └── ...
│   ├── structural_features.pkl      # Features estruturais
│   └── phylogenetic_graph.pkl       # Grafo cached
│
├── results/                         # Resultados de experimentos
│   ├── experiment_001/
│   │   ├── config_used.yaml
│   │   ├── output.log
│   │   ├── apfd_per_build_FULL_testcsv.csv
│   │   ├── prioritized_test_cases.csv
│   │   ├── confusion_matrix.png
│   │   ├── best_model.pt
│   │   └── ...
│   ├── experiment_002/
│   └── ...
│
├── logs/                            # Logs gerais
│   └── experiment_vX.log
│
├── archive/                         # Arquivos obsoletos
│   ├── old_mains/
│   ├── configs/
│   ├── scripts/
│   └── docs/
│
├── models/                          # Modelos salvos
│   └── best_model_v8.pt
│
├── figures/                         # Figuras e visualizações
│
├── docs/                            # Documentação adicional
│
└── README.md                        # Documentação principal
└── PROJECT_RULES.md                 # Regras do projeto
└── MIGRATION_V8_TO_V9.md            # Guia de migração
└── RELATORIO_COMPLETO_PROJETO.md    # Este relatório
```

### 8.2 Convenções de Nomenclatura

#### 8.2.1 Arquivos Python

- **Módulos:** `snake_case.py` (ex: `qodo_encoder.py`)
- **Classes:** `PascalCase` (ex: `QodoEncoder`)
- **Funções:** `snake_case()` (ex: `extract_features()`)
- **Constantes:** `UPPER_SNAKE_CASE` (ex: `MAX_LENGTH`)

#### 8.2.2 Experimentos

- **Formato:** `experiment_XXX/` onde XXX é número de 3 dígitos
- **Auto-increment:** Próximo número disponível
- **Exemplos:** `experiment_001/`, `experiment_042/`

#### 8.2.3 Arquivos de Configuração

- **Ativo:** `configs/experiment.yaml` (ÚNICO)
- **Arquivados:** Dentro de `experiment_XXX/config_used.yaml`
- **Legacy:** `archive/configs/`

#### 8.2.4 Cache

- **Embeddings:** `cache/embeddings_{model_name}/`
- **Features:** `cache/structural_features.pkl`
- **Grafos:** `cache/phylogenetic_graph.pkl`

### 8.3 Gitignore

Arquivos não versionados:

```gitignore
# Cache
cache/
*.pkl
*.npy

# Logs
logs/
*.log

# Results
results/
runs/

# Models (exceto best models documentados)
*.pt
*.pth

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Virtual env
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## 9. Workflow de Experimentação

### 9.1 Setup Inicial (Uma vez)

```bash
# 1. Clonar repositório
git clone <repo_url>
cd filo_priori_v8

# 2. Executar setup
./setup_experiment.sh

# O script faz:
# - Cria virtual environment
# - Instala dependências (requirements.txt)
# - Verifica CUDA/GPU
# - Cria diretórios necessários (cache, results, logs)
```

### 9.2 Executar Experimento

#### 9.2.1 Workflow Padrão

```bash
# 1. Editar configuração
vim configs/experiment.yaml

# 2. Executar experimento (número automático)
./run_experiment.sh

# 3. Monitorar execução
tail -f results/experiment_XXX/output.log

# 4. Revisar resultados
cat results/experiment_XXX/output.log | grep "Final Test"
```

#### 9.2.2 Opções de Execução

```bash
# Usar config customizada
./run_experiment.sh --config configs/custom.yaml

# Forçar CPU (sem GPU)
./run_experiment.sh --device cpu

# Testar em amostra pequena
./run_experiment.sh --sample 1000

# Combinação
./run_experiment.sh --device cuda --sample 500
```

#### 9.2.3 Script `run_experiment.sh` (Internamente)

```bash
#!/bin/bash

# 1. Detectar próximo número de experimento
NEXT_NUM=$(ls results/ | grep "experiment_" | wc -l)
NEXT_NUM=$((NEXT_NUM + 1))
EXP_DIR="results/experiment_$(printf "%03d" $NEXT_NUM)"

# 2. Criar diretório
mkdir -p $EXP_DIR

# 3. Salvar config e comando
cp configs/experiment.yaml $EXP_DIR/config_used.yaml
echo "$@" > $EXP_DIR/command.txt

# 4. Executar main.py
python main.py \
    --config configs/experiment.yaml \
    --output_dir $EXP_DIR \
    $@ \
    2>&1 | tee $EXP_DIR/output.log

# 5. Salvar timestamp
date > $EXP_DIR/finished_at.txt
```

### 9.3 Análise de Resultados

#### 9.3.1 Arquivos Gerados

Cada experimento gera:

```
experiment_XXX/
├── config_used.yaml              # Config snapshot
├── command.txt                   # Comando executado
├── output.log                    # Log completo
├── timestamps.txt                # Start/end times
│
├── apfd_per_build_FULL_testcsv.csv  # APFD por build
├── prioritized_test_cases.csv       # Tests ranqueados
│
├── confusion_matrix.png          # Confusion matrix
├── pr_curve.png                  # Precision-Recall curve
│
├── best_model.pt                 # Melhor modelo (val F1)
├── last_model.pt                 # Último modelo
│
├── train_metrics.json            # Métricas de treino
├── val_metrics.json              # Métricas de validação
└── test_metrics.json             # Métricas de teste
```

#### 9.3.2 Extrair Métricas Principais

```bash
# Mean APFD
grep "Mean APFD" results/experiment_XXX/output.log

# Test Metrics
grep "Final Test" results/experiment_XXX/output.log

# Ou via JSON
cat results/experiment_XXX/test_metrics.json | jq '.f1_macro, .accuracy, .mean_apfd'
```

#### 9.3.3 Comparar Experimentos

```bash
# Script para comparar múltiplos experimentos
for exp in results/experiment_*/; do
    echo "=== $exp ==="
    grep "Mean APFD" $exp/output.log
    grep "Final Test F1" $exp/output.log
done
```

### 9.4 Iteração e Refinamento

#### 9.4.1 Ciclo de Experimentação

```
1. HIPÓTESE
   ↓
2. MODIFICAR CONFIG (configs/experiment.yaml)
   ↓
3. EXECUTAR EXPERIMENTO (./run_experiment.sh)
   ↓
4. ANALISAR RESULTADOS (results/experiment_XXX/)
   ↓
5. DOCUMENTAR APRENDIZADOS
   ↓
6. VOLTAR AO PASSO 1
```

#### 9.4.2 Tipos de Mudanças Comuns

**A) Hiperparâmetros:**
```yaml
# Aumentar learning rate
training:
  learning_rate: 1e-4  # Era 7.5e-5

# Aumentar dropout
model:
  classifier:
    dropout: 0.35  # Era 0.25
```

**B) Arquitetura:**
```yaml
# Adicionar camada no classifier
model:
  classifier:
    hidden_dims: [256, 128, 64]  # Era [128, 64]
```

**C) Features:**
```yaml
# Usar mais commits
text:
  num_commits_to_keep: 10  # Era 5
```

**D) Grafo:**
```yaml
# Relaxar filtro de arestas
graph:
  weight_threshold: 0.05  # Era 0.1
```

#### 9.4.3 Boas Práticas

1. **Mudar 1 coisa por vez:**
   - Facilita atribuição de causa-efeito
   - Experimentos mais interpretáveis

2. **Documentar hipótese:**
   ```yaml
   experiment:
     description: "Test higher dropout (0.35) to reduce overfitting"
   ```

3. **Manter histórico:**
   - Não deletar experimentos
   - Revisar resultados anteriores

4. **Validar em amostra primeiro:**
   ```bash
   ./run_experiment.sh --sample 1000  # Teste rápido
   # Se promissor:
   ./run_experiment.sh  # Full dataset
   ```

### 9.5 Debugging e Troubleshooting

#### 9.5.1 Problemas Comuns

**1. CUDA Out of Memory:**

```bash
# Solução 1: Reduzir batch size
vim configs/experiment.yaml
# training.batch_size: 16 (era 32)

# Solução 2: Usar CPU
./run_experiment.sh --device cpu
```

**2. Colapso de Predição:**

```
Output: Model predicts 100% Pass

Causas:
- Class imbalance muito alto
- Learning rate muito alto
- Loss function inadequada

Soluções:
1. Ativar class weights:
   loss:
     weighted_ce:
       use_class_weights: true
       class_weights: [1.0, 6.0]  # Favor Fail

2. Reduzir learning rate:
   training:
     learning_rate: 3e-5

3. Usar Focal Loss:
   loss:
     type: "focal"
     focal:
       alpha: 0.25
       gamma: 2.0
```

**3. NaN Loss:**

```
Output: Loss = nan

Causas:
- Learning rate muito alto
- Gradientes explodindo
- Dados com NaN/Inf

Soluções:
1. Reduzir learning rate
2. Verificar gradient clipping:
   training:
     grad_clip_norm: 0.5

3. Verificar dados:
   python -c "import pandas as pd; print(pd.read_csv('datasets/train.csv').isna().sum())"
```

**4. Overfitting:**

```
Output: Train F1=0.95, Val F1=0.40

Soluções:
1. Aumentar dropout
2. Aumentar weight decay
3. Reduzir model capacity (hidden_dims)
4. Early stopping mais agressivo (menor patience)
5. Data augmentation (se aplicável)
```

---

## 10. Evolução do Projeto (V8 → V9)

### 10.1 Timeline de Versões

```
V1-V5: Early prototypes
  ↓
V6: Initial dual-stream
  ↓
V7: BGE embeddings + k-NN graph
  ↓
V8: Phylogenetic graph + GAT
  ├── V8.0: Initial (APFD: 0.5481)
  ├── V8.1: Improved (APFD: 0.5967)
  └── V8.2: Fine-tuned BGE (APFD: 0.5481) ← Degradação!
  ↓
V9: Qodo-Embed + Separate encoding ← CURRENT
```

### 10.2 Principais Mudanças V8 → V9

#### 10.2.1 Embedding Model

| Aspecto | V8 | V9 |
|---------|----|----|
| **Modelo** | BGE-large-en-v1.5 | Qodo-Embed-1-1.5B |
| **Parâmetros** | 335M | 1.5B (4.5x maior) |
| **Embedding Dim** | 1024 | 1536 (por encoding) |
| **Combined Dim** | 1024 | 3072 (3x maior) |
| **Fine-tuning** | Sim (triplet loss) | Não (pré-treinado) |
| **Encoding** | Concatenado (TC+Commit) | Separado (TC / Commit) |
| **Domínio** | Geral | Código + Docs técnicas |

#### 10.2.2 Arquitetura do Modelo

| Componente | V8 | V9 |
|------------|----|----|
| **Semantic Stream Input** | 1024 | 3072 |
| **Semantic Hidden** | 256 | 256 |
| **Structural Input** | 6 | 6 |
| **Structural Type** | GAT (4 heads) | GAT (4 heads) |
| **Fusion** | Cross-attention | Cross-attention |
| **Classifier** | [128, 64] | [128, 64] |

#### 10.2.3 Hiperparâmetros

| Parâmetro | V8 | V9 | Mudança |
|-----------|----|----|---------|
| **Learning Rate** | 5e-5 | 7.5e-5 | +50% |
| **Weight Decay** | 1e-4 | 3e-5 | -70% |
| **Dropout (Semantic)** | 0.3 | 0.15 | -50% |
| **Dropout (Classifier)** | 0.3 | 0.25 | -17% |
| **Batch Size** | 32 | 32 | - |
| **Epochs** | 50 | 50 | - |
| **Early Stopping Patience** | 15 | 20 | +33% |

#### 10.2.4 Loss Function

| Aspecto | V8 | V9 |
|---------|----|----|
| **Type** | Weighted CE | Weighted CE |
| **Class Weights** | [60.0, 1.0] | [1.0, 1.0] (balanced) |
| **Label Smoothing** | 0.0 | 0.0 |

### 10.3 Motivação para V9

#### 10.3.1 Problema com Fine-tuning (V8)

**Experimento V8 Fine-tuned:**
```
Strategy: Fine-tune BGE with triplet loss
- Positive: Same TC, different builds
- Negative: Different TC

Results:
- Train loss: Decreased ✓
- Embeddings: More separated ✓
- APFD: 0.5967 → 0.5481 ✗ (DEGRADAÇÃO!)

Causa:
- Overfitting nos dados de treino
- Perda de capacidade de generalização
- Embeddings too task-specific
```

#### 10.3.2 Hipótese V9

**Princípios:**

1. **Bigger is Better:**
   - Modelo maior (1.5B) → Mais capacidade representacional
   - Menos necessidade de fine-tuning

2. **Separate Encoding:**
   - TCs e Commits têm estruturas semânticas diferentes
   - Encoding separado preserva informação

3. **Pre-trained Knowledge:**
   - Qodo treinado em código/docs → Domínio relevante
   - Não fine-tuning → Evita overfitting

4. **Higher Capacity:**
   - 3072 dims vs 1024 → 3x informação
   - Mais contexto para modelo downstream

### 10.4 Resultados Comparativos

#### 10.4.1 Métricas Esperadas

| Métrica | V8 (Best) | V9 (Target) | V9 (Stretch) |
|---------|-----------|-------------|--------------|
| **Mean APFD** | 0.5967 | ≥ 0.58 | ≥ 0.65 |
| **Test F1 Macro** | 0.5360 | ≥ 0.54 | ≥ 0.60 |
| **Test Accuracy** | 0.6500 | ≥ 0.60 | ≥ 0.70 |
| **Recall (Fail)** | 0.18 | ≥ 0.20 | ≥ 0.40 |
| **Precision (Fail)** | 0.22 | ≥ 0.25 | ≥ 0.35 |

#### 10.4.2 Análise de Trade-offs

**Vantagens V9:**
- ✅ Modelo maior e mais poderoso
- ✅ Encoding separado (mais informação)
- ✅ Sem risco de overfitting no fine-tuning
- ✅ Domínio específico (código/docs)

**Desvantagens V9:**
- ⚠️ Maior uso de memória (3072 dims)
- ⚠️ Processamento mais lento (1.5B params)
- ⚠️ Embedding fixo (não adaptado ao task)

**Trade-off Accepted:**
- Preferimos generalização sobre otimização task-specific
- Memória/tempo aceitáveis para ganhos de performance

---

## 11. Resultados Esperados

### 11.1 Objetivos Primários

#### 11.1.1 APFD (Objetivo Principal)

```
Target: Mean APFD ≥ 0.58

Interpretação:
- 0.58: 58% das falhas detectadas na primeira metade dos testes
- Melhoria de ~16% sobre random (0.50)
- Comparável ou melhor que V8 (0.5481-0.5967)

Impacto Prático:
- Em 1000 testes, ~580 falhas detectadas nas primeiras 500 execuções
- Feedback 2x mais rápido para desenvolvedores
```

#### 11.1.2 F1 Macro (Objetivo Secundário)

```
Target: Test F1 Macro ≥ 0.54

Interpretação:
- Balanço entre Precision e Recall para ambas classes
- Melhor que V8 Fine-tuned (0.5360)

Classes:
- Pass: F1 ~ 0.90 (fácil, classe majoritária)
- Fail: F1 ~ 0.35-0.40 (difícil, classe minoritária)
- Macro: (0.90 + 0.35) / 2 = 0.625 (target realistic: 0.54)
```

#### 11.1.3 Recall (Fail) - Crítico

```
Target: Recall (Fail) ≥ 0.20

Interpretação:
- Detectar pelo menos 20% das falhas reais
- Trade-off: Preferir False Positives a False Negatives

Exemplo:
- 100 falhas reais
- Recall 0.20 → Detecta 20 falhas
- Recall 0.40 → Detecta 40 falhas (stretch goal)
```

### 11.2 Métricas Complementares

#### 11.2.1 Prediction Diversity

```
Target: ≥ 10% para ambas classes

Crítico: Evitar colapso de predição
- Se Diversity < 5%: Experimento rejeitado
- Se Diversity ≥ 30%: Excelente
```

#### 11.2.2 AUPRC (Area Under PR Curve)

```
Target: ≥ 0.30

Interpretação:
- Curva Precision-Recall integrada
- Melhor métrica para classe minoritária (Fail)
- 0.30: Bom para dados tão desbalanceados
```

#### 11.2.3 Generalization Gap

```
Target: Val-Test Gap < 0.10

Exemplo:
- Val F1: 0.55
- Test F1: 0.54
- Gap: 0.01 ✓ (excelente generalização)

Warning se:
- Gap > 0.15: Possível overfitting
```

### 11.3 Análise de Casos de Uso

#### 11.3.1 Cenário Real

**Setup:**
- Build com 1000 testes
- 100 falhas (10% failure rate)
- Tempo: 5min/teste → 5000min total (~83h)
- Budget: 500 testes (50% do total)

**Estratégia Baseline (Random):**
```
Execução: Ordem aleatória
Falhas detectadas em 500 testes: ~50 (50% das falhas)
APFD: ~0.50
Tempo de feedback: 41.5h
```

**Estratégia Filo-Priori V9 (APFD=0.58):**
```
Execução: Ordenado por risk_score
Falhas detectadas em 500 testes: ~65 (65% das falhas)
APFD: 0.58
Tempo de feedback: 41.5h

Ganho:
- +15 falhas detectadas (30% a mais)
- Mesmo tempo de execução
- Feedback mais rico para devs
```

#### 11.3.2 ROI (Return on Investment)

**Custo:**
- Treino inicial: ~3h (one-time)
- Inferência por build: ~2min (1000 testes)

**Benefício:**
- 30% mais falhas detectadas no mesmo tempo
- Detecção mais cedo → Correção mais barata
- Menos context switching para devs

**Break-even:**
- Após ~10 builds (custo de treino amortizado)

---

## 12. Guia de Uso

### 12.1 Quick Start (5 minutos)

```bash
# 1. Clone e setup
git clone <repo>
cd filo_priori_v8
./setup_experiment.sh

# 2. Teste rápido (amostra)
./run_experiment.sh --sample 1000

# 3. Ver resultados
tail -50 results/experiment_001/output.log
```

### 12.2 Uso Completo

#### 12.2.1 Preparar Dados

```bash
# Verificar formato dos dados
head -5 datasets/train.csv
head -5 datasets/test.csv

# Colunas obrigatórias:
# - TC_Key
# - Build_ID
# - TE_Summary
# - TC_Steps
# - TE_Test_Result
# - commit
# - CR
```

#### 12.2.2 Configurar Experimento

```bash
# Editar config
vim configs/experiment.yaml

# Principais parâmetros:
# - training.learning_rate
# - training.num_epochs
# - model.*.dropout
# - loss.weighted_ce.class_weights
```

#### 12.2.3 Executar Treinamento

```bash
# Full training
./run_experiment.sh

# Com opções
./run_experiment.sh --device cuda --config configs/experiment.yaml

# Background (tmux/screen)
tmux new -s experiment
./run_experiment.sh
# Ctrl+B, D (detach)
```

#### 12.2.4 Monitorar Execução

```bash
# Seguir log em tempo real
tail -f results/experiment_XXX/output.log

# Ver métricas por época
grep "Epoch" results/experiment_XXX/output.log

# Ver early stopping
grep "Early stopping" results/experiment_XXX/output.log
```

#### 12.2.5 Analisar Resultados

```bash
# Métricas finais
grep "Final Test" results/experiment_XXX/output.log

# APFD
grep "Mean APFD" results/experiment_XXX/output.log

# Confusion matrix
open results/experiment_XXX/confusion_matrix.png

# Top falhas preditas
head -20 results/experiment_XXX/prioritized_test_cases.csv
```

### 12.3 Customizações Comuns

#### 12.3.1 Ajustar para Dataset Pequeno

```yaml
# configs/experiment.yaml

training:
  num_epochs: 30  # Reduzir epochs
  batch_size: 16  # Menor batch
  early_stopping:
    patience: 10  # Menos patience

model:
  classifier:
    hidden_dims: [64, 32]  # Menor capacity
    dropout: 0.4  # Mais dropout
```

#### 12.3.2 Ajustar para Dataset Grande

```yaml
training:
  num_epochs: 100  # Mais epochs
  batch_size: 64   # Maior batch
  early_stopping:
    patience: 30

model:
  classifier:
    hidden_dims: [256, 128, 64]  # Mais capacity
    dropout: 0.2  # Menos dropout
```

#### 12.3.3 Lidar com Extreme Imbalance

```yaml
loss:
  type: "focal"  # Focal Loss
  focal:
    alpha: 0.25
    gamma: 2.0

# Ou
loss:
  type: "weighted_ce"
  weighted_ce:
    use_class_weights: true
    class_weights: [1.0, 10.0]  # Forte favor para Fail
```

### 12.4 Troubleshooting

#### 12.4.1 GPU Memory Issues

```bash
# Opção 1: Reduzir batch size
vim configs/experiment.yaml
# training.batch_size: 16

# Opção 2: CPU mode
./run_experiment.sh --device cpu

# Opção 3: Limpar cache
rm -rf cache/*
```

#### 12.4.2 Slow Training

```bash
# Verificar cache de embeddings
ls -lh cache/embeddings_qodo/

# Se falta cache, primeira época será lenta (encoding)
# Épocas seguintes: Rápido (cache hit)

# Forçar recálculo de cache (se corrompido)
rm -rf cache/embeddings_qodo/
```

#### 12.4.3 Poor Performance

```bash
# Verificar prediction diversity
grep "Prediction Diversity" results/experiment_XXX/output.log

# Se < 5%: Colapso de predição
# Soluções:
# 1. Ajustar class weights
# 2. Reduzir learning rate
# 3. Usar Focal Loss
```

### 12.5 Exportar e Compartilhar

#### 12.5.1 Pacote de Resultados

```bash
# Criar pacote
tar -czf experiment_XXX_results.tar.gz \
    results/experiment_XXX/ \
    --exclude="*.pt"  # Excluir modelos (grandes)

# Compartilhar
scp experiment_XXX_results.tar.gz user@server:/path/
```

#### 12.5.2 Modelo Treinado

```bash
# Salvar apenas best model
cp results/experiment_XXX/best_model.pt models/filo_priori_v9_best.pt

# Carregar em outro script
import torch
model = torch.load('models/filo_priori_v9_best.pt')
```

#### 12.5.3 Relatório Resumido

```bash
# Gerar relatório automático
python scripts/generate_report.py \
    --experiment results/experiment_XXX \
    --output reports/experiment_XXX_report.md
```

---

## 13. Conclusão

### 13.1 Resumo do Sistema

O **Filo-Priori V9** é um sistema completo de **Test Case Prioritization** que combina:

1. **Embeddings semânticos poderosos** (Qodo-Embed-1-1.5B)
2. **Grafos filogenéticos estruturais** (co-failure relationships)
3. **Graph Neural Networks** (GAT com multi-head attention)
4. **Fusão por cross-attention** (integração semântica-estrutural)

**Resultado:** Priorização inteligente que detecta falhas mais cedo, otimizando tempo de CI/CD.

### 13.2 Pontos Fortes

✅ **Arquitetura Dual-Stream:** Combina texto e grafo
✅ **Modelo State-of-Art:** Qodo-Embed 1.5B
✅ **Encoding Separado:** Preserva estrutura TC/Commit
✅ **Grafos Verdadeiros:** Co-failure patterns reais
✅ **Atenção Multi-Head:** GAT + Cross-Attention
✅ **Pipeline Completo:** End-to-end automatizado
✅ **Reprodutível:** Seeds, configs, caching

### 13.3 Limitações e Trabalhos Futuros

#### Limitações Atuais:

1. **Dependência de Histórico:** Requer dados históricos suficientes
2. **Cold Start:** Novos testes sem histórico → Features imputadas
3. **Computational Cost:** 1.5B params → GPU necessária
4. **Dados Desbalanceados:** ~10-15% falhas → Difícil aprender

#### Trabalhos Futuros:

1. **Meta-Learning:** Adaptar a novos projetos rapidamente
2. **Active Learning:** Selecionar testes mais informativos para rotular
3. **Multi-Task Learning:** Predizer também tempo de execução, flakiness
4. **Explainability:** Por que um teste foi priorizado? (SHAP, Attention viz)
5. **Online Learning:** Atualizar modelo incrementalmente com novos builds
6. **Graph Contrastive Learning:** Melhorar representações de grafo

### 13.4 Referências e Links

**Papers Relacionados:**
- Graph Attention Networks (GAT): Veličković et al., 2018
- Focal Loss: Lin et al., 2017
- Test Case Prioritization: Rothermel et al., 2001

**Modelos:**
- Qodo-Embed-1-1.5B: https://huggingface.co/Qodo/Qodo-Embed-1-1.5B
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

**Repositório:**
- GitHub: <repo_url>
- Docs: `README.md`, `PROJECT_RULES.md`

---

**Última Atualização:** 2025-11-12
**Versão do Relatório:** 1.0
**Autor:** Equipe Filo-Priori

---
