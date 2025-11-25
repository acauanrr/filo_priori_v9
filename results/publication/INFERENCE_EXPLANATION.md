# EXPLICAÇÃO DIDÁTICA: COMO FUNCIONA A INFERÊNCIA NO PROJETO

## RESUMO EXECUTIVO

Este documento explica de forma didática como o modelo faz **inferência (predições)** em novos dados de teste (test.csv), especialmente para test cases que **nunca foram vistos durante o treinamento**.

---

## 1. POR QUE "NOVO! (global_idx = -1)"?

### Conceito de Global Index

Durante o **treinamento**, o modelo constrói um **mapeamento** de todos os test cases únicos que aparecem no train.csv:

```
MAPEAMENTO CRIADO NO TREINAMENTO (tc_key_to_global_idx):
┌─────────────────┬──────────────┐
│ TC_Key          │ global_idx   │
├─────────────────┼──────────────┤
│ MCA-1015        │      0       │
│ MCA-101956      │      1       │
│ MCA-102345      │      2       │
│ ...             │     ...      │
│ MCA-999888      │     160      │  ← 161 test cases únicos no train
└─────────────────┴──────────────┘
```

**Como é criado?** (main.py linhas 470-472):
```python
tc_keys_train = df_train['TC_Key'].unique().tolist()
tc_key_to_global_idx = {tc_key: i for i, tc_key in enumerate(tc_keys_train)}
# Resultado: {'MCA-1015': 0, 'MCA-101956': 1, ...}
```

### O que acontece com Test Cases Novos?

Quando o modelo processa o **test.csv**, ele tenta mapear cada test case:

```python
# Para cada test case no test.csv (linhas 475-476):
test_data['global_indices'] = np.array([
    tc_key_to_global_idx.get(tc_key, -1)  # .get(key, default=-1)
    for tc_key in df_test['TC_Key']
])
```

**Resultado**:
```
TEST.CSV:
┌─────────────────┬──────────────┬─────────────────────┐
│ TC_Key          │ global_idx   │ Status              │
├─────────────────┼──────────────┼─────────────────────┤
│ MCA-1015        │      0       │ ✓ Conhecido (treino)│
│ MCA-NEW-123     │     -1       │ ✗ NOVO!             │
│ MCA-101956      │      1       │ ✓ Conhecido         │
│ MCA-NEW-456     │     -1       │ ✗ NOVO!             │
└─────────────────┴──────────────┴─────────────────────┘
```

**Por que -1?**
- O método `.get(tc_key, -1)` do Python retorna `-1` se a chave não existir no dicionário
- `-1` é um **valor sentinela** (flag) que indica: "este test case não estava no treinamento"
- Permite identificar facilmente test cases novos: `if global_idx == -1: # novo!`

---

## 2. OS NÓS NOVOS SÃO ADICIONADOS AO GRAFO GLOBAL?

### RESPOSTA CURTA: **NÃO!** ❌

O grafo permanece **estático** (fixo) após o treinamento. Nós novos **NÃO são adicionados** ao grafo.

### Por que essa decisão?

```
╔═══════════════════════════════════════════════════════════════╗
║  DESIGN DECISION: GRAFO ESTÁTICO                              ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Problema: Como processar test cases novos em um GNN?        ║
║                                                               ║
║  Opção A (NÃO usada): Adicionar nós novos ao grafo           ║
║    ✗ Requer reconstruir o grafo (lento)                      ║
║    ✗ Novos nós não têm arestas (sem histórico)               ║
║    ✗ GAT não consegue agregar informação útil                ║
║                                                               ║
║  Opção B (USADA): Grafo estático + Orphan Handling           ║
║    ✓ Grafo construído uma vez (rápido)                       ║
║    ✓ Nós novos detectados via global_idx = -1                ║
║    ✓ Nós novos usam apenas Semantic Stream                   ║
║    ✓ Predição default conservadora [0.5, 0.5]                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Visualização do Processo

```
TREINAMENTO (uma vez):
┌──────────────────────────────────────────────────────────────┐
│ CONSTRUÇÃO DO GRAFO (train.csv)                             │
│                                                              │
│  Nós: 161 test cases únicos do train.csv                    │
│  Arestas: Relações de co-falha e co-commit                  │
│                                                              │
│     MCA-1015 ←--co-falha-→ MCA-101956                       │
│         │                        │                           │
│    co-commit               co-commit                         │
│         │                        │                           │
│         ↓                        ↓                           │
│    MCA-102345 ←--co-falha-→ MCA-999888                      │
│                                                              │
│  SALVO: edge_index (grafo fixo)                             │
└──────────────────────────────────────────────────────────────┘
              │
              │ Grafo NÃO muda durante inferência
              ▼
INFERÊNCIA (test.csv):
┌──────────────────────────────────────────────────────────────┐
│ PROCESSAMENTO DE BATCH                                      │
│                                                              │
│  Batch contém:                                              │
│    - MCA-1015     (global_idx = 0)   → ✓ Está no grafo     │
│    - MCA-NEW-123  (global_idx = -1)  → ✗ NÃO está no grafo │
│    - MCA-101956   (global_idx = 1)   → ✓ Está no grafo     │
│    - MCA-NEW-456  (global_idx = -1)  → ✗ NÃO está no grafo │
│                                                              │
│  FILTRO (main.py linha 624):                                │
│    valid_mask = (global_indices != -1)                      │
│    → [True, False, True, False]                             │
│                                                              │
│  SUBGRAFO EXTRAÍDO:                                         │
│    Apenas MCA-1015 e MCA-101956 processados pelo GAT        │
│                                                              │
│    MCA-1015 ←--co-falha-→ MCA-101956                        │
│        │                        │                            │
│   GAT processa             GAT processa                      │
│                                                              │
│  ORPHANS (global_idx = -1):                                 │
│    MCA-NEW-123 e MCA-NEW-456 → Predição [0.5, 0.5]         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Código Relevante

**Construção do grafo (TREINO, main.py linhas 460-466):**
```python
# Grafo construído APENAS com train.csv
all_tc_keys = df_train['TC_Key'].unique().tolist()
edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
    tc_keys=all_tc_keys,  # APENAS test cases do train
    return_torch=True
)
logger.info(f"Graph: {edge_index.shape[1]} edges among {len(all_tc_keys)} nodes")
```

**Extração de subgrafo (INFERÊNCIA, main.py linhas 637-643):**
```python
# Durante inferência: extrai subgrafo do MESMO grafo de treino
sub_edge_index, sub_edge_weights = subgraph(
    subset=global_indices_valid,  # Apenas nós válidos deste batch
    edge_index=edge_index,         # GRAFO FIXO do treino
    edge_attr=edge_weights,
    relabel_nodes=True,
    num_nodes=num_nodes_global     # 161 nós do treino
)
```

**Tratamento de orphans (main.py linhas 791-794):**
```python
# Nós com global_idx = -1 recebem predição default
if return_full_probs and dataset_size is not None:
    full_probs = np.full((dataset_size, 2), 0.5)  # [0.5, 0.5] para orphans
    full_probs[all_batch_indices] = all_probs     # Substitui válidos
```

---

## 3. COMO SÃO EXTRAÍDAS AS FEATURES ESTRUTURAIS DO TEST.CSV?

### Overview

As **features estruturais** são estatísticas históricas sobre cada test case:
1. **test_age**: Quantos builds desde a primeira aparição
2. **failure_rate**: Taxa de falhas histórica
3. **recent_failure_rate**: Taxa de falhas recente
4. **flakiness_rate**: Taxa de flakiness
5. **commit_count**: Número de commits no build atual
6. **test_novelty**: 1.0 se novo, 0.0 se conhecido

### Processo de Extração

```
┌─────────────────────────────────────────────────────────────────┐
│  EXTRAÇÃO DE FEATURES ESTRUTURAIS (test.csv)                   │
└─────────────────────────────────────────────────────────────────┘

Para cada test case no test.csv:

┌─────────────────────────────────────────────────────────────────┐
│  PASSO 1: Verificar se test case está no histórico de treino   │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─ SIM (TC_Key in tc_history)
         │         │
         │         ▼
         │  ┌───────────────────────────────────────────────┐
         │  │ FEATURES REAIS (do histórico)                 │
         │  │                                               │
         │  │ Exemplo: MCA-1015                            │
         │  │ - test_age = 45 builds                       │
         │  │ - failure_rate = 0.23 (23% de falhas)        │
         │  │ - recent_failure_rate = 0.15                 │
         │  │ - flakiness_rate = 0.08                      │
         │  │ - commit_count = 3 (commits no build atual)  │
         │  │ - test_novelty = 0.0 (não é novo)            │
         │  └───────────────────────────────────────────────┘
         │
         └─ NÃO (TC_Key NOT in tc_history)
                   │
                   ▼
         ┌──────────────────────────────────────────────────────┐
         │ FEATURES DEFAULT (test case novo)                   │
         │                                                      │
         │ Exemplo: MCA-NEW-123                                │
         │ - test_age = 0.0 (acabou de aparecer)               │
         │ - failure_rate = population_mean (ex: 0.31)         │
         │ - recent_failure_rate = population_mean (ex: 0.28)  │
         │ - flakiness_rate = population_median (ex: 0.12)     │
         │ - commit_count = 2 (extraído do build atual)        │
         │ - test_novelty = 1.0 (NOVO!)                        │
         │                                                      │
         │ ⚠️ IMPORTANTE: NÃO usa zeros!                       │
         │    Zeros implica "nunca falha" → bias perigoso      │
         │    Usa estatísticas da população = conservador      │
         └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PASSO 2: Imputação (se necessário)                            │
└─────────────────────────────────────────────────────────────────┘

Se test case tem histórico INSUFICIENTE (ex: apareceu 1-2 vezes):
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ IMPUTAÇÃO BASEADA EM VIZINHOS SEMÂNTICOS                      │
│                                                                │
│ 1. Calcula similaridade semântica (usando embeddings)         │
│    - Compara embedding do test novo com todos do train        │
│    - Cosine similarity                                         │
│                                                                │
│ 2. Encontra K vizinhos mais similares (K=10)                  │
│    - Threshold: similarity > 0.5                               │
│                                                                │
│ 3. Imputa features como média ponderada dos vizinhos          │
│    - Peso = similaridade semântica                             │
│                                                                │
│ Exemplo:                                                       │
│   MCA-NEW-123 (novo, mas similar a MCA-1015)                  │
│   → Empresta features de MCA-1015                             │
│   → failure_rate ≈ 0.23 (similar ao vizinho)                  │
└────────────────────────────────────────────────────────────────┘
```

### Código Real (structural_feature_extractor.py)

**Para test cases conhecidos (linhas 327-341):**
```python
if tc_key in self.tc_history:
    history = self.tc_history[tc_key]

    # Estatísticas reais do histórico
    test_age = current_build_idx - history['first_build_idx']
    failure_rate = history['failure_rate']
    recent_failure_rate = history['recent_failure_rate']
    flakiness_rate = history['flakiness_rate']
```

**Para test cases NOVOS (linhas 343-357):**
```python
else:
    # Test case NOVO (não visto no treino)
    test_age = 0.0  # Recém-criado

    if self.feature_means is not None:
        # Usa estatísticas da POPULAÇÃO (conservador)
        failure_rate = float(self.feature_means[1])
        recent_failure_rate = float(self.feature_means[2])
        flakiness_rate = float(self.feature_medians[3])
    else:
        # Fallback: zeros (menos ideal)
        failure_rate = 0.0
        recent_failure_rate = 0.0
        flakiness_rate = 0.0
```

**Imputação (main.py linhas 293-328):**
```python
# Detecta quais test cases precisam de imputação
needs_imputation_test = extractor.get_imputation_mask(tc_keys_test)

if needs_imputation_test.sum() > 0:
    # Imputa usando vizinhos semânticos
    test_struct, test_imputation_stats = impute_structural_features(
        train_embeddings, train_struct, tc_keys_train,  # Dados de treino
        test_embeddings, test_struct, tc_keys_test,      # Dados de teste
        extractor.tc_history,
        k_neighbors=10,              # 10 vizinhos mais próximos
        similarity_threshold=0.5,     # Similaridade mínima
        verbose=False
    )

    logger.info(f"Imputação - Test: {needs_imputation_test.sum()} samples")
```

### Exemplo Completo

```
TEST CASE: MCA-NEW-123 (NOVO, não visto no treino)

STEP 1: Extração Inicial
├─ test_age = 0.0                    (novo)
├─ failure_rate = 0.31               (população média)
├─ recent_failure_rate = 0.28        (população média)
├─ flakiness_rate = 0.12             (população mediana)
├─ commit_count = 2                  (do build atual)
└─ test_novelty = 1.0                (NOVO!)

STEP 2: Verificação de Imputação
└─ needs_imputation = True (histórico insuficiente)

STEP 3: Imputação por Vizinhos Semânticos
├─ Embedding de MCA-NEW-123: [0.12, -0.45, 0.78, ...]
│
├─ 10 vizinhos mais similares (por embedding):
│   1. MCA-1015      (similarity = 0.89) → failure_rate = 0.23
│   2. MCA-102345    (similarity = 0.85) → failure_rate = 0.45
│   3. MCA-201567    (similarity = 0.78) → failure_rate = 0.12
│   ... (7 mais)
│
└─ failure_rate final = weighted_average([0.23, 0.45, 0.12, ...],
                                         weights=[0.89, 0.85, 0.78, ...])
                      ≈ 0.28 (média ponderada pelos vizinhos)

RESULTADO FINAL: [0.0, 0.28, 0.25, 0.11, 2.0, 1.0]
                  ↑    ↑     ↑     ↑    ↑    ↑
                  age  fail  rcnt  flak cmt  nov
```

---

## 4. COMO FUNCIONA A PARTE SEMÂNTICA (EMBEDDINGS)?

### Conceito

A **Semantic Stream** usa **embeddings de texto** gerados por um modelo de linguagem (SBERT) para capturar o **significado semântico** de cada test case, independentemente de ter histórico ou não.

### Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────┐
│           GERAÇÃO DE EMBEDDINGS (test.csv)                     │
└─────────────────────────────────────────────────────────────────┘

PARA CADA TEST CASE NO TEST.CSV:

┌─────────────────────────────────────────────────────────────────┐
│  PASSO 1: Preparação do Texto                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  CAMPO 1: Test Case Description                               │
│  ─────────────────────────────────────────────────────────────│
│                                                                │
│  tc_summary = "Verify login functionality"                    │
│  tc_steps = "1. Open app\n2. Enter credentials\n3. Click OK" │
│                                                                │
│  Texto formatado:                                              │
│  "Summary: Verify login functionality                         │
│   Steps: 1. Open app                                          │
│          2. Enter credentials                                  │
│          3. Click OK"                                          │
│                                                                │
│  SBERT (all-mpnet-base-v2)                                    │
│         │                                                      │
│         ▼                                                      │
│  TC Embedding: [0.12, -0.45, 0.78, ..., 0.34]  (768 dims)    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  CAMPO 2: Commit Description                                   │
│  ─────────────────────────────────────────────────────────────│
│                                                                │
│  commit_msg = "Fix authentication bug in login module"        │
│  commit_diff = "- if (user == null)\n+ if (user != null)"    │
│                                                                │
│  Texto formatado:                                              │
│  "Message: Fix authentication bug in login module             │
│   Diff: - if (user == null)                                   │
│         + if (user != null)"                                   │
│                                                                │
│  SBERT (all-mpnet-base-v2)                                    │
│         │                                                      │
│         ▼                                                      │
│  Commit Embedding: [-0.23, 0.56, -0.12, ..., 0.89] (768 dims)│
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PASSO 2: Concatenação                                         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  EMBEDDING FINAL (CONCATENADO)                                 │
│                                                                │
│  [TC_Embedding (768) | Commit_Embedding (768)]                │
│                                                                │
│  = [0.12, -0.45, ..., 0.34, -0.23, 0.56, ..., 0.89]          │
│                                                                │
│  Dimensão total: 1536                                          │
└────────────────────────────────────────────────────────────────┘
```

### Modelo SBERT (all-mpnet-base-v2)

```
╔═══════════════════════════════════════════════════════════════╗
║  SENTENCE-BERT (SBERT)                                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Modelo: all-mpnet-base-v2                                   ║
║  Base: Microsoft MPNet (Masked and Permuted Pre-training)   ║
║  Treinamento: 1B+ sentence pairs (paraphrase detection)     ║
║                                                               ║
║  Características:                                             ║
║  ✓ Estado da arte em similaridade semântica                  ║
║  ✓ 768 dimensões por embedding                               ║
║  ✓ Normalizado (norma euclidiana = 1)                        ║
║  ✓ Cosine similarity = dot product                           ║
║                                                               ║
║  Propriedades importantes:                                    ║
║  • Textos similares → embeddings próximos no espaço          ║
║  • "Verify login" ≈ "Test authentication"                    ║
║  • Independente de vocabulário exato                         ║
║  • Captura sinônimos e conceitos                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Código Real (embedding_manager.py)

**Preparação do texto TC (linhas 49-75):**
```python
def _prepare_tc_texts(self, df: pd.DataFrame) -> list:
    texts = []
    for _, row in df.iterrows():
        # Extrai campos
        summary = row.get('tc_summary', row.get('summary', ''))
        steps = row.get('tc_steps', row.get('steps', ''))

        # Formata texto
        if summary and steps:
            text = f"Summary: {summary}\nSteps: {steps}"
        elif summary:
            text = f"Summary: {summary}"
        else:
            text = "No test case information"

        texts.append(text)
    return texts
```

**Preparação do texto Commit (linhas 77-103):**
```python
def _prepare_commit_texts(self, df: pd.DataFrame) -> list:
    texts = []
    for _, row in df.iterrows():
        # Extrai campos
        commit_msg = row.get('commit_messages', '')
        commit_diff = row.get('diff', '')

        # Formata texto
        if commit_msg and commit_diff:
            text = f"Message: {commit_msg}\nDiff: {commit_diff}"
        elif commit_msg:
            text = f"Message: {commit_msg}"
        else:
            text = "No commit information"

        texts.append(text)
    return texts
```

**Geração de embeddings (main.py linhas 1258-1271):**
```python
# Gera embeddings para TODOS os test cases do test.csv
full_test_embeddings_dict = embedding_manager.get_embeddings(
    test_df_full,
    test_df_full
)

# Extrai embeddings individuais
test_tc_embeddings_full = full_test_embeddings_dict['train_tc']       # [N, 768]
test_commit_embeddings_full = full_test_embeddings_dict['train_commit']# [N, 768]

# Concatena TC + Commit
test_embeddings_full = np.concatenate([
    test_tc_embeddings_full,
    test_commit_embeddings_full
], axis=1)  # [N, 1536]
```

### Visualização do Espaço de Embeddings

```
ESPAÇO DE EMBEDDINGS (1536 dimensões, projetado em 2D):

        Test Novos (•)          Test Conhecidos (○)

    │
    │        •                  ○
    │     MCA-NEW-123      MCA-1015
  E │   (login test)    (login test)
  m │
  b │                      ○
  e │                 MCA-102345
  d │              (database test)
  d │
  i │
  n │     •
  g │  MCA-NEW-456
    │  (UI test)          ○
  2 │               MCA-201567
    │                (UI test)
    │
    ├──────────────────────────────────────────────────
                    Embedding 1

INTERPRETAÇÃO:
- Distância no espaço ≈ Similaridade semântica
- MCA-NEW-123 próximo de MCA-1015 → ambos testam login
- MCA-NEW-456 próximo de MCA-201567 → ambos testam UI
- Modelo APRENDE padrões: "login" → alto risco de falha
                          "UI" → médio risco
```

### Como o Modelo USA os Embeddings?

```
┌─────────────────────────────────────────────────────────────────┐
│  SEMANTIC STREAM (dual_stream_v8.py)                           │
└─────────────────────────────────────────────────────────────────┘

INPUT: Embedding concatenado [batch_size, 1536]

         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  CAMADA 1: Linear + Activation                                 │
│  ──────────────────────────────────────────────────────────────│
│                                                                │
│  Linear(1536 → 512)                                           │
│  BatchNorm1d(512)                                             │
│  ReLU                                                          │
│  Dropout(0.3)                                                  │
│                                                                │
│  [batch, 1536] → [batch, 512]                                 │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  CAMADA 2: Linear + Activation                                 │
│  ──────────────────────────────────────────────────────────────│
│                                                                │
│  Linear(512 → 256)                                            │
│  BatchNorm1d(256)                                             │
│  ReLU                                                          │
│  Dropout(0.3)                                                  │
│                                                                │
│  [batch, 512] → [batch, 256]                                  │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
OUTPUT: Semantic Features [batch_size, 256]

APRENDIZADO:
- Rede neural APRENDE quais padrões semânticos são indicativos de falha
- Exemplo: palavras como "authentication", "login", "database"
           podem estar correlacionadas com maior risco
- Funciona para test cases NOVOS porque embeddings capturam significado
```

---

## 5. OS EMBEDDINGS DO TEST.CSV SÃO COMPARADOS COM OS DO TREINO?

### RESPOSTA: **SIM e NÃO** (depende da etapa)

Há **duas formas** de uso dos embeddings:

### FORMA 1: Imputação de Features (SIM, comparação explícita)

```
┌─────────────────────────────────────────────────────────────────┐
│  IMPUTAÇÃO: COMPARAÇÃO EXPLÍCITA                               │
└─────────────────────────────────────────────────────────────────┘

Objetivo: Emprestar features estruturais de test cases similares

ALGORITMO:
1. Para cada test novo no test.csv:
   ├─ Calcula cosine similarity com TODOS os test cases do train
   └─ similarity(test_new, train_i) = dot(emb_test, emb_train_i)

2. Seleciona K=10 vizinhos mais próximos:
   ├─ Ordena por similarity (descendente)
   └─ Filtra: similarity > 0.5 (threshold)

3. Imputa features como média ponderada:
   └─ feature_imputed = Σ(feature_i × similarity_i) / Σ(similarity_i)

EXEMPLO:
┌──────────────┬─────────────┬──────────────┐
│ Vizinho      │ Similarity  │ failure_rate │
├──────────────┼─────────────┼──────────────┤
│ MCA-1015     │    0.89     │     0.23     │
│ MCA-102345   │    0.85     │     0.45     │
│ MCA-201567   │    0.78     │     0.12     │
│ ...          │    ...      │     ...      │
└──────────────┴─────────────┴──────────────┘

Imputação:
failure_rate = (0.23×0.89 + 0.45×0.85 + 0.12×0.78 + ...) / (0.89+0.85+0.78+...)
             ≈ 0.28
```

**Código (src/preprocessing/imputation.py linhas 50-95):**
```python
def impute_structural_features(
    train_embeddings, train_struct, tc_keys_train,
    test_embeddings, test_struct, tc_keys_test,
    tc_history, k_neighbors=10, similarity_threshold=0.5
):
    # Calcula similaridade: test vs train
    similarities = cosine_similarity(
        test_embeddings,   # [N_test, 1536]
        train_embeddings   # [N_train, 1536]
    )  # [N_test, N_train]

    for i, tc_key in enumerate(tc_keys_test):
        if tc_key not in tc_history:  # Test novo
            # Encontra K vizinhos mais similares
            neighbor_sims = similarities[i]  # [N_train]
            top_k_indices = np.argsort(neighbor_sims)[-k_neighbors:][::-1]

            # Filtra por threshold
            valid_neighbors = [
                idx for idx in top_k_indices
                if neighbor_sims[idx] > similarity_threshold
            ]

            if len(valid_neighbors) > 0:
                # Média ponderada
                weights = neighbor_sims[valid_neighbors]
                neighbor_features = train_struct[valid_neighbors]

                imputed_features = np.average(
                    neighbor_features,
                    axis=0,
                    weights=weights
                )

                test_struct[i] = imputed_features

    return test_struct
```

### FORMA 2: Predição via Rede Neural (NÃO, comparação implícita)

```
┌─────────────────────────────────────────────────────────────────┐
│  PREDIÇÃO: APRENDIZADO DE PADRÕES (SEM COMPARAÇÃO DIRETA)     │
└─────────────────────────────────────────────────────────────────┘

Durante TREINAMENTO:
- Modelo aprende PESOS que mapeiam embeddings → risco de falha
- NÃO memoriza embeddings específicos
- APRENDE padrões: "palavras X → alta prob de falha"

Durante INFERÊNCIA:
- Embedding do test novo passa pela MESMA rede neural
- Rede aplica pesos aprendidos
- NÃO compara com embeddings do treino
- Simplesmente aplica: f(embedding_test) = prob_falha

ANALOGIA:
┌────────────────────────────────────────────────────────────────┐
│  Treinamento: Professor ensina REGRAS                          │
│  "Se vejo palavras como 'login', 'auth' → alto risco"         │
│                                                                │
│  Inferência: Aluno aplica REGRAS em texto novo                │
│  "Este test tem 'login' → aplico regra → alto risco"          │
│                                                                │
│  Aluno NÃO precisa comparar com textos antigos!               │
│  Aluno apenas aplica as regras aprendidas                      │
└────────────────────────────────────────────────────────────────┘
```

**Visualização:**

```
TREINAMENTO (aprendizado de padrões):

Train Embeddings:
┌─────────────────┬────────────────┬────────┐
│ TC_Key          │ Embedding      │ Label  │
├─────────────────┼────────────────┼────────┤
│ MCA-1015        │ [0.12, -0.45,  │  FAIL  │
│ (login test)    │  ..., 0.78]    │        │
│                 │                │        │
│ MCA-102345      │ [0.15, -0.42,  │  FAIL  │
│ (login test)    │  ..., 0.75]    │        │
│                 │                │        │
│ MCA-201567      │ [-0.56, 0.23,  │  PASS  │
│ (UI test)       │  ..., -0.12]   │        │
└─────────────────┴────────────────┴────────┘
         │
         │ Backpropagation
         ▼
┌────────────────────────────────────────────┐
│  REDE NEURAL APRENDE:                      │
│  - Dimensões [0.12, -0.45, ...] → FAIL    │
│  - Padrão semântico de "login" → risco    │
│  - NÃO memoriza embeddings específicos     │
│  - Aprende TRANSFORMAÇÃO: emb → prob      │
└────────────────────────────────────────────┘

INFERÊNCIA (aplicação de padrões):

Test Embedding:
┌─────────────────┬────────────────┐
│ MCA-NEW-123     │ [0.14, -0.43,  │  ← Similar a MCA-1015
│ (login test)    │  ..., 0.76]    │     mas modelo NÃO compara!
└─────────────────┴────────────────┘
         │
         │ Forward pass
         ▼
┌────────────────────────────────────────────┐
│  REDE NEURAL APLICA PESOS APRENDIDOS:      │
│  - Detecta padrão semântico de "login"     │
│  - Aplica transformação aprendida          │
│  - Output: prob_fail = 0.72 (alto risco)   │
└────────────────────────────────────────────┘
```

### Código (dual_stream_v8.py linhas 580-615)

```python
def forward(self, semantic_input, structural_input, edge_index, edge_weights):
    # Semantic stream: NÃO compara com treino!
    # Apenas aplica transformação aprendida
    semantic_features = self.semantic_stream(semantic_input)
    # Input: [batch, 1536] → Output: [batch, 256]

    # Structural stream (GAT)
    structural_features = self.structural_stream(
        structural_input, edge_index, edge_weights
    )
    # Input: [batch, 6] → Output: [batch, 256]

    # Fusão
    fused = self.fusion(semantic_features, structural_features)
    # Output: [batch, 512]

    # Classificação
    logits = self.classifier(fused)
    # Output: [batch, 2] (Pass, Fail)

    return logits
```

---

## PIPELINE COMPLETO DE INFERÊNCIA

### Diagrama Geral

```
┌═══════════════════════════════════════════════════════════════╗
║                  INFERÊNCIA COMPLETA                          ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│  1. CARREGAMENTO DO TEST.CSV                                   │
│     - 31,333 samples, 277 builds                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. GERAÇÃO DE EMBEDDINGS (TODOS os test cases)                │
│     TC: SBERT(summary + steps) → [N, 768]                      │
│     Commit: SBERT(msg + diff) → [N, 768]                       │
│     Concatena: [N, 1536]                                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. EXTRAÇÃO DE FEATURES ESTRUTURAIS                           │
│     Para cada test case:                                        │
│     - Se conhecido: features históricas reais                   │
│     - Se novo: population defaults + imputação                  │
│     Resultado: [N, 6]                                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. MAPEAMENTO PARA GLOBAL INDICES                             │
│     - Conhecido: global_idx ∈ [0, 160]                         │
│     - Novo: global_idx = -1                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. LOOP DE BATCH INFERENCE                                     │
│                                                                 │
│  Para cada batch (ex: Build_789):                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Batch: [MCA-1015, MCA-NEW-123, MCA-101956, MCA-NEW-456] │  │
│  │ Indices: [0, -1, 1, -1]                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         ├─ 5.1 FILTRO: valid_mask = (indices != -1)           │
│         │      → [True, False, True, False]                    │
│         │                                                       │
│         ├─ 5.2 EXTRAÇÃO DE SUBGRAFO                           │
│         │      subset = [0, 1] (apenas MCA-1015, MCA-101956)  │
│         │      subgraph extrai arestas do grafo de treino      │
│         │                                                       │
│         ├─ 5.3 FORWARD PASS (apenas nós válidos)              │
│         │      ┌────────────────────────────────────────┐     │
│         │      │ Semantic Stream:                       │     │
│         │      │   [2, 1536] → [2, 256]                │     │
│         │      │                                        │     │
│         │      │ Structural Stream (GAT):               │     │
│         │      │   [2, 6] + subgraph → [2, 256]        │     │
│         │      │                                        │     │
│         │      │ Fusion:                                │     │
│         │      │   [2, 512]                             │     │
│         │      │                                        │     │
│         │      │ Classifier:                            │     │
│         │      │   [2, 2] logits                        │     │
│         │      └────────────────────────────────────────┘     │
│         │                                                       │
│         └─ 5.4 PREENCHER ORPHANS COM [0.5, 0.5]              │
│              Resultado final: [4, 2] probabilities            │
│              - MCA-1015: [0.28, 0.72] (GAT + Semantic)        │
│              - MCA-NEW-123: [0.5, 0.5] (orphan)               │
│              - MCA-101956: [0.88, 0.12] (GAT + Semantic)      │
│              - MCA-NEW-456: [0.5, 0.5] (orphan)               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. RANKING E APFD                                              │
│     - Ordena por prob_fail (decrescente)                        │
│     - Calcula APFD por build                                    │
│     - Retorna métricas                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Exemplo Detalhado de um Batch

```
╔═══════════════════════════════════════════════════════════════╗
║  BATCH INFERENCE: Build_789 (exemplo)                         ║
╚═══════════════════════════════════════════════════════════════╝

INPUT:
┌─────────────────┬──────────┬────────────┬─────────────────┐
│ TC_Key          │ global   │ Semantic   │ Structural      │
│                 │ _idx     │ Embed      │ Features        │
├─────────────────┼──────────┼────────────┼─────────────────┤
│ MCA-1015        │    0     │ [0.12,...] │ [45,0.23,...]   │
│ MCA-NEW-123     │   -1     │ [0.14,...] │ [0,0.28,...]    │
│ MCA-101956      │    1     │ [-0.56,...]│ [30,0.08,...]   │
│ MCA-NEW-456     │   -1     │ [0.45,...] │ [0,0.35,...]    │
└─────────────────┴──────────┴────────────┴─────────────────┘

STEP 1: Filtro (main.py linha 624)
valid_mask = [True, False, True, False]
Batch válido: 2 samples (MCA-1015, MCA-101956)

STEP 2: Subgrafo (main.py linha 637)
global_indices_valid = [0, 1]

Grafo de treino (edge_index):
  MCA-1015 (0) ←──co-falha──→ MCA-101956 (1)
       │                            │
  co-commit                    co-commit
       │                            │
       ↓                            ↓
  MCA-102345 (2)              MCA-999888 (160)

Subgrafo extraído (subset=[0,1]):
  Node 0 ←──co-falha──→ Node 1
  (MCA-1015)           (MCA-101956)

  Relabeled: [0, 1] → [0, 1] (já são os primeiros)

STEP 3: Forward Pass (dual_stream_v8.py linha 580)

SEMANTIC STREAM:
  Input: [[0.12, -0.45, ..., 0.78],    ← MCA-1015
          [-0.56, 0.23, ..., -0.12]]   ← MCA-101956
  Shape: [2, 1536]

  Layer 1: Linear(1536→512) + ReLU + Dropout
    → [2, 512]

  Layer 2: Linear(512→256) + ReLU + Dropout
    → [2, 256]

  Output: [[0.34, -0.12, ..., 0.56],   ← Semantic features MCA-1015
           [-0.23, 0.45, ..., -0.34]]  ← Semantic features MCA-101956

STRUCTURAL STREAM (GAT):
  Input: [[45.0, 0.23, 0.15, 0.08, 3.0, 0.0],   ← MCA-1015
          [30.0, 0.08, 0.05, 0.02, 2.0, 0.0]]   ← MCA-101956
  Shape: [2, 6]

  Edge_index: [[0, 1],    ← Source nodes
               [1, 0]]    ← Target nodes (bidirecional)
  Edge_weights: [0.85, 0.85]  (co-failure strength)

  GAT Layer 1 (4 heads):
    - Node 0 agrega de si mesmo + Node 1
    - Node 1 agrega de si mesmo + Node 0
    - Attention weights aprendidos
    → [2, 128]

  GAT Layer 2 (1 head):
    - Segunda camada de agregação
    → [2, 256]

  Output: [[0.12, 0.67, ..., -0.23],   ← Structural features MCA-1015
           [0.45, -0.12, ..., 0.89]]   ← Structural features MCA-101956

FUSION (Gated Fusion):
  Semantic: [2, 256]
  Structural: [2, 256]

  Gate = sigmoid(Linear(concat(sem, struct)))
  Fused = Gate ⊙ Semantic + (1-Gate) ⊙ Structural
  Final = concat(Semantic, Structural)

  Output: [2, 512]

CLASSIFIER:
  Input: [2, 512]
  Linear(512 → 2)

  Logits: [[0.12, 1.45],    ← MCA-1015
           [2.34, -0.67]]   ← MCA-101956

  Softmax: [[0.28, 0.72],   ← P(Pass)=0.28, P(Fail)=0.72
            [0.88, 0.12]]   ← P(Pass)=0.88, P(Fail)=0.12

STEP 4: Preencher Orphans (main.py linha 791)
full_probs = np.full((4, 2), 0.5)  # [0.5, 0.5] para todos
full_probs[[0, 2]] = [[0.28, 0.72], [0.88, 0.12]]  # Válidos

Resultado:
┌─────────────────┬────────────────┬────────────┐
│ TC_Key          │ P(Pass)        │ P(Fail)    │
├─────────────────┼────────────────┼────────────┤
│ MCA-1015        │     0.28       │    0.72    │  ← GAT+Semantic
│ MCA-NEW-123     │     0.50       │    0.50    │  ← Orphan
│ MCA-101956      │     0.88       │    0.12    │  ← GAT+Semantic
│ MCA-NEW-456     │     0.50       │    0.50    │  ← Orphan
└─────────────────┴────────────────┴────────────┘

RANKING (por P(Fail)):
1. MCA-1015        (0.72) → Alto risco
2. MCA-NEW-123     (0.50) → Incerto (novo)
3. MCA-NEW-456     (0.50) → Incerto (novo)
4. MCA-101956      (0.12) → Baixo risco

EXECUÇÃO:
MCA-1015 → FAIL ✓ (detectado cedo!)
MCA-NEW-123 → PASS
MCA-NEW-456 → PASS
MCA-101956 → PASS

APFD = (1×1 + 0) / (1×4) = 0.25
```

---

## RESUMO: PERGUNTAS RESPONDIDAS

### 1. Por que NOVO! (global_idx = -1)?
- `-1` é um **valor sentinela** retornado por `dict.get(key, -1)` quando a chave não existe
- Indica que o test case **não estava no train.csv**
- Permite detectar facilmente test cases novos: `if idx == -1`

### 2. Os nós novos são adicionados ao grafo global?
- **NÃO!** ❌
- O grafo permanece **estático** após o treinamento
- Test cases novos são classificados como **orphans** (global_idx = -1)
- Orphans são **filtrados antes** do processamento GAT
- Recebem predição default **[0.5, 0.5]** (máxima incerteza)

### 3. Como são extraídas as features estruturais do test.csv?
- **Test cases conhecidos**: Features históricas reais do treino
- **Test cases novos**:
  - Defaults conservadores (population statistics, NÃO zeros)
  - Imputação via vizinhos semânticos (K=10, similarity > 0.5)
  - `test_novelty = 1.0` (flag de novo)

### 4. Como funciona a parte semântica (embeddings)?
- **Geração**: SBERT (all-mpnet-base-v2) processa textos
  - TC: summary + steps → [768]
  - Commit: message + diff → [768]
  - Concatena → [1536]
- **Processamento**: Rede neural aprende padrões semânticos
  - NÃO memoriza embeddings específicos
  - APRENDE transformação: embedding → probabilidade de falha
- **Vantagem**: Funciona para test cases novos!

### 5. Embeddings do test.csv são comparados com os do treino?
- **Imputação**: SIM ✓ (comparação explícita via cosine similarity)
  - Encontra vizinhos similares para emprestar features
- **Predição**: NÃO ❌ (aprendizado de padrões)
  - Modelo aplica transformação aprendida
  - NÃO compara com embeddings do treino

---

## DIAGRAMA FINAL: VISÃO COMPLETA

```
┌═══════════════════════════════════════════════════════════════╗
║              ARQUITETURA DUAL-STREAM (INFERÊNCIA)             ║
╚═══════════════════════════════════════════════════════════════╝

TEST.CSV (31,333 samples)
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │ TC_Key   │      │ Text     │      │ Build    │
   │ Mapping  │      │ Data     │      │ Info     │
   └──────────┘      └──────────┘      └──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │ global   │      │ SBERT    │      │ Struct   │
   │ _idx     │      │ Embed    │      │ Features │
   │ [0..160  │      │ [N,1536] │      │ [N, 6]   │
   │ ou -1]   │      │          │      │          │
   └──────────┘      └──────────┘      └──────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
      Valid (idx≠-1)              Orphan (idx=-1)
              │                           │
              ▼                           ▼
      ┌───────────────┐          ┌──────────────┐
      │  DUAL-STREAM  │          │  Default     │
      │               │          │  [0.5, 0.5]  │
      │  Semantic:    │          └──────────────┘
      │  [1536→256]   │
      │               │
      │  Structural:  │
      │  GAT [6→256]  │
      │               │
      │  Fusion:      │
      │  [512]        │
      │               │
      │  Classifier:  │
      │  [2]          │
      └───────────────┘
              │
              └──────────────┬────────────────┐
                             │                │
                             ▼                ▼
                     ┌──────────────┐  ┌──────────────┐
                     │ Valid Probs  │  │ Orphan Probs │
                     │ [N_valid, 2] │  │ [N_orphan,2] │
                     └──────────────┘  └──────────────┘
                             │                │
                             └────────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Full Probs   │
                              │ [N, 2]       │
                              └──────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Ranking      │
                              │ (by P(Fail)) │
                              └──────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ APFD Calc    │
                              │ Per Build    │
                              └──────────────┘
```

---

## CONCLUSÃO

Este sistema implementa uma **abordagem híbrida conservadora**:

1. **Test cases conhecidos**: Máxima informação (histórico + grafo + semântica)
2. **Test cases novos**: Degradação graciosa (semântica + defaults + incerteza)
3. **Segurança**: Defaults conservadores evitam bias "falso negativo"
4. **Escalabilidade**: Subgrafo extraction permite processar grandes datasets

É um **design intencional** para um sistema de priorização de testes, onde **falsos negativos** (não detectar falhas) são **muito custosos**.

---

**Autor**: Sistema Dual-Stream para Test Case Prioritization
**Data**: 2025-11-15
**Versão**: v8
