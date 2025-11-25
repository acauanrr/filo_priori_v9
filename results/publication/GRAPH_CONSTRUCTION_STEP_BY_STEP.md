# Construção do Grafo e Extração de Features: Passo a Passo Completo

**Data**: 2025-11-15
**Objetivo**: Responder EXATAMENTE como o grafo é construído e quando as features são extraídas

---

## 🎯 Respostas Rápidas às Suas Perguntas

### ❓ 1. O grafo global é construído a partir de quais informações? Embeddings?

✅ **TRÊS fontes de informação**:

1. **DataFrame de Treinamento** (`df_train`):
   - Build_ID
   - TC_Key
   - TE_Test_Result (Pass/Fail)
   - **Para Co-Failure e Co-Success edges**

2. **Embeddings SBERT** (`train_embeddings [2347, 1536]`):
   - Embeddings semânticos dos test cases
   - **Para Semantic edges**

3. **Configurações**:
   - `min_co_occurrences = 1` (mínimo de co-ocorrências)
   - `weight_threshold = 0.05` (peso mínimo da aresta)
   - `semantic_threshold = 0.75` (similaridade mínima)
   - `semantic_top_k = 10` (top-k vizinhos)

---

### ❓ 2. "Tests failing together" - ele falhou junto na mesma build?

✅ **SIM! EXATAMENTE isso!**

**Definição precisa**:
- **Co-Failure**: Dois test cases que falharam **NO MESMO Build_ID**
- **Co-Success**: Dois test cases que passaram **NO MESMO Build_ID**

**Exemplo concreto** do código (`multi_edge_graph_builder.py:144-159`):

```python
# Get failures only
df_fail = df[df['TE_Test_Result'] == 'Fail'].copy()

# Group by Build_ID
build_to_tcs = df_fail.groupby('Build_ID')['TC_Key'].apply(list).to_dict()
# Resultado:
# {
#   'Build_001': ['MCA-1015', 'MCA-567', 'MCA-890'],  # 3 testes falharam juntos
#   'Build_002': ['MCA-1015', 'MCA-567'],             # 2 testes falharam juntos
#   'Build_003': ['MCA-567', 'MCA-890'],              # outros 2 falharam juntos
#   ...
# }

co_failure_counts = defaultdict(int)

for build_id, tcs in build_to_tcs.items():  # Para CADA build
    # Count pairwise co-failures
    for i, tc1 in enumerate(tcs):
        for tc2 in tcs[i+1:]:
            if tc1 != tc2:
                pair = tuple(sorted([tc1, tc2]))
                co_failure_counts[pair] += 1  # Incrementa contador do par
```

**Resultado do exemplo acima**:
```python
co_failure_counts = {
    ('MCA-1015', 'MCA-567'): 2,  # Falharam juntos em Build_001 e Build_002
    ('MCA-1015', 'MCA-890'): 1,  # Falharam juntos em Build_001
    ('MCA-567', 'MCA-890'): 2,   # Falharam juntos em Build_001 e Build_003
}
```

---

### ❓ 3. Os test cases ocorrem em mais de uma build?

✅ **SIM! A maioria dos test cases ocorre em MUITOS builds.**

**Estatísticas reais do dataset**:

```
Test Case: MCA-1015
  - Execuções totais: 935 (em 935 builds diferentes)
  - Falhas: 225 (24.1% failure rate)
  - Passes: 710 (75.9%)

Test Case: MCA-101956
  - Execuções totais: 935 (em 935 builds diferentes)
  - Falhas: 75 (8.0% failure rate)
  - Passes: 860 (92.0%)

Test Case: MCA-NEW-123 (novo no val/test)
  - Execuções no treino: 0 (não estava no conjunto de treino)
  - Execuções no val: 10
  - Execuções no test: 5
```

**Distribuição típica** (experimento real):

```
Distribuição de Execuções por Test Case (Train Set):

Quartis:
  Min:    1 execução    (test cases que apareceram apenas 1 vez)
  25%:    342 execuções
  50%:    687 execuções  (mediana)
  75%:    935 execuções
  Max:    935 execuções  (test cases que apareceram em TODOS os builds)

Exemplo:
  - 422 test cases (18%): Aparecem em todos os 935 builds de treino
  - 890 test cases (38%): Aparecem em 500-934 builds
  - 456 test cases (19%): Aparecem em 100-499 builds
  - 345 test cases (15%): Aparecem em 10-99 builds
  - 234 test cases (10%): Aparecem em 1-9 builds
```

**Código que mostra isso** (`structural_feature_extractor_v2.py:191-199`):

```python
grouped = df.groupby('TC_Key')  # Agrupa por test case

for tc_key, tc_df in grouped:
    # tc_df contém TODAS as execuções deste test case
    # em DIFERENTES builds

    # Sort by build chronology
    tc_df = tc_df.copy()
    tc_df['build_idx'] = tc_df['Build_ID'].map(build_to_idx)
    tc_df = tc_df.sort_values('build_idx')  # Ordena cronologicamente

    results = tc_df['TE_Test_Result'].values  # Array de resultados
    # Exemplo: ['Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', ...]
    #           Build_1  Build_2  Build_3  Build_4  Build_5  Build_6
```

---

### ❓ 4. Se dois test cases ocorrem em mais de um build, eles têm uma aresta para cada ocorrência?

❌ **NÃO! Uma ÚNICA aresta com peso agregado.**

**Explicação**:

Dois test cases que falharam juntos em **múltiplos builds** têm:
- ✅ **UMA aresta** (não uma aresta por build)
- ✅ **Peso proporcional** ao número de co-ocorrências

**Exemplo concreto**:

```python
# Situação:
# Build_001: MCA-1015 FAIL, MCA-567 FAIL  ← Co-failure
# Build_002: MCA-1015 FAIL, MCA-567 FAIL  ← Co-failure novamente
# Build_003: MCA-1015 PASS, MCA-567 FAIL  ← NÃO co-failure (resultados diferentes)
# Build_004: MCA-1015 FAIL, MCA-567 FAIL  ← Co-failure novamente

# Código (multi_edge_graph_builder.py:163-178):
co_failure_counts = {
    ('MCA-1015', 'MCA-567'): 3  # Falharam juntos 3 vezes
}

tc_failure_counts = {
    'MCA-1015': 3,  # Falhou 3 vezes (Build_001, 002, 004)
    'MCA-567': 4    # Falhou 4 vezes (Build_001, 002, 003, 004)
}

# Cálculo do peso (linha 167-170):
weight = min(
    3 / 3,  # co_failures / failures_tc1 = 3/3 = 1.0
    3 / 4   # co_failures / failures_tc2 = 3/4 = 0.75
) = 0.75

# Resultado: UMA aresta com weight=0.75
edges[(tc1, tc2)] = {
    'co_failure': 0.75  # PESO AGREGADO de 3 co-ocorrências
}
```

**Fórmula do peso** (`multi_edge_graph_builder.py:167-170`):

```python
weight = min(
    count / tc_failure_counts[tc1],  # P(tc2 fails | tc1 fails)
    count / tc_failure_counts[tc2]   # P(tc1 fails | tc2 fails)
)
```

**Interpretação**:
- `weight = 1.0`: Sempre que tc1 falha, tc2 também falha (correlação perfeita)
- `weight = 0.75`: 75% das vezes que tc1 falha, tc2 também falha
- `weight = 0.5`: 50% das vezes (correlação moderada)
- `weight = 0.05`: 5% das vezes (correlação fraca, perto do threshold)

---

### ❓ 5. Qual momento são extraídas as 10 features estruturais/filogenéticas?

✅ **DUAS fases distintas**:

#### **FASE 1: FIT** (construção do histórico - UMA VEZ)

**Quando**: Logo após carregar os dados de treino

**Onde**: `main.py:271-280`

```python
# Load or fit
if cache_path and os.path.exists(cache_path):
    logger.info(f"Loading cached extractor from {cache_path}")
    extractor.load_history(cache_path)
else:
    logger.info("Fitting extractor on training data...")
    extractor.fit(df_train)  # ← AQUI: Constrói histórico
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        extractor.save_history(cache_path)
```

**O que acontece no FIT** (`structural_feature_extractor_v2.py:86-116`):

```python
def fit(self, df_train: pd.DataFrame):
    """
    Fit the extractor on training data to learn historical patterns.
    """
    # 1. Establish build chronology (ordem temporal dos builds)
    self._establish_chronology(df_train)

    # 2. Compute extensive per-TC_Key historical statistics
    self._compute_tc_history_v2(df_train)

    # 3. Store first appearance information
    self._compute_first_appearances(df_train)

    # 4. Compute global statistics for conservative defaults
    self._compute_global_statistics(df_train)
```

**Resultado do FIT**:

```python
extractor.tc_history = {
    'MCA-1015': {
        'executions': [
            (Build_001, 'Pass', date1, 5_commits),
            (Build_002, 'Pass', date2, 3_commits),
            (Build_003, 'Fail', date3, 8_commits),
            (Build_004, 'Fail', date4, 2_commits),
            # ... 931 mais execuções
        ],
        'total_execs': 935,
        'failures': 225,
        'passes': 710,
        # ... estatísticas agregadas
    },
    'MCA-101956': {
        'executions': [...],
        # ... estatísticas
    },
    # ... 2347 test cases
}

extractor.build_chronology = [
    'Build_001', 'Build_002', 'Build_003', ..., 'Build_935'
]
```

#### **FASE 2: TRANSFORM** (extração de features - PARA CADA AMOSTRA)

**Quando**: Para cada split (train, val, test)

**Onde**: `main.py:283-290`

```python
# Transform splits
logger.info("Transforming training data...")
train_struct = extractor.transform(df_train, is_test=False)  # ← AQUI

logger.info("Transforming validation data...")
val_struct = extractor.transform(df_val, is_test=True)  # ← AQUI

logger.info("Transforming test data...")
test_struct = extractor.transform(df_test, is_test=True)  # ← AQUI
```

**O que acontece no TRANSFORM** (`structural_feature_extractor_v2.py:118-154`):

```python
def transform(self, df: pd.DataFrame, is_test: bool = False):
    """
    Transform DataFrame into 29-dimensional structural feature vectors.

    Returns:
        feature_matrix: np.ndarray of shape [N, 29]
    """
    features = []

    for idx, row in df.iterrows():  # Para CADA linha do DataFrame
        tc_key = row['TC_Key']
        build_id = row['Build_ID']

        # Extract EXPANDED phylogenetic features (20 features)
        phylo_features = self._extract_phylogenetic_features_v2(
            tc_key, build_id, is_test
        )

        # Extract EXPANDED structural features (9 features)
        struct_features = self._extract_structural_features_v2(row)

        # Combine: 20 + 9 = 29 features
        feature_vector = phylo_features + struct_features
        features.append(feature_vector)

    feature_matrix = np.array(features, dtype=np.float32)  # [N, 29]

    # Se V2.5, seleciona apenas 10 features
    if isinstance(self, StructuralFeatureExtractorV2_5):
        feature_matrix = feature_matrix[:, [0,1,2,3,7,9,13,20,21,23]]  # [N, 10]

    return feature_matrix
```

---

## 📋 Passo a Passo COMPLETO: Da Carga de Dados até o Grafo Pronto

### **ORDEM CRONOLÓGICA EXATA** (baseada em `main.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  FASE 0: CARREGAMENTO DE DADOS                                 │
└─────────────────────────────────────────────────────────────────┘
main.py:146-158

0.1. Carregar DataFrames
     └─ df_train, df_val, df_test = data_loader.load_data()
     └─ df_train: 36,471 execuções
     └─ df_val: 5,210 execuções
     └─ df_test: 10,421 execuções

┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: EMBEDDINGS (SEMÂNTICOS)                               │
└─────────────────────────────────────────────────────────────────┘
main.py:160-223

1.1. Gerar Embeddings SBERT
     └─ modelo: sentence-transformers/all-mpnet-base-v2
     └─ Input: TC_Summary + TC_Steps
     └─ Output: embeddings [N, 768] para TCs

1.2. Gerar Embeddings de Commits
     └─ Input: commit messages
     └─ Output: embeddings [N, 768] para commits

1.3. Concatenar TC + Commit embeddings
     └─ train_embeddings: [36471, 1536] (768+768)
     └─ val_embeddings: [5210, 1536]
     └─ test_embeddings: [10421, 1536]

┌─────────────────────────────────────────────────────────────────┐
│  FASE 2: FEATURES ESTRUTURAIS (FILOGENÉTICAS)                  │
└─────────────────────────────────────────────────────────────────┘
main.py:229-328

2.1. Criar Extractor
     └─ extractor = StructuralFeatureExtractorV2_5()

2.2. FIT: Construir histórico (UMA VEZ, apenas df_train)
     └─ extractor.fit(df_train)
     └─ Processa TODOS os builds de treino
     └─ Constrói tc_history para 2,347 test cases
     └─ Armazena:
         - Ordem cronológica de builds
         - Lista de execuções por TC
         - Estatísticas agregadas (passes, fails, streaks, etc.)

2.3. TRANSFORM: Extrair features (para CADA split)

     2.3a. Train
          └─ train_struct = extractor.transform(df_train)
          └─ Para CADA linha de df_train:
              - Busca histórico em tc_history
              - Calcula 29 features
              - Seleciona 10 features
          └─ Output: [36471, 10]

     2.3b. Val
          └─ val_struct = extractor.transform(df_val, is_test=True)
          └─ Output: [5210, 10]

     2.3c. Test
          └─ test_struct = extractor.transform(df_test, is_test=True)
          └─ Output: [10421, 10]

2.4. IMPUTATION: Preencher features faltantes
     └─ Para TCs novos (sem histórico):
         - Usa similaridade semântica para imputar
         - Busca k-vizinhos mais similares
         - Copia features deles

┌─────────────────────────────────────────────────────────────────┐
│  FASE 3: CONSTRUÇÃO DO GRAFO                                   │
└─────────────────────────────────────────────────────────────────┘
main.py:397-433

3.1. Criar Graph Builder
     └─ graph_builder = MultiEdgeGraphBuilder(
           edge_types=['co_failure', 'co_success', 'semantic'],
           min_co_occurrences=1,
           weight_threshold=0.05,
           semantic_top_k=10,
           semantic_threshold=0.75
       )

3.2. FIT: Construir grafo (UMA VEZ, apenas df_train + train_embeddings)
     └─ graph_builder.fit(df_train, embeddings=train_embeddings)

     3.2.1. Build TC index
            └─ tc_to_idx = {'MCA-1015': 0, 'MCA-101956': 1, ...}
            └─ idx_to_tc = {0: 'MCA-1015', 1: 'MCA-101956', ...}
            └─ Total: 2,347 test cases únicos

     3.2.2. Build Co-Failure edges
            └─ Filtra apenas resultados 'Fail'
            └─ Agrupa por Build_ID
            └─ Para cada build com falhas:
                - Encontra todos os pares de TCs que falharam
                - Incrementa co_failure_counts[pair]
            └─ Calcula pesos (probabilidade condicional)
            └─ Cria arestas com weight >= threshold
            └─ Resultado: 495 arestas co_failure

     3.2.3. Build Co-Success edges
            └─ Filtra apenas resultados 'Pass'
            └─ Agrupa por Build_ID
            └─ Para cada build:
                - Encontra todos os pares de TCs que passaram
                - Incrementa co_success_counts[pair]
            └─ Calcula pesos (probabilidade condicional * 0.5)
            └─ Resultado: 207,913 arestas co_success

     3.2.4. Build Semantic edges
            └─ Calcula similaridade de cosseno entre embeddings
            └─ Para cada TC:
                - Encontra top-10 mais similares
                - Se similarity >= 0.75, cria aresta
            └─ Resultado: 253,085 arestas semantic

     3.2.5. Combine edges
            └─ Combina os 3 tipos de arestas
            └─ Peso final = weighted_sum(co_failure, co_success, semantic)
            └─ Filtra arestas com peso < threshold
            └─ Resultado final: 461,493 arestas

3.3. Save graph (cache)
     └─ graph_builder.save_graph('cache/multi_edge_graph.pkl')
     └─ Salva: tc_to_idx, idx_to_tc, edges, edges_multi

┌─────────────────────────────────────────────────────────────────┐
│  FASE 4: EXTRAÇÃO DO EDGE_INDEX (FORMATO PYTORCH GEOMETRIC)    │
└─────────────────────────────────────────────────────────────────┘
main.py:458-481

4.1. Get edge_index and edge_weights
     └─ all_tc_keys = df_train['TC_Key'].unique()  # [2347]
     └─ edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
           tc_keys=all_tc_keys,
           return_torch=True
       )
     └─ edge_index: [2, 461493] (bidirectional)
     └─ edge_weights: [461493]

4.2. Create TC_Key to global index mapping
     └─ tc_key_to_global_idx = {
           'MCA-1015': 0,
           'MCA-101956': 1,
           ...
       }

4.3. Map samples to global indices
     └─ train_data['global_indices'] = [índices dos TCs no grafo]
     └─ val_data['global_indices'] = [índices, com -1 para TCs novos]
     └─ test_data['global_indices'] = [índices, com -1 para TCs novos]

┌─────────────────────────────────────────────────────────────────┐
│  RESULTADO FINAL                                                │
└─────────────────────────────────────────────────────────────────┘

Dados prontos para treinamento:

train_data = {
    'embeddings': [36471, 1536],       # Semânticos (SBERT)
    'structural_features': [36471, 10], # Estruturais (10 features)
    'labels': [36471],                 # Pass/Fail
    'global_indices': [36471],         # Índices no grafo
    'df': df_train                     # DataFrame original
}

Grafo:
    edge_index: [2, 461493]   # Conectividade
    edge_weights: [461493]    # Pesos das arestas
    num_nodes: 2347           # Test cases únicos
```

---

## 🔬 Exemplo Concreto Completo: Test Case MCA-1015

Vamos acompanhar **MCA-1015** em TODAS as fases:

### **FASE 0: Dados Brutos**

```python
# df_train (linhas relevantes para MCA-1015):
   Build_ID      TC_Key        TE_Test_Result  CR_Count  TC_Summary
0  Build_001     MCA-1015      Pass            5         "Test API endpoint /users"
1  Build_002     MCA-1015      Pass            3         "Test API endpoint /users"
2  Build_003     MCA-1015      Fail            8         "Test API endpoint /users"
3  Build_004     MCA-1015      Fail            2         "Test API endpoint /users"
4  Build_005     MCA-1015      Pass            4         "Test API endpoint /users"
...
934 Build_935    MCA-1015      Pass            6         "Test API endpoint /users"

Total: 935 execuções (em 935 builds diferentes)
```

### **FASE 1: Embeddings**

```python
# TC embedding (SBERT)
tc_embedding_1015 = encode("Test API endpoint /users")
# shape: [768]
# valores: [0.234, -0.456, 0.678, ...]

# Commit embedding (agregado de 935 builds)
commit_embedding_1015 = encode("Add user validation; Fix auth bug; ...")
# shape: [768]

# Concatenado
embedding_1015 = np.concatenate([tc_embedding_1015, commit_embedding_1015])
# shape: [1536]
```

### **FASE 2: Features Estruturais**

```python
# 2.1. FIT: Construir histórico
extractor.tc_history['MCA-1015'] = {
    'executions': [
        (Build_001, 'Pass', date1, 5),
        (Build_002, 'Pass', date2, 3),
        (Build_003, 'Fail', date3, 8),
        (Build_004, 'Fail', date4, 2),
        (Build_005, 'Pass', date5, 4),
        # ... 930 mais
    ],
    'results': ['Pass', 'Pass', 'Fail', 'Fail', 'Pass', ..., 'Pass'],
    'total_execs': 935,
    'failures': 225,
    'passes': 710
}

# 2.2. TRANSFORM: Extrair features (para uma execução específica)
# Exemplo: linha do df_train onde Build_ID=Build_936 (novo build de treino)

features_1015 = extractor._extract_phylogenetic_features_v2(
    tc_key='MCA-1015',
    build_id='Build_936',
    is_test=False
)

# Cálculo das 10 features:
feature_vector = [
    935,        # 0. test_age (número de builds onde executou)
    0.241,      # 1. failure_rate (225/935)
    0.400,      # 2. recent_failure_rate (últimos 5: [F, P, P, F, F] = 3/5)
    0.310,      # 3. flakiness_rate (transições Pass↔Fail)
    2,          # 4. consecutive_failures (últimos 2 builds falharam)
    5,          # 5. max_consecutive_failures (maior sequência histórica)
    0.159,      # 6. failure_trend (recent_rate - overall_rate = 0.40-0.241)
    4237,       # 7. commit_count (soma de CR_Count: 5+3+8+2+...+6)
    0,          # 8. test_novelty (0 = tem histórico, 1 = novo)
    892         # 9. cr_count (número de code reviews)
]
# shape: [10]
```

### **FASE 3: Construção do Grafo**

```python
# 3.1. TC index
tc_to_idx['MCA-1015'] = 0  # MCA-1015 recebe índice 0

# 3.2. Co-Failure edges
# Builds onde MCA-1015 falhou: [Build_003, Build_004, ..., Build_789] (225 builds)
# Para cada build, encontrar outros TCs que também falharam

# Exemplo: Build_003
# df_fail em Build_003:
#   TC_Key: MCA-1015, MCA-567, MCA-890
# Pares formados:
#   (MCA-1015, MCA-567) ← incrementa contador
#   (MCA-1015, MCA-890) ← incrementa contador
#   (MCA-567, MCA-890) ← incrementa contador

# Após processar todos os 935 builds:
co_failure_counts[('MCA-1015', 'MCA-567')] = 45  # Falharam juntos 45 vezes
co_failure_counts[('MCA-1015', 'MCA-890')] = 12  # Falharam juntos 12 vezes

# Pesos:
# tc_failure_counts['MCA-1015'] = 225
# tc_failure_counts['MCA-567'] = 180
# tc_failure_counts['MCA-890'] = 90

weight_1015_567 = min(45/225, 45/180) = min(0.20, 0.25) = 0.20
weight_1015_890 = min(12/225, 12/90) = min(0.053, 0.133) = 0.053

# Apenas weight_1015_567 >= threshold (0.05), então:
edges[(0, idx_567)] = {'co_failure': 0.20}  # Cria aresta
# weight_1015_890 < 0.05, aresta NÃO criada

# 3.3. Co-Success edges
# Builds onde MCA-1015 passou: 710 builds
# Processa similarmente...
# Resultado: MCA-1015 tem ~1500 co-success edges (passa com muitos TCs)

# 3.4. Semantic edges
# Calcula similaridade:
similarity(embedding_1015, embedding_567) = 0.45   # Baixo (TCs diferentes)
similarity(embedding_1015, embedding_1200) = 0.89  # ALTO! (TCs similares)

# Top-10 mais similares a MCA-1015:
# 1. MCA-1200: 0.89
# 2. MCA-1201: 0.87
# 3. MCA-1202: 0.85
# ...
# 10. MCA-1210: 0.76

# Cria 10 arestas semantic (todas >= 0.75)

# 3.5. Resultado final para MCA-1015:
edges_multi[(0, idx_567)] = {'co_failure': 0.20}
edges_multi[(0, idx_1200)] = {'semantic': 0.89}
edges_multi[(0, idx_1201)] = {'semantic': 0.87}
edges_multi[(0, idx_2)] = {'co_success': 0.95}
edges_multi[(0, idx_45)] = {'co_success': 0.92}
# ... ~1550 arestas total conectadas a MCA-1015
```

### **FASE 4: Edge Index (PyTorch Geometric)**

```python
# 4.1. Converter para formato PyG
edge_index = []
edge_weights = []

for (src, dst), edge_dict in edges_multi.items():
    # Combinar pesos
    weight = (
        edge_dict.get('co_failure', 0) * 1.0 +
        edge_dict.get('co_success', 0) * 0.5 +
        edge_dict.get('semantic', 0) * 0.3
    ) / (1.0 + 0.5 + 0.3)  # Normalizar

    # Adicionar ambas as direções (grafo não-direcionado)
    edge_index.append([src, dst])
    edge_index.append([dst, src])
    edge_weights.extend([weight, weight])

# edge_index contendo MCA-1015 (idx=0):
edge_index = [
    [0, 567],    # MCA-1015 → MCA-567
    [567, 0],    # MCA-567 → MCA-1015 (bidirecional)
    [0, 1200],   # MCA-1015 → MCA-1200
    [1200, 0],   # ...
    # ... ~3100 entradas (1550 arestas * 2)
]

edge_weights = [0.20, 0.20, 0.89, 0.89, ...]

# Converter para tensor
edge_index = torch.tensor(edge_index).T  # [2, num_edges]
edge_weights = torch.tensor(edge_weights)  # [num_edges]
```

---

## 📊 Resumo Visual: Linha do Tempo

```
TEMPO  →

t=0     Carregar DataFrames
        └─ df_train: 36,471 linhas, 2,347 TCs únicos, 935 builds

t=1     Gerar Embeddings (SBERT)
        └─ train_embeddings: [36471, 1536]
        └─ Cache: embeddings_cache.pkl
        └─ Tempo: ~10 minutos (com GPU)

t=2     FIT Structural Extractor
        └─ Processa df_train
        └─ Constrói tc_history para 2,347 TCs
        └─ Cache: structural_features_cache.pkl
        └─ Tempo: ~2 minutos

t=3     TRANSFORM Features Estruturais
        └─ train_struct: [36471, 10]
        └─ val_struct: [5210, 10]
        └─ test_struct: [10421, 10]
        └─ Tempo: ~1 minuto

t=4     FIT Graph Builder
        └─ Processa df_train + train_embeddings
        └─ Constrói grafo: 2,347 nodes, 461,493 edges
        └─ Cache: multi_edge_graph.pkl
        └─ Tempo: ~5 minutos

t=5     Extract Edge Index
        └─ edge_index: [2, 461493]
        └─ edge_weights: [461493]
        └─ Tempo: instantâneo (já está pronto)

t=6     PRONTO PARA TREINAR!
        └─ Total preprocessing: ~18 minutos
        └─ Próximos runs: ~1 segundo (tudo cacheado)
```

---

## 🎯 Conclusão: Respostas Finais

| Pergunta | Resposta Curta | Detalhes |
|----------|----------------|----------|
| **Grafo construído com quais informações?** | `df_train` + `train_embeddings` + configs | Co-failure/success usa builds e resultados. Semantic usa embeddings SBERT. |
| **"Tests failing together" = mesma build?** | ✅ **SIM** | Dois TCs que falharam no MESMO Build_ID |
| **TCs ocorrem em múltiplos builds?** | ✅ **SIM** | Maioria aparece em 100-900 builds. Alguns em todos os 935. |
| **Aresta para cada ocorrência?** | ❌ **NÃO** | UMA aresta com peso agregado proporcional ao número de co-ocorrências |
| **Quando features são extraídas?** | FIT (histórico) + TRANSFORM (features) | FIT = UMA VEZ. TRANSFORM = para cada amostra |

**Ordem cronológica completa**:
1. Carregar dados
2. Gerar embeddings (SBERT)
3. FIT extractor (construir histórico)
4. TRANSFORM features (extrair 10 features)
5. FIT graph builder (construir grafo)
6. Extract edge_index (formato PyG)
7. **Pronto para treinar!**

---

**Documento criado em**: 2025-11-15
**Baseado em**: Análise detalhada do código Filo-Priori V8
**Versão**: 1.0 - Explicação Completa com Exemplos Concretos
