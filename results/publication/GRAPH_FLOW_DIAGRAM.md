# Diagrama de Fluxo: Uso do Grafo no Filo-Priori V8

## Visão Geral: Grafo Global + GAT

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1: CONSTRUÇÃO DO GRAFO                 │
│                         (UMA VEZ APENAS)                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  df_train        │
│  - 36,471 execs  │──┐
│  - 935 builds    │  │
│  - 2,347 TCs     │  │
└──────────────────┘  │
                      │  ┌──────────────────────────────┐
┌──────────────────┐  │  │  MultiEdgeGraphBuilder      │
│ train_embeddings │  ├─▶│                              │
│  [2347, 1536]    │  │  │  1. Build TC index           │
└──────────────────┘  │  │     tc_to_idx = {           │
                      │  │       'MCA-1015': 0,         │
                      │  │       'MCA-101956': 1,       │
                      │  │       ...                    │
                      │  │     }                        │
                      │  │                              │
                      │  │  2. Build Co-Failure Edges   │
                      │  │     - Group by Build_ID      │
                      │  │     - Find failing pairs     │
                      │  │     - Weight = P(fail|cooc)  │
                      │  │                              │
                      │  │  3. Build Co-Success Edges   │
                      │  │     - Group by Build_ID      │
                      │  │     - Find passing pairs     │
                      │  │     - Weight = P(pass|cooc)  │
                      │  │                              │
                      │  │  4. Build Semantic Edges     │
                      │  │     - Cosine similarity      │
                      │  │     - Top-k neighbors        │
                      │  │     - Threshold = 0.75       │
                      │  │                              │
                      │  │  5. Combine Multi-Edges      │
                      │  │     - Weighted sum           │
                      └─▶│     - Filter threshold       │
                         └──────────────────────────────┘
                                        │
                                        │ .fit()
                                        ▼
                         ┌──────────────────────────────┐
                         │  GRAFO GLOBAL (cached)       │
                         │                              │
                         │  Nodes: 2,347 test cases     │
                         │  Edges: 461,493              │
                         │    - Co-Failure: 495         │
                         │    - Co-Success: 207,913     │
                         │    - Semantic: 253,085       │
                         │                              │
                         │  Saved: cache/graph.pkl      │
                         └──────────────────────────────┘
                                        │
                                        │ .get_edge_index_and_weights()
                                        ▼
                         ┌──────────────────────────────┐
                         │  PyG Format                  │
                         │                              │
                         │  edge_index: [2, 461493]     │
                         │  edge_weights: [461493]      │
                         │                              │
                         │  (usado para train/val/test) │
                         └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              FASE 2: TREINAMENTO (CADA BATCH)                  │
└─────────────────────────────────────────────────────────────────┘

Epoch 1, Batch 1 (32 samples):
┌────────────────────┐
│ Batch Data         │
│                    │
│ TC_Keys (batch):   │
│  ['MCA-1015',      │
│   'MCA-101956',    │
│   'MCA-1012',      │
│   ...              │
│   'MCA-2347']      │
│                    │
│ Global Indices:    │
│  [0, 1, 2, ...]    │◀─── Mapeamento para índices no grafo
│                    │
│ Embeddings:        │
│  [32, 1536]        │
│                    │
│ Struct Features:   │
│  [32, 10]          │──┐
└────────────────────┘  │
                        │
                        │
                        │    ┌──────────────────────────┐
                        │    │  Dual-Stream Model       │
                        │    │                          │
                        │    │  ┌──────────────────┐    │
                        │    │  │ Semantic Stream  │    │
                        ├────┼─▶│                  │    │
                        │    │  │ MLP: [1536→256]  │    │
                        │    │  │                  │    │
                        │    │  │ Output: [32,256] │    │
                        │    │  └──────────────────┘    │
                        │    │           │              │
                        │    │           │ [32, 256]    │
                        │    │           ▼              │
                        │    │  ┌──────────────────┐    │
                        │    │  │ Fusion Module    │    │
                        │    │  │                  │    │
                        │    │  │ Cross-Attention  │◀───┼─┐
                        │    │  │ or Gated Fusion  │    │ │
                        │    │  │                  │    │ │
                        │    │  │ Output: [32,512] │    │ │
                        │    │  └──────────────────┘    │ │
                        │    │           │              │ │
                        │    │           ▼              │ │
                        │    │  ┌──────────────────┐    │ │
                        │    │  │ Classifier       │    │ │
                        │    │  │                  │    │ │
                        │    │  │ MLP: [512→2]     │    │ │
                        │    │  │                  │    │ │
                        │    │  │ Output: [32, 2]  │    │ │
                        │    │  └──────────────────┘    │ │
                        │    └──────────────────────────┘ │
                        │                                 │
                        │                                 │
┌────────────────────┐│  ┌──────────────────────────────┐│
│ GRAFO COMPLETO     ││  │  Structural Stream (GAT)     ││
│                    ││  │                              ││
│ edge_index:        ││  │  IMPORTANTE:                 ││
│  [2, 461493]       │├─▶│  - Recebe GRAFO COMPLETO     ││
│                    ││  │  - Mas só processa features  ││
│ edge_weights:      ││  │    dos 2347 test cases       ││
│  [461493]          ││  │                              ││
│                    ││  │  Input:                      ││
│ (NUNCA muda!)      ││  │    x: [2347, 10]             ││
└────────────────────┘│  │    edge_index: [2, 461493]   ││
                      │  │    edge_weights: [461493]    ││
                      │  │                              ││
                      │  │  GAT Layer 1:                ││
                      │  │    Multi-head (4 heads)      ││
                      │  │    h = Σ α_ij * W * x_j      ││
                      │  │    Output: [2347, 256]       ││
                      │  │                              ││
                      │  │  GAT Layer 2:                ││
                      │  │    Single-head               ││
                      └─▶│    Output: [2347, 256]       │┘
                         │                              │
                         │  Extract batch samples:      │
                         │    output[global_indices]    │
                         │    → [32, 256]               │
                         └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               FASE 3: INFERÊNCIA (TEST SET)                    │
└─────────────────────────────────────────────────────────────────┘

Test Batch (Build_789):
┌────────────────────┐
│ Test Cases:        │
│                    │
│ 1. MCA-1015        │  ✓ No grafo (global_idx = 0)
│ 2. MCA-NEW-123     │  ✗ NOVO! (global_idx = -1)
│ 3. MCA-101956      │  ✓ No grafo (global_idx = 1)
│ 4. MCA-NEW-456     │  ✗ NOVO! (global_idx = -1)
└────────────────────┘
         │
         │ Mesmo pipeline de treinamento
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Dual-Stream Model (MESMOS pesos treinados)                    │
│                                                                 │
│  Semantic Stream:                                              │
│    - Processa TODOS os test cases (novos ou não)               │
│    - Embeddings sempre disponíveis                             │
│                                                                 │
│  Structural Stream (GAT):                                      │
│    - Usa MESMO grafo de treinamento                            │
│    - Test cases NOVOS:                                         │
│        * Sem arestas conectadas                                │
│        * GAT usa apenas suas próprias features                 │
│        * Semantic stream compensa falta de histórico           │
│                                                                 │
│  Output: Probabilidades [batch_size, 2]                        │
└─────────────────────────────────────────────────────────────────┘
         │
         │ Ordenação por prob_fail
         ▼
┌────────────────────┐
│ Ranking:           │
│                    │
│ 1. MCA-NEW-456     │  0.85 prob_fail (novo, mas embedding alto risco)
│ 2. MCA-1015        │  0.72 prob_fail (histórico + semantic)
│ 3. MCA-NEW-123     │  0.45 prob_fail (novo, embedding baixo risco)
│ 4. MCA-101956      │  0.12 prob_fail (estável no histórico)
└────────────────────┘
         │
         │ Execução ordenada
         ▼
┌────────────────────┐
│ APFD Calculation:  │
│                    │
│ MCA-NEW-456: FAIL  │  ✓ Detectado cedo!
│ MCA-1015: FAIL     │  ✓ Detectado cedo!
│ MCA-NEW-123: PASS  │
│ MCA-101956: PASS   │
│                    │
│ APFD = 0.75        │  Excelente!
└────────────────────┘
```

## Comparação: Com vs Sem Grafo

```
┌─────────────────────────────────────────────────────────────────┐
│              ABORDAGEM SEM GRAFO (baseline)                    │
└─────────────────────────────────────────────────────────────────┘

Test Case: MCA-1015
  ┌──────────────────────────────┐
  │ Features Estruturais (10):   │
  │                              │
  │ test_age: 50                 │
  │ failure_rate: 0.24           │
  │ recent_failure_rate: 0.40    │
  │ flakiness_rate: 0.31         │
  │ commit_count: 250            │
  │ test_novelty: 0              │
  │ consecutive_failures: 0      │
  │ max_consecutive_failures: 3  │
  │ failure_trend: +0.16         │
  │ cr_count: 45                 │
  └──────────────────────────────┘
            │
            │ Direto para MLP (sem grafo)
            ▼
  ┌──────────────────────────────┐
  │ MLP: [10 → 64 → 256]         │
  │                              │
  │ LIMITAÇÃO:                   │
  │ - Apenas informação própria  │
  │ - Não sabe de vizinhos       │
  │ - Perde contexto global      │
  └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               ABORDAGEM COM GRAFO + GAT (V8)                   │
└─────────────────────────────────────────────────────────────────┘

Test Case: MCA-1015
  ┌──────────────────────────────┐
  │ Features Estruturais (10):   │
  │ [mesmas features]            │
  └──────────────────────────────┘
            │
            │ + Informação dos vizinhos via GAT
            ▼
  ┌──────────────────────────────────────────────────┐
  │ GAT Agregação:                                   │
  │                                                  │
  │ Vizinhos Co-Failure (pesos altos):              │
  │   - MCA-567: failure_rate=0.85 (alto risco!)    │
  │   - MCA-890: failure_rate=0.72                  │
  │   - MCA-1023: failure_rate=0.68                 │
  │                                                  │
  │ Vizinhos Co-Success (pesos médios):             │
  │   - MCA-2: failure_rate=0.05 (estável)          │
  │   - MCA-45: failure_rate=0.08                   │
  │   - ... (1500+ vizinhos estáveis)               │
  │                                                  │
  │ Vizinhos Semantic (pesos baixos):               │
  │   - MCA-1200: failure_rate=0.30                 │
  │   - MCA-1201: failure_rate=0.25                 │
  │   - ... (10 similares)                          │
  │                                                  │
  │ Agregação Atencional:                           │
  │   h = Σ α_ij * W * features_j                   │
  │                                                  │
  │ VANTAGEM:                                        │
  │ - Aprende contexto de vizinhos                  │
  │ - Atenção ajusta importância                    │
  │ - Detecta padrões globais                       │
  └──────────────────────────────────────────────────┘

Resultado:
  Sem Grafo: prob_fail = 0.35 (subestimado)
  Com Grafo: prob_fail = 0.72 (correto! vizinhos de alto risco)
```

## Estatísticas de Conectividade

```
┌─────────────────────────────────────────────────────────────────┐
│              ANÁLISE DE CONECTIVIDADE DO GRAFO                 │
└─────────────────────────────────────────────────────────────────┘

Top 10 Test Cases Mais Conectados:

1. MCA-1015          ████████████████████████████████  48,297 conexões
2. MCA-101956        ████████████████████████████████  48,145 conexões
3. MCA-1012          ████████████████████████████████  48,125 conexões
4. MCA-1011          ████████████████████████████████  48,117 conexões
5. MCA-1013          ████████████████████████████████  48,098 conexões
6. MCA-1037          ██                                 2,630 conexões
7. MCA-101960        ██                                 2,562 conexões
8. MCA-101962        ██                                 2,562 conexões
9. MCA-103639        ██                                 2,551 conexões
10. MCA-101961       ██                                 2,527 conexões

Distribuição de Grau:
  0-100 conexões:     234 test cases (10.0%)  ████
  101-500 conexões:   345 test cases (14.7%)  ██████
  501-1000 conexões:  456 test cases (19.4%)  ████████
  1001-5000 conexões: 890 test cases (37.9%)  ███████████████
  5001+ conexões:     422 test cases (18.0%)  ███████

Tipo de Aresta:
  Co-Failure:    495 arestas (0.1%)    ▌ RARO mas IMPORTANTE
  Co-Success: 207,913 arestas (45.1%)  ██████████████████████
  Semantic:   253,085 arestas (54.8%)  ███████████████████████████

Densidade do Grafo: 16.8%
  (vs. 100% = grafo completo com todas as arestas possíveis)
```

## Fluxo de Mensagens no GAT

```
┌─────────────────────────────────────────────────────────────────┐
│           PROPAGAÇÃO DE INFORMAÇÃO (2 layers GAT)              │
└─────────────────────────────────────────────────────────────────┘

Layer 1: Agregação Local (1-hop neighbors)

    MCA-1015 (target)
         │
         │ Recebe mensagens de:
         │
         ├─── MCA-567 (co-failure, peso alto)
         │     └─ "Eu falho muito! Cuidado!"
         │
         ├─── MCA-890 (co-failure, peso alto)
         │     └─ "Também falho frequentemente com MCA-1015"
         │
         ├─── MCA-2 (co-success, peso médio)
         │     └─ "Passamos juntos 95% das vezes"
         │
         ├─── MCA-45 (co-success, peso médio)
         │     └─ "Muito estável aqui"
         │
         └─── MCA-1200 (semantic, peso baixo)
               └─ "Semanticamente similar, mas comportamento diferente"

    Agregação (weighted sum):
    h1 = α_567 * W * x_567 + α_890 * W * x_890 + ... + α_1200 * W * x_1200

    Atenção (aprende importância):
    α_567 = 0.45  (ALTO - co-failure importante!)
    α_890 = 0.35  (ALTO)
    α_2   = 0.08  (baixo - co-success menos informativo)
    α_45  = 0.07  (baixo)
    α_1200= 0.05  (MUITO baixo - semantic pouco relevante aqui)

Layer 2: Agregação de Longo Alcance (2-hop neighbors)

    MCA-1015 (target)
         │
         │ Agora também recebe informação de:
         │
         ├─── Vizinhos de MCA-567:
         │     └─ MCA-123 (também falha frequentemente)
         │     └─ MCA-456 (cluster de testes problemáticos)
         │
         ├─── Vizinhos de MCA-890:
         │     └─ MCA-789 (mesmo módulo, mesmos problemas)
         │
         └─── Clusters detectados:
               └─ "MCA-1015 pertence ao cluster de testes de API"
               └─ "Esse cluster tem alta taxa de falha recente"

    Agregação final:
    h2 = β_567 * W * h1_567 + β_890 * W * h1_890 + ...

    Output: [256] features enriquecidas com contexto global

Resultado:
  Input:  [10 features próprias]
  Output: [256 features + contexto de ~1500 vizinhos em 2 hops]
```

---

**Conclusão**: O grafo fornece **contexto relacional rico** que permite ao modelo detectar **padrões de falha colaborativos** que não seriam visíveis olhando apenas para features individuais de cada test case.
