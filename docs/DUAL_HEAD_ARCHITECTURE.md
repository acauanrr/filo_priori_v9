# Filo-Priori V9: Arquitetura Dual-Head (DeepOrder-inspired)

## Sumário

1. [Visão Geral](#visão-geral)
2. [Evolução da Arquitetura](#evolução-da-arquitetura)
3. [FASE 1: Priority Score como Feature Auxiliar](#fase-1-priority-score-como-feature-auxiliar)
4. [FASE 2: Modelo Dual-Head](#fase-2-modelo-dual-head)
5. [Pipeline de Treinamento](#pipeline-de-treinamento)
6. [Pipeline de Ranking (APFD)](#pipeline-de-ranking-apfd)
7. [Comparação: Antes vs Depois](#comparação-antes-vs-depois)
8. [Arquivos e Configuração](#arquivos-e-configuração)

---

## Visão Geral

O Filo-Priori V9 implementa uma arquitetura **Dual-Head** inspirada no paper DeepOrder (ICSME 2021), que combina:

- **Classificação**: Predição binária Fail/Pass (usando Focal Loss)
- **Regressão**: Predição de Priority Score contínuo (usando MSE Loss)

A insight fundamental do DeepOrder é que **priorização de testes é fundamentalmente um problema de REGRESSÃO**, não apenas classificação. O modelo deve aprender a ordenar testes, não apenas classificá-los.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DUAL-HEAD MODEL                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Semantic Stream (SBERT)──┐                                            │
│   [batch, 1536]            │                                            │
│                            ├──→ Cross-Attention ──→ Fused Features      │
│   Structural Stream (GAT)──┘    Fusion              [batch, 512]        │
│   [batch, 19]                                            │              │
│                                                          │              │
│                              ┌────────────────────────────┤              │
│                              │                            │              │
│                              ▼                            ▼              │
│                    ┌─────────────────┐         ┌─────────────────┐      │
│                    │ Classification  │         │   Regression    │      │
│                    │     Head        │         │      Head       │      │
│                    │                 │         │                 │      │
│                    │ MLP → Softmax   │         │ MLP → Sigmoid   │      │
│                    │ [batch, 2]      │         │ [batch, 1]      │      │
│                    │                 │         │                 │      │
│                    │   P(Fail|x)     │         │ priority_score  │      │
│                    │   P(Pass|x)     │         │    ∈ [0, 1]     │      │
│                    └─────────────────┘         └─────────────────┘      │
│                              │                            │              │
│                              ▼                            ▼              │
│                    ┌─────────────────┐         ┌─────────────────┐      │
│                    │   Focal Loss    │         │    MSE Loss     │      │
│                    │  (α_focal=0.75) │         │                 │      │
│                    └─────────────────┘         └─────────────────┘      │
│                              │                            │              │
│                              └──────────┬─────────────────┘              │
│                                         │                                │
│                                         ▼                                │
│                              ┌─────────────────┐                        │
│                              │  Combined Loss  │                        │
│                              │                 │                        │
│                              │ L = α×Focal +   │                        │
│                              │     β×MSE       │                        │
│                              │ (α=1.0, β=0.5)  │                        │
│                              └─────────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Evolução da Arquitetura

### Arquitetura Original (V7/V8) - Apenas Classificação

```
┌──────────────┐     ┌──────────────┐
│   Semantic   │     │  Structural  │
│   Stream     │     │   Stream     │
│   (SBERT)    │     │   (GAT)      │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
         ┌──────▼──────┐
         │   Fusion    │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ Classifier  │────→ P(Fail), P(Pass)
         │  (Softmax)  │
         └─────────────┘
                │
         ┌──────▼──────┐
         │ Focal Loss  │
         └─────────────┘
```

**Problema**: O modelo só aprendia a separar Fail/Pass, não a ordenar por prioridade.

### Arquitetura V9 - Dual-Head (Classificação + Regressão)

O modelo agora tem **duas cabeças de saída**:

1. **Classification Head**: Mantém a tarefa de Fail/Pass
2. **Regression Head**: Nova tarefa de prever Priority Score

---

## FASE 1: Priority Score como Feature Auxiliar

### O que é Priority Score?

O Priority Score é uma métrica contínua em `(0, 1)` que representa quão importante é executar um teste cedo. É calculado baseado no histórico de execuções do teste:

```
p(ti) = Σ(j=1..m) wj × max(ES(i,j), 0)
```

Onde:
- `ES(i,j)` = Status de execução no ciclo j: `{1 = failed, 0 = passed, -1 = not executed}`
- `wj` = Peso do ciclo j (ciclos recentes têm peso maior)
- `Σwj = 1` (pesos normalizados)

### Implementação (priority_score_generator.py)

```python
# Exemplo: TC com histórico [Pass, Pass, Fail, Fail, Fail] nos últimos 5 ciclos
# ES = [0, 0, 1, 1, 1]
# Weights (exponential decay 0.8): [0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11, 0.14, 0.18, 0.22]

# Priority = 0.05×1 + 0.06×1 + 0.07×1 = 0.18 (considerando só os últimos ciclos relevantes)
# Quanto mais falhas recentes, maior o priority score!
```

### DeepOrder Features (9 features adicionais)

Além do priority score, extraímos 9 features do histórico:

| Feature | Descrição |
|---------|-----------|
| `priority_score` | Score calculado pela fórmula DeepOrder |
| `fail_rate_all` | Taxa de falha histórica total |
| `fail_rate_recent` | Taxa de falha nos últimos N ciclos |
| `avg_fail_position` | Posição média das falhas no histórico |
| `consecutive_failures` | Número de falhas consecutivas recentes |
| `cycles_since_last_fail` | Ciclos desde a última falha |
| `execution_count` | Total de execuções do TC |
| `recent_executions` | Execuções recentes |
| `history_length` | Tamanho do histórico disponível |

**Resultado FASE 1**: Structural features aumentaram de 10 para 19 (10 + 9 DeepOrder)

---

## FASE 2: Modelo Dual-Head

### Por que Dual-Head?

A classificação binária tem limitações para ranking:

| Cenário | Classificação | Problema |
|---------|---------------|----------|
| TC_A: 90% prob Fail | Fail | ✓ Correto |
| TC_B: 89% prob Fail | Fail | ✓ Correto |
| **Ordenação?** | **Empate!** | ❌ Ambos são "Fail", mas qual vem primeiro? |

Com **Regressão**, temos scores contínuos:

| TC | Priority Score | Rank |
|----|----------------|------|
| TC_A | 0.95 | 1º |
| TC_B | 0.72 | 2º |
| TC_C | 0.45 | 3º |

### Implementação (dual_head_model.py)

```python
class DualHeadModel(nn.Module):
    def __init__(self, ...):
        # Streams (idênticos ao V8)
        self.semantic_stream = SemanticStream(...)
        self.structural_stream = StructuralStreamV8(...)  # GAT
        self.fusion = CrossAttentionFusion(...)

        # DUAS CABEÇAS DE SAÍDA
        self.classifier = ClassificationHead(...)  # → [batch, 2]
        self.regressor = RegressionHead(...)       # → [batch, 1]

    def forward(self, semantic_input, structural_input, edge_index, edge_weights):
        # Processar streams
        semantic_features = self.semantic_stream(semantic_input)
        structural_features = self.structural_stream(structural_input, edge_index, edge_weights)

        # Fusão
        fused = self.fusion(semantic_features, structural_features)

        # DUAL OUTPUT
        logits = self.classifier(fused)           # Para classificação
        priority_scores = self.regressor(fused)   # Para ranking

        return logits, priority_scores
```

### DualHeadLoss

A loss combinada otimiza ambas as tarefas simultaneamente:

```python
L_total = α × L_focal + β × L_mse
```

Onde:
- `α = 1.0` (peso da classificação)
- `β = 0.5` (peso da regressão)
- `L_focal = FocalLoss(logits, labels)` com `focal_alpha=0.75`, `gamma=2.5`
- `L_mse = MSE(pred_priority, true_priority)`

```python
class DualHeadLoss(nn.Module):
    def forward(self, logits, priority_pred, labels, priority_true):
        # Classificação (Focal Loss - handles class imbalance)
        loss_focal = self.focal_loss(logits, labels)

        # Regressão (MSE)
        loss_mse = F.mse_loss(priority_pred.squeeze(), priority_true)

        # Combined
        total_loss = self.alpha * loss_focal + self.beta * loss_mse

        return total_loss, {'focal': loss_focal, 'mse': loss_mse}
```

---

## Pipeline de Treinamento

### Fluxo Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1.1 Load datasets (train.csv, test.csv)                                │
│       ↓                                                                 │
│  1.1.1 Compute class weights [19.13, 0.51] for Fail/Pass                │
│       ↓                                                                 │
│  1.1.2 Compute Priority Scores (DeepOrder-style)  ← NOVO EM V9!         │
│       - Para cada TC, calcular p(ti) baseado no histórico               │
│       - Extrair 9 features DeepOrder adicionais                         │
│       ↓                                                                 │
│  1.2 Extract commits                                                    │
│       ↓                                                                 │
│  1.3 Generate embeddings (SBERT all-mpnet-base-v2)                      │
│       - TC embeddings [768]                                             │
│       - Commit embeddings [768]                                         │
│       - Combined [1536]                                                 │
│       ↓                                                                 │
│  1.4 Extract structural features V2.5 [10 features]                     │
│       ↓                                                                 │
│  1.5 Concatenate: structural [10] + DeepOrder [9] = [19]               │
│       ↓                                                                 │
│  1.6 Build phylogenetic graph (co-failure, co-success, semantic)        │
│       ↓                                                                 │
│  1.7 Create dataloaders WITH priority_score (for dual-head)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: MODEL INITIALIZATION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  model = create_model(config)  → DualHeadModel                          │
│  criterion = create_dual_head_loss(config) → DualHeadLoss               │
│  optimizer = AdamW(lr=3e-5, weight_decay=1e-4)                          │
│  scheduler = CosineAnnealingLR                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEP 3: TRAINING                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for epoch in range(50):                                                │
│      for batch in train_loader:                                         │
│          embeddings, structural, labels, indices, priority_scores = batch│
│                                                                         │
│          # Forward pass (DUAL OUTPUT)                                   │
│          logits, pred_priority = model(embeddings, structural, ...)     │
│                                                                         │
│          # Combined loss                                                │
│          loss, loss_dict = criterion(                                   │
│              logits,           # [batch, 2]                             │
│              pred_priority,    # [batch, 1]                             │
│              labels,           # [batch]                                │
│              priority_scores   # [batch] - TARGET para regressão!       │
│          )                                                              │
│                                                                         │
│          loss.backward()                                                │
│          optimizer.step()                                               │
│                                                                         │
│      # Validation                                                       │
│      val_loss, val_metrics = evaluate(...)                              │
│                                                                         │
│      # Early stopping on val_f1_macro                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Batch Structure (com Dual-Head)

```python
# Cada batch contém 5 elementos:
batch = (
    embeddings,         # [batch_size, 1536] - Combined TC+Commit
    structural_features,# [batch_size, 19]   - V2.5 + DeepOrder
    labels,             # [batch_size]       - 0=Fail, 1=Pass
    global_indices,     # [batch_size]       - Índice no grafo
    priority_scores     # [batch_size]       - TARGET para regressão! ← NOVO
)
```

---

## Pipeline de Ranking (APFD)

### Como o Ranking é Feito ATUALMENTE

O ranking **ainda usa probabilidades de classificação**, não os priority scores preditos:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: TEST EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  logits, priority_pred = model(test_embeddings, test_structural, ...)   │
│  probs = softmax(logits)                                                │
│                                                                         │
│  # Probabilidade de FAIL (classe 0)                                     │
│  test_df['probability'] = probs[:, 0]  ← USADO PARA RANKING             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: APFD CALCULATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Para cada Build:                                                       │
│      1. Ordenar TCs por probability (descendente)                       │
│      2. Atribuir ranks (1 = maior prioridade)                           │
│      3. Calcular APFD:                                                  │
│                                                                         │
│         APFD = 1 - (Σ ranks_of_failures) / (n_failures × n_tests)       │
│                  + 1 / (2 × n_tests)                                    │
│                                                                         │
│  Exemplo:                                                               │
│    10 testes, 2 falhas nas posições 2 e 5:                              │
│    APFD = 1 - (2+5)/(2×10) + 1/(2×10) = 1 - 0.35 + 0.05 = 0.70          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Fluxo Detalhado de Ranking

```python
# 1. Após avaliação, adicionar probabilidades ao DataFrame
test_df['probability'] = test_probs[:, 0]  # P(Fail)

# 2. Calcular ranks POR BUILD (não globalmente!)
def calculate_ranks_per_build(df):
    df['rank'] = df.groupby('Build_ID')['probability'].rank(
        method='first',
        ascending=False  # Maior prob = rank 1 (maior prioridade)
    )
    return df

# 3. Calcular APFD por build
for build_id, build_df in df.groupby('Build_ID'):
    ranks = build_df['rank'].values
    labels = (build_df['TE_Test_Result'] == 'Fail').astype(int).values

    apfd = calculate_apfd_single_build(ranks, labels)
```

---

## Comparação: Antes vs Depois

### Estrutura de Features

| Versão | Structural Features | Descrição |
|--------|---------------------|-----------|
| V8 | 10 | Features V2.5 básicas |
| V9 | **19** | V2.5 (10) + DeepOrder (9) |

### Modelo

| Aspecto | V8 (Apenas Classificação) | V9 (Dual-Head) |
|---------|---------------------------|----------------|
| **Output** | `logits [batch, 2]` | `logits [batch, 2]` + `priority [batch, 1]` |
| **Loss** | Focal Loss | `α×Focal + β×MSE` |
| **Target** | `labels` (0/1) | `labels` + `priority_score` (contínuo) |
| **Ranking** | P(Fail) | P(Fail) (ainda) |

### Training Loop

```python
# V8 (classificação apenas)
logits = model(embeddings, structural, edge_index, edge_weights)
loss = focal_loss(logits, labels)

# V9 (dual-head)
logits, priority_pred = model(embeddings, structural, edge_index, edge_weights)
loss, loss_dict = dual_head_loss(logits, priority_pred, labels, priority_true)
# loss_dict = {'focal': ..., 'mse': ..., 'total': ...}
```

### O que Mudou no Ranking?

**FASE 4 (Implementado - ATUALIZADO)**: Estratégia híbrida com **P(Fail) como sinal primário**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│             FASE 4: HYBRID RANKING STRATEGY (ATUALIZADO)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INSIGHT: P(Fail) é mais confiável que priority_pred isolado            │
│           (regression head pode ter baixa variância nas predições)      │
│                                                                         │
│  Para cada Test Case na inferência:                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ SE (TC está no grafo de treinamento):                           │   │
│  │                                                                  │   │
│  │   SE (priority_pred tem variância > 0.01):                      │   │
│  │       normalized_priority = normalizar priority_pred para [0,1] │   │
│  │       hybrid_score = λ × P(Fail) + (1-λ) × normalized_priority │   │
│  │       onde λ = 0.7 (P(Fail) é sinal primário)                  │   │
│  │   SENÃO:                                                        │   │
│  │       hybrid_score = P(Fail)  ← Fallback quando regressão falha│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ SENÃO (TC é órfão - não visto durante treinamento):             │   │
│  │                                                                  │   │
│  │   1. Encontrar K vizinhos mais próximos (K=10) no grafo         │   │
│  │      usando similaridade de cosseno dos embeddings semânticos   │   │
│  │                                                                  │   │
│  │   2. Calcular média ponderada dos hybrid_scores dos vizinhos:   │   │
│  │      knn_score = Σ (sim_i × hybrid_score_i) / Σ sim_i           │   │
│  │                                                                  │   │
│  │   3. Blend com P(Fail) do próprio órfão:                        │   │
│  │      hybrid_score = α × knn_score + (1-α) × P(Fail)            │   │
│  │      onde α = 0.5 (balanceado)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Ranking final: ordenar por hybrid_score (descendente)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Weighted MSE Loss (Correção Crítica)

O problema original: **priority_score = 0 para ~92% dos TCs** (TCs que nunca falharam).
MSE padrão incentivava o modelo a prever 0 para tudo.

**Solução**: Weighted MSE que dá peso 20x maior para amostras com priority_score > 0:

```python
def weighted_mse_loss(self, pred, target):
    mse_per_sample = (pred - target) ** 2

    # Samples com priority_score > 0 recebem peso 20x maior
    weights = torch.where(
        target > 0,
        torch.tensor(20.0),  # mse_nonzero_weight
        torch.tensor(1.0)
    )

    return (weights * mse_per_sample).sum() / weights.sum()
```

**Código da Estratégia Híbrida:**

```python
# P(Fail) como sinal primário
lambda_pfail = 0.7
hybrid_score = np.copy(failure_probs_full)  # Inicializa com P(Fail)

# Para in-graph: blend P(Fail) com priority_pred normalizado
if priority_std > 0.01:  # Só usa priority_pred se tiver variância
    normalized = (priority_pred - min) / (max - min)
    hybrid_score[in_graph] = lambda_pfail * P_fail + (1-lambda_pfail) * normalized

# Para órfãos: KNN + P(Fail)
knn_score = weighted_avg(neighbors_hybrid_scores)
hybrid_score[orphan] = 0.5 * knn_score + 0.5 * orphan_P_fail
```

**Vantagens da Nova Abordagem:**

| Aspecto | Antes (priority_pred direto) | Agora (P(Fail) primário) |
|---------|------------------------------|--------------------------|
| Robustez | Falha se regressão não aprende | P(Fail) garante ranking base |
| Órfãos | KNN de predições ruins | KNN de hybrid_scores bons |
| Fallback | Nenhum | P(Fail) se priority_pred sem variância |
| MSE Loss | Padrão (prediz 0) | Weighted (20x para non-zero) |

---

## Arquivos e Configuração

### Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| `src/models/dual_head_model.py` | DualHeadModel, DualHeadLoss, RegressionHead |
| `src/models/model_factory.py` | Factory que cria modelo baseado em `type: dual_head` |
| `src/preprocessing/priority_score_generator.py` | Gerador de priority scores (DeepOrder) |
| `configs/experiment_industry_dual_head.yaml` | Config para experimento dual-head |
| `main.py` | Pipeline principal (atualizado para dual-head) |

### Configuração YAML

```yaml
# configs/experiment_industry_dual_head.yaml

model:
  type: "dual_head"  # Ativa DualHeadModel
  num_classes: 2

  semantic:
    input_dim: 1536
    hidden_dim: 256

  structural:
    input_dim: 19  # 10 V2.5 + 9 DeepOrder
    hidden_dim: 64

  # Configuração específica para regression head
  regressor:
    hidden_dims: [128, 64]
    dropout: 0.3

training:
  loss:
    type: "dual_head"  # Ativa DualHeadLoss
    focal_alpha: 0.75
    focal_gamma: 2.5
    dual_head:
      alpha: 1.0   # Peso classificação
      beta: 0.5    # Peso regressão

priority_score:
  enabled: true
  num_cycles: 10
  decay_type: 'exponential'
  decay_factor: 0.8
```

---

## Resumo

| Pergunta | Resposta |
|----------|----------|
| **Temos classificação?** | SIM - Classification Head com Focal Loss |
| **Temos regressão?** | SIM - Regression Head com MSE Loss |
| **Como funciona a loss?** | `L = α×Focal + β×MSE` (α=1.0, β=0.5) |
| **O que é priority score?** | Score contínuo [0,1] baseado no histórico de falhas |
| **Quem calcula priority score?** | `PriorityScoreGenerator` (antes do treinamento) |
| **Ranking usa o quê?** | Ainda usa P(Fail) da classificação |
| **Ranking é feita quando?** | Após treinamento (STEP 5: APFD Calculation) |
| **APFD é por build?** | SIM - calculado separadamente para cada Build_ID |

---

## Executando o Experimento

```bash
# Ativar ambiente
source venv/bin/activate

# Rodar experimento dual-head
python main.py --config configs/experiment_industry_dual_head.yaml
```

**Métricas esperadas (após correção da DualHeadLoss):**
- Val F1 Macro: > 0.50
- Val Accuracy: > 0.90
- APFD: Melhoria em relação ao baseline (~0.65)

---

*Documento criado em: Dezembro 2024*
*Versão: Filo-Priori V9 (Dual-Head Architecture)*
