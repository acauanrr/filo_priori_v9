# Relatório de Análise do Pipeline Filo-Priori V9

**Data:** 2025-12-01
**Objetivo:** Análise detalhada das etapas de classificação e ordenação para identificar oportunidades de melhoria do APFD

---

## 1. Visão Geral do Pipeline

O pipeline Filo-Priori V9 é um sistema de priorização de casos de teste que combina:
- **Embeddings Semânticos** (SBERT all-mpnet-base-v2)
- **Features Estruturais** (histórico de execução)
- **Grafo Filogenético** (co-falha, co-sucesso, similaridade semântica)
- **Arquitetura Dual-Stream com GAT** (Graph Attention Network)

### 1.1. Fluxo Principal (`main.py`)

```
STEP 1: DATA PREPARATION
├── 1.1: Load datasets (train.csv, test.csv)
├── 1.1.1: Compute class weights
├── 1.2: Extract semantic embeddings (SBERT)
├── 1.4: Extract structural features (V2.5 - 10 features)
├── 1.4b: Cold-Start imputation (MLP para TCs sem histórico)
├── 1.6: Build phylogenetic graph (multi-edge)
├── 1.7: Extract edge_index and edge_weights
└── 1.8: Create TC_Key to global index mapping

STEP 2: MODEL INITIALIZATION
├── Create DualStreamModelV8
├── Initialize WeightedFocalLoss
└── Initialize AdamW optimizer

STEP 3: TRAINING
├── Train epoch (classification loss only)
├── Validate
├── Early stopping
└── Save best model

STEP 4: TEST EVALUATION
├── Threshold optimization (opcional)
└── Evaluate on test set

STEP 5: APFD CALCULATION (test split)
├── Generate probabilities
├── Calculate ranks per build
└── Calculate APFD per build

STEP 6: FULL TEST.CSV APFD (277 builds)
├── Load full test.csv
├── Generate embeddings
├── Generate predictions
├── Calculate ranks
└── Calculate final APFD
```

---

## 2. Etapa de Classificação

### 2.1. Arquitetura do Modelo (`src/models/dual_stream_v8.py`)

```
DualStreamModelV8
├── SemanticStream
│   ├── Input: [batch, 1536] (TC + Commit embeddings)
│   ├── Linear projection: 1536 → 256
│   ├── 2x FFN layers with residual connections
│   └── Output: [batch, 256]
│
├── StructuralStreamV8 (GAT-based)
│   ├── Input: [batch, 10] (structural features)
│   ├── GATConv Layer 1: 4 heads, output [batch, 256]
│   ├── GATConv Layer 2: 1 head, output [batch, 256]
│   └── Output: [batch, 256]
│
├── CrossAttentionFusion
│   ├── Bidirectional cross-attention
│   ├── Semantic → Structural attention
│   ├── Structural → Semantic attention
│   └── Output: [batch, 512] (concatenated)
│
└── SimpleClassifier
    ├── Linear: 512 → 128 → 64 → 2
    └── Output: [batch, 2] (logits)
```

### 2.2. Função de Perda (`src/training/losses.py`)

**WeightedFocalLoss** (configuração atual):
```python
# Parâmetros:
alpha: 0.75           # Peso para classe minoritária
gamma: 2.5            # Foco em exemplos difíceis
class_weights: [19.13, 0.51]  # Calculados automaticamente
```

**Fórmula:**
```
Loss = α × (1 - p_t)^γ × CE(p, y)
```

Onde:
- `p_t` = probabilidade da classe correta
- `CE` = Cross-Entropy com class_weights

### 2.3. Processamento durante Avaliação (`main.py:799-940`)

```python
# 1. Forward pass
logits = model(semantic_input, structural_input, edge_index, edge_weights)

# 2. Probabilidades via Softmax
probs = torch.softmax(logits, dim=1)  # [batch, 2]
# probs[:, 0] = P(Fail)
# probs[:, 1] = P(Pass)

# 3. Tratamento de órfãos (TCs não no grafo)
full_probs = np.full((dataset_size, 2), 0.5)  # Default: máxima incerteza
full_probs[valid_indices] = actual_probs
```

### 2.4. Problemas Identificados na Classificação

#### PROBLEMA CRÍTICO #1: Ranking Loss Não Está Sendo Usado!

**Configuração define ranking loss:**
```yaml
training:
  ranking:
    enabled: true
    weight: 0.3
    loss_type: "logistic"
    # ...
```

**MAS o código `train_epoch()` usa apenas classification loss:**
```python
# main.py:784-785
loss = criterion(logits, labels_valid)  # Apenas WeightedFocalLoss!
# NÃO há combinação com ranking loss
```

**Impacto:** O modelo é treinado para **classificar** (prever Pass/Fail), não para **ordenar** (rankear falhas antes de passes).

#### PROBLEMA #2: Class Imbalance Extremo

```
Class distribution:
  Pass:      61224 (97.4%)
  Fail:       1654 (2.6%)

Ratio: 37:1
```

O WeightedFocalLoss tenta compensar, mas o modelo ainda pode colapsar para a classe majoritária.

#### PROBLEMA #3: Tratamento de Órfãos

```python
# ~22.7% dos samples de teste são órfãos
# Recebem probabilidade padrão [0.5, 0.5]
# Isso afeta negativamente o APFD
```

---

## 3. Etapa de Ordenação (Ranking)

### 3.1. Cálculo de Ranks (`src/evaluation/apfd.py:311-340`)

```python
def calculate_ranks_per_build(df, probability_col, build_col):
    """
    Ranks são calculados POR BUILD:
    - Higher probability = Lower rank (rank 1 = mais prioritário)
    - method='first' para desempate determinístico
    """
    df['rank'] = df.groupby(build_col)[probability_col] \
                   .rank(method='first', ascending=False) \
                   .astype(int)
```

### 3.2. Qual Probabilidade é Usada para Ranking?

```python
# main.py:1353
test_df['probability'] = test_probs[:, 0]  # P(Fail) - classe 0
```

**Lógica:**
- P(Fail) alto → rank baixo → executado primeiro
- P(Fail) baixo → rank alto → executado depois

### 3.3. Cálculo do APFD (`src/evaluation/apfd.py:79-122`)

```python
def calculate_apfd_single_build(ranks, labels):
    """
    APFD = 1 - (soma dos ranks das falhas) / (n_falhas × n_testes) + 1 / (2 × n_testes)

    Exemplo:
    - 10 testes, 2 falhas nos ranks [2, 5]
    - APFD = 1 - (2+5)/(2×10) + 1/(2×10)
    - APFD = 1 - 0.35 + 0.05 = 0.70
    """
```

**Regras de Negócio:**
1. Apenas builds com pelo menos 1 falha são considerados
2. Builds com apenas 1 TC têm APFD = 1.0
3. Esperado: 277 builds no dataset industry

### 3.4. Problemas Identificados na Ordenação

#### PROBLEMA #4: Classificação ≠ Ranking

O modelo é otimizado para **classificar corretamente**, não para **ordenar corretamente**.

```
Cenário problemático:
- TC1: P(Fail) = 0.6, Label = Fail
- TC2: P(Fail) = 0.55, Label = Fail
- TC3: P(Fail) = 0.4, Label = Pass

Classificação: Todos acima de threshold 0.5 são "Fail" ✓
Ranking: TC1 (rank 1), TC2 (rank 2), TC3 (rank 3) ✓

MAS se:
- TC1: P(Fail) = 0.45, Label = Fail  ← Erro de classificação
- TC2: P(Fail) = 0.55, Label = Pass  ← Erro de classificação
- TC3: P(Fail) = 0.35, Label = Fail

Ranking: TC2 (rank 1), TC1 (rank 2), TC3 (rank 3)
- TC2 (Pass) executado primeiro! ← Prejudica APFD
```

#### PROBLEMA #5: Probabilidades Não Calibradas

As probabilidades do softmax podem não refletir a "confiança real":
- Modelo pode produzir P(Fail) entre 0.48-0.52 para muitos casos
- Pequenas diferenças de probabilidade determinam ranks
- Não há calibração de probabilidades (e.g., Platt scaling)

#### PROBLEMA #6: Órfãos com P=0.5 Afetam Ranking

```
22.7% dos samples têm P(Fail) = 0.5
- Se um órfão é realmente Fail, ele fica no meio do ranking
- Isso prejudica o APFD
```

---

## 4. Análise de Gargalos e Oportunidades

### 4.1. Gargalo Principal: Falta de Ranking Loss

**Status Atual:** Modelo treinado apenas com classification loss
**Impacto:** Não otimiza diretamente para ordenação

**Solução Proposta:**
```python
# Combinar classification loss + ranking loss
total_loss = classification_loss + 0.3 * ranking_loss
```

Opções de Ranking Loss disponíveis em `src/training/ranking_losses.py`:
- **ListNet**: Listwise, cross-entropy em distribuições
- **ListMLE**: Maximum likelihood para permutações
- **LambdaRank**: Pairwise com gradientes NDCG-aware

### 4.2. Gargalo: Tratamento de Órfãos

**Status Atual:** Órfãos recebem P = 0.5 (neutro)
**Impacto:** 22.7% dos samples afetam negativamente o APFD

**Soluções Propostas:**
1. **Cold-Start melhorado**: Usar embedding similarity para transferir probabilidades
2. **Fallback conservador**: P(Fail) = 0.7 para órfãos (assumir risco)
3. **Ensemble**: Combinar predição do modelo com heurísticas

### 4.3. Gargalo: Calibração de Probabilidades

**Status Atual:** Probabilidades raw do softmax
**Impacto:** Podem não refletir confiança real

**Soluções Propostas:**
1. **Temperature Scaling**: Calibrar probabilidades no validation set
2. **Platt Scaling**: Regressão logística nas probabilidades
3. **Isotonic Regression**: Calibração não-paramétrica

### 4.4. Oportunidade: Features de Ranking Diretas

Adicionar features que capturam diretamente a "urgência" de um TC:
- Tempo desde última falha
- Frequência de falhas recentes
- Severidade histórica das falhas

---

## 5. Fluxo Detalhado de Dados

### 5.1. Da Predição ao APFD

```
1. PREDIÇÃO
   model(semantic, structural, graph) → logits [N, 2]

2. PROBABILIDADES
   softmax(logits) → probs [N, 2]
   P(Fail) = probs[:, 0]

3. TRATAMENTO DE ÓRFÃOS
   Para cada TC não no grafo: P(Fail) = 0.5

4. RANKING POR BUILD
   Para cada build:
     ranks = P(Fail).rank(ascending=False)
     # Maior P(Fail) → rank 1

5. APFD POR BUILD
   Para cada build com falhas:
     APFD = 1 - sum(fail_ranks)/(n_fails × n_tests) + 1/(2 × n_tests)

6. APFD FINAL
   Mean APFD across all 277 builds
```

### 5.2. Exemplo Numérico

```
Build X com 5 TCs:
TC    P(Fail)  Label   Rank
A     0.82     Fail    1
B     0.65     Pass    2
C     0.51     Fail    3
D     0.45     Pass    4
E     0.30     Pass    5

Falhas: A (rank 1), C (rank 3)
n_fails = 2, n_tests = 5

APFD = 1 - (1+3)/(2×5) + 1/(2×5)
     = 1 - 0.4 + 0.1
     = 0.70
```

---

## 6. Recomendações de Melhoria

### 6.1. Alta Prioridade (Impacto Direto no APFD)

| # | Melhoria | Complexidade | Impacto Esperado |
|---|----------|--------------|------------------|
| 1 | **Ativar Ranking Loss** | Média | +5-10% APFD |
| 2 | **Melhorar tratamento de órfãos** | Baixa | +2-5% APFD |
| 3 | **Calibração de probabilidades** | Baixa | +1-3% APFD |

### 6.2. Média Prioridade

| # | Melhoria | Complexidade | Impacto Esperado |
|---|----------|--------------|------------------|
| 4 | Threshold otimizado para ranking | Baixa | +1-2% APFD |
| 5 | Ensemble com heurísticas | Média | +2-4% APFD |
| 6 | Features de urgência temporal | Média | +2-3% APFD |

### 6.3. Baixa Prioridade (Experimental)

| # | Melhoria | Complexidade | Impacto Esperado |
|---|----------|--------------|------------------|
| 7 | Learning-to-Rank end-to-end | Alta | Incerto |
| 8 | Graph Transformer | Alta | Incerto |
| 9 | Multi-task learning | Alta | +0-5% APFD |

---

## 7. Código para Implementar Ranking Loss

### 7.1. Modificação em `train_epoch()`

```python
# Adicionar ranking loss combinada
from src.training.ranking_losses import ListNetLoss

ranking_criterion = ListNetLoss(temperature=1.0)

def train_epoch(..., ranking_config=None):
    for batch in loader:
        # Classification loss (existente)
        classification_loss = criterion(logits, labels)

        # Ranking loss (NOVO)
        if ranking_config and ranking_config.get('enabled', False):
            # scores = logits[:, 0] - logits[:, 1]  # Score diferencial
            scores = logits[:, 0]  # P(Fail) score
            relevance = labels.float()  # 1 para Fail, 0 para Pass
            ranking_loss = ranking_criterion(scores, relevance)

            # Combinar losses
            weight = ranking_config.get('weight', 0.3)
            total_loss = classification_loss + weight * ranking_loss
        else:
            total_loss = classification_loss

        # Backward
        total_loss.backward()
```

---

## 8. Conclusão

O pipeline atual tem **arquitetura sólida** mas **não otimiza diretamente para ranking**. Os principais gargalos são:

1. **Ranking Loss não utilizado** (mesmo estando configurado)
2. **Órfãos tratados com probabilidade neutra** (22.7% dos samples)
3. **Probabilidades não calibradas**

A implementação do Ranking Loss combinado com melhor tratamento de órfãos pode potencialmente elevar o APFD de **0.62 → 0.68+**.

---

## Anexos

### A. Configuração Atual (experiment_industry.yaml)

```yaml
# Loss
loss:
  type: "weighted_focal"
  focal_alpha: 0.75
  focal_gamma: 2.5

# Ranking (NÃO ESTÁ SENDO USADO!)
ranking:
  enabled: true
  weight: 0.3
  loss_type: "logistic"
```

### B. Estatísticas do Dataset

```
Train: 50,621 samples (2,453 builds)
Val:    6,062 samples (307 builds)
Test:   6,195 samples (307 builds)

Class Distribution:
  Pass: 97.4%
  Fail: 2.6%

Graph:
  Nodes: 2,347 (unique TCs)
  Edges: 335,148 (combined)
  Density: 12.2%
```

### C. Métricas Atuais

```
Best Val F1: 0.4928
Test F1: 0.4935
Mean APFD (test split): 0.5711
Mean APFD (FULL, 277 builds): 0.6169
```
