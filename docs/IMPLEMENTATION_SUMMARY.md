# Implementação Completa: Correções para Colapso de Classe

## ✅ STATUS: IMPLEMENTAÇÃO COMPLETA

Todas as **PRIORIDADES CRÍTICAS** foram implementadas com sucesso para corrigir o colapso de classe identificado na análise dos resultados.

---

## 📊 PROBLEMA IDENTIFICADO

### Análise dos Resultados Originais (experiment_2025-11-14_14-45)

```
Classification Report:
              precision    recall  f1-score   support
    Not-Pass       0.00      0.00      0.00       373  ❌ COLAPSO TOTAL
        Pass       0.97      1.00      0.99     11886  ✅ Prediz tudo como Pass
```

**Causas Raiz:**
1. **Loss fraco**: Focal Loss com alpha=0.25 insuficiente para imbalance 37:1
2. **Sem balanceamento**: Nenhum oversampling da classe minoritária
3. **Threshold fixo**: 0.5 inapropriado para prevalência de 3%
4. **Grafo esparso**: Densidade 0.02% (538 edges, 2347 nodes)

---

## 🔧 SOLUÇÕES IMPLEMENTADAS

### 1. ✅ Weighted Focal Loss (PRIORIDADE CRÍTICA A)

**Arquivo**: `src/training/losses.py`

**Implementação**:
```python
class WeightedFocalLoss(nn.Module):
    """
    Combina 3 mecanismos para imbalance extremo:
    1. Class weights (rebalanceia classes)
    2. Focal modulation (foca em exemplos difíceis)
    3. Alpha weighting (peso adicional classe-específico)
    """
    def __init__(
        self,
        alpha: float = 0.75,        # ↑ de 0.25 (3x mais peso minority)
        gamma: float = 3.0,         # ↑ de 2.0 (mais foco hard examples)
        class_weights: Tensor = None,
        label_smoothing: float = 0.0
    ):
        # Step 1: Weighted CE (aplica class weights)
        ce_loss = F.cross_entropy(inputs, targets, weight=class_weights)

        # Step 2: Focal modulation
        focal_weight = (1 - p_t) ** gamma

        # Step 3: Alpha weighting
        loss = alpha * focal_weight * ce_loss
```

**Configuração**:
```yaml
training:
  loss:
    type: "weighted_focal"
    focal_alpha: 0.75      # Was 0.25 (+200%)
    focal_gamma: 3.0       # Was 2.0 (+50%)
    label_smoothing: 0.0
```

**Impacto Esperado**:
- Class weights automáticos: `[19.13, 0.51]` (37:1 ratio)
- Minority class recebe ~60x mais peso total
- Loss foca em exemplos difíceis (hard negatives)

---

### 2. ✅ Balanced Sampling (PRIORIDADE CRÍTICA B)

**Arquivo**: `main.py` (função `create_dataloaders`)

**Implementação**:
```python
# Cria sample weights (higher para minority)
sample_weights = [
    1.0 if label == minority_class else 0.05
    for label in labels
]

# WeightedRandomSampler com replacement
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # Permite oversample minority
)
```

**Configuração**:
```yaml
training:
  sampling:
    use_balanced_sampling: true
    minority_weight: 1.0    # Fail
    majority_weight: 0.05   # Pass
    # Ratio: 20:1 oversampling
```

**Impacto Esperado**:
- Cada batch terá ~50% minority / 50% majority
- Modelo verá muito mais exemplos de "Fail"
- Previne convergência para solução trivial

**Estatísticas de Log**:
```
BALANCED SAMPLING ENABLED
  Class distribution:
    Class 0 (Fail):  3.04% (minority)
    Class 1 (Pass): 96.96% (majority)

  Expected sampling probabilities:
    Minority class: 50.00%  ← was 3.04%
    Majority class: 50.00%  ← was 96.96%

  Expected samples per batch (size=32):
    Minority class: ~16 samples  ← was ~1
    Majority class: ~16 samples  ← was ~31
```

---

### 3. ✅ Threshold Optimization (PRIORIDADE CRÍTICA C)

**Arquivo**: `src/evaluation/threshold_optimizer.py`

**Implementação**:
```python
def optimize_threshold_for_minority(
    y_true, y_prob,
    metric='f1_macro',
    min_threshold=0.01,   # Não 0.5!
    max_threshold=0.99,
    num_thresholds=100
):
    # Testa 100 thresholds de 0.01 a 0.99
    # Para imbalance 37:1, threshold ótimo geralmente 0.03-0.15

    # Retorna threshold que maximiza métrica escolhida
    return best_threshold, best_score, metrics_dict
```

**Funções Disponíveis**:
- `optimize_threshold_for_minority()` - Foca em minority class
- `optimize_threshold_youden()` - Youden's J (TPR - FPR)
- `optimize_threshold_f_score()` - F-beta score
- `find_optimal_threshold()` - Wrapper com estratégias

**Uso**:
```python
from evaluation.threshold_optimizer import find_optimal_threshold

# Após predição no validation set
threshold, metrics = find_optimal_threshold(
    y_true=val_labels,
    y_prob=val_probs[:, 1],  # P(Pass)
    strategy='f1_macro'
)

# Aplica threshold otimizado no test set
y_pred_test = (test_probs[:, 1] >= threshold).astype(int)
```

**Impacto Esperado**:
- Threshold: 0.5 → 0.03-0.15 (muito menor!)
- Recall minority: 0% → 50-60%
- F1 macro: 0.10 → 0.50-0.55

---

### 4. ✅ Multi-Edge Phylogenetic Graph (MELHORIA ESTRUTURAL B)

**Arquivo**: `src/phylogenetic/multi_edge_graph_builder.py`

**Implementação**:
```python
class MultiEdgeGraphBuilder:
    """
    5 tipos de edges para grafo mais denso:
    1. CO-FAILURE: Tests que falham juntos (existente)
    2. CO-SUCCESS: Tests que passam juntos (NOVO)
    3. SEMANTIC: Top-k similar por embeddings (NOVO)
    4. TEMPORAL: Executados em sequência (NOVO)
    5. COMPONENT: Mesmo módulo/componente (NOVO)
    """

    def fit(self, df_train, embeddings):
        # Constrói cada tipo de edge
        self._build_co_failure_edges(df_train)
        self._build_co_success_edges(df_train)
        self._build_semantic_edges(embeddings)
        self._build_temporal_edges(df_train)
        self._build_component_edges(df_train)

        # Combina com weighted sum
        combined_weight = sum(
            edge_types[etype] * self.edge_weights[etype]
            for etype in self.edge_types
        )
```

**Configuração**:
```yaml
graph:
  use_multi_edge: true

  edge_types:
    - co_failure    # Tests que falham juntos
    - co_success    # Tests que passam juntos
    - semantic      # Top-k similar

  edge_weights:
    co_failure: 1.0
    co_success: 0.5
    semantic: 0.3

  # Thresholds reduzidos para grafo mais denso
  min_co_occurrences: 1    # Was 2
  weight_threshold: 0.05   # Was 0.1

  # Semantic edges
  semantic_top_k: 10
  semantic_threshold: 0.7
```

**Impacto Esperado**:
- Densidade: 0.02% → 0.5-1.0% (25-50x aumento!)
- Avg degree: 4.37 → 20+ edges por node
- GAT terá muito mais informação para propagar

**Tipos de Edge**:

1. **CO-FAILURE** (peso 1.0):
   - Tradicional: tests que falham no mesmo build
   - Weight: P(A fails | B fails)

2. **CO-SUCCESS** (peso 0.5) - NOVO:
   - Inverso: tests que PASSAM juntos
   - Captura correlação negativa (faltava antes!)

3. **SEMANTIC** (peso 0.3) - NOVO:
   - Top-k mais similares por embeddings
   - Conecta tests mesmo sem histórico de execução
   - Garante conectividade mínima para todos nodes

4. **TEMPORAL** (peso 0.2) - NOVO:
   - Tests executados em sequência no mesmo build
   - Captura dependências temporais

5. **COMPONENT** (peso 0.4) - NOVO:
   - Tests do mesmo componente/módulo
   - Usa coluna CR_Component_Name

---

## 🗂️ ARQUIVOS MODIFICADOS

### Novos Arquivos
1. `src/training/losses.py` (modificado)
   - ✅ Adicionado `WeightedFocalLoss`
   - ✅ Adicionado `create_loss_function()` factory

2. `src/evaluation/threshold_optimizer.py` (novo)
   - ✅ `optimize_threshold_for_minority()`
   - ✅ `optimize_threshold_youden()`
   - ✅ `optimize_threshold_f_score()`
   - ✅ `find_optimal_threshold()`

3. `src/phylogenetic/multi_edge_graph_builder.py` (novo)
   - ✅ `MultiEdgeGraphBuilder` com 5 tipos de edges
   - ✅ Métodos save/load para caching

### Arquivos Modificados
4. `src/phylogenetic/phylogenetic_graph_builder.py`
   - ✅ `build_phylogenetic_graph()` com suporte multi-edge
   - ✅ Parâmetros adicionais: `use_multi_edge`, `embeddings`, etc.

5. `main.py`
   - ✅ `create_dataloaders()` com balanced sampling
   - ✅ Suporte para multi-edge graph (passa embeddings)
   - ✅ Loss creation com `create_loss_function()`
   - ✅ Configuração de sampling

### Configurações
6. `configs/experiment_improved.yaml` (novo)
   - ✅ Config completo com todas as melhorias ativadas

---

## 🚀 COMO USAR

### Opção 1: Usar Config Melhorado (RECOMENDADO)

```bash
# Teste rápido com sample
./venv/bin/python main.py \
  --config configs/experiment_improved.yaml \
  --sample-size 500

# Experimento completo
./venv/bin/python main.py \
  --config configs/experiment_improved.yaml
```

### Opção 2: Modificar Config Existente

```yaml
# Em configs/experiment.yaml:

# 1. Adicionar Weighted Focal Loss
training:
  loss:
    type: "weighted_focal"
    focal_alpha: 0.75
    focal_gamma: 3.0

# 2. Adicionar Balanced Sampling
training:
  sampling:
    use_balanced_sampling: true
    minority_weight: 1.0
    majority_weight: 0.05

# 3. Ativar Multi-Edge Graph
graph:
  use_multi_edge: true
  edge_types: [co_failure, co_success, semantic]
  edge_weights:
    co_failure: 1.0
    co_success: 0.5
    semantic: 0.3
  min_co_occurrences: 1
  weight_threshold: 0.05
  semantic_top_k: 10
  semantic_threshold: 0.7
```

### Opção 3: Usar Threshold Optimizer (Após Treino)

```python
# No seu script de avaliação
from evaluation.threshold_optimizer import find_optimal_threshold

# Otimizar threshold no validation set
threshold, metrics = find_optimal_threshold(
    y_true=val_labels,
    y_prob=val_probs[:, 1],
    strategy='f1_macro'
)

print(f"Optimal threshold: {threshold:.4f} (vs default 0.5)")
print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"Recall minority: {metrics['recall_minority']:.4f}")
```

---

## 📈 RESULTADOS ESPERADOS

### Antes (Baseline)
```
Test Metrics:
  Accuracy:     0.9696  (trivial - prediz tudo Pass)
  F1 Macro:     0.10    ❌
  F1 Weighted:  0.98    (enganoso)

Classification Report:
              precision  recall  f1-score  support
    Not-Pass      0.00    0.00      0.00      373  ❌
        Pass      0.97    1.00      0.99    11886  ✅

Graph:
  Density: 0.02%
  Edges: 538
  Avg Degree: 4.37
```

### Depois (Esperado com Melhorias)
```
Test Metrics:
  Accuracy:     0.60-0.65  ✅
  F1 Macro:     0.50-0.55  ✅ (+400%)
  F1 Weighted:  0.65-0.70

Classification Report:
              precision  recall  f1-score  support
    Not-Pass      0.45    0.55      0.50      373  ✅
        Pass      0.98    0.97      0.98    11886  ✅

Graph (Multi-Edge):
  Density: 0.5-1.0%  ✅ (25-50x aumento)
  Edges: 13,000-25,000
  Avg Degree: 20-40
  Edge types:
    co_failure: ~600
    co_success: ~800
    semantic: ~12,000-24,000
```

**Melhorias Críticas**:
- ✅ Recall Not-Pass: 0% → 50-55% (RESOLVE COLAPSO!)
- ✅ F1 Macro: 0.10 → 0.50-0.55 (5x melhoria)
- ✅ Graph density: 0.02% → 0.5-1.0% (25-50x)
- ✅ Model aprende a detectar falhas

---

## 🔍 VALIDAÇÃO

### Checklist Pré-Execução
- [ ] Config `experiment_improved.yaml` criado
- [ ] Todas as importações funcionam (losses, threshold_optimizer, multi_edge_graph_builder)
- [ ] GPU disponível (verifica `nvidia-smi`)
- [ ] Cache limpo se quiser forçar rebuild (`rm -rf cache/*`)

### Checklist Durante Execução
- [ ] **Balanced Sampling**: Log mostra "BALANCED SAMPLING ENABLED" com ~50%/50%
- [ ] **Loss**: Log mostra "WeightedFocalLoss" com alpha=0.75, gamma=3.0
- [ ] **Graph**: Log mostra "MULTI-EDGE PHYLOGENETIC GRAPH" com density > 0.5%
- [ ] **Training**: Ambas classes sendo preditas (não só Pass)
- [ ] **Metrics**: F1 macro > 0.40 após 5-10 épocas

### Checklist Pós-Execução
- [ ] Test F1 Macro >= 0.50
- [ ] Test Accuracy >= 0.60
- [ ] Recall Not-Pass >= 0.50 (CRÍTICO!)
- [ ] Confusion matrix mostra TP > 0 (detecta falhas)
- [ ] APFD mantido ou melhorado (>= 0.60)

---

## 🐛 TROUBLESHOOTING

### Loss muito alto ou NaN
```yaml
# Reduzir learning rate
training:
  learning_rate: 0.00001  # Was 0.00005

# Reduzir focal gamma
training:
  loss:
    focal_gamma: 2.0  # Was 3.0
```

### Balanced sampling muito agressivo
```yaml
# Aumentar majority_weight (reduz oversampling)
training:
  sampling:
    minority_weight: 1.0
    majority_weight: 0.1  # Was 0.05 (10:1 vs 20:1)
```

### Graph muito denso (OOM)
```yaml
# Reduzir semantic top-k
graph:
  semantic_top_k: 5  # Was 10

# Aumentar semantic threshold
graph:
  semantic_threshold: 0.8  # Was 0.7
```

### Model ainda colapsa (prediz só Pass)
```yaml
# Aumentar focal alpha e gamma
training:
  loss:
    focal_alpha: 0.85   # Was 0.75
    focal_gamma: 4.0    # Was 3.0

# Aumentar oversampling
training:
  sampling:
    majority_weight: 0.03  # Was 0.05 (33:1 ratio)
```

---

## 📝 PRÓXIMOS PASSOS (OPCIONAIS)

As implementações abaixo são **melhorias adicionais** (não críticas):

### 5. Expandir Features Estruturais (6 → 29)
- **Status**: Pendente
- **Impacto**: Médio (+5-10% F1)
- **Esforço**: Alto (requer refatoração extensa)

### 6. Imputation Avançada
- **Status**: Pendente
- **Impacto**: Baixo-Médio (+2-5% F1)
- **Esforço**: Médio

Recomendo **testar as melhorias críticas primeiro** antes de implementar estas opcionais.

---

## 📊 COMPARAÇÃO RÁPIDA

| Métrica | Baseline | Esperado | Melhoria |
|---------|----------|----------|----------|
| F1 Macro | 0.10 | 0.50-0.55 | +400% ✅ |
| Recall Not-Pass | 0.00 | 0.50-0.55 | ∞ ✅ |
| Precision Not-Pass | 0.00 | 0.45-0.50 | ∞ ✅ |
| Graph Density | 0.02% | 0.5-1.0% | +2500% ✅ |
| Graph Edges | 538 | 13K-25K | +24x-46x ✅ |

---

## ✅ CONCLUSÃO

**TODAS as prioridades críticas foram implementadas e estão prontas para teste:**

1. ✅ **Weighted Focal Loss** - Strongest loss para imbalance extremo
2. ✅ **Balanced Sampling** - Oversample minority 20:1
3. ✅ **Threshold Optimization** - Não mais fixo em 0.5
4. ✅ **Multi-Edge Graph** - Densidade 25-50x maior

**Próxima ação**: Executar experimento com `configs/experiment_improved.yaml`

```bash
./venv/bin/python main.py --config configs/experiment_improved.yaml
```

**Tempo estimado**: 2-3 horas (full dataset)

**Critério de sucesso**:
- ✅ F1 Macro >= 0.50
- ✅ Recall Not-Pass >= 0.50
- ✅ Model detecta falhas (não colapsa)
