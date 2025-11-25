# Análise Completa dos Resultados - Experiment 2025-11-14

**Data:** 2025-11-14
**Experimento:** Filo-Priori V8 com SBERT
**Dataset:** Completo (62,878 samples, 3,067 builds)

---

## 📊 Resumo Executivo

### ✅ Pontos Positivos
1. **APFD Excelente:** 0.6133 (277 builds) - **Acima da média**
2. **Sem Data Leakage:** Group-aware split funcionou perfeitamente
3. **Pipeline Estável:** Execução completa sem erros
4. **Embeddings Eficientes:** Cache funcionando (60x speedup)

### ⚠️ **PROBLEMA CRÍTICO IDENTIFICADO**
**O modelo colapsou para a classe majoritária (Pass)**
- **Precision Not-Pass: 0.00** ❌
- **Recall Not-Pass: 0.00** ❌
- **Precision Pass: 0.97** ✅
- **Recall Pass: 1.00** ✅

**Tradução:** O modelo **sempre prevê "Pass"**, ignorando completamente a classe "Fail".

---

## 1. ❓ Houve Data Leakage?

### ✅ **RESPOSTA: NÃO**

**Evidências:**

1. **Group-Aware Split por Build_ID:**
```
INFO: 🔒 Using GROUP-AWARE split (by Build_ID) to prevent leakage
INFO: ✅ No build leakage: All splits are disjoint by Build_ID
```

2. **Splits Completamente Disjuntos:**
```
Train:  50,621 samples (2,453 builds)
Val:     6,062 samples (307 builds)
Test:    6,195 samples (307 builds)
─────────────────────────────────────
Total:  62,878 samples (3,067 builds)
```

3. **Nenhum Build Compartilhado:**
- Train builds: 2,453
- Val builds: 307
- Test builds: 307
- Total: 3,067 (100% dos builds únicos)

4. **Features Estruturais:**
- Extraídas **apenas** do histórico de treino
- Imputation baseada em similaridade semântica com treino
- Nenhuma informação de val/test vazou para treino

5. **Grafo Filogenético:**
- Construído **apenas** com dados de treino (2,453 builds)
- Val/Test nodes mapeados para estrutura do grafo de treino
- Edges calculados apenas com co-failures de treino

### 📊 Validação de Leakage

**Distribuição de Features Estruturais:**
```
Train mean:  [838.90, 0.026, 0.025, 0.023, 93.53,  0.083]
Val mean:   [2038.49, 0.029, 0.026, 0.024, 78.58,  0.009]
Test mean:  [2058.02, 0.027, 0.025, 0.022, 106.64, 0.007]
```
- Val/Test médias **diferentes** de Train → ✅ Sem leakage
- Val/Test têm valores maiores no primeiro feature (ID sequencial) → ✅ Temporal split correto

**Samples Needing Imputation:**
- Val: 79/6,062 (1.3%) precisam imputação
- Test: 51/6,195 (0.8%) precisam imputação
- Muito baixo porque os TCs já têm histórico de builds anteriores

**Conclusão sobre Leakage:** ✅ **NENHUM LEAKAGE DETECTADO**

---

## 2. ⚖️ O Modelo Está Estável e Consolidado?

### ❌ **RESPOSTA: NÃO - Modelo Instável com Colapso de Classe**

### Problema 1: Colapso para Classe Majoritária

**Evidência do Colapso:**

```
Classification Report (Validation - Todas as Épocas):
              precision    recall  f1-score   support

    Not-Pass       0.00      0.00      0.00       170
        Pass       0.97      1.00      0.99      5836

    accuracy                           0.97      6006
   macro avg       0.49      0.50      0.49      6006
weighted avg       0.94      0.97      0.96      6006
```

**Interpretação:**
- **Precision Not-Pass = 0.00:** Modelo **nunca** prevê "Not-Pass"
- **Recall Not-Pass = 0.00:** Modelo **não detecta** nenhuma falha
- **Recall Pass = 1.00:** Modelo **sempre** prevê "Pass"

**F1 Macro = 0.49 é ILUSÓRIO:**
- F1(Pass) = 0.99 (excelente)
- F1(Not-Pass) = 0.00 (completamente inútil)
- Macro avg = (0.99 + 0.00) / 2 = **0.495 ≈ 0.49**

**Modelo classificador trivial:** Prevê sempre "Pass" → 97.2% accuracy!

### Problema 2: Estagnação Total de Aprendizado

**Training History:**
```
Epoch 1:  Train Loss=0.0096, Val Loss=0.0096, Val F1=0.4928
Epoch 2:  Train Loss=0.0092, Val Loss=0.0096, Val F1=0.4928
Epoch 3:  Train Loss=0.0092, Val Loss=0.0095, Val F1=0.4928
...
Epoch 13: Train Loss=0.0089, Val Loss=0.0095, Val F1=0.4928
```

**Análise:**
- **Val F1:** Exatamente 0.4928 em **todas as 13 épocas** ❌
- **Val Loss:** Praticamente constante (~0.0095)
- **Train Loss:** Pequena queda (0.0096 → 0.0089)

**Interpretação:**
1. Modelo converge rapidamente para solução trivial (epoch 1)
2. Não há aprendizado adicional após epoch 1
3. Early stopping correto (epoch 13), mas modelo já estava "morto" desde epoch 1

### Problema 3: Imbalance Extremo Não Tratado

**Distribuição de Classes:**
```
Train:  Pass = 49,082 (96.96%)  |  Fail = 1,539 (3.04%)
Val:    Pass =  5,836 (97.17%)  |  Fail =   170 (2.83%)
Test:   Pass =  5,995 (97.45%)  |  Fail =   157 (2.55%)
```

**Class Weights Calculados:**
```
Class weights: [19.13, 0.51]
Weight ratio: 37.26:1 (minority/majority)
```

**Focal Loss Configurado:**
```
alpha: 0.25
gamma: 2.0
```

**Problema:** Mesmo com class weights e Focal Loss, modelo ignora classe minoritária!

### Por Que o APFD é Bom Apesar do Colapso?

**APFD Results:**
```
Mean APFD: 0.6133 (277 builds)
Median APFD: 0.6450
Builds with APFD ≥ 0.7: 45.5%
Builds with APFD ≥ 0.5: 63.2%
```

**Explicação:**

O **APFD não depende de classificação binária correta!**

1. **APFD usa probabilidades para ranking:**
   - Modelo gera `P(Fail)` para cada teste
   - Ranking é feito por `P(Fail)` decrescente
   - Mesmo modelo prevendo sempre "Pass", as **probabilidades variam**

2. **Exemplo:**
   ```
   TC1: P(Fail)=0.02 → Prediction: Pass
   TC2: P(Fail)=0.05 → Prediction: Pass
   TC3: P(Fail)=0.08 → Prediction: Pass

   Ranking: TC3, TC2, TC1 (correto!)
   Classificação: Pass, Pass, Pass (errado se TC3 falha)
   ```

3. **APFD mede ordenação, não classificação:**
   - Se testes com maior P(Fail) realmente falharem mais
   - APFD será bom, **independente do threshold**
   - Modelo pode ter P(Fail) baixas demais para classificar, mas altas **o suficiente para rankear**

**Conclusão:** APFD = 0.6133 indica que o modelo **aprendeu padrões relevantes** para ordenação, mas **falhou em calibração** para classificação.

---

## 3. 🔍 Pontos de Melhoria

### A. Parte Estrutural/Filogenética (Sua Pergunta Principal)

#### Problema 1: Grafo Muito Esparso

**Estatísticas do Grafo:**
```
Nodes: 2,347 (unique test cases)
Edges: 538
Avg Degree: 4.37 edges/node
Avg Edge Weight: 1.14
```

**Análise:**
- **Densidade:** 538 edges / (2347 * 2346 / 2) ≈ **0.02%** (muito esparso!)
- **Avg Degree:** 4.37 → Maioria dos nodes têm poucas conexões
- **Coverage no Test:** 6,152/6,195 (99.3%) → boa coverage
- **Orphan samples:** 43/6,195 (0.7%) → muito baixo

**Implicações:**
1. GAT não tem informação suficiente para propagar
2. Subgraphs extraídos são pequenos e desconectados
3. Stream estrutural contribui pouco

**Melhorias Propostas:**

1. **Adicionar Mais Tipos de Edges:**
   ```python
   # Atual: apenas co-failure
   - co_failure: TCs que falharam juntos

   # Proposto: múltiplos tipos
   - co_failure: Falham juntos
   - co_success: Passam juntos (correlação negativa útil!)
   - temporal: Execução sequencial
   - semantic_similarity: Similaridade de embeddings (top-k neighbors)
   - component_based: Mesmo componente/módulo
   ```

2. **Reduzir Thresholds:**
   ```yaml
   # Atual
   min_co_occurrences: 2
   weight_threshold: 0.1

   # Proposto (mais edges)
   min_co_occurrences: 1  # Aceitar single co-occurrence
   weight_threshold: 0.05  # Threshold menor
   ```

3. **Grafo Hierárquico:**
   - Layer 1: Node-level (test cases)
   - Layer 2: Component-level (grupos de TCs)
   - Layer 3: Build-level (temporal)
   - Use Heterogeneous Graph (HGT/HAN)

4. **Edge Features:**
   ```python
   # Atual: apenas peso escalar
   edge_weight = co_occurrence_count

   # Proposto: edge features (multidimensional)
   edge_features = [
       co_failure_rate,
       co_success_rate,
       temporal_distance,
       semantic_similarity,
       shared_commits_ratio
   ]
   ```

#### Problema 2: Features Estruturais Pouco Informativas

**Features Atuais (6):**
```python
1. Pass rate (historical)
2. Fail rate (historical)
3. Recent pass rate (window=5)
4. Recent fail rate (window=5)
5. Days since last test
6. Total executions
```

**Análise:**
- Features muito básicas
- Não capturam complexidade temporal
- Faltam features de contexto

**Melhorias Propostas:**

1. **Features Temporais Avançadas:**
   ```python
   # Time-series features
   - fail_rate_trend: Slope da fail rate (últimos 10 builds)
   - fail_rate_volatility: Std dev da fail rate
   - time_since_last_fail: Dias desde última falha
   - consecutive_passes: Streak de passes
   - consecutive_fails: Streak de fails
   - seasonal_patterns: Dia da semana, hora do dia
   ```

2. **Features de Contexto:**
   ```python
   # Build context
   - num_commits_in_build: Quantidade de commits no build
   - build_size: Total de TCs no build
   - build_complexity: LOC changed, files changed
   - author_experience: Experiência do autor dos commits
   - commit_message_sentiment: Análise de sentimento
   ```

3. **Features de Relacionamento:**
   ```python
   # Network features
   - node_centrality: Centralidade no grafo
   - clustering_coefficient: Coeficiente de agrupamento
   - pagerank: Importância no grafo
   - community_id: Comunidade detectada (Louvain)
   - neighbor_fail_rate: Taxa de falha dos vizinhos
   ```

4. **Features de Similaridade:**
   ```python
   # Semantic features
   - avg_similarity_to_failed_tcs: Similaridade com TCs que falharam
   - avg_similarity_to_passed_tcs: Similaridade com TCs que passaram
   - novelty_score: Distância do centroid de treino
   ```

#### Problema 3: Imputation Muito Simplista

**Método Atual:**
- k-NN semântico (k=10)
- Threshold de similaridade: 0.50
- Fallback: Conservative defaults

**Melhorias:**

1. **Imputation Mais Sofisticada:**
   ```python
   # Weighted k-NN com múltiplas fontes
   - Semantic similarity (embeddings)
   - Structural similarity (features)
   - Temporal similarity (time window)
   - Component similarity (metadata)

   # Combine com pesos aprendidos
   imputed_value = w1*semantic + w2*structural + w3*temporal
   ```

2. **Imputation Iterativa:**
   ```python
   # MICE (Multiple Imputation by Chained Equations)
   1. Impute feature 1 baseado em outras features
   2. Impute feature 2 usando feature 1 imputada
   3. Iterate até convergência
   ```

3. **Uncertainty-Aware Imputation:**
   ```python
   # Retornar distribuição, não ponto único
   imputed_mean, imputed_std = impute_with_uncertainty(...)

   # Usar na loss function
   loss += uncertainty_penalty * imputed_std
   ```

### B. Modelo e Training

#### Problema 1: Colapso para Classe Majoritária

**Causas Raiz:**

1. **Focal Loss Insuficiente:**
   - alpha=0.25 muito baixo para imbalance 37:1
   - gamma=2.0 padrão, pode ser aumentado

2. **Class Weights Não Usados:**
   - Calculados: [19.13, 0.51]
   - **Mas Focal Loss não os usa diretamente!**

3. **Threshold Fixo em 0.5:**
   - Para classe com 3% prevalência, threshold deve ser ~0.03-0.05

**Soluções Imediatas:**

1. **Combinar Focal + Class Weights:**
   ```python
   # Weighted Focal Loss
   class FocalLossWeighted(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
           self.alpha = alpha
           self.gamma = gamma
           self.class_weights = class_weights  # [19.13, 0.51]

       def forward(self, inputs, targets):
           ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
           p_t = torch.exp(-ce_loss)
           focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
           return focal_loss.mean()
   ```

2. **Aumentar alpha/gamma:**
   ```yaml
   loss:
     focal:
       alpha: 0.75  # Mais peso para classe minoritária
       gamma: 3.0   # Mais foco em hard examples
   ```

3. **Threshold Optimization:**
   ```python
   # Atual: threshold fixo em 0.5
   # Proposto: otimizar threshold no validation set

   best_threshold = find_best_threshold(
       y_true=val_labels,
       y_prob=val_probs,
       metric='f1_macro'  # ou 'recall_minority'
   )
   # Esperado: ~0.03-0.10 para classe 3%
   ```

4. **Oversampling/Undersampling:**
   ```python
   # SMOTE para classe minoritária
   from imblearn.over_sampling import SMOTE

   smote = SMOTE(sampling_strategy=0.1)  # 10% ratio
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

5. **Two-Stage Training:**
   ```python
   # Stage 1: Train com balanced sampling
   sampler = WeightedRandomSampler(
       weights=sample_weights,  # Mais peso para Fail
       num_samples=len(dataset),
       replacement=True
   )

   # Stage 2: Fine-tune com distribuição original
   ```

#### Problema 2: Modelo Não Aprende Após Epoch 1

**Causas:**

1. **Learning Rate Muito Baixo:**
   - lr = 5e-5 pode ser muito pequeno
   - Modelo converge para mínimo local ruim rapidamente

2. **Regularização Excessiva:**
   - weight_decay = 1e-4
   - dropout = 0.15-0.3 em múltiplas camadas

**Soluções:**

1. **Learning Rate Schedule Adaptativo:**
   ```python
   # Atual: CosineAnnealing (fixa)
   # Proposto: ReduceLROnPlateau

   scheduler = ReduceLROnPlateau(
       optimizer,
       mode='max',
       factor=0.5,
       patience=3,
       verbose=True
   )
   ```

2. **Warmup Mais Longo:**
   ```yaml
   training:
     learning_rate: 1e-4  # Maior
     warmup_epochs: 10     # Mais longo
   ```

3. **Gradual Unfreezing:**
   ```python
   # Freeze embeddings inicialmente
   # Unfreeze progressivamente

   if epoch < 5:
       freeze_embeddings()
   elif epoch < 10:
       unfreeze_top_layers()
   else:
       unfreeze_all()
   ```

### C. Arquitetura

#### Problema: Fusion Pode Não Estar Funcionando

**Arquitetura Atual:**
```
Semantic Stream:  [1536] → [256]
Structural Stream: [6] → [256] (via GAT)
Fusion: [512] → Classifier
```

**Questões:**

1. **Structural Stream com GAT:**
   - Input: 6 features estruturais
   - Processamento: GAT com grafo esparso
   - **Problema:** 6 features muito poucas para GAT aprender

2. **Dimensões Desbalanceadas:**
   - Semantic: 1536 dims (rico)
   - Structural: 6 dims (pobre)
   - Modelo pode ignorar structural stream

**Melhorias:**

1. **Aumentar Features Estruturais:**
   - De 6 → 20-30 features (conforme proposto)
   - Dar mais capacidade para structural stream

2. **Attention-Based Fusion:**
   ```python
   # Atual: Concatenação simples
   fused = torch.cat([semantic_out, structural_out], dim=-1)

   # Proposto: Cross-Attention
   class CrossAttentionFusion(nn.Module):
       def forward(self, semantic, structural):
           # Semantic attends to structural
           attn_s = self.attention(semantic, structural, structural)
           # Structural attends to semantic
           attn_st = self.attention(structural, semantic, semantic)
           # Combine
           return torch.cat([attn_s, attn_st], dim=-1)
   ```

3. **Gated Fusion:**
   ```python
   # Aprende pesos para cada stream
   class GatedFusion(nn.Module):
       def forward(self, semantic, structural):
           gate = torch.sigmoid(self.gate_layer(torch.cat([semantic, structural], -1)))
           return gate * semantic + (1 - gate) * structural
   ```

4. **Multi-Scale Structural Processing:**
   ```python
   # Processar structural features em múltiplas escalas
   - Local: GAT com 1-hop neighbors
   - Medium: GAT com 2-hop neighbors
   - Global: Graph pooling (mean/max)

   # Combine all scales
   structural_multi = torch.cat([local, medium, global], dim=-1)
   ```

---

## 4. 📋 Recomendações Prioritizadas

### 🔴 **Prioridade CRÍTICA** (Resolver Primeiro)

1. **Corrigir Colapso de Classe:**
   - [ ] Implementar Weighted Focal Loss
   - [ ] Aumentar alpha para 0.75
   - [ ] Otimizar threshold no validation set
   - [ ] Adicionar SMOTE ou balanced sampling

2. **Validar Que Modelo Está Aprendendo:**
   - [ ] Monitorar precision/recall **por classe** durante treino
   - [ ] Adicionar early stopping baseado em **recall da classe minoritária**
   - [ ] Plot confusion matrix a cada época

### 🟡 **Prioridade ALTA** (Melhorias Significativas)

3. **Enriquecer Features Estruturais:**
   - [ ] Adicionar 10-15 novas features temporais
   - [ ] Adicionar features de contexto (build, commits)
   - [ ] Implementar features de rede (centrality, clustering)

4. **Melhorar Grafo Filogenético:**
   - [ ] Adicionar edges de co-success
   - [ ] Adicionar edges de similaridade semântica
   - [ ] Reduzir thresholds para aumentar densidade
   - [ ] Adicionar edge features (não apenas peso)

### 🟢 **Prioridade MÉDIA** (Otimizações)

5. **Hyperparameter Tuning:**
   - [ ] Grid search em learning_rate, dropout, weight_decay
   - [ ] Test diferentes configurações de Focal Loss
   - [ ] Experimentar diferentes arquiteturas de fusion

6. **Arquitetura:**
   - [ ] Implementar cross-attention fusion
   - [ ] Test multi-scale structural processing
   - [ ] Adicionar uncertainty estimation

### ⚪ **Prioridade BAIXA** (Futuro)

7. **Advanced Techniques:**
   - [ ] Hierarchical graph (multi-level)
   - [ ] Heterogeneous graph networks (HAN/HGT)
   - [ ] Temporal graph networks
   - [ ] Meta-learning para few-shot failure detection

---

## 5. 🎯 Experimentos Sugeridos

### Experimento 1: Fix Class Collapse (URGENTE)

**Objetivo:** Fazer modelo detectar falhas

**Mudanças:**
```yaml
loss:
  type: "weighted_focal"
  focal:
    alpha: 0.75
    gamma: 3.0
    use_class_weights: true  # Usar [19.13, 0.51]

training:
  learning_rate: 1e-4  # Maior
  sampling_strategy: "balanced"  # WeightedRandomSampler

evaluation:
  threshold_search:
    enabled: true
    optimize_for: "recall_minority"  # Foco em detectar fails
```

**Critério de Sucesso:**
- Recall Not-Pass > 0.30
- Precision Not-Pass > 0.10
- F1 Not-Pass > 0.15

### Experimento 2: Richer Structural Features

**Objetivo:** Dar mais informação para structural stream

**Mudanças:**
```python
# Adicionar features
structural_features = [
    # Atuais (6)
    'pass_rate', 'fail_rate', 'recent_pass_rate',
    'recent_fail_rate', 'days_since_last', 'total_execs',

    # Novas (14)
    'fail_rate_trend', 'fail_rate_volatility',
    'time_since_last_fail', 'consecutive_passes', 'consecutive_fails',
    'num_commits', 'build_size', 'locs_changed',
    'node_centrality', 'clustering_coef', 'pagerank',
    'neighbor_fail_rate', 'semantic_novelty', 'component_diversity'
]
# Total: 20 features
```

**Critério de Sucesso:**
- Structural stream contribui mais (visualizar attention weights)
- F1 Macro aumenta em 5-10%

### Experimento 3: Denser Phylogenetic Graph

**Objetivo:** Aumentar densidade do grafo

**Mudanças:**
```yaml
graph:
  type: "multi_edge"  # Co-failure + Co-success + Semantic
  min_co_occurrences: 1  # Mais edges
  weight_threshold: 0.05  # Threshold menor
  semantic_edge_top_k: 10  # Top-10 neighbors semânticos
  include_co_success: true  # Edges de co-success
```

**Critério de Sucesso:**
- Densidade > 1% (vs atual 0.02%)
- Avg degree > 20 (vs atual 4.37)
- APFD aumenta em 3-5%

### Experimento 4: Advanced Fusion

**Objetivo:** Melhor integração de semantic + structural

**Mudanças:**
```python
# Substituir fusion por cross-attention
fusion = CrossAttentionFusion(
    semantic_dim=256,
    structural_dim=256,
    num_heads=4
)
```

**Critério de Sucesso:**
- Visualizar que ambos streams são usados
- F1 aumenta em 2-5%

---

## 6. 📊 Métricas de Sucesso Revisadas

### Atual (Baseline)
```
Val F1 Macro:    0.4928 (ilusório - modelo prevê sempre Pass)
Val F1 Not-Pass: 0.00 ❌
Val F1 Pass:     0.99 ✅

Test F1 Macro:   0.4935
Test F1 Not-Pass: 0.00 ❌
Test F1 Pass:    0.99 ✅

APFD:            0.6133 ✅ (bom para ranking)
```

### Target (Após Correções)
```
Val F1 Macro:    > 0.60 (+22%)
Val F1 Not-Pass: > 0.30 (vs 0.00) ⭐ CRÍTICO
Val F1 Pass:     > 0.90 (manter alto)

Test F1 Macro:   > 0.62 (+26%)
Test F1 Not-Pass: > 0.30 ⭐ CRÍTICO
Test F1 Pass:    > 0.92

Recall Not-Pass: > 0.40 ⭐ CRÍTICO (vs 0.00)
Precision Not-Pass: > 0.20 ⭐ CRÍTICO (vs 0.00)

APFD:            > 0.65 (+6%)
```

**Meta Principal:** **Fazer o modelo detectar falhas!**

---

## 7. ✅ Checklist de Ações

### Imediatas (Esta Semana)

- [ ] Implementar Weighted Focal Loss
- [ ] Adicionar balanced sampling
- [ ] Otimizar threshold no validation set
- [ ] Monitorar recall por classe durante treino
- [ ] Rodar Experimento 1 (Fix Class Collapse)

### Curto Prazo (Próximas 2 Semanas)

- [ ] Adicionar 10-15 novas features estruturais
- [ ] Implementar edges de co-success no grafo
- [ ] Adicionar edges semânticas (top-k)
- [ ] Reduzir thresholds do grafo
- [ ] Rodar Experimentos 2 e 3

### Médio Prazo (Próximo Mês)

- [ ] Implementar cross-attention fusion
- [ ] Adicionar edge features (multidimensional)
- [ ] Implementar imputation sofisticada
- [ ] Experimentar hierarchical graph
- [ ] Rodar Experimento 4

---

## 8. 🎓 Conclusões Finais

### ✅ O Que Está Funcionando

1. **Data Pipeline:** Sem leakage, splits corretos
2. **Embeddings:** SBERT rápido e eficiente
3. **APFD:** 0.6133 é excelente para ranking
4. **Infraestrutura:** Pipeline estável, cache funcional

### ❌ O Que Precisa Correção Urgente

1. **Modelo Colapsado:** Não detecta falhas (precision/recall = 0.00)
2. **Loss Function:** Focal Loss insuficiente para imbalance 37:1
3. **Threshold:** Fixo em 0.5 é inadequado para classe 3%
4. **Monitoring:** Falta monitoramento de métricas por classe

### 🔧 O Que Precisa Evolução

1. **Features Estruturais:** Muito básicas (apenas 6)
2. **Grafo:** Muito esparso (0.02% densidade)
3. **Fusion:** Concatenação simples, pode ser melhor
4. **Imputation:** k-NN simples, pode ser sofisticado

### 🎯 Próximo Passo Recomendado

**PRIORITÁRIO:** Rodar Experimento 1 (Fix Class Collapse)

**Objetivo:** Fazer modelo detectar ao menos 30% das falhas

**Mudanças Mínimas:**
1. Weighted Focal Loss (alpha=0.75, gamma=3.0, use_class_weights=true)
2. Balanced sampling (WeightedRandomSampler)
3. Threshold optimization (optimize_for="recall_minority")
4. Monitor recall por classe

**Tempo Estimado:** 1-2 dias para implementar e testar

**Impacto Esperado:**
- Recall Not-Pass: 0.00 → 0.30-0.40
- F1 Not-Pass: 0.00 → 0.20-0.30
- APFD: 0.6133 → 0.63-0.65 (pequeno aumento)

**Se funcionar:** Adicionar melhorias estruturais/filogenéticas (Exp 2-4)

---

## 📚 Referências para Melhorias

### Class Imbalance
- Focal Loss (original paper): https://arxiv.org/abs/1708.02002
- Class Balanced Loss: https://arxiv.org/abs/1901.05555
- SMOTE: https://arxiv.org/abs/1106.1813

### Graph Neural Networks
- Graph Attention Networks (GAT): https://arxiv.org/abs/1710.10903
- Heterogeneous Graph Transformer: https://arxiv.org/abs/2003.01332
- Temporal Graph Networks: https://arxiv.org/abs/2006.10637

### Test Prioritization
- Learning to Rank for Test Prioritization: Multiple papers
- APFD and variants: Survey papers

### Feature Engineering
- Time Series Features (tsfresh): https://github.com/blue-yonder/tsfresh
- Network Features (NetworkX): https://networkx.org/

---

**Status:** 📋 **ANÁLISE COMPLETA**

**Próxima Ação:** Implementar Experimento 1 (Fix Class Collapse)

*Análise realizada em: 2025-11-14*
