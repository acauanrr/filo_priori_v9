# 🎯 ESTRATÉGIA: Experimentos 04 - Abordagem Conservadora

## 📊 SITUAÇÃO ATUAL

Todos os experimentos anteriores **colapsaram**:

| Exp | Técnicas Usadas | Resultado |
|-----|----------------|-----------|
| 01 | Focal (0.25, 2.0) | ❌ Colapso → Pass (baseline) |
| 02 | Focal (0.75, 3.0) + Weights + Sampling (20:1) | ❌ Colapso → Fail |
| 03 | Focal (0.25, 2.0) + Weights + Sampling (5:1) | ❌ Colapso → Fail |

**DIAGNÓSTICO**: Combinar múltiplas técnicas de rebalanceamento causa **overengineering** e colapso.

**SOLUÇÃO**: Testar técnicas **ISOLADAMENTE**.

---

## 🎯 EXPERIMENTOS 04: TESTE ISOLADO

### Experimento 04a: WEIGHTED CE APENAS ⭐ RECOMENDADO

**Configuração**: `configs/experiment_04a_weighted_ce_only.yaml`

```yaml
Loss: Weighted Cross-Entropy
  - Class weights automáticos [19.13, 0.51]
  - SEM Focal Loss
  - SEM Balanced Sampling

Model: Simplificado
  - GAT: 1 layer, 2 heads (de 2 layers, 4 heads)
  - Dropout: 0.1-0.2 (de 0.15-0.3)
  - LR: 3e-5 (de 5e-5)

Graph: Menos denso
  - semantic_top_k: 5 (de 10)
  - semantic_threshold: 0.75 (de 0.7)
```

**Por quê começar aqui?**
- ✅ Mais simples e estável
- ✅ Amplamente usado em produção
- ✅ Apenas 1 mecanismo de rebalanceamento
- ✅ Menos hiper-parâmetros

**Expectativa**:
- F1 Macro: **0.30-0.40** (vs 0.025 atual)
- Recall Not-Pass: **0.20-0.40** (vs 1.00 colapsado)
- Recall Pass: **0.95-0.98** (vs 0.00 colapsado)
- **AMBAS classes preditas** (sem colapso)

---

### Experimento 04b: FOCAL LOSS APENAS

**Configuração**: `configs/experiment_04b_focal_only.yaml`

```yaml
Loss: Focal Loss
  - focal_alpha: 0.5 (moderado)
  - focal_gamma: 2.0
  - use_class_weights: false  ← SEM weights extras
  - SEM Balanced Sampling

Model: Simplificado (igual 04a)
```

**Quando usar?**
- Se 04a falhar ou tiver F1 < 0.30
- Focal pode focar melhor em hard examples
- Alternativa ao class weights

**Expectativa**:
- F1 Macro: **0.25-0.35**
- Pode precisar ajuste de alpha (0.3-0.7)

---

## 🚀 PLANO DE EXECUÇÃO

### Passo 1: Executar 04a (PRIMEIRA PRIORIDADE)

```bash
# Limpar cache de graph (usar novo semantic_top_k=5)
rm cache/multi_edge_graph.pkl

# Executar
./venv/bin/python main.py --config configs/experiment_04a_weighted_ce_only.yaml
```

**Monitorar durante execução**:
1. **Build graph**: Densidade deve ser ~10-15% (não 21%)
2. **Epoch 1**: Ver se prediz ambas classes
3. **Epoch 5**: Ver se métricas estão melhorando
4. **Epoch 10**: F1 > 0.20?

**Critérios de SUCESSO**:
- [ ] Val F1 Macro **varia** entre épocas (não constante!)
- [ ] Val Accuracy **varia** (não 2.8% fixo)
- [ ] Confusion matrix mostra **ambas classes preditas**
- [ ] F1 Macro > 0.30 no final
- [ ] Recall Pass > 0.95

**Critérios de FALHA** (parar experimento):
- [ ] Val F1 = 0.0275 em todas épocas (colapso)
- [ ] Prediz só 1 classe (0% diversity)
- [ ] Loss não converge (> 0.15 após 10 épocas)

### Passo 2: Análise dos Resultados

#### Se 04a FUNCIONAR ✅

**Próximos passos**:
1. Adicionar threshold optimization (já implementado)
2. Tentar adicionar **1 técnica por vez**:
   - Exp 05a: Weighted CE + Sampling LEVE (2:1)
   - Exp 05b: Weighted CE + Focal LEVE (alpha=0.1, gamma=1.5)
3. Comparar F1 antes/depois de cada adição

#### Se 04a FALHAR ❌

**Alternativas**:
1. Executar **Exp 04b** (Focal apenas)
2. Se 04b também falhar:
   - SMOTE agressivo (oversample até 1:1)
   - Two-stage training
   - Simplificar modelo ainda mais (MLP sem GAT)
   - Considerar problema intratável

---

## 📊 MÉTRICAS ESPERADAS (REALISTAS)

Para imbalance **37:1**, literatura mostra:

| Métrica | Baseline | Mínimo | Alvo | Excelente |
|---------|----------|--------|------|-----------|
| **F1 Macro** | 0.10 | 0.25 | **0.35** | 0.50 |
| **Recall Not-Pass** | 0.00 | 0.15 | **0.30** | 0.50 |
| **Precision Not-Pass** | 0.00 | 0.20 | **0.35** | 0.50 |
| **Recall Pass** | 1.00 | 0.95 | **0.97** | 0.98 |
| **APFD** | 0.61 | 0.55 | **0.60** | 0.65 |

**Nota**: Nossas metas originais (F1=0.50-0.55) eram **otimistas demais** para ratio 37:1!

---

## 🔧 MODIFICAÇÕES APLICADAS

### 1. Model Simplificado

**ANTES** (Exp 01-03):
```yaml
gnn:
  num_layers: 2
  num_heads: 4
  dropout: 0.2

classifier:
  dropout: 0.3
```

**AGORA** (Exp 04a/04b):
```yaml
gnn:
  num_layers: 1  # ↓ 50% parâmetros
  num_heads: 2   # ↓ 50% parâmetros
  dropout: 0.1   # ↓ regularização

classifier:
  dropout: 0.2   # ↓ regularização
```

**Razão**: Modelo complexo demais para 1,323 samples minority

### 2. Graph Menos Denso

**ANTES**:
```yaml
semantic_top_k: 10
semantic_threshold: 0.7
Density: 21.36%
```

**AGORA**:
```yaml
semantic_top_k: 5      # ↓ 50% edges
semantic_threshold: 0.75  # ↑ mais seletivo
Expected density: 10-15%
```

**Razão**: Graph muito denso pode propagar ruído

### 3. Learning Rate Reduzido

**ANTES**: 5e-5
**AGORA**: 3e-5 (40% redução)

**Razão**: LR alto pode causar instabilidade com imbalance

### 4. Early Stopping Mais Paciente

**ANTES**: patience=12
**AGORA**: patience=15

**Razão**: Dar mais tempo para convergir

---

## ⚠️ SINAIS DE ALERTA

Durante treinamento, **PARAR** se ver:

1. **Colapso detectado**:
   - Val Acc = 0.0283 ou 0.9717 (constante)
   - Val F1 = 0.0275 ou 0.9800 (constante)
   - Classification report: 1 classe com recall=0%

2. **Loss divergente**:
   - Train loss > 0.20 após 5 épocas
   - Val loss aumentando consistentemente

3. **Gradientes explodindo**:
   - Loss = NaN ou Inf
   - Warnings de gradient clipping

Se qualquer um ocorrer: **CTRL+C e ajustar config**

---

## 📋 CHECKLIST DE EXECUÇÃO

### Antes de executar
- [ ] Cache de graph limpo (`rm cache/multi_edge_graph.pkl`)
- [ ] GPU disponível (`nvidia-smi`)
- [ ] Config correto escolhido (04a recomendado)

### Durante execução (monitorar)
- [ ] Graph building: Edges ~200K-300K (não 588K)
- [ ] Epoch 1: Ambas classes preditas?
- [ ] Epoch 5: Val F1 > 0.15?
- [ ] Epoch 10: Val F1 > 0.25?

### Após execução
- [ ] Test F1 Macro > 0.30?
- [ ] Recall Pass > 0.95?
- [ ] Recall Not-Pass > 0.20?
- [ ] APFD > 0.55?
- [ ] Confusion matrix balanceada?

---

## 🎯 DECISÃO APÓS EXP 04a

### Cenário A: SUCESSO (F1 > 0.30, sem colapso)

**Ação**: Gradualmente adicionar técnicas
1. Threshold optimization
2. Sampling leve (2:1)
3. Focal leve (alpha=0.1)

**Objetivo**: Chegar em F1 ~ 0.40-0.45

### Cenário B: PARCIAL (F1 = 0.20-0.30, sem colapso)

**Ação**:
- Aceitar como baseline
- Tentar Exp 04b (Focal)
- Considerar SMOTE

**Objetivo**: Melhorar para F1 ~ 0.30-0.35

### Cenário C: FALHA (F1 < 0.20 ou colapso)

**Ação**:
- Executar Exp 04b
- Se também falhar, considerar:
  - SMOTE agressivo
  - Two-stage training
  - Aceitar limitação do problema

---

## 💡 SE TUDO FALHAR

### Última Linha de Defesa

1. **Treinar apenas para ranking** (não classificação):
   - Usar loss de ranking (Triplet, ArcFace)
   - Focar em APFD (já razoável)
   - Aceitar que classificação é intratável

2. **Ensemble de modelos simples**:
   - 5 modelos com seeds diferentes
   - Voting para classificação
   - Média de probabilities para ranking

3. **Aceitar limitação**:
   - Ratio 37:1 pode ser **limite do tratável**
   - APFD=0.60 é **aceitável** para ranking
   - Classificação pode não ser necessária

---

## ✅ RESUMO EXECUTIVO

**PROBLEMA**: Todos experimentos colapsam por overengineering

**SOLUÇÃO**: Teste isolado de técnicas simples

**PRÓXIMA AÇÃO**:

```bash
# Limpar cache
rm cache/multi_edge_graph.pkl

# Executar Exp 04a
./venv/bin/python main.py --config configs/experiment_04a_weighted_ce_only.yaml
```

**TEMPO ESTIMADO**: 2-3 horas

**CRITÉRIO SUCESSO**: F1 > 0.30, ambas classes preditas, sem colapso

**SE FUNCIONAR**: Adicionar técnicas gradualmente

**SE FALHAR**: Tentar Exp 04b ou abordagens alternativas

---

**Boa sorte! 🍀**
